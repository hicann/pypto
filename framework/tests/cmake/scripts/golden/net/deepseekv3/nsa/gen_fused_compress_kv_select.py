#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import math
import sys
import logging
from pathlib import Path
from typing import List
from common_func import dump_file

import numpy as np
from ml_dtypes import bfloat16

if __name__ == "__main__":
    """单独调试时配置"""
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import (
        GoldenRegister,
    )  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def trans_bsnd_to_bsh(tensor):
    shape = tensor.shape
    if len(shape) == 4:
        b = shape[0]
        s = shape[1]
        n = shape[2]
        d = shape[3]
        return np.reshape(tensor, (b, s, n * d))
    else:
        return tensor


def gen_cache_tensor(x, block_table, block_num, block_size, b):
    logging.info("Entering into gen_cache_tensor!")
    dtype = x.dtype
    b, s, n, d = x.shape
    k_cache = np.zeros([block_num, block_size, n * d]).astype(dtype)

    k_tensor_bsh_raw = trans_bsnd_to_bsh(x)  # (b, s2, n2 * d_qk)

    # kv padding
    k_tensor_bsh = np.zeros(
        (b, block_table.shape[1] * block_size, n * d)
    ).astype(
        dtype
    )  # (b, max_block_num * block_size, n2 * d_qk)
    k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx == -1:
                continue
            else:
                k_cache[cache_block_idx, :, :] = k_tensor_bsh[
                                                 b_idx, block_offset: (block_offset + block_size), :
                                                 ]
    k_cache = np.reshape(
        k_cache, (block_num * block_size, n, d)
    )  # (block_num * block_size, n2, d_qk)

    return k_cache


def gen_uniform_data(shape, min_value, max_value, dtype):
    # np.random.seed(None)
    np.random.seed(0)
    if min_value == 0 and max_value == 0:
        return np.zeros(shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=shape)
    return np.random.uniform(low=min_value, high=max_value, size=shape).astype(
        dtype
    )


def gen_block_table(b, block_size, max_kv, act_kv):
    logging.info("Entering into gen_block_table!")
    block_num = 0
    block_num_each = []
    for cur_s in act_kv:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    shape_bt = [b, math.ceil(max_kv / block_size)]
    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * shape_bt[1]
    block_table = np.tile(block_table, (shape_bt[0], 1)).astype(np.int32)

    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    return block_num, block_table


def gen_p_slc_ast(p_cmp, l_prime=64, l=32, d=16):
    b, n, s, kv_cmp_len = p_cmp.shape
    out_loop = l_prime // d
    inner_loop = l // d
    reduce_len = kv_cmp_len // out_loop + 1
    s_slc = (kv_cmp_len + 3) // 4
    p_cmp = p_cmp.reshape(b * n * s, kv_cmp_len)
    trans0 = p_cmp.transpose(1, 0)
    reduce0 = np.zeros((s_slc, b * n * s))
    for i in range(s_slc):
        part0 = trans0[i * out_loop:i * out_loop + out_loop, :]
        part1 = trans0[1 + i * out_loop:1 + i * out_loop + out_loop, :]
        part0Sum = np.sum(part0, axis=0)
        part1Sum = np.sum(part1, axis=0)
        reduce0[i, :] = part0Sum + part1Sum
    trans1 = reduce0.transpose(1, 0)
    reduce1 = np.sum(trans1, axis=0)
    return trans0, reduce0, trans1, reduce1


def gen_topk_indices(p_slc, front=1, near=2, topk=16, actual_len=0):
    b, s, reduce_len = p_slc.shape
    front_indices = np.arange(front)
    near_indices = np.arange((reduce_len if actual_len == 0 else actual_len) - near, reduce_len)
    required_indices = np.concatenate([front_indices, near_indices])

    mask = np.zeros_like(p_slc, dtype=bool)
    mask[:, :, required_indices] = True
    x_masked = np.where(mask, -np.inf, p_slc)

    k = topk - front - near
    additional_indices = np.argpartition(x_masked, -k, axis=-1)[:, :, -k:]
    additional_indices = np.sort(additional_indices)
    topk_indices = np.concatenate([
        np.tile(front_indices, (b, s, 1)),
        additional_indices,
        np.tile(near_indices, (b, s, 1)),
    ], axis=-1)

    return additional_indices, topk_indices


def safe_sigmoid(x):
    logging.info("Entering into safe_sigmoid!")
    x_dtype = x.dtype
    x_cast = x.astype(np.float32)
    x_sig = float(1.0) / (float(1.0) + np.exp((-1.0) * x_cast))
    return x_sig.astype(x_dtype)


def rotate_half(x):
    logging.info("Entering into rotate_half!")
    x_shape = x.shape
    last_dim = x_shape[-1]
    x_left = x[..., : last_dim // 2]
    x_right = x[..., last_dim // 2:]
    res = np.concatenate(((float(-1.0)) * x_right, x_left), axis=-1)
    return res


def mlp_single_rope(x, cos_in, sin_in):
    logging.info("Entering into mlp_single_rope!")
    # x: (1, cmp_bs, n2, dr), cos_in: (1, cmp_bs, dr), sin_in: (1, cmp_bs, dr)
    x_dtype = x.dtype
    _, s, n, d = x.shape
    x_cast = x.astype(np.float32)
    cos_cast = cos_in.astype(np.float32)
    sin_cast = sin_in.astype(np.float32)
    cos_re = np.expand_dims(cos_cast, axis=2)  # (1, cmp_bs, 1, dr)
    sin_re = np.expand_dims(sin_cast, axis=2)  # (1, cmp_bs, 1, dr)
    x_re = np.reshape(x_cast, (1, s, n, d // 2, 2))  # (1, cmp_bs, n2, dr // 2, 2)
    x_trans = np.transpose(x_re, (0, 1, 2, 4, 3))  # (1, cmp_bs, n2, 2, dr // 2)
    x_re1 = np.reshape(x_trans, (1, s, n, d))  # (1, cmp_bs, n2, dr)
    res = x_re1 * cos_re + rotate_half(x_re1) * sin_re  # (1, cmp_bs, n2, dr)
    return res.astype(x_dtype)


def softmax(x, dim=-1):
    logging.info("Entering into softmax!")
    x = x.astype(np.float32)
    x_max = x.max(dim, keepdims=True)
    x_sub = x - x_max
    x_exp = np.exp(x_sub)
    x_sum = np.sum(x_exp, axis=dim, keepdims=True)
    return x_exp / x_sum


def mlp_compress(x, w1, w2):
    logging.info("Entering into mlp_compress!")
    # x: (1, cmp_bs, n2, d), w1: (cmp_bs * d, 2 * cmp_bs * d), w2: (2 * cmp_bs * d, d)
    x_dtype = x.dtype
    _, s, n, d = x.shape
    x_new = np.reshape(
        np.transpose(x, (0, 2, 1, 3)), (1 * n, s * d)
    )  # (n2, cmp_bs * d)
    mm1 = np.matmul(
        x_new.astype(np.float32), w1.astype(np.float32)
    )  # (n2, 2 * cmp_bs * d)
    sig = safe_sigmoid(mm1)  # (n2, 2 * cmp_bs * d)
    res = np.matmul(sig.astype(np.float32), w2.astype(np.float32)).astype(x_dtype)  # (n2, d)
    res = np.reshape(res, (1, n * d))
    return res


def compress_attention_data_gen(out_path: Path, params, other_input):
    # define input files
    in_params_path = Path(out_path, "input_param_compress.bin")
    q_nope_path = Path(out_path, "q_nope_compress.bin")
    q_rope_path = Path(out_path, "q_rope_compress.bin")
    kv_cache_path = Path(out_path, "kv_cache_compress.bin")
    kr_cache_path = Path(out_path, "kr_cache_compress.bin")

    block_table_path = Path(out_path, "block_table_compress.bin")
    cmp_block_table_path = Path(out_path, "cmp_block_table_compress.bin")
    act_seq_path = Path(out_path, "act_seq_compress.bin")
    act_cmp_seq_path = Path(out_path, "act_cmp_seq_compress.bin")
    mlp_wk1_path = Path(out_path, "mlp_wk1_compress.bin")
    mlp_wk2_path = Path(out_path, "mlp_wk2_compress.bin")
    mlp_cos_path = Path(out_path, "mlp_cos_compress.bin")
    mlp_sin_path = Path(out_path, "mlp_sin_compress.bin")

    # gen block_table, here block_num = sum of blocks of each act_seq

    b = params.get("b")
    s1 = params.get("s")
    n1 = params.get("n1")
    d_qk = params.get("d")
    q_dtype = params.get("dtype")
    s2 = params.get("s2")
    n2 = params.get("n2")
    dv = params.get("dv")
    dr = d_qk - dv
    k_dtype = params.get("dtype")
    act_seq = params.get("act_seq")
    block_size = params.get("block_size")
    cmp_bs = params.get("cmp_block_size")
    cmp_stride = params.get("cmp_stride")
    act_cmp_seq = params.get("act_cmp_seq")
    scmp = params.get("scmp")

    block_num, block_table = gen_block_table(b, block_size, s2, act_seq)
    cmp_block_num, cmp_block_table = gen_block_table(
        b, block_size, scmp, act_cmp_seq
    )

    # construct input shape
    shape_q = [b, s1, n1, d_qk]
    shape_k = [b, s2, n2, d_qk]
    shape_wk1 = [cmp_bs * d_qk, 2 * cmp_bs * d_qk]
    shape_wk2 = [2 * cmp_bs * d_qk, d_qk]
    shape_cos = [b, cmp_bs, dr]
    shape_sin = shape_cos

    # construct tensor by shapes
    q_bsnd = gen_uniform_data(shape_q, -1, 1, q_dtype)
    k_bsnd = gen_uniform_data(shape_k, -1, 1, k_dtype)  # (b, s, n, d)

    if other_input != None:
        mode = True
        q, k, bt = other_input
        if mode and q.shape == q_bsnd.shape and k.shape == k_bsnd.shape and bt.shape == block_table.shape:
            print(f'####################################use mla prolog output as input for compress attention')
            q_bsnd = q
            k_bsnd = k
            block_table = bt

    q_nope = q_bsnd[..., : dv]
    q_rope = q_bsnd[..., dv:]

    threhold = 1
    wk1 = gen_uniform_data(shape_wk1, -1 * threhold, 1 * threhold, k_dtype)
    wk2 = gen_uniform_data(shape_wk2, -1 * threhold, 1 * threhold, k_dtype)
    mlp_cos = gen_uniform_data(shape_cos, -1 * threhold, 1 * threhold, k_dtype)
    mlp_sin = gen_uniform_data(shape_sin, -1 * threhold, 1 * threhold, k_dtype)

    k_cache = gen_cache_tensor(
        k_bsnd, block_table, block_num, block_size, b
    )  # (block_num * block_size, n2, d)
    kv_cache = k_cache[..., : dv]  # (block_num * block_size, n2, dv)
    kr_cache = k_cache[..., dv:]  # (block_num * block_size, n2, dr)

    # construct output tensor
    shape_attn_out = [b, s1, n1, dv]
    cmp_attn_out = np.zeros(shape_attn_out, dtype=np.float32)
    shape_softmax_out = [b, s1, n1, scmp]
    cmp_softmax_out = np.zeros(shape_softmax_out, dtype=np.float32)
    topk_full = np.zeros([b, s1, 16], dtype=np.int32)
    topk_input = np.zeros([b, s1, 1 , (scmp+3)//4], dtype=np.float32)

    k_tensor_out = np.zeros(
        [block_table.shape[1] * block_size, n2 * d_qk], dtype=q_dtype
    )
    shape_cmp_cache = [b, scmp, n2 * d_qk]
    k_cmp_tensor = np.zeros(shape_cmp_cache, dtype=k_dtype)
    first_rope = np.zeros((act_cmp_seq[0], 1, cmp_bs, 1, dr), dtype=k_dtype)
    first_rope_input = np.zeros((act_cmp_seq[0], 1, cmp_bs, 1, dr), dtype=k_dtype)

    input_data_map = {}
    input_data_map["q_bsnd"] = q_bsnd
    input_data_map["k_bsnd"] = k_bsnd
    input_data_map["k_tensor_out"] = k_tensor_out
    input_data_map["mlp_cos"] = mlp_cos
    input_data_map["mlp_sin"] = mlp_sin
    input_data_map["first_rope"] = first_rope
    input_data_map["first_rope_input"] = first_rope_input
    input_data_map["k_cmp_tensor"] = k_cmp_tensor
    input_data_map["cmp_softmax_out"] = cmp_softmax_out
    input_data_map["cmp_attn_out"] = cmp_attn_out
    input_data_map["cmp_block_table"] = cmp_block_table
    input_data_map["cmp_block_num"] = cmp_block_num
    input_data_map["wk1"] = wk1
    input_data_map["wk2"] = wk2
    input_data_map["act_seq"] = act_seq
    input_data_map["act_cmp_seq"] = act_cmp_seq
    input_data_map["topk_full"] = topk_full
    input_data_map["topk_input"] = topk_input


    input_params = [
        b,
        s1,
        n1,
        d_qk,
        s2,
        n2,
        dv,
        block_size,
        cmp_bs,
        cmp_stride,
    ]
    # dump golden file
    dtypeStr = "bf16" if q_dtype == bfloat16 else "fp16"
    dump_file(input_params, in_params_path, "int32")
    dump_file(q_nope, q_nope_path, dtypeStr)
    dump_file(q_rope, q_rope_path, dtypeStr)
    dump_file(kv_cache, kv_cache_path, dtypeStr)
    dump_file(kr_cache, kr_cache_path, dtypeStr)
    dump_file(block_table, block_table_path, "int32")
    dump_file(cmp_block_table, cmp_block_table_path, "int32")
    dump_file(wk1, mlp_wk1_path, dtypeStr)
    dump_file(wk2, mlp_wk2_path, dtypeStr)
    dump_file(mlp_cos, mlp_cos_path, dtypeStr)
    dump_file(mlp_sin, mlp_sin_path, dtypeStr)
    dump_file(act_seq, act_seq_path, "int32")
    dump_file(act_cmp_seq, act_cmp_seq_path, "int32")

    return input_data_map


def thread_func(b_idx, thread_params):
    q_dtype, dtypeStr, k_bsnd, k_tensor_out, dv, act_seq, b, s2, n2, d_qk, mlp_cos, mlp_sin, act_cmp_seq, k_dtype, s, cmp_stride, cmp_bs, wk1, wk2, first_rope, first_rope_input, k_cmp_tensor, group, q_bsnd, out_path, softmax_scale, cmp_softmax_out, cmp_attn_out, topk_full, topk_input \
        = thread_params
    cur_k = k_bsnd[b_idx: b_idx + 1]  # (1, s2, n2, d_qk)
    cur_v = k_bsnd[b_idx: b_idx + 1, :, :, : dv]  # (1, s2, n2, dv)
    k_tensor_out[: act_seq[b_idx], :] = (np.reshape(cur_k, (s2, n2 * d_qk)))[: act_seq[b_idx], :]
    cur_cos = mlp_cos[b_idx: b_idx + 1]  # (1, cmp_bs, dr)
    cur_sin = mlp_sin[b_idx: b_idx + 1]  # (1, cmp_bs, dr)
    cur_seq = act_seq[b_idx]
    cur_cmp_seq = act_cmp_seq[b_idx]

    k_cmp = np.zeros([cur_cmp_seq, n2 * d_qk], k_dtype)  # (cur_cmp_seq, n2 * d_qk)
    # kv compress
    for loop_idx in range(cur_cmp_seq):
        print(f'b_idx:{b_idx} b:{b} loop_idx:{loop_idx} cur_cmp_seq:{cur_cmp_seq}')
        cmp_offset = loop_idx * cmp_stride
        k_local = cur_k[:, cmp_offset: (cmp_offset + cmp_bs), :, :]  # (1, cmp_bs, n2, d_qk)
        k_rope = mlp_single_rope(k_local[..., dv:], cur_cos, cur_sin)  # (1, cmp_bs, n2, dr)
        k_cat = np.concatenate((k_local[..., : dv], k_rope), axis=-1)  # (1, cmp_bs, n2, d_qk)
        k_cmp[loop_idx, :] = mlp_compress(k_cat, wk1, wk2)  # (1, n2 * d_qk)
        # if loop_idx == 0:
        first_rope[loop_idx, ...] = k_rope
        first_rope_input[loop_idx, ...] = k_local[..., dv:]
    k_cmp_tensor[b_idx: (b_idx + 1), :cur_cmp_seq, :] = k_cmp
    # compress attention
    for s1_idx in range(s):
        # think of casual calculation
        cur_off = s - s1_idx - 1
        eff_seq = (cur_seq - cur_off - cmp_bs) // cmp_stride + 1
        for n2_idx in range(n2):
            g_offset = n2_idx * group
            dk_offset = n2_idx * d_qk
            # get q_attn
            cur_q = np.reshape(
                q_bsnd[
                b_idx: (b_idx + 1),
                s1_idx: (s1_idx + 1),
                g_offset: (g_offset + group),
                :,
                ],
                (group, d_qk),
            )  # (g, d_qk)

            # get k_attn
            act_cmp_k = k_cmp[:eff_seq, dk_offset: (dk_offset + d_qk)]  # (eff_seq, d_qk)
            act_cmp_v = act_cmp_k[..., : dv]  # (eff_seq, dv)
            cmp_c1 = np.matmul(cur_q.astype(np.float32),
                               act_cmp_k.transpose(1, 0).astype(np.float32))  # (g, eff_seq)
            cmp_v1 = softmax(cmp_c1)
            cmp_scale = cmp_v1 * softmax_scale
            cmp_softmax_out[
            b_idx: (b_idx + 1),
            s1_idx: (s1_idx + 1),
            g_offset: (g_offset + group),
            :eff_seq,
            ] = np.reshape(cmp_v1, (1, 1, group, eff_seq))
            softmax_out_thread = np.reshape(cmp_v1, (1, 1, group, eff_seq))
            cmp_attn_out[
            b_idx: (b_idx + 1),
            s1_idx: (s1_idx + 1),
            g_offset: (g_offset + group),
            :,
            ] = np.reshape(
                np.matmul(cmp_scale, act_cmp_v.astype(np.float32)), (1, 1, group, dv)
            )  # (g, dv)

            gen_topk_actual_len = cur_seq
            p_cmp = cmp_softmax_out
            s_cmp_len = p_cmp.shape[-1]
            s_slc = (s_cmp_len + 3) // 4
            trans0, reduce0, trans1, reduce1 = gen_p_slc_ast(softmax_out_thread)

            reduce1 = reduce1.reshape(1, 1, s_slc)
            tmp_s_smp = (gen_topk_actual_len - 32) // 16 + 1
            tmp_s_slc = (tmp_s_smp + 3) // 4

            topk_indices, _ = gen_topk_indices(reduce1, actual_len=tmp_s_slc)
            p_cmp_path = Path(out_path, 'p_cmp_compress.bin')
            p_cmp.astype(q_dtype).tofile(p_cmp_path)
            dump_file(p_cmp, p_cmp_path, dtypeStr)
            topk_indices_path = Path(out_path, 'topk_indices_compress.bin')
            topk_indices.astype(np.float32).tofile(topk_indices_path)
            topk_full[b_idx,s1_idx ,:13] = topk_indices
            topk_input[b_idx,s1_idx ,:,:] = reduce1



def compress_attention_compute(out_path: Path, input_data_map, params):
    q_bsnd = input_data_map.get("q_bsnd")
    k_bsnd = input_data_map.get("k_bsnd")
    k_tensor_out = input_data_map.get("k_tensor_out")
    mlp_cos = input_data_map.get("mlp_cos")
    mlp_sin = input_data_map.get("mlp_sin")
    first_rope = input_data_map.get("first_rope")
    first_rope_input = input_data_map.get("first_rope_input")
    k_cmp_tensor = input_data_map.get("k_cmp_tensor")
    cmp_softmax_out = input_data_map.get("cmp_softmax_out")
    cmp_attn_out = input_data_map.get("cmp_attn_out")
    cmp_block_table = input_data_map.get("cmp_block_table")
    cmp_block_num = input_data_map.get("cmp_block_num")
    wk1 = input_data_map.get("wk1")
    wk2 = input_data_map.get("wk2")
    act_seq = input_data_map.get("act_seq")
    act_cmp_seq = input_data_map.get("act_cmp_seq")
    topk_full = input_data_map.get("topk_full")
    topk_input = input_data_map.get("topk_input")
    b = params.get("b")
    dv = params.get("dv")
    s2 = params.get("s2")
    n2 = params.get("n2")
    n1 = params.get("n1")
    d_qk = params.get("d")
    cmp_stride = params.get("cmp_stride")
    k_dtype = params.get("dtype")
    q_dtype = params.get("dtype")
    cmp_bs = params.get("cmp_block_size")
    s = params.get("s")
    softmax_scale = params.get("softmax_scale")
    block_size = params.get("block_size")
    scmp = params.get("scmp")
    group = n1 // n2
    dtypeStr = "bf16" if q_dtype == bfloat16 else "fp16"


    import threading

    threads = []

    thread_params = \
        q_dtype, dtypeStr, k_bsnd, k_tensor_out, dv, act_seq, b, s2, n2, d_qk, mlp_cos, mlp_sin, act_cmp_seq, k_dtype, s, cmp_stride, cmp_bs, wk1, wk2, first_rope, first_rope_input, k_cmp_tensor, group, q_bsnd, out_path, softmax_scale, cmp_softmax_out, cmp_attn_out ,topk_full, topk_input

    for b_idx in range(b):
        thread = threading.Thread(target=thread_func, args=(b_idx, thread_params))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    k_cmp_bsnd = np.reshape(k_cmp_tensor, (b, scmp, n2, d_qk))
    k_cmp_cache = gen_cache_tensor(
        k_cmp_bsnd, cmp_block_table, cmp_block_num, block_size, b
    )  # (cmp_block_num * block_size, n2 * d_qk)
    cmp_kv_cache = k_cmp_cache[..., : dv]
    cmp_kr_cache = k_cmp_cache[..., dv:]

    k_cmp_out = k_cmp_tensor[:, :act_cmp_seq[0]].astype(np.float32)

    # # input params

    cmp_kv_cache_path = Path(out_path, "cmp_kv_cache_compress.bin")
    cmp_kr_cache_path = Path(out_path, "cmp_kr_cache_compress.bin")
    cmp_attn_path = Path(out_path, "cmp_attn_compress.bin")
    cmp_attn16_path = Path(out_path, "cmp_attn16_compress.bin")
    cmp_softmax_path = Path(out_path, "cmp_softmax_compress.bin")
    k_tensor_path = Path(out_path, "k_tensor_out_compress.bin")
    k_cmp_path = Path(out_path, "k_cmp_out_compress.bin")
    first_rope_path = Path(out_path, "first_rope_compress.bin")
    first_rope_input_path = Path(out_path, "first_rope_input_compress.bin")
    topk_full_path = Path(out_path, "topk_full.bin")
    topk_input_path = Path(out_path, "topk_input.bin")

    dump_file(cmp_kv_cache, cmp_kv_cache_path, dtypeStr)
    dump_file(cmp_kr_cache, cmp_kr_cache_path, dtypeStr)
    dump_file(cmp_attn_out, cmp_attn_path, "fp32")
    dump_file(cmp_attn_out, cmp_attn16_path, "fp16")
    dump_file(cmp_softmax_out, cmp_softmax_path, "fp32")
    dump_file(k_tensor_out, k_tensor_path, dtypeStr)
    dump_file(k_cmp_out, k_cmp_path, "fp32")
    dump_file(first_rope, first_rope_path, dtypeStr)
    dump_file(first_rope_input, first_rope_input_path, dtypeStr)
    dump_file(topk_full, topk_full_path, "int32")
    dump_file(topk_input, topk_input_path, "fp32")

    logging.info("Finish gen golden of fused_compress_kv_select!")
    return cmp_attn_out,topk_full


def fused_cmp_kv_sel(out_path: Path, params, other_input=None):
    # define input files
    input_data_map = compress_attention_data_gen(out_path, params,other_input)
    compress_attention_compute(out_path, input_data_map, params)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicCmpKvSel.dynamic_NSA_case_no_flash",
        "DynamicCmpKvSel.debug_dynamic_NSA_case_no_flash",
    ]
)
def fuse_compress_kv_select(case_name: str, output: Path) -> bool:
    # init initial params
    b, s1, n1, d_qk, q_dtype = 32, 2, 128, 576, np.float16
    s2, n2, d_qk, dv, k_dtype = 8192, 8, 576, 512, np.float16
    act_seq = [s2 for _ in range(b)]
    block_size = 128
    cmp_block_size = 32
    cmp_stride = 16
    if case_name.startswith("DynamicCmpKvSel.dynamic_NSA_case_no_flash"):
        b = 2
        s1 = 1
        n1 = 128
        # d_qk = 192
        n2 = 1
        # dv = 128
        s2 = 1024
        act_seq = [s2 for _ in range(b)]
    if case_name.startswith("DynamicCmpKvSel.debug_dynamic_NSA_case_no_flash"):
        b = 1
        s1 = 1
        n1 = 2
        d_qk = 192
        n2 = 1
        dv = 128
        s2 = 8
        cmp_block_size = 8
        cmp_stride = 4
        act_seq = [s2 for _ in range(b)]
        logging.info("get func to gen golden, Case(%s)", case_name)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    params = {}
    params["b"] = b
    params["s"] = s1
    params["n1"] = n1
    params["d"] = d_qk
    params["dtype"] = q_dtype
    params["s2"] = s2
    params["n2"] = n2
    params["dv"] = dv
    params["act_seq"] = act_seq
    params["block_size"] = block_size
    params["cmp_block_size"] = cmp_block_size
    params["cmp_stride"] = cmp_stride
    params["softmax_scale"] = float(1.0) / np.sqrt(d_qk)
    act_cmp_seq = [
        (cur_s - cmp_block_size) // cmp_stride + 1 for cur_s in act_seq
    ]
    params["act_cmp_seq"] = act_cmp_seq
    print("act_cmp_seq: ", act_cmp_seq)
    scmp = max(act_cmp_seq)
    print("scmp: ", scmp)
    params["scmp"] = scmp
    fused_cmp_kv_sel(output, params)
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicCmpKvSel.dynamic_NSA_case_no_flash",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = fuse_compress_kv_select(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
