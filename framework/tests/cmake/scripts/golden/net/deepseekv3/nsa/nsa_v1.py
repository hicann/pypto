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
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import math
import sys
import logging
from pathlib import Path
from typing import List
import time

import numpy as np
from ml_dtypes import bfloat16
import os
# np.random.seed(0)
# 添加 golden 所在目录的父路径（例如项目根目录）
project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)

from golden.net.deepseekv3.nsa.gen_slc_attn import compute_attention
from golden.op.kv_slc import kv_slc_compute
from golden.net.deepseekv3.nsa.attention_post_golden import PostConfig, post_compute, gen_post_input_data
from golden.net.deepseekv3.nsa.win_atten import win_attn_calc
from golden.net.deepseekv3.mla.mla_prolog_golden_v2 import gen_prolog_input_data, mla_prolog_compute, gen_block_table
from golden.net.deepseekv3.nsa.gen_fused_compress_kv_select import compress_attention_data_gen, compress_attention_compute



if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gated_score_mlp_standard(x, w_1, w_2, output: Path):
    b, s, h = x.shape
    _, n3 = w_2.shape
    n = n3 // 3
    # if True:
    #     w_1 = np.random.rand(h, 4 * h)
    #     w_2 = np.random.rand(4*h, 3 * n)
    print(f'b {b} s {s} h {h} n {n} \n')
    # x_path = Path(output, 'x.bin')
    w1_path = Path(output, 'w1.bin')
    w2_path = Path(output, 'w2.bin')
    score_path = Path(output, 'score.bin')
    _, n_heads = w_2.shape
    n = n_heads // 3
    x_2d = x.reshape(-1, h)
    mm1 = np.matmul(x_2d, w_1)
    mm1_sigmoid = sigmoid(mm1)
    mm2 = np.matmul(mm1_sigmoid.astype(w_2.dtype), w_2)
    gating_score = mm2.reshape(b, s, 3, n)
    return gating_score, mm1_sigmoid, mm2


def gated_score_mlp_simple(x, w_1, output: Path):
    b, s, h = x.shape
    _, n_heads = w_1.shape
    n = n_heads // 3
    x_2d = x.reshape(-1, h)
    mm1 = np.matmul(x_2d, w_1)
    mm1_sigmoid = sigmoid(mm1)
    gating_score = mm1_sigmoid.reshape(b, s, n, 3)

    return gating_score


def gen_gated_score(x, gate_sim_w1, gate_w1, gate_w2, output: Path, mode='standard'):
    if mode == 'standard':
        gating_score, mm1, mm2 = gated_score_mlp_standard(x, gate_w1, gate_w2, output)
    else:
        gating_score = gated_score_mlp_simple(x, gate_sim_w1, output)

    gating_score = gating_score.transpose((0, 1, 3, 2))
    return gating_score, mm1, mm2


def dump_file(data_pool, data_path, dtype):
    np.array(data_pool).astype(dtype).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def gen_kv_cache(params, actual_seq_list, dtype, output_dir):
    '''
    生成kv_cache, 包括kv_nope_cache, k_rope_cache, block_table
    '''
    b = params.get("b")
    s1 = params.get("s")
    n2 = params.get("n2")
    rope_dim = params.get("rope_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    front = params.get("front")
    near = params.get("near")
    topk = params.get("topk")
    block_size = params.get("block_size")

    block_num, block_table = gen_block_table(b, actual_seq_list, block_size)

    shape_topk_indices = [b, s1, topk - front - near]
    # shape_kv_nope_cache = [block_num * block_size, n2 * kv_lora_rank]
    # shape_k_rope_cache = [block_num * block_size, n2 * rope_dim]

    # kv_nope_cache = gen_uniform_data(shape_kv_nope_cache, -1, 1, dtype)
    # k_rope_cache = gen_uniform_data(shape_k_rope_cache, -1, 1, dtype)

    # kv_nope_cache_path = Path(output_dir, 'kv_nope_cache.bin')
    # kr_cache_path = Path(output_dir, 'k_rope_cache.bin')
    block_table_path = Path(output_dir, 'block_table.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    # dump_file(kv_nope_cache, kv_nope_cache_path, dtype)
    # dump_file(k_rope_cache, kr_cache_path, dtype)
    dump_file(block_table, block_table_path, np.int32)
    dump_file(actual_seq_list, kv_cache_actual_seq_path, np.int32)

    # return kv_nope_cache, k_rope_cache, block_table
    return block_table, block_num


def gen_atten_golden_data(cmp_atten, sel_atten, win_atten, gating_score, dtype):
    '''
    内部以fp32进行运算
    '''
    fp32 = np.float32
    cmp_atten_fp32 = cmp_atten.astype(fp32)
    sel_atten_fp32 = sel_atten.astype(fp32)
    win_atten_fp32 = win_atten.astype(fp32)
    gating_score_fp32 = gating_score.astype(fp32)
    w_cmp, w_slc, w_win = np.split(gating_score_fp32, 3, axis = -1)
    attention_out_fp32 = (w_cmp * cmp_atten_fp32 + w_slc * sel_atten_fp32 + w_win * win_atten_fp32)
    attention_out = attention_out_fp32.astype(dtype)
    return attention_out


def dump_gen_win_attn_file(win_attn, output_dir):
    win_attn_path = Path(output_dir, 'winAttn.bin')
    dump_file(win_attn, win_attn_path, np.float32)


def dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, kv_slc_out, kr_slc_out, kv_slc_actual_seqs, dtype, output_dir):
    topk_tensor_path = Path(output_dir, 'topk_tensor.bin')
    topk_tensor_shape_path = Path(output_dir, 'topk_tensor_shape.bin')
    kv_slc_out_path = Path(output_dir, 'kv_slc_out.bin')
    kr_slc_out_path = Path(output_dir, 'kr_slc_out.bin')
    kv_slc_actual_seqs_path = Path(output_dir, 'kv_slc_actual_seqs.bin')

    dump_file(topk_indices, topk_tensor_path, np.int32)
    dump_file(topk_tensor_shape, topk_tensor_shape_path, np.int32)
    dump_file(kv_slc_out, kv_slc_out_path, dtype)
    dump_file(kr_slc_out, kr_slc_out_path, dtype)
    dump_file(kv_slc_actual_seqs, kv_slc_actual_seqs_path, np.int32)


def dump_slc_atten_file(q_bsnd, k_bsnd, v_bsnd, actual_seq, kv_lora_rank, dtype, output_dir, atten_out, input_params):
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]

    q_nope_path = Path(output_dir, 'q_nope.bin')
    q_rope_path = Path(output_dir, 'q_rope.bin')
    k_slc_path = Path(output_dir, 'k_slc.bin')
    v_slc_path = Path(output_dir, 'v_slc.bin')
    # actual_seq_path = Path(output_dir, 'slc_actual_seq.bin')

    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(k_bsnd, k_slc_path, dtype)
    dump_file(v_bsnd, v_slc_path, dtype)
    # dump_file(actual_seq, actual_seq_path, np.int32)


def dump_gen_atten_file(cmp_atten, sel_atten, win_atten, attention_out, dtype, output_dir):
    cmp_atten_path = Path(output_dir, 'cmp_atten.bin')
    sel_atten_path = Path(output_dir, 'sel_atten.bin')
    win_atten_path = Path(output_dir, 'win_atten.bin')
    attention_out_path = Path(output_dir, 'attention_out.bin')

    dump_file(cmp_atten, cmp_atten_path, dtype)
    dump_file(sel_atten, sel_atten_path, np.float32) # slc_attn, fp32
    dump_file(win_atten, win_atten_path, dtype)
    dump_file(attention_out, attention_out_path, dtype)


def dump_gated_score_file(gate_sim_w1, gate_w1, gate_w2, gating_score, dtype, output_dir):
    # x_path = Path(output_dir, 'x.bin')
    gate_sim_w1_path = Path(output_dir, 'gate_sim_w1.bin')
    gate_w1_path = Path(output_dir, 'gate_w1.bin')
    gate_w2_path = Path(output_dir, 'gate_w2.bin')
    gating_score_path = Path(output_dir, 'gating_score.bin')

    # x.astype(dtype).tofile(x_path)
    gate_sim_w1.astype(dtype).tofile(gate_sim_w1_path)
    gate_w1.astype(dtype).tofile(gate_w1_path)
    gate_w2.astype(dtype).tofile(gate_w2_path)
    gating_score.astype(dtype).tofile(gating_score_path)


def kv_cache_concat_bsnd(concat_parms, kr_cache_out, kv_cache_out, block_table, block_num, kv_cache_actual_seq, dtype):
    b = concat_parms[0]
    s = concat_parms[1]
    n2 = concat_parms[2]
    kv_lora_rank = concat_parms[3]
    rope_dim= concat_parms[4]
    block_size = concat_parms[5]

    kv_max = max(kv_cache_actual_seq)
    k_cache = np.zeros([b, kv_max, n2, kv_lora_rank], dtype = dtype)
    v_cache = np.zeros([b, kv_max, n2, rope_dim], dtype = dtype)
    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = np.zeros([1, kv_max, n2, kv_lora_rank], dtype = dtype)
        kv_rope_temp_tensor = np.zeros([1, kv_max, n2, rope_dim], dtype = dtype)
        s_idx = 0
        for _, block_idx in enumerate(block_list):
            kv_nope_temp_tensor[:, s_idx * block_size : (s_idx + 1) * block_size, :, :] = kv_cache_out[block_idx : block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, s_idx * block_size : (s_idx + 1) * block_size, :, :] = kr_cache_out[block_idx : block_idx + 1, :, :, :]
            s_idx += 1
        k_cache[b_idx : b_idx + 1, :, :, :] = kv_nope_temp_tensor
        v_cache[b_idx : b_idx + 1, :, :, :] = kv_rope_temp_tensor
    k_cache_bsnd = np.concatenate([k_cache, v_cache], axis = -1)
    v_cache_bsnd = k_cache
    return k_cache_bsnd, v_cache_bsnd


def gen_nsa_golden(params, dtypes, output_dir: Path, is_nz=False):
    '''
    将整个nsa分为6个子图进行串联
    subgragh 1: gen_win_attn
    subgragh 2: kv_compression
    subgragh 3: gen_cmp_atten
    subgragh 4: gen_slc_atten, 其中包括: gen_kv_slc及slc_attn
    subgragh 5: gen_gated_score
    subgragh 6: gen_attn
    subgragh 7: post
    '''
    print("=========== start =============: nsa golden")

    dtype, w_dtype = dtypes
    logging.debug(f"gen_nsa_golden  dtype:{dtype}, w_dtype:{w_dtype}")
    b = params.get("b")
    s = params.get("s")
    s1 = params.get("s")
    s2 = params.get("s2")
    h = params.get("h")
    n1 = params.get("n1")
    n2 = params.get("n2")
    q_dim = params.get("q_dim")
    k_dim = params.get("k_dim")
    v_dim = params.get("v_dim")
    rope_dim = params.get("rope_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    cmp_block_size = params.get("cmp_block_size")
    cmp_stride = params.get("cmp_stride")
    slc_block_size = params.get("slc_block_size")
    front = params.get("front")
    near = params.get("near")
    topk = params.get("topk")
    block_size = params.get("block_size")
    win_size = params.get("win_size")
    epsilon = params.get("epsilon")
    cache_mode = params.get("cache_mode")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    v_head_dim = params.get("v_head_dim")
    is_quant = params.get("is_quant")
    has_smooth = params.get("is_smooth")

    softmax_scale = q_dim ** -0.5
    slc_s_max = topk * slc_block_size

    # kv cache actual_seq
    kv_cache_actual_seq_p = params.get("kv_cache_actual_seq")
    if isinstance(kv_cache_actual_seq_p, int):
        kv_cache_actual_seq = [kv_cache_actual_seq_p] * b
    elif isinstance(kv_cache_actual_seq_p, list):
        if len(kv_cache_actual_seq_p) == b:
            kv_cache_actual_seq = kv_cache_actual_seq_p
        else:
            raise RuntimeError("unsupported this kv_cache_actual_seq")
    else:
        raise RuntimeError("unsupported kv_cache_actual_seq data type")
    skv_max = max(kv_cache_actual_seq)
    # 1. 设置shape
    # gen kv_slc
    shape_topk_indices = [b, s, topk - front - near]

    # gen slc atten
    # slc_q_shape = [b, s, n1, q_dim]
    slc_k_shape = [b, s, n2, slc_s_max, k_dim]
    slc_v_shape = [b, s, n2, slc_s_max, v_dim]
    slc_atten_out_shape = [b, s, n1, v_dim]
    # gen gated_score

    x_shape = [b, s, h]
    gate_sim_w1_shape = [h, n1 * 3]
    gate_w1_shape = [h, h * 4]
    gate_w2_shape = [h * 4, n1 * 3]

    # gen attn
    cmp_atten_shape = [b, s, n1, v_dim]
    sel_atten_shape = [b, s, n1, v_dim]
    win_atten_shape = [b, s, n1, v_dim]
    gating_score_shape = [b, s, n1, 3]

    np.random.seed(int(time.time()))

    # 2. 生成数据
    # mla_prolog
    block_table, block_num = gen_kv_cache(params, kv_cache_actual_seq, dtype, output_dir) # 生成 block_table

    prolog_params = {
        "b": b,
        "s": s,
        "s2": s2,
        "h": h,
        "num_heads": n1,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": rope_dim,
        "kv_lora_rank": kv_lora_rank,
        "v_head_dim": v_head_dim,
        "block_num": block_num,
        "block_table": block_table,
        "skv_max": skv_max,
    }
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache = \
        gen_prolog_input_data(prolog_params, [dtype, dtype], epsilon, output_dir, (False, is_quant), is_nz, has_smooth,
                              block_size, cache_mode)

    # 计算公式：s_slc = (act_seq_len[bIdx] + s1Idx - s1 + 1 - cmp_block_size + slc_block_size) // slc_block_size
    topk_tensor_shape = np.zeros([b, s], dtype=np.int32)
    topk_indices = np.zeros(shape_topk_indices, dtype=np.int32)
    for bIdx in range(b):
        for s1Idx in range(s):
            s_slc = (kv_cache_actual_seq[bIdx] + s1Idx - s + 1 - cmp_block_size + slc_block_size) // slc_block_size
            topk_tensor_shape[bIdx][s1Idx] = s_slc
            topk_indices[bIdx][s1Idx] = gen_uniform_data([shape_topk_indices[-1]], 0, s_slc, dtype=np.int32)

    # gen gated_score
    # x = gen_uniform_data(x_shape, -1, 1, dtype)
    gate_sim_w1 = gen_uniform_data(gate_sim_w1_shape, -0.1, 0.1, dtype)
    gate_w1 = gen_uniform_data(gate_w1_shape, -0.1, 0.1, dtype)
    gate_w2 = gen_uniform_data(gate_w2_shape, -0.1, 0.1, dtype)

    # gen attn
    cmp_atten = np.random.uniform(-1, 1, cmp_atten_shape).astype(dtype)
    # slc_atten = np.random.uniform(-1, 1, sel_atten_shape).astype(dtype)
    # win_atten = np.random.uniform(-1, 1, win_atten_shape).astype(dtype)

    # post
    post_config = PostConfig((b, n1, s, h, kv_lora_rank, v_head_dim), dtype, False, False, is_quant, has_smooth, is_nz)
    w_uv, w_uv_scale, smooth_wuv, w_o, w_o_scale, smooth_wo = gen_post_input_data(output_dir, post_config)

    # 3. 计算 & dump file
    # mla_prolog
    prolog_inputs = {"dtype": dtype, "is_quant_a": False, "is_quant_b": is_quant, "has_smooth": has_smooth}
    prolog_inputs["cache_mode"] = cache_mode
    prolog_inputs["gamma_cq"] = gamma_cq
    prolog_inputs["gamma_ckv"] = gamma_ckv
    prolog_inputs["epsilon"] = epsilon
    prolog_inputs["x"] = x
    prolog_inputs["w_dq"] = w_dq
    prolog_inputs["w_uqqr"] = w_uqqr
    prolog_inputs["w_uk"] = w_uk
    prolog_inputs["w_dkvkr"] = w_dkvkr
    prolog_inputs["cos"] = cos
    prolog_inputs["sin"] = sin
    prolog_inputs["kv_cache"] = kv_cache
    prolog_inputs["kr_cache"] = kr_cache
    prolog_inputs["cache_index"] = kv_len
    if is_quant:
        prolog_inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            prolog_inputs["smooth_cq"] = smooth_cq
    # q_out: [b, s, n1, kv_lora_rank], q_rope_out: [b, s, n1, rope_dim]
    # kv_cache_out: [block_num, block_size, n2, kv_lora_rank], kr_cache_out: [block_num, block_size, n2, rope_dim]
    q_out, q_rope_out, kv_cache_out, kr_cache_out = mla_prolog_compute(prolog_inputs)
    q_out.tofile(Path(output_dir, 'q_golden.bin'))
    q_rope_out.tofile(Path(output_dir, 'q_rope_golden.bin'))
    kv_cache_out.tofile(Path(output_dir, 'kv_cache_golden.bin'))
    kr_cache_out.tofile(Path(output_dir, 'kr_cache_golden.bin'))

    # reshape
    kv_nope_cache = kv_cache_out.reshape([block_num * block_size, n2 * kv_lora_rank])
    k_rope_cache = kr_cache_out.reshape([block_num * block_size, n2 * rope_dim])

    q_bsnd = np.concatenate([q_out, q_rope_out], axis=-1)  # [b, s, n1, kv_lora_rank + rope_dim]
    concat_parms = [b, s, n2, kv_lora_rank, rope_dim, block_size]
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(concat_parms, kr_cache_out, kv_cache_out, block_table, block_num, kv_cache_actual_seq, dtype)

    # kv compression
    # gen_kv_compression()

    # cmp atten
    # gen_cmp_attn()
    params = {}
    b_compress = b
    s2_compress = s2
    params["b"] = b_compress
    params["s"] = s1
    params["n1"] = n1
    params["d"] = q_dim
    params["dtype"] = dtype
    params["s2"] = s2_compress
    params["n2"] = n2
    params["dv"] = v_dim
    act_seq = [s2 for _ in range(b_compress)]
    params["act_seq"] = act_seq
    params["block_size"] = block_size
    params["cmp_block_size"] = cmp_block_size
    params["cmp_stride"] = cmp_stride
    params["softmax_scale"] = float(1.0) / np.sqrt(q_dim)
    act_cmp_seq = [
        (cur_s - cmp_block_size) // cmp_stride + 1 for cur_s in act_seq
    ]
    params["act_cmp_seq"] = act_cmp_seq
    scmp = max(act_cmp_seq)
    params["scmp"] = scmp

    input_data_map = compress_attention_data_gen(output_dir, params, (q_bsnd,k_cache_bsnd,block_table))
    # compress_attention_out = compress_attention_compute(output_dir, input_data_map, params)
    compress_attention_out, topk_full = compress_attention_compute(output_dir, input_data_map, params)



# win atten
    win_atten = np.zeros(win_atten_shape, dtype = np.float32)
    input_params_win_attn = [b, s, n2, n1, q_dim, win_size, k_dim, v_dim, softmax_scale]
    win_attn_calc(input_params_win_attn, kv_cache_actual_seq, q_bsnd, k_cache_bsnd, v_cache_bsnd, dtypes, win_atten)
    dump_gen_win_attn_file(win_atten, output_dir)

    # gen kv_slc
    print("========== gen kv_slc ==============")
    compute_input_params = [block_size, n2, front, near, topk, slc_block_size]
    # k_slc_out, v_slc_out, kv_slc_actual_seqs = kv_slc_compute(compute_input_params, topk_indices, topk_tensor_shape, kv_nope_cache, k_rope_cache, block_table, kv_cache_actual_seq)
    k_slc_out, v_slc_out, kv_slc_actual_seqs = kv_slc_compute(compute_input_params, topk_full, topk_tensor_shape, kv_nope_cache, k_rope_cache, block_table, kv_cache_actual_seq)
    dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, k_slc_out, v_slc_out, kv_slc_actual_seqs, dtype, output_dir)

    # slc atten
    print("========== gen slc_attn ==============")
    input_params = [b, s, n1, n2, kv_lora_rank, rope_dim, slc_s_max, slc_s_max]
    k_slc = np.reshape(k_slc_out, slc_k_shape) # [b*s*n2*slc_s_max, k_dim] -> [b, s, n2, slc_s_max, k_dim]
    v_slc = np.reshape(v_slc_out, slc_v_shape) # [b*s*n2*slc_s_max, v_dim] -> [b, s, n2, slc_s_max, v_dim]
    slc_atten = compute_attention(q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, softmax_scale, slc_atten_out_shape) # 输出fp64
    dump_slc_atten_file(q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, kv_lora_rank, dtype, output_dir, slc_atten, input_params)

    # gen gated_score
    print("========== gen gated_score ==============")
    gating_score, _, _ = gen_gated_score(x, gate_sim_w1, gate_w1, gate_w2, output_dir, mode='standard') # 升精度运算
    dump_gated_score_file(gate_sim_w1, gate_w1, gate_w2, gating_score, dtype, output_dir)

    # gen atten
    print("========== gen attn ==============")
    cmp_atten = compress_attention_out.reshape(cmp_atten_shape)
    attention_out = gen_atten_golden_data(cmp_atten, slc_atten, win_atten, gating_score, dtype)
    dump_gen_atten_file(cmp_atten, slc_atten, win_atten, attention_out, dtype, output_dir)
    print("========== attention_out: ", attention_out.shape, attention_out.dtype)

    # post
    print("========== gen post output ==============")
    post_inputs = {"dtype": dtype, "is_quant_w_uv": False, "has_smooth_w_uv": False, "is_quant_w_o": is_quant, "has_smooth_w_o": has_smooth}
    post_inputs["x"] = attention_out
    post_inputs["w_uv"] = w_uv
    post_inputs["w_o"] = w_o
    if is_quant:
        post_inputs["w_o_scale"] = w_o_scale
        if has_smooth:
            post_inputs["smooth_w_o"] = smooth_wo
    post_out = post_compute(post_inputs)

    # dump output to file
    output_path = Path(output_dir, 'golden_output.bin')
    post_out.tofile(output_path)

    return True


def nsa_entry(dtypes, bs1s2h, quant_smooth, output_dir: Path):
    b, s1, s2, h = bs1s2h
    is_quant, is_smooth = quant_smooth
    kv_lora_rank = 512
    rope_dim = 64
    q_dim = kv_lora_rank + rope_dim
    k_dim = kv_lora_rank + rope_dim
    v_dim = kv_lora_rank
    topk = 16
    slc_block_size = 64
    v_head_dim = 128
    epsilon = 1e-5
    cache_mode = "PA_BSND"

    params = {
        "b": b,
        "s": s1,
        "s2": s2,
        "n1": 128,
        "n2": 1,
        "h": h,
        "q_lora_rank": 1536,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rope_dim": rope_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "cmp_block_size": 32,
        "cmp_stride": 16,
        "slc_block_size": slc_block_size,
        "front": 1,
        "near": 2,
        "topk": topk,
        "block_size": 128,
        "win_size": 512,
        "kv_cache_actual_seq": s2,
        "epsilon": epsilon,
        "cache_mode": cache_mode,
        "v_head_dim": v_head_dim,
        "is_quant": is_quant,
        "is_smooth": is_smooth,
    }
    gen_nsa_golden(params, dtypes, output_dir)

    # 将变化的参数保存到文件中，供测试用例直接读取
    input_params = [params.get("b"), params.get("s"), params.get("s2"), params.get("n1"), params.get("n2")]
    input_params.append(1 if is_quant else 0)
    input_params.append(1 if is_smooth else 0)
    dump_file(input_params, Path(output_dir, 'input_params.bin'), np.int32)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicNSATest.nsa_b_16_s1_1_s2_8192_h_7168_fp16",
        "DynamicNSATest.nsa_b_1_s1_1_s2_8192_h_7168_fp16",
        "DynamicNSATest.nsa_b_16_s1_1_s2_8192_h_7168_fp16_quant",
        "DynamicNSATest.s2_1024",
        "DynamicNSATest.s2_2048",
        "DynamicNSATest.s2_8192",
        "DynamicNSATest.s2_4096",
        "DynamicNSATest.mini",
        "DynamicNSATest.mini_debug",
    ]
)
def gen_nsa_v1_func(case_name: str, output: Path) -> bool:
    input_path = Path(output, 'x.bin')
    complete = input_path.exists()
    if complete:
        file_mod_time = input_path.stat().st_mtime
        # 获取当前时间（Unix 时间戳）
        current_time = time.time()
        # 判断文件的修改时间是否超过1小时（3600秒）
        if current_time - file_mod_time > 3600 * 24 *30:
            logging.info("文件的修改时间超过1小时，重新生成文件...")
            complete = False
        else:
            logging.info("文件的修改时间在1小时内，无需重新生成。")

    # complete = False if case_name != "DynamicNSATest.mini" else True # TODO: del complete
    if complete:
        logging.info("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "DynamicNSATest.nsa_b_16_s1_1_s2_8192_h_7168_fp16": # gen_slc_attn + gen_gated_score + gen_attn nsa_b_16_s1_1_s2_8192_h_7168_fp16
            nsa_entry((np.float16, np.float16), (16, 1, 1024, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.nsa_b_16_s1_1_s2_8192_h_7168_fp16_quant": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 8192, 7168), (True, True), output)
        elif case_name == "DynamicNSATest.nsa_b_1_s1_1_s2_8192_h_7168_fp16": # quant
            nsa_entry((np.float16, np.float16), (1, 1, 1024, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.s2_1024": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 1024, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.s2_2048": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 2048, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.s2_8192": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 8192, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.s2_4096": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 4096, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.mini": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 1024, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.mini_debug": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 1024, 7168), (False, False), output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicNSATest.s2_1024",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = gen_nsa_v1_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
