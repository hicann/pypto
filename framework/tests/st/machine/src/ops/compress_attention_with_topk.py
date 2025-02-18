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
from enum import Enum
import math
import os
import sys
import logging
import torch
from pathlib import Path
from typing import List
from common_func import dump_file

import numpy as np
from ml_dtypes import bfloat16
import copy


project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)

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


def gen_cache_tensor(k_tensor, block_table, block_num, block_size, b):
    logging.info("Entering into gen_cache_tensor!")
    dtype = k_tensor.dtype
    b, s, n, d = k_tensor.shape
    k_cache = torch.zeros([block_num, block_size, n * d], dtype=dtype)
    k_tensor_bsh_raw = k_tensor.reshape(b, s, n * d)

    # kv padding
    k_tensor_bsh = torch.zeros(
        (b, block_table.shape[1] * block_size, n * d), dtype=dtype
    )
    k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx != -1:
                k_cache[cache_block_idx, :, :] = k_tensor_bsh[
                    b_idx, block_offset : (block_offset + block_size), :
                ]

    k_cache = k_cache.reshape(block_num * block_size, n, d)
    return k_cache


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


def gen_data_for_compute(
    out_path: Path, params, force_update=True, kv_compress_params=None
):
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    dn = params.get("dn")
    dr = params.get("dr")
    dtype = params.get("dtype")
    s2 = params.get("s2")
    n2 = params.get("n2")
    block_size = params.get("block_size")
    cmp_block_size = params.get("cmp_block_size")
    cmp_stride = params.get("cmp_stride")
    slc_block_size = params.get("slc_block_size")
    softmax_scale = params.get("softmax_scale")
    topk = params.get("topk")
    front = params.get("front")
    near = params.get("near")
    act_seq_len = params.get("act_seq")
    act_cmp_seq = params.get("act_cmp_seq")
    s_cmp_max = params.get("s_cmp_max")

    # define input files
    in_params_path = Path(out_path, "in_params.bin")
    q_nope_path = Path(out_path, "q_nope.bin")
    q_rope_path = Path(out_path, "q_rope.bin")
    k_bsnd_path = Path(out_path, "k.bin")
    cmp_kv_cache_path = Path(out_path, "cmp_kv_cache.bin")
    cmp_kr_cache_path = Path(out_path, "cmp_kr_cache.bin")
    cmp_block_table_path = Path(out_path, "cmp_block_table.bin")
    act_seq_path = Path(out_path, "act_seq.bin")
    aux_tensor_path = Path(out_path, "aux_tensor.bin")

    # construct input shape
    if os.path.exists(q_nope_path) and force_update == False:
        q_nope = torch.from_numpy(
            np.fromfile(q_nope_path, dtype=bfloat16)
            .reshape([b * s1 * n1, dn])
            .astype(float)
        ).to(torch.bfloat16)
        q_rope = torch.from_numpy(
            np.fromfile(q_rope_path, dtype=bfloat16)
            .reshape([b * s1 * n1, dr])
            .astype(float)
        ).to(torch.bfloat16)
        k_bsnd = torch.from_numpy(
            np.fromfile(k_bsnd_path, dtype=bfloat16)
            .reshape([b, s_cmp_max, n2, (dn + dr)])
            .astype(float)
        ).to(torch.bfloat16)
    else:
        q_nope = torch.randn([b * s1 * n1, dn], dtype=dtype)
        q_rope = torch.randn([b * s1 * n1, dr], dtype=dtype)
        k_bsnd = torch.randn([b, s_cmp_max, n2, (dn + dr)], dtype=dtype)
    cmp_block_num, cmp_block_table_list = gen_block_table(
        b, block_size, s_cmp_max, act_cmp_seq
    )
    cmp_block_table = torch.tensor(cmp_block_table_list, dtype=torch.int32)
    act_seq = torch.tensor(act_seq_len, dtype=torch.int32)

    cmp_k_cache = gen_cache_tensor(
        k_bsnd, cmp_block_table_list, cmp_block_num, block_size, b
    )
    cmp_kv_cache = cmp_k_cache[..., :dn].reshape(cmp_block_num * block_size, n2 * dn)
    cmp_kr_cache = cmp_k_cache[..., dn:].reshape(cmp_block_num * block_size, n2 * dr)

    # 需要满足slc_block_size / cmp_stride
    aux_size = slc_block_size // cmp_stride + cmp_block_size // cmp_stride - 1
    aux_tensor = torch.zeros(
        slc_block_size // cmp_stride + cmp_block_size // cmp_stride - 1, n1
    )
    aux_tensor_temp = torch.ones(slc_block_size // cmp_stride, n1)
    for i in range(cmp_block_size // cmp_stride):
        aux_tensor[i : i + slc_block_size // cmp_stride :, :] += aux_tensor_temp

    if kv_compress_params is not None:
        q_nope = (
            kv_compress_params.get("q_bsnd")[:, :dn]
            if "q_bsnd" in kv_compress_params
            else q_nope
        )
        q_rope = (
            kv_compress_params.get("q_bsnd")[:, dn:]
            if "q_bsnd" in kv_compress_params
            else q_rope
        )
        cmp_block_table = kv_compress_params.get("cmp_block_table")
        act_seq = kv_compress_params.get("act_seq")
        cmp_kv_cache = kv_compress_params.get("cmp_kv_cache")
        cmp_kr_cache = kv_compress_params.get("cmp_kr_cache")
        aux_tensor = kv_compress_params.get("aux_tensor")

    # construct output tensor
    cmp_attn_out = torch.zeros([b * s1 * n1, dn], dtype=torch.float32)
    topk_res = torch.zeros([b, s1, topk], dtype=torch.int32)

    input_data_map = {}
    input_data_map["q_nope"] = q_nope
    input_data_map["q_rope"] = q_rope
    input_data_map["k_bsnd"] = k_bsnd
    input_data_map["cmp_kv_cache"] = cmp_kv_cache
    input_data_map["cmp_kr_cache"] = cmp_kr_cache
    input_data_map["cmp_block_table"] = cmp_block_table
    input_data_map["act_seq"] = act_seq
    input_data_map["aux_tensor"] = aux_tensor
    input_data_map["cmp_attn_out"] = cmp_attn_out
    input_data_map["topk_res"] = topk_res

    input_params = [
        b,
        s1,
        n1,
        dn,
        dr,
        n2,
        block_size,
        cmp_block_size,
        cmp_stride,
        slc_block_size,
        topk,
        front,
        near,
    ]

    # dump golden file
    dump_file(input_params, in_params_path, "int32")
    dump_file(q_nope.to(torch.float32).numpy().astype(bfloat16), q_nope_path, "bf16")
    dump_file(q_rope.to(torch.float32).numpy().astype(bfloat16), q_rope_path, "bf16")
    dump_file(k_bsnd.to(torch.float32).numpy().astype(bfloat16), k_bsnd_path, "bf16")
    dump_file(
        cmp_kv_cache.to(torch.float32).numpy().astype(bfloat16),
        cmp_kv_cache_path,
        "bf16",
    )
    dump_file(
        cmp_kr_cache.to(torch.float32).numpy().astype(bfloat16),
        cmp_kr_cache_path,
        "bf16",
    )
    dump_file(cmp_block_table.numpy(), cmp_block_table_path, "int32")
    dump_file(act_seq.numpy(), act_seq_path, "int32")
    dump_file(aux_tensor.numpy(), aux_tensor_path, "fp32")

    return input_data_map


def compress_attention_with_topk_compute(input_data_map, params):
    # q_nope: [b*s1*n1, dn]
    # q_rope: [b*s1*n1, dr]
    # cmp_kv_cache: [cmp_block_num * block_size, n2 * dn]
    # cmp_kr_cache: [cmp_block_num * block_size, n2 * dr]
    # cmp_block_table: [b, max_cmp_block]
    # act_seq_len: [b]
    # aux_tensor: [slc_block_size / cmp_stride + cmp_block_size / cmp_stride - 1, n1]
    # cmp_attn_out: [b*s1*n1, dn]
    # topk_res: [b, s1, topk]

    # get compute params
    block_size = params.get("block_size")  # 128
    cmp_block_size = params.get("cmp_block_size")  # 32
    cmp_stride = params.get("cmp_stride")  # 16
    slc_block_size = params.get("slc_block_size")  # 64
    softmax_scale = params.get("softmax_scale")
    topk = params.get("topk")
    front = params.get("front")
    near = params.get("near")
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    dn = params.get("dn")
    dr = params.get("dr")
    dtype = params.get("dtype")

    # get input tensors
    q_nope = input_data_map.get("q_nope")
    q_rope = input_data_map.get("q_rope")
    cmp_kv_cache = input_data_map.get("cmp_kv_cache")
    cmp_kr_cache = input_data_map.get("cmp_kr_cache")
    cmp_block_table = input_data_map.get("cmp_block_table")
    act_seq_len = input_data_map.get("act_seq")
    aux_tensor = input_data_map.get("aux_tensor")

    q_dtype = q_nope.dtype
    k_dtype = cmp_kv_cache.dtype

    dqk = dn + dr
    max_cmp_block = cmp_block_table.shape[1]
    cmp_size = cmp_block_size // cmp_stride  # 2
    slc_size = slc_block_size // cmp_stride  # 4
    block_slc_num = block_size // slc_size  # 32
    slc_window = slc_size + cmp_size - 1

    cmp_attn_out = torch.zeros([b * s1 * n1, dn], dtype=torch.float32)
    topk_res = torch.zeros([b, s1, topk], dtype=torch.float32)
    max_act_seq = int(torch.max(act_seq_len).item())
    p_slc = torch.zeros([b, s1, max_cmp_block * block_slc_num], dtype=torch.float32)

    inc_seq = torch.zeros([max_cmp_block * block_slc_num], dtype=torch.int32)
    for i in range(max_cmp_block * block_slc_num):
        inc_seq[i] = i

    for b_idx in range(b):
        cur_seq = act_seq_len[b_idx]
        for s_idx in range(s1):
            cas_cmp_seq = (
                cur_seq - (s1 - s_idx - 1) - cmp_block_size
            ) // cmp_stride + 1
            cur_cmp_block = (cas_cmp_seq + block_size - 1) // block_size
            slc_loop = (cas_cmp_seq + slc_size - 1) // slc_size
            q_offset = b_idx * s1 * n1 + s_idx * n1

            cur_qn = q_nope[q_offset : (q_offset + n1), :]
            cur_qr = q_rope[q_offset : (q_offset + n1), :]
            cur_q = torch.cat((cur_qn, cur_qr), axis=-1)
            slc_before_g_reduce = torch.zeros(
                [max_cmp_block * block_slc_num, n1], dtype=torch.float32
            )
            local_max_gather = torch.zeros([max_cmp_block, n1], dtype=torch.float32)
            oi_update = torch.zeros([n1, dn], dtype=torch.float32)
            li_update = torch.zeros([1, n1], dtype=torch.float32)
            mi_update = torch.zeros([1, n1], dtype=torch.float32)
            slc_pre = torch.zeros([block_slc_num, n1], dtype=torch.float32)

            for block_idx in range(cur_cmp_block):
                ll = (b_idx * s1 + s_idx) * max_cmp_block + block_idx
                # LOOP(1)
                cur_block_idx = cmp_block_table[b_idx][block_idx]
                cur_valid_seq = min(cas_cmp_seq - block_idx * block_size, block_size)
                cur_slc_loop = (
                    cur_valid_seq + slc_size - 1
                ) // slc_size  # cur_slc_loop <= block_slc_num

                cur_cmp_kv = cmp_kv_cache[
                    cur_block_idx
                    * block_size : (cur_block_idx * block_size + cur_valid_seq),
                    :,
                ]  # (block_size|cur_valid_seq, n2*dn)
                cur_cmp_kr = cmp_kr_cache[
                    cur_block_idx
                    * block_size : (cur_block_idx * block_size + cur_valid_seq),
                    :,
                ]  # (block_size|cur_valid_seq, n2*dr)
                cur_cmp_k = torch.cat(
                    (cur_cmp_kv, cur_cmp_kr), axis=-1
                )  # (block_size|cur_valid_seq, n1 * dqk)
                cur_cmp_v = cur_cmp_kv

                sij = torch.matmul(
                    cur_cmp_k.to(torch.float32), cur_q.t().to(torch.float32)
                ).to(
                    torch.float32
                )  # (block_size|cur_valid_seq, dqk), (n1, dqk) -> (block_size|cur_valid_seq, n1)

                sij_scale = sij * softmax_scale  # (block_size|cur_valid_seq, n1)
                tilda_mij = sij_scale.amax(
                    axis=0, keepdim=True
                )  # (block_size|cur_valid_seq, n1) -> (1, n1)
                t_sub = (
                    sij_scale - tilda_mij
                )  # (block_size|cur_valid_seq, n1), (1, n1) -> (block_size|cur_valid_seq, n1)
                tilda_pij = t_sub.exp()  # (block_size|cur_valid_seq, n1)
                tilda_pij_b16 = tilda_pij.to(k_dtype)  # (block_size|cur_valid_seq, n1)
                tilda_lij = tilda_pij.sum(dim=0, keepdim=True)  # (1, n1)
                if block_idx == 0:
                    oi_tmp = torch.matmul(
                        tilda_pij_b16.t().to(torch.float32), cur_cmp_v.to(torch.float32)
                    ).to(
                        torch.float32
                    )  # (block_size|cur_valid_seq, n1), (block_size|cur_valid_seq, dn) -> (n1, dn)
                    if (
                        block_idx == cur_cmp_block - 1
                    ):  # notice here is not block_idx == cur_cmp_block!!!
                        oi_update = oi_tmp / tilda_lij.reshape(
                            n1, 1
                        )  # (n1, dn), (n1, 1) -> (n1, dn)
                        cmp_attn_out[q_offset : (q_offset + n1), :] = oi_update
                    else:
                        oi_update = oi_tmp
                    li_update = tilda_lij
                    mi_update = tilda_mij
                else:
                    oi = oi_update
                    li = li_update
                    mi = mi_update
                    mi_new = torch.maximum(mi, tilda_mij)  # (1, n1), (1, n1) -> (1, n1)
                    t1 = mi - mi_new
                    t2 = t1.exp()
                    t3 = tilda_mij - mi_new
                    t4 = t3.exp()
                    t5 = t4 * tilda_lij
                    t6 = t2 * li
                    li_new = t6 + t5  # (1, n1), (1, n1) -> (1, n1)
                    q3 = oi * t2.reshape(n1, 1)  # (n1, dn), (n1, 1) -> (n1, dn)

                    q1 = torch.matmul(
                        tilda_pij_b16.t().to(torch.float32), cur_cmp_v.to(torch.float32)
                    )  # (block_size|cur_valid_seq, n1), (block_size|cur_valid_seq, dn) -> (n1, dn)
                    q2 = q1 * t4.reshape(n1, 1)  # (n1, dn), (n1, 1) -> (n1, dn)
                    oi_tmp = q3 + q2
                    if block_idx == cur_cmp_block - 1:
                        oi_update = oi_tmp / li_new.reshape(
                            n1, 1
                        )  # (n1, dn), (n1, 1) -> (n1, dn)
                        cmp_attn_out[q_offset : (q_offset + n1), :] = oi_update
                    else:
                        oi_update = oi_tmp
                    mi_update = mi_new
                    li_update = li_new

                    sub_cur = tilda_mij - mi_update  # (1, n1), (1, n1) -> (1, n1)
                    exp_cur = sub_cur.exp()
                    tilda_pij = (
                        tilda_pij * exp_cur
                    )  # (block_size|cur_valid_seq, n1), (1, n1) -> (block_size|cur_valid_seq, n1)

                slc_cur = torch.zeros([block_slc_num, n1], dtype=torch.float32)
                # LOOP(1)

                for slc_idx in range(cur_slc_loop):
                    slc_valid = min(cur_valid_seq - slc_idx * slc_size, slc_window)
                    last_view = tilda_pij[
                        slc_idx * slc_size : (slc_idx * slc_size + slc_valid), :
                    ]  # (slc_valid, n1)
                    aux_tmp_tensor = aux_tensor[:slc_valid, :]  # (slc_valid, n1)
                    slc_last_no_reduce = last_view * aux_tmp_tensor  # (slc_valid, n1)
                    slc_last_reduce = slc_last_no_reduce.sum(
                        axis=0, keepdim=False
                    )  # (n1)
                    slc_cur[slc_idx, :] = slc_last_reduce

                # LOOP(1)
                if block_idx == cur_cmp_block - 1:  # 尾块处理
                    slc_before_g_reduce[
                        block_idx * block_slc_num : (block_idx + 1) * block_slc_num, :
                    ] = slc_cur
                if block_idx != 0:
                    local_max_gather[block_idx - 1 : block_idx, :] = mi_new
                    sub_pre = mi - mi_new
                    exp_pre = sub_pre.exp()
                    slc_pre = slc_pre * exp_pre
                    modify_tensor = tilda_pij[
                        : min(cmp_size - 1, cur_valid_seq), :
                    ]  # local
                    last_aux_tensor = aux_tensor[
                        (aux_tensor.shape[0] - min(cmp_size - 1, cur_valid_seq)) :, :
                    ]  # local
                    modify_tensor = modify_tensor * last_aux_tensor  # (cmp_size -1, n1)
                    modify_tensor_reduce = torch.sum(
                        modify_tensor, dim=0, keepdims=False
                    )  # [n1]
                    pre_view_tensor = slc_pre[block_slc_num - 1, :]  # [n1]
                    pre_view_tensor = pre_view_tensor + modify_tensor_reduce  # [n1]
                    slc_before_g_reduce[
                        (block_idx - 1) * block_slc_num : block_idx * block_slc_num - 1,
                        :,
                    ] = slc_pre[: block_slc_num - 1, :]
                    slc_before_g_reduce[block_idx * block_slc_num - 1, :] = (
                        pre_view_tensor
                    )

                slc_pre = slc_cur

            slc_before_g_reduce2 = torch.zeros(
                [max_cmp_block * block_slc_num, n1], dtype=torch.float32
            )
            for block_idx in range(cur_cmp_block - 1):
                slc_before_g_reduce_block = copy.deepcopy(
                    slc_before_g_reduce[
                        block_idx * block_slc_num : (block_idx + 1) * block_slc_num, :
                    ]
                )  # (block_slc_num, n1)
                row_max_block = local_max_gather[
                    block_idx : (block_idx + 1), :
                ]  # (1, n1)
                sub_tmp = row_max_block - mi_update
                exp_tmp = sub_tmp.exp()
                slc_before_g_reduce_block = (
                    slc_before_g_reduce_block * exp_tmp
                )  # (block_slc_num, n1), (1, n1) -> (block_slc_num, n1)
                slc_before_g_reduce_block = (
                    slc_before_g_reduce_block / li_update
                )  # (block_slc_num, n1), (1, n1) -> (block_slc_num, n1)
                slc_before_g_reduce2[
                    block_idx * block_slc_num : (block_idx + 1) * block_slc_num, :
                ] = slc_before_g_reduce_block

            slc_before_g_reduce_last_block = slc_before_g_reduce[
                (cur_cmp_block - 1) * block_slc_num : (cur_cmp_block * block_slc_num), :
            ]
            slc_before_g_reduce_last_block = (
                slc_before_g_reduce_last_block / li_update
            )  # (block_slc_num), n1)
            slc_before_g_reduce2[
                (cur_cmp_block - 1) * block_slc_num : (cur_cmp_block * block_slc_num), :
            ] = slc_before_g_reduce_last_block

            slc_before_g_reduce_actual = slc_before_g_reduce2[:slc_loop, :]
            slc_reduce = slc_before_g_reduce_actual.sum(axis=1, keepdim=True).reshape(
                1, 1, slc_loop
            )  # (slc_loop, n1) -> (1, 1, slc_loop)
            p_slc[b_idx : b_idx + 1, s_idx : s_idx + 1, :slc_loop] = slc_reduce

            if slc_loop < topk:
                topk_res[b_idx, s_idx, :slc_loop] = inc_seq[:slc_loop]
            else:
                slc_review = slc_reduce[:, :, front : (slc_loop - near)]
                _, inner_topk = torch.topk(slc_review, k=topk - front - near, dim=2)
                inner_topk = inner_topk + 1
                topk_res[b_idx, s_idx, :front] = inc_seq[:front]
                topk_res[
                    b_idx : (b_idx + 1), s_idx : (s_idx + 1), front : (topk - near)
                ] = inner_topk
                topk_res[b_idx, s_idx, (topk - near) :] = inc_seq[
                    slc_loop - near : slc_loop
                ]

    return cmp_attn_out, p_slc, topk_res


################
# Old cmp_attn #
################


def softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum


def cmp_attn_old(input_data_map: dict, params: dict):
    # get compute params
    block_size = params.get("block_size")
    cmp_block_size = params.get("cmp_block_size")
    cmp_stride = params.get("cmp_stride")
    slc_block_size = params.get("slc_block_size")
    softmax_scale = params.get("softmax_scale")
    topk = params.get("topk")
    front = params.get("front")
    near = params.get("near")
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    n2 = params.get("n2")
    dn = params.get("dn")
    dr = params.get("dr")
    dtype = params.get("dtype")

    # get input tensors
    q_nope = input_data_map.get("q_nope")
    q_rope = input_data_map.get("q_rope")
    k_bsnd = input_data_map.get("k_bsnd")
    cmp_kv_cache = input_data_map.get("cmp_kv_cache")
    cmp_kr_cache = input_data_map.get("cmp_kr_cache")
    cmp_block_table = input_data_map.get("cmp_block_table")
    act_seq_len = input_data_map.get("act_seq")
    aux_tensor = input_data_map.get("aux_tensor")

    q_bsnd = torch.cat((q_nope, q_rope), dim=-1).reshape(b, s1, n1, dn + dr)
    act_seq = act_seq_len

    aux_vec1 = aux_tensor[:, 0]
    kv_lora_rank = dn
    stride = cmp_stride
    max_cmp_block = cmp_block_table.shape[1]
    slc_size = slc_block_size // cmp_stride  # 4
    block_slc_num = block_size // slc_size  # 32

    rs = slc_block_size // stride  # 4
    rc = cmp_block_size // stride  # 2

    attn_res = torch.zeros(b, s1, n1, kv_lora_rank)
    max_act_seq = int(torch.max(act_seq).item())
    total_p_slc = torch.zeros(
        [b, s1, max_cmp_block * block_slc_num], dtype=torch.float32
    )
    for bi in range(b):
        for si in range(s1):
            act_cmp_seq_local = (
                act_seq[bi] - s1 + si + 1 - cmp_block_size
            ) // stride + 1
            q_act = q_bsnd[bi, si, :, :]  # [128, 576]
            k_act = k_bsnd[bi, :act_cmp_seq_local, 0, :]  # [act_cmp_seq_local, 576]
            v_act = k_act[:, :kv_lora_rank]  # [act_cmp_seq_local, 512]

            mm1 = torch.matmul(q_act.to(torch.float32), k_act.t().to(torch.float32))

            mm1_muls_scale = mm1 * softmax_scale
            softmax_res = softmax(mm1_muls_scale, dim=1)  # [128, act_cmp_seq_local]
            softmax_res_16 = softmax_res.to(dtype=dtype)
            loop_count = (act_cmp_seq_local - 1) // rs + 1
            p_slc_before_reduce_g = torch.zeros(n1, loop_count)  # [128, loop_count]
            for loop in range(loop_count):
                softmax_res_loop = softmax_res[
                    :, loop * rs : min((loop + 1) * rs + rc - 1, act_cmp_seq_local)
                ]
                aux_vec1_loop = aux_vec1[
                    : min(rs + rc - 1, act_cmp_seq_local - loop * rs)
                ]
                vmuls_res = softmax_res_loop * aux_vec1_loop  # [128, rs+1]
                reduce_s2_res = torch.sum(vmuls_res, dim=1, keepdim=True)  # [128, 1]
                p_slc_before_reduce_g[:, loop : loop + 1] = (
                    reduce_s2_res  # [128, loop_count]
                )
            p_slc = torch.sum(p_slc_before_reduce_g, dim=0)  # [loop_count]
            attn_out = torch.matmul(
                softmax_res_16.to(torch.float32), v_act.to(torch.float32)
            )
            attn_res[bi, si, :, :] = attn_out
            total_p_slc[bi, si, :loop_count] = p_slc

    return attn_res.reshape(b * s1 * n1, kv_lora_rank), total_p_slc


def compare_results(new, old, name):
    if new.shape != old.shape:
        print(f"{name} shape mismatch: {new.shape} vs {old.shape}")
        return False

    new_float = new.float()
    old_float = old.float()

    total_elements = new.numel()
    abs_diff = (new_float - old_float).abs()

    atol = 1 * 1e-3
    rtol = 1 * 1e-3
    close_mask = abs_diff <= (atol + rtol * old_float.abs())
    equal_mask = abs_diff == 0
    non_zero_mask = (new_float != 0) | (old_float != 0)

    close_count = close_mask.sum().item()
    equal_count = equal_mask.sum().item()
    non_zero_count = non_zero_mask.sum().item()

    mismatch_mask = ~close_mask
    mismatch_indices = mismatch_mask.nonzero(as_tuple=True)
    mismatch_count = len(mismatch_indices[0])

    print(f"\n{name} comparison report:")
    print(f"  Total elements: {total_elements}")
    print(f"  Non-zero elements: {non_zero_count}")
    print(
        f"  Close elements (atol={atol}, rtol={rtol}): {close_count} ({close_count/total_elements:.2%})"
    )
    print(f"  Exactly equal elements: {equal_count} ({equal_count/total_elements:.2%})")
    print(
        f"  Mismatched elements: {mismatch_count} ({mismatch_count/total_elements:.2%})"
    )

    max_display = 20
    if mismatch_count > 0:
        print("\nMismatched elements [index | new value | old value | absolute diff]:")
        for i in range(min(mismatch_count, max_display)):
            idx = tuple(mi[i].item() for mi in mismatch_indices)
            new_val = new_float[idx].item()
            old_val = old_float[idx].item()
            diff = abs_diff[idx].item()
            print(f"  {idx}: {new_val:.8f} vs {old_val:.8f} | diff={diff:.8f}")

        if mismatch_count > max_display:
            print(f"  ... and {mismatch_count - max_display} more mismatches")

    return mismatch_count < total_elements * 0.001


def check_two_ver_goldens(input_data_map: dict, params: dict):
    new_attn_out, new_p_slc, new_topk_res = compress_attention_with_topk_compute(
        input_data_map, params
    )
    old_attn_out, old_p_slc = cmp_attn_old(input_data_map, params)

    if new_attn_out.shape != old_attn_out.shape:
        print(
            f"Shapes mismatch: new_attn={new_attn_out.shape}, old_attn={old_attn_out.shape}"
        )
        return

    max_diff = (new_attn_out - old_attn_out).max().item()
    min_diff = (new_attn_out - old_attn_out).min().item()
    max_relative_diff = (new_attn_out / old_attn_out).max().item() - 1
    min_relative_diff = (new_attn_out / old_attn_out).min().item() - 1
    print(f"new_golden: [{new_attn_out.min()}, {new_attn_out.max()}]")
    print(f"old_golden: [{old_attn_out.min()}, {old_attn_out.max()}]")
    print(
        f"diff=[{min_diff}, {max_diff}], relative_diff=[{min_relative_diff}, {max_relative_diff}]"
    )

    attn_out_result = compare_results(new_attn_out, old_attn_out, "AttnOut")
    print(f"attn out compare={attn_out_result}")

    max_diff = (new_p_slc - old_p_slc).max().item()
    min_diff = (new_p_slc - old_p_slc).min().item()
    max_relative_diff = (new_p_slc / old_p_slc).max().item() - 1
    min_relative_diff = (new_p_slc / old_p_slc).min().item() - 1
    print(f"new_golden: [{new_p_slc.min()}, {new_p_slc.max()}]")
    print(f"old_golden: [{old_p_slc.min()}, {old_p_slc.max()}]")
    print(
        f"diff=[{min_diff}, {max_diff}], relative_diff=[{min_relative_diff}, {max_relative_diff}]"
    )

    p_slc_result = compare_results(new_p_slc, old_p_slc, "AttnOut")
    print(f"p_slc compare={p_slc_result}")


#####################
# Old cmp_attn ends #
#####################


@GoldenRegister.reg_golden_func(
    case_names=[
        "CmpAttnTopk.cmp_attn_with_topk_singleop_bf16",
    ]
)
def compress_attention_with_topk(case_name: str, output: Path) -> bool:
    b, s1, n1, dn, dr = 48, 1, 128, 512, 64
    s2, n2 = 128 * 1024, 1
    dtype = torch.bfloat16
    if b % 2 == 0:
        act_seq = [s2] * (b // 2) + [s2 // 2 + 1] * (b // 2)
        logging.info(act_seq)
    else:
        act_seq = [s2] * (b // 2) + [s2 // 2 + 1] * (b // 2) + [s2 // 2 + 1]
        logging.info(act_seq)
    block_size = 128
    cmp_block_size = 32
    cmp_stride = 16
    slc_block_size = 64
    topk = 16
    front = 1
    near = 2

    cur_case = "CmpAttnTopk.cmp_attn_with_topk_singleop_bf16"
    if case_name != cur_case:
        logging.error("Fail to gen golden for Case(%s)", case_name)
        return False

    params = {}
    params["b"] = b
    params["s1"] = s1
    params["n1"] = n1
    params["dn"] = dn
    params["dr"] = dr
    params["dtype"] = dtype
    params["s2"] = s2
    params["n2"] = n2
    params["act_seq"] = act_seq
    params["block_size"] = block_size
    params["cmp_block_size"] = cmp_block_size
    params["cmp_stride"] = cmp_stride
    params["softmax_scale"] = float(1.0) / np.sqrt(dn + dr)
    params["slc_block_size"] = slc_block_size
    params["topk"] = topk
    params["front"] = front
    params["near"] = near
    act_cmp_seq = [(cur_s - cmp_block_size) // cmp_stride + 1 for cur_s in act_seq]
    params["act_cmp_seq"] = act_cmp_seq
    s_cmp_max = max(act_cmp_seq)
    params["s_cmp_max"] = s_cmp_max

    input_data_map = gen_data_for_compute(output, params)
    cmp_attn_out, slc_result, topk_res = compress_attention_with_topk_compute(
        input_data_map, params
    )

    check_two_ver_goldens(input_data_map, params)

    # dump golden for compare res
    cmp_attn_out_path = Path(output, "cmp_attn_out.bin")
    slc_result_path = Path(output, "slc_result.bin")
    topk_res_path = Path(output, "topk_res.bin")
    dump_file(cmp_attn_out.numpy(), cmp_attn_out_path, "fp32")
    dump_file(slc_result.numpy(), slc_result_path, "fp32")
    dump_file(topk_res.numpy(), topk_res_path, "int32")

    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "CmpAttnTopk.cmp_attn_with_topk_singleop_bf16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = compress_attention_with_topk(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
