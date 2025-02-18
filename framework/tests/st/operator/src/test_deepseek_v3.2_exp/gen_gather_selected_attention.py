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

import numpy as np
import torch
from ml_dtypes import bfloat16

from gen_mla_prolog_quant_golden_v32 import gen_block_table

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


def dump_file(data, data_path, dtype):
    """将PyTorch张量保存到文件，支持BFloat16类型转换"""
    if dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.int32:
        np_dtype = np.int32
    elif dtype == torch.bfloat16:
        np_dtype = bfloat16
    elif dtype == torch.int8:
        np_dtype = np.int8
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    if isinstance(data, torch.Tensor):
        # 处理BFloat16类型：转换为float32后再转NumPy（NumPy不支持BFloat16）
        if data.dtype == torch.bfloat16:
            data_np = data.cpu().to(torch.float32).numpy()
        else:
            data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)
    # 确保最终类型与指定dtype一致
    data_np = data_np.astype(np_dtype)
    data_np.tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def softmax(x, input_dtype):
    """PyTorch实现的softmax函数"""
    x = x.float()
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    y = y.to(input_dtype)
    x_sum = torch.sum(y, dim=-1, keepdim=True)
    ans = y
    return ans, x_sum, x_max


def compute_attention(input_data, params):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    """
    q, kn, kr, kn_scales, topk_indcies, block_table, actual_seq = input_data
    block_size, scalar, topk, d_v, is_kn_quant = params
    # 提取维度信息
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape

    s2_tile = 2048
    if topk_indcies.ndim > 2:
        topk_indcies = topk_indcies.reshape(b * s1, topk)

    atten_out_shape = [b, s1, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, n1], dtype=input_dtype)

    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        for s1_idx in range(s1):
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            bn_per_batch = math.ceil(cur_seq / s2_tile)

            qi = q[b_idx, s1_idx, :, :] # (n1, dk)

            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur
                cur_bs1_idx = b_idx * s1 + s1_idx
                topk_indcies_tmp = topk_indcies[cur_bs1_idx, s2_start:s2_end]
                slc_kn = torch.zeros([s2_tile_cur, dk], dtype=kn_dtype)
                slc_kr = torch.zeros([s2_tile_cur, dv], dtype=input_dtype)
                slc_kn_scales = torch.zeros([s2_tile_cur, 4], dtype=torch.float32)

                # 当前b&s1&s2 topk_index  --->  kvCache的offset
                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    s2_idx_tmp = s2_start + cur_s2_idx
                    topk_index = topk_indcies_tmp[s2_idx_tmp]
                    block_idx_in_batch = topk_index // block_size
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]
                    tail = topk_index % block_size
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail

                # 索引 kvCache
                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_kn[cur_s2_idx, :] = kn[slc_idx, :]
                    slc_kr[cur_s2_idx, :] = kr[slc_idx, :]
                    slc_kn_scales[cur_s2_idx, :] = kn_scales[slc_idx, :]
                
                qn_tmp = qi[..., :dk]
                qr_tmp = qi[..., dk:]
                if is_kn_quant:
                    kn_bs = slc_kn.reshape(-1, 128).to(torch.float)
                    kn_scales_tmp = slc_kn_scales.reshape(-1, 1)
                    kn_tmp = kn_bs * kn_scales_tmp
                    kn_tmp = kn_tmp.reshape(-1, 512).to(input_dtype)
                else:
                    kn_tmp = slc_kn
                kr_tmp = slc_kr
                vj = kn_tmp

                # C1
                qkn_bmm = torch.matmul(qn_tmp, kn_tmp.transpose(1, 0)).to(torch.float)
                qkr_bmm = torch.matmul(qr_tmp, kr_tmp.transpose(1, 0)).to(torch.float)
                sij = qkn_bmm + qkr_bmm
                sij_scale = sij * scalar # (n1, s2_tile)
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True) # (n1, 1)
                t_sub = sij_scale - tilda_mij # (n1, s2_tile)
                tilda_pij = torch.exp(t_sub) # (n1, s2_tile)
                tilda_pij_f16 = tilda_pij.to(input_dtype)
                q1 = torch.matmul(tilda_pij_f16, vj)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True) # (n1, 1)

                if s2_idx == 0:
                    oi_tmp = q1
                    if bn_per_batch == 1:
                        oi_update = oi_tmp / tilda_lij
                    else:
                        oi_update = oi_tmp
                    li_update = tilda_lij
                    mi_update = tilda_mij
                    tmp_out[b_idx, s1_idx, :] = tilda_lij.reshape(n1)
                    continue

                oi = oi_update
                li = li_update
                mi = mi_update

                mi_new = torch.maximum(mi, tilda_mij)
                t1 = mi - mi_new
                t2 = torch.exp(t1)
                t3 = tilda_mij - mi_new
                t4 = torch.exp(t3)
                t5 = t4 * tilda_lij
                t6 = t2 * li
                li_new = t6 + t5
                q3 = oi * t2
                q2 = q1 * t4
                oi_tmp = q3 + q2
                if s2_idx == bn_per_batch - 1:
                    oi_update = oi_tmp / li_new
                else:
                    oi_update = oi_tmp
                li_update = li_new
                mi_update = mi_new

            attention_output[b_idx, s1_idx, :, :] = oi_update

    return attention_output, tmp_out


def gen_dsa_gather_sa_entry(dtype, bn1n2s1, is_kn_quant, actual_seq, output):
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # 48, 128, 1, 1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)
    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank
    scalar = d_q ** -0.5
    if isinstance(actual_seq, int):
        actual_seq = [actual_seq] * b
    elif isinstance(actual_seq, list):
        if len(actual_seq) == b:
            actual_seq = actual_seq
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")
    # 1. 定义shape
    shape_q = [b, s_q, n_q, d_q]

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq_tmp in actual_seq:
        block_num_per_batch.append(math.ceil(actual_seq_tmp / block_size))
        block_num_min += math.ceil(actual_seq_tmp / block_size)
    block_num = block_num_min

    shape_kn = [block_num, block_size, kv_lora_rank]
    shape_kr = [block_num, block_size, qk_rope_dim]

    max_kv_seq = max(actual_seq)
    block_num, block_table = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    topk_indcies = torch.zeros(b, s_q, topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))
    for b_i in range(b):
        for s_q_i in range(s_q):
            if slc_actual_seq[b_i] < topk:
                topk_indcies[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indcies[b_i, s_q_i, :] = perm[:topk]
    topk_indcies = topk_indcies.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kn_bsnd_tmp = gen_uniform_data(shape_kn, -1, 1, dtype)
    
    kn_bsnd_reshape = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    if is_kn_quant == 1:
        kn_quant = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128) / kn_scales
        kn = torch.round(kn_quant).clamp(-128, 127).to(torch.int8)
    else:
        kn = kn_bsnd_tmp
    kr = gen_uniform_data(shape_kr, -1, 1, dtype)
    # 2D
    kn = kn.reshape(block_num * block_size, kv_lora_rank)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim)

    # 3. 计算attention
    params = [block_size, scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, topk_indcies, block_table, actual_seq]
    atten_out, tmp_out = compute_attention(input_data, params)

    # 4.dump 数据
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)
    # input params
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, is_kn_quant]
    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    kn_path = Path(output, 'k_nope.bin')
    kr_path = Path(output, 'k_rope.bin')
    kn_scales_path = Path(output, 'kn_scales.bin')
    topk_indcies_path = Path(output, 'topk_indcies.bin')
    block_table_path = Path(output, 'block_table.bin')
    actual_seq_path = Path(output, 'actual_seq.bin')
    atten_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')
    # dump golden file
    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(kn, kn_path, kn.dtype)
    dump_file(kr, kr_path, dtype)
    dump_file(kn_scales, kn_scales_path, kn_scales.dtype)
    dump_file(topk_indcies, topk_indcies_path, torch.int32)
    dump_file(block_table, block_table_path, torch.int32)
    dump_file(actual_seq, actual_seq_path, torch.int32)
    dump_file(atten_out, atten_out_path, dtype)
    dump_file(input_params, input_param_path, torch.int32)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicGatherSlcFlashAttnDSASTest.SFA_b4_s2_seq64K_int8_perf",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8",
    ],
    version=0,
    timeout=0
)
def dsa_sa_func(case_name: str, output: Path) -> bool:
    if case_name == "DynamicGatherSlcFlashAttnDSASTest.SFA_b4_s2_seq64K_int8_perf":
        bn1n2s1 = (4, 128, 1, 2)
        is_kn_quant = 1
        actual_seq = [65536] * 4
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 2)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511":
        # bn1n2s1数据: b, n_q, n_kv, s_q; n_kv=1
        bn1n2s1 = (32, 128, 1, 1)
        # 0为kn非量化情况，1为kn量化情况
        is_kn_quant = 0
        actual_seq = 511
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8":
        bn1n2s1 = (32, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 511
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = 2049
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 2049
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 0
        actual_seq = 2047
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 1
        actual_seq = 2047
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = 8096
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 8096
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = 131072  # 128k
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 131072
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
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
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = dsa_sa_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
