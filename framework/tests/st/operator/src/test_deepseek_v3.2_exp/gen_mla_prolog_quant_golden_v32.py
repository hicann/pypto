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

""" mla_prolog_quant_v32 子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import math
import time
import logging
from pathlib import Path
from typing import List

import torch

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../cmake").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def rms_norm(x, gamma):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True)
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def scatter_update(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [block_number,block_size,n2,d], n2=1
    # key_states shape: [b*s1*1, d]
    # indices shape: [b, s1], s1=1
    cache, key_states, indices = inputs
    block_number, block_size, n2, d = cache.shape
    res = cache.reshape(block_number * block_size * n2, d)
    b, s1 = indices.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]
    return res.reshape(block_number, block_size, n2, d)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=2):
    input_dtype = q.dtype
    if input_dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=unsqueeze_dim)  # [b,s,1,qk_d]
    sin = torch.unsqueeze(sin, dim=unsqueeze_dim)  # [b,s,1,qk_d]
    logging.debug("expand sin.shape: %s", sin.shape)
    logging.debug("expand cos.shape: %s", cos.shape)

    b, s, h, d = q.shape
    q = q.reshape(b, s, h, d // 2, 2).permute(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,n,qk_d]

    b, s, h, d = k.shape
    k = k.reshape(b, s, h, d // 2, 2).permute(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,1,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != torch.float32:
        q_embed, k_embed = q_embed.to(input_dtype), k_embed.to(input_dtype)
    return q_embed, k_embed


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_cq=None):
    input_fp32 = input_t.to(torch.float32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_cq
    abs_res = torch.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = torch.max(abs_res, dim=reduce_idx, keepdims=True)[0]
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_fp16 = out_int32.to(torch.float16)
    out_int8 = torch.trunc(out_fp16).to(torch.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def tensor_to_file(t: torch.Tensor, output: Path):
    with open(str(output), "wb") as f:
        dtype = t.dtype
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).numpy().tobytes())


def mla_prolog_quant_v32_compute(inputs):
    dtype = inputs.get("dtype")
    is_quant_a = inputs.get("is_quant_a")
    is_quant_b = inputs.get("is_quant_b")
    has_smooth = inputs.get("has_smooth")
    cache_mode = inputs.get("cache_mode")
    gamma_cq = inputs.get("gamma_cq")
    gamma_ckv = inputs.get("gamma_ckv")
    x = inputs.get("x")
    w_dq = inputs.get("w_dq")
    w_uqqr = inputs.get("w_uqqr")
    w_uk = inputs.get("w_uk")
    w_dkvkr = inputs.get("w_dkvkr")
    cos = inputs.get("cos")
    sin = inputs.get("sin")
    kv_cache = inputs.get("kv_cache")
    kr_cache = inputs.get("kr_cache")
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_quant_scale_cache = inputs.get("kv_quant_scale_cache")
    cache_index = inputs.get("cache_index")
    if is_quant_a:
        w_qa_scale = inputs.get("w_qa_scale")
        w_kva_scale = inputs.get("w_kva_scale")
    if is_quant_b:
        w_qb_scale = inputs.get("w_qb_scale")
        if has_smooth:
            smooth_cq = inputs.get("smooth_cq")

    b, s, h = x.shape
    qk_rope_head_dim = cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    """ q """
    x_2d = x.reshape(b * s, h)
    # shape is: [b * s, h] @ [h, q_lora_rank] -> [b * s, q_lora_rank]
    if is_quant_a:
        # no smooth
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        q_a_proj = torch.matmul(x_2d_quant.to(torch.int32), w_dq.to(torch.int32))

        """ dequant """
        q_a_proj_fp32 = q_a_proj.to(torch.float32)
        q_a_proj_fp32_dequant = q_a_proj_fp32 * x_2d_scale_dequant
        q_a_proj = q_a_proj_fp32_dequant * w_qa_scale
    else:
        q_a_proj = torch.matmul(x_2d.to(torch.float32), w_dq.to(torch.float32))  # [b * s, q_lora_rank]

    q_a_proj = q_a_proj.to(dtype)

    q_a_layernorm = rms_norm(q_a_proj, gamma_cq)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    # shape is: [b * s, q_lora_rank] @ [q_lora_rank, n * q_head_dim] -> [b * s, n * q_head_dim]
    q_a_layernorm_scale_dequant = None
    if is_quant_b:
        if has_smooth:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True, True, smooth_cq)
        else:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True)  # scale: [b*s,1]
        q_b_proj = torch.matmul(q_a_layernorm.to(torch.int32), w_uqqr.to(torch.int32)).to(q_a_layernorm.device)  # q_b_proj

        """ dequant """
        q_b_proj_fp32 = q_b_proj.to(torch.float32)
        q_b_proj_fp32_dequant = q_b_proj_fp32 * q_a_layernorm_scale_dequant
        q_b_proj = q_b_proj_fp32_dequant * w_qb_scale
    else:
        q_b_proj = torch.matmul(q_a_layernorm.to(torch.float32), w_uqqr.to(torch.float32))  # [b * s, n * q_head_dim]

    q_b_proj = q_b_proj.to(dtype)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)

    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    q_nope = q_reshape[:, :, :, 0:qk_nope_head_dim]  # [b, s, n, qk_nope_head_dim]
    q_nope_r = q_nope.reshape(b * s, n, qk_nope_head_dim)
    q_nope_t = q_nope_r.permute(1, 0, 2)  # [n, b*s, qk_nope_head_dim]
    # shape is: [n, b*s, qk_nope_head_dim] @ [n, qk_nope_head_dim, kv_lora_rank] -> [n, b*s, kv_lora_rank]
    q_nope_new = torch.matmul(q_nope_t.to(torch.float32), w_uk.to(torch.float32))
    q_nope_new = q_nope_new.to(dtype)
    q_nope_new_t = q_nope_new.permute(1, 0, 2)  # [b*s, n, kv_lora_rank]
    q_out = q_nope_new_t.reshape(b, s, n, kv_lora_rank)  # [b, s, n, kv_lora_rank]

    """ kv """
    # shape is: [b*s, h] @ [h, kv_lora_rank + qk_rope_head_dim] -> [b*s, kv_lora_rank + qk_rope_head_dim]
    if is_quant_a:
        # no smooth
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        kv_a_proj = torch.matmul(x_2d_quant.to(torch.int32), w_dkvkr.to(torch.int32))
        """ dequant """
        kv_a_proj_fp32 = kv_a_proj.to(torch.float32)
        kv_a_proj_fp32_dequant = kv_a_proj_fp32 * x_2d_scale_dequant
        kv_a_proj = kv_a_proj_fp32_dequant * w_kva_scale
    else:
        kv_a_proj = torch.matmul(x_2d.to(torch.float32),
                                 w_dkvkr.to(torch.float32))  # [b*s, kv_lora_rank + qk_rope_head_dim]

    kv_a_proj = kv_a_proj.to(dtype)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    compressed_kv = kv_reshape[:, :, 0:kv_lora_rank]  # [b, s, kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, gamma_ckv)
    compressed_kv_quant_scale = None
    if is_quant_b:
        compressed_kv_norm_split = compressed_kv_norm.reshape(b * s, 4, kv_lora_rank // 4)
        compressed_kv_norm, compressed_kv_quant_scale = quant(compressed_kv_norm_split, True)
        compressed_kv_quant_scale = compressed_kv_quant_scale.reshape(b, s, 1, 4)
    compressed_kv_r = compressed_kv_norm.reshape(b, s, 1, kv_lora_rank)
    k_nope = compressed_kv_r.reshape(b * s * 1, kv_lora_rank)

    """ RoPE """
    q_pe = q_reshape[:, :, :, qk_nope_head_dim:]  # [b, s, n, qk_rope_head_dim]

    k_pe = kv_reshape[:, :, kv_lora_rank:]  # [b, s, qk_rope_head_dim]
    k_pe_r = k_pe.reshape(b, s, 1, qk_rope_head_dim)

    # q_embed: [b, s, n, qk_rope_head_dim], k_embed: [b, s, 1, qk_rope_head_dim]
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 2)
    k_embed_r = k_embed.reshape(b * 1 * s, qk_rope_head_dim)

    """ kv_cache output, [b,1,s2,kv_lora_rank] """
    kv_cache_out = scatter_update([kv_cache, k_nope, cache_index], -2)

    """ kr_cache output, [b,1,s2,qk_rope_head_dim] """
    kr_cache_out = scatter_update([kr_cache, k_embed_r, cache_index], -2)

    if is_quant_b:
        compressed_kv_quant_scale = compressed_kv_quant_scale.reshape(-1, 4)
        kv_quant_scale_cache_out = scatter_update([kv_quant_scale_cache, compressed_kv_quant_scale, cache_index], -2)
    else:
        kv_quant_scale_cache_out = None

    return q_out, q_embed, q_a_layernorm, q_a_layernorm_scale_dequant, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    b = act_seq.shape[0]
    block_num = 0
    block_num_each = []
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_idx = 0
    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int64)
        for i in range(b):
            cur_act = act_seq[i]
            for j in range(s1):
                pos = cur_act - s1 + j
                block_idx_in_seq = pos // block_size
                global_block_id = block_table[i, block_idx_in_seq]

                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    else:
        cache_index = None

    if need_indices:
        return block_num, block_table, cache_index
    else:
        return block_num, block_table


def gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, output_dir: Path, is_quant=(False, False),
                                        is_nz=False, has_smooth=False, block_size=128, cache_mode="BSND"):
    dtype, w_dtype = dtypes
    logging.debug(f"gen_mla_prolog_quant_v32_input_data  dtype:{dtype}, w_dtype:{w_dtype}")
    is_quant_a, is_quant_b = is_quant
    b = params.get("b")
    s = params.get("s")  # s=1 or 2
    s1 = params.get("s1")  # s2=4k
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    block_num, block_table, cache_index = gen_block_table(actual_seq, block_size, s1, need_indices=True)

    skv_max = actual_seq.max()
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    cos_shape = [b, s, qk_rope_head_dim]
    kv_bsnd_shape = [b, skv_max, 1, kv_lora_rank + qk_rope_head_dim]
    kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
    kr_cache_shape = [block_num, block_size, 1, qk_rope_head_dim]
    kv_quant_scale_cache_shape = [block_num, block_size, 1, 4]
    smooth_cq_shape = [1, q_lora_rank]
    logging.debug("x shape is %s", x_shape)
    logging.debug("w_dq shape is %s", w_qa_shape)
    logging.debug("w_uqqr shape is %s", w_qb_shape)
    logging.debug("w_dkvkr shape is %s", w_kv_a_shape)
    logging.debug("w_uk shape is %s", w_kv_b_k_shape)
    logging.debug("cos sin shape is %s", cos_shape)
    logging.debug("cgamma_cq shape is %s", gamma_cq_shape)
    logging.debug("cgamma_ckv shape is %s", gamma_ckv_shape)
    logging.debug("kv_len shape is %s", cache_index.shape)
    logging.debug("kv_cache shape is %s", kv_cache_shape)
    logging.debug("kr_cache shape is %s", kr_cache_shape)
    logging.debug("block_num is %s", block_num)
    logging.debug("block_table shape is %s", block_table.shape)
    logging.debug("actual_seq is %s", actual_seq)
    if is_quant_b:
        logging.debug("kv_quant_scale_cache shape is %s", kv_quant_scale_cache_shape)

    x_path = Path(output_dir, 'x.bin')
    w_dq_path = Path(output_dir, 'wDq.bin')
    w_qa_scale_path = Path(output_dir, 'w_qa_scale.bin')
    w_uqqr_path = Path(output_dir, 'wUqQr.bin')
    w_qb_scale_path = Path(output_dir, 'w_qb_scale.bin')
    w_dkvkr_path = Path(output_dir, 'wDkvKr.bin')
    w_kva_scale_path = Path(output_dir, 'w_kva_scale.bin')
    w_uk_path = Path(output_dir, 'wUk.bin')  # kv_b_proj_w_k
    gamma_cq_path = Path(output_dir, 'gamma_cq.bin')
    gamma_ckv_path = Path(output_dir, 'gamma_ckv.bin')
    cos_path = Path(output_dir, 'cos.bin')
    sin_path = Path(output_dir, 'sin.bin')
    kv_len_path = Path(output_dir, 'kv_len.bin')
    kv_cache_path = Path(output_dir, 'kv_cache.bin')
    kr_cache_path = Path(output_dir, 'kr_cache.bin')
    kv_quant_scale_cache_path = Path(output_dir, 'kv_quant_scale_cache.bin')
    smooth_cq_path = Path(output_dir, 'smooth_cq.bin')

    res = [None] * 17
    x = torch.empty(x_shape).uniform_(-1, 1).to(dtype)
    tensor_to_file(x, x_path)
    res[0] = x
    w_dq = torch.empty(w_qa_shape).uniform_(-0.1, 0.1).to(w_dtype)
    w_uqqr = torch.empty(w_qb_shape).uniform_(-0.1, 0.1).to(w_dtype)
    w_dkvkr = torch.empty(w_kv_a_shape).uniform_(-0.1, 0.1).to(w_dtype)
    res[4] = dict()

    if is_quant_a:
        w_dq, w_qa_scale = quant(w_dq, False)
        w_dkvkr, w_kva_scale = quant(w_dkvkr, False)
        tensor_to_file(w_qa_scale, w_qa_scale_path)
        tensor_to_file(w_kva_scale, w_kva_scale_path)
        res[4]["w_dq"] = w_qa_scale
        res[4]["w_dkvkr"] = w_kva_scale
        if is_nz:
            tensor_to_file(w_dq.reshape(h, q_lora_rank // 32, 32).permute(1, 0, 2), w_dq_path)
            tensor_to_file(w_dkvkr.reshape(h, (kv_lora_rank + qk_rope_head_dim) // 32, 32).permute(1, 0, 2),
                           w_dkvkr_path)
        else:
            tensor_to_file(w_dq, w_dq_path)
            tensor_to_file(w_dkvkr, w_dkvkr_path)
    else:
        if is_nz:
            tensor_to_file(w_dq.reshape(h, q_lora_rank // 16, 16).permute(1, 0, 2), w_dq_path)
            tensor_to_file(w_dkvkr.reshape(h, (kv_lora_rank + qk_rope_head_dim) // 16, 16).permute(1, 0, 2),
                           w_dkvkr_path)
        else:
            tensor_to_file(w_dq, w_dq_path)
            tensor_to_file(w_dkvkr, w_dkvkr_path)

    if is_quant_b:
        w_uqqr, w_qb_scale = quant(w_uqqr, False)
        tensor_to_file(w_qb_scale, w_qb_scale_path)
        res[4]["w_uqqr"] = w_qb_scale
        # smooth_data
        if has_smooth:
            smooth_cq = torch.empty(smooth_cq_shape).uniform_(-1, 1).to(torch.float32)
            tensor_to_file(smooth_cq, smooth_cq_path)
            res[3] = smooth_cq
        if is_nz:
            tensor_to_file(w_uqqr.reshape(q_lora_rank, n * q_head_dim // 32, 32).permute(1, 0, 2), w_uqqr_path)
        else:
            tensor_to_file(w_uqqr, w_uqqr_path)
    else:
        if is_nz:
            tensor_to_file(w_uqqr.reshape(q_lora_rank, n * q_head_dim // 16, 16).permute(1, 0, 2), w_uqqr_path)
        else:
            tensor_to_file(w_uqqr, w_uqqr_path)

    res[1] = w_dq
    res[2] = w_uqqr
    res[5] = w_dkvkr

    w_uk = torch.empty(w_kv_b_k_shape).uniform_(-0.1, 0.1).to(w_dtype)
    tensor_to_file(w_uk, w_uk_path)
    res[6] = w_uk
    gamma_cq = torch.empty(gamma_cq_shape).uniform_(-1, 1).to(dtype)  # [q_lora_rank]
    gamma_ckv = torch.empty(gamma_ckv_shape).uniform_(-1, 1).to(dtype)  # [kv_lora_rank]
    tensor_to_file(gamma_cq, gamma_cq_path)
    tensor_to_file(gamma_ckv, gamma_ckv_path)
    res[7] = gamma_cq
    res[8] = gamma_ckv
    cos = torch.empty(cos_shape).uniform_(-0.1, 0.1).to(dtype)  # [b, s, qk_rope_head_dim]
    sin = torch.empty(cos_shape).uniform_(-0.1, 0.1).to(dtype)  # [b, s, qk_rope_head_dim]
    tensor_to_file(cos, cos_path)
    tensor_to_file(sin, sin_path)
    res[9] = cos
    res[10] = sin
    tensor_to_file(cache_index, kv_len_path)
    res[11] = cache_index
    k_bsnd = torch.empty(kv_bsnd_shape).uniform_(-1, 1).to(dtype)
    # kv paddIng
    per_batch_max_num = math.ceil(skv_max / block_size)
    k_tensor_bsnd = torch.zeros((b, per_batch_max_num * block_size, 1, kv_lora_rank + qk_rope_head_dim)).to(dtype)
    k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
    # kv_cache
    k_cache_tensor = torch.zeros([block_num, block_size, 1, kv_lora_rank + qk_rope_head_dim]).to(dtype)
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache_tensor[kv_cache_blk_id, 0:block_size, :, :] = k_tensor_bsnd[
                    b_idx, block_offset:(block_offset + block_size), :, :]
    kv_cache = k_cache_tensor[:, :, :, : kv_lora_rank]
    kr_cache = k_cache_tensor[:, :, :, kv_lora_rank:]
    kv_quant_scale_cache = None
    if is_quant_b:
        kv_cache_split = kv_cache.reshape(-1, 4, kv_lora_rank // 4)
        kv_cache, kv_quant_scale_cache = quant(kv_cache_split, True)
        kv_cache = kv_cache.reshape(kv_cache_shape)
        kv_quant_scale_cache = kv_quant_scale_cache.reshape(kv_quant_scale_cache_shape)
        tensor_to_file(kv_quant_scale_cache, kv_quant_scale_cache_path)
    tensor_to_file(kr_cache, kr_cache_path)  # kr_cache in
    tensor_to_file(kv_cache, kv_cache_path)  # kv_cache in
    res[12] = kv_cache
    res[13] = kr_cache
    res[14] = kv_quant_scale_cache
    res[15] = block_num
    res[16] = block_table

    return res


def gen_mla_prolog_quant_v32_data(params, dtypes, actual_seq, output_dir: Path, is_quant=(False, False), is_nz=False,
                                  has_smooth=False, block_size=128, cache_mode="BSND"):
    torch.manual_seed(int(time.time()))
    dtype, w_dtype = dtypes
    logging.debug(f"gen_mla_prolog_quant_v32_data  dtype:{dtype}, w_dtype:{w_dtype}")
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache, kv_quant_scale_cache, block_num, block_table = \
        gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, output_dir, is_quant, is_nz, has_smooth,
                                            block_size, cache_mode)
    is_quant_a, is_quant_b = is_quant

    # output
    q_golden_path = Path(output_dir, 'q_golden.bin')
    q_rope_golden_path = Path(output_dir, 'q_rope_golden.bin')
    rms_norm_golden_path = Path(output_dir, 'rms_norm_golden.bin')
    rms_norm_scale_golden_path = Path(output_dir, 'rms_norm_scale_golden.bin')
    kv_golden_path = Path(output_dir, 'kv_cache_golden.bin')
    kr_golden_path = Path(output_dir, 'kr_cache_golden.bin')
    kv_quant_scale_cache_golden_path = Path(output_dir, 'kv_quant_scale_cache_golden.bin')

    inputs = {"dtype": dtype, "is_quant_a": is_quant_a, "is_quant_b": is_quant_b, "has_smooth": has_smooth}
    inputs["cache_mode"] = cache_mode
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv
    inputs["x"] = x
    inputs["w_dq"] = w_dq
    inputs["w_uqqr"] = w_uqqr
    inputs["w_uk"] = w_uk
    inputs["w_dkvkr"] = w_dkvkr
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["kv_cache"] = kv_cache
    inputs["kr_cache"] = kr_cache
    inputs["kv_quant_scale_cache"] = kv_quant_scale_cache
    inputs["cache_index"] = kv_len
    if is_quant_a:
        inputs["w_qa_scale"] = scale_data["w_dq"]
        inputs["w_kva_scale"] = scale_data["w_dkvkr"]
    if is_quant_b:
        inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            inputs["smooth_cq"] = smooth_cq

    q_out, q_embed, rms_norm, rms_norm_scale, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out = mla_prolog_quant_v32_compute(
        inputs)

    tensor_to_file(q_out, q_golden_path)  # [b,s,n,kv_lora_rank]
    tensor_to_file(q_embed, q_rope_golden_path)  # [b,s,n,qk_rope_head_dim]
    tensor_to_file(kr_cache_out, kr_golden_path)
    tensor_to_file(kv_cache_out, kv_golden_path)
    if kv_quant_scale_cache_out is not None:
        tensor_to_file(kv_quant_scale_cache_out, kv_quant_scale_cache_golden_path)
    tensor_to_file(rms_norm, rms_norm_golden_path)
    if rms_norm_scale is not None:
        tensor_to_file(rms_norm_scale, rms_norm_scale_golden_path)
    return q_out, q_embed, kv_cache_out, kr_cache_out


def gen_mla_prolog_v32_quantB_test(dtypes, bn1s1s2, output_dir: Path, is_quant=False, is_nz=False,
                                   is_smooth=False, block_size=128, cache_mode="BSND"):
    b, n, s1, s2 = bn1s1s2
    quant_choice = (False, is_quant)
    params = {
        "b": b,
        "s": s1,
        "s1": s1,
        "s2": s2,
        "h": 7168,
        "num_heads": n,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    actual_seq = torch.tensor([s2] * b, dtype=torch.int32).unsqueeze(-1)
    gen_mla_prolog_quant_v32_data(params, dtypes, actual_seq, output_dir, quant_choice, is_nz, is_smooth, block_size,
                                  cache_mode)


def gen_mla_prolog_quant_v32_data_wrap(case_name: str, output: Path):
    # fp16, quant, weight nd, "PA_BSND"
    if case_name == "MlaPrologQuantV32STest.b1_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (1, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b4_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (4, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b8_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (8, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b16_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (16, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b64_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (64, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b128_s64k2_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (128, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s64k1_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 1, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 4, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s1k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 4, 1 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s4k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 4, 4 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s16k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 4, 16 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s128k4_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (32, 128, 4, 128 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # quant small shape
    elif case_name == "MlaPrologQuantV32STest.b1_s11_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (1, 128, 1, 1), output, True, False, False, 128,
                                       "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b1_s129_1_pa_nd_fp16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.float16, torch.float16), (1, 128, 1, 129), output, True, False, False,
                                       128, "PA_BSND")

    # bf16, quant, weight nd, "PA_BSND"
    elif case_name == "MlaPrologQuantV32STest.b1_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (1, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b4_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (4, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b8_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (8, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b16_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (16, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b64_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (64, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b128_s64k2_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (128, 128, 2, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s64k1_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 1, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s64k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 4, 64 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # quant
    elif case_name == "MlaPrologQuantV32STest.b32_s1k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 4, 1 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s4k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 4, 4 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s16k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 4, 16 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b32_s128k4_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 4, 128 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # quant small shape
    elif case_name == "MlaPrologQuantV32STest.b1_s11_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (1, 128, 1, 1), output, True, False, False,
                                       128, "PA_BSND")
    elif case_name == "MlaPrologQuantV32STest.b1_s129_1_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (1, 128, 1, 129), output, True, False, False,
                                       128, "PA_BSND")
    # unaligned shape
    elif case_name == "MlaPrologQuantV32STest.b104_s8k1_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (104, 128, 1, 8 * 1024), output, True, False,
                                       False, 128, "PA_BSND")
    # special case from test
    elif case_name == "MlaPrologQuantV32STest.b32_s127104_3_pa_nd_bf16_quantB":
        gen_mla_prolog_v32_quantB_test((torch.bfloat16, torch.bfloat16), (32, 128, 3, 127104), output, True, False,
                                       False, 128, "PA_BSND")
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


# MlaPrologQuantV32
case_names = [
    # fp16, quant, weight nd, "PA_BSND"
    "MlaPrologQuantV32STest.b1_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b4_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b8_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b16_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b64_s64k2_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b128_s64k2_pa_nd_fp16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s64k1_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s64k4_pa_nd_fp16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s1k4_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s4k4_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s16k4_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b32_s128k4_pa_nd_fp16_quantB",
    # small shape
    "MlaPrologQuantV32STest.b1_s11_pa_nd_fp16_quantB",
    "MlaPrologQuantV32STest.b1_s129_1_pa_nd_fp16_quantB",

    # bf16, quant, weight nd, "PA_BSND"
    "MlaPrologQuantV32STest.b1_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b4_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b8_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b16_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b64_s64k2_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b128_s64k2_pa_nd_bf16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s64k1_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s64k4_pa_nd_bf16_quantB",
    #
    "MlaPrologQuantV32STest.b32_s1k4_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s4k4_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s16k4_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b32_s128k4_pa_nd_bf16_quantB",
    # small shape
    "MlaPrologQuantV32STest.b1_s11_pa_nd_bf16_quantB",
    "MlaPrologQuantV32STest.b1_s129_1_pa_nd_bf16_quantB",

    # unaligned shape
    "MlaPrologQuantV32STest.b104_s8k1_pa_nd_bf16_quantB",

    # special case from test
    "MlaPrologQuantV32STest.b32_s127104_3_pa_nd_bf16_quantB",
]


@GoldenRegister.reg_golden_func(
    case_names=[
        *case_names
    ]
)
def gen_mla_prolog_quant_v32_date_one_case(case_name: str, output: Path) -> bool:
    if case_name in case_names:
        gen_mla_prolog_quant_v32_data_wrap(case_name, output)
        return True
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MlaPrologQuantV32STest.b4_s64k2_pa_nd_bf16_quantB"
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "../../build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_mla_prolog_quant_v32_date_one_case(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    # 只有当脚本作为主程序执行时，才会调用 main()
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    exit(0 if main() else 1)
