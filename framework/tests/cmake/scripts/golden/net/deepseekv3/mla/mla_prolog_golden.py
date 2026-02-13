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

""" MLA_prolog 子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
import math
from pathlib import Path
from typing import List

import numpy as np
import torch

from ml_dtypes import bfloat16

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

np.random.seed(0)

fp32 = np.float32


def tensor_bf16_tofile(t: torch.Tensor, output: Path):
    input_file_bin = open(str(output), "wb")
    for each in t:
        if t.dtype == torch.bfloat16:
            input_file_bin.write(each.view(torch.int16).numpy().tobytes())
        elif t.dtype == torch.float32:
            input_file_bin.write(each.view(torch.int32).numpy().tobytes())
        elif t.dtype == torch.int32:
            input_file_bin.write(each.numpy().tobytes())
        elif t.dtype == torch.int8:
            input_file_bin.write(each.numpy().tobytes())
        else:
            raise ValueError(f"Unsupported dtype: {t.dtype}")
    input_file_bin.close()


def gen_data_func(input_b, input_s, input_n, input_h,
                  kv_lora_rank, v_head_dim, dtype, case_name: str, output: Path) -> bool:
    params_path = Path(output, 'params.bin')
    input_path = Path(output, 'input.bin')
    t1_path = Path(output, 't1.bin')
    r1_path = Path(output, 'r1.bin')
    t2_path = Path(output, 't2.bin')
    w_uv_path = Path(output, 'w_uv.bin')
    bmm4_path = Path(output, 'bmm4.bin')
    t3_path = Path(output, 't3.bin')
    r2_path = Path(output, 'r2.bin')
    w_o_path = Path(output, 'w_o.bin')
    bmm5_path = Path(output, 'bmm5.bin')
    attn_output_path = Path(output, 'attn_output.bin')
    complete = (params_path.exists() and input_path.exists() and t1_path.exists() and r1_path.exists() and
                t2_path.exists() and w_uv_path.exists() and bmm4_path.exists() and t3_path.exists() and
                r2_path.exists() and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists())
    complete = False
    if complete:
        logging.debug("======= Case(%s), Golden complete.", case_name)
    else:
        dtype_num = 0
        if dtype == torch.float32:
            dtype_num = 0
        elif dtype == torch.float16:
            dtype_num = 1
        elif dtype == torch.bfloat16:
            dtype_num = 2
        params = torch.tensor([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, dtype_num],
                              dtype=torch.int64)

        # # (0, 1)
        input_t = torch.randn([input_b, input_n, input_s, kv_lora_rank], dtype=dtype)  # [32, 32, 1, 512]
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)  # [32, 512, 128]
        w_o = torch.randn([input_n * v_head_dim, input_h], dtype=dtype)  # [32 * 128, 7168]

        params.numpy().tofile(params_path)
        input_t.numpy().tofile(input_path)
        w_uv.numpy().tofile(w_uv_path)
        w_o.numpy().tofile(w_o_path)

        # 原[input_b, input_n, input_s, kv_lora_rank] -> [input_b, input_s, input_n, kv_lora_rank] ->
        # 原[input_b*input_s, input_n, kv_lora_rank] -> [input_n, input_b*input_s, kv_lora_rank]
        t1 = input_t.transpose(1, 2)
        t1.numpy().tofile(t1_path)
        r1 = t1.reshape(input_b * input_s, input_n, kv_lora_rank)
        r1.numpy().tofile(r1_path)
        t2 = r1.transpose(0, 1)
        t2.numpy().tofile(t2_path)
        calc_input = t2

        # 原[input_n, input_b*input_s, kv_lora_rank] @ [input_n, kv_lora_rank, v_head_dim] ->
        # 原[input_n ,input_b*input_s, v_head_dim]
        bmm4 = torch.matmul(calc_input.to(torch.float32), w_uv.to(torch.float32))
        if dtype != torch.float32:
            bmm4 = bmm4.to(dtype)
        bmm4.numpy().tofile(bmm4_path)

        # 原[input_n ,input_b*input_s, v_head_dim] -> [input_b*input_s, input_n ,v_head_dim] ->
        # 原[input_b, input_s, input_n*v_head_dim]
        t3 = bmm4.transpose(0, 1)
        t3.numpy().tofile(t3_path)
        r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
        r2.numpy().tofile(r2_path)
        bmm5_i = r2

        # 原[input_b*input_s, input_n*v_head_dim] @ [input_n*v_head_dim, input_h] -> [input_b*input_s, input_h]
        bmm5 = torch.matmul(bmm5_i.to(torch.float32), w_o.to(torch.float32))
        if dtype != torch.float32:
            bmm5 = bmm5.to(dtype)
        bmm5.numpy().tofile(bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        bmm5.numpy().tofile(attn_output_path)
    return True


def gen_data_func_bf16(shape_and_atten_res, dtype, output: Path):
    input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, atten_res = shape_and_atten_res
    params_path = Path(output, 'params.bin')
    input_path = Path(output, 'input.bin')
    t1_path = Path(output, 't1.bin')
    r1_path = Path(output, 'r1.bin')
    t2_path = Path(output, 't2.bin')
    w_uv_path = Path(output, 'w_uv.bin')
    bmm4_path = Path(output, 'bmm4.bin')
    t3_path = Path(output, 't3.bin')
    r2_path = Path(output, 'r2.bin')
    w_o_path = Path(output, 'w_o.bin')
    bmm5_path = Path(output, 'bmm5.bin')
    attn_output_path = Path(output, 'attn_output.bin')
    complete = (params_path.exists() and input_path.exists() and t1_path.exists() and r1_path.exists() and
                t2_path.exists() and w_uv_path.exists() and bmm4_path.exists() and t3_path.exists() and
                r2_path.exists() and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists())
    complete = False
    if complete:
        logging.info("Golden complete.", )
    else:
        dtype_num = 0
        if dtype == torch.float32:
            dtype_num = 0
        elif dtype == torch.float16:
            dtype_num = 1
        elif dtype == torch.bfloat16:
            dtype_num = 2
        params = torch.tensor([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, dtype_num],
                              dtype=torch.int64)

        input_t = torch.randn([input_b, input_n, input_s, kv_lora_rank], dtype=dtype)  # [32, 32, 1, 512]
        if atten_res.max() != 0:
            input_t = torch.from_numpy(atten_res).to(torch.bfloat16)
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)  # [32, 512, 128]
        w_o = torch.randn([input_n * v_head_dim, input_h], dtype=dtype)  # [32 * 128, 7168]

        params.numpy().tofile(params_path)
        tensor_bf16_tofile(input_t, input_path)
        tensor_bf16_tofile(w_uv, w_uv_path)
        tensor_bf16_tofile(w_o, w_o_path)

        t1 = input_t.transpose(1, 2)
        tensor_bf16_tofile(t1, t1_path)
        r1 = t1.reshape(input_b * input_s, input_n, kv_lora_rank)
        tensor_bf16_tofile(r1, r1_path)
        t2 = r1.transpose(0, 1)
        tensor_bf16_tofile(t2, t2_path)
        calc_input = t2

        bmm4 = torch.matmul(calc_input.to(torch.float32), w_uv.to(torch.float32))
        if dtype != torch.float32:
            bmm4 = bmm4.to(dtype)
        tensor_bf16_tofile(bmm4, bmm4_path)

        t3 = bmm4.transpose(0, 1)
        tensor_bf16_tofile(t3, t3_path)
        r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
        tensor_bf16_tofile(r2, r2_path)
        bmm5_i = r2

        bmm5 = torch.matmul(bmm5_i.to(torch.float32), w_o.to(torch.float32))
        if dtype != torch.float32:
            bmm5 = bmm5.to(dtype)
        tensor_bf16_tofile(bmm5, bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        tensor_bf16_tofile(bmm5, attn_output_path)


def quantize_torch(input_fp32):
    abs_res = torch.abs(input_fp32)
    max_value, _ = torch.max(abs_res, dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_int8 = torch.clamp(out_int32, -128, 127).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_int8, scale_dequant


def gen_quant_mm_torch(a, w, scale_w):
    a_fp32 = a.to(torch.float32)
    quantized_a, scale_dequant_a = quantize_torch(a_fp32)

    a_int32 = quantized_a.to(torch.int32)
    w_int32 = w.to(torch.int32)
    res_int32 = torch.matmul(a_int32, w_int32)
    res = res_int32.to(torch.float32)
    res = res * scale_dequant_a
    res = res * scale_w
    return res.to(a.dtype)


def gen_data_func_bf16_quant(shape_size, dtype, case_name: str, output: Path):
    input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim = shape_size
    params_path = Path(output, 'params.bin')
    input_path = Path(output, 'input.bin')
    t1_path = Path(output, 't1.bin')
    r1_path = Path(output, 'r1.bin')
    t2_path = Path(output, 't2.bin')
    w_uv_path = Path(output, 'w_uv.bin')
    w_uv_scale_w_path = Path(output, 'w_uv_scale_w.bin')
    cast0_out_path = Path(output, 'cast0_out.bin')
    abs_out_path = Path(output, 'abs_out.bin')
    mul0_out_path = Path(output, 'mul0_out.bin')
    rms_out_path = Path(output, 'rms_out.bin')
    quant1_int8_path = Path(output, 'quant0_int8.bin')
    quant1_fp32_path = Path(output, 'quant0_fp32.bin')
    bmm4_int32_path = Path(output, 'bmm4_int32.bin')
    bmm4_path = Path(output, 'bmm4.bin')
    t3_path = Path(output, 't3.bin')
    r2_path = Path(output, 'r2.bin')
    w_o_path = Path(output, 'w_o.bin')
    w_o_scale_w_path = Path(output, 'w_o_scale_w.bin')
    bmm5_path = Path(output, 'bmm5.bin')
    attn_output_path = Path(output, 'attn_output.bin')
    complete = (params_path.exists() and input_path.exists() and t1_path.exists() and r1_path.exists() and
                t2_path.exists() and w_uv_path.exists() and bmm4_path.exists() and t3_path.exists() and
                r2_path.exists() and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists()
                and w_uv_scale_w_path.exists() and w_o_scale_w_path.exists())
    complete = False
    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        dtype_num = 0
        if dtype == torch.float32:
            dtype_num = 0
        elif dtype == torch.float16:
            dtype_num = 1
        elif dtype == torch.bfloat16:
            dtype_num = 2
        params = torch.tensor([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, dtype_num],
                              dtype=torch.int64)

        input_t = torch.randn([input_b, input_n, input_s, kv_lora_rank], dtype=dtype)
        w_uv = torch.randint(size=(input_n, kv_lora_rank, v_head_dim), low=-128, high=128, dtype=torch.int8)
        w_uv_scale_w = torch.randn([input_n, 1, v_head_dim], dtype=torch.float32) * 0.001

        w_o = torch.randint(size=(input_n * v_head_dim, input_h), low=-128, high=128, dtype=torch.int8)
        w_o_scale_w = torch.randn([input_h], dtype=torch.float32) * 0.001

        params.numpy().tofile(params_path)
        tensor_bf16_tofile(input_t, input_path)
        tensor_bf16_tofile(w_uv, w_uv_path)
        tensor_bf16_tofile(w_uv_scale_w, w_uv_scale_w_path)
        tensor_bf16_tofile(w_o, w_o_path)
        tensor_bf16_tofile(w_o_scale_w, w_o_scale_w_path)

        t1 = input_t.transpose(1, 2)
        tensor_bf16_tofile(t1, t1_path)
        r1 = t1.reshape(input_b * input_s, input_n, kv_lora_rank)
        tensor_bf16_tofile(r1, r1_path)
        t2 = r1.transpose(0, 1)
        tensor_bf16_tofile(t2, t2_path)
        calc_input = t2

        a_fp32 = calc_input.to(torch.float32)
        tensor_bf16_tofile(a_fp32, cast0_out_path)

        abs_res = torch.abs(a_fp32)
        tensor_bf16_tofile(abs_res, abs_out_path)
        max_value, _ = torch.max(abs_res, dim=-1, keepdim=True)
        tensor_bf16_tofile(max_value, rms_out_path)
        scale_quant = 127.0 / max_value
        out_fp32 = a_fp32 * scale_quant
        tensor_bf16_tofile(out_fp32, mul0_out_path)
        out_int32 = torch.round(out_fp32).to(torch.int32)
        out_int8 = torch.clamp(out_int32, -128, 127).to(torch.int8)
        scale_dequant = 1.0 / scale_quant
        quantized_a = out_int8
        tensor_bf16_tofile(quantized_a, quant1_int8_path)
        scale_dequant_a = scale_dequant
        tensor_bf16_tofile(scale_dequant_a, quant1_fp32_path)
        a_int32 = quantized_a.to(torch.int32)
        w_int32 = w_uv.to(torch.int32)
        res_int32 = torch.matmul(a_int32, w_int32)
        tensor_bf16_tofile(res_int32, bmm4_int32_path)
        res = res_int32.to(torch.float32)
        res = res * scale_dequant_a
        res = res * w_uv_scale_w
        bmm4 = res.to(calc_input.dtype)

        # 原bmm4 = gen_quant_mm_torch(calc_input, w_uv, w_uv_scale_w)
        tensor_bf16_tofile(bmm4, bmm4_path)

        t3 = bmm4.transpose(0, 1)
        tensor_bf16_tofile(t3, t3_path)
        r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
        tensor_bf16_tofile(r2, r2_path)
        bmm5_i = r2

        bmm5 = gen_quant_mm_torch(bmm5_i, w_o, w_o_scale_w)
        tensor_bf16_tofile(bmm5, bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        tensor_bf16_tofile(bmm5, attn_output_path)
    return True


def rms_norm_torch(x):
    import torch
    x_dtype = x.dtype
    eps = 1e-6
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    red_sum = torch.sum(mean_res, dim=-1, keepdim=True)
    red_brc = red_sum.expand_as(x)
    red_sqrt = torch.sqrt(red_brc + eps)
    res = x_f32 / red_sqrt

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def rms_norm(x):
    x_dtype = x.dtype
    eps = 1e-6
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.astype(fp32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = np.sum(mean_res, axis=-1, keepdims=True) + eps
    reduce_sqrt = np.sqrt(reduce_sum)
    res = x_f32 / reduce_sqrt

    if x_dtype != fp32:
        res = res.astype(x_dtype)
    return res


def scatter_update(inputs, axis, kv_lora_rank, qk_rope_head_dim):
    # inputs: past_key_states, key_states, indices
    # past_key_states shape: [b, 1, s2, d], [b, 1, s2, kv_lora_rank + qk_rope_head_dim]
    # key_states shape: [b, 1, 1, kv_lora_rank + qk_rope_head_dim]
    # indices shape: [1]
    past_key_states, key_states, indices = inputs
    b, n, s2, d = past_key_states.shape
    index = indices[0]
    logging.debug(f"scatter_update  past_key_states, b:{b}, n:{n}, s2:{s2}, d:{d}")
    logging.debug(f"scatter_update  indices, index:{index}")
    res = past_key_states
    if axis == -2:
        for b in range(b):
            for s_i in range(s2):
                if s_i == index:
                    logging.debug("find the index value and to replace!")
                    res[b][0][index][:] = key_states[b][0][0][:]
    return res


def scatter_update_new(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [b, 1, s2, d]
    # key_states shape: [b, 1, s1, d]
    # indices shape: [b, s1]
    cache, key_states, indices = inputs
    b, n2, s2, d = cache.shape  # n2=1
    s1 = indices.shape[1]
    logging.debug(f"scatter_update  cache, b:{b}, n2:{n2}, s2:{s2}, d:{d}")
    res = cache
    if axis == -2:
        for b_i in range(b):
            for s2_i in range(s2):
                for s1_i in range(s1):
                    index_value = indices[b_i][s1_i]
                    if s2_i == index_value:
                        logging.debug("find the index value and to replace!")
                        res[b_i][0][s2_i][:] = key_states[b_i][0][s1_i][:]
    return res


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(qk, cossin, position_ids, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    """
    q, k = qk
    cos, sin = cossin
    input_dtype = q.dtype
    if input_dtype != fp32:
        q = q.astype(fp32)
        k = k.astype(fp32)
    if cos.dtype != fp32:
        cos = cos.astype(fp32)
        sin = sin.astype(fp32)

    cos = np.expand_dims(cos[position_ids], axis=unsqueeze_dim)  # [b,1,s,qk_d]
    sin = np.expand_dims(sin[position_ids], axis=unsqueeze_dim)  # [b,1,s,qk_d]
    logging.debug("expand sin.shape: %s", sin.shape)
    logging.debug("expand cos.shape: %s", cos.shape)

    b, n, s, d = q.shape
    q = q.reshape(b, n, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, n, s, d)  # [b,n,s,qk_d]

    b, n, s, d = k.shape
    k = k.reshape(b, n, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, n, s, d)  # [b,1,s,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != fp32:
        q_embed, k_embed = q_embed.astype(input_dtype), k_embed.astype(input_dtype)
    return q_embed, k_embed


def quant(input_t, is_pertoken=True):
    input_fp32 = input_t.astype(fp32)
    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def gen_mla_prolog_data(params, dtype, w_dtype, output_dir: Path, is_quant=False, new_scatter=False):
    logging.debug(f"gen_mla_prolog_data  dtype:{dtype}, w_dtype:{w_dtype}")
    b = params.get("b")
    s = params.get("s")  # s=1
    s2 = params.get("s2")  # s2=4k
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    v_head_dim = params.get("v_head_dim")
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    position_ids_shape = [b, s]
    cos_shape = [s, qk_rope_head_dim]
    past_key_states_shape = [b, 1, s2, kv_lora_rank + qk_rope_head_dim]
    kv_len_shape = [1]
    if new_scatter:
        kv_len_shape = [b, s]
    logging.debug("x shape is %s", x_shape)
    logging.debug("w_qa shape is %s", w_qa_shape)
    logging.debug("w_qb shape is %s", w_qb_shape)
    logging.debug("w_kv_a shape is %s", w_kv_a_shape)
    logging.debug("w_kv_b_k shape is %s", w_kv_b_k_shape)
    logging.debug("position_ids shape is %s", position_ids_shape)
    logging.debug("cos sin shape is %s", cos_shape)
    logging.debug("past_key_states shape is %s", past_key_states_shape)
    logging.debug("kv_len shape is %s", kv_len_shape)

    x_path = Path(output_dir, 'x.bin')
    w_qa_path = Path(output_dir, 'w_qa.bin')
    w_qb_path = Path(output_dir, 'w_qb.bin')
    w_qb_scale_path = Path(output_dir, 'w_qb_scale.bin')
    w_kv_a_path = Path(output_dir, 'w_kv_a.bin')
    w_kv_b_k_path = Path(output_dir, 'w_kv_b_k.bin')  # kv_b_proj_w_k
    position_ids_path = Path(output_dir, 'position_ids.bin')
    cos_path = Path(output_dir, 'cos.bin')
    sin_path = Path(output_dir, 'sin.bin')
    past_key_states_path = Path(output_dir, 'past_key_states.bin')
    kv_len_path = Path(output_dir, 'kv_len.bin')
    q_golden_path = Path(output_dir, 'q_golden.bin')
    kv_golden_path = Path(output_dir, 'kv_golden.bin')
    q0_golden_path = Path(output_dir, 'q0_golden.bin')
    q1_golden_path = Path(output_dir, 'q1_golden.bin')
    k0_golden_path = Path(output_dir, 'k0_golden.bin')
    k1_golden_path = Path(output_dir, 'k1_golden.bin')
    v0_golden_path = Path(output_dir, 'v0_golden.bin')

    x = np.random.uniform(-1, 1, x_shape).astype(dtype)
    x.tofile(x_path)
    w_qa = np.random.uniform(-0.1, 0.1, w_qa_shape).astype(w_dtype)
    w_qa.tofile(w_qa_path)
    w_qb = np.random.uniform(-0.1, 0.1, w_qb_shape).astype(w_dtype)
    w_qb_quant, w_qb_scale = w_qb, w_qb
    if is_quant:
        w_qb_quant, w_qb_scale = quant(w_qb, False)
        w_qb_quant.tofile(w_qb_path)
        w_qb_scale.tofile(w_qb_scale_path)
        logging.debug("w_qb_scale shape is %s", w_qb_scale.shape)
        logging.debug("%s", w_qb_scale)
    else:
        w_qb.tofile(w_qb_path)

    w_kv_a = np.random.uniform(-0.1, 0.1, w_kv_a_shape).astype(w_dtype)
    w_kv_a.tofile(w_kv_a_path)
    w_kv_b_k = np.random.uniform(-0.1, 0.1, w_kv_b_k_shape).astype(w_dtype)
    w_kv_b_k.tofile(w_kv_b_k_path)
    position_ids = np.random.randint(0, cos_shape[0], size=position_ids_shape).astype(np.int32)
    position_ids.tofile(position_ids_path)
    cos = np.random.uniform(-0.1, 0.1, cos_shape).astype(dtype)  # [s, qk_d]
    sin = np.random.uniform(-0.1, 0.1, cos_shape).astype(dtype)  # [s, qk_d]
    cos.tofile(cos_path)
    sin.tofile(sin_path)
    past_key_states = np.random.uniform(-1, 1, past_key_states_shape).astype(dtype)
    past_key_states.tofile(past_key_states_path)
    kv_len = np.random.randint(0, s2, size=kv_len_shape).astype(np.int32)
    if new_scatter:
        kv_len = np.random.randint(0, s2, size=kv_len_shape).astype(np.int64)
    kv_len.tofile(kv_len_path)

    # numpy
    # q
    logging.debug("================ numpy ================")
    x_2d = x.reshape(b * s, h)
    # shape is: [b * s, h] * [h, q_lora_rank] = [b * s, q_lora_rank]
    q_a_proj = np.matmul(x_2d.astype(fp32), w_qa.astype(fp32))  # q_a_proj
    q_a_proj = q_a_proj.astype(dtype)
    q_a_layernorm = rms_norm(q_a_proj)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    # shape is: [b * s, q_lora_rank] * [q_lora_rank, n * q_head_dim] = [b * s, n * q_head_dim]
    if is_quant:
        q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm)
        q_b_proj = np.matmul(q_a_layernorm.astype(np.int32), w_qb_quant.astype(np.int32))  # q_b_proj

        # dequant
        q_b_proj_fp32 = q_b_proj.astype(fp32)
        q_b_proj_fp32_dequant = q_b_proj_fp32 * q_a_layernorm_scale_dequant
        q_b_proj = q_b_proj_fp32_dequant * w_qb_scale
    else:
        q_b_proj = np.matmul(q_a_layernorm.astype(fp32), w_qb.astype(fp32))  # q_b_proj

    q_b_proj = q_b_proj.astype(dtype)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)

    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    q_nope = q_reshape[:, :, :, 0:qk_nope_head_dim]  # [b, s, n, qk_nope_head_dim]
    q_nope_r = q_nope.reshape(b * s, n, qk_nope_head_dim)
    q_nope_t = q_nope_r.transpose(1, 0, 2)  # [n, b*s, qk_nope_head_dim]
    # shape is: [n, b*s, qk_nope_head_dim] * [n, qk_nope_head_dim, kv_lora_rank] = [n, b*s, kv_lora_rank]
    q_nope_new = np.matmul(q_nope_t.astype(fp32), w_kv_b_k.astype(fp32))
    q_nope_new = q_nope_new.astype(dtype)
    q_nope_new_t = q_nope_new.transpose(1, 0, 2)  # [b*s, n, kv_lora_rank]
    q_nope_new_r = q_nope_new_t.reshape(b, s, n, kv_lora_rank)  # [b, s, n, kv_lora_rank]
    q_nope_new_t2 = q_nope_new_r.transpose(0, 2, 1, 3)  # [b, n, s, kv_lora_rank]

    # kv
    # shape is: [b * s, h] * [h, kv_lora_rank + qk_rope_head_dim] = [b * s, kv_lora_rank + qk_rope_head_dim]
    kv_a_proj = np.matmul(x_2d.astype(fp32), w_kv_a.astype(fp32))  # kv_a_proj
    kv_a_proj = kv_a_proj.astype(dtype)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    compressed_kv = kv_reshape[:, :, 0:kv_lora_rank]  # [b, s, kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv)
    compressed_kv_r = compressed_kv_norm.reshape(b, s, 1, kv_lora_rank)
    k_nope = compressed_kv_r.transpose(0, 2, 1, 3)  # [b, 1, s, kv_lora_rank]

    # RoPE
    q_pe = q_reshape[:, :, :, qk_nope_head_dim:]  # [b, s, n, qk_rope_head_dim]
    q_pe_t = q_pe.transpose(0, 2, 1, 3)  # [b, n, s, qk_rope_head_dim]

    k_pe = kv_reshape[:, :, kv_lora_rank:]  # [b, s, qk_rope_head_dim]
    k_pe_r = k_pe.reshape(b, 1, s, qk_rope_head_dim)

    # q_embed: [b, n, s, qk_rope_head_dim], k_embed: [b, 1, s, qk_rope_head_dim]
    q_embed, k_embed = apply_rotary_pos_emb([q_pe_t, k_pe_r], [cos, sin], position_ids)

    """ q output """
    q_res = np.concatenate((q_nope_new_t2, q_embed), axis=-1)  # [b, n, s, kv_lora_rank + qk_rope_head_dim]
    q_res.tofile(q_golden_path)
    q_nope_new_t2.tofile(q0_golden_path)
    q_embed.tofile(q1_golden_path)

    """ kv output """
    key_states = np.concatenate((k_nope, k_embed), axis=-1)  # [b, 1, s, kv_lora_rank + qk_rope_head_dim]
    # kv cache: scatter_update
    past_key_states_new = scatter_update([past_key_states, key_states, kv_len], -2, kv_lora_rank, qk_rope_head_dim)
    if new_scatter:
        past_key_states_new = scatter_update_new([past_key_states, key_states, kv_len], -2)
    past_key_states_new.tofile(kv_golden_path)  # [b, 1, s2, kv_lora_rank + qk_rope_head_dim]

    k0 = past_key_states_new[:, :, :, :kv_lora_rank]
    k1 = past_key_states_new[:, :, :, kv_lora_rank:]
    v0 = past_key_states_new[:, :, :, :kv_lora_rank]
    k0.tofile(k0_golden_path)
    k1.tofile(k1_golden_path)
    v0.tofile(v0_golden_path)

    return q_res, past_key_states_new


def gen_mla_prolog_test1(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 2,  #
        "q_lora_rank": 512,  #
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_mla_prolog_test3(dtype, w_dtype, output_dir: Path):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 1024,  # 7168
        "num_heads": 32,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir)


def gen_mla_prolog_test_net1(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,  # 4096
        "h": 7168,
        "num_heads": 32,  # 128
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_mla_prolog_test_net1_b4(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 4,
        "s": 1,
        "s2": 256,  # 4096
        "h": 7168,
        "num_heads": 32,  # 128
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_mla_prolog_test_net2(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 32,
        "s": 1,
        "s2": 4096,
        "h": 7168,
        "num_heads": 128,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant)


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == 'fp16':
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == 'fp32':
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == 'fp64':
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == 'int8':
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == 'int16':
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == 'int32':
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == 'int64':
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == 'uint8':
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == 'uint16':
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == 'uint32':
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == 'uint64':
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == 'complex64':
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == 'complex128':
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == 'bool':
        np.array(data_pool).astype(np.bool_).tofile(data_path)
    elif type_str.lower() == 'bf16':
        np.array(data_pool).astype(bfloat16).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def trans_bnsd_to_bsh(tensor, shape):
    if len(shape) == 4:
        b = shape[0]
        n = shape[1]
        s = shape[2]
        d = shape[3]
        h = n * d
        return tensor.transpose(0, 2, 1, 3).reshape(b, s, h)
    else:
        return tensor


def split_tensor_shape_by_b(input_list):
    # [[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    list_len = input_list[0]
    list_new = []
    for _ in range(0, list_len):
        list_new_item = [1, input_list[1], input_list[2], input_list[3]]
        list_new.append(list_new_item)
    return list_new


def split_tensor_by_b(input_tensor):
    # tensor:[[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    return split_data


def softmax(x):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_sum, x_max


def net1(info, output: Path):
    dtype = bfloat16
    b, n_q, s2, q_res, kv_res = info

    # n_q = 1
    n_kv = 1

    kv_lora_rank = 512
    qk_rope_dim = 64

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim

    # k head dim
    d_k = kv_lora_rank + qk_rope_dim

    # v head dim
    d_v = kv_lora_rank

    sq = 1
    block_size = 256
    scalar = 0.8  # 临时
    actual_seq_len = [s2] * b
    s_max = max(actual_seq_len)

    shape_q = [b, n_q, sq, d_q]
    shape_k = [b, n_kv, s_max, d_k]
    shape_v = [b, n_kv, s_max, d_v]

    atten_out_shape = [b, n_q, sq, d_v]

    block_num_per_block = []
    block_num_min = 0
    block_num = 0

    # gen q k v data
    q_bnsd = gen_uniform_data(shape_q, -1, 1, dtype)
    k_bnsd = gen_uniform_data(shape_k, -1, 1, dtype)
    v_bnsd = gen_uniform_data(shape_v, -1, 1, dtype)

    if q_res.max() != 0 and kv_res.max() != 0:
        q_bnsd = q_res
        k_bnsd = kv_res
        v_bnsd = kv_res[:, :, :, :512]  # k0

    for actual_seq in actual_seq_len:
        block_num_per_block.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    # 处理pageatten场景（block table, kv cache处理不涉及cpu、真值计算，仅为npu生成输入）：
    # 1、生成随机的block_table，并覆写原有bin文件
    # 2、将kv shape 统一转换成bsh后处理
    # 3、生成kv cache
    # 4、将kv cache dump成新的bin文件，供aclnn接口调用

    # gen block table [b, s_max/block_size]
    block_table_shape = [b, math.ceil(s_max / block_size)]
    block_num = block_num_min

    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * block_table_shape[1]

    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_block:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    # input_t = np.arange(0, b *n *s *d, 1).reshape(qkv_shape).astype(dtype_f32)
    block_table = np.arange(0, b * s2 // block_size, 1).reshape(b, s2 // block_size).astype(np.int32)
    logging.debug(f"block_table :%s", block_table)

    # gen kv cache. [block_num , block_size, H]
    k_cache = np.zeros([block_num, block_size, n_kv * d_k]).astype(dtype)
    v_cache = np.zeros([block_num, block_size, n_kv * d_v]).astype(dtype)

    logging.debug(f"dtype %s %s, shape %s", type(k_bnsd), k_bnsd.shape)

    k_tensor_bsh_raw = trans_bnsd_to_bsh(k_bnsd, shape_k)
    v_tensor_bsh_raw = trans_bnsd_to_bsh(v_bnsd, shape_v)

    # kv paddIng
    k_tensor_bsh = np.zeros((b, block_table_shape[1] * block_size, n_kv * d_k)).astype(dtype)
    v_tensor_bsh = np.zeros((b, block_table_shape[1] * block_size, n_kv * d_v)).astype(dtype)

    k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
    v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]

    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache[kv_cache_blk_id, 0:block_size, :] = k_tensor_bsh[
                                                            b_idx, block_offset:(block_offset + block_size), :]
                v_cache[kv_cache_blk_id, 0:block_size, :] = v_tensor_bsh[
                                                            b_idx, block_offset:(block_offset + block_size), :]

    # calculate result
    attent_out = np.zeros(atten_out_shape, dtype=np.float32)

    # 处理连续场景：将单个tensor依据B值拆成列表

    k_tensor_list = split_tensor_by_b(k_bnsd)
    v_tensor_list = split_tensor_by_b(v_bnsd)

    for b_index in range(b):
        matmul_dtype = np.float32

        act_seq = actual_seq_len[b_index]

        k_sub_tensor = k_tensor_list[b_index]
        v_sub_tensor = v_tensor_list[b_index]

        q_tensor_cur = q_bnsd[b_index:(b_index + 1), :, :, :]
        k_cur = k_sub_tensor[:, :, :act_seq, :]
        v_cur = v_sub_tensor[:, :, :act_seq, :]

        # MM1
        qk_bmm_res = np.matmul(q_tensor_cur, k_cur.transpose(0, 1, 3, 2), dtype=matmul_dtype)
        qk_ele_res = qk_bmm_res * scalar
        softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)

        # MM2
        bmm2_res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
        attent_out[b_index:(b_index + 1), :, :, :] = bmm2_res

    # data split to [nope + rope]
    q_nope = q_bnsd[:, :, :, : kv_lora_rank]
    q_rope = q_bnsd[:, :, :, kv_lora_rank:]

    # BBH split [B B kv_lora_rank]  + [B B rope]
    k_cache_nope_h = kv_lora_rank * n_kv
    k_cache_nope = k_cache[:, :, : k_cache_nope_h]
    k_cache_rope = k_cache[:, :, k_cache_nope_h:]

    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    k_cache_nope_path = Path(output, 'k_cache_nope.bin')
    k_cache_rope_path = Path(output, 'k_cache_rope.bin')
    v_cache_path = Path(output, 'v_cache.bin')
    block_table_path = Path(output, 'block_table.bin')
    actual_seq_len_path = Path(output, 'actual_seq_len.bin')
    block_size_path = Path(output, 'block_size.bin')
    attent_out_path = Path(output, 'atten_out.bin')

    # dump golden file
    dump_file(q_nope, q_nope_path, "bf16")
    dump_file(q_rope, q_rope_path, "bf16")
    dump_file(k_cache_nope, k_cache_nope_path, "bf16")  # 100% k0
    dump_file(k_cache_rope, k_cache_rope_path, "bf16")
    dump_file(v_cache, v_cache_path, "bf16")
    dump_file(block_table, block_table_path, "int32")
    dump_file(actual_seq_len, actual_seq_len_path, "int32")
    dump_file(block_size, block_size_path, "int64")
    dump_file(attent_out, attent_out_path, "fp32")
    return attent_out


def gen_mla_prolog_test_net2_b4(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 4,
        "s": 1,
        "s2": 4096,
        "h": 7168,
        "num_heads": 128,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_mla_prolog_test_net3(dtype, w_dtype, output_dir: Path):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 7168,
        "num_heads": 128,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir)


def gen_mla_prolog_test_net3_b4(dtype, w_dtype, output_dir: Path):
    params = {
        "b": 4,
        "s": 1,
        "s2": 256,
        "h": 7168,
        "num_heads": 128,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir)


def gen_mla_prolog_test_net4_b4(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 4,
        "s": 1,
        "s2": 4096,
        "h": 7168,
        "num_heads": 32,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_attention_data(params, dtype, w_dtype, output_dir: Path, is_quant=False, new_scatter=True):
    input_b = params.get("b")
    input_s = params.get("s")
    input_n = params.get("num_heads")
    input_h = params.get("h")
    kv_lora_rank = params.get("kv_lora_rank")
    v_head_dim = params.get("v_head_dim")
    s2 = params.get("s2")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")

    q_res, kv_res = gen_mla_prolog_data(params, dtype, w_dtype, output_dir, is_quant, new_scatter)
    info = (input_b, input_n, s2, q_res, kv_res)
    atten_res = net1(info, output_dir)
    dtype = torch.bfloat16
    # 原torch.float32 torch.float16   torch.bfloat16
    gen_data_func_bf16((input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, atten_res), dtype, output_dir)


@GoldenRegister.reg_golden_func(
    case_names=[
        # MLA prolog
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_2_1_256_256_512",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_32_1_256_1024_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_32_1_256_1024_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_32_1_256_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_32_1_256_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_128_1_4096_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_128_1_256_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_128_1_4096_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_256_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_4096_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_128_1_256_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_128_1_4096_7168_1536",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_2_1_256_256_512_quant",
        "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_256_7168_1536_quant",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_32_1_256_7168_1536_quant",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant",
        "MlaPrologOnBoardCostTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant",
        "MlaPrologOnBoardTest.test_MlaProlog_float16_2_32_1_4096_7168_1536",
        "MlaPrologOnBoardTest.attention_bf16_4_1024_1024_32_256",
        "MlaPrologOnBoardTest.attention_bf16_test",
        "MlaPrologOnBoardTest.attention_bf16_high",
        "MlaPrologOnBoardTest.attention_bf16_low",
    ]
)
def gen_mla_prolog_date(case_name: str, output: Path) -> bool:
    x_path = Path(output, 'x.bin')
    w_qa_path = Path(output, 'w_qa.bin')  # q_a_proj_w
    w_qb_path = Path(output, 'w_qb.bin')  # q_b_proj_w
    w_kv_a_path = Path(output, 'w_kv_a.bin')  # kv_a_proj_with_mqa_w
    w_kv_b_k_path = Path(output, 'w_kv_b_k.bin')  # kv_b_proj_w_k
    position_ids_path = Path(output, 'position_ids.bin')
    cos_path = Path(output, 'cos.bin')
    sin_path = Path(output, 'sin.bin')
    past_key_states_path = Path(output, 'past_key_states.bin')
    kv_len_path = Path(output, 'kv_len.bin')
    q_golden_path = Path(output, 'q_golden.bin')
    kv_golden_path = Path(output, 'kv_golden.bin')

    complete = (x_path.exists() and w_qa_path.exists() and w_qb_path.exists() and w_kv_a_path.exists()
                and w_kv_b_k_path.exists() and position_ids_path.exists() and cos_path.exists() and sin_path.exists()
                and past_key_states_path.exists() and kv_len_path.exists()
                and q_golden_path.exists() and kv_golden_path.exists())
    complete = False

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True
    else:
        # b_n_s_s2_h_q_lora_rank
        if case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_2_1_256_256_512":
            gen_mla_prolog_test1(np.float16, np.float16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_32_1_256_1024_1536":
            gen_mla_prolog_test3(np.float16, np.float16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_32_1_256_1024_1536":
            gen_mla_prolog_test3(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_32_1_256_7168_1536":
            gen_mla_prolog_test_net1(np.float16, np.float16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_32_1_256_7168_1536":
            gen_mla_prolog_test_net1(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_128_1_4096_7168_1536":
            gen_mla_prolog_test_net2(np.float16, np.float16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_128_1_4096_7168_1536":
            gen_mla_prolog_test_net2(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_32_128_1_256_7168_1536":
            gen_mla_prolog_test_net3(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_4096_7168_1536":
            gen_mla_prolog_test_net4_b4(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_256_7168_1536":
            gen_mla_prolog_test_net1_b4(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_128_1_4096_7168_1536":
            gen_mla_prolog_test_net2_b4(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_128_1_256_7168_1536":
            gen_mla_prolog_test_net3_b4(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_2_1_256_256_512_quant":
            gen_mla_prolog_test1(np.float16, np.float16, output, True)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_bfloat16_4_32_1_256_7168_1536_quant":
            gen_mla_prolog_test_net1_b4(bfloat16, bfloat16, output, True)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_32_1_256_7168_1536_quant":
            gen_mla_prolog_test_net1(np.float16, np.float16, output, True)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant":
            gen_mla_prolog_test_net2(np.float16, np.float16, output, True)
        elif case_name == "MlaPrologOnBoardCostTest.test_MlaProlog_float16_32_128_1_4096_7168_1536_quant":
            gen_mla_prolog_test_net2(np.float16, np.float16, output, True)
        elif case_name == "MlaPrologOnBoardTest.test_MlaProlog_float16_2_32_1_4096_7168_1536":
            params = {
                "b": 4,
                "s": 1,
                "s2": 4096,
                "h": 7168,
                "num_heads": 32,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "v_head_dim": 128,
            }
            gen_attention_data(params, bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.attention_bf16_4_1024_1024_32_256":
            params = {
                "b": 4,
                "s": 1,
                "s2": 1024,
                "h": 1024,
                "num_heads": 32,
                "q_lora_rank": 256,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "v_head_dim": 128,
            }
            gen_attention_data(params, bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.attention_bf16_test":
            params = {
                "b": 4,
                "s": 1,
                "s2": 256,
                "h": 128,
                "num_heads": 32,
                "q_lora_rank": 128,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "v_head_dim": 128,
            }
            gen_attention_data(params, bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.attention_bf16_high":
            params = {
                "b": 32,
                "s": 1,
                "s2": 4096,
                "h": 7168,
                "num_heads": 128,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "v_head_dim": 128,
            }
            gen_attention_data(params, bfloat16, bfloat16, output)
        elif case_name == "MlaPrologOnBoardTest.attention_bf16_low":
            params = {
                "b": 4,
                "s": 1,
                "s2": 256,
                "h": 7168,
                "num_heads": 32,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "v_head_dim": 128,
            }
            gen_attention_data(params, bfloat16, bfloat16, output)
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
        "MlaPrologOnBoardTest.attention_bf16_4_1024_1024_32_256",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_mla_prolog_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
