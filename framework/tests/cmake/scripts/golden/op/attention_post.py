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


# =================================== PaPost
def gen_post_quantize_np(input_fp32):
    abs_res = np.abs(input_fp32)
    max_value = np.max(abs_res, axis=-1, keepdims=True)
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_int8 = np.clip(out_int32, -128, 127).astype(np.int8)
    scale_dequant = 1.0 / scale_quant
    return out_int8, scale_dequant


def gen_post_quant_mm_np(a, w, scale_w, output: Path):
    a_fp32 = a.astype(np.float32)
    quantized_a, scale_dequant_a = gen_post_quantize_np(a_fp32)
    quant1_int8_path = Path(output, 'quant0_int8.bin')
    quant1_fp32_path = Path(output, 'quant0_fp32.bin')
    quantized_a.tofile(quant1_int8_path)
    scale_dequant_a.tofile(quant1_fp32_path)

    a_int32 = quantized_a.astype(np.int32)
    w_int32 = w.astype(np.int32)

    res_int32 = np.matmul(a_int32, w_int32)
    mm5_int32_path = Path(output, 'mm5_int32.bin')
    res_int32.tofile(mm5_int32_path)
    res = res_int32.astype(np.float32)
    mm5_fp32_path = Path(output, 'mm5_fp32.bin')
    res.tofile(mm5_fp32_path)
    res = res * scale_dequant_a
    res = res * scale_w
    return res.astype(a.dtype)


def gen_post_data(output_dir: Path, params, dtype, pa_out=None, is_quant=True, is_nz=True):
    input_b = params.get("b")
    input_s = params.get("s")
    input_n = params.get("num_heads")
    input_h = params.get("h")
    kv_lora_rank = params.get("kv_lora_rank")
    v_head_dim = params.get("v_head_dim")

    params_path = Path(output_dir, 'params.bin')
    input_path = Path(output_dir, 'input.bin')
    t1_path = Path(output_dir, 't1.bin')
    r1_path = Path(output_dir, 'r1.bin')
    cast1_path = Path(output_dir, 'cast1.bin')
    w_uv_path = Path(output_dir, 'w_uv.bin')
    w_uv_scale_w_path = Path(output_dir, 'w_uv_scale_w.bin')
    cast0_out_path = Path(output_dir, 'cast0_out.bin')
    abs_out_path = Path(output_dir, 'abs_out.bin')
    mul0_out_path = Path(output_dir, 'mul0_out.bin')
    rms_out_path = Path(output_dir, 'rms_out.bin')
    quant1_int8_path = Path(output_dir, 'quant0_int8.bin')
    quant1_fp32_path = Path(output_dir, 'quant0_fp32.bin')
    bmm4_int32_path = Path(output_dir, 'bmm4_int32.bin')
    bmm4_path = Path(output_dir, 'bmm4.bin')
    t3_path = Path(output_dir, 't3.bin')
    r2_path = Path(output_dir, 'r2.bin')
    w_o_path = Path(output_dir, 'w_o.bin')
    w_o_nd_path = Path(output_dir, 'w_o_nd.bin')
    w_o_scale_w_path = Path(output_dir, 'w_o_scale_w.bin')
    bmm5_path = Path(output_dir, 'bmm5.bin')
    attn_output_path = Path(output_dir, 'attn_output.bin')
    complete = (params_path.exists() and input_path.exists() and attn_output_path.exists())
    complete = False

    if complete:
        logging.debug("Case(), Golden complete.")
    else:
        logging.debug("Start generate papost golden ...")
        dtype_num = 0
        if dtype == np.float32:
            dtype_num = 0
        elif dtype == np.float16:
            dtype_num = 1
        elif dtype == bfloat16:
            dtype_num = 2

        params_num = np.array([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, dtype_num], dtype=np.int64)
        np.random.seed(0)
        input_t = pa_out
        if pa_out is None:
            input_t = np.random.randn(input_b * input_n * input_s, kv_lora_rank).astype(np.float32)

        w_uv = np.random.randn(input_n, kv_lora_rank, v_head_dim).astype(dtype)
        w_uv_scale_w = np.random.randn(input_n, 1, v_head_dim).astype(np.float32) * 0.001

        w_o = np.random.randint(-128, 128, size=(input_n * v_head_dim, input_h), dtype=np.int8)
        w_o_scale_w = np.random.randn(input_h).astype(np.float32) * 0.001
        w_o_scale_w.tofile(w_o_scale_w_path)

        params_num.tofile(params_path)
        input_t.tofile(input_path)
        w_uv.tofile(w_uv_path)
        w_uv_scale_w.tofile(w_uv_scale_w_path)

        w_o.tofile(w_o_nd_path)  # ND
        w_o_nz = np.reshape(w_o, (input_n * v_head_dim, input_h // 32, 32)) # INT8
        w_o_nz = np.transpose(w_o_nz, (1, 0, 2))
        w_o_nz.tofile(w_o_path)  # NZ

        cast1 = input_t.astype(dtype)
        cast1.tofile(cast1_path)

        r1 = np.reshape(cast1, (input_b * input_s, input_n, kv_lora_rank))
        r1.tofile(r1_path)
        t1 = np.transpose(r1, (1, 0, 2))
        t1.tofile(t1_path)

        calc_input = t1
        bmm4 = np.matmul(calc_input.astype(np.float32), w_uv.astype(np.float32))
        if dtype != np.float32:
            bmm4 = bmm4.astype(dtype)
        bmm4.tofile(bmm4_path)

        t3 = np.transpose(bmm4, (1, 0, 2))
        t3.tofile(t3_path)

        r2 = np.reshape(t3, (input_b * input_s, input_n * v_head_dim))
        r2.tofile(r2_path)
        bmm5_i = r2

        bmm5 = gen_post_quant_mm_np(bmm5_i, w_o, w_o_scale_w, output_dir)
        bmm5.tofile(bmm5_path)

        bmm5 = np.reshape(bmm5, (input_b, input_s, input_h))
        bmm5.tofile(attn_output_path)
    return bmm5


# =================================== Pa
def split_pa_tensor_by_b(input_tensor):
    # tensor:[[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    return split_data


def gen_pa_softmax(x):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_sum, x_max


def gen_pa_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def trans_pa_bnsd_to_bsh(tensor, shape):
    if len(shape) == 4:
        b = shape[0]
        n = shape[1]
        s = shape[2]
        d = shape[3]
        h = n * d
        return tensor.transpose(0, 2, 1, 3).reshape(b, s, h)
    else:
        return tensor


def gen_pa_data(output_dir: Path, params, dtype, q_out, q_rope_out, kv_cache_out, kr_cache_out, block_size=4096,
                n_tile=128, is_nz=False):
    np.random.seed(0)

    # b, n_q, skv, s_q, n_kv, kv_lora_rank, qk_rope_dim = params
    b = params.get("b")
    n_q = params.get("num_heads")
    skv = params.get("s2")
    s_q = params.get("s")
    n_kv = params.get("n2")
    kv_lora_rank = params.get("kv_lora_rank")
    qk_rope_dim = params.get("qk_rope_head_dim")

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim

    # k head dim
    d_k = kv_lora_rank + qk_rope_dim

    # v head dim
    d_v = kv_lora_rank

    scalar = d_q ** -0.5

    if isinstance(skv, int):
        actual_seq_len = [skv] * b
    elif isinstance(skv, list):
        if len(skv) == b:
            actual_seq_len = skv
        else:
            raise RuntimeError("unsupported skv list length")
    else:
        raise RuntimeError("unsupported skv data type")

    s_max = max(actual_seq_len)

    shape_q1 = [b, n_q, s_q, kv_lora_rank]
    shape_q2 = [b, n_q, s_q, qk_rope_dim]

    shape_k1 = [b, n_kv, s_max, kv_lora_rank]
    shape_k2 = [b, n_kv, s_max, qk_rope_dim]

    shape_k = [b, n_kv, s_max, d_k]
    shape_v = [b, n_kv, s_max, d_v]

    atten_out_shape = [b, n_q, s_q, d_v]

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0

    # gen q k v data
    q_bnsd1 = q_out
    q_bnsd2 = q_rope_out
    k_bnsd1 = kv_cache_out
    k_bnsd2 = kr_cache_out
    v_bnsd = k_bnsd1
    if q_out is None or q_rope_out is None or kv_cache_out is None or kr_cache_out is None:
        q_bnsd1 = gen_pa_uniform_data(shape_q1, -1, 1, dtype)
        q_bnsd2 = gen_pa_uniform_data(shape_q2, -1, 1, dtype)
        k_bnsd1 = gen_pa_uniform_data(shape_k1, -1, 1, dtype)
        k_bnsd2 = gen_pa_uniform_data(shape_k2, -1, 1, dtype)
        v_bnsd = k_bnsd1
    q_bnsd = np.concatenate((q_bnsd1, q_bnsd2), axis=-1)
    k_bnsd = np.concatenate((k_bnsd1, k_bnsd2), axis=-1)

    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
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
    # block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * block_table_shape[1]

    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    logging.debug("block_table %s", block_table)

    # gen kv cache. [block_num , block_size, H]
    k_cache = np.zeros([block_num, block_size, n_kv * d_k]).astype(dtype)
    v_cache = np.zeros([block_num, block_size, n_kv * d_v]).astype(dtype)

    logging.debug("dtype %s shape %s ", type(k_bnsd), k_bnsd.shape)

    k_tensor_bsh_raw = trans_pa_bnsd_to_bsh(k_bnsd, shape_k)
    v_tensor_bsh_raw = trans_pa_bnsd_to_bsh(v_bnsd, shape_v)

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
    k_tensor_list = split_pa_tensor_by_b(k_bnsd)
    v_tensor_list = split_pa_tensor_by_b(v_bnsd)

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
        softmax_res, softmax_sum, softmax_max = gen_pa_softmax(qk_ele_res)

        # MM2
        bmm2_res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
        attent_out[b_index:(b_index + 1), :, :, :] = bmm2_res
    attent_out = np.reshape(attent_out, (b * n_q * s_q, d_v))


    # data split to [nope + rope]
    q_nope = q_bnsd[:, :, :, : kv_lora_rank]
    q_rope = q_bnsd[:, :, :, kv_lora_rank:]

    # BBH split [B B kv_lora_rank]  + [B B rope]
    k_cache_nope_h = kv_lora_rank * n_kv
    k_cache_nope = k_cache[:, :, : k_cache_nope_h]
    k_cache_rope = k_cache[:, :, k_cache_nope_h:]

    # NZ 支持bf16/fp16的 NZ
    k_cache_nope_nz = k_cache_nope.reshape(k_cache_nope.shape[0], k_cache_nope.shape[1], k_cache_nope.shape[2] // 16,
                                           16)
    k_cache_rope_nz = k_cache_rope.reshape(k_cache_rope.shape[0], k_cache_rope.shape[1], k_cache_rope.shape[2] // 16,
                                           16)
    v_cache_nz = v_cache.reshape(v_cache.shape[0], v_cache.shape[1], v_cache.shape[2] // 16, 16)

    k_cache_nope_nz = np.transpose(k_cache_nope_nz, (0, 2, 1, 3))
    k_cache_rope_nz = np.transpose(k_cache_rope_nz, (0, 2, 1, 3))
    v_cache_nz = np.transpose(v_cache_nz, (0, 2, 1, 3))

    # input params
    # input_params = [b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, block_size, 128]
    input_params = np.array([b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, block_size, n_tile], dtype=np.int32)

    # dump golden file
    q_nope_path = Path(output_dir, 'q_nope.bin')
    q_nope.tofile(q_nope_path)

    q_rope_path = Path(output_dir, 'q_rope.bin')
    q_rope.tofile(q_rope_path)

    k_cache_nope_path = Path(output_dir, 'k_cache_nope.bin')
    k_cache_nope.tofile(k_cache_nope_path)

    k_cache_rope_path = Path(output_dir, 'k_cache_rope.bin')
    k_cache_rope.tofile(k_cache_rope_path)

    v_cache_path = Path(output_dir, 'v_cache.bin')
    v_cache.tofile(v_cache_path)

    k_cache_nope_nz_path = Path(output_dir, 'k_cache_nope_nz.bin')
    k_cache_nope_nz.tofile(k_cache_nope_nz_path )

    kv_cache_nope_nz_path = Path(output_dir, 'kv_cache_nope_nz.bin')
    k_cache_nope_nz.tofile(kv_cache_nope_nz_path)

    k_cache_rope_nz_path = Path(output_dir, 'k_cache_rope_nz.bin')
    k_cache_rope_nz.tofile(k_cache_rope_nz_path )

    v_cache_nz_path = Path(output_dir, 'v_cache_nz.bin')
    v_cache_nz.tofile(v_cache_nz_path)

    block_table_path = Path(output_dir, 'block_table.bin')
    block_table.tofile(block_table_path)

    actual_seq_len_path = Path(output_dir, 'actual_seq_len.bin')
    np.array(actual_seq_len).astype(np.int32).tofile(actual_seq_len_path)

    attent_out_path = Path(output_dir, 'atten_out.bin')
    attent_out.tofile(attent_out_path)

    input_param_path = Path(output_dir, 'input_param.bin')
    input_params.tofile(input_param_path)

    return attent_out


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


def gen_data_func_bf16(shape_size, dtype, case_name: str, output: Path):
    input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim = shape_size
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
    complete = (params_path.exists() and input_path.exists() and t1_path.exists()
                and r1_path.exists() and t2_path.exists() and w_uv_path.exists()
                and bmm4_path.exists() and t3_path.exists() and r2_path.exists()
                and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists())
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

        input_t = torch.randn([input_b, input_n, input_s, kv_lora_rank], dtype=dtype)  # [32, 32, 1, 512]
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
    return True


def quantize_torch(input_fp32):
    abs_res = torch.abs(input_fp32)
    max_value, _ = torch.max(abs_res, dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_int8 = torch.clamp(out_int32, -128, 127).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_int8, scale_dequant


def gen_quant_mm_torch(a, w, scale_w, output: Path):
    a_fp32 = a.to(torch.float32)
    quantized_a, scale_dequant_a = quantize_torch(a_fp32)
    quant1_int8_path = Path(output, 'quant0_int8.bin')
    quant1_fp32_path = Path(output, 'quant0_fp32.bin')
    tensor_bf16_tofile(quantized_a, quant1_int8_path)
    tensor_bf16_tofile(scale_dequant_a, quant1_fp32_path)

    a_int32 = quantized_a.to(torch.int32)
    w_int32 = w.to(torch.int32)
    res_int32 = torch.matmul(a_int32, w_int32)
    mm5_int32_path = Path(output, 'mm5_int32.bin')
    tensor_bf16_tofile(res_int32, mm5_int32_path)
    res = res_int32.to(torch.float32)
    mm5_fp32_path = Path(output, 'mm5_fp32.bin')
    tensor_bf16_tofile(res, mm5_fp32_path)
    res = res * scale_dequant_a
    res = res * scale_w
    return res.to(a.dtype)


def gen_data_func_bf16_quant_dynamic(shape_size, dtype, case_name: str, output: Path):
    input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim = shape_size
    params_path = Path(output, 'params.bin')
    input_path = Path(output, 'input.bin')
    t1_path = Path(output, 't1.bin')
    r1_path = Path(output, 'r1.bin')
    cast1_path = Path(output, 'cast1.bin')
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
    complete = (params_path.exists() and input_path.exists() and t1_path.exists() and r1_path.exists()
                and cast1_path.exists() and w_uv_path.exists() and bmm4_path.exists() and t3_path.exists()
                and r2_path.exists() and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists()
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

        input_t = torch.randn([input_b * input_n * input_s, kv_lora_rank], dtype=torch.float32)
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)
        w_uv_scale_w = torch.randn([input_n, 1, v_head_dim], dtype=torch.float32) * 0.001

        w_o = torch.randint(size=(input_n * v_head_dim, input_h), low=-128, high=128, dtype=torch.int8)
        w_o_scale_w = torch.randn([input_h], dtype=torch.float32) * 0.001

        params.numpy().tofile(params_path)
        tensor_bf16_tofile(input_t, input_path)
        tensor_bf16_tofile(w_uv, w_uv_path)
        tensor_bf16_tofile(w_uv_scale_w, w_uv_scale_w_path)

        w_o_nz = w_o.reshape(input_n * v_head_dim, input_h // 32, 32)
        w_o_nz = w_o_nz.transpose(0, 1)
        tensor_bf16_tofile(w_o_nz, w_o_path)  # (224, 16k, 32)     (16k, 7168)   NZ
        # 原tensor_bf16_tofile(w_o, w_o_path)  # ND
        tensor_bf16_tofile(w_o_scale_w, w_o_scale_w_path)

        r1 = input_t.reshape(input_b * input_s, input_n, kv_lora_rank)
        tensor_bf16_tofile(r1, r1_path)

        cast1 = r1.to(dtype)
        tensor_bf16_tofile(cast1, cast1_path)

        t1 = cast1.transpose(0, 1)
        tensor_bf16_tofile(t1, t1_path)

        calc_input = t1
        bmm4 = torch.matmul(calc_input.to(torch.float32), w_uv.to(torch.float32))
        if dtype != torch.float32:
            bmm4 = bmm4.to(dtype)

        tensor_bf16_tofile(bmm4, bmm4_path)

        t3 = bmm4.transpose(0, 1)
        tensor_bf16_tofile(t3, t3_path)
        r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
        tensor_bf16_tofile(r2, r2_path)
        bmm5_i = r2

        bmm5 = gen_quant_mm_torch(bmm5_i, w_o, w_o_scale_w, output)
        tensor_bf16_tofile(bmm5, bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        tensor_bf16_tofile(bmm5, attn_output_path)
    return True


def gen_data_func_bf16_quant_dynamic_cast(shape_size, dtype, case_name: str, output: Path):
    input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim = shape_size
    params_path = Path(output, 'params.bin')
    input_path = Path(output, 'input.bin')
    t1_path = Path(output, 't1.bin')
    r1_path = Path(output, 'r1.bin')
    cast1_path = Path(output, 'cast1.bin')
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
    w_o_nd_path = Path(output, 'w_o_nd.bin')
    w_o_scale_w_path = Path(output, 'w_o_scale_w.bin')
    bmm5_path = Path(output, 'bmm5.bin')
    attn_output_path = Path(output, 'attn_output.bin')
    complete = (params_path.exists() and input_path.exists()
                and t1_path.exists() and r1_path.exists()
                and cast1_path.exists() and w_uv_path.exists()
                and bmm4_path.exists() and t3_path.exists()
                and r2_path.exists() and w_o_path.exists()
                and attn_output_path.exists() and bmm5_path.exists()
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

        input_t = torch.randn([input_b * input_n * input_s, kv_lora_rank], dtype=torch.float32)
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)
        # w_uv = torch.full([input_n, kv_lora_rank, v_head_dim], 0.1, dtype=dtype)
        w_uv_scale_w = torch.randn([input_n, 1, v_head_dim], dtype=torch.float32) * 0.001

        w_o = torch.randint(size=(input_n * v_head_dim, input_h), low=-128, high=128, dtype=torch.int8)
        w_o_scale_w = torch.randn([input_h], dtype=torch.float32) * 0.001

        params.numpy().tofile(params_path)
        tensor_bf16_tofile(input_t, input_path)
        tensor_bf16_tofile(w_uv, w_uv_path)
        tensor_bf16_tofile(w_uv_scale_w, w_uv_scale_w_path)

        w_o_nz = w_o.reshape(input_n * v_head_dim, input_h // 32, 32)
        w_o_nz = w_o_nz.transpose(0, 1)
        tensor_bf16_tofile(w_o_nz, w_o_path)  # (224, 16k, 32)     (16k, 7168)   NZ
        tensor_bf16_tofile(w_o, w_o_nd_path)  # ND
        tensor_bf16_tofile(w_o_scale_w, w_o_scale_w_path)

        cast1 = input_t.to(dtype)
        tensor_bf16_tofile(cast1, cast1_path)

        r1 = cast1.reshape(input_b * input_s, input_n, kv_lora_rank)
        tensor_bf16_tofile(r1, r1_path)

        t1 = r1.transpose(0, 1)
        tensor_bf16_tofile(t1, t1_path)

        calc_input = t1
        bmm4 = torch.matmul(calc_input.to(torch.float32), w_uv.to(torch.float32))
        if dtype != torch.float32:
            bmm4 = bmm4.to(dtype)

        tensor_bf16_tofile(bmm4, bmm4_path)

        t3 = bmm4.transpose(0, 1)
        tensor_bf16_tofile(t3, t3_path)
        r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
        tensor_bf16_tofile(r2, r2_path)
        bmm5_i = r2

        bmm5 = gen_quant_mm_torch(bmm5_i, w_o, w_o_scale_w, output)
        tensor_bf16_tofile(bmm5, bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        tensor_bf16_tofile(bmm5, attn_output_path)
    return True


def gen_data_func_bf16_quant_onlymm5(shape_size, dtype, case_name: str, output: Path):
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
    complete = (params_path.exists() and input_path.exists() and t1_path.exists() and r1_path.exists()
                and t2_path.exists() and w_uv_path.exists() and bmm4_path.exists() and t3_path.exists()
                and r2_path.exists() and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists()
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
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)
        w_uv_scale_w = torch.randn([input_n, 1, v_head_dim], dtype=torch.float32) * 0.001

        w_o = torch.randint(size=(input_n * v_head_dim, input_h), low=-128, high=128, dtype=torch.int8)
        w_o_scale_w = torch.randn([input_h], dtype=torch.float32) * 0.001

        params.numpy().tofile(params_path)
        tensor_bf16_tofile(input_t, input_path)
        tensor_bf16_tofile(w_uv, w_uv_path)
        tensor_bf16_tofile(w_uv_scale_w, w_uv_scale_w_path)

        w_o_nz = w_o.reshape(input_n * v_head_dim, input_h // 32, 32)
        w_o_nz = w_o_nz.transpose(0, 1)
        tensor_bf16_tofile(w_o_nz, w_o_path)  # (224, 16k, 32)     (16k, 7168)   NZ
        # ND存储 tensor_bf16_tofile(w_o, w_o_path)  # ND
        tensor_bf16_tofile(w_o_scale_w, w_o_scale_w_path)

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

        bmm5 = gen_quant_mm_torch(bmm5_i, w_o, w_o_scale_w, output)
        tensor_bf16_tofile(bmm5, bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        tensor_bf16_tofile(bmm5, attn_output_path)
    return True


def attention_post_func_torch_bf16(case_name: str, output: Path):
    if case_name == "OnBoardTest.test_attention_post_bf16_real_batch4":
        input_b = 4
        input_s = 1
        input_n = 32
        input_h = 7168
        kv_lora_rank = 512
        v_head_dim = 128
        dtype = torch.bfloat16
        # 原torch.float32 torch.float16   torch.bfloat16
        gen_data_func_bf16((input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim), dtype, case_name, output)
    elif case_name == "OnBoardTest.test_attention_post_bf16_real_n128":
        input_b = 32
        input_s = 1
        input_n = 128
        input_h = 7168
        kv_lora_rank = 512
        v_head_dim = 128
        dtype = torch.bfloat16
        # 原torch.float32 torch.float16   torch.bfloat16
        gen_data_func_bf16((input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim), dtype, case_name, output)


def attention_post_func_torch_mm5(case_name: str, output: Path):
    if (case_name == "OnBoardCostTest.test_attention_post_bf16_real_quant_n128_onlymm5"
          or case_name == "OnBoardCostTest.test_attention_post_bf16_real_quant_n128_onlymm5K"
          or case_name == "OnBoardCostTest.test_attention_post_bf16_real_quant_n128_onlymm5He"):
        input_b = 32
        input_s = 1
        input_n = 128
        input_h = 7168
        kv_lora_rank = 512
        v_head_dim = 128
        dtype = torch.bfloat16
        # 原torch.float32 torch.float16   torch.bfloat16
        gen_data_func_bf16_quant_onlymm5((input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim), dtype,
                                         case_name, output)
    elif (case_name == "OnBoardCostTest.test_attention_post_bf16_real_quant_batch4_onlymm5"
          or case_name == "OnBoardCostTest.test_attention_post_bf16_real_quant_batch4_onlymm5K"):
        input_b = 4
        input_s = 1
        input_n = 32
        input_h = 7168
        kv_lora_rank = 512
        v_head_dim = 128
        dtype = torch.bfloat16
        # 原torch.float32 torch.float16   torch.bfloat16
        gen_data_func_bf16_quant_onlymm5((input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim), dtype,
                                         case_name, output)


def attention_post_func_numpy_b32(case_name: str, output: Path):
    if (case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_normal_unsplitk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_splitk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_r2"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_t3r2"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4tr"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_r1"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlyt1"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_t1"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlybmm4_fail"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlybmm4"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_bmm4"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_t3"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_quant"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4tr_quant_fail"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4tr_quant"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4trq_mm5nd"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4trq_mm5ndk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_unquant_r3"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_nd"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_nz"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_ndk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_mm5ndk_unquant_r3"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_nzk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_unsplitk"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_unsplitk"):
        input_b = 32
        input_s = 1
        input_n = 128
        input_h = 7168
        kv_lora_rank = 512
        v_head_dim = 128
        dtype = bfloat16
        params = {
            "b": input_b,
            "s": input_s,
            "h": input_h,
            "num_heads": input_n,
            "kv_lora_rank": kv_lora_rank,
            "v_head_dim": v_head_dim,
        }
        gen_post_data(output, params, dtype, None, True, True)


def attention_post_func_numpy_b4(case_name: str, output: Path):
    if (case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk_low"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_splitk_low"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_unsplitk_low"
          or case_name == "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_unsplitk_low"):
        input_b = 4
        input_s = 1
        input_n = 32
        input_hh = 7168
        kv_lora_rank1 = 512
        v_head_dim1 = 128
        dtype1 = bfloat16
        params1 = {
            "b": input_b,
            "s": input_s,
            "h": input_hh,
            "num_heads": input_n,
            "kv_lora_rank": kv_lora_rank1,
            "v_head_dim": v_head_dim1,
        }
        gen_post_data(output, params1, dtype1, None, True, True)


@GoldenRegister.reg_golden_func(
    case_names=[
        # attention post
        "OnBoardTest.test_attention_post_bf16_real_batch4",
        "OnBoardTest.test_attention_post_bf16_real_n128",
        "OnBoardCostTest.test_attention_post_bf16_real_quant_n128_onlymm5",
        "OnBoardCostTest.test_attention_post_bf16_real_quant_batch4_onlymm5",
        "OnBoardCostTest.test_attention_post_bf16_real_quant_n128_onlymm5K",
        "OnBoardCostTest.test_attention_post_bf16_real_quant_batch4_onlymm5K",
        "OnBoardCostTest.test_attention_post_bf16_real_quant_n128_onlymm5He",
        "OnBoardCostTest.dynamic_pa_post_static_cast_first",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_normal_unsplitk",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_splitk",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_r2",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_t3r2",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4tr",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_r1",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlyt1",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_t1",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlybmm4_fail",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlybmm4",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_bmm4",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_t3",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_quant",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4tr_quant_fail",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4tr_quant",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4trq_mm5nd",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_crtb4trq_mm5ndk",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_unquant_r3",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_nd",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_nz",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_ndk",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_mm5ndk_unquant_r3",
        "DynamicAttentionPostTest.dynamic_pa_post_cast_first_onlymm5_nzk",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_unsplitk",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_unsplitk",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_splitk_low",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_splitk_low",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nz_unsplitk_low",
        "DynamicAttentionPostTest.dynamic_pa_post_new_mm5nd_unsplitk_low",
        "DynamicAttentionPostTest.dynamic_pa_papost_bf16_b48",
    ]
)


def attention_post_func(case_name: str, output: Path) -> bool:
    attention_post_func_torch_bf16(case_name, output)
    attention_post_func_torch_mm5(case_name, output)
    attention_post_func_numpy_b32(case_name, output)
    attention_post_func_numpy_b4(case_name, output)
    if case_name == "OnBoardCostTest.dynamic_pa_post_static_cast_first":
        input_b = 32
        input_s = 1
        input_n = 128
        input_h = 7168
        kv_lora_rank = 512
        v_head_dim = 128
        dtype = torch.bfloat16
        # 原torch.float32 torch.float16   torch.bfloat16  hehre
        gen_data_func_bf16_quant_dynamic_cast((input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim),
                                              dtype, case_name, output)
    elif (case_name == "DynamicAttentionPostTest.dynamic_pa_papost_bf16_b48"):
        b = 48
        n_q = 128
        skv = 4096
        block_size = 4096
        n_tile = 128
        s_q = 1
        n_kv = 1
        kv_lora_rank = 512
        qk_rope_dim = 64
        dtype = bfloat16
        input_h = 7168
        v_head_dim = 128
        params = {
            "b": b,
            "s": s_q,
            "s2": skv,
            "num_heads": n_q,
            "n2": n_kv,
            "qk_rope_head_dim": qk_rope_dim,
            "kv_lora_rank": kv_lora_rank,
            "h": input_h,
            "v_head_dim": v_head_dim,
        }
        pa_out = gen_pa_data(output, params, dtype, None, None, None, None, block_size, n_tile, False)
        gen_post_data(output, params, dtype, pa_out, True, True)
    else:
        return True
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "OnBoardTest.test_attention_post_bf16_real_n128",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = attention_post_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
