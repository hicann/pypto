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


def quantize_np(input_fp32):
    abs_res = np.abs(input_fp32)
    max_value = np.max(abs_res, axis=-1, keepdims=True)
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_int8 = np.clip(out_int32, -128, 127).astype(np.int8)
    scale_dequant = 1.0 / scale_quant
    return out_int8, scale_dequant


def gen_quant_mm_np(a, w, scale_w, output: Path):
    a_fp32 = a.astype(np.float32)
    quantized_a, scale_dequant_a = quantize_np(a_fp32)
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


def gen_data_func_bf16_quant_dynamic_cast_np(attention_out, shape_size, dtype, case_name: str, output: Path):
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
        if dtype == np.float32:
            dtype_num = 0
        elif dtype == np.float16:
            dtype_num = 1
        elif dtype == bfloat16:
            dtype_num = 2

        params = np.array([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, dtype_num], dtype=np.int64)
        np.random.seed(0)
        # input_t = np.random.randn(input_b * input_n * input_s, kv_lora_rank).astype(np.float32)
        input_t = attention_out
        w_uv = np.random.randn(input_n, kv_lora_rank, v_head_dim).astype(dtype)
        w_uv_scale_w = np.random.randn(input_n, 1, v_head_dim).astype(np.float32) * 0.001

        w_o = np.random.randint(-128, 128, size=(input_n * v_head_dim, input_h), dtype=np.int8)
        w_o_scale_w = np.random.randn(input_h).astype(np.float32) * 0.001

        params.tofile(params_path)
        input_t.tofile(input_path)
        w_uv.tofile(w_uv_path)
        w_uv_scale_w.tofile(w_uv_scale_w_path)

        w_o.tofile(w_o_nd_path)  # ND
        w_o_nz = np.reshape(w_o, (input_n * v_head_dim, input_h // 32, 32))
        w_o_nz = np.transpose(w_o_nz, (1, 0, 2))
        w_o_nz.tofile(w_o_path)  # NZ
        w_o_scale_w.tofile(w_o_scale_w_path)

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

        bmm5 = gen_quant_mm_np(bmm5_i, w_o, w_o_scale_w, output)
        bmm5.tofile(bmm5_path)

        bmm5 = np.reshape(bmm5, (input_b, input_s, input_h))
        bmm5.tofile(attn_output_path)
    return True


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


@GoldenRegister.reg_golden_func(
    case_names=[
        # ifa
        "OnBoardPaCostTest.test_page_attention_low_latency_cost",
        "OnBoardPaCostTest.test_page_attention_low_latency_cost_precision",
        "OnBoardPaCostTest.test_page_attention_hight_throughput_cost",
        "OnBoardPaCostTest.test_page_attention_hight_throughput_cost_precision",
        "DynamicPATest.dynamic_pa_low_lantency",
        "DynamicPATest.dynamic_pa_low_lantency_imm_scalar",
        "DynamicPATest.dynamic_pa_low_lantency_unroll",
        "DynamicPATest.dynamic_pa_low_lantency_manual_unroll",
        "DynamicPATest.dynamic_pa_low_lantency_dyn_valid_shape",
        "DynamicPATest.dynamic_pa_high_throughput_dview_large",
        "DynamicPATest.dynamic_pa_noflash_unalign",
        "DynamicPATest.dynamic_pa_noflash",
        "DynamicPATest.dynamic_pa_low_lantency_dyn_unalign",
        "DynamicPAPOSTTest.dynamic_page_attention_adds_high_throughput_dview_large",
        "DynamicPAPOSTTest.dynamic_page_attention_adds_high_throughput_dview_large_single_out",
        "DynamicPAPOSTTest.dynamic_prolog_post_high_throughput_dview_large",
        "DynamicPATest.dynamic_pa_high_throughput_only_batch_loop",
        "DynamicPATest.dynamic_pa_high_throughput_dview_large_dyn_unalign",
    ]
)
def ifa_pa_func(case_name: str, output: Path) -> bool:
    gen_data_debug_mode = False
    n_tile = 32
    if case_name.startswith('OnBoardPaCostTest.test_page_attention_low_latency_cost'):
        b = 4
        n_q = 32
        skv = 256
        block_size = 256
    elif case_name.startswith('OnBoardPaCostTest.test_page_attention_hight_throughput_cost'):
        b = 32
        n_q = 128
        skv = 4096
        block_size = 512
    elif (case_name == "DynamicPATest.dynamic_pa_low_lantency"
          or case_name == "DynamicPATest.dynamic_pa_low_lantency_unroll"
          or case_name == "DynamicPATest.dynamic_pa_low_lantency_manual_unroll"
          or case_name == "DynamicPATest.dynamic_pa_low_lantency_dyn_valid_shape"):
        b = 4
        n_q = 32
        skv = 256
        block_size = 128
        n_tile = 32
    elif (case_name == "DynamicPATest.dynamic_pa_low_lantency_imm_scalar"):
        b = 1
        n_q = 128
        skv = 256
        block_size = 256
        n_tile = 32
    elif (case_name == "DynamicPATest.dynamic_pa_high_throughput_dview_large"
          or case_name == "DynamicPATest.dynamic_pa_high_throughput_only_batch_loop"
          or case_name == "DynamicPAPOSTTest.dynamic_page_attention_adds_high_throughput_dview_large"
          or case_name == "DynamicPAPOSTTest.dynamic_page_attention_adds_high_throughput_dview_large_single_out"
          or case_name == "DynamicPAPOSTTest.dynamic_prolog_post_high_throughput_dview_large"):
        b = 32
        n_q = 128
        skv = 4096
        block_size = 4096
        n_tile = 128
    elif case_name == "DynamicPATest.dynamic_pa_high_throughput_dview_large_dyn_unalign":
        b = 32
        n_q = 128
        skv = 4087
        block_size = 4096
        n_tile = 128
    elif case_name == "DynamicPATest.dynamic_pa_noflash_unalign":
        b = 4
        n_q = 32
        skv = [48, 100, 120, 123]
        block_size = 128
        n_tile = 32
    elif case_name == "DynamicPATest.dynamic_pa_noflash":
        b = 4
        n_q = 32
        skv = 128
        block_size = 128
        n_tile = 32
    elif case_name == "DynamicPATest.dynamic_pa_low_lantency_dyn_unalign":
        b = 4
        n_q = 32
        skv = 248
        block_size = 128
        n_tile = 32
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    np.random.seed(None)
    dtype = bfloat16
    s_q = 1
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

    shape_q = [b, n_q, sq, d_q]
    shape_k = [b, n_kv, s_max, d_k]
    shape_v = [b, n_kv, s_max, d_v]

    atten_out_shape = [b, n_q, sq, d_v]

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0

    # gen q k v data
    q_bnsd = gen_uniform_data(shape_q, -1, 1, dtype)
    k_bnsd = gen_uniform_data(shape_k, -1, 1, dtype)
    v_bnsd = k_bnsd[:, :, :, : kv_lora_rank]

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
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

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

    if gen_data_debug_mode == False:
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
    else:
        tiled_out = []
        n_loop = math.ceil(n_q / n_tile)
        for b_index in range(b):
            matmul_dtype = np.float32
            cur_seq = actual_seq_len[b_index]
            bn_per_batch =math.ceil(cur_seq / block_size)
            for n_idx in range(n_loop) :
                oiUpdate = []
                liUpdate = []
                miUpdate = []

                qi = q_bnsd[b_index, n_idx * n_tile: (n_idx + 1) * n_tile, :, :]
                qi = qi.reshape(-1, qi.shape[-1])

                for bn in range(block_num_per_batch[b_index]):
                    cur_block_idx = block_table[b_index][bn]
                    s2_tile_cur = min(block_size, cur_seq - bn * block_size)
                    kj = k_cache[cur_block_idx, 0 : s2_tile_cur, :]
                    vj = v_cache[cur_block_idx, 0 : s2_tile_cur, :]

                    kj = kj.reshape(s2_tile_cur , d_k)
                    vj = vj.reshape(s2_tile_cur, d_v)

                    # C1
                    sij = np.matmul(qi, np.transpose(kj), dtype=matmul_dtype)
                    sij_scale = sij * scalar
                    tilda_mij = sij_scale.max(axis=-1, keepdims=True)
                    t_sub = sij_scale - tilda_mij
                    tilda_pij = np.exp(t_sub)
                    tilda_lij = tilda_pij.sum(axis=-1, keepdims=True)

                    if bn == 0:
                        oi_Tmp = np.matmul(tilda_pij, vj, dtype=matmul_dtype)
                        if bn_per_batch == 1:
                            oiUpdate = oi_Tmp / tilda_lij
                        else:
                            oiUpdate = oi_Tmp
                        liUpdate = tilda_lij
                        miUpdate = tilda_mij
                        continue

                    oi = oiUpdate
                    li = liUpdate
                    mi = miUpdate

                    miNew = np.maximum(mi, tilda_mij)
                    t1 = mi - miNew
                    t2 = np.exp(t1)
                    t3 = tilda_mij - miNew
                    t4 = np.exp(t3)
                    t5 = t4 * tilda_lij
                    t6 = t2 * li
                    liNew = t6 + t5
                    q3 = oi * t2
                    q1 = np.matmul(tilda_pij, vj)
                    q2 = q1 * t4
                    oi_Tmp = q3 + q2
                    if bn == block_num_per_batch[b_index] - 1:
                        oiUpdate = oi_Tmp / liNew
                    else:
                        oiUpdate = oi_Tmp
                    liUpdate = liNew
                    miUpdate = miNew

                tiled_out.append(oiUpdate)
        attent_out = np.concatenate(tiled_out, 0)

    # data split to [nope + rope]
    q_nope = q_bnsd[:, :, :, : kv_lora_rank]
    q_rope = q_bnsd[:, :, :, kv_lora_rank:]

    # BBH split [B B kv_lora_rank]  + [B B rope]
    k_cache_nope_h = kv_lora_rank * n_kv
    k_cache_nope = k_cache[:, :, : k_cache_nope_h]
    k_cache_rope = k_cache[:, :, k_cache_nope_h:]

    # NZ
    k_cache_nope_nz = k_cache_nope.reshape(k_cache_nope.shape[0], k_cache_nope.shape[1], k_cache_nope.shape[2] // 16,
                                           16)
    k_cache_rope_nz = k_cache_rope.reshape(k_cache_rope.shape[0], k_cache_rope.shape[1], k_cache_rope.shape[2] // 16,
                                           16)
    v_cache_nz = v_cache.reshape(v_cache.shape[0], v_cache.shape[1], v_cache.shape[2] // 16, 16)

    k_cache_nope_nz = np.transpose(k_cache_nope_nz, (0, 2, 1, 3))
    k_cache_rope_nz = np.transpose(k_cache_rope_nz, (0, 2, 1, 3))
    v_cache_nz = np.transpose(v_cache_nz, (0, 2, 1, 3))

    # input params
    input_params = [b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, block_size, n_tile]

    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')

    k_cache_nope_path = Path(output, 'k_cache_nope.bin')
    k_cache_rope_path = Path(output, 'k_cache_rope.bin')
    v_cache_path = Path(output, 'v_cache.bin')

    k_cache_nope_nz_path = Path(output, 'k_cache_nope_nz.bin')
    kv_cache_nope_nz_path = Path(output, 'kv_cache_nope_nz.bin')
    k_cache_rope_nz_path = Path(output, 'k_cache_rope_nz.bin')
    v_cache_nz_path = Path(output, 'v_cache_nz.bin')

    block_table_path = Path(output, 'block_table.bin')
    actual_seq_len_path = Path(output, 'actual_seq_len.bin')
    block_size_path = Path(output, 'block_size.bin')
    attent_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')

    # dump golden file
    dump_file(q_nope, q_nope_path, "bf16")
    dump_file(q_rope, q_rope_path, "bf16")
    dump_file(k_cache_nope, k_cache_nope_path, "bf16")
    dump_file(k_cache_rope, k_cache_rope_path, "bf16")
    dump_file(v_cache, v_cache_path, "bf16")

    dump_file(k_cache_nope_nz, k_cache_nope_nz_path, "bf16")
    dump_file(k_cache_nope_nz, kv_cache_nope_nz_path, "bf16")
    dump_file(k_cache_rope_nz, k_cache_rope_nz_path, "bf16")
    dump_file(v_cache_nz, v_cache_nz_path, "bf16")

    dump_file(block_table, block_table_path, "int32")
    dump_file(actual_seq_len, actual_seq_len_path, "int32")
    dump_file(block_size, block_size_path, "int64")
    dump_file(attent_out, attent_out_path, "fp32")
    dump_file(input_params, input_param_path, "int32")

    if case_name.startswith('DynamicPAPOSTTest.dynamic_prolog_post_high_throughput_dview_large'):
        input_b = b
        input_s = s_q
        input_n = n_q
        input_h = 7168
        v_head_dim = 128
        gen_data_func_bf16_quant_dynamic_cast_np(attent_out,
                                                 (input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim),
                                                 dtype, case_name, output)
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "OnBoardPaCostTest.test_page_attention_low_latency_cost",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = ifa_pa_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
