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

TYPE_CONVERT = {
    np.float16: "fp16",
    bfloat16: "bf16",
    np.float32: "fp32",
    np.int32: "int32",
    np.int64: "int64"
}


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() in ('fp16', 'float16', 'half'):
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() in ('bf16', 'bfloat16'):
        np.array(data_pool).astype(bfloat16).tofile(data_path)
    elif type_str.lower() in ('fp32', 'float', 'float32'):
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() in ('fp64', 'float64', 'double'):
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == 'int8':
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == 'int16':
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() in ('int32', 'int'):
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


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype)


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


def generate_block_table(block_table_shape, actual_seq_len, block_size):
    # 处理pageatten场景（block table, kv cache处理不涉及cpu、真值计算，仅为npu生成输入）：
    # 1、生成随机的block_table，并覆写原有bin文件
    # 2、将kv shape 统一转换成bsh后处理
    # 3、生成kv cache
    # 4、将kv cache dump成新的bin文件，供aclnn接口调用

    # gen block table [b, s_max / block_size]

    block_num_per_block = []
    block_num_min = 0
    block_num = 0

    for actual_seq in actual_seq_len:
        block_num_per_block.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

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

    return block_num, block_table


def generate_query(q_bnsd, kv_lora_rank):
    q_nope = q_bnsd[:, :, :, : kv_lora_rank]
    q_rope = q_bnsd[:, :, :, kv_lora_rank:]
    return q_nope, q_rope


def generate_cache(tensor_bnsd, shape, block_table, block_num, block_size, dtype):
    # bnsd->bsh
    b = tensor_bnsd.shape[0]
    tensor_bsh_raw = trans_bnsd_to_bsh(tensor_bnsd, shape)
    # kv bsh paddIng
    tensor_bsh = np.zeros((b, block_table.shape[1] * block_size, shape[1] * shape[-1])).astype(dtype)
    tensor_bsh[:, :tensor_bsh_raw.shape[1], :] = tensor_bsh_raw[:, :, :]
    tensor_cache = np.zeros([block_num, block_size, shape[1] * shape[-1]]).astype(dtype)

    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                tensor_cache[kv_cache_blk_id, 0:block_size, :] = tensor_bsh[
                                                                 b_idx, block_offset:(block_offset + block_size), :]

    return tensor_cache


def calc_attention(atten_out_shape, q_bnsd, k_bnsd, v_bnsd, actual_seq_len, softmax_scale):
    # calculate result
    attent_out = np.zeros(atten_out_shape, dtype=np.float32)

    # 处理连续场景：将单个tensor依据B值拆成列表
    k_tensor_list = split_tensor_by_b(k_bnsd)
    v_tensor_list = split_tensor_by_b(v_bnsd)

    b = atten_out_shape[0]

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
        qk_ele_res = qk_bmm_res * softmax_scale
        softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)

        # MM2
        bmm2_res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
        attent_out[b_index:(b_index + 1), :, :, :] = bmm2_res

    return attent_out


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicPAPOSTTest.dynamic_prolog_post_low_lantency",
    ]
)
def pro_log_post_func(case_name: str, output: Path) -> bool:
    dtype = bfloat16
    softmax_scale = 0.8

    s_q = 1
    n_kv = 1
    kv_lora_rank = 128
    qk_rope_dim = 64
    v_head_dim = 32
    hidden_size = 64

    if case_name == "DynamicPAPOSTTest.dynamic_prolog_post_low_lantency":
        b = 2
        n_q = 32
        s_kv = 128
        block_size = 128
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    actual_seq_len = [s_kv for _ in range(b)]
    s_max = max(actual_seq_len)

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank

    shape_q = [b, n_q, s_q, d_q]
    shape_k = [b, n_kv, s_max, d_k]
    shape_v = [b, n_kv, s_max, d_v]
    atten_out_shape = [b, n_q, s_q, d_v]
    shape_w_uv = [n_q, kv_lora_rank, v_head_dim]
    shape_w_o = [n_q * v_head_dim, hidden_size]

    # calc max block_num and generate block_table
    block_table_shape = [b, math.ceil(s_max / block_size)]
    block_num, block_table = generate_block_table(block_table_shape, actual_seq_len, block_size)

    q_bnsd = gen_uniform_data(shape_q, -1, 1, dtype)
    q_nope, q_rope = generate_query(q_bnsd, kv_lora_rank)

    k_bnsd = gen_uniform_data(shape_k, -1, 1, dtype)
    k_cache = generate_cache(k_bnsd, shape_k, block_table, block_num, block_size, dtype)
    k_cache_nope = k_cache[:, :, : kv_lora_rank * n_kv]
    k_cache_rope = k_cache[:, :, kv_lora_rank * n_kv:]

    v_bnsd = gen_uniform_data(shape_v, -1, 1, dtype)
    v_cache = generate_cache(v_bnsd, shape_v, block_table, block_num, block_size, dtype)

    # calc res
    attent_out = calc_attention(atten_out_shape, q_bnsd, k_bnsd, v_bnsd, actual_seq_len, softmax_scale)

    weight_uv = gen_uniform_data(shape_w_uv, -1, 1, dtype)
    weight_o = gen_uniform_data(shape_w_o, -1, 1, dtype)

    # (n_q, b *s_q, kv_lora_rank)
    atten_trans = attent_out.transpose(0, 2, 1, 3).reshape(b * s_q, n_q, kv_lora_rank).transpose(1, 0, 2)

    # (n_q, b *s_q, kv_lora_rank) * (n_q, kv_lora_rank, v_head_dim) -> (n_q, b * s_q, v_head_dim)
    bmm_res = np.matmul(atten_trans, weight_uv, dtype=np.float32)

    bmm_trans = bmm_res.transpose(1, 0, 2).reshape(b * s_q, n_q * v_head_dim)

    # (b * s_q, n_q * v_head_dim) * (n_q * v_head_dim, hiiden_size) -> (b * s_q, hidden_size)
    mm_res = np.matmul(bmm_trans, weight_o, dtype=np.float32)

    post_out = mm_res.reshape(b, s_q, hidden_size)

    # get bin and dump data
    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    k_cache_nope_path = Path(output, 'k_cache_nope.bin')
    k_cache_rope_path = Path(output, 'k_cache_rope.bin')
    v_cache_path = Path(output, 'v_cache.bin')
    block_table_path = Path(output, 'block_table.bin')
    actual_seq_len_path = Path(output, 'actual_seq_len.bin')
    weight_uv_path = Path(output, "weight_uv.bin")
    weight_k_path = Path(output, "weight_o.bin")
    block_size_path = Path(output, 'block_size.bin')
    post_out_path = Path(output, 'post_out.bin')

    type_str = TYPE_CONVERT.get(dtype)
    dump_file(q_nope, q_nope_path, type_str)
    dump_file(q_rope, q_rope_path, type_str)
    dump_file(k_cache_nope, k_cache_nope_path, type_str)
    dump_file(k_cache_rope, k_cache_rope_path, type_str)
    dump_file(v_cache, v_cache_path, type_str)
    dump_file(weight_uv, weight_uv_path, type_str)
    dump_file(weight_o, weight_k_path, type_str)
    dump_file(block_table, block_table_path, "int32")
    dump_file(actual_seq_len, actual_seq_len_path, "int32")
    dump_file(block_size, block_size_path, "int64")
    dump_file(post_out, post_out_path, "fp32")
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicPAPOSTTest.dynamic_prolog_post_low_lantency",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = pro_log_post_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
