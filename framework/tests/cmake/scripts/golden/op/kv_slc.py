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

CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
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
    g_src_root: Path = Path(Path(file).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

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

def numpy_topk(input_array, k, axis=-1):
    """
    实现类似PyTorch的torch.topk功能，返回指定维度上的前k个最大值及其索引。

    参数:
    input_array (np.ndarray): 输入数组
    k (int): 需要提取的最大值的数量
    axis (int): 操作的维度，默认为最后一个维度

    返回:
    values (np.ndarray): 前k个最大值
    indices (np.ndarray): 对应的索引
    """
    if k <= 0:
        raise ValueError("k必须为正整数")

    # 使用argpartition高效获取前k大元素的索引
    partitioned_indices = np.argpartition(input_array, -k, axis=axis)[..., -k:]

    # 提取对应值并生成降序排序的索引
    partitioned_values = np.take_along_axis(input_array, partitioned_indices, axis=axis)
    sorted_order = np.argsort(-partitioned_values, axis=axis)  # 负号实现降序

    # 调整索引顺序并获取最终结果
    final_indices = np.take_along_axis(partitioned_indices, sorted_order, axis=axis)
    final_values = np.take_along_axis(input_array, final_indices, axis=axis)

    return final_values, final_indices

def kv_slc_compute(compute_input_params, topk_indecies, topk_tensor_shape, kvNopeCache, krCache, block_table, actual_seq_len):
    block_size = compute_input_params[0]
    n2 = compute_input_params[1]
    front = compute_input_params[2]
    near = compute_input_params[3]
    topK = compute_input_params[4]
    l_prime = compute_input_params[5]

    b = topk_indecies.shape[0]
    s = topk_indecies.shape[1]
    rope_dim = krCache.shape[1]
    kv_lora_rank = kvNopeCache.shape[1]
    kv_cache_axis1 = kvNopeCache.shape[0]

    shape_k_slc_out = [b * n2 * s * topK * l_prime, rope_dim + kv_lora_rank]
    shape_v_slc_out = [b * n2 * s * topK * l_prime, kv_lora_rank]

    k_slc_out = np.zeros(shape_k_slc_out, kvNopeCache.dtype)
    v_slc_out = np.zeros(shape_v_slc_out, kvNopeCache.dtype)
    kv_slc_actual_seqs = np.zeros([b, s], dtype=np.int32)

    for batchIdx in range(b):
        for seqIdx in range(s):
            slcSeqLen = 0
            s_slc = topk_tensor_shape[batchIdx][seqIdx]
            for nkvIdx in range(n2):
                for topKIdx in range(topK):
                    if topKIdx < front:
                        position = topKIdx
                    elif topKIdx > topK - near - front:
                        position = s_slc - near + (topKIdx - (topK - front - near) - 1)
                    else:
                        position = topk_indecies[batchIdx][seqIdx][topKIdx - front]
                    block_idx_in_batch = int(position * l_prime / block_size)
                    tail = int(position * l_prime % block_size)
                    slcBlockIdx = block_table[batchIdx][block_idx_in_batch]
                    slcSeqLen = slcSeqLen + max(l_prime - max(position * l_prime + l_prime - actual_seq_len[batchIdx], 0), 0)
                    preIdx_out_base = batchIdx * s * n2 * topK * l_prime + seqIdx * n2 * topK * l_prime + nkvIdx * topK * l_prime + topKIdx * l_prime
                    preIdx_cache_base = slcBlockIdx * block_size + tail

                    k_slc_out[preIdx_out_base : preIdx_out_base + l_prime, 0:kv_lora_rank] = kvNopeCache[preIdx_cache_base : preIdx_cache_base + l_prime, 0:kv_lora_rank]
                    k_slc_out[preIdx_out_base : preIdx_out_base + l_prime, kv_lora_rank:kv_lora_rank + rope_dim] = krCache[preIdx_cache_base : preIdx_cache_base + l_prime, 0:rope_dim]
                    v_slc_out[preIdx_out_base : preIdx_out_base + l_prime, 0:kv_lora_rank] = kvNopeCache[preIdx_cache_base : preIdx_cache_base + l_prime, 0:kv_lora_rank]
            kv_slc_actual_seqs[batchIdx][seqIdx] = slcSeqLen

    return k_slc_out, v_slc_out, kv_slc_actual_seqs

def gen_block_table(b, actual_seq_len, block_size, output: Path):
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    s_max = max(actual_seq_len)
    # gen block table [b, s_max/block_size]
    block_table_shape = [b, math.ceil(s_max / block_size)]
    block_num = block_num_min

    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    block_table = [-1] * block_table_shape[1]

    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        block_idx = 0
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    logging.debug("block_table %s", block_table)
    block_table_path = Path(output, 'block_table.bin')
    dump_file(block_table, block_table_path, "int32")
    return block_num, block_table

def gen_i_o_tensor(input_param, s_slc, s2, dtype, output: Path):
    block_size = input_param[9]
    b = input_param[0]
    s = input_param[1]
    n2 = input_param[2]
    kv_lora_rank = input_param[3]
    rope_dim = input_param[4]
    front = input_param[5]
    near = input_param[6]
    topK = input_param[7]
    l_prime = input_param[8]

    actual_seq_len = [s2] * b
    actual_seq_len_path = Path(output, 'actual_seq_len.bin')
    dump_file(actual_seq_len, actual_seq_len_path, "int32")

    block_num, block_table = gen_block_table(b, actual_seq_len, block_size, output)

    shape_topk_indecies = [b, s, topK - front - near]
    shape_kvNopeCache = [block_num * block_size, n2 * kv_lora_rank]
    shape_krCache = [block_num * block_size, n2 * rope_dim]

    topk_indecies = gen_uniform_data(shape_topk_indecies, 0, s_slc, dtype=np.int32)
    topk_tensor_shape = np.zeros([b, s], dtype=np.int32)
    for batchIdx in range(b):
        for seqIdx in range(s):
            topk_tensor_shape[batchIdx][seqIdx] = s_slc

    kvNopeCache = gen_uniform_data(shape_kvNopeCache, -1, 1, dtype)
    krCache = gen_uniform_data(shape_krCache, -1, 1, dtype)

    kv_slc_actual_seqs = np.zeros([b, s], dtype=np.int32)

    if dtype == bfloat16:
        dump_dtype = "bf16"
    if dtype == np.float16:
        dump_dtype = "fp16"

    topk_tensor_path = Path(output, 'topk_tensor.bin')
    topk_indecies_path = Path(output, 'topk_tensor.bin')
    kv_nope_cache_path = Path(output, 'kv_nope_cache.bin')
    kr_cache_path = Path(output, 'k_rope_cache.bin')
    topk_tensor_shape_path = Path(output, 'topk_tensor_shape.bin')

    dump_file(topk_indecies, topk_tensor_path, "int32")
    dump_file(topk_tensor_shape, topk_tensor_shape_path, "int32")
    dump_file(kvNopeCache, kv_nope_cache_path, dump_dtype)
    dump_file(krCache, kr_cache_path, dump_dtype)

    shape_k_slc_out = [b * n2 * s * topK * l_prime, rope_dim + kv_lora_rank]
    shape_v_slc_out = [b * n2 * s * topK * l_prime, kv_lora_rank]

    k_slc_out = np.zeros(shape_k_slc_out, dtype)
    v_slc_out = np.zeros(shape_v_slc_out, dtype)

    compute_input_params = [block_size, n2, front, near, topK, l_prime]
    k_slc_out, v_slc_out, kv_slc_actual_seqs = kv_slc_compute(compute_input_params, topk_indecies, topk_tensor_shape, kvNopeCache, krCache, block_table, actual_seq_len)

    k_slc_out_path = Path(output, 'k_slc_out.bin')
    v_slc_out_path = Path(output, 'v_slc_out.bin')
    kv_slc_actual_seqs_path = Path(output, 'kv_slc_actual_seqs.bin')

    dump_file(k_slc_out, k_slc_out_path, dump_dtype)
    dump_file(v_slc_out, v_slc_out_path, dump_dtype)
    dump_file(kv_slc_actual_seqs, kv_slc_actual_seqs_path, "int32")

@GoldenRegister.reg_golden_func(
    case_names=[
        # slc
        "DynamicSlcTest.dynamic_p_slc_fp16",
        "DynamicSlcTest.dynamic_p_slc_bf16",
    ]
)

def kv_slc_func(case_name: str, output: Path) -> bool:
    gen_data_debug_mode = False
    if case_name.startswith('DynamicSlcTest.dynamic_p_slc_fp16'):
        block_size = 128
        b = 32
        s = 1
        s_slc = 128
        n2 = 1
        s2 = 8192
        kv_lora_rank = 512
        rope_dim = 64
        front = 1
        near = 2
        topK = 16
        l_prime = 64
        golden_input_params = [b, s, n2, kv_lora_rank, rope_dim, front, near, topK, l_prime, block_size]
        dtype = np.float16

    if case_name.startswith('DynamicSlcTest.dynamic_p_slc_bf16'):
        block_size = 128
        b = 16
        s = 1
        s_slc = 32
        n2 = 1
        s2 = 4096
        kv_lora_rank = 512
        rope_dim = 64
        front = 1
        near = 2
        topK = 16
        l_prime = 64
        golden_input_params = [b, s, n2, kv_lora_rank, rope_dim, front, near, topK, l_prime, block_size]
        dtype = bfloat16
    input_param_path = Path(output, 'input_param.bin')
    dump_file(golden_input_params, input_param_path, "int32")
    gen_i_o_tensor(golden_input_params, s_slc, s2, dtype, output=output)
    return True

def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicSlcTest.dynamic_p_slc_fp16",
        "DynamicSlcTest.dynamic_p_slc_bf16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = kv_slc_func(case_name=cs, output=output)
    return ret

if __name__ == "__main__":
    exit(0 if main() else 1)