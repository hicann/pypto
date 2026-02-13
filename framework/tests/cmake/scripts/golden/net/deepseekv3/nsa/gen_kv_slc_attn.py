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
import time

import torch
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


def gen_block_table(b, actual_seq_len, block_size):
    """生成块表，返回PyTorch张量"""
    block_num_per_batch = []
    block_num_min = 0
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    slc_s_max = max(actual_seq_len)
    block_table_shape = [b, math.ceil(slc_s_max / block_size)]
    block_num = block_num_min

    block_idx_list = torch.arange(0, block_num, 1, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_num)]

    block_table = torch.full((block_table_shape[0], block_table_shape[1]), -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0

    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1

    return block_num, block_table


def compute_attention(q, k, v, actualSeq, scalar, atten_out_shape):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    """
    # 提取维度信息
    b, s_q, n_q, d_q = q.shape
    _, _, n_kv, s_max, d_k = k.shape
    _, _, _, _, d_v = v.shape

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=torch.float32)

    # 遍历每个批次
    for i in range(b):
        # 遍历每个s_q
        for j in range(s_q):
            # 获取当前批次的实际序列长度
            kv_seq_len = actualSeq[i][j].item()

            # s_q!=1 MTP场景下的casual计算
            seq_len = max(kv_seq_len - s_q + 1 + j, 0)
            print("==============cur s1 seq_len: ",  seq_len)

            # 获取当前批次和s_q的q [n_q, d_q]
            q_bs = q[i, j]

            # 获取当前批次、s_q和n_kv的k和v [seq_len, d_k/d_v]
            k_bs = k[i, j, 0, :seq_len]  # n_kv=1
            v_bs = v[i, j, 0, :seq_len]  # n_kv=1

            # MM1: 矩阵乘法
            qk_bmm_res = torch.matmul(q_bs.float(), k_bs.transpose(1, 0).float())
            qk_ele_res = qk_bmm_res * scalar

            # Softmax计算
            softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)

            # MM2: 矩阵乘法
            bmm2_res = torch.matmul(softmax_res / softmax_sum, v_bs.float())

            # 存储结果
            attention_output[i, j] = bmm2_res

    return attention_output


def softmax(x):
    """PyTorch实现的softmax函数"""
    x = x.float()
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = torch.sum(y, dim=-1, keepdim=True)
    ans = y
    return ans, x_sum, x_max


def kv_slc_compute(compute_input_params, topk_indecies, topk_tensor_shape, kvNopeCache, krCache, block_table, actual_seq_len):
    """PyTorch实现的kv切片计算"""
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

    k_slc_out = torch.zeros(shape_k_slc_out, dtype=kvNopeCache.dtype)
    v_slc_out = torch.zeros(shape_v_slc_out, dtype=kvNopeCache.dtype)
    kv_slc_actual_seqs = torch.zeros([b, s], dtype=torch.int32)

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

                    slcSeqLen += min(l_prime, actual_seq_len[batchIdx] - position * l_prime)

                    preIdx_out_base = batchIdx * s * n2 * topK * l_prime + seqIdx * n2 * topK * l_prime + nkvIdx * topK * l_prime + topKIdx * l_prime
                    preIdx_cache_base = slcBlockIdx * block_size + tail

                    # 切片操作
                    k_slc_out[preIdx_out_base : preIdx_out_base + l_prime, 0:kv_lora_rank] = kvNopeCache[preIdx_cache_base : preIdx_cache_base + l_prime, 0:kv_lora_rank]
                    k_slc_out[preIdx_out_base : preIdx_out_base + l_prime, kv_lora_rank:kv_lora_rank + rope_dim] = krCache[preIdx_cache_base : preIdx_cache_base + l_prime, 0:rope_dim]
                    v_slc_out[preIdx_out_base : preIdx_out_base + l_prime, 0:kv_lora_rank] = kvNopeCache[preIdx_cache_base : preIdx_cache_base + l_prime, 0:kv_lora_rank]

            kv_slc_actual_seqs[batchIdx][seqIdx] = slcSeqLen

    return k_slc_out, v_slc_out, kv_slc_actual_seqs


def gen_kv_cache(params, actual_seq_list, dtype, output_dir):
    '''生成kv_cache, 包括kv_nope_cache, k_rope_cache, block_table'''
    print("============= gen kv_cache ================")
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
    shape_kv_nope_cache = [block_num * block_size, n2 * kv_lora_rank]
    shape_k_rope_cache = [block_num * block_size, n2 * rope_dim]

    kv_nope_cache = gen_uniform_data(shape_kv_nope_cache, -1, 1, dtype)
    k_rope_cache = gen_uniform_data(shape_k_rope_cache, -1, 1, dtype)

    kv_nope_cache_path = Path(output_dir, 'kv_nope_cache.bin')
    kr_cache_path = Path(output_dir, 'k_rope_cache.bin')
    block_table_path = Path(output_dir, 'block_table.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    dump_file(kv_nope_cache, kv_nope_cache_path, dtype)
    dump_file(k_rope_cache, kr_cache_path, dtype)
    dump_file(block_table, block_table_path, torch.int32)
    dump_file(actual_seq_list, kv_cache_actual_seq_path, torch.int32)

    return kv_nope_cache, k_rope_cache, block_table, block_num


def dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, kv_slc_out, kr_slc_out, kv_slc_actual_seqs, kv_cache_actual_seq, dtype, output_dir):
    topk_tensor_path = Path(output_dir, 'topk_tensor.bin')
    topk_tensor_shape_path = Path(output_dir, 'topk_tensor_shape.bin')
    kv_slc_out_path = Path(output_dir, 'kv_slc_out.bin')
    kr_slc_out_path = Path(output_dir, 'kr_slc_out.bin')
    kv_slc_actual_seqs_path = Path(output_dir, 'kv_slc_actual_seqs.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    dump_file(topk_indices, topk_tensor_path, torch.int32)
    dump_file(topk_tensor_shape, topk_tensor_shape_path, torch.int32)
    dump_file(kv_slc_out, kv_slc_out_path, dtype)
    dump_file(kr_slc_out, kr_slc_out_path, dtype)
    dump_file(kv_slc_actual_seqs, kv_slc_actual_seqs_path, torch.int32)
    dump_file(kv_cache_actual_seq, kv_cache_actual_seq_path, torch.int32)


def dump_slc_atten_file(q_bsnd, k_bsnd, v_bsnd, actual_seq, kv_lora_rank, dtype, output_dir, atten_out, input_params):
    # 数据分割为 [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]

    q_nope_path = Path(output_dir, 'q_nope.bin')
    q_rope_path = Path(output_dir, 'q_rope.bin')
    k_slc_path = Path(output_dir, 'k_slc.bin')
    v_slc_path = Path(output_dir, 'v_slc.bin')
    actual_seq_path = Path(output_dir, 'slc_actual_seq.bin')

    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(k_bsnd, k_slc_path, dtype)
    dump_file(v_bsnd, v_slc_path, dtype)
    dump_file(actual_seq, actual_seq_path, torch.int32)


def gen_kv_slc_attn_golden(params, dtypes, output_dir: Path, is_nz=False):
    '''gen_kv_slc_atten, 其中包括: gen_kv_slc及slc_attn'''
    dtype, w_dtype = dtypes
    logging.debug(f"gen_kv_slc_attn_golden  dtype:{dtype}, w_dtype:{w_dtype}")

    b = params.get("b")
    s = params.get("s")
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

    print("======kv_cache_actual_seq: ", kv_cache_actual_seq)

    skv_max = max(kv_cache_actual_seq)

    # 1. 设置shape
    # gen kv_slc
    shape_topk_indices = [b, s, topk - front - near]
    slc_k_shape = [b, s, n2, slc_s_max, k_dim]
    slc_v_shape = [b, s, n2, slc_s_max, v_dim]

    # gen slc atten
    shape_q = [b, s, n1, q_dim]
    slc_atten_out_shape = [b, s, n1, v_dim]

    # 设置随机种子
    torch.manual_seed(int(time.time()))

    # 2. 生成数据
    kv_nope_cache, k_rope_cache, block_table, block_num = gen_kv_cache(params, kv_cache_actual_seq, dtype, output_dir)

    # 计算公式：s_slc = (act_seq_len[bIdx] + s1Idx - s1 + 1 - cmp_block_size + slc_block_size) // slc_block_size
    topk_tensor_shape = torch.zeros([b, s], dtype=torch.int32)
    topk_indices = torch.zeros(shape_topk_indices, dtype=torch.int32)
    for bIdx in range(b):
        for s1Idx in range(s):
            s_slc = (kv_cache_actual_seq[bIdx] + s1Idx - s + 1 - cmp_block_size + slc_block_size) // slc_block_size
            topk_tensor_shape[bIdx][s1Idx] = s_slc
            topk_indices[bIdx][s1Idx] = gen_uniform_data([shape_topk_indices[-1]], 0, s_slc, torch.int32)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)

    # gen kv_slc
    print("========== gen kv_slc ==============")
    compute_input_params = [block_size, n2, front, near, topk, slc_block_size]
    k_slc_out, v_slc_out, kv_slc_actual_seqs = kv_slc_compute(
        compute_input_params, topk_indices, topk_tensor_shape,
        kv_nope_cache, k_rope_cache, block_table, kv_cache_actual_seq
    )
    dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, k_slc_out, v_slc_out, kv_slc_actual_seqs, kv_cache_actual_seq, dtype, output_dir)

    # slc atten
    print("========== gen slc_attn ==============")
    input_params = [b, s, n1, n2, kv_lora_rank, rope_dim, slc_s_max, slc_s_max]
    k_slc = torch.reshape(k_slc_out, slc_k_shape)  # [b*s*n2*slc_s_max, k_dim] -> [b, s, n2, slc_s_max, k_dim]
    v_slc = torch.reshape(v_slc_out, slc_v_shape)  # [b*s*n2*slc_s_max, v_dim] -> [b, s, n2, slc_s_max, v_dim]
    slc_atten = compute_attention(q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, softmax_scale, slc_atten_out_shape)
    dump_slc_atten_file(q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, kv_lora_rank, dtype, output_dir, slc_atten, input_params)

    dump_file(slc_atten, Path(output_dir, 'slc_attn_out.bin'), torch.float32)

    return True


def gen_kv_slc_attn_entry(dtypes, bs1s2h, quant_smooth, kv_cache_actual_seq, output_dir: Path):
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
        "kv_cache_actual_seq": kv_cache_actual_seq,
        "epsilon": epsilon,
        "cache_mode": cache_mode,
        "v_head_dim": v_head_dim,
        "is_quant": is_quant,
        "is_smooth": is_smooth,
    }
    gen_kv_slc_attn_golden(params, dtypes, output_dir)

    # 将变化的参数保存到文件中，供测试用例直接读取
    input_params = [params.get("b"), params.get("s"), params.get("s2"), params.get("n1"), params.get("n2")]
    input_params.append(1 if is_quant else 0)
    input_params.append(1 if is_smooth else 0)
    dump_file(input_params, Path(output_dir, 'input_params.bin'), torch.int32)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicKvSATest.kv_slc_attn_b48_s1_fp16_perf",
        "DynamicKvSATest.kv_slc_attn_b32_s2_bf16_perf",
        "DynamicKvSATest.kv_slc_attn_b2_s1_fp16",
    ]
)
def gen_kv_slc_attn_func(case_name: str, output: Path) -> bool:
    input_params_path = Path(output, 'input_params.bin')
    slc_attn_out_path = Path(output, 'slc_attn_out.bin')
    complete = (input_params_path.exists() and slc_attn_out_path.exists())

    # complete = False  # TODO: del complete
    if complete:
        logging.info("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "DynamicKvSATest.kv_slc_attn_b48_s1_fp16_perf":
            gen_kv_slc_attn_entry((torch.float16, torch.float16), (48, 1, 8192, 7168), (False, False), [8192] * 48, output)
        elif case_name == "DynamicKvSATest.kv_slc_attn_b32_s2_bf16_perf":
            gen_kv_slc_attn_entry((torch.bfloat16, torch.bfloat16), (32, 2, 32768, 7168), (False, False), 32768, output)
        elif case_name == "DynamicKvSATest.kv_slc_attn_b2_s1_fp16":
            gen_kv_slc_attn_entry((torch.float16, torch.float16), (2, 1, 131072, 7168), (False, False), [131072, 64 * 1024 + 35], output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def main() -> bool:
    """单独调试 入口函数"""
    # 用例名称
    case_name_list: List[str] = [
        "DynamicKvSATest.kv_slc_attn_b48_s1_fp16_perf",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = gen_kv_slc_attn_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
