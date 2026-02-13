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


def compute_attention(q, k, v, actualSeq, scalar, atten_out_shape):
    """
    计算注意力机制，支持不同批次的序列长度不同

    参数:
    q: 查询张量，形状 [b, s_q, n_q, d_q]
    k: 键张量，形状 [b, s_q, n_kv, s_max, d_k]
    v: 值张量，形状 [b, s_q, n_kv, s_max, d_v]
    actualSeq: 实际序列长度，形状 [b]

    返回:
    attention_output: 注意力输出，形状 [b, s_q, n_q, d_v]
    """
    # 提取维度信息
    b, s_q, n_q, d_q = q.shape
    _, _, n_kv, s_max, d_k = k.shape
    _, _, _, _, d_v = v.shape

    # 初始化输出张量
    attention_output = np.zeros(atten_out_shape)

    # 遍历每个批次
    for i in range(b):

        # 遍历每个s_q
        for j in range(s_q):
            # 获取当前批次的实际序列长度
            kv_seq_len = actualSeq[i][j]

            seq_len = max(kv_seq_len - s_q + 1 + j, 0) # s_q!=1 MTP场景下的casual计算
            print("==============cur s1 seq_len: ", seq_len)

            # 获取当前批次和s_q的q [n_q, d_q]
            q_bs = q[i, j]

            # 获取当前批次、s_q和n_kv的k和v [seq_len, d_k/d_v]
            k_bs = k[i, j, 0, :seq_len]  # n_kv=1
            v_bs = v[i, j, 0, :seq_len]  # n_kv=1

            # MM1
            qk_bmm_res = np.matmul(q_bs, k_bs.transpose(1, 0), dtype=np.float32)
            qk_ele_res = qk_bmm_res * scalar
            softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)

            # MM2
            bmm2_res = np.matmul(softmax_res, v_bs, dtype=np.float32) / softmax_sum

            # 存储结果
            attention_output[i, j] = bmm2_res

    return attention_output


@GoldenRegister.reg_golden_func(
    case_names=[
        # sa
        "DynamicSATest.slc_attn_fp16",
        "DynamicSATest.slc_attn_mtp_s1_2_fp16",
        "DynamicSATest.slc_attn_bf16_b48_s1_perf",
    ]
)
def sa_func(case_name: str, output: Path) -> bool:
    # print("========================sa golden")
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 16
    selectBlockSize = 64

    if case_name == "DynamicSATest.slc_attn_fp16":
        b = 32
        n_q = 128
        n_kv = 1
        s_q = 1
        skv = 1024
        dtype = np.float16
    elif case_name == "DynamicSATest.slc_attn_mtp_s1_2_fp16":
        b = 24
        n_q = 128
        n_kv = 1
        s_q = 2
        skv = 1024
        dtype = np.float16
    elif case_name == "DynamicSATest.slc_attn_bf16_b48_s1_perf":
        b = 48
        n_q = 128
        n_kv = 1
        s_q = 1
        skv = 1024
        dtype = bfloat16
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    np.random.seed(None)

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank

    scalar = d_q ** -0.5

    if isinstance(skv, int):
        actual_seq = [[skv] * s_q for _ in range(b)]
    elif isinstance(skv, list):
        if len(skv) == b and len(skv[0]) == s_q:
            actual_seq = skv
        else:
            raise RuntimeError("unsupported skv list length")
    else:
        raise RuntimeError("unsupported skv data type")

    s_max = topk * selectBlockSize

    shape_q = [b, s_q, n_q, d_q]
    shape_k = [b, s_q, n_kv, s_max, d_k]
    shape_v = [b, s_q, n_kv, s_max, d_v]

    atten_out_shape = [b, s_q, n_q, d_v]

    # gen q k v data
    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    k_bsnd = gen_uniform_data(shape_k, -1, 1, dtype)
    v_bsnd = k_bsnd[:, :, :, :, :kv_lora_rank]

    atten_out = compute_attention(q_bsnd, k_bsnd, v_bsnd, actual_seq, scalar, atten_out_shape)

    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]

    # input params
    input_params = [b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, s_max, s_max] # 保留一位

    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    k_slc_path = Path(output, 'k_slc.bin')
    v_slc_path = Path(output, 'v_slc.bin')

    actual_seq_path = Path(output, 'actual_seq.bin')
    atten_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')

    # dump golden file
    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(k_bsnd, k_slc_path, dtype)
    dump_file(v_bsnd, v_slc_path, dtype)

    # print("atten_out: ", atten_out)
    dump_file(actual_seq, actual_seq_path, np.int32)
    dump_file(atten_out, atten_out_path, np.float32)
    dump_file(input_params, input_param_path, np.int32)

    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicSATest.slc_attn_fp16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = sa_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
