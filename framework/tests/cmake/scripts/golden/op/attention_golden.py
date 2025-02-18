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
""" Attention 相关用例 Golden 生成逻辑.
"""
import logging

from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import math
import torch

from golden.net.deepseekv3.mla.mla_prolog_golden_v2 import gen_mla_prolog_data
from golden.op.attention_post import gen_pa_data, gen_post_data

from golden_register import GoldenRegister

fp32 = np.float32


def gen_attention_test_data(dtypes, bns2, epsilon, output_dir: Path, is_quant=False, is_nz=False, is_smooth=False, cache_mode="BNSD"):
    b, n, s2, block_size, n_tile = bns2
    n2, s = 1, 1
    params = {
        "b": b,
        "s": s,
        "s2": s2,
        "h": 7168,
        "num_heads": n,
        "n2": n2,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    quant_choice = (False, is_quant)
    # [b,s,n,kv_lora_rank], [b,s,n,qk_rope_head_dim], [b,n2,s2,kv_lora_rank], [b,n2,s2,qk_rope_head_dim]
    print("gen_attention_test_data cache_mode is ", cache_mode)
    q_out, q_rope_out, kv_cache_out, kr_cache_out = \
        gen_mla_prolog_data(params, dtypes, epsilon, output_dir, quant_choice, is_nz, is_smooth, block_size, cache_mode)

    # reshape
    q_out = q_out.reshape(b, n, s, params["kv_lora_rank"])
    q_rope_out = q_rope_out.reshape(b, n, s, params["qk_rope_head_dim"])

    # pa_out: [b*n*s, kv_lora_rank], fp32
    pa_out = gen_pa_data(output_dir, params, dtypes[0], q_out, q_rope_out, kv_cache_out, kr_cache_out,
                         block_size, n_tile, is_nz)

    post_out = gen_post_data(output_dir, params, dtypes[0], pa_out, is_quant, is_nz)

    return post_out


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicAttention.dynamic_attention_low",
        "DynamicAttention.dynamic_attention_high",
        "DynamicAttention.low_latency_quant_smooth_nz",
        "DynamicAttention.high_throughput_quant_smooth_nz",
    ]
)
def gen_attention_date(case_name: str, output: Path) -> bool:
    print("gen_attention_date-------------------------- ")
    # b_n_s_s2_h_q_lora_rank
    if case_name == "DynamicAttention.dynamic_attention_low":
        gen_attention_test_data((np.float16, np.float16), (4, 32, 256, 256, 32), 1e-5, output, False, False, False)
    elif case_name == "DynamicAttention.dynamic_attention_high":
        gen_attention_test_data((np.float16, np.float16), (32, 128, 4096, 128, 128), 1e-5, output, False, False, False)
    elif case_name == "DynamicAttention.low_latency_quant_smooth_nz":
        gen_attention_test_data((np.float16, np.float16), (4, 32, 256, 128, 32), 1e-5, output, True, True, True)
    elif case_name == "DynamicAttention.high_throughput_quant_smooth_nz":
        gen_attention_test_data((np.float16, np.float16), (32, 128, 4096, 4096, 128), 1e-5, output, True, True, True, "PA_NZ")
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


if __name__ == "__main__":
    # 只有当脚本作为主程序执行时，才会调用 main()
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    gen_attention_date("DynamicAttention.dynamic_attention_low", "./")
