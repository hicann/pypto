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
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np

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

np.random.seed(0)
B, N, S, D = [1, 1, 128, 128]
BSN = B * N * S
ND = N * D
BS = B * S

fp16 = np.float16
fp32 = np.float32


def rms_norm(x):
    res = x
    eps = 1e-6

    square = x * x
    mean_coff = 1.0 / x.shape[1]
    mean_res = square * mean_coff

    red_sum = np.sum(mean_res, axis=-1, keepdims=True)
    red_brc = np.broadcast_to(red_sum, x.shape)
    red_sqrt = np.sqrt(red_brc + eps)

    res = x / red_sqrt
    return res

    for i in range(x.shape[0]):
        res[i] = res[i] / red_sqrt[i]
    return res


def softmax(x, axis=None):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def multi_attention(x, attn_weight):
    # qkv matmul
    logging.debug(f'B {B} S {S} N {N} D {D}')
    qkv_fp16 = np.matmul(x.astype(fp32), attn_weight.astype(fp32)).astype(fp16)  # (bs,nd) @ (nd,nd*3) -> (bs,3*nd)
    q, k, v = qkv_fp16[:, 0:ND], qkv_fp16[:, ND:ND * 2], qkv_fp16[:, ND * 2:ND * 3]
    q = q.reshape(B, S, N, D).transpose(0, 2, 1, 3)  # b,n,s,d
    k = k.reshape(B, S, N, D).transpose(0, 2, 1, 3)  # b,n,s,d
    v = v.reshape(B, S, N, D).transpose(0, 2, 1, 3)  # b,n,s,d
    # flash attention
    qk = np.matmul(q.astype(fp32), k.transpose(0, 1, 3, 2).astype(fp32))  # b,n,s,s
    softmax_res, x_max, x_sum = softmax(qk.astype(fp32))  # b,n,s,s
    drop_res = softmax_res * 1  # assume mask is 1    b,n,s,s
    y = np.matmul(drop_res.astype(fp32), v.astype(fp32))  # b,n,s,d
    y = y.transpose(0, 2, 1, 3).reshape(BS, ND)  # bs,nd
    return y


def llama_mlp(x, ffn_weight):
    # gate_proj
    t_gate = np.matmul(x.astype(fp32), ffn_weight.astype(fp32))

    # swish: x / (1 + e^(-x))
    t_swish = t_gate * (-1.0)
    t_swish = np.exp(t_swish)
    t_swish = t_swish + 1.0
    t_swish = t_gate / t_swish

    # up_proj
    t_up = np.matmul(x.astype(fp32), ffn_weight.astype(fp32))
    t_swish = t_swish * t_up

    # down_proj
    res = np.matmul(t_swish.astype(fp32), ffn_weight.transpose(1, 0).astype(fp32))

    return res


def llama_layer(x, attn_weight, dense_weight, ffn_weight):
    residual = x

    # pre-norm
    hidden_states = rms_norm(x)
    # hiddenStates = x

    # self attention
    attention_out = multi_attention(hidden_states, attn_weight)  # bs,nd
    attention_out_fp16 = attention_out.astype(fp16)

    # dense
    dense_out = np.matmul(attention_out_fp16.astype(fp32), dense_weight.astype(fp32))
    hidden_states = residual + dense_out

    # fully connect
    residual = hidden_states
    hidden_states = rms_norm(hidden_states)

    mlp_res = llama_mlp(hidden_states, ffn_weight)
    hidden_states = residual + mlp_res

    return hidden_states


@GoldenRegister.reg_golden_func(
    case_names=[
        "LLamaLayerTest.llama_1_1_128_128",
        "LLamaLayerTest.llama_1_1_256_128",
        "LLamaLayerTest.llama_1_1_512_128",
        "LLamaLayerTest.llama_1_1_1024_128",
        "LLamaLayerTest.llama_1_1_4096_128",
    ]
)
def gen_llama_layer_golden(case_name: str, output: Path) -> bool:
    case_name_2_shape = {
        "LLamaLayerTest.llama_1_1_128_128": [1, 1, 128, 128],
        "LLamaLayerTest.llama_1_1_256_128": [1, 1, 256, 128],
        "LLamaLayerTest.llama_1_1_512_128": [1, 1, 512, 128],
        "LLamaLayerTest.llama_1_1_1024_128": [1, 1, 1024, 128],
        "LLamaLayerTest.llama_1_1_4096_128": [1, 1, 4096, 128]
    }
    value = case_name_2_shape.get(case_name)
    if value is None:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    llama_layer_main(case_name, value, output)
    return True


def llama_layer_main(case_name: str, shape, output: Path):
    global B, N, S, D
    B, N, S, D = shape
    global BSN, ND, BS
    BSN = B * N * S
    ND = N * D
    BS = B * S
    logging.debug("Case(%s), shape is %d %d %d %d BSN %d ND %d BS %d.", case_name, B, N, S, D, BSN, ND, BS)
    hidden_states_path = Path(output, 'hiddenStates.bin')
    atten_weight_path = Path(output, 'attnWeight.bin')
    dense_weight_path = Path(output, 'denseWeight.bin')
    ffn_weight_path = Path(output, 'ffnWeight.bin')
    res_path = Path(output, 'llama_layer_golden_res.bin')
    if (hidden_states_path.exists() and atten_weight_path.exists() and dense_weight_path.exists() and
            ffn_weight_path.exists() and res_path.exists()):
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        hidden_states = np.random.uniform(-1, 1, (B * S, N * D)).astype(fp32)
        attn_weight = np.random.uniform(-0.03, 0.03, (N * D, N * D * 3)).astype(fp16)
        dense_weight = np.random.uniform(-0.03, 0.03, (N * D, N * D)).astype(fp16)
        ffn_weight = np.random.uniform(-0.03, 0.03, (N * D, N * D * 3)).astype(fp16)

        # hiddenStates = np.ones((B*S, N*D)).astype(fp32)
        # attn_weight = np.ones((N*D, N*D*3)).astype(fp16)
        # dense_weight =  np.ones((N*D, N*D)).astype(fp16)
        # ffn_weight =  np.ones((N*D, N*D*3)).astype(fp16)

        # write golden
        hidden_states.tofile(hidden_states_path)
        attn_weight.tofile(atten_weight_path)
        dense_weight.tofile(dense_weight_path)
        ffn_weight.tofile(ffn_weight_path)
        # call layer function
        res = llama_layer(hidden_states, attn_weight, dense_weight, ffn_weight)
        res.tofile(res_path)
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "QuantMMOnBoardTest.test_QuantMM_32_16384_times_16384_7168_np",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_llama_layer_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
