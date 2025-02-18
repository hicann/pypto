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
import math
import sys
import logging
from pathlib import Path
from typing import List

import torch

if __name__ == "__main__":
    """单独调试时配置"""
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "/tests/st/operator/src/test_deepseek_v3.2_exp")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    sys.path.append(str(g_src_root))
    from framework.tests.cmake.scripts.golden_register import (
        GoldenRegister,
    )  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def tensor_tofile(t: torch.Tensor, output: Path, dtype: torch.dtype):
    with open(str(output), "wb") as f:
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).numpy().tobytes())


def inputs_tofile(inputs: dict, output: Path):
    for name, tensor in inputs.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_tofile(tensor, Path(output, f"{name}.bin"), tensor.dtype)


def golden_tofile(golden: dict, output: Path):
    for name, tensor in golden.items():
        tensor_tofile(tensor, Path(output, f"{name}_golden.bin"), tensor.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def single_rope(x, cos_in, sin_in):
    logging.info("Entering into single_rope")
    # x: (b, s, n, d), cos_in: (b, s, d), sin_in: (b, s, d)
    x_dtype = x.dtype
    b, s, n, d = x.shape
    x_cast = x.to(torch.float32)
    cos_cast = cos_in.to(torch.float32)
    sin_cast = sin_in.to(torch.float32)
    cos_re = cos_cast.unsqueeze(2)  # (b, s, 1, d)
    sin_re = sin_cast.unsqueeze(2)  # (b, s, 1, d)
    res = x_cast * cos_re + rotate_half(x_cast) * sin_re  # (b, s, n, d)
    return res.to(x_dtype)


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x_dtype = x.dtype
    if x_dtype != torch.float32:
        x = x.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return (x * gamma.to(torch.float32) + beta.to(torch.float32)).to(x_dtype)


def quant_int8(x: torch.Tensor):
    # pertoken
    x_dtype = x.dtype  # bf16, (b, s, n, d)
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_int32 = torch.round(y_fp32).to(torch.int32)  # rint mode
    y_int8 = torch.trunc(y_int32.to(x_dtype)).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) int8, (b, s, n, 1) fp32
    return y_int8, scale_dequant


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

    block_idx = 0
    block_table_bidx = 0
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

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


def gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size):
    dtype = k_cache_bsnd.dtype
    b, s2, n_kv, d = k_cache_bsnd.shape
    k_cache = torch.zeros((block_num, block_size, n_kv, d), dtype=dtype)
    s2_new = ((s2 + block_size - 1) // block_size) * block_size  # ceil to block_size
    k_cache_raw = torch.zeros((b, s2_new, n_kv, d), dtype=dtype)
    k_cache_raw[:, :s2, :, :] = k_cache_bsnd

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx == -1:
                continue
            else:
                k_cache[cache_block_idx, :, :, :] = k_cache_raw[
                    b_idx, block_offset: (block_offset + block_size), :, :
                ]

    return k_cache


def scatter_update_pa_bsnd(cache, k_bsnd, cache_index, axis):
    block_number, block_size, n_kv, d = cache.shape
    res = cache.reshape(block_number * block_size * n_kv, d)
    b, s1 = cache_index.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = cache_index[b_i][s1_i]
                res[index_value, :] = k_bsnd[b_i, s1_i, :, :]

    return res.reshape(block_number, block_size, n_kv, d)


def indexer_prolog(inputs: dict, dims: dict):
    # input
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b

    rope_head_dim = dims["rope_head_dim"]
    x = inputs["token_x"]  # (b, s, h)
    q_norm = inputs["q_norm"]  # (b, s, q_lora_rank), int8
    q_norm_scale = inputs["q_norm_scale"]  # (b, s, 1), fp32
    w_idx_qb = inputs["w_idx_qb"]  # (q_lora_rank, n * d), int8
    w_idx_qb_scale = inputs["w_idx_qb_scale"]  # (n * d, 1), fp32
    w_idx_k = inputs["w_idx_k"]  # (h, d)
    w_idx_proj = inputs["w_idx_proj"]  # (h, n)
    layer_norm_gamma = inputs["layer_norm_gamma"]  # (d,)
    layer_norm_beta = inputs["layer_norm_beta"]  # (d,)
    cos = inputs["cos_idx_rope"]  # (b, s, rope_head_dim)
    sin = inputs["sin_idx_rope"]  # (b, s, rope_head_dim)
    hadamard_q = inputs["hadamard_q"]  # (d, d)
    hadamard_k = inputs["hadamard_k"]  # (d, d)
    idx_k_cache = inputs["idx_k_cache"]  # input13, int8
    idx_k_scale_cache = inputs["idx_k_scale_cache"]  # input14, fp16
    cache_index = inputs["idx_k_cache_index"]  # (b, s), int32
    x_dtype = x.dtype

    # calculate
    q = torch.matmul(q_norm.to(torch.int32), w_idx_qb.to(torch.int32))  # (b, s, n * d)
    q_fp32 = q.to(torch.float32)
    q_fp32 = q_fp32 * q_norm_scale
    q_fp32 = q_fp32 * w_idx_qb_scale.reshape(1, n * d)
    q_bf16 = q_fp32.reshape(b, s, n, d).to(torch.bfloat16)
    q_rope, q_nope = torch.split(q_bf16, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_rope = single_rope(q_rope, cos, sin)
    q = torch.cat([q_rope, q_nope], dim=-1)
    # hadamard
    q = torch.matmul(q, hadamard_q)  # (b, s, n, d)
    q_int8, q_scale = quant_int8(q)  # (b, s, n, d) int8, (b, s, n, 1) fp32
    q_scale = q_scale.to(torch.float16)

    k = torch.matmul(x.to(torch.float32), w_idx_k.to(torch.float32))  # (b, s, d)
    k = layer_norm(k, layer_norm_gamma, layer_norm_beta).to(x_dtype)
    k_rope, k_nope = torch.split(k, [rope_head_dim, d - rope_head_dim], dim=-1)
    k_rope = single_rope(k_rope.unsqueeze(2), cos, sin).squeeze(2)
    k = torch.cat([k_rope, k_nope], dim=-1)
    # hadamard
    k = torch.matmul(k.to(torch.float32), hadamard_k.to(torch.float32)).to(x_dtype)  # (b, s, d)
    k_int8, k_scale = quant_int8(k)  # (b, s, d) int8, (b, s, 1) fp32
    k_scale = k_scale.to(torch.float16)
    # cache update
    k_cache = idx_k_cache.clone()  # (block_num, block_size, n_kv, d)
    k_scale_cache = idx_k_scale_cache.clone()  # (block_num, block_size, n_kv, 1)
    scatter_update_pa_bsnd(k_cache, k_int8.reshape(b, s, 1, d), cache_index, -2)
    scatter_update_pa_bsnd(k_scale_cache, k_scale.reshape(b, s, 1, 1), cache_index, -2)

    weights = torch.matmul(x, w_idx_proj).to(torch.float32)  # (b, s, n)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    weights = weights.to(torch.float16)

    # output dtype: int8, fp16, int8, fp16, fp16
    outputs = {"query": q_int8, "query_scale": q_scale,
               "idx_k_cache_out": k_cache, "idx_k_scale_cache_out": k_scale_cache,
               "weights": weights}
    return outputs


def gen_dims(params):
    dims = {}
    dims["s2"] = params["s2"]
    dims["b"] = params["b"]
    dims["t"] = params["b"] * params["s1"]
    dims["h"] = 7168
    dims["q_lora_rank"] = 1536
    dims["idx_head_dim"] = 128
    dims["idx_n_heads"] = 64
    dims["rope_head_dim"] = 64
    dims["block_size"] = 128
    dims["block_num"] = dims["b"] * dims["s2"] // dims["block_size"]
    dims["n_kv"] = 1
    return dims


def gen_indexer_prolog_inputs(dims, dtype=torch.bfloat16, qunat_dtype=torch.int8, eps=1e-6):
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    block_num = dims["block_num"]
    block_size = dims["block_size"]
    n_kv = dims["n_kv"]
    s2 = dims["s2"]
    rope_head_dim = dims["rope_head_dim"]

    x = torch.empty((b, s, h), dtype=dtype).uniform_(-1, 1)
    q_norm = torch.randint(low=-128, high=128, size=(b, s, q_lora_rank), dtype=qunat_dtype)
    q_norm_scale = torch.empty((b, s, 1), dtype=torch.float32).uniform_(-1, 1)
    w_idx_qb = torch.randint(low=-128, high=128, size=(q_lora_rank, n * d), dtype=qunat_dtype)
    w_idx_qb_nz = w_idx_qb.reshape(q_lora_rank // 16, 16, n * d // 32, 32).permute(2, 0, 1, 3)  # int8, C0=32
    w_idx_qb_scale = torch.empty((n * d, 1), dtype=torch.float32).uniform_(-1, 1)

    w_idx_k = torch.empty((h, d), dtype=dtype).uniform_(-1, 1)
    w_idx_k_nz = w_idx_k.reshape(h // 16, 16, d // 16, 16).permute(2, 0, 1, 3)

    w_idx_proj = torch.empty((h, n), dtype=dtype).uniform_(-1, 1)
    w_idx_proj_nz = w_idx_proj.reshape(h // 16, 16, n // 16, 16).permute(2, 0, 1, 3)

    ln_gamma = torch.ones((d,), dtype=dtype)
    ln_beta = torch.zeros((d,), dtype=dtype)

    random_angles = (torch.rand(b, s, rope_head_dim, dtype=torch.float32) * 2 * torch.pi)
    cos = torch.cos(random_angles).to(dtype)
    sin = torch.sin(random_angles).to(dtype)

    hadamard_q = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)  # (128, 128)
    hadamard_k = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)

    act_seq = torch.tensor([s2] * b)  # (b,)
    k_cache_bsnd = torch.randint(low=-128, high=128, size=(b, s2, n_kv, d), dtype=qunat_dtype)
    k_scale_cache_bsnd = torch.empty((b, s2, n_kv, 1), dtype=torch.float16).uniform_(-1, 1)
    block_num, block_table, k_cache_index = gen_block_table(act_seq, block_size, s, need_indices=True)
    k_cache = gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size)
    k_scale_cache = gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num, block_size)

    return {
        "token_x": x,  # input0, bf16
        "q_norm": q_norm,  # input1, int8
        "q_norm_scale": q_norm_scale,  # input2, fp32
        "w_idx_qb": w_idx_qb,  # input3, int8
        "w_idx_qb_nz": w_idx_qb_nz,
        "w_idx_qb_scale": w_idx_qb_scale,  # input4, fp32
        "w_idx_k": w_idx_k,  # input5, bf16
        "w_idx_k_nz": w_idx_k_nz,
        "w_idx_proj": w_idx_proj,  # input6, bf16
        "weights_proj_nz": w_idx_proj_nz,
        "layer_norm_gamma": ln_gamma,  # input7, bf16
        "layer_norm_beta": ln_beta,  # input8, bf16
        "cos_idx_rope": cos,  # input9, bf16
        "sin_idx_rope": sin,  # input10, bf16
        "hadamard_q": hadamard_q,  # input11, bf16
        "hadamard_k": hadamard_k,  # input12, bf16
        "idx_k_cache": k_cache,  # input13, int8  # (block_num, block_size, n_kv, d)
        "idx_k_scale_cache": k_scale_cache,  # input14, fp16  # (block_num, block_size, n_kv, 1)
        "idx_k_cache_index": k_cache_index,  # input15, int64  (b, s)/（t,)
        "idx_block_table": block_table,  # input16, int32  (b, ceil(s2, block_size))
        "act_seq": act_seq,  # input17, int32
        "layernorm_epsilon_k": eps,  # attr0, fp32
    }


def gen_indexer_golden(params, output):
    seed = 0
    # PyTorch 随机数生成器
    torch.manual_seed(seed)

    dims = gen_dims(params)
    dim_tensor = torch.tensor(list(dims.values()), dtype=torch.int32)
    tensor_tofile(dim_tensor, Path(output, f"input_param.bin"), torch.int32)

    inputs = gen_indexer_prolog_inputs(dims, torch.bfloat16)
    inputs_tofile(inputs, output)
    outputs = indexer_prolog(inputs, dims)
    golden_tofile(outputs, output)


@GoldenRegister.reg_golden_func(
    case_names=[
        "QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k",
        "QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k",
        "QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k",
        "QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k",
        "QuantLightningIndexerPrologSTest.b128_s1_4_s2_8k"
    ]
)
def indexer_test(case_name: str, output: Path) -> bool:
    if case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k"):
        params = {
            "b": 4,
            "s1": 2,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k"):
        params = {
            "b": 8,
            "s1": 2,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k"):
        params = {
            "b": 1,
            "s1": 1024 * 4,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k"):
        params = {
            "b": 2,
            "s1": 1024 * 4,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b128_s1_4_s2_8k"):
        params = {
            "b": 128,
            "s1": 4,
            "s2": 1024 * 8
        }
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    gen_indexer_golden(params, output)
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k",
        "QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k",
        "QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k",
        "QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k"
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/framework/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = indexer_test(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
