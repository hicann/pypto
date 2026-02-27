#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import math
import logging
import torch
import torch_npu
import pytest
import pypto
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

from lightning_indexer_prolog_quant_mxfp8_impl import (
    IndexerPrologQuantInput, IndexerPrologQuantOutput, IndexerPrologQuantAttr, IndexerPrologQuantConfigs,
    lightning_indexer_prolog_quant)
from utils.compare import precision_compare_triple


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("lightning_indexer_prolog_quant_mxfp8(Tensor x, Tensor q_norm, Tensor q_norm_scale, \
    Tensor w_qb, Tensor w_qb_scale, Tensor wk, Tensor w_proj, Tensor ln_gamma_k, \
    Tensor ln_beta_k, Tensor cos_idx_rope, Tensor sin_idx_rope, Tensor hadamard_q, \
    Tensor hadamard_k, Tensor k_cache, Tensor k_cache_scale, Tensor k_cache_index) -> \
    (Tensor q_fp8e4m3, Tensor q_scale, Tensor k_fp8e4m3, Tensor k_scale, Tensor weights)")


def gen_dims(params):
    dims = {}
    dims["s2"] = params["s2"]
    dims["b"] = params["b"]
    dims["t"] = params["b"] * params["s1"]
    dims["h"] = 2560
    dims["q_lora_rank"] = 1024
    dims["idx_head_dim"] = 128
    dims["idx_n_heads"] = 24
    dims["rope_head_dim"] = 64
    dims["block_size"] = 128
    dims["block_num"] = dims["b"] * dims["s2"] // dims["block_size"]
    dims["n_kv"] = 1
    return dims


def gen_block_table(act_seq, block_size, s1):
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

    return block_num, block_table, cache_index


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


def gen_inputs(dims, dtype=torch.bfloat16, quant_dtype=torch.float8_e4m3fn):
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    block_size = dims["block_size"]
    n_kv = dims["n_kv"]
    s2 = dims["s2"]
    rope_head_dim = dims["rope_head_dim"]

    x = torch.empty((b, s, h), dtype=dtype).uniform_(-1, 1)
    q_norm = torch.empty((b, s, q_lora_rank), dtype=torch.float32).uniform_(-10, 10).to(quant_dtype)
    q_norm_scale = torch.empty((b, s, q_lora_rank // 32), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    w_idx_qb = torch.empty((q_lora_rank, n * d), dtype=torch.float32).uniform_(-10, 10).to(quant_dtype)
    w_idx_qb_scale = torch.empty((n * d, q_lora_rank // 32), dtype=torch.float32) \
        .uniform_(0, 1).to(torch.float8_e8m0fnu)
    w_idx_k = torch.empty((h, d), dtype=dtype).uniform_(-1, 1)
    w_idx_proj = torch.empty((h, n), dtype=dtype).uniform_(-1, 1)
    ln_gamma = torch.ones((d,), dtype=dtype)
    ln_beta = torch.zeros((d,), dtype=dtype)

    random_angles = (torch.rand(b, s, rope_head_dim, dtype=torch.float32) * 2 * torch.pi)
    cos = torch.cos(random_angles).to(dtype)
    sin = torch.sin(random_angles).to(dtype)

    hadamard_q = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)  # (128, 128)
    hadamard_k = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)

    act_seq = torch.tensor([s2] * b)  # (b,)
    k_cache_bsnd = torch.empty(size=(b, s2, n_kv, d), dtype=torch.float32).uniform_(-10, 10).to(quant_dtype)
    k_scale_cache_bsnd = torch.empty((b, s2, n_kv, 1), dtype=torch.float32).uniform_(-1, 1)
    block_num, block_table, k_cache_index = gen_block_table(act_seq, block_size, s)
    k_cache = gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size)
    k_scale_cache = gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num, block_size)

    return {
        "token_x": x,  # input0, bf16
        "q_norm": q_norm,  # input1, mxfp8e4m3
        "q_norm_scale": q_norm_scale,  # input2, e8m0
        "w_idx_qb": w_idx_qb,  # input3, mxfp8e4m3
        "w_idx_qb_scale": w_idx_qb_scale,  # input4, e8m0
        "w_idx_k": w_idx_k,  # input5, bf16
        "w_idx_proj": w_idx_proj,  # input6, bf16
        "layer_norm_gamma": ln_gamma,  # input7, bf16
        "layer_norm_beta": ln_beta,  # input8, bf16
        "cos_idx_rope": cos,  # input9, bf16
        "sin_idx_rope": sin,  # input10, bf16
        "hadamard_q": hadamard_q,  # input11, bf16
        "hadamard_k": hadamard_k,  # input12, bf16
        "idx_k_cache": k_cache,  # input13, fp8e4m3  # (block_num, block_size, n_kv, d)
        "idx_k_scale_cache": k_scale_cache,  # input14, fp32  # (block_num, block_size, n_kv, 1)
        "idx_k_cache_index": k_cache_index,  # input15, int64  (b, s)/（t,)
        "idx_block_table": block_table,  # input16, int32  (b, ceil(s2, block_size))
        "act_seq": act_seq,  # input17, int32
    }


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


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x_dtype = x.dtype
    if x_dtype != torch.float32:
        x = x.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return (x * gamma.to(torch.float32) + beta.to(torch.float32)).to(x_dtype)


def quant_fp8e4m3(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def single_rope(x, cos_in, sin_in):
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


def indexer_prolog(inputs_initial: dict, dims: dict, platform: str = "cpu"):
    # input
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b

    if platform == "cpu":
        inputs = {k: v.to(torch.float32) if v.dtype in [torch.bfloat16, torch.float16] else v.clone() 
                    for k, v in inputs_initial.items()}
    elif platform == "npu":
        inputs = {k: v.clone().npu() for k, v in inputs_initial.items()}

    rope_head_dim = dims["rope_head_dim"]
    x = inputs["token_x"]  # (b, s, h)
    q_norm = inputs["q_norm"]  # (b, s, q_lora_rank), mxfp8e4m3
    q_norm_scale = inputs["q_norm_scale"]  # (b, s, q_lora_rank//32), e8m0
    w_idx_qb = inputs["w_idx_qb"]  # (q_lora_rank, n * d), mxfp8e4m3
    w_idx_qb_scale = inputs["w_idx_qb_scale"]  # (q_lora_rank//32, n * d), e8m0
    w_idx_k = inputs["w_idx_k"]  # (h, d)
    w_idx_proj = inputs["w_idx_proj"]  # (h, n)
    layer_norm_gamma = inputs["layer_norm_gamma"]  # (d,)
    layer_norm_beta = inputs["layer_norm_beta"]  # (d,)
    cos = inputs["cos_idx_rope"]  # (b, s, rope_head_dim)
    sin = inputs["sin_idx_rope"]  # (b, s, rope_head_dim)
    hadamard_q = inputs["hadamard_q"]  # (d, d)
    hadamard_k = inputs["hadamard_k"]  # (d, d)
    idx_k_cache = inputs["idx_k_cache"]  # input13, fp8e4m3
    idx_k_scale_cache = inputs["idx_k_scale_cache"]  # input14, fp16
    cache_index = inputs["idx_k_cache_index"]  # (b, s), int32
    x_dtype = x.dtype

    # calculate
    q_norm_scale_expand = q_norm_scale.cpu().repeat_interleave(32, dim=2)
    w_idx_qb_scale_expand = w_idx_qb_scale.cpu().transpose(0, 1).repeat_interleave(32, dim=0) # TO DO Trans

    q_norm_fp32 = q_norm.to(torch.float32)
    q_norm_scale_expand_fp32 = q_norm_scale_expand.to(torch.float32)
    w_idx_qb_fp32 = w_idx_qb.to(torch.float32)
    w_idx_qb_scale_expand_fp32 = w_idx_qb_scale_expand.to(torch.float32)
    if platform == "npu":
        q_norm_scale_expand_fp32 = q_norm_scale_expand_fp32.npu()
        w_idx_qb_scale_expand_fp32 = w_idx_qb_scale_expand_fp32.npu()

    q_matmul_mx = torch.matmul(q_norm_fp32 * q_norm_scale_expand_fp32, w_idx_qb_fp32 * w_idx_qb_scale_expand_fp32)

    q_bf16 = q_matmul_mx.reshape(b, s, n, d).to(x_dtype)
    q_rope, q_nope = torch.split(q_bf16, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_rope = single_rope(q_rope, cos, sin)
    q_cat = torch.cat([q_rope, q_nope], dim=-1)
    # hadamard
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    q_hadamard = torch.matmul(q_cat.to(torch.float32), hadamard_q.to(torch.float32)).to(x_dtype)  # (b, s, n, d)
    q_fp8e4m3, q_scale = quant_fp8e4m3(q_hadamard)  # (b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    q_scale = q_scale.to(torch.float32)

    k_matmul = torch.matmul(x.to(torch.float32), w_idx_k.to(torch.float32))  # (b, s, d)
    k = layer_norm(k_matmul, layer_norm_gamma, layer_norm_beta).to(x_dtype)
    k_rope, k_nope = torch.split(k, [rope_head_dim, d - rope_head_dim], dim=-1)
    k_rope = single_rope(k_rope.unsqueeze(2), cos, sin).squeeze(2)
    k = torch.cat([k_rope, k_nope], dim=-1)
    # hadamard
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    k_hadamard = torch.matmul(k.to(torch.float32), hadamard_k.to(torch.float32)).to(x_dtype)  # (b, s, d)
    k_fp8e4m3, k_scale = quant_fp8e4m3(k_hadamard)  # (b, s, d) fp8e4m3, (b, s, 1) fp32
    k_scale = k_scale.to(torch.float32)
    # cache update
    k_cache = idx_k_cache.clone()  # (block_num, block_size, n_kv, d)
    k_scale_cache = idx_k_scale_cache.clone()  # (block_num, block_size, n_kv, 1)
    scatter_update_pa_bsnd(k_cache, k_fp8e4m3.reshape(b, s, 1, d), cache_index, -2)
    scatter_update_pa_bsnd(k_scale_cache, k_scale.reshape(b, s, 1, 1), cache_index, -2)

    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    weights = torch.matmul(x.to(torch.float32), \
        w_idx_proj.to(torch.float32)).to(x_dtype).to(torch.float32)  # (b, s, n)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    weights = weights.to(x_dtype)

    # output dtype: fp8e4m3, fp32, fp8e4m3, fp32, bf16
    outputs = {"query": q_fp8e4m3, "query_scale": q_scale,
               "idx_k_cache_out": k_cache, "idx_k_scale_cache_out": k_scale_cache,
               "weights": weights}
    return outputs


def gen_data(case_name):
    if case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_8k_s2_8k"):
        params = {"b": 1, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b2_s1_8k_s2_8k"):
        params = {"b": 2, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_8k_s2_8k"):
        params = {"b": 4, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_4_s2_8k"):
        params = {"b": 1, "s1": 4, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b64_s1_2_s2_8k"):
        params = {"b": 64, "s1": 2, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b192_s1_1_s2_8k"):
        params = {"b": 192, "s1": 1, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b16_s1_8k_s2_8k"):
        params = {"b": 16, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b32_s1_8k_s2_8k"):
        params = {"b": 32, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b64_s1_8k_s2_8k"):
        params = {"b": 64, "s1": 1024 * 8, "s2": 1024 * 8}
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 0
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.bfloat16)
    outputs_cpu = indexer_prolog(inputs, dims, "cpu")
    outputs_npu = indexer_prolog(inputs, dims, "npu")

    return dims, inputs, outputs_cpu, outputs_npu


def gen_zero_tensor(t):
    return torch.zeros_like(t).npu()


def ascend_operator_accuracy_standard_version_2_1(pypto_out, npu_out, golden_out, out_name):
    result, mare, mere, rmse, small_err = precision_compare_triple(pypto_out, npu_out, golden_out)
    assert result != "FAILED", f"{out_name} FAILED: result={result}, \
        mare={mare:.6f}, mere={mere:.6f}, rmse={rmse:.6f}, small_errors={small_err}"
    logging.info(f"{out_name} PASSED - \
        mare={mare:.6f}, mere={mere:.6f}, rmse={rmse:.6f}, small_errors={small_err}")


@torch.library.impl(pyptolib, "lightning_indexer_prolog_quant_mxfp8", "Meta")
def lightning_indexer_prolog_quant_mxfp8_meta(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj,
                                           ln_gamma_k, ln_beta_k, cos_idx_rope, sin_idx_rope, hadamard_q,
                                           hadamard_k, k_cache, k_cache_scale, k_cache_index):
    t = x.shape[0]
    head_num = w_proj.shape[1]
    block_num, block_size, n_kv, head_dim = k_cache.shape
    q_fp8e4m3 = torch.empty((t, head_num, head_dim), device="meta", dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((t, head_num, 1), device="meta", dtype=torch.float32)
    k_fp8e4m3 = torch.empty((block_num, block_size, n_kv, head_dim), device="meta", dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((block_num, block_size, n_kv, 1), device="meta", dtype=torch.float32)
    weights = torch.empty((t, head_num), device="meta", dtype=torch.bfloat16)

    return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights


@torch.library.impl(pyptolib, "lightning_indexer_prolog_quant_mxfp8", "NPU")
@allow_in_graph
def lightning_indexer_prolog_quant_mxfp8_npu(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj,
                                           ln_gamma_k, ln_beta_k, cos_idx_rope, sin_idx_rope, hadamard_q,
                                           hadamard_k, k_cache, k_cache_scale, k_cache_index):
    t = x.shape[0]
    head_num = w_proj.shape[1]
    block_num, block_size, n_kv, head_dim = k_cache.shape

    device = x.device
    q_fp8e4m3 = torch.empty((t, head_num, head_dim), device=device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((t, head_num, 1), device=device, dtype=torch.float32)
    k_fp8e4m3 = torch.empty((block_num, block_size, n_kv, head_dim), device=device, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((block_num, block_size, n_kv, 1), device=device, dtype=torch.float32)
    weights = torch.empty((t, head_num), device=device, dtype=torch.bfloat16)

    if isinstance(x, FakeTensor):
        return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights

    attrs = IndexerPrologQuantAttr(
        eps=1e-6,
        layerout_query="TND",
        layerout_key="PA_BSND",
    )
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )

    input_tensors = {
        x: [0],
        q_norm: [0],
        q_norm_scale: [0],
        w_qb: [],
        w_qb_scale: [],
        wk: [],
        w_proj: [],
        ln_gamma_k: [],
        ln_beta_k: [],
        cos_idx_rope: [0],
        sin_idx_rope: [0],
        hadamard_q: [],
        hadamard_k: [],
        k_cache: [0],
        k_cache_scale: [0],
        k_cache_index: [0],
    }
    output_tensors = {
        q_fp8e4m3: [0],
        q_scale: [0],
        k_fp8e4m3: [0],
        k_scale: [0],
        weights: [0],
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
    lightning_indexer_prolog_quant(*pto_inputs, *pto_outputs, attrs, configs)

    return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights


def pypto_lightning_indexer_prolog_quant_mxfp8(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj,
                                        ln_gamma_k, ln_beta_k, cos_idx_rope, sin_idx_rope, hadamard_q,
                                        hadamard_k, k_cache, k_cache_scale, k_cache_index):
    
    return torch.ops.pypto.lightning_indexer_prolog_quant_mxfp8(x, q_norm, q_norm_scale, w_qb, w_qb_scale,
        wk, w_proj, ln_gamma_k, ln_beta_k, cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k, k_cache,
        k_cache_scale, k_cache_index)


def lightning_indexer_prolog_quant_dyn(inputs: IndexerPrologQuantInput, outputs: IndexerPrologQuantOutput,
                                      attrs: IndexerPrologQuantAttr, configs: IndexerPrologQuantConfigs):
    input_tensors = {
        inputs.x: [0],
        inputs.q_norm: [0],
        inputs.q_norm_scale: [0],
        inputs.w_qb: [],
        inputs.w_qb_scale: [],
        inputs.wk: [],
        inputs.w_proj: [],
        inputs.ln_gamma_k: [],
        inputs.ln_beta_k: [],
        inputs.cos_idx_rope: [0],
        inputs.sin_idx_rope: [0],
        inputs.hadamard_q: [],
        inputs.hadamard_k: [],
        inputs.k_cache: [0],
        inputs.k_cache_scale: [0],
        inputs.k_cache_index: [0],
    }
    output_tensors = {
        outputs.q_fp8e4m3: [0],
        outputs.q_scale: [0],
        outputs.k_fp8e4m3: [0],
        outputs.k_scale: [0],
        outputs.weights: [0],
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
    lightning_indexer_prolog_quant(*pto_inputs, *pto_outputs, attrs, configs)
    torch_npu.npu.synchronize()


def do_test_lightning_indexer_prolog_quant(case_name, configs):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch_npu.npu.set_device(device_id)

    logging.info(f"=== run test case: {case_name} ===")

    dims, inputs_data, golden_data, npu_data = gen_data(case_name)

    t = dims["t"]
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    idx_head_dim = dims["idx_head_dim"]
    head_num = dims["idx_n_heads"]
    rope_head_dim = dims["rope_head_dim"]

    torch_npu.npu.config.allow_internal_format = True

    inputs = IndexerPrologQuantInput(
        x=inputs_data["token_x"].npu().reshape(t, h),
        q_norm=inputs_data["q_norm"].npu().reshape(t, q_lora_rank),
        q_norm_scale=inputs_data["q_norm_scale"].npu().reshape(t, q_lora_rank // 64, 2),
        w_qb=inputs_data["w_idx_qb"].npu(),  # TO DO ND->NZ
        w_qb_scale=inputs_data["w_idx_qb_scale"].npu() \
            .reshape(head_num * idx_head_dim, q_lora_rank // 64, 2),  # TO DO ND->DN
        wk=inputs_data["w_idx_k"].npu(),  # TO DO ND->NZ
        w_proj=inputs_data["w_idx_proj"].npu(),  # TO DO ND->NZ
        ln_gamma_k=inputs_data["layer_norm_gamma"].npu(),
        ln_beta_k=inputs_data["layer_norm_beta"].npu(),
        cos_idx_rope=inputs_data["cos_idx_rope"].npu().reshape(t, rope_head_dim),
        sin_idx_rope=inputs_data["sin_idx_rope"].npu().reshape(t, rope_head_dim),
        hadamard_q=inputs_data["hadamard_q"].npu(),
        hadamard_k=inputs_data["hadamard_k"].npu(),
        k_cache=inputs_data["idx_k_cache"].npu(),
        k_cache_scale=inputs_data["idx_k_scale_cache"].npu(),
        k_cache_index=inputs_data["idx_k_cache_index"].npu().reshape(t)
    )

    q_fp8e4m3_golden = golden_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_golden = golden_data["query_scale"].reshape(t, head_num, 1)
    k_cache_golden = golden_data["idx_k_cache_out"]
    k_cache_scale_golden = golden_data["idx_k_scale_cache_out"]
    weights_golden = golden_data["weights"].reshape(t, head_num)

    q_fp8e4m3_npu = npu_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_npu = npu_data["query_scale"].reshape(t, head_num, 1)
    k_cache_npu = npu_data["idx_k_cache_out"]
    k_cache_scale_npu = npu_data["idx_k_scale_cache_out"]
    weights_npu = npu_data["weights"].reshape(t, head_num)

    outputs = IndexerPrologQuantOutput(
        q_fp8e4m3=gen_zero_tensor(q_fp8e4m3_npu),
        q_scale=gen_zero_tensor(q_scale_npu),
        k_fp8e4m3=inputs.k_cache,
        k_scale=inputs.k_cache_scale,
        weights=gen_zero_tensor(weights_npu),
    )

    # ---- Attrs ----
    attrs = IndexerPrologQuantAttr(
        eps=1e-6,
        layerout_query="TND",
        layerout_key="PA_BSND",
    )

    logging.info("==================finish torch==================")
    lightning_indexer_prolog_quant_dyn(inputs, outputs, attrs, configs)

    logging.info("==================finish pypto==================")
    ascend_operator_accuracy_standard_version_2_1(outputs.q_fp8e4m3, q_fp8e4m3_npu, q_fp8e4m3_golden, "q_fp8e4m3")
    ascend_operator_accuracy_standard_version_2_1(outputs.q_scale, q_scale_npu, q_scale_golden, "q_scale")
    ascend_operator_accuracy_standard_version_2_1(outputs.k_fp8e4m3, k_cache_npu, k_cache_golden, "k_fp8e4m3")
    ascend_operator_accuracy_standard_version_2_1(outputs.k_scale, k_cache_scale_npu, k_cache_scale_golden, "k_scale")
    ascend_operator_accuracy_standard_version_2_1(outputs.weights, weights_npu, weights_golden, "weights")

    logging.info(f"=== {case_name}: PASS ===")


def test_b1_s1_8k_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_8k_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b2_s1_8k_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b2_s1_8k_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b4_s1_8k_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_8k_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b1_s1_4_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_4_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b64_s1_2_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b64_s1_2_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b192_s1_1_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b192_s1_1_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b16_s1_8k_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b16_s1_8k_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b32_s1_8k_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b32_s1_8k_s2_8k", configs)


@pytest.mark.skip(reason="large test case")
def test_b64_s1_8k_s2_8k():
    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b64_s1_8k_s2_8k", configs)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, *args):
        x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj, ln_gamma_k, ln_beta_k, cos_idx_rope, \
        sin_idx_rope, hadamard_q, hadamard_k, k_cache, k_cache_scale, k_cache_index = args
        q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights = torch.ops.pypto.lightning_indexer_prolog_quant_mxfp8(
            x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj, ln_gamma_k, ln_beta_k, cos_idx_rope,
            sin_idx_rope, hadamard_q, hadamard_k, k_cache, k_cache_scale, k_cache_index)
        return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights


def test_acl():
    params = {"b": 1, "s1": 1024 * 8, "s2": 1024 * 8}
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.bfloat16)
    golden_data = indexer_prolog(inputs, dims, "cpu")
    npu_data = indexer_prolog(inputs, dims, "npu")
    logging.info("==================finish torch==================")

    t = dims["t"]
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    idx_head_dim = dims["idx_head_dim"]
    head_num = dims["idx_n_heads"]
    rope_head_dim = dims["rope_head_dim"]

    q_fp8e4m3_golden = golden_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_golden = golden_data["query_scale"].reshape(t, head_num, 1)
    k_cache_golden = golden_data["idx_k_cache_out"]
    k_cache_scale_golden = golden_data["idx_k_scale_cache_out"]
    weights_golden = golden_data["weights"].reshape(t, head_num)

    q_fp8e4m3_npu = npu_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_npu = npu_data["query_scale"].reshape(t, head_num, 1)
    k_cache_npu = npu_data["idx_k_cache_out"]
    k_cache_scale_npu = npu_data["idx_k_scale_cache_out"]
    weights_npu = npu_data["weights"].reshape(t, head_num)

    model = Model()
    compile_forward = torch.compile(model, fullgraph=True, backend="npugraph_ex", dynamic=False)
    q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights = compile_forward(inputs["token_x"].npu().reshape(t, h),
        inputs["q_norm"].npu().reshape(t, q_lora_rank), inputs["q_norm_scale"].npu().reshape(t, q_lora_rank // 64, 2),
        inputs["w_idx_qb"].npu(), inputs["w_idx_qb_scale"].npu().reshape(head_num * idx_head_dim, q_lora_rank // 64, 2),
        inputs["w_idx_k"].npu(), inputs["w_idx_proj"].npu(), inputs["layer_norm_gamma"].npu(),
        inputs["layer_norm_beta"].npu(), inputs["cos_idx_rope"].npu().reshape(t, rope_head_dim),
        inputs["sin_idx_rope"].npu().reshape(t, rope_head_dim), inputs["hadamard_q"].npu(), inputs["hadamard_k"].npu(),
        inputs["idx_k_cache"].npu(), inputs["idx_k_scale_cache"].npu(), inputs["idx_k_cache_index"].npu().reshape(t))

    logging.info("==================finish pypto==================")
    ascend_operator_accuracy_standard_version_2_1(q_fp8e4m3, q_fp8e4m3_npu, q_fp8e4m3_golden, "q_fp8e4m3")
    ascend_operator_accuracy_standard_version_2_1(q_scale, q_scale_npu, q_scale_golden, "q_scale")
    ascend_operator_accuracy_standard_version_2_1(k_fp8e4m3, k_cache_npu, k_cache_golden, "k_fp8e4m3")
    ascend_operator_accuracy_standard_version_2_1(k_scale, k_cache_scale_npu, k_cache_scale_golden, "k_scale")
    ascend_operator_accuracy_standard_version_2_1(weights, weights_npu, weights_golden, "weights")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b1_s1_8k_s2_8k()