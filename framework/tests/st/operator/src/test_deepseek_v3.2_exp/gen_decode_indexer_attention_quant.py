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

import numpy as np
import os
import torch

from pathlib import Path
from typing import List
from ml_dtypes import bfloat16

# 添加 golden 所在目录的父路径（例如项目根目录）
project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)
golden_parent2 = os.path.join(golden_parent, "cmake/scripts")
sys.path.insert(0, golden_parent2)

import gen_mla_prolog_quant_golden_v32
import gen_quant_lightning_indexer_prolog
import gen_lightning_indexer
import gen_gather_selected_attention

torch.manual_seed(42)

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.getLogger('').handlers.clear()

    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../cmake/").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def dump_file_torch(data, data_path):
    """将PyTorch张量保存到文件，支持BFloat16类型转换"""
    dtype = data.dtype
    if dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.int32:
        np_dtype = np.int32
    elif dtype == torch.int64:
        np_dtype = np.int64
    elif dtype == torch.int8:
        np_dtype = np.int8
    elif dtype == torch.bfloat16:
        np_dtype = bfloat16
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")

    if isinstance(data, torch.Tensor):
        # 处理BFloat16类型：转换为float32后再转NumPy（NumPy不支持BFloat16）
        if data.dtype == torch.bfloat16:
            data_np = data.to(torch.float32).numpy()
        else:
            data_np = data.numpy()
    else:
        data_np = np.array(data)

    # 确保最终类型与指定dtype一致
    data_np = data_np.astype(np_dtype)
    data_np.tofile(data_path)


def gen_indexer_prolog_inputs(params, output_dir, block_num, block_table, x, q_norm, q_norm_scale, cos, sin,
                              cache_index, actual_seq):
    dtype = torch.bfloat16
    quant_dtype = torch.int8

    b = params.get("b")
    s1 = params.get("s1")
    s2 = params.get("s2")
    h = params.get("h")
    block_size = params.get("block_size")
    eps = params.get("epsilon")
    q_lora_rank = params.get("q_lora_rank")
    n2 = params.get("n2")
    idx_head_dim = params.get("idx_head_dim")
    idx_n_heads = params.get("idx_n_heads")
    rope_dim = params.get("rope_dim")

    w_idx_qb = torch.randint(low=-128, high=128, size=(q_lora_rank, idx_n_heads * idx_head_dim), dtype=quant_dtype)
    w_idx_qb_nz = w_idx_qb.reshape(q_lora_rank // 16, 16, idx_n_heads * idx_head_dim // 32, 32).permute(2, 0, 1,
                                                                                                        3)  # int8, C0=32
    w_idx_qb_scale = torch.empty((1, idx_n_heads * idx_head_dim), dtype=torch.float32).uniform_(-1, 1)  # TODO  shape

    w_idx_k = torch.empty((h, idx_head_dim), dtype=dtype).uniform_(-1, 1)
    w_idx_k_nz = w_idx_k.reshape(h // 16, 16, idx_head_dim // 16, 16).permute(2, 0, 1, 3)

    w_idx_proj = torch.empty((h, idx_n_heads), dtype=dtype).uniform_(-1, 1)
    w_idx_proj_nz = w_idx_proj.reshape(h // 16, 16, idx_n_heads // 16, 16).permute(2, 0, 1, 3)

    ln_gamma = torch.ones((idx_head_dim,), dtype=dtype)
    ln_beta = torch.zeros((idx_head_dim,), dtype=dtype)

    hadamard_q = torch.empty((idx_head_dim, idx_head_dim), dtype=dtype).uniform_(-1, 1)  # (128, 128)
    hadamard_k = torch.empty((idx_head_dim, idx_head_dim), dtype=dtype).uniform_(-1, 1)

    k_cache_bsnd = torch.rand((b * s2 * n2, idx_head_dim), dtype=torch.float32) * 2 - 1
    k_cache_bsnd, k_scale_cache_bsnd = gen_mla_prolog_quant_golden_v32.quant(k_cache_bsnd)
    k_cache_bsnd = k_cache_bsnd.reshape(b, s2, n2, idx_head_dim).to(dtype=quant_dtype)
    k_scale_cache_bsnd = k_scale_cache_bsnd.reshape(b, s2, n2, 1).to(torch.float16)

    k_cache = gen_quant_lightning_indexer_prolog.gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size)
    k_scale_cache = gen_quant_lightning_indexer_prolog.gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num,
                                                                        block_size)

    dump_file_torch(w_idx_qb_nz, Path(output_dir, 'w_idx_qb_nz.bin'))
    dump_file_torch(w_idx_qb_scale, Path(output_dir, 'w_idx_qb_scale.bin'))
    dump_file_torch(w_idx_k_nz, Path(output_dir, 'w_idx_k_nz.bin'))
    dump_file_torch(w_idx_proj_nz, Path(output_dir, 'w_idx_proj_nz.bin'))
    dump_file_torch(ln_gamma, Path(output_dir, 'ln_gamma.bin'))
    dump_file_torch(ln_beta, Path(output_dir, 'ln_beta.bin'))
    dump_file_torch(hadamard_q, Path(output_dir, 'hadamard_q.bin'))
    dump_file_torch(hadamard_k, Path(output_dir, 'hadamard_k.bin'))
    dump_file_torch(k_cache, Path(output_dir, 'k_cache.bin'))
    dump_file_torch(k_scale_cache, Path(output_dir, 'k_scale_cache.bin'))

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
        "idx_k_cache_index": cache_index,  # input15, int64  (b, s)/（t,)
        "idx_block_table": block_table,  # input16, int32  (b, ceil(s2, block_size))
        "act_seq": actual_seq,  # input17, int32
        "layernorm_epsilon_k": eps,  # attr0, fp32
    }, {
        "s2": s2,
        "b": b,
        "t": b * s1,
        "h": h,
        "q_lora_rank": q_lora_rank,
        "idx_head_dim": idx_head_dim,
        "idx_n_heads": idx_n_heads,
        "rope_head_dim": rope_dim,
        "block_size": block_size,
        "block_num": block_num,
        "n_kv": n2
    }


def mla_prolog_golden(params, actual_seq, output_dir):
    block_size = params.get("block_size")
    epsilon = params.get("epsilon")
    cache_mode = params.get("cache_mode")

    # mla_prolog 数据
    dtype = torch.bfloat16
    dtypes = (dtype, dtype)
    is_nz = True
    is_quant_a, is_quant_b = (False, True)
    is_quant = (is_quant_a, is_quant_b)
    has_smooth = False
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, cache_index, kv_cache, \
        kr_cache, kv_quant_scale_cache, block_num, block_table = \
        gen_mla_prolog_quant_golden_v32.gen_mla_prolog_quant_v32_input_data(params, dtypes, actual_seq, output_dir,
                                                                            is_quant, is_nz, has_smooth, block_size,
                                                                            cache_mode)

    inputs = {"dtype": dtype, "is_quant_a": is_quant_a, "is_quant_b": is_quant_b, "has_smooth": has_smooth}
    inputs["cache_mode"] = cache_mode
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv
    inputs["epsilon"] = epsilon
    inputs["x"] = x
    inputs["w_dq"] = w_dq
    inputs["w_uqqr"] = w_uqqr
    inputs["w_uk"] = w_uk
    inputs["w_dkvkr"] = w_dkvkr
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["kv_cache"] = kv_cache
    inputs["kr_cache"] = kr_cache
    inputs["kv_quant_scale_cache"] = kv_quant_scale_cache
    inputs["cache_index"] = cache_index
    if is_quant_b:
        inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            inputs["smooth_cq"] = smooth_cq

    q_out, q_embed, rms_norm, rms_norm_scale, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out = \
        gen_mla_prolog_quant_golden_v32.mla_prolog_quant_v32_compute(inputs)

    # output
    q_golden_path = Path(output_dir, 'q_golden.bin')
    q_rope_golden_path = Path(output_dir, 'q_rope_golden.bin')
    rms_norm_golden_path = Path(output_dir, 'rms_norm_golden.bin')
    rms_norm_scale_golden_path = Path(output_dir, 'rms_norm_scale_golden.bin')
    kv_golden_path = Path(output_dir, 'kv_cache_golden.bin')
    kr_golden_path = Path(output_dir, 'kr_cache_golden.bin')
    kv_quant_scale_cache_golden_path = Path(output_dir, 'kv_quant_scale_cache_golden.bin')

    dump_file_torch(q_out, q_golden_path)  # [b,s,n,kv_lora_rank]
    dump_file_torch(q_embed, q_rope_golden_path)  # [b,s,n,qk_rope_head_dim]
    dump_file_torch(rms_norm, rms_norm_golden_path)
    dump_file_torch(rms_norm_scale, rms_norm_scale_golden_path)
    dump_file_torch(kr_cache_out, kr_golden_path)
    dump_file_torch(kv_cache_out, kv_golden_path)
    dump_file_torch(kv_quant_scale_cache_out, kv_quant_scale_cache_golden_path)

    return block_num, block_table, q_out, q_embed, rms_norm, rms_norm_scale, kv_cache_out, kr_cache_out, \
        kv_quant_scale_cache_out, x, cos, sin, cache_index


def indexer_prolog_golden(params, output_dir, block_num, block_table, x, q_norm, q_norm_scale, cos, sin, cache_index,
                          actual_seq):
    indexer_prolog_inputs, indexer_prolog_params = gen_indexer_prolog_inputs(params, output_dir, block_num, \
                                                                             block_table, x, q_norm, q_norm_scale, cos,
                                                                             sin, cache_index, actual_seq)
    indexer_outputs = gen_quant_lightning_indexer_prolog.indexer_prolog(indexer_prolog_inputs, indexer_prolog_params)

    query = indexer_outputs["query"]  # shape is [b, s1, idx_n_heads]
    query_scale = indexer_outputs["query_scale"]  # shape is [b, s1, idx_n_heads]
    idx_k_cache = indexer_outputs["idx_k_cache_out"]  # shape is [b, s1, idx_n_heads]
    idx_k_scale_cache = indexer_outputs["idx_k_scale_cache_out"]  # shape is [b, s1, idx_n_heads]
    weights = indexer_outputs["weights"]  # shape is [b, s1, idx_n_heads]

    dump_file_torch(query, Path(output_dir, "query_golden.bin"))
    dump_file_torch(query_scale, Path(output_dir, "query_scale_golden.bin"))
    dump_file_torch(idx_k_cache, Path(output_dir, "idx_k_cache_golden.bin"))
    dump_file_torch(idx_k_scale_cache, Path(output_dir, "idx_k_scale_cache_golden.bin"))
    dump_file_torch(weights, Path(output_dir, "weights_golden.bin"))

    return query, query_scale, idx_k_cache, idx_k_scale_cache, weights


def lightning_index_golden(params, idx_query, idx_k_cache, idx_query_scale, idx_k_scale_cache, weights, block_table,
                           output_dir):
    kv_cache_actual_seq = params.get("kv_cache_actual_seq")
    block_size = params.get("block_size")
    idx_n_heads = params.get("idx_n_heads")
    idx_head_dim = params.get("idx_head_dim")
    is_quant = params.get("is_quant")

    max_block_num = (max(kv_cache_actual_seq) + block_size - 1) // block_size
    n1_scale = 1.0 / math.sqrt(idx_n_heads)
    idx_softmax_scale = 1.0 / math.sqrt(idx_head_dim)

    lightning_indexer_params = {
        "block_size": block_size,
        "b": params.get("b"),
        "s1": params.get("s1"),
        "n1": idx_n_heads,
        "d": idx_head_dim,
        "n2": params.get("n2"),
        "block_num": params.get("block_num"),  # mla_prolog输出后添加的
        "max_block_num": max_block_num,
        "score_scale": n1_scale * idx_softmax_scale,
        "dtype": idx_query.type,
        "selected_count": params.get("topk")
    }

    input_data_map = {
        "query": idx_query,
        "key": idx_k_cache,
        "q_scale": idx_query_scale,
        "k_scale": idx_k_scale_cache,
        "weights": weights,
        "act_seq": kv_cache_actual_seq,
        "block_table": block_table
    }
    print(f"{idx_query.shape=}")
    topk_value, topk_res, tmp_out = gen_lightning_indexer.indexer_topk_compute(input_data_map, lightning_indexer_params,
                                                                               is_quant)

    # dump golden for compare res
    dump_file_torch(topk_value, Path(output_dir, "topk_value.bin"))
    dump_file_torch(topk_res, Path(output_dir, "topk_res.bin"))
    dump_file_torch(tmp_out, Path(output_dir, "tmp_out.bin"))
    return topk_res


def gather_slc_attn_golden(params, input_data, output_dir):
    q_out, q_embed, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out, topk_indcies, block_table = input_data
    is_quant = params.get("is_quant")
    b = params.get("b")
    s1 = params.get("s1")
    topk = params.get("topk")
    n_q = params.get("num_heads")
    q_dim = params.get("q_dim")
    d_v = params.get("v_dim")
    block_size = params.get("block_size")
    topk = params.get("topk")
    kv_cache_actual_seq = params.get("kv_cache_actual_seq")
    scalar = q_dim ** -0.5
    atten_out_shape = [b, s1, n_q, d_v]
    q_out_tensor = q_out.contiguous().to(torch.bfloat16)
    q_embed_tensor = q_embed.contiguous().to(torch.bfloat16)
    q = torch.concat([q_out_tensor, q_embed_tensor], dim=-1)

    kv_cache_tensor = kv_cache_out.contiguous()
    kr_cache_tensor = kr_cache_out.contiguous().to(q.dtype)  # bf16
    kv_lora_rank = kv_cache_out.shape[-1]
    kv_quant_scale_cache_tensor = kv_quant_scale_cache_out.contiguous()

    kn = kv_cache_tensor.reshape([-1, 512])
    kr = kr_cache_tensor.reshape([-1, 64])
    kn_scales = kv_quant_scale_cache_tensor.reshape([-1, 4])
    params = [block_size, scalar, topk, kv_lora_rank, is_quant]
    input_tensor = [q, kn, kr, kn_scales, topk_indcies, block_table, kv_cache_actual_seq]

    attn_golden, _ = gen_gather_selected_attention.compute_attention(input_tensor, params)

    input_params = [b, s1, n_q, 1, kv_lora_rank, 64, kv_cache_tensor.shape[0], kv_cache_tensor.shape[1], topk, 1]
    input_params_tensor = torch.tensor(input_params, dtype=torch.int32)
    dump_file_torch(input_params_tensor, Path(output_dir, "input_param.bin"))
    dump_file_torch(q_out_tensor, Path(output_dir, "q_nope.bin"))
    dump_file_torch(q_embed_tensor, Path(output_dir, "q_rope.bin"))
    dump_file_torch(kn, Path(output_dir, "k_nope.bin"))
    dump_file_torch(kr, Path(output_dir, "k_rope.bin"))
    dump_file_torch(kn_scales, Path(output_dir, "kn_scales.bin"))
    dump_file_torch(attn_golden, Path(output_dir, "atten_out.bin"))

    return attn_golden


def gen_deepseek_indexer_attention_golden(params, actual_seq, output_dir: Path):
    logging.debug(f"gen_deepseek_indexer_attention_golden")
    # 2. 计算 & dump file
    # mla-prolog 子图
    print("============ mla prolog ==================")
    block_num, block_table, q_out, q_embed, rms_norm, rms_norm_scale, kv_cache_out, kr_cache_out, \
        kv_quant_scale_cache_out, x, cos, sin, cache_index = mla_prolog_golden(params, actual_seq, output_dir)
    params["block_num"] = block_num
    x = x.view(torch.int16).view(torch.bfloat16)
    cos = cos.view(torch.int16).view(torch.bfloat16)
    sin = sin.view(torch.int16).view(torch.bfloat16)
    dump_file_torch(block_table, Path(output_dir, 'block_table.bin'))
    dump_file_torch(actual_seq, Path(output_dir, 'actual_seq.bin'))

    # Lightning Indexer prolog子图
    print("============ Lightning Indexer prolog ==================")
    idx_query, idx_query_scale, idx_k_cache, idx_k_scale_cache, weights = indexer_prolog_golden(params, output_dir, \
                                                                                                block_num, block_table,
                                                                                                x, rms_norm,
                                                                                                rms_norm_scale, cos,
                                                                                                sin, cache_index,
                                                                                                actual_seq)

    # Lightning Indexer 子图
    print("============ Lightning Indexer ==================")
    topk_indcies = lightning_index_golden(params, idx_query, idx_k_cache, idx_query_scale, idx_k_scale_cache, weights, block_table, output_dir)
    # sfa 子图
    print("============ sfa ==================")
    input_data = [q_out, q_embed, kv_cache_out, kr_cache_out, kv_quant_scale_cache_out, topk_indcies, block_table]
    attn_golden = gather_slc_attn_golden(params, input_data, output_dir)

    dump_file_torch(attn_golden, Path(output_dir, "attn_golden.bin"))
    logging.debug(f"gen_deepseek_indexer_attention_golden done")

    return True


def deepseek_indexer_attention_entry(bs1s2h, actual_seq, output_dir: Path):
    b, s1, s2, h = bs1s2h
    kv_lora_rank = 512
    rope_dim = 64
    q_dim = kv_lora_rank + rope_dim
    k_dim = kv_lora_rank + rope_dim
    v_dim = kv_lora_rank
    v_head_dim = 128
    epsilon = 1e-5
    cache_mode = "PA_BSND"
    actual_seq = torch.tensor(actual_seq, dtype=torch.int32).unsqueeze(-1)

    # index 参数
    idx_n_heads = 64
    idx_head_dim = 128
    topk = 2048
    is_quant = True

    params = {
        "b": b,
        "s": s1,
        "s1": s1,
        "s2": s2,
        "n1": 128,
        "n2": 1,
        "h": h,
        "num_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rope_dim": rope_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "topk": topk,
        "block_size": 128,
        "kv_cache_actual_seq": actual_seq,
        "epsilon": epsilon,
        "cache_mode": cache_mode,
        "v_head_dim": v_head_dim,
        "idx_n_heads": idx_n_heads,
        "idx_head_dim": idx_head_dim,
        "is_quant": is_quant,
    }
    print("cur actual seq: ", actual_seq)
    gen_deepseek_indexer_attention_golden(params, actual_seq, output_dir)

    # 将变化的参数保存到文件中，供测试用例直接读取
    input_params = torch.tensor([params.get("b"), params.get("s"), params.get("s2"), params.get("n1"),
                                 params.get("n2"), params.get("block_num"), params.get("topk")], dtype=torch.int32)
    dump_file_torch(input_params, Path(output_dir, 'input_params.bin'))


@GoldenRegister.reg_golden_func(
    case_names=[
        "DeepSeekIndexerAttentionQuantSTest.4B_mtp",
        "DeepSeekIndexerAttentionQuantSTest.4B_mtp_perf",
        "DeepSeekIndexerAttentionQuantSTest.32B"
    ],
    version=0,
    timeout=0
)
def gen_deepseek_indexer_attention_func(case_name: str, output: Path) -> bool:
    input_path = Path(output, 'x.bin')
    complete = input_path.exists()
    complete = False
    if complete:
        logging.info("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "DeepSeekIndexerAttentionQuantSTest.4B_mtp":
            b, s1, s2 = 4, 2, 128 * 1024
            kv_act_seq = [768, 4097, 8192, 131071]
            deepseek_indexer_attention_entry((b, s1, s2, 7168), kv_act_seq, output)
        elif case_name == "DeepSeekIndexerAttentionQuantSTest.4B_mtp_perf":
            b, s1, s2 = 4, 2, 128 * 1024
            kv_act_seq = [65536, 65537, 65538, 65539]
            deepseek_indexer_attention_entry((b, s1, s2, 7168), kv_act_seq, output)
        elif case_name == "DeepSeekIndexerAttentionQuantSTest.32B":
            b, s1, s2 = 32, 1, 128 * 1024
            kv_act_seq = [8192] * b
            deepseek_indexer_attention_entry((b, s1, s2, 7168), kv_act_seq, output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DeepSeekIndexerAttentionQuantSTest.4B_mtp_perf",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "../../build/output/bin/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(output_dir)
        ret = gen_deepseek_indexer_attention_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
