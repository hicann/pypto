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
"""
from dataclasses import dataclass
import math
import torch
import torch_npu
import pypto
from typing import List
import lightning_indexer_prolog_quant_impl as ip
import mla_prolog_quant_impl as mla
"""
MLA Indexer Prolog Quantization Module

This module implements fused MLA Prolog and Lightning Indexer Prolog computation
for DeepSeek V32 model. It combines both operators to enable pipeline parallelism
and improve overall performance.

Main Functions:
    - mla_indexer_prolog_quant_p: Fused computation for prefill phase
    - mla_indexer_prolog_quant_d: Fused computation for decode phase

Example:
    See deepseekv32_mla_indexer_prolog_quant.py for usage examples.
"""


@pypto.jit(
    # prefill版本融合算子优化参数
    runtime_options={"stitch_function_inner_memory": 512, "stitch_function_outcast_memory": 512,
                     "device_sched_mode": 2}
)
def mla_indexer_prolog_quant_p(token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                             mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                             mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
                             ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
                             ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
                             mla_kv_cache_out, mla_kr_cache_out,
                             mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
                             ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
                             mla_cache_mode, mla_tile_config,
                             ip_attrs, ip_configs, rope_cfg):
    """Fused MLA and Indexer Prolog quantization for prefill phase.
    
    Combines MLA Prolog and Lightning Indexer Prolog computations in a single
    fused operator for prefill phase. This enables pipeline parallelism and
    reduces memory transfers between operators.
    
    The computation flow:
    1. MLA Prolog: Computes MLA query, key, and value projections
    2. Indexer Prolog: Uses MLA's q_norm output to compute indexer query, key, and weights
    
    Args:
        token_x: Input token tensor, shape (t, h), dtype BF16
        mla_w_dq: MLA down-projection weight for query, NZ format
        mla_w_uq_qr: MLA up-projection weight for query and RoPE, NZ format
        mla_dequant_scale: MLA dequantization scale, FP32
        mla_w_uk: MLA up-projection weight for key, BF16
        mla_w_dkv_kr: MLA down-projection weight for key-value and RoPE, NZ format
        mla_gamma_cq: MLA RMSNorm scale for query, BF16
        mla_gamma_ckv: MLA RMSNorm scale for key-value, BF16
        cos: Cosine values for RoPE, BF16
        sin: Sine values for RoPE, BF16
        cache_index: Cache index for scatter update, INT64
        mla_kv_cache: MLA key-value cache input/output, INT8
        mla_kr_cache: MLA key RoPE cache input/output, BF16
        mla_k_scale_cache: MLA key scale cache input/output, FP16
        ip_w_qb_in: Indexer query projection weight, INT8, NZ format
        ip_w_qb_scale_in: Indexer query weight dequantization scale, FP32
        ip_wk_in: Indexer key projection weight, BF16, NZ format
        ip_w_proj_in: Indexer weight projection matrix, BF16, NZ format
        ip_ln_gamma_k_in: Indexer LayerNorm scale for key, BF16
        ip_ln_beta_k_in: Indexer LayerNorm shift for key, BF16
        ip_hadamard_q_in: Indexer Hadamard matrix for query, BF16
        ip_hadamard_k_in: Indexer Hadamard matrix for key, BF16
        ip_k_cache: Indexer key cache input/output, INT8
        ip_k_cache_scale: Indexer key cache scale input/output, FP16
        mla_query_nope_out: Output MLA query without RoPE, BF16
        mla_query_rope_out: Output MLA query with RoPE, BF16
        mla_kv_cache_out: Output MLA key-value cache
        mla_kr_cache_out: Output MLA key RoPE cache
        mla_k_scale_cache_out: Output MLA key scale cache
        ip_q_int8_out: Output indexer quantized query, INT8
        ip_q_scale_out: Output indexer query quantization scale, FP16
        ip_k_int8_out: Output indexer key cache
        ip_k_scale_out: Output indexer key cache scale
        ip_weights_out: Output indexer weights, FP16
        mla_epsilon_cq: MLA RMSNorm epsilon for query
        mla_epsilon_ckv: MLA RMSNorm epsilon for key-value
        mla_cache_mode: MLA cache mode
        mla_tile_config: MlaTileConfig object for MLA computation
        ip_attrs: IndexerPrologQuantAttr object for indexer computation
        ip_configs: IndexerPrologQuantConfigs object for indexer computation
        rope_cfg: RopeTileShapeConfig object for RoPE computation
        
    Note:
        The function creates intermediate tensors (mla_q_norm_out, mla_q_norm_scale_out)
        to pass data from MLA Prolog to Indexer Prolog. Pipeline parallelism is
        enabled through device_sched_mode=2.
    """
    ##################### mla #######################
    # mla输出的q_norm和q_norm_scale结果用于ip模块的输入， t和q_lora_rank表示输出tensor轴大小
    t = token_x.shape[0]
    q_lora_rank = ip_w_qb_in.shape[0]
    mla_q_norm_out = pypto.Tensor([t, q_lora_rank], pypto.DT_INT8)
    mla_q_norm_scale_out = pypto.Tensor([t, 1], pypto.DT_FP32)

    # 设置mla模块的优化参数
    pypto.set_pass_options(vec_nbuffer_mode=mla_tile_config.vec_nbuffer_mode,
                           cube_l1_reuse_setting=mla_tile_config.cube_l1_reuse_setting,
                           cube_nbuffer_setting=mla_tile_config.cube_nbuffer_setting,
                           mg_copyin_upper_bound=mla_tile_config.mg_copyin_upper_bound)
    
    # mla模块的输入输出tensor
    mla_input_tensors = (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                         mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                         mla_k_scale_cache)
    mla_output_tensors = (mla_q_norm_out, mla_q_norm_scale_out, mla_query_nope_out, mla_query_rope_out,
                          mla_kv_cache_out, mla_kr_cache_out, mla_k_scale_cache_out)
    
    # mla模块计算流
    mla.mla_prolog_quant_compute(*mla_input_tensors, *mla_output_tensors, mla_epsilon_cq, mla_epsilon_ckv,
                                   mla_cache_mode, mla_tile_config, rope_cfg)

    ##################### ip #######################
    # 设置ip模块的优化参数
    pypto.set_pass_options(vec_nbuffer_mode=ip_configs.vec_nbuffer_mode)
    pypto.set_pass_options(cube_l1_reuse_setting=ip_configs.cube_l1_reuse_setting)
    pypto.set_pass_options(mg_copyin_upper_bound=ip_configs.mg_copyin_upper_bound)
    pypto.set_pass_options(pg_upper_bound=ip_configs.pg_upper_bound)
    
    # ip模块的输入输出tensor
    ip_input_tensors = (token_x, mla_q_norm_out, mla_q_norm_scale_out, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in,
                        ip_w_proj_in, ip_ln_gamma_k_in, ip_ln_beta_k_in, cos, sin, ip_hadamard_q_in, ip_hadamard_k_in,
                        ip_k_cache, ip_k_cache_scale, cache_index)
    ip_output_tensors = (ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out)
    # mla模块计算流
    ip.lightning_indexer_prolog_quant_compute(*ip_input_tensors, *ip_output_tensors, ip_attrs, ip_configs)


@pypto.jit
def mla_indexer_prolog_quant_d(token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                             mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                             mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
                             ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
                             ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
                             mla_kv_cache_out, mla_kr_cache_out,
                             mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
                             ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
                             mla_cache_mode, mla_tile_config,
                             ip_attrs, ip_configs, rope_cfg):
    """Fused MLA and Indexer Prolog quantization for decode phase.
    
    Combines MLA Prolog and Lightning Indexer Prolog computations in a single
    fused operator for decode phase. Optimized for low latency processing.
    
    The computation flow:
    1. MLA Prolog: Computes MLA query, key, and value projections
    2. Indexer Prolog: Uses MLA's q_norm output to compute indexer query, key, and weights
    
    Args:
        token_x: Input token tensor, shape (t, h), dtype BF16
        mla_w_dq: MLA down-projection weight for query, NZ format
        mla_w_uq_qr: MLA up-projection weight for query and RoPE, NZ format
        mla_dequant_scale: MLA dequantization scale, FP32
        mla_w_uk: MLA up-projection weight for key, BF16
        mla_w_dkv_kr: MLA down-projection weight for key-value and RoPE, NZ format
        mla_gamma_cq: MLA RMSNorm scale for query, BF16
        mla_gamma_ckv: MLA RMSNorm scale for key-value, BF16
        cos: Cosine values for RoPE, BF16
        sin: Sine values for RoPE, BF16
        cache_index: Cache index for scatter update, INT64
        mla_kv_cache: MLA key-value cache input/output, INT8
        mla_kr_cache: MLA key RoPE cache input/output, BF16
        mla_k_scale_cache: MLA key scale cache input/output, FP16
        ip_w_qb_in: Indexer query projection weight, INT8, NZ format
        ip_w_qb_scale_in: Indexer query weight dequantization scale, FP32
        ip_wk_in: Indexer key projection weight, BF16, NZ format
        ip_w_proj_in: Indexer weight projection matrix, BF16, NZ format
        ip_ln_gamma_k_in: Indexer LayerNorm scale for key, BF16
        ip_ln_beta_k_in: Indexer LayerNorm shift for key, BF16
        ip_hadamard_q_in: Indexer Hadamard matrix for query, BF16
        ip_hadamard_k_in: Indexer Hadamard matrix for key, BF16
        ip_k_cache: Indexer key cache input/output, INT8
        ip_k_cache_scale: Indexer key cache scale input/output, FP16
        mla_query_nope_out: Output MLA query without RoPE, BF16
        mla_query_rope_out: Output MLA query with RoPE, BF16
        mla_kv_cache_out: Output MLA key-value cache
        mla_kr_cache_out: Output MLA key RoPE cache
        mla_k_scale_cache_out: Output MLA key scale cache
        ip_q_int8_out: Output indexer quantized query, INT8
        ip_q_scale_out: Output indexer query quantization scale, FP16
        ip_k_int8_out: Output indexer key cache
        ip_k_scale_out: Output indexer key cache scale
        ip_weights_out: Output indexer weights, FP16
        mla_epsilon_cq: MLA RMSNorm epsilon for query
        mla_epsilon_ckv: MLA RMSNorm epsilon for key-value
        mla_cache_mode: MLA cache mode
        mla_tile_config: MlaTileConfig object for MLA computation
        ip_attrs: IndexerPrologQuantAttr object for indexer computation
        ip_configs: IndexerPrologQuantConfigs object for indexer computation
        rope_cfg: RopeTileShapeConfig object for RoPE computation
        
    Note:
        The function creates intermediate tensors (mla_q_norm_out, mla_q_norm_scale_out)
        to pass data from MLA Prolog to Indexer Prolog. Optimized for decode phase
        with minimal latency.
    """
    ##################### mla #######################
    # mla输出的q_norm和q_norm_scale结果用于ip模块的输入， t和q_lora_rank表示输出tensor轴大小
    t = token_x.shape[0]
    q_lora_rank = ip_w_qb_in.shape[0]
    mla_q_norm_out = pypto.Tensor([t, q_lora_rank], pypto.DT_INT8)
    mla_q_norm_scale_out = pypto.Tensor([t, 1], pypto.DT_FP32)

    # 设置mla模块的优化参数
    pypto.set_pass_options(vec_nbuffer_mode=mla_tile_config.vec_nbuffer_mode,
                           cube_l1_reuse_setting=mla_tile_config.cube_l1_reuse_setting,
                           cube_nbuffer_setting=mla_tile_config.cube_nbuffer_setting,
                           mg_copyin_upper_bound=mla_tile_config.mg_copyin_upper_bound)
    
    # mla模块的输入输出tensor
    mla_input_tensors = (token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
                         mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
                         mla_k_scale_cache)
    mla_output_tensors = (mla_q_norm_out, mla_q_norm_scale_out, mla_query_nope_out, mla_query_rope_out,
                          mla_kv_cache_out, mla_kr_cache_out, mla_k_scale_cache_out)
    # mla模块计算流
    mla.mla_prolog_quant_compute(*mla_input_tensors, *mla_output_tensors, mla_epsilon_cq, mla_epsilon_ckv,
                                   mla_cache_mode, mla_tile_config, rope_cfg)

    ##################### ip #######################
    # 设置ip模块的优化参数
    pypto.set_pass_options(vec_nbuffer_mode=ip_configs.vec_nbuffer_mode)
    pypto.set_pass_options(cube_l1_reuse_setting=ip_configs.cube_l1_reuse_setting)
    pypto.set_pass_options(mg_copyin_upper_bound=ip_configs.mg_copyin_upper_bound)
    pypto.set_pass_options(pg_upper_bound=ip_configs.pg_upper_bound)
    
    # ip模块的输入输出tensor
    ip_input_tensors = (token_x, mla_q_norm_out, mla_q_norm_scale_out, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in,
                        ip_w_proj_in, ip_ln_gamma_k_in, ip_ln_beta_k_in, cos, sin, ip_hadamard_q_in, ip_hadamard_k_in,
                        ip_k_cache, ip_k_cache_scale, cache_index)
    ip_output_tensors = (ip_q_int8_out, ip_q_scale_out, ip_k_int8_out, ip_k_scale_out, ip_weights_out)
    # mla模块计算流
    ip.lightning_indexer_prolog_quant_compute(*ip_input_tensors, *ip_output_tensors, ip_attrs, ip_configs)