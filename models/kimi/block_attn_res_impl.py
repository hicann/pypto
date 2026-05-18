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
Block Attention Residuals Implementation Module

This module implements the core computation functions for Block Attention Residuals.
It provides both forward and backward passes, with optional RMSNorm support and cache outputs.

Main Functions:
    - ai_infra_block_attn_res: Forward pass for block attention residuals
    - ai_infra_block_attn_res_backward: Backward pass for block attention residuals

Kernels:
    - ai_infra_block_attn_res_kernel: 前向 kernel
    - ai_infra_block_attn_res_backward_kernel: 反向 kernel
"""

import typing
from typing import List, Optional

import pypto
import torch


L_MAX = 128


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 64,
    }
)
def ai_infra_block_attn_res_kernel(
    v_in: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    q_in: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma_in: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale: float,
    rms_norm_eps: float,
    enable_rmsnorm: bool,
):
    """PyPTO jit ai_infra_block_attn_res_kernel

    v_in:          [B*T, L, D]    bf16/fp16
    q_in:          [1, 1, D]      bf16/fp16
    gamma_in:      [1, D]         bf16/fp16 (enable_rmsnorm=False 时不读取)
    h_out:         [B*T, 1, D]    bf16/fp16
    rms_out:       [B*T, L, 1]    fp32  (RMSNorm denominator cache)
    alpha_out:     [B*T, L]       fp32  (softmax attention weights cache)
    scale:         softmax 缩放因子
    rms_norm_eps:  RMSNorm epsilon
    enable_rmsnorm: 是否启用 RMSNorm
    """
    t, l, d = v_in.shape
    dtype = v_in.dtype
    unroll_list = [1, 64]
    norm_m_tile = 64 * 1024 // (4 * d)
    norm_m_tile = max(norm_m_tile, 1)

    if d == 512:
        pypto.set_pass_options(
            vec_nbuffer_setting={"DEFAULT": 8, "func8_2": 4, "func8_1": 16, "func8_7": 16, "func8_8": 16},
            cube_nbuffer_setting={-1: 16},
            cube_l1_reuse_setting={-1: 2}
        )
    elif d == 1536:
        pypto.set_pass_options(
            vec_nbuffer_setting={"DEFAULT": 8, "func8_8": 32},
            cube_nbuffer_setting={-1: 16},
            cube_l1_reuse_setting={-1: 2}
        )

    pypto.experimental.set_operation_options(combine_axis=True)

    for t_idx, unroll_length in pypto.loop_unroll(
        0, t, 1,
        name="Block_Attn_Res_T_Loop",
        idx_name="t_idx",
        unroll_list=unroll_list,
    ):
        t_tile = unroll_length

        v = pypto.view(v_in, [t_tile, L_MAX, d], [t_idx, 0, 0],
                       valid_shape=[t_tile, l, d])

        if enable_rmsnorm:
            pypto.set_semantic_label("RmsNorm")
            pypto.set_vec_tile_shapes(1, norm_m_tile, d)
            v_fp32 = pypto.cast(v, pypto.DT_FP32)
            v_sq = pypto.mul(v_fp32, v_fp32)
            v_sq_sum = pypto.sum(v_sq, dim=-1, keepdim=True)
            mean_val = pypto.div(v_sq_sum, d)
            rms_val = pypto.sqrt(pypto.add(mean_val, rms_norm_eps))
            k_fp32 = pypto.div(v_fp32, rms_val)
            gamma_fp32 = pypto.cast(gamma_in, pypto.DT_FP32)
            k_fp32 = pypto.mul(k_fp32, gamma_fp32)

            pypto.assemble(rms_val, [t_idx, 0, 0], rms_out)

        pypto.set_semantic_label("Logits Matmul")
        if enable_rmsnorm:
            matmul_lhs = k_fp32
        else:
            pypto.set_vec_tile_shapes(1, norm_m_tile, D)
            matmul_lhs = pypto.cast(v, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 1, d)
        q_in_fp32 = pypto.cast(q_in, pypto.DT_FP32)
        pypto.set_cube_tile_shapes([128, 128], [128, 512], [16, 16])
        logits_3d = pypto.matmul(matmul_lhs, q_in_fp32, pypto.DT_FP32, a_trans=False, b_trans=True)
        logits = pypto.reshape(logits_3d, [t_tile, L_MAX], valid_shape=[t_tile, l])

        pypto.set_semantic_label("Softmax")
        pypto.set_vec_tile_shapes(128, 128)
        if scale != 1.0:
            logits = pypto.mul(logits, scale)
        alpha = pypto.softmax(logits, dim=-1)

        pypto.assemble(alpha, [t_idx, 0], alpha_out)

        alpha_3d = pypto.reshape(alpha, [t_tile, 1, L_MAX], valid_shape=[t_tile, 1, l])
        if not enable_rmsnorm:
            pypto.set_vec_tile_shapes(1, norm_m_tile, d)
            v_fp32 = pypto.cast(v, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(128, 1, 128)
        alpha_3d = alpha_3d + 0.0

        pypto.set_semantic_label("Weighted Summation")
        pypto.set_cube_tile_shapes([1, 1], [128, 128], [128, 128])
        h = pypto.matmul(alpha_3d, v_fp32, dtype)

        pypto.assemble(h, [t_idx, 0, 0], h_out)


def _validate_inputs(
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    partial_block: Optional[torch.Tensor],
    rmsnorm_gamma: Optional[torch.Tensor],
    enable_rmsnorm: bool,
    rms_out_flag: bool,
):
    if not blocks:
        raise ValueError("blocks must not be empty")
    n = len(blocks)
    if n < 1 or n > 127:
        raise ValueError(f"N (number of blocks) must be in [1, 127], got {n}")

    b, t, d = blocks[0].shape
    dtype = blocks[0].dtype

    for i, block in enumerate(blocks):
        if block.shape != (b, t, d):
            raise ValueError(f"blocks[{i}] shape {block.shape} != expected {(b, t, d)}")
        if block.dtype != dtype:
            raise ValueError(f"blocks[{i}] dtype {block.dtype} != expected {dtype}")

    if partial_block is not None:
        if partial_block.shape != (b, t, d):
            raise ValueError(f"partial_block shape {partial_block.shape} != expected {(b, t, d)}")
        if partial_block.dtype != dtype:
            raise ValueError(f"partial_block dtype {partial_block.dtype} != expected {dtype}")

    if proj_weight.shape != (d,):
        raise ValueError(f"proj_weight shape must be [D], got {proj_weight.shape}")
    if proj_weight.dtype != dtype:
        raise ValueError(f"proj_weight dtype {proj_weight.dtype} != expected {dtype}")

    if enable_rmsnorm and rmsnorm_gamma is None:
        raise ValueError("rmsnorm_gamma is required when enable_rmsnorm=True")

    if rmsnorm_gamma is not None:
        if rmsnorm_gamma.shape != (d,):
            raise ValueError(f"rmsnorm_gamma shape must be [D], got {rmsnorm_gamma.shape}")
        if rmsnorm_gamma.dtype != dtype:
            raise ValueError(f"rmsnorm_gamma dtype {rmsnorm_gamma.dtype} != expected {dtype}")

    if rms_out_flag and not enable_rmsnorm:
        raise ValueError("rms_out_flag=True is invalid when enable_rmsnorm=False (no RMSNorm output to cache)")


def ai_infra_block_attn_res(
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    partial_block: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    rmsnorm_eps: float = 1e-6,
    rmsnorm_gamma: Optional[torch.Tensor] = None,
    enable_rmsnorm: bool = True,
    rms_out_flag: bool = False,
    alpha_out_flag: bool = False,
):
    """Block Attention Residuals 前向算子调用入口。

    Args:
        blocks: N 个已完成 block 表示, 每个 [B, T, D]
        proj_weight: 伪查询投影权重 [D]
        partial_block: 当前 block 部分和 [B, T, D]
        scale: softmax 缩放因子
        rmsnorm_eps: RMSNorm epsilon
        rmsnorm_gamma: RMSNorm 缩放参数 [D]
        enable_rmsnorm: 是否启用 RMSNorm
        rms_out_flag: 是否输出 rms_cache 供反向计算复用, 默认 False
        alpha_out_flag: 是否输出 alpha_cache 供反向计算复用, 默认 False

    Returns:
        - 仅 output [B,T,D] (两个 flag 均为 False)
        - (output, rms_cache [B,T,L,1]) (仅 rms_out_flag=True)
        - (output, alpha_cache [B,T,L]) (仅 alpha_out_flag=True)
        - (output, rms_cache, alpha_cache) (两个 flag 均为 True)
    """
    _validate_inputs(blocks, proj_weight, partial_block,
                     rmsnorm_gamma, enable_rmsnorm, rms_out_flag)
    n = len(blocks)
    l = n + 1 if partial_block is not None else n
    b, t, d = blocks[0].shape
    dtype = blocks[0].dtype
    device = blocks[0].device

    need_cache = rms_out_flag or alpha_out_flag

    if partial_block is not None:
        v = torch.stack(blocks + [partial_block], dim=2)
    else:
        v = torch.stack(blocks, dim=2)

    v = v.reshape(b * t, l, d)
    q = proj_weight.reshape(1, 1, d)
    gamma = (rmsnorm_gamma.reshape(1, d) if rmsnorm_gamma is not None
             else torch.ones(1, d, dtype=dtype, device=device))
    h_out = torch.zeros(b * t, 1, d, dtype=dtype, device=device)

    rms_cache_flat = torch.zeros(b * t, l, 1, dtype=torch.float32, device=device)
    alpha_cache_flat = torch.zeros(b * t, l, dtype=torch.float32, device=device)
    ai_infra_block_attn_res_kernel(
        v, q, gamma, h_out, rms_cache_flat, alpha_cache_flat,
        scale, rmsnorm_eps, enable_rmsnorm)

    output = h_out.reshape(b, t, d)

    if not need_cache:
        return output

    results = [output]
    if rms_out_flag:
        results.append(rms_cache_flat.reshape(b, t, l, 1))
    if alpha_out_flag:
        results.append(alpha_cache_flat.reshape(b, t, l))
    return tuple(results)
    


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 32,
    }
)
def ai_infra_block_attn_res_backward_kernel(
    v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_h_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    rms_cache: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    alpha_cache_3d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.DYNAMIC]),
    proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    gamma: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_v_flat: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]),
    grad_proj_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    grad_gamma: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    scale: float,
    enable_rmsnorm: bool,
):
    """PyPTO jit ai_infra_block_attn_res_backward_kernel

    v_flat:            [B*T, L, D]    bf16/fp16
    grad_h_3d:         [B*T, 1, D]    bf16/fp16
    rms_cache:         [B*T, L, 1]    fp32 (enable_rmsnorm=False 时为空)
    alpha_cache_3d:    [B*T, 1, L]    fp32
    proj_weight:       [1, 1, D]      bf16/fp16
    gamma:             [1, 1, D]      bf16/fp16
    grad_v_flat:       [B*T, L, D]    bf16/fp16
    grad_proj_weight:  [1, 1, D]      bf16/fp16
    grad_gamma:        [1, 1, D]      bf16/fp16
    scale:         softmax 缩放因子
    enable_rmsnorm: 是否启用 RMSNorm
    """
    bt, l, d = v_flat.shape
    dtype = v_flat.dtype
    unroll_list = [1, 64]
    l_tile = max(16384 // d, 1)

    if d == 512:
        pypto.set_pass_options(
            vec_nbuffer_setting={"DEFAULT": 16, "func5_0": 2, "func8_7": 32, "func8_6": 8, "func8_2": 2, "func8_1": 32,
            "func8_9": 32, "func8_4": 16, "func8_12": 64, "func8_11": 32, "func8_0": 8, "func8_16": 32, "func8_17": 32,
            "func8_13": 32, "func8_14": 4, "func8_18": 4},
            cube_nbuffer_setting={"DEFAULT": 16, "func8_1": 32, "func8_2": 32},
            cube_l1_reuse_setting={"DEFAULT": 2}
            )
    elif d == 1536:
        pypto.set_pass_options(
            vec_nbuffer_setting={"DEFAULT": 16, "func5_0": 2, "func8_6": 8, "func8_2": 6, "func8_1": 64, "func8_8": 4,
            "func8_10": 32, "func8_14": 32, "func8_21": 32, "func8_23": 32, "func8_22": 32, "func8_12": 32,
            "func8_24": 32, "func8_15": 32, "func8_17": 32, "func8_3": 3, "func8_16": 32},
            cube_nbuffer_setting={"DEFAULT": 16, "func8_0": 32, "func8_2": 32},
            cube_l1_reuse_setting={"DEFAULT": 2}
            )

    pypto.experimental.set_operation_options(combine_axis=True)

    pypto.set_vec_tile_shapes(1, 1, d)
    grad_proj_weight_acc = pypto.zeros([1, 1, d], dtype=pypto.DT_FP32)
    if enable_rmsnorm:
        grad_gamma_acc = pypto.zeros([1, 1, d], dtype=pypto.DT_FP32)

    for bt_idx, tile_bt in pypto.loop_unroll(0, bt, 1, name="BT_LOOP", idx="bt_idx", unroll_list=unroll_list):
        pypto.set_semantic_label("Weighted Summation Backward")
        pypto.set_vec_tile_shapes(1, l_tile, d)
        grad_h_tile_3d = pypto.view(grad_h_3d, [tile_bt, 1, d], [bt_idx, 0, 0])

        # matmul1: [tile_bt, 1, L_MAX] grad_alpha = grad_h @ V^T
        v_tile = pypto.view(v_flat, [tile_bt, L_MAX, d], [bt_idx, 0, 0],
                            valid_shape=[tile_bt, l, d])
        pypto.set_cube_tile_shapes([1, 1], [256, 512], [128, 128])
        grad_alpha_fp32 = pypto.matmul(grad_h_tile_3d, v_tile, pypto.DT_FP32, b_trans=True)
        
        # matmul2: [tile_bt, L_MAX, D] grad_V_agg = alpha^T @ grad_h
        alpha_tile_3d = pypto.view(alpha_cache_3d, [tile_bt, 1, L_MAX], [bt_idx, 0, 0],
                                   valid_shape=[tile_bt, 1, l])
        pypto.set_cube_tile_shapes([128, 128], [16, 16], [256, 256])
        pypto.set_vec_tile_shapes(l_tile, 1, d)
        grad_h_tile_3d_fp32 = pypto.cast(grad_h_tile_3d, pypto.DT_FP32)
        grad_v_agg_bmm_fp32 = pypto.matmul(alpha_tile_3d, grad_h_tile_3d_fp32, pypto.DT_FP32, a_trans=True)
        grad_v_agg_fp32 = pypto.view(grad_v_agg_bmm_fp32, [tile_bt, L_MAX, d], [0, 0, 0], valid_shape=[tile_bt, l, d])

        pypto.set_semantic_label("Attention Softmax Backward")
        pypto.set_vec_tile_shapes(128, 128, 16)
        grad_alpha_2d_fp32 = pypto.reshape(grad_alpha_fp32, [tile_bt, L_MAX],
                                   valid_shape=[tile_bt, l])
        alpha_2d = pypto.reshape(alpha_tile_3d, [tile_bt, L_MAX],
                                 valid_shape=[tile_bt, l])
        pypto.set_vec_tile_shapes(128, 128)
        alpha_fp32_2d = pypto.cast(alpha_2d, pypto.DT_FP32)

        # Softmax 反向：[tile_bt, 1] dot = sum(grad_alpha * alpha)
        dot_fp32 = pypto.sum(grad_alpha_2d_fp32 * alpha_fp32_2d, dim=-1, keepdim=True)

        # Softmax 反向：[tile_bt, L_MAX] grad_logits = alpha * (grad_alpha - dot)
        grad_logits_fp32 = alpha_fp32_2d * (grad_alpha_2d_fp32 - dot_fp32)
        grad_logits_reshaped = pypto.reshape(grad_logits_fp32, [tile_bt, 1, L_MAX],
                                             valid_shape=[tile_bt, 1, l])
        pypto.set_vec_tile_shapes(128, 1, 128)
        grad_logits_reshaped_fp32 = grad_logits_reshaped + 0.0

        pypto.set_semantic_label("Attention Scores Backward")
        pypto.set_vec_tile_shapes(1, l_tile, d)
        v_tile_fp32 = pypto.cast(v_tile, pypto.DT_FP32)
        proj_weight_fp32 = pypto.cast(proj_weight, pypto.DT_FP32)
        proj_weight_scale = scale * proj_weight_fp32
        proj_weight_scale_fp32 = proj_weight_scale
        if enable_rmsnorm:
            pypto.set_vec_tile_shapes(128, 128, 1)
            rms_tile = pypto.view(rms_cache, [tile_bt, L_MAX, 1], [bt_idx, 0, 0],
                                valid_shape=[tile_bt, l, 1])
            rms_tile_fp32 = pypto.cast(rms_tile, pypto.DT_FP32)
            pypto.set_vec_tile_shapes(1, l_tile, d)
            gamma_fp32 = pypto.cast(gamma, pypto.DT_FP32)

            # RMSNorm 正向: [t_tile, L_MAX, D] K = V / rms * gamma
            k_tile_fp32 = v_tile_fp32 / rms_tile_fp32 * gamma_fp32
        else:
            k_tile_fp32 = v_tile_fp32

        # matmul3: [tile_bt, L_MAX, D] grad_K = grad_logits @ proj_weight
        pypto.set_cube_tile_shapes([128, 128], [16, 16], [128, 128])
        grad_k_bmm_fp32 = pypto.matmul(grad_logits_reshaped_fp32, proj_weight_scale_fp32, pypto.DT_FP32, a_trans=True)
        grad_k_fp32 = pypto.view(grad_k_bmm_fp32, [tile_bt, L_MAX, d], [0, 0, 0], valid_shape=[tile_bt, l, d])

        pypto.set_vec_tile_shapes(1, l_tile, d)
        grad_logits_reshaped = pypto.reshape(grad_logits_fp32 + 0.0, [tile_bt, L_MAX, 1],
                                             valid_shape=[tile_bt, l, 1])
        grad_proj_weight_tile_fp32 = pypto.mul(grad_logits_reshaped, k_tile_fp32)
        pypto.set_vec_tile_shapes(tile_bt, 1, 16384 // tile_bt)
        grad_proj_weight_tile_fp32 = pypto.sum(grad_proj_weight_tile_fp32, dim=0, keepdim=True)
        pypto.set_pass_options(sg_set_scope=1)
        pypto.set_vec_tile_shapes(1, L_MAX, 16384 // L_MAX)
        grad_proj_weight_tile_fp32 = pypto.sum(grad_proj_weight_tile_fp32, dim=1, keepdim=True)
        pypto.set_pass_options(sg_set_scope=-1)

        pypto.set_vec_tile_shapes(1, 1, d)
        grad_proj_weight_scaled_fp32 = grad_proj_weight_tile_fp32 * scale
        grad_proj_weight_acc[:] = grad_proj_weight_acc + grad_proj_weight_scaled_fp32

        if enable_rmsnorm:
            pypto.set_semantic_label("RmsNorm Backward")
            pypto.set_vec_tile_shapes(1, l_tile, d)
            # RMSNorm 反向: [tile_bt, L_MAX, 1] c = sum(gamma * grad_K * V)（必须用 sum，不切规约轴）
            c_fp32 = pypto.sum(gamma_fp32 * grad_k_fp32 * v_tile_fp32, dim=-1, keepdim=True)

            # RMSNorm 反向: [tile_bt, L_MAX, D] grad_V_rms = (gamma * grad_K - V * c / (D * rms²)) / rms
            rms_sq_fp32 = rms_tile_fp32 * rms_tile_fp32
            grad_v_rms_fp32 = (gamma_fp32 * grad_k_fp32 - v_tile_fp32 * c_fp32 / (d * rms_sq_fp32)) / rms_tile_fp32

            # grad_gamma 累加：[1, 1, D] sum(grad_K * V / rms)
            grad_gamma_tile_fp32 = grad_k_fp32 * (v_tile_fp32 / rms_tile_fp32)
            pypto.set_vec_tile_shapes(tile_bt, 1, 16384 // tile_bt)
            grad_gamma_tile_fp32 = pypto.sum(grad_gamma_tile_fp32, dim=0, keepdim=True)
            pypto.set_pass_options(sg_set_scope=1)
            pypto.set_vec_tile_shapes(1, L_MAX, 16384 // L_MAX)
            grad_gamma_tile_fp32 = pypto.sum(grad_gamma_tile_fp32, dim=1, keepdim=True)
            pypto.set_pass_options(sg_set_scope=-1)

            pypto.set_vec_tile_shapes(1, 1, d)
            grad_gamma_acc[:] = grad_gamma_acc + grad_gamma_tile_fp32
        else:
            grad_v_rms_fp32 = grad_k_fp32

        pypto.set_vec_tile_shapes(1, l_tile, d)
        grad_v_fp32 = grad_v_agg_fp32 + grad_v_rms_fp32
        grad_v_tile = pypto.cast(grad_v_fp32, dtype)
        pypto.assemble(grad_v_tile, [bt_idx, 0, 0], grad_v_flat)

    grad_proj_weight[:] = pypto.cast(grad_proj_weight_acc, dtype)
    if enable_rmsnorm:
        grad_gamma[:] = pypto.cast(grad_gamma_acc, dtype)


def ai_infra_block_attn_res_backward(
    grad_h: torch.Tensor,
    blocks: List[torch.Tensor],
    proj_weight: torch.Tensor,
    alpha_cache: torch.Tensor,
    partial_block: Optional[torch.Tensor] = None,
    rmsnorm_gamma: Optional[torch.Tensor] = None,
    rms_cache: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    enable_rmsnorm: bool = True,
):
    """Block Attention Residuals 反向算子调用入口。

    Args:
        grad_h: 上游传递的输出梯度 [B, T, D]
        blocks: 正向使用的已完成的 block 表示列表, 每个 tensor shape 均为 [B, T, D]
        proj_weight: 伪查询投影权重 [D]
        alpha_cache: 正向缓存的 softmax 权重 [B, T, L]
        partial_block: 正向使用的当前 block 部分和 [B, T, D]
        rmsnorm_gamma: RMSNorm 缩放参数 [D]
        rms_cache: 正向缓存的 RMSNorm 分母 [B, T, L, 1]
        scale: softmax 缩放因子, 默认值为 1.0
        enable_rmsnorm: 是否启用 RMSNorm, 默认为 True

    Returns:
        - (grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma)
        - (grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma, 
                grad_rmsnorm_gamma) (enable_rmsnorm为True时)
    """
    n = len(blocks)
    if n == 0:
        raise ValueError("blocks 不能为空")

    has_partial = partial_block is not None

    b, t, d = blocks[0].shape
    l = n + 1 if has_partial else n
    dtype = blocks[0].dtype
    device = blocks[0].device

    if grad_h.shape != (b, t, d):
        raise ValueError(f"grad_h shape must be {(b, t, d)}, but got {tuple(grad_h.shape)}")
    if grad_h.dtype != dtype or grad_h.device != device:
        raise ValueError(f"grad_h dtype/device must be {dtype}/{device}, but got {grad_h.dtype}/{grad_h.device}")

    for i, block in enumerate(blocks):
        if block.shape != (b, t, d):
            raise ValueError(f"blocks[{i}] shape must be {(b, t, d)}, but got {tuple(block.shape)}")
        if block.dtype != dtype or block.device != device:
            raise ValueError(f"blocks[{i}] dtype/device must be {dtype}/{device}, but got {block.dtype}/{block.device}")

    if has_partial:
        if partial_block.shape != (b, t, d):
            raise ValueError(f"partial_block shape must be {(b, t, d)}, but got {tuple(partial_block.shape)}")
        if partial_block.dtype != dtype or partial_block.device != device:
            raise ValueError(
                f"partial_block dtype/device must be {dtype}/{device}, \
                    but got {partial_block.dtype}/{partial_block.device}"
            )

    if proj_weight.shape != (d,):
        raise ValueError(f"proj_weight shape must be {(d,)}, but got {tuple(proj_weight.shape)}")
    if proj_weight.dtype != dtype or proj_weight.device != device:
        raise ValueError(f"proj_weight dtype/device must be {dtype}/{device}, \
                        but got {proj_weight.dtype}/{proj_weight.device}")

    if alpha_cache is None:
        raise ValueError("alpha_cache must be provided and cannot be None")
    if alpha_cache.shape != (b, t, l):
        raise ValueError(f"alpha_cache shape must be {(b, t, l)}, but got {tuple(alpha_cache.shape)}")

    if enable_rmsnorm:
        if rmsnorm_gamma is None:
            raise ValueError("when enable_rmsnorm is True, rmsnorm_gamma must be provided")
        if rmsnorm_gamma.shape != (d,):
            raise ValueError(f"rmsnorm_gamma shape must be {(d,)}, but got {tuple(rmsnorm_gamma.shape)}")
        if rmsnorm_gamma.dtype != dtype or rmsnorm_gamma.device != device:
            raise ValueError(
                f"rmsnorm_gamma dtype/device must be {dtype}/{device}, \
                    but got {rmsnorm_gamma.dtype}/{rmsnorm_gamma.device}"
            )

        if rms_cache is None:
            raise ValueError("when enable_rmsnorm is True, rms_cache must be provided")
        if rms_cache.shape != (b, t, l, 1):
            raise ValueError(f"rms_cache shape must be {(b, t, l, 1)}, but got {tuple(rms_cache.shape)}")


    tensors = blocks + ([partial_block] if has_partial else [])
    v = torch.stack(tensors, dim=2)
    v_flat = v.reshape(b * t, l, d)

    if rmsnorm_gamma is not None:
        gamma_reshaped = rmsnorm_gamma.reshape(1, 1, d)
    else:
        gamma_reshaped = torch.ones(1, 1, d, dtype=dtype, device=device)

    proj_weight_reshaped = proj_weight.reshape(1, 1, d)

    grad_h_3d = grad_h.reshape(b * t, 1, d)

    alpha_cache_3d = alpha_cache.reshape(b * t, 1, l)

    if rms_cache is not None:
        rms_cache_flat = rms_cache.reshape(b * t, l, 1)
    else:
        rms_cache_flat = torch.empty(b * t, l, 1, dtype=dtype, device=device)

    grad_v_flat = torch.zeros(b * t, l, d, dtype=dtype, device=device)
    grad_proj_weight_out = torch.zeros(1, 1, d, dtype=dtype, device=device)
    grad_gamma_out = torch.zeros(1, 1, d, dtype=dtype, device=device)

    ai_infra_block_attn_res_backward_kernel(
        v_flat,
        grad_h_3d,
        rms_cache_flat,
        alpha_cache_3d,
        proj_weight_reshaped,
        gamma_reshaped,
        grad_v_flat,
        grad_proj_weight_out,
        grad_gamma_out,
        scale,
        enable_rmsnorm,
    )

    grad_v = grad_v_flat.reshape(b, t, l, d)
    grad_blocks = [
        grad_v[:, :, i, :].contiguous()
        for i in range(n)
    ]

    if has_partial:
        grad_partial_block = grad_v[:, :, n, :].contiguous()
    else:
        grad_partial_block = None

    grad_proj_weight_final = grad_proj_weight_out.reshape(d)

    if enable_rmsnorm:
        grad_rmsnorm_gamma_final = grad_gamma_out.reshape(d)
        return grad_blocks, grad_partial_block, grad_proj_weight_final, grad_rmsnorm_gamma_final

    return grad_blocks, grad_partial_block, grad_proj_weight_final