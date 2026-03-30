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
Flash Attention Score Implementation with Online Softmax

This module implements Flash Attention using block-wise computation with
online softmax algorithm for numerical stability.
"""

import math
import pypto


NUM_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE_KV = 64
BLOCK_SIZE_Q = 64


@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "cube_l1_reuse_setting": {0: 8},
        "cube_nbuffer_setting": {0: 4},
        "vec_nbuffer_setting": {0: 4},
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 1,
    },
    debug_options={
        "runtime_debug_mode": 0,
    }
)
def flash_attention_score_kernel_with_mask(
    query: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    value: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
    atten_mask: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_BF16),
):
    """
    Flash Attention Score kernel with online softmax (with mask).
    Dynamic axes: batch (dim 0), seq_len_q (dim 2), seq_len_kv (dim 3 for key/value, dim 1 for mask)
    """
    batch_size = query.shape[0]
    seq_len_q = query.shape[2]
    seq_len_kv = key.shape[2]
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
    pypto.set_vec_tile_shapes(64, 512)
    
    num_blocks_kv = (seq_len_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (seq_len_q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    
    for b_idx in pypto.loop(0, batch_size, 1, name="LOOP_B", idx_name="b_idx"):
        for n_idx in pypto.loop(0, NUM_HEADS, 1, name="LOOP_N", idx_name="n_idx"):
            for q_block_idx in pypto.loop(0, num_blocks_q, 1, name="LOOP_Q_BLOCK", idx_name="q_block_idx"):
                q_start = q_block_idx * BLOCK_SIZE_Q
                cur_q_size = pypto.min(BLOCK_SIZE_Q, seq_len_q - q_start)
                
                oi_update = pypto.tensor([BLOCK_SIZE_Q, HEAD_DIM], pypto.DT_FP32, "oi_update")
                li_update = pypto.tensor([BLOCK_SIZE_Q, 1], pypto.DT_FP32, "li_update")
                mi_update = pypto.tensor([BLOCK_SIZE_Q, 1], pypto.DT_FP32, "mi_update")
                
                q_block = pypto.view(query, [1, 1, BLOCK_SIZE_Q, HEAD_DIM], 
                                    [b_idx, n_idx, q_start, 0],
                                    valid_shape=[1, 1, cur_q_size, HEAD_DIM])
                q_block_2d = pypto.reshape(q_block, [BLOCK_SIZE_Q, HEAD_DIM])
                q_block_2d_valid = pypto.view(q_block_2d, [BLOCK_SIZE_Q, HEAD_DIM],
                                              [0, 0],
                                              valid_shape=[cur_q_size, HEAD_DIM])
                
                for kv_block_idx, _ in pypto.loop_unroll(0, num_blocks_kv, 1,
                                                         name="LOOP_KV_BLOCK", 
                                                         idx_name="kv_block_idx",
                                                         unroll_list=[4, 2, 1]):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = pypto.min(BLOCK_SIZE_KV, seq_len_kv - kv_start)
                    
                    k_block = pypto.view(key, [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                                        [b_idx, n_idx, kv_start, 0],
                                        valid_shape=[1, 1, cur_block_size, HEAD_DIM])
                    k_block_2d = pypto.reshape(k_block, [BLOCK_SIZE_KV, HEAD_DIM])
                    k_block_2d_valid = pypto.view(k_block_2d, [BLOCK_SIZE_KV, HEAD_DIM],
                                                  [0, 0],
                                                  valid_shape=[cur_block_size, HEAD_DIM])
                    
                    scores = pypto.matmul(q_block_2d_valid, k_block_2d_valid, pypto.DT_FP32, 
                                         a_trans=False, b_trans=True)
                    scores_scaled = pypto.mul(scores, scale)
                    
                    mask_block = pypto.view(atten_mask, [BLOCK_SIZE_Q, BLOCK_SIZE_KV],
                                           [q_start, kv_start],
                                           valid_shape=[cur_q_size, cur_block_size])
                    valid_mask = pypto.add(mask_block, -1.0)
                    valid_mask = pypto.mul(valid_mask, -1.0)
                    
                    m_ij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                    
                    s_ij_sub_m = pypto.sub(scores_scaled, m_ij)
                    p_ij = pypto.exp(s_ij_sub_m)
                    p_ij = pypto.mul(p_ij, valid_mask)
                    l_ij = pypto.sum(p_ij, dim=-1, keepdim=True)
                    
                    v_block = pypto.view(value, [1, 1, BLOCK_SIZE_KV, HEAD_DIM],
                                        [b_idx, n_idx, kv_start, 0],
                                        valid_shape=[1, 1, cur_block_size, HEAD_DIM])
                    v_block_2d = pypto.reshape(v_block, [BLOCK_SIZE_KV, HEAD_DIM])
                    v_block_2d_valid = pypto.view(v_block_2d, [BLOCK_SIZE_KV, HEAD_DIM],
                                                  [0, 0],
                                                  valid_shape=[cur_block_size, HEAD_DIM])
                    v_block_fp32 = pypto.cast(v_block_2d_valid, pypto.DT_FP32)
                    
                    o_ij = pypto.matmul(p_ij, v_block_fp32, pypto.DT_FP32)
                    
                    if pypto.is_loop_begin(kv_block_idx):
                        if pypto.is_loop_end(kv_block_idx):
                            o_final = pypto.div(o_ij, l_ij)
                            o_final_bf16 = pypto.cast(o_final, pypto.DT_BF16)
                            o_final_4d = pypto.reshape(o_final_bf16, [1, 1, BLOCK_SIZE_Q, HEAD_DIM])
                            output[b_idx: b_idx + 1, n_idx: n_idx + 1, q_start: q_start + BLOCK_SIZE_Q, :] = o_final_4d
                        else:
                            oi_update[:] = o_ij
                        li_update[:] = l_ij
                        mi_update[:] = m_ij
                    else:
                        mi_new = pypto.maximum(mi_update, m_ij)
                        
                        alpha = pypto.exp(pypto.sub(mi_update, mi_new))
                        beta = pypto.exp(pypto.sub(m_ij, mi_new))
                        
                        li_new = pypto.add(
                            pypto.mul(alpha, li_update),
                            pypto.mul(beta, l_ij)
                        )
                        
                        oi_scaled = pypto.mul(oi_update, alpha)
                        o_ij_scaled = pypto.mul(o_ij, beta)
                        oi_new = pypto.add(oi_scaled, o_ij_scaled)
                        
                        if pypto.is_loop_end(kv_block_idx):
                            o_final = pypto.div(oi_new, li_new)
                            o_final_bf16 = pypto.cast(o_final, pypto.DT_BF16)
                            o_final_4d = pypto.reshape(o_final_bf16, [1, 1, BLOCK_SIZE_Q, HEAD_DIM])
                            output[b_idx: b_idx + 1, n_idx: n_idx + 1, q_start: q_start + BLOCK_SIZE_Q, :] = o_final_4d
                        else:
                            oi_update[:] = oi_new
                        li_update[:] = li_new
                        mi_update[:] = mi_new