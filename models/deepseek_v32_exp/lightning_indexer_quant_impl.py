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
Lightning Indexer Quantization Module

This module implements lightning indexer with quantization support for DeepSeek V32.
It is a high-performance tensor computation function based on the PyPTO framework,
primarily used for optimizing index calculations during the decoding phase.

Main Functions:
    - lightning_indexer_decode_compute: JIT-compiled decode version

Example:
    See deepseekv32_lightning_indexer_quant.py for usage examples.
"""
import sys
from pypto.operation import op_wrapper
import pypto
from pypto import pypto_impl
from deepseekv32_lightning_indexer_quant import LightningIndexerConfigs

MAX_LI_S1 = 4
MAX_LI_S2 = 128 * 1024

SHAPE_DIM1 = 1
SHAPE_DIM2 = 2


@op_wrapper
def topk_sort(x, idx_start):
    return pypto_impl.topk_sort(x, idx_start)


@op_wrapper
def topk_merge(x, merge_size):
    return pypto_impl.topk_merge(x, merge_size)


@op_wrapper
def topk_extract(x, k, is_index):
    return pypto_impl.topk_extract(x, k, is_index)


def lightning_indexer_decode_compute(
    idx_query: pypto.tensor,
    idx_query_scale: pypto.tensor,
    idx_key_cache: pypto.tensor,
    idx_key_scale: pypto.tensor,
    idx_weight: pypto.tensor,
    act_seq_key: pypto.tensor,
    block_table: pypto.tensor,
    topk_res: pypto.tensor,
    unroll_list: list,
    configs: LightningIndexerConfigs,
    selected_count: int):

    """Compute lightning indexer with quantization support.
    It obtains the top-k positions corresponding to each token based on a series of operations.
    Args:
        idx_query: Non-contiguous data is not supported, shape (t, n_q, idx_head_dim), dtype INT8. 
        idx_query_scale: It represents the scaling factor for idx_query. shape (t, n_q, idx_head_dim), dtype FP16. 
        idx_key_cache: Non-contiguous data is not supported, shape (t, n_kv, idx_head_dim), dtype INT8. 
        idx_key_scale: It represents the scaling factor for idx_key_cache, shape (t, n_kv, idx_head_dim), dtype FP16. 
        idx_weight: Non-contiguous data is not supported. The data format supports ND, shape (t, n_q), dtype FP16.
        act_seq_key: It represents the number of valid tokens for `key` in different batches. shape (b), dtype INT32.
        block_table: It represents the block mapping table used for KV storage in PageAttention,
                    shape (b, ceilDiv(max(s2), block_size), dtype INT32.
        unroll_list: It represents the multi-level tiling configuration.
        configs: It is a LightningIndexerConfigs configuration structure that represents
                tiling configuration and optimization options.
        selected_count: Required parameter. It represents the number of topk selections, with a default value of 2048.
    """

    # graph fuse/split thresold
    pypto.set_pass_options(mg_copyin_upper_bound=configs.mg_copy_in_upper_bound)
    pypto.set_pass_options(pg_upper_bound=configs.pg_upper_bound)

    # vector graph fuse optimization
    pypto.set_pass_options(vec_nbuffer_mode=configs.vec_merge_mode)
    pypto.set_pass_options(vec_nbuffer_setting=configs.vec_nbuffer_setting)

    # cube graph fuse optimization
    pypto.set_pass_options(cube_l1_reuse_setting=configs.cube_l1_reuse_setting)

    # get tile params from configs
    s1_tile = configs.s1_tile # s1 need to be divided by s1_tile
    topk_tile = configs.topk_tile
    c1_tile = configs.c1_tile
    c2_tile = configs.c2_tile

    # symbolization of params
    t = idx_query.shape[0]
    b = act_seq_key.shape[0]
    block_num = idx_key_cache.shape[0]

    idx_n_heads = idx_query.shape[SHAPE_DIM1]
    index_d = idx_query.shape[SHAPE_DIM2]
    block_size = idx_key_cache.shape[SHAPE_DIM1]
    qk_dtype = idx_query.dtype
    scale_dtype = idx_query_scale.dtype
    w_dtype = idx_weight.dtype

    s1 = t // b # s1 of each batch is equal in decode process
    s1_loop = (s1 + s1_tile - 1) // s1_tile

    xdtype = pypto.DT_FP32
    dxdtype = pypto.DT_INT32
    pad_idx_value = -1
    descending = True
    pad_value = -sys.float_info.max if descending else sys.float_info.max
    length_2k = selected_count
    length_8k = 1024 * 8
    length_32k = 1024 * 32

    query_2d = pypto.tensor([t * idx_n_heads, index_d], qk_dtype, "query_2d")
    q_scale_3d = pypto.tensor([t, 1, idx_n_heads], scale_dtype, "q_scale_3d")
    key_2d = pypto.tensor([block_num * block_size, index_d], qk_dtype, "key_2d")
    k_scale_2d = pypto.tensor([block_num, block_size], scale_dtype, "k_scale_2d")
    weight_3d = pypto.tensor([t, 1, idx_n_heads], w_dtype, "weight_3d")

    # reshape will not generate real datamove
    query_2d = pypto.reshape(idx_query, [t * idx_n_heads, index_d], inplace=True)
    q_scale_3d = pypto.reshape(idx_query_scale, [t, 1, idx_n_heads], inplace=True)
    key_2d = pypto.reshape(idx_key_cache, [block_num * block_size, index_d], inplace=True)
    weight_3d = pypto.reshape(idx_weight, [t, 1, idx_n_heads], inplace=True)
    k_scale_2d = pypto.reshape(idx_key_scale, [block_num, block_size], inplace=True)

    for b_idx in pypto.loop(0, b, 1, name="LI_LOOP_BATCH", idx_name="b_idx"):
        cur_seq = act_seq_key[b_idx]
        cur_block = (cur_seq + block_size - 1) // block_size
        last_seq = cur_seq - (cur_block - 1) * block_size
        # static tensor for rawShape assemble
        max_tensor = pypto.tensor([MAX_LI_S1, MAX_LI_S2], pypto.DT_FP32, "max_tensor")
        for s1_tile_idx in pypto.loop(0, s1_loop, 1, name="LI_LOOP_S1", idx_name="s1_loop"):
            w_scale = pypto.tensor([s1_tile, 1, idx_n_heads], pypto.DT_FP16, "w_scale")
            pypto.set_vec_tile_shapes(s1_tile, 1, idx_n_heads)
            cur_qs = pypto.view(q_scale_3d, [s1_tile, 1, idx_n_heads],
                                [b_idx * s1 + s1_tile * s1_tile_idx, 0, 0])
            cur_w = pypto.view(weight_3d, [s1_tile, 1, idx_n_heads],
                                [b_idx * s1 + s1_tile * s1_tile_idx, 0, 0])
            w_scale = pypto.mul(cur_qs, cur_w) # (s1_tile, 1, idx_n_heads), fp16 * fp16
            q_offset = b_idx * s1 * idx_n_heads + s1_tile_idx * s1_tile * idx_n_heads
            cur_q = pypto.view(query_2d, [s1_tile * idx_n_heads, index_d], [q_offset, 0])
            for bn_idx, unroll_loop in pypto.loop_unroll(
                0, cur_block, 1, name="LOOP_BLOCK_NUM", idx_name="bn_idx", unroll_list=unroll_list,):
                # static unroll into bigger block to reduce tasks
                first_mm_collect = pypto.tensor([s1_tile * idx_n_heads, block_size * unroll_loop],
                                                pypto.DT_FP16, "first_mm_collect")
                for sub_bn_idx in range(unroll_loop):
                    idx_in_block = bn_idx + sub_bn_idx
                    cur_block_idx = block_table[b_idx, idx_in_block]
                    tail_seq = pypto.min(block_size, cur_seq - (idx_in_block * block_size))
                    k_block = pypto.view(key_2d, [block_size, index_d], [cur_block_idx * block_size, 0],
                        valid_shape=[tail_seq, index_d]) # (blockSize, indexD)
                    pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]], [c1_tile[2],
                                                                          c1_tile[3]], [c1_tile[4], c1_tile[5]])
                    # use fixpipe
                    qk_dot = pypto.matmul(cur_q, k_block, pypto.DT_FP16, a_trans=False, b_trans=True,
                                            extend_params=configs.extend_param) # (s1Tile * idxNHeads, blockSize)
                    pypto.assemble(qk_dot, [0, sub_bn_idx * block_size], first_mm_collect)

                pypto.set_vec_tile_shapes(pypto.min(s1_tile * idx_n_heads, block_size), block_size)
                valid_cat_shape = (unroll_loop - 1) * block_size + last_seq
                qk_3d = pypto.reshape(first_mm_collect, [s1_tile, idx_n_heads, unroll_loop * block_size],
                    valid_shape=[s1_tile, idx_n_heads, pypto.min(unroll_loop * block_size, valid_cat_shape)],
                                        inplace=True)
                pypto.set_cube_tile_shapes(
                    [c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                w_qk = pypto.matmul(w_scale, qk_3d, pypto.DT_FP32, a_trans=False, b_trans=False)
                second_mm = pypto.reshape(w_qk, [s1_tile, unroll_loop * block_size],
                    valid_shape=[s1_tile, pypto.min(unroll_loop * block_size, valid_cat_shape)], inplace=True)

                ks_assemble = pypto.tensor([1, unroll_loop * block_size], pypto.DT_FP16, "ks_assemble")
                pypto.set_vec_tile_shapes(1, block_size)

                for idx in range(unroll_loop):
                    cur_block_idx = block_table[b_idx, bn_idx + idx]
                    k_s_block = pypto.view(k_scale_2d, [1, block_size], [cur_block_idx, 0],
                            valid_shape=[1, pypto.min(block_size, cur_seq - (cur_block - 1) * block_size)])
                    pypto.assemble(pypto.clone(k_s_block), [0, idx * block_size], ks_assemble)
                    pypto.set_vec_tile_shapes(1, 16 * block_size)

                k_res = pypto.mul(second_mm, pypto.cast(ks_assemble, pypto.DT_FP32))
                pypto.assemble(k_res, [s1_tile_idx * s1_tile, bn_idx * block_size], max_tensor)

            for s1_idx in pypto.loop(0, s1, 1, name="LOOP_TOPK_S1", idx_name="s1_idx"):
                # TopK Process
                pypto.set_vec_tile_shapes(1, topk_tile)
                casual_offset = s1 - s1_idx - 1
                eff_seq = cur_seq - casual_offset
                src_offset = s1_idx
                dst_offset = b_idx * s1 + s1_idx
                length_is_le2k = eff_seq <= length_2k
                length_is_gt2k = eff_seq > length_2k
                for _ in pypto.loop(length_is_le2k, name="2K_TOPK", idx_name="un_used"):
                    pad_x2k = pypto.tensor([1, length_2k], xdtype, "pad_x2k")
                    pypto.set_pass_options(pg_skip_partition=True)
                    pypto.set_vec_tile_shapes(1, length_2k)
                    eff_2k = pypto.view(max_tensor, [1, length_2k], [src_offset, 0], valid_shape=[1, eff_seq])
                    ax = pypto.view(eff_2k, [1, length_2k], [0, 0], valid_shape=[1, eff_seq])
                    bx = pypto.full([1, length_2k], pad_value, pypto.DT_FP32, valid_shape=[1, length_2k - eff_seq])
                    pypto.assemble(pypto.clone(ax), [0, 0], pad_x2k)
                    pypto.assemble(bx, [0, eff_seq], pad_x2k)
                    pypto.set_pass_options(pg_skip_partition=False)
                    res, _ = topk_sort(pad_x2k, 0)
                    res_idx = topk_extract(res, selected_count, True)
                    pypto.set_vec_tile_shapes(1, 1, selected_count)
                    cur_res = pypto.reshape(pypto.view(res_idx, [1, selected_count], [0, 0],
                                valid_shape=[1, eff_seq]), [1, 1, selected_count],
                                            valid_shape=[1, 1, eff_seq], inplace=True)
                    pypto.assemble(pypto.clone(cur_res), [dst_offset, 0, 0], topk_res)
                    topk_indices_pad = pypto.full([1, 1, selected_count], pad_idx_value, dxdtype,
                                        valid_shape=[1, 1, selected_count - eff_seq])
                    pypto.assemble(topk_indices_pad, [dst_offset, 0, eff_seq], topk_res)
                length_is_le8k = eff_seq <= length_8k
                length_is_gt8k = eff_seq > length_8k
                pad_x8k = pypto.tensor([1, length_8k], xdtype, "pad_x8k")
                pypto.set_vec_tile_shapes(1, topk_tile)

                for _ in pypto.loop(length_is_gt2k * length_is_le8k, name="8K_TOPK", idx_name="unused"):
                    pypto.set_pass_options(pg_skip_partition=True)
                    pypto.set_vec_tile_shapes(1, length_2k)
                    eff_8k = pypto.view(max_tensor, [1, length_8k], [src_offset, 0], valid_shape=[1, eff_seq])
                    ax = pypto.view(eff_8k, [1, length_8k], [0, 0], valid_shape=[1, eff_seq])
                    bx = pypto.full([1, length_8k], pad_value, pypto.DT_FP32, valid_shape=[1, length_8k - eff_seq])
                    pypto.assemble(pypto.clone(ax), [0, 0], pad_x8k)
                    pypto.assemble(bx, [0, eff_seq], pad_x8k)
                    pypto.set_pass_options(pg_skip_partition=False)

                    res, _ = topk_sort(pypto.view(pad_x8k, [1, length_8k], [0, 0]), 0)
                    res_idx = topk_extract(res, selected_count, True)
                    pypto.set_vec_tile_shapes(1, 1, topk_tile)
                    topk_3d = pypto.reshape(res_idx, [1, 1, selected_count])
                    pypto.assemble(pypto.clone(topk_3d), [dst_offset, 0, 0], topk_res)
                    pypto.set_vec_tile_shapes(1, topk_tile)

                # 128K TOPK
                total_size_y1 = MAX_LI_S2 // length_8k * selected_count
                total_size_y2 = MAX_LI_S2 // length_8k * selected_count // length_8k * selected_count
                local_y1 = pypto.tensor([1, total_size_y1 * 2], pypto.DT_FP32, "local_y1")
                local_y2 = pypto.tensor([1, total_size_y2 * 2], pypto.DT_FP32, "local_y2")
                max_num_of_8k = MAX_LI_S2 // length_8k
                num_of_8k = (eff_seq - 1) // length_8k + 1
                valid_size_y1 = num_of_8k * selected_count
                pad_size_y1 = total_size_y1 - valid_size_y1
                num_of_32k = (eff_seq - 1) // length_32k + 1
                valid_size_y2 = num_of_32k * selected_count
                pad_size_y2 = total_size_y2 - valid_size_y2

                pypto.set_vec_tile_shapes(1, topk_tile)
                for _ in pypto.loop(1 * length_is_gt8k * (num_of_8k != max_num_of_8k),
                                    name="128K_PAD_Y1Y2", idx_name="unused"):
                    pypto.assemble(pypto.full([1, total_size_y1 * 2], pad_value, xdtype,
                                            valid_shape=[1, pad_size_y1 * 2]), [0, valid_size_y1 * 2], local_y1)
                    pypto.assemble(pypto.full([1, total_size_y2 * 2], pad_value, xdtype,
                                            valid_shape=[1, pad_size_y2 * 2]), [0, valid_size_y2 * 2], local_y2)

                need_8k_pad_tail = (eff_seq % length_8k) != 0
                num_of_8k_full_block = num_of_8k - need_8k_pad_tail

                for idx1 in pypto.loop(num_of_8k_full_block * length_is_gt8k,
                                       name="128K_TO_32K_FULL_SORT", idx_name="idx1"):
                    cur_in = pypto.view(max_tensor, [1, MAX_LI_S2], [src_offset, 0], valid_shape=[1, eff_seq])
                    ax = pypto.view(cur_in, [1, length_8k], [0, idx1 * length_8k])
                    res, _ = topk_sort(ax, idx1)
                    pypto.assemble(pypto.clone(pypto.view(res, [1, selected_count * 2], [0, 0])),
                                [0, idx1 * selected_count * 2], local_y1)

                for _ in pypto.loop(need_8k_pad_tail * length_is_gt8k,
                                       name="128K_TO_32K_TAIL", idx_name="unused"):
                    x_start_offset = num_of_8k_full_block * length_8k
                    tail_block_length = eff_seq - x_start_offset
                    pypto.set_pass_options(pg_skip_partition=True)
                    pypto.set_vec_tile_shapes(1, topk_tile)
                    cur_in = pypto.view(max_tensor, [1, MAX_LI_S2], [src_offset, 0], valid_shape=[1, eff_seq])
                    ax = pypto.view(cur_in, [1, length_8k], [0, x_start_offset],
                                    valid_shape=[1, tail_block_length])
                    bx = pypto.full([1, length_8k], pad_value, pypto.DT_FP32,
                        valid_shape=[1, length_8k - tail_block_length])
                    pypto.assemble(pypto.clone(ax), [0, 0], pad_x8k)
                    pypto.assemble(bx, [0, tail_block_length], pad_x8k)
                    pypto.set_pass_options(pg_skip_partition=False)
                    for _ in pypto.loop(0, 1, 1, name="128K_TO_32K_TAIL_SORT", idx_name="un_used0"):
                        res, _ = topk_sort(pypto.view(pad_x8k, [1, length_8k], [0, 0]), num_of_8k_full_block)
                        pypto.assemble(pypto.clone(pypto.view(res, [1, selected_count * 2], [0, 0])),
                                            [0, num_of_8k_full_block * selected_count * 2], local_y1)

                for idx2 in pypto.loop(num_of_32k * length_is_gt8k, name="32K_TO_8K_MERGE", idx_name="idx2"):
                    res = topk_merge(pypto.view(local_y1, [1, length_8k * 2],
                                                [0, idx2 * length_8k * 2]), selected_count)
                    pypto.assemble(pypto.clone(pypto.view(res, [1, selected_count * 2], [0, 0])),
                                [0, idx2 * selected_count * 2], local_y2)

                pypto.set_vec_tile_shapes(1, topk_tile)
                for _ in pypto.loop(1 * length_is_gt8k, name="8K_TO_2K_MERGE", idx_name="unused"):
                    res = topk_merge(pypto.view(local_y2, [1, length_8k * 2], [0, 0]), selected_count)
                    res_idx = topk_extract(res, selected_count, True)
                    pypto.set_vec_tile_shapes(1, 1, topk_tile)
                    topk_3d = pypto.reshape(res_idx, [1, 1, selected_count])
                    pypto.assemble(pypto.clone(topk_3d), [dst_offset, 0, 0], topk_res)
                    pypto.set_vec_tile_shapes(1, topk_tile)


def lightning_indexer_decode(
    idx_n_heads, idx_head_dim, block_size, block_num,
    unroll_list, configs, selected_count=2048):
    """Factory function for Lightning Indexer Decode kernel.

    Factory Parameters (fixed dimensions):
        idx_n_heads: Number of index attention heads (n_q)
        idx_head_dim: Dimension of each index head
        block_size: Size of each block in PageAttention
        block_num: Total number of blocks in KV cache
        unroll_list: Multi-level tiling configuration
        configs: LightningIndexerConfigs configuration
        selected_count: Number of topk selections (default: 2048)

    Returns:
        Compiled kernel function for lightning indexer decode computation.
    """

    # Define dynamic dimensions
    t = pypto.frontend.dynamic("t")  # Total tokens = b * s1
    b = pypto.frontend.dynamic("b")  # Batch size
    max_blocks = pypto.frontend.dynamic("max_blocks")

    # Assemble tensor shapes
    idx_query_shape = (t, idx_n_heads, idx_head_dim)
    idx_query_scale_shape = (t, idx_n_heads)
    idx_key_cache_shape = (block_num, block_size, 1, idx_head_dim)
    idx_key_scale_shape = (block_num, block_size, 1)
    idx_weight_shape = (t, idx_n_heads)
    act_seq_key_shape = (b,)
    block_table_shape = (b, max_blocks)
    topk_res_shape = (t, 1, selected_count)

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_inner_memory": 8192,
            "stitch_function_outcast_memory": 4096,
            "stitch_function_num_initial": 128,
            "device_sched_mode": 1
        }
    )
    def lightning_indexer_decode_kernel(
        idx_query: pypto.Tensor(idx_query_shape, pypto.DT_INT8),
        idx_query_scale: pypto.Tensor(idx_query_scale_shape, pypto.DT_FP16),
        idx_key_cache: pypto.Tensor(idx_key_cache_shape, pypto.DT_INT8),
        idx_key_scale: pypto.Tensor(idx_key_scale_shape, pypto.DT_FP16),
        idx_weight: pypto.Tensor(idx_weight_shape, pypto.DT_FP16),
        act_seq_key: pypto.Tensor(act_seq_key_shape, pypto.DT_INT32),
        block_table: pypto.Tensor(block_table_shape, pypto.DT_INT32),
    ) -> (
        pypto.Tensor(topk_res_shape, pypto.DT_INT32),
    ):
        """JIT-compiled Lightning Indexer for decode phase.

        Args:
            idx_query: (t, idx_n_heads, idx_head_dim), dtype INT8
                Query indices for lightning indexing
            idx_query_scale: (t, idx_n_heads, idx_head_dim), dtype FP16
                Quantization scale for idx_query
            idx_key_cache: (block_num, block_size, 1, idx_head_dim), dtype INT8
                Key cache in PageAttention format
            idx_key_scale: (block_num, block_size, 1, idx_head_dim), dtype FP16
                Quantization scale for idx_key_cache
            idx_weight: (t, idx_n_heads), dtype FP16
                Attention weights for indexing
            act_seq_key: (b,), dtype INT32
                Actual sequence length per batch
            block_table: (b, max_blocks), dtype INT32
                Block mapping table for PageAttention

        Returns:
            topk_res: (t, 1, selected_count), dtype INT32
                TopK indices for sparse attention
        """
        # Create output tensor
        topk_res = pypto.Tensor(topk_res_shape, pypto.DT_INT32)

        # Call original compute function
        lightning_indexer_decode_compute(
            idx_query,
            idx_query_scale,
            idx_key_cache,
            idx_key_scale,
            idx_weight,
            act_seq_key,
            block_table,
            topk_res,
            unroll_list,
            configs,
            selected_count
        )

        return topk_res

    return lightning_indexer_decode_kernel