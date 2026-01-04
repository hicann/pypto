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
import sys
from dataclasses import dataclass, field
from typing import List, Set, Optional
import logging
import pytest
import pypto

SHAPE_DIM0 = 0
SHAPE_DIM1 = 1
SHAPE_DIM2 = 2
SHAPE_DIM3 = 3

NUM_NEG1 = -1
NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_100 = 100
NUM_128 = 128
NUM_1024 = 1024
NUM_1127 = 1127
NUM_2048 = 2048
NUM_4096 = 4096
NUM_8192 = 8192
AVOID_FP32_TO_FP16_OVERFLOW_SCALE = 1.0 / 2048.0


@dataclass
class LightningIndexerTileConfig:
    weight_tile: List[int]
    c1_tile: List[List[int]]
    v1_tile: List[int]
    topk_tile: List[int]
    adds_tile: List[int]


@dataclass
class LightningIndexerParams:
    b: int
    s1: int
    index_n1: int
    qk_nope: int
    qk_rope: int
    n2: int
    block_size: int
    block_num: int
    selected_count: int
    is_quant: bool = False


@dataclass
class LightningIndexerInputs:
    query: pypto.Tensor
    key: pypto.Tensor
    weights: pypto.Tensor
    act_seq_key: pypto.Tensor
    block_table: pypto.Tensor
    topk_res: pypto.Tensor
    q_scale: Optional[pypto.Tensor]
    k_scale: Optional[pypto.Tensor]
    tmp_out: Optional[pypto.Tensor]
    topk_value: Optional[pypto.Tensor]
    tile_config: LightningIndexerTileConfig
    unroll_list: Set[int]
    params: LightningIndexerParams


def lightning_indexer_topk_impl(args: LightningIndexerInputs):
    query = args.query
    key = args.key
    weights = args.weights
    act_seq_key = args.act_seq_key
    block_table = args.block_table
    topk_res = args.topk_res
    q_scale = args.q_scale
    k_scale = args.k_scale
    tmp_out = args.tmp_out
    topk_value = args.topk_value
    tile_config = args.tile_config
    unroll_list = args.unroll_list
    params = args.params
    is_quant = params.is_quant
    selected_count = params.selected_count

    b = query.shape[SHAPE_DIM0]
    s1 = query.shape[SHAPE_DIM1]
    block_num = key.shape[SHAPE_DIM0]

    index_n1 = query.shape[SHAPE_DIM2]
    index_d = query.shape[SHAPE_DIM3]
    block_size = key.shape[SHAPE_DIM1]
    n2 = key.shape[SHAPE_DIM2]

    qk_dtype = query.dtype
    scale_dtype = q_scale.dtype if is_quant else pypto.DT_FP16
    w_dtype = weights.dtype

    group = index_n1 // n2

    c1_tile = tile_config.c1_tile
    max_batch = NUM_128
    max_s1 = NUM_4
    max_n2 = NUM_1
    max_s2 = NUM_128 * NUM_1024

    query_2d = pypto.tensor([b * s1 * index_n1, index_d], qk_dtype, "query2D")
    key_2d = pypto.tensor([block_num * block_size, n2 * index_d], qk_dtype, "key2D")
    q_scale_2d = pypto.tensor([b * s1 * index_n1, 1], scale_dtype, "qScale2D")
    k_scale_2d = pypto.tensor([block_num * block_size, n2], scale_dtype, "kScale2D")
    weight_2d = pypto.tensor([b * s1 * index_n1, 1], w_dtype, "weight2D")
    local_sum = pypto.tensor(
        [max_batch * max_s1 * max_n2, max_s2],
        pypto.DT_FP32,
        "localSum",
    )

    for _ in pypto.loop(0, 1, 1, name="INPUT_4D_2_2D", idx_name="unUsedIdx"):
        query_2d[:] = pypto.reshape(query, [b * s1 * index_n1, index_d], inplace=True)
        key_2d[:] = pypto.reshape(
            key, [block_num * block_size, n2 * index_d], inplace=True
        )
        weight_2d[:] = pypto.reshape(weights, [b * s1 * index_n1, 1], inplace=True)
        if is_quant:
            q_scale_2d[:] = pypto.reshape(q_scale, [b * s1 * index_n1, 1], inplace=True)
            k_scale_2d[:] = pypto.reshape(
                k_scale, [block_num * block_size, n2], inplace=True
            )

    for b_idx in pypto.loop(0, b, 1, name="INDEX_LOOP_BATCH", idx_name="bIdx"):
        cur_seq = act_seq_key[b_idx]
        for s1_idx in pypto.loop(0, s1, 1, name="INDEX_LOOP_S1", idx_name="s1Idx"):
            causal_offset = s1 - s1_idx - 1
            eff_seq = cur_seq - causal_offset
            act_block = (eff_seq + block_size - 1) // block_size
            for n2_idx in pypto.loop(
                0, n2, 1, name="INDEX_LOOP_N2", idx_name="n2Idx"
            ):
                bs1n2_offset = b_idx * s1 * n2 + s1_idx * n2 + n2_idx
                q_offset = (
                    b_idx * s1 * index_n1
                    + s1_idx * index_n1
                    + n2_idx * group
                )

                def unrolling_process(
                    unroll_length: int,
                    first_block_idx: pypto.symbolic_scalar,
                    b_idx,
                    s1_idx,
                    n2_idx,
                    eff_seq,
                    bs1n2_offset,
                    q_offset,
                ):
                    cur_q = pypto.view(
                        query_2d, [group, index_d], [q_offset, 0]
                    )

                    concat_srcs = []

                    for sub_block_idx in range(unroll_length):
                        block_idx = first_block_idx + sub_block_idx
                        cur_block_idx = block_table[b_idx, block_idx]

                        cur_k = pypto.view(
                            key_2d,
                            [block_size, index_d],
                            [cur_block_idx * block_size, n2_idx * index_d],
                            valid_shape=[
                                pypto.min(
                                    block_size,
                                    eff_seq - (block_idx * block_size),
                                ),
                                index_d,
                            ],
                        )

                        pypto.set_cube_tile_shapes(
                            c1_tile[0], c1_tile[1], c1_tile[2], False
                        )

                        mm_res = pypto.matmul(
                            cur_q,
                            cur_k,
                            pypto.DT_FP32,
                            a_trans=False,
                            b_trans=True,
                        )
                        concat_srcs.append(mm_res)

                    pypto.set_vec_tile_shapes(*tile_config.weight_tile)

                    cur_w = pypto.view(weight_2d, [group, 1], [q_offset, 0])
                    w_b32 = pypto.cast(cur_w, pypto.DT_FP32)

                    mm_res_cat = pypto.concat(concat_srcs, -1)

                    pypto.set_vec_tile_shapes(*tile_config.v1_tile)

                    relu_res = pypto.maximum(mm_res_cat, 0.0)
                    mul_res = relu_res * w_b32
                    sum_res = pypto.sum(mul_res, 0)

                    pypto.assemble(
                        sum_res,
                        [bs1n2_offset, first_block_idx * block_size],
                        local_sum,
                    )
                    if tmp_out is not None:
                        pypto.assemble(
                            sum_res,
                            [bs1n2_offset, first_block_idx * block_size],
                            tmp_out,
                        )

                def unrolling_process_quant(
                    unroll_length: int,
                    first_block_idx: pypto.symbolic_scalar,
                    b_idx,
                    s1_idx,
                    n2_idx,
                    eff_seq,
                    bs1n2_offset,
                    q_offset,
                ):
                    cur_q = pypto.view(
                        query_2d, [group, index_d], [q_offset, 0]
                    )
                    cur_q_scale = pypto.view(
                        q_scale_2d, [group, 1], [q_offset, 0]
                    )

                    mm_res_quant_concat_srcs = []
                    k_scale_concat_srcs = []

                    for sub_block_idx in range(unroll_length):
                        block_idx = first_block_idx + sub_block_idx
                        cur_block_idx = block_table[b_idx, block_idx]

                        cur_k = pypto.view(
                            key_2d,
                            [block_size, index_d],
                            [cur_block_idx * block_size, n2_idx * index_d],
                            valid_shape=[
                                pypto.min(
                                    block_size,
                                    eff_seq - (block_idx * block_size),
                                ),
                                index_d,
                            ],
                        )

                        pypto.set_cube_tile_shapes(
                            c1_tile[0], c1_tile[1], c1_tile[2], False
                        )

                        mm_res = pypto.matmul(
                            cur_q,
                            cur_k,
                            pypto.DT_INT32,
                            a_trans=False,
                            b_trans=True,
                        )
                        mm_res_quant_concat_srcs.append(mm_res)

                        cur_k_scale = pypto.view(
                            k_scale_2d,
                            [block_size, 1],
                            [cur_block_idx * block_size, n2_idx],
                            valid_shape=[
                                pypto.min(
                                    block_size,
                                    eff_seq - (block_idx * block_size),
                                ),
                                1,
                            ],
                        )
                        k_scale_concat_srcs.append(cur_k_scale)

                    pypto.set_vec_tile_shapes(*tile_config.weight_tile)

                    cur_w = pypto.view(weight_2d, [group, 1], [q_offset, 0])
                    w_f16 = pypto.cast(cur_w, pypto.DT_FP16)

                    pypto.set_vec_tile_shapes(*tile_config.v1_tile)

                    cur_k_scale = pypto.concat(k_scale_concat_srcs, 0)
                    mm_res_i32 = pypto.concat(mm_res_quant_concat_srcs, -1)
                    mm_res_fp32 = (
                        pypto.cast(mm_res_i32, pypto.DT_FP32)
                        * AVOID_FP32_TO_FP16_OVERFLOW_SCALE
                    )
                    mm_res_fp16 = pypto.cast(
                        mm_res_fp32, pypto.DT_FP16
                    )
                    mm_res_dequant = (
                        mm_res_fp16
                        * cur_q_scale
                        * pypto.transpose(cur_k_scale, 0, 1)
                    )
                    relu_res = pypto.maximum(mm_res_dequant, 0.0)
                    mul_res = relu_res * w_f16

                    sum_res = pypto.sum(
                        pypto.cast(mul_res, pypto.DT_FP32),
                        0,
                        True
                    )

                    pypto.assemble(
                        sum_res,
                        [bs1n2_offset, first_block_idx * block_size],
                        local_sum,
                    )
                    if tmp_out is not None:
                        pypto.assemble(
                            sum_res,
                            [bs1n2_offset, first_block_idx * block_size],
                            tmp_out,
                        )

                for loop_block_idx, unroll_length in pypto.loop_unroll(
                    0,
                    act_block,
                    1,
                    name="INDEX_LOOP_MATMUL",
                    idx_name="loopBlockIdx",
                    unroll_list=unroll_list,
                ):
                    if is_quant:
                        unrolling_process_quant(
                            unroll_length,
                            loop_block_idx,
                            b_idx,
                            s1_idx,
                            n2_idx,
                            eff_seq,
                            bs1n2_offset,
                            q_offset,
                        )
                    else:
                        unrolling_process(
                            unroll_length,
                            loop_block_idx,
                            b_idx,
                            s1_idx,
                            n2_idx,
                            eff_seq,
                            bs1n2_offset,
                            q_offset,
                        )

    assert selected_count == NUM_2048

    x_dtype = local_sum.dtype
    idx_dtype = topk_res.dtype
    pad_idx_value = NUM_NEG1
    tile_size = NUM_8192
    descending = True
    pad_value = -sys.float_info.max if descending else sys.float_info.max

    length_2k = selected_count
    length_8k = NUM_1024 * NUM_8
    length_64k = NUM_1024 * NUM_64
    length_128k = max_s2

    pypto.set_vec_tile_shapes(1, tile_size)

    for bs1n2_offset in pypto.loop(
        0,
        b * s1 * n2,
        1,
        name="INDEX_LOOP_TOPK_bs1n2Offset",
        idx_name="bs1n2Offset",
    ):

        def _inside_bs1n2(bs1n2_offset):
            b_idx = bs1n2_offset // (s1 * n2)
            s1_idx = (bs1n2_offset % (s1 * n2)) // n2
            n2_idx = bs1n2_offset % n2

            cur_seq = act_seq_key[b_idx]
            causal_offset = s1 - s1_idx - 1
            eff_seq = cur_seq - causal_offset

            length_is_le2k = eff_seq <= length_2k
            length_is_gt2k = eff_seq > length_2k

            pad_x_2k = pypto.tensor(
                [max_batch * max_s1 * max_n2, length_2k],
                x_dtype,
                "padX2K",
            )
            pypto.set_vec_tile_shapes(1, tile_size)

            for unused in pypto.loop(
                0, length_is_le2k, 1, name="2K_LOOP", idx_name="unused"
            ):

                def _inside_2k(unused):
                    pypto.set_pass_options(pg_skip_partition=True)

                    for unused1 in pypto.loop(0, 1, 1, name="2K_PAD", idx_name="unused1"):

                        def _inside_2k_pad(unused1):
                            pypto.set_vec_tile_shapes(1, length_2k)

                            eff_sum_res = pypto.view(
                                local_sum,
                                [1, length_2k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pypto.view(
                                eff_sum_res,
                                [1, length_2k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pypto.full(
                                [1, length_2k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_2k - eff_seq,
                                ],
                            )

                            pypto.assemble(
                                pypto.clone(ax),
                                [bs1n2_offset, 0],
                                pad_x_2k,
                            )
                            pypto.assemble(
                                bx,
                                [bs1n2_offset, eff_seq],
                                pad_x_2k,
                            )

                        _inside_2k_pad(unused1)

                    pypto.set_pass_options(pg_skip_partition=False)

                    for unused2 in pypto.loop(
                        0, 1, 1, name="2K_TOPK", idx_name="unused2"
                    ):

                        def _inside_2k_topk(unused2):
                            res, res_idx = pypto.topk(
                                pypto.view(
                                    pad_x_2k,
                                    [1, length_2k],
                                    [bs1n2_offset, 0],
                                ),
                                selected_count,
                                1,
                            )
                            pypto.set_vec_tile_shapes(*tile_config.adds_tile)

                            topk_4d = pypto.reshape(
                                pypto.view(
                                    res_idx,
                                    [1, selected_count],
                                    [0, 0],
                                    valid_shape=[1, eff_seq],
                                ),
                                [1, 1, 1, selected_count],
                                valid_shape=[1, 1, 1, eff_seq],
                            )
                            pypto.assemble(
                                pypto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            topk_indices_pad = pypto.full(
                                [1, 1, 1, selected_count],
                                pad_idx_value,
                                idx_dtype,
                                valid_shape=[
                                    1,
                                    1,
                                    1,
                                    selected_count - eff_seq,
                                ],
                            )
                            pypto.assemble(
                                topk_indices_pad,
                                [b_idx, s1_idx, n2_idx, eff_seq],
                                topk_res,
                            )

                            if topk_value is not None:
                                topk_4d_value = pypto.reshape(
                                    pypto.view(
                                        res,
                                        [1, selected_count],
                                        [0, 0],
                                        valid_shape=[1, eff_seq],
                                    ),
                                    [1, 1, 1, selected_count],
                                    valid_shape=[
                                        1,
                                        1,
                                        1,
                                        eff_seq,
                                    ],
                                )
                                pypto.assemble(
                                    pypto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )
                                topk_value_pad = pypto.full(
                                    [1, 1, 1, selected_count],
                                    pad_value,
                                    pypto.DT_FP32,
                                    valid_shape=[
                                        1,
                                        1,
                                        1,
                                        selected_count - eff_seq,
                                    ],
                                )
                                pypto.assemble(
                                    topk_value_pad,
                                    [
                                        b_idx,
                                        s1_idx,
                                        n2_idx,
                                        eff_seq,
                                    ],
                                    topk_value,
                                )

                            pypto.set_vec_tile_shapes(1, tile_size)

                        _inside_2k_topk(unused2)

                _inside_2k(unused)

            length_is_le8k = eff_seq <= length_8k
            length_is_gt8k = eff_seq > length_8k

            pad_x_8k = pypto.tensor(
                [max_batch * max_s1 * max_n2, length_8k], x_dtype, "padX8K"
            )

            for unused in pypto.loop(
                0,
                length_is_gt2k * length_is_le8k,
                1,
                name="8K_LOOP",
                idx_name="unused",
            ):

                def _inside_8k(unused):
                    for unused0 in pypto.loop(0, 1, 1, name="8K_PAD", idx_name="unused0"):

                        def _inside_8k_pad(unused0):
                            pypto.set_vec_tile_shapes(1, tile_size)

                            eff_sum_res = pypto.view(
                                local_sum,
                                [1, length_8k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pypto.view(
                                eff_sum_res,
                                [1, length_8k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pypto.full(
                                [1, length_8k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_8k - eff_seq,
                                ],
                            )
                            pypto.assemble(pypto.clone(ax), [bs1n2_offset, 0], pad_x_8k)
                            pypto.assemble(bx, [bs1n2_offset, eff_seq], pad_x_8k)

                        _inside_8k_pad(unused0)

                    pypto.set_vec_tile_shapes(1, tile_size)

                    for unused1 in pypto.loop(
                        0, 1, 1, name="8K_TOPK", idx_name="unused1"
                    ):

                        def _inside_8k_topk(unused1):
                            res, res_idx = pypto.topk(
                                pypto.view(pad_x_8k, [1, length_8k], [bs1n2_offset, 0]),
                                selected_count,
                                1,
                            )
                            pypto.set_vec_tile_shapes(*tile_config.adds_tile)

                            topk_4d = pypto.reshape(
                                res_idx,
                                [1, 1, 1, selected_count],
                            )
                            pypto.assemble(
                                pypto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            if topk_value is not None:
                                pypto.set_vec_tile_shapes(*tile_config.adds_tile)
                                topk_4d_value = pypto.reshape(
                                    res, [1, 1, 1, selected_count]
                                )
                                pypto.assemble(
                                    pypto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )

                            pypto.set_vec_tile_shapes(1, tile_size)

                        _inside_8k_topk(unused1)

                _inside_8k(unused)

            length_is_le64k = eff_seq <= length_64k
            length_is_gt64k = eff_seq > length_64k

            pad_x_64k = pypto.tensor(
                [max_batch * max_s1 * max_n2, length_64k], x_dtype, "padX64K"
            )

            for unused in pypto.loop(
                0,
                length_is_gt8k * length_is_le64k,
                1,
                name="64K_LOOP",
                idx_name="unused",
            ):

                def _inside_64k(unused):
                    for unused0 in pypto.loop(
                        0, 1, 1, name="64K_PAD", idx_name="unused0"
                    ):

                        def _inside_64k_pad(unused0):
                            pypto.set_vec_tile_shapes(1, tile_size)

                            eff_sum_res = pypto.view(
                                local_sum,
                                [1, length_64k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pypto.view(
                                eff_sum_res,
                                [1, length_64k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pypto.full(
                                [1, length_64k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_64k - eff_seq,
                                ],
                            )
                            pypto.assemble(pypto.clone(ax), [bs1n2_offset, 0], pad_x_64k)
                            pypto.assemble(bx, [bs1n2_offset, eff_seq], pad_x_64k)

                        _inside_64k_pad(unused0)

                    pypto.set_vec_tile_shapes(1, tile_size)

                    for unused1 in pypto.loop(
                        0, 1, 1, name="64K_TOPK", idx_name="unused1"
                    ):

                        def _inside_64k_topk(unused1):
                            res, res_idx = pypto.topk(
                                pypto.view(pad_x_64k, [1, length_64k], [bs1n2_offset, 0]),
                                selected_count,
                                1,
                            )
                            pypto.set_vec_tile_shapes(*tile_config.adds_tile)
                            topk_4d = pypto.reshape(res_idx, [1, 1, 1, selected_count])
                            pypto.assemble(
                                pypto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            if topk_value is not None:
                                pypto.set_vec_tile_shapes(*tile_config.adds_tile)
                                topk_4d_value = pypto.reshape(
                                    res, [1, 1, 1, selected_count]
                                )
                                pypto.assemble(
                                    pypto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )

                            pypto.set_vec_tile_shapes(1, tile_size)

                        _inside_64k_topk(unused1)

                _inside_64k(unused)

            pad_x_128k = pypto.tensor(
                [max_batch * max_s1 * max_n2, length_128k], x_dtype, "padX128K"
            )

            for unused in pypto.loop(
                0, length_is_gt64k, 1, name="128K_LOOP", idx_name="unused"
            ):

                def _inside_128k(unused):
                    for unused0 in pypto.loop(
                        0, 1, 1, name="128K_PAD", idx_name="unused0"
                    ):

                        def _inside_128k_pad(unused0):
                            pypto.set_vec_tile_shapes(1, tile_size)

                            eff_sum_res = pypto.view(
                                local_sum,
                                [1, length_128k],
                                [bs1n2_offset, 0],
                                valid_shape=[1, eff_seq],
                            )
                            ax = pypto.view(
                                eff_sum_res,
                                [1, length_128k],
                                [0, 0],
                                valid_shape=[1, eff_seq],
                            )
                            bx = pypto.full(
                                [1, length_128k],
                                pad_value,
                                x_dtype,
                                valid_shape=[
                                    1,
                                    length_128k - eff_seq,
                                ],
                            )
                            pypto.assemble(pypto.clone(ax), [bs1n2_offset, 0], pad_x_128k)
                            pypto.assemble(bx, [bs1n2_offset, eff_seq], pad_x_128k)

                        _inside_128k_pad(unused0)

                    pypto.set_vec_tile_shapes(1, tile_size)

                    for unused1 in pypto.loop(
                        0, 1, 1, name="128K_TOPK", idx_name="unused1"
                    ):

                        def _inside_128k_topk(unused1):
                            res, res_idx = pypto.topk(
                                pypto.view(
                                    pad_x_128k, [1, length_128k], [bs1n2_offset, 0]
                                ),
                                selected_count,
                                1,
                            )
                            pypto.set_vec_tile_shapes(*tile_config.adds_tile)
                            topk_4d = pypto.reshape(res_idx, [1, 1, 1, selected_count])
                            pypto.assemble(
                                pypto.clone(topk_4d),
                                [b_idx, s1_idx, n2_idx, 0],
                                topk_res,
                            )

                            if topk_value is not None:
                                pypto.set_vec_tile_shapes(*tile_config.adds_tile)
                                topk_4d_value = pypto.reshape(
                                    res, [1, 1, 1, selected_count]
                                )
                                pypto.assemble(
                                    pypto.clone(topk_4d_value),
                                    [b_idx, s1_idx, n2_idx, 0],
                                    topk_value,
                                )

                            pypto.set_vec_tile_shapes(1, tile_size)

                        _inside_128k_topk(unused1)

                _inside_128k(unused)

        _inside_bs1n2(bs1n2_offset)


def lightning_indexer_topk_inner(args: LightningIndexerInputs):
    input_tensors = [
        args.query,
        args.key,
        args.weights,
        args.act_seq_key,
        args.block_table,
    ]
    if args.params.is_quant:
        input_tensors += [args.q_scale, args.k_scale]

    output_tensors = [args.topk_res]
    if args.tmp_out is not None:
        output_tensors.append(args.tmp_out)
    if args.topk_value is not None:
        output_tensors.append(args.topk_value)

    with pypto.function("LightningIndexerTopkInner", *input_tensors, *output_tensors):

        def inside_main_function():
            lightning_indexer_topk_impl(args)

        inside_main_function()


@dataclass
class LightningIndexerBuildConfig:
    b: int = NUM_4
    s1: int = NUM_2
    index_n1: int = NUM_64
    qk_nope: int = NUM_128
    qk_rope: int = NUM_0
    n2: int = NUM_1
    block_size: int = NUM_128
    block_num: int = NUM_1127
    selected_count: int = NUM_2048
    is_quant: bool = True
    c1_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_64, NUM_64],
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
        ]
    )
    v1_tile: List[int] = field(default_factory=lambda: [NUM_64, NUM_128])
    topk_tile: List[int] = field(default_factory=lambda: [NUM_1, NUM_4096])
    adds_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_1, NUM_1, NUM_4096]
    )


def setup_lightning_indexer_topk_config():
    pypto.set_pass_options(mg_copyin_upper_bound=NUM_100 * NUM_1024 * NUM_1024,
                         pg_lower_bound=NUM_1024,
                         pg_upper_bound=NUM_1024 * NUM_1024,
                         cube_l1_reuse_mode=NUM_32,
                         pg_parallel_lower_bound=NUM_2,
                         vec_nbuffer_mode=NUM_2,
                         vec_nbuffer_setting={NUM_NEG1: NUM_16})
    pypto.set_runtime_options(device_sched_mode=NUM_3,
                            stitch_function_inner_memory=NUM_128,
                            stitch_function_outcast_memory=NUM_128)


def build_lightning_indexer_topk_args(
    cfg: LightningIndexerBuildConfig = LightningIndexerBuildConfig(),
):
    d_bf16 = pypto.DT_FP16
    d_i32 = pypto.DT_INT32
    d_int8 = pypto.DT_INT8
    d_f16 = pypto.DT_FP16

    index_d = cfg.qk_nope + cfg.qk_rope
    max_block_num = NUM_1024

    if cfg.is_quant:
        qk_dtype = d_int8
        scale_dtype = d_f16
    else:
        qk_dtype = d_bf16
        scale_dtype = d_f16

    query = pypto.tensor(
        [cfg.b, cfg.s1, cfg.index_n1, index_d],
        qk_dtype,
        "query",
    )

    key = pypto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, index_d],
        qk_dtype,
        "key",
    )

    weights = pypto.tensor(
        [cfg.b, cfg.s1, cfg.index_n1],
        d_bf16,
        "weights",
    )

    act_seq_key = pypto.tensor(
        [cfg.b],
        d_i32,
        "actSeqKey",
    )

    block_table = pypto.tensor(
        [cfg.b, max_block_num],
        d_i32,
        "blockTable",
    )

    topk_res = pypto.tensor(
        [cfg.b, cfg.s1, cfg.n2, cfg.selected_count],
        d_i32,
        "topkRes",
    )

    q_scale = (
        pypto.tensor(
            [cfg.b, cfg.s1, cfg.index_n1, 1],
            scale_dtype,
            "qScale",
        )
        if cfg.is_quant
        else None
    )
    k_scale = (
        pypto.tensor(
            [cfg.block_num, cfg.block_size, cfg.n2, 1],
            scale_dtype,
            "kScale",
        )
        if cfg.is_quant
        else None
    )

    tmp_out = None
    topk_value = None

    tile_cfg = LightningIndexerTileConfig(
        weight_tile=[NUM_64, NUM_128],
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        topk_tile=cfg.topk_tile,
        adds_tile=cfg.adds_tile,
    )

    unroll_list: List[int] = [1, 2, 4, 8, 16, 32, 64]

    params = LightningIndexerParams(
        b=cfg.b,
        s1=cfg.s1,
        index_n1=cfg.index_n1,
        qk_nope=cfg.qk_nope,
        qk_rope=cfg.qk_rope,
        n2=cfg.n2,
        block_size=cfg.block_size,
        block_num=cfg.block_num,
        selected_count=cfg.selected_count,
        is_quant=cfg.is_quant,
    )

    args = LightningIndexerInputs(
        query=query,
        key=key,
        weights=weights,
        act_seq_key=act_seq_key,
        block_table=block_table,
        topk_res=topk_res,
        q_scale=q_scale,
        k_scale=k_scale,
        tmp_out=tmp_out,
        topk_value=topk_value,
        tile_config=tile_cfg,
        unroll_list=unroll_list,
        params=params,
    )

    meta = {
        "B": cfg.b,
        "S1": cfg.s1,
        "indexN1": cfg.index_n1,
        "indexD": index_d,
        "N2": cfg.n2,
        "blockSize": cfg.block_size,
        "blockNum": cfg.block_num,
        "maxBlockNum": max_block_num,
        "selectedCount": cfg.selected_count,
        "isQuant": cfg.is_quant,
        "dims": {
            "query": [cfg.b, cfg.s1, cfg.index_n1, index_d],
            "key": [cfg.block_num, cfg.block_size, cfg.n2, index_d],
            "weights": [cfg.b, cfg.s1, cfg.index_n1],
            "actSeqKey": [cfg.b],
            "blockTable": [cfg.b, max_block_num],
            "topkRes": [cfg.b, cfg.s1, cfg.n2, cfg.selected_count],
            "qScale": ([cfg.b, cfg.s1, cfg.index_n1, 1] if cfg.is_quant else None),
            "kScale": (
                [cfg.block_num, cfg.block_size, cfg.n2, 1] if cfg.is_quant else None
            ),
        },
        "tiles": {
            "weightTile": tile_cfg.weight_tile,
            "c1Tile": tile_cfg.c1_tile,
            "v1Tile": tile_cfg.v1_tile,
            "topkTile": tile_cfg.topk_tile,
            "addsTile": tile_cfg.adds_tile,
        },
        "unrollList": sorted(list(unroll_list)),
    }

    return args, meta


@pytest.mark.skip(reason="Case is no longer maintained")
def test_lightning_indexer_topk():
    logging.basicConfig(level=logging.INFO)
    setup_lightning_indexer_topk_config()
    args, meta = build_lightning_indexer_topk_args()
    logging.info({"Sanity": meta})
    lightning_indexer_topk_inner(args)
    assert True
