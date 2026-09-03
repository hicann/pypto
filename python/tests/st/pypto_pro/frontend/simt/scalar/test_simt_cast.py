# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT cast interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
ELEMENTS = 256


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_fp16(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_tile,
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int16_rint: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int16_floor: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int16_ceil: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int16_trunc: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    tensor_value = src_tensor[0, tid]
    tile_value = src_tile[0, tid]
    out_fp32[0, tid] = pl.simt.cast(tensor_value, pl.DT_FP32)
    out_bf16[0, tid] = pl.simt.cast(tile_value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    out_int8[0, tid] = pl.simt.cast(tensor_value, pl.DT_INT8, mode=pl.RoundMode.CAST_TRUNC)
    out_int16_rint[0, tid] = pl.simt.cast(tile_value, pl.DT_INT16, mode=pl.RoundMode.CAST_RINT)
    out_int16_floor[0, tid] = pl.simt.cast(tensor_value, pl.DT_INT16, mode=pl.RoundMode.CAST_FLOOR)
    out_int16_ceil[0, tid] = pl.simt.cast(tile_value, pl.DT_INT16, mode=pl.RoundMode.CAST_CEIL)
    out_int16_trunc[0, tid] = pl.simt.cast(tensor_value, pl.DT_INT16, mode=pl.RoundMode.CAST_TRUNC)
    out_int32[0, tid] = pl.simt.cast(tile_value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_int64[0, tid] = pl.simt.cast(tensor_value, pl.DT_INT64, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_fp16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int16_rint: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int16_floor: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int16_ceil: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int16_trunc: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 2,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_fp16,
            threads=ELEMENTS,
            args=(
                source,
                source_tile,
                out_fp32,
                out_bf16,
                out_int8,
                out_int16_rint,
                out_int16_floor,
                out_int16_ceil,
                out_int16_trunc,
                out_int32,
                out_int64,
            ),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_bf16(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_tile,
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    tid = pl.simt.linear_thread_idx()
    tensor_value = src_tensor[0, tid]
    tile_value = src_tile[0, tid]
    out_fp32[0, tid] = pl.simt.cast(tensor_value, pl.DT_FP32)
    out_fp16[0, tid] = pl.simt.cast(tile_value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    out_uint8[0, tid] = pl.simt.cast(tensor_value, pl.DT_UINT8, mode=pl.RoundMode.CAST_TRUNC)
    out_uint16[0, tid] = pl.simt.cast(tile_value, pl.DT_UINT16, mode=pl.RoundMode.CAST_RINT)
    out_uint32[0, tid] = pl.simt.cast(tensor_value, pl.DT_UINT32, mode=pl.RoundMode.CAST_RINT)
    out_uint64[0, tid] = pl.simt.cast(tile_value, pl.DT_UINT64, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_bf16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 2,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_bf16,
            threads=ELEMENTS,
            args=(
                source,
                source_tile,
                out_fp32,
                out_fp16,
                out_uint8,
                out_uint16,
                out_uint32,
                out_uint64,
            ),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_int8(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    src_tile,
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    out_int32[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_INT32)
    out_int64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_INT64)


@pl.jit()
def simt_cast_from_int8(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(cast_from_int8, threads=ELEMENTS, args=(source, source_tile, out_int32, out_int64))


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_int16(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    src_tile,
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    out_int32[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_INT32)
    out_fp16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_int16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 2,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(cast_from_int16, threads=ELEMENTS, args=(source, source_tile, out_int32, out_fp16))


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_int32(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_tile,
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    out_int64[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_INT64)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_fp16[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_int32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 4,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_int32,
            threads=ELEMENTS,
            args=(source, source_tile, out_int64, out_fp32, out_fp16),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_int64(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    src_tile,
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    out_int32[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_INT32)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_fp16[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_int64(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 8,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_int64,
            threads=ELEMENTS,
            args=(source, source_tile, out_int32, out_fp32, out_fp16),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_fp32(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_tile,
    out_fp16_rint: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_fp16_odd: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_int32_rint: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_round: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_floor: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_ceil: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_trunc: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    tid = pl.simt.linear_thread_idx()
    tensor_value = src_tensor[0, tid]
    tile_value = src_tile[0, tid]
    out_fp16_rint[0, tid] = pl.simt.cast(tensor_value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    out_fp16_odd[0, tid] = pl.simt.cast(tile_value, pl.DT_FP16, mode=pl.RoundMode.CAST_ODD)
    out_bf16[0, tid] = pl.simt.cast(tensor_value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    out_int32_rint[0, tid] = pl.simt.cast(tile_value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_int32_round[0, tid] = pl.simt.cast(tensor_value, pl.DT_INT32, mode=pl.RoundMode.CAST_ROUND)
    out_int32_floor[0, tid] = pl.simt.cast(tile_value, pl.DT_INT32, mode=pl.RoundMode.CAST_FLOOR)
    out_int32_ceil[0, tid] = pl.simt.cast(tensor_value, pl.DT_INT32, mode=pl.RoundMode.CAST_CEIL)
    out_int32_trunc[0, tid] = pl.simt.cast(tile_value, pl.DT_INT32, mode=pl.RoundMode.CAST_TRUNC)
    out_uint32[0, tid] = pl.simt.cast(tensor_value, pl.DT_UINT32, mode=pl.RoundMode.CAST_RINT)
    out_int64[0, tid] = pl.simt.cast(tile_value, pl.DT_INT64, mode=pl.RoundMode.CAST_RINT)
    out_uint64[0, tid] = pl.simt.cast(tensor_value, pl.DT_UINT64, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_fp32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16_rint: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_fp16_odd: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_int32_rint: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_round: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_floor: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_ceil: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int32_trunc: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 4,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_fp32,
            threads=ELEMENTS,
            args=(
                source,
                source_tile,
                out_fp16_rint,
                out_fp16_odd,
                out_bf16,
                out_int32_rint,
                out_int32_round,
                out_int32_floor,
                out_int32_ceil,
                out_int32_trunc,
                out_uint32,
                out_int64,
                out_uint64,
            ),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_uint8(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    src_tile,
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    tid = pl.simt.linear_thread_idx()
    out_uint32[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_UINT32)
    out_uint64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_UINT64)


@pl.jit()
def simt_cast_from_uint8(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(cast_from_uint8, threads=ELEMENTS, args=(source, source_tile, out_uint32, out_uint64))


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_uint16(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    src_tile,
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    tid = pl.simt.linear_thread_idx()
    out_uint32[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_UINT32)
    out_bf16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_uint16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 2,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(cast_from_uint16, threads=ELEMENTS, args=(source, source_tile, out_uint32, out_bf16))


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_uint32(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_tile,
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    tid = pl.simt.linear_thread_idx()
    out_uint64[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_UINT64)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_bf16[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_uint32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 4,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_uint32,
            threads=ELEMENTS,
            args=(source, source_tile, out_uint64, out_fp32, out_bf16),
        )


@pl.simt.function(max_threads=ELEMENTS)
def cast_from_uint64(
    src_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    src_tile,
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    tid = pl.simt.linear_thread_idx()
    out_uint64[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_UINT64)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_bf16[0, tid] = pl.simt.cast(src_tensor[0, tid], pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)


@pl.jit()
def simt_cast_from_uint64(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    source_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
        size=ELEMENTS * 8,
    )
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(
            cast_from_uint64,
            threads=ELEMENTS,
            args=(source, source_tile, out_uint64, out_fp32, out_bf16),
        )


def _repeat(values, dtype):
    values = torch.tensor(values, dtype=dtype)
    repeats = (ELEMENTS + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:ELEMENTS].reshape(1, ELEMENTS)


def _saturating_rint(values, dtype):
    info = torch.iinfo(dtype)
    result = []
    for value in values.reshape(-1).tolist():
        rounded = round(value)
        result.append(min(max(rounded, info.min), info.max))
    return torch.tensor(result, dtype=dtype).reshape_as(values)


def _saturating_round(values, dtype, rounding):
    info = torch.iinfo(dtype)
    rounded = rounding(values.to(torch.float32))
    return torch.clamp(rounded, info.min, info.max).to(dtype)


def _run_cast(kernel, source, output_dtypes):
    _require_a5()
    outputs = [torch.empty((1, ELEMENTS), dtype=dtype).to(ST_DEVICE) for dtype in output_dtypes]
    kernel(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    return [output.cpu() for output in outputs]


def _assert_outputs(actual, expected):
    assert len(actual) == len(expected)
    for result, golden in zip(actual, expected):
        torch.testing.assert_close(result, golden, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_cast_from_fp16():
    source = _repeat([-65504.0, -200.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 200.0, 65504.0], torch.float16)
    output_dtypes = [
        torch.float32,
        torch.bfloat16,
        torch.int8,
        torch.int16,
        torch.int16,
        torch.int16,
        torch.int16,
        torch.int32,
        torch.int64,
    ]
    expected = [
        source.to(torch.float32),
        source.to(torch.bfloat16),
        _saturating_round(source, torch.int8, torch.trunc),
        _saturating_round(source, torch.int16, torch.round),
        _saturating_round(source, torch.int16, torch.floor),
        _saturating_round(source, torch.int16, torch.ceil),
        _saturating_round(source, torch.int16, torch.trunc),
        _saturating_rint(source, torch.int32),
        _saturating_rint(source, torch.int64),
    ]
    _assert_outputs(_run_cast(simt_cast_from_fp16, source, output_dtypes), expected)


@pytest.mark.soc("950")
def test_cast_from_bf16():
    source = _repeat([-70000.0, -300.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 300.0, 70000.0], torch.bfloat16)
    output_dtypes = [torch.float32, torch.float16, torch.uint8, torch.uint16, torch.uint32, torch.uint64]
    expected = [
        source.to(torch.float32),
        source.to(torch.float16),
        _saturating_round(source, torch.uint8, torch.trunc),
        _saturating_round(source, torch.uint16, torch.round),
        _saturating_rint(source, torch.uint32),
        _saturating_rint(source, torch.uint64),
    ]
    _assert_outputs(_run_cast(simt_cast_from_bf16, source, output_dtypes), expected)


@pytest.mark.soc("950")
def test_cast_from_fp32():
    source = ((torch.arange(ELEMENTS, dtype=torch.float32) - 128) / 2).reshape(1, ELEMENTS)
    source[0, 0] = 1.0001
    expected_odd = source.to(torch.float16)
    expected_odd[0, 0] = 1.0009765625
    round_away = torch.sign(source) * torch.floor(torch.abs(source) + 0.5)
    output_dtypes = [
        torch.float16,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.uint32,
        torch.int64,
        torch.uint64,
    ]
    expected = [
        source.to(torch.float16),
        expected_odd,
        source.to(torch.bfloat16),
        torch.round(source).to(torch.int32),
        round_away.to(torch.int32),
        torch.floor(source).to(torch.int32),
        torch.ceil(source).to(torch.int32),
        torch.trunc(source).to(torch.int32),
        _saturating_rint(source, torch.uint32),
        _saturating_rint(source, torch.int64),
        _saturating_rint(source, torch.uint64),
    ]
    _assert_outputs(_run_cast(simt_cast_from_fp32, source, output_dtypes), expected)


@pytest.mark.soc("950")
def test_cast_from_fp32_wide_integer_saturation():
    source = _repeat([-1.0e30, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 1.0e30], torch.float32)
    output_dtypes = [
        torch.float16,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.uint32,
        torch.int64,
        torch.uint64,
    ]
    actual = _run_cast(simt_cast_from_fp32, source, output_dtypes)
    expected = [
        _saturating_rint(source, torch.int32),
        _saturating_rint(source, torch.uint32),
        _saturating_rint(source, torch.int64),
        _saturating_rint(source, torch.uint64),
    ]
    _assert_outputs([actual[3], actual[8], actual[9], actual[10]], expected)


@pytest.mark.soc("950")
def test_cast_from_int8():
    source = _repeat([torch.iinfo(torch.int8).min, -1, 0, 1, torch.iinfo(torch.int8).max], torch.int8)
    expected = [source.to(torch.int32), source.to(torch.int64)]
    _assert_outputs(_run_cast(simt_cast_from_int8, source, [torch.int32, torch.int64]), expected)


@pytest.mark.soc("950")
def test_cast_from_int16():
    source = _repeat(
        [torch.iinfo(torch.int16).min, -257, -1, 0, 1, 257, torch.iinfo(torch.int16).max],
        torch.int16,
    )
    expected = [source.to(torch.int32), source.to(torch.float16)]
    _assert_outputs(_run_cast(simt_cast_from_int16, source, [torch.int32, torch.float16]), expected)


@pytest.mark.soc("950")
def test_cast_from_int32():
    source = _repeat(
        [torch.iinfo(torch.int32).min, -(2**24 + 1), -1, 0, 1, 2**24 + 1, torch.iinfo(torch.int32).max],
        torch.int32,
    )
    expected = [source.to(torch.int64), source.to(torch.float32), source.to(torch.float32).to(torch.float16)]
    _assert_outputs(_run_cast(simt_cast_from_int32, source, [torch.int64, torch.float32, torch.float16]), expected)


@pytest.mark.soc("950")
def test_cast_from_int64():
    source = _repeat(
        [torch.iinfo(torch.int32).min, -(2**24 + 1), -1, 0, 1, 2**24 + 1, torch.iinfo(torch.int32).max],
        torch.int64,
    )
    expected = [source.to(torch.int32), source.to(torch.float32), source.to(torch.float32).to(torch.float16)]
    _assert_outputs(_run_cast(simt_cast_from_int64, source, [torch.int32, torch.float32, torch.float16]), expected)


@pytest.mark.soc("950")
def test_cast_from_uint8():
    source = _repeat([0, 1, 127, torch.iinfo(torch.uint8).max], torch.uint8)
    expected = [source.to(torch.uint32), source.to(torch.uint64)]
    _assert_outputs(_run_cast(simt_cast_from_uint8, source, [torch.uint32, torch.uint64]), expected)


@pytest.mark.soc("950")
def test_cast_from_uint16():
    source = _repeat([0, 1, 255, 256, 32767, 32768, torch.iinfo(torch.uint16).max], torch.uint16)
    expected = [source.to(torch.uint32), source.to(torch.bfloat16)]
    _assert_outputs(_run_cast(simt_cast_from_uint16, source, [torch.uint32, torch.bfloat16]), expected)


@pytest.mark.soc("950")
def test_cast_from_uint32():
    source = _repeat([0, 1, 65504, 2**24 + 1, torch.iinfo(torch.uint32).max], torch.uint32)
    expected = [
        source.to(torch.uint64),
        source.to(torch.float32),
        source.to(torch.float32).to(torch.bfloat16),
    ]
    _assert_outputs(_run_cast(simt_cast_from_uint32, source, [torch.uint64, torch.float32, torch.bfloat16]), expected)


@pytest.mark.soc("950")
def test_cast_from_uint64():
    source = _repeat([0, 1, 2**32 + 1, 2**40 + 1, torch.iinfo(torch.uint64).max], torch.uint64)
    expected = [
        source,
        source.to(torch.float32),
        source.to(torch.float32).to(torch.bfloat16),
    ]
    _assert_outputs(_run_cast(simt_cast_from_uint64, source, [torch.uint64, torch.float32, torch.bfloat16]), expected)
