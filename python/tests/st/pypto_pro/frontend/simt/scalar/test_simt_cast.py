# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for the SIMT cast interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
ELEMENTS = 256

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

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_fp16(
    src_tile,
    out_fp32,
    out_bf16,
    out_int8,
    out_int16_rint,
    out_int16_floor,
    out_int16_ceil,
    out_int16_trunc,
    out_int32,
    out_int64,
):
    tid = pl.simt.linear_thread_idx()
    value = src_tile[0, tid]
    out_fp32[0, tid] = pl.simt.cast(value, pl.DT_FP32)
    out_bf16[0, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    out_int8[0, tid] = pl.simt.cast(value, pl.DT_INT8, mode=pl.RoundMode.CAST_TRUNC)
    out_int16_rint[0, tid] = pl.simt.cast(value, pl.DT_INT16, mode=pl.RoundMode.CAST_RINT)
    out_int16_floor[0, tid] = pl.simt.cast(value, pl.DT_INT16, mode=pl.RoundMode.CAST_FLOOR)
    out_int16_ceil[0, tid] = pl.simt.cast(value, pl.DT_INT16, mode=pl.RoundMode.CAST_CEIL)
    out_int16_trunc[0, tid] = pl.simt.cast(value, pl.DT_INT16, mode=pl.RoundMode.CAST_TRUNC)
    out_int32[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_int64[0, tid] = pl.simt.cast(value, pl.DT_INT64, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
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
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_int8_tile = out_int8_tile_group.current()
    out_int16_rint_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_rint_tile = out_int16_rint_tile_group.current()
    out_int16_floor_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_floor_tile = out_int16_floor_tile_group.current()
    out_int16_ceil_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_ceil_tile = out_int16_ceil_tile_group.current()
    out_int16_trunc_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_trunc_tile = out_int16_trunc_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_fp16[ELEMENTS](
            source_tile,
            out_fp32_tile,
            out_bf16_tile,
            out_int8_tile,
            out_int16_rint_tile,
            out_int16_floor_tile,
            out_int16_ceil_tile,
            out_int16_trunc_tile,
            out_int32_tile,
            out_int64_tile,
        )
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_int8, out_int8_tile, [0, 0])
        pl.store(out_int16_rint, out_int16_rint_tile, [0, 0])
        pl.store(out_int16_floor, out_int16_floor_tile, [0, 0])
        pl.store(out_int16_ceil, out_int16_ceil_tile, [0, 0])
        pl.store(out_int16_trunc, out_int16_trunc_tile, [0, 0])
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_int64, out_int64_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_fp16():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor(
        [-65504.0, -200.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 200.0, 65504.0],
        dtype=torch.float16,
    )
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
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
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_fp16(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_bf16(
    src_tile,
    out_fp32,
    out_fp16,
    out_uint8,
    out_uint16,
    out_uint32,
    out_uint64,
):
    tid = pl.simt.linear_thread_idx()
    value = src_tile[0, tid]
    out_fp32[0, tid] = pl.simt.cast(value, pl.DT_FP32)
    out_fp16[0, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    out_uint8[0, tid] = pl.simt.cast(value, pl.DT_UINT8, mode=pl.RoundMode.CAST_TRUNC)
    out_uint16[0, tid] = pl.simt.cast(value, pl.DT_UINT16, mode=pl.RoundMode.CAST_RINT)
    out_uint32[0, tid] = pl.simt.cast(value, pl.DT_UINT32, mode=pl.RoundMode.CAST_RINT)
    out_uint64[0, tid] = pl.simt.cast(value, pl.DT_UINT64, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_bf16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_uint8_tile = out_uint8_tile_group.current()
    out_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_uint16_tile = out_uint16_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_bf16[ELEMENTS](
            source_tile,
            out_fp32_tile,
            out_fp16_tile,
            out_uint8_tile,
            out_uint16_tile,
            out_uint32_tile,
            out_uint64_tile,
        )
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_uint8, out_uint8_tile, [0, 0])
        pl.store(out_uint16, out_uint16_tile, [0, 0])
        pl.store(out_uint32, out_uint32_tile, [0, 0])
        pl.store(out_uint64, out_uint64_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_bf16():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor(
        [-70000.0, -300.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 300.0, 70000.0],
        dtype=torch.bfloat16,
    )
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [
        source.to(torch.float32),
        source.to(torch.float16),
        _saturating_round(source, torch.uint8, torch.trunc),
        _saturating_round(source, torch.uint16, torch.round),
        _saturating_rint(source, torch.uint32),
        _saturating_rint(source, torch.uint64),
    ]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_bf16(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_int8(
    src_tile,
    out_int32,
    out_int64,
):
    tid = pl.simt.linear_thread_idx()
    out_int32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_INT32)
    out_int64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_INT64)

@pl.jit(auto_mutex=True)
def simt_cast_from_int8(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_int8[ELEMENTS](
            source_tile,
            out_int32_tile,
            out_int64_tile,
        )
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_int64, out_int64_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_int8():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor([torch.iinfo(torch.int8).min, -1, 0, 1, torch.iinfo(torch.int8).max], dtype=torch.int8)
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [source.to(torch.int32), source.to(torch.int64)]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_int8(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_int16(
    src_tile,
    out_int32,
    out_fp16,
):
    tid = pl.simt.linear_thread_idx()
    out_int32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_INT32)
    out_fp16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_int16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_int16[ELEMENTS](
            source_tile,
            out_int32_tile,
            out_fp16_tile,
        )
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_fp16, out_fp16_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_int16():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor(
        [torch.iinfo(torch.int16).min, -257, -1, 0, 1, 257, torch.iinfo(torch.int16).max],
        dtype=torch.int16,
    )
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [source.to(torch.int32), source.to(torch.float16)]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_int16(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_int32(
    src_tile,
    out_int64,
    out_fp32,
    out_fp16,
):
    tid = pl.simt.linear_thread_idx()
    out_int64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_INT64)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_fp16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_int32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_int32[ELEMENTS](
            source_tile,
            out_int64_tile,
            out_fp32_tile,
            out_fp16_tile,
        )
        pl.store(out_int64, out_int64_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_fp16, out_fp16_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_int32():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor(
        [torch.iinfo(torch.int32).min, -(2**24 + 1), -1, 0, 1, 2**24 + 1, torch.iinfo(torch.int32).max],
        dtype=torch.int32,
    )
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [source.to(torch.int64), source.to(torch.float32), source.to(torch.float32).to(torch.float16)]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_int32(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_int64(
    src_tile,
    out_int32,
    out_fp32,
    out_fp16,
):
    tid = pl.simt.linear_thread_idx()
    out_int32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_INT32)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_fp16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_int64(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_int64[ELEMENTS](
            source_tile,
            out_int32_tile,
            out_fp32_tile,
            out_fp16_tile,
        )
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_fp16, out_fp16_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_int64():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor(
        [torch.iinfo(torch.int32).min, -(2**24 + 1), -1, 0, 1, 2**24 + 1, torch.iinfo(torch.int32).max],
        dtype=torch.int64,
    )
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [source.to(torch.int32), source.to(torch.float32), source.to(torch.float32).to(torch.float16)]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_int64(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_fp32(
    src_tile,
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
):
    tid = pl.simt.linear_thread_idx()
    value = src_tile[0, tid]
    out_fp16_rint[0, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    out_fp16_odd[0, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_ODD)
    out_bf16[0, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    out_int32_rint[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_int32_round[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_ROUND)
    out_int32_floor[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_FLOOR)
    out_int32_ceil[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_CEIL)
    out_int32_trunc[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_TRUNC)
    out_uint32[0, tid] = pl.simt.cast(value, pl.DT_UINT32, mode=pl.RoundMode.CAST_RINT)
    out_int64[0, tid] = pl.simt.cast(value, pl.DT_INT64, mode=pl.RoundMode.CAST_RINT)
    out_uint64[0, tid] = pl.simt.cast(value, pl.DT_UINT64, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
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
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_fp16_rint_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_rint_tile = out_fp16_rint_tile_group.current()
    out_fp16_odd_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_odd_tile = out_fp16_odd_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_int32_rint_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_rint_tile = out_int32_rint_tile_group.current()
    out_int32_round_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_round_tile = out_int32_round_tile_group.current()
    out_int32_floor_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_floor_tile = out_int32_floor_tile_group.current()
    out_int32_ceil_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_ceil_tile = out_int32_ceil_tile_group.current()
    out_int32_trunc_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_trunc_tile = out_int32_trunc_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x3000,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_fp32[ELEMENTS](
            source_tile,
            out_fp16_rint_tile,
            out_fp16_odd_tile,
            out_bf16_tile,
            out_int32_rint_tile,
            out_int32_round_tile,
            out_int32_floor_tile,
            out_int32_ceil_tile,
            out_int32_trunc_tile,
            out_uint32_tile,
            out_int64_tile,
            out_uint64_tile,
        )
        pl.store(out_fp16_rint, out_fp16_rint_tile, [0, 0])
        pl.store(out_fp16_odd, out_fp16_odd_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_int32_rint, out_int32_rint_tile, [0, 0])
        pl.store(out_int32_round, out_int32_round_tile, [0, 0])
        pl.store(out_int32_floor, out_int32_floor_tile, [0, 0])
        pl.store(out_int32_ceil, out_int32_ceil_tile, [0, 0])
        pl.store(out_int32_trunc, out_int32_trunc_tile, [0, 0])
        pl.store(out_uint32, out_uint32_tile, [0, 0])
        pl.store(out_int64, out_int64_tile, [0, 0])
        pl.store(out_uint64, out_uint64_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_fp32():
    torch.npu.set_device(ST_DEVICE)
    source = ((torch.arange(ELEMENTS, dtype=torch.float32) - 128) / 2).reshape(1, ELEMENTS)
    source[0, 0] = 1.0001
    source[0, -2:] = torch.tensor([-1.0e30, 1.0e30])
    expected_odd = source.to(torch.float16)
    expected_odd[0, 0] = 1.0009765625
    round_away = torch.sign(source) * torch.floor(torch.abs(source) + 0.5)
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
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_fp32(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu()[:, :-2], golden[:, :-2], rtol=0, atol=0)
    for output, dtype in zip((outputs[3], outputs[8], outputs[9], outputs[10]),
                             (torch.int32, torch.uint32, torch.int64, torch.uint64)):
        torch.testing.assert_close(output.cpu()[:, -2:], _saturating_rint(source[:, -2:], dtype), rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_uint8(
    src_tile,
    out_uint32,
    out_uint64,
):
    tid = pl.simt.linear_thread_idx()
    out_uint32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_UINT32)
    out_uint64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_UINT64)

@pl.jit(auto_mutex=True)
def simt_cast_from_uint8(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_uint8[ELEMENTS](
            source_tile,
            out_uint32_tile,
            out_uint64_tile,
        )
        pl.store(out_uint32, out_uint32_tile, [0, 0])
        pl.store(out_uint64, out_uint64_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_uint8():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor([0, 1, 127, torch.iinfo(torch.uint8).max], dtype=torch.uint8)
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [source.to(torch.uint32), source.to(torch.uint64)]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_uint8(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_uint16(
    src_tile,
    out_uint32,
    out_bf16,
):
    tid = pl.simt.linear_thread_idx()
    out_uint32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_UINT32)
    out_bf16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_uint16(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_uint16[ELEMENTS](
            source_tile,
            out_uint32_tile,
            out_bf16_tile,
        )
        pl.store(out_uint32, out_uint32_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_uint16():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor([0, 1, 255, 256, 32767, 32768, torch.iinfo(torch.uint16).max], dtype=torch.uint16)
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [source.to(torch.uint32), source.to(torch.bfloat16)]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_uint16(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_uint32(
    src_tile,
    out_uint64,
    out_fp32,
    out_bf16,
):
    tid = pl.simt.linear_thread_idx()
    out_uint64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_UINT64)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_bf16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_uint32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_uint32[ELEMENTS](
            source_tile,
            out_uint64_tile,
            out_fp32_tile,
            out_bf16_tile,
        )
        pl.store(out_uint64, out_uint64_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_uint32():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor([0, 1, 65504, 2**24 + 1, torch.iinfo(torch.uint32).max], dtype=torch.uint32)
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [
        source.to(torch.uint64),
        source.to(torch.float32),
        source.to(torch.float32).to(torch.bfloat16),
    ]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_uint32(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cast_from_uint64(
    src_tile,
    out_uint64,
    out_fp32,
    out_bf16,
):
    tid = pl.simt.linear_thread_idx()
    out_uint64[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_UINT64)
    out_fp32[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    out_bf16[0, tid] = pl.simt.cast(src_tile[0, tid], pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)

@pl.jit(auto_mutex=True)
def simt_cast_from_uint64(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        cast_from_uint64[ELEMENTS](
            source_tile,
            out_uint64_tile,
            out_fp32_tile,
            out_bf16_tile,
        )
        pl.store(out_uint64, out_uint64_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])

@pytest.mark.soc("950")
def test_cast_from_uint64():
    torch.npu.set_device(ST_DEVICE)
    source = torch.tensor([0, 1, 2**32 + 1, 2**40 + 1, torch.iinfo(torch.uint64).max], dtype=torch.uint64)
    source = source.repeat((ELEMENTS + source.numel() - 1) // source.numel())[:ELEMENTS].reshape(1, ELEMENTS)
    expected = [
        source,
        source.to(torch.float32),
        source.to(torch.float32).to(torch.bfloat16),
    ]
    outputs = [torch.empty_like(golden, device=ST_DEVICE) for golden in expected]
    simt_cast_from_uint64(source.to(ST_DEVICE), *outputs)
    torch.npu.synchronize()
    for output, golden in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)
