# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def bitcast_all_scalar_pairs(
    src_int16,
    src_uint16,
    src_fp16,
    src_bf16,
    src_int32,
    src_uint32,
    src_fp32,
    out_fp16,
    out_bf16,
    out_int16,
    out_uint16,
    out_fp32,
    out_int32,
    out_uint32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.bitcast(src_int16[0, tid], pl.DT_FP16)
    out_fp16[1, tid] = pl.simt.bitcast(src_uint16[0, tid], pl.DT_FP16)
    out_bf16[0, tid] = pl.simt.bitcast(src_int16[0, tid], pl.DT_BF16)
    out_bf16[1, tid] = pl.simt.bitcast(src_uint16[0, tid], pl.DT_BF16)
    out_int16[0, tid] = pl.simt.bitcast(src_fp16[0, tid], pl.DT_INT16)
    out_int16[1, tid] = pl.simt.bitcast(src_bf16[0, tid], pl.DT_INT16)
    out_uint16[0, tid] = pl.simt.bitcast(src_fp16[0, tid], pl.DT_UINT16)
    out_uint16[1, tid] = pl.simt.bitcast(src_bf16[0, tid], pl.DT_UINT16)
    out_fp32[0, tid] = pl.simt.bitcast(src_int32[0, tid], pl.DT_FP32)
    out_fp32[1, tid] = pl.simt.bitcast(src_uint32[0, tid], pl.DT_FP32)
    out_int32[0, tid] = pl.simt.bitcast(src_fp32[0, tid], pl.DT_INT32)
    out_uint32[0, tid] = pl.simt.bitcast(src_fp32[0, tid], pl.DT_UINT32)


@pl.jit(auto_mutex=True)
def simt_bitcast_all_scalar_pairs(
    src_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    src_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[2, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[2, ELEMENTS], pl.DT_BF16],
    out_int16: pl.Tensor[[2, ELEMENTS], pl.DT_INT16],
    out_uint16: pl.Tensor[[2, ELEMENTS], pl.DT_UINT16],
    out_fp32: pl.Tensor[[2, ELEMENTS], pl.DT_FP32],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
):
    src_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_int16_tile = src_int16_tile_group.current()
    src_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_uint16_tile = src_uint16_tile_group.current()
    src_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    src_fp16_tile = src_fp16_tile_group.current()
    src_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    src_bf16_tile = src_bf16_tile_group.current()
    src_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    src_int32_tile = src_int32_tile_group.current()
    src_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    src_uint32_tile = src_uint32_tile_group.current()
    src_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    src_fp32_tile = src_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_tile = out_int16_tile_group.current()
    out_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_uint16_tile = out_uint16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x3000,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x3400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    with pl.section_vector():
        pl.load(src_int16_tile, src_int16, [0, 0])
        pl.load(src_uint16_tile, src_uint16, [0, 0])
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_bf16_tile, src_bf16, [0, 0])
        pl.load(src_int32_tile, src_int32, [0, 0])
        pl.load(src_uint32_tile, src_uint32, [0, 0])
        pl.load(src_fp32_tile, src_fp32, [0, 0])
        bitcast_all_scalar_pairs[ELEMENTS](
            src_int16_tile,
            src_uint16_tile,
            src_fp16_tile,
            src_bf16_tile,
            src_int32_tile,
            src_uint32_tile,
            src_fp32_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_int16_tile,
            out_uint16_tile,
            out_fp32_tile,
            out_int32_tile,
            out_uint32_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_int16, out_int16_tile, [0, 0])
        pl.store(out_uint16, out_uint16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_uint32, out_uint32_tile, [0, 0])


@pytest.mark.soc("950")
def test_bitcast_preserves_all_supported_scalar_bit_patterns():
    torch.npu.set_device(ST_DEVICE)
    bits16 = torch.tensor(
        [0x0000, 0x3C00, 0x7C00, 0x7E00, 0x8000, 0xBC00, 0xFC00, 0xFFFF],
        dtype=torch.uint16,
    ).view(torch.int16).repeat(ELEMENTS // 8).reshape(1, ELEMENTS)
    bits32 = torch.tensor(
        [
            0x00000000,
            0x3F800000,
            0x7F800000,
            0x7FC00000,
            0x80000000,
            0xBF800000,
            0xFF800000,
            0xFFFFFFFF,
        ],
        dtype=torch.uint32,
    ).view(torch.int32).repeat(ELEMENTS // 8).reshape(1, ELEMENTS)
    sources = (
        bits16,
        bits16.view(torch.uint16),
        bits16.view(torch.float16),
        bits16.view(torch.bfloat16),
        bits32,
        bits32.view(torch.uint32),
        bits32.view(torch.float32),
    )
    outputs = (
        torch.empty((2, ELEMENTS), dtype=torch.float16, device=ST_DEVICE),
        torch.empty((2, ELEMENTS), dtype=torch.bfloat16, device=ST_DEVICE),
        torch.empty((2, ELEMENTS), dtype=torch.int16, device=ST_DEVICE),
        torch.empty((2, ELEMENTS), dtype=torch.uint16, device=ST_DEVICE),
        torch.empty((2, ELEMENTS), dtype=torch.float32, device=ST_DEVICE),
        torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE),
        torch.empty((1, ELEMENTS), dtype=torch.uint32, device=ST_DEVICE),
    )

    simt_bitcast_all_scalar_pairs(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()

    fp16, bf16, int16, uint16, fp32, int32, uint32 = (output.cpu() for output in outputs)
    for output in (fp16[0], fp16[1], bf16[0], bf16[1], int16[0], int16[1], uint16[0], uint16[1]):
        assert torch.equal(output.contiguous().view(torch.int16), bits16[0])
    for output in (fp32[0], fp32[1], int32[0], uint32[0]):
        assert torch.equal(output.contiguous().view(torch.int32), bits32[0])
