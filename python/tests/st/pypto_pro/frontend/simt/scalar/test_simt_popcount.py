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

ELEMENTS = 128

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def count_bits(
    src32,
    src64,
    count32,
    count64,
):
    tid = pl.simt.linear_thread_idx()
    count32[0, tid] = pl.simt.popcount(src32[0, tid])
    count64[0, tid] = pl.simt.popcount(src64[0, tid])


@pl.jit(auto_mutex=True)
def simt_popcount(
    src32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    count32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    count64: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    src32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src32_tile = src32_tile_group.current()
    src64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src64_tile = src64_tile_group.current()
    count32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    count32_tile = count32_tile_group.current()
    count64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    count64_tile = count64_tile_group.current()
    with pl.section_vector():
        pl.load(src32_tile, src32, [0, 0])
        pl.load(src64_tile, src64, [0, 0])
        count_bits[ELEMENTS](
            src32_tile,
            src64_tile,
            count32_tile,
            count64_tile,
        )
        pl.store(count32, count32_tile, [0, 0])
        pl.store(count64, count64_tile, [0, 0])


@pytest.mark.soc("950")
def test_popcount_uint32_uint64():
    torch.npu.set_device(ST_DEVICE)
    sources = []
    goldens = []
    for bits, dtype in ((32, torch.uint32), (64, torch.uint64)):
        mask = (1 << bits) - 1
        patterns = [0, mask, mask // 3, (mask // 3) << 1, 1, 1 << (bits // 2), 1 << (bits - 1), mask ^ 1]
        values = patterns * (ELEMENTS // len(patterns))
        sources.append(torch.tensor(values, dtype=dtype).reshape(1, ELEMENTS))
        goldens.append(torch.tensor([value.bit_count() for value in values], dtype=torch.int32).reshape(1, ELEMENTS))

    count32 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    count64 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    simt_popcount(sources[0].to(ST_DEVICE), sources[1].to(ST_DEVICE), count32, count64)
    torch.npu.synchronize()

    torch.testing.assert_close(count32.cpu(), goldens[0], rtol=0, atol=0)
    torch.testing.assert_close(count64.cpu(), goldens[1], rtol=0, atol=0)
