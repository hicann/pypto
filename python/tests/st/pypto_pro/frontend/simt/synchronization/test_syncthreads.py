# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""每个线程写入共享UB，块内同步后读取另一线程写入的数据。"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 128


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.simt.function(max_threads=THREADS)
def exchange_after_syncthreads(out: pl.Tensor[[1, THREADS], pl.DT_UINT32], shared):
    tid = pl.simt.linear_thread_idx()
    shared[0, tid] = tid
    pl.simt.syncthreads()
    out[0, tid] = shared[0, THREADS - 1 - tid]


@pl.jit()
def simt_syncthreads(out: pl.Tensor[[1, THREADS], pl.DT_UINT32]):
    shared = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0,
        size=THREADS * 4,
    )
    with pl.section_vector():
        pl.simt.launch(exchange_after_syncthreads, threads=THREADS, args=(out, shared))


@pytest.mark.soc("950")
def test_syncthreads():
    _require_a5()
    out = torch.empty((1, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    simt_syncthreads(out)
    torch.npu.synchronize()

    expected = torch.arange(THREADS - 1, -1, -1, dtype=torch.int64).to(torch.uint32).reshape(1, THREADS)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
