# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 128


@pl.simt.function(max_threads=ELEMENTS)
def count_bits(
    src32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    count32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    count64: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    count32[0, tid] = pl.simt.popcount(src32[0, tid])
    count64[0, tid] = pl.simt.popcount(src64[0, tid])


@pl.jit()
def simt_popcount(
    src32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    count32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    count64: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    with pl.section_vector():
        pl.simt.launch(count_bits, threads=ELEMENTS, args=(src32, src64, count32, count64))


@pytest.mark.soc("950")
def test_popcount_uint32_uint64(a5_device, assert_simt_close):
    sources = []
    goldens = []
    for bits, dtype in ((32, torch.uint32), (64, torch.uint64)):
        mask = (1 << bits) - 1
        patterns = [0, mask, mask // 3, (mask // 3) << 1, 1, 1 << (bits // 2), 1 << (bits - 1), mask ^ 1]
        values = patterns * (ELEMENTS // len(patterns))
        sources.append(torch.tensor(values, dtype=dtype).reshape(1, ELEMENTS))
        goldens.append(torch.tensor([value.bit_count() for value in values], dtype=torch.int32).reshape(1, ELEMENTS))

    count32 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=a5_device)
    count64 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=a5_device)
    simt_popcount(sources[0].to(a5_device), sources[1].to(a5_device), count32, count64)
    torch.npu.synchronize()

    assert_simt_close(count32, goldens[0])
    assert_simt_close(count64, goldens[1])
