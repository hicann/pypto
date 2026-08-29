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
def multiply_high(
    lhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    lhs_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    rhs_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    lhs_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    rhs_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    tid = pl.simt.linear_thread_idx()
    out_i32[0, tid] = pl.simt.mul_hi(lhs_i32[0, tid], rhs_i32[0, tid])
    out_u32[0, tid] = pl.simt.mul_hi(lhs_u32[0, tid], rhs_u32[0, tid])
    out_i64[0, tid] = pl.simt.mul_hi(lhs_i64[0, tid], rhs_i64[0, tid])
    out_u64[0, tid] = pl.simt.mul_hi(lhs_u64[0, tid], rhs_u64[0, tid])


@pl.jit()
def simt_mul_hi(
    lhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    lhs_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    rhs_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    lhs_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    rhs_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    with pl.section_vector():
        pl.simt.launch(
            multiply_high,
            threads=ELEMENTS,
            args=(
                lhs_i32,
                rhs_i32,
                lhs_u32,
                rhs_u32,
                lhs_i64,
                rhs_i64,
                lhs_u64,
                rhs_u64,
                out_i32,
                out_u32,
                out_i64,
                out_u64,
            ),
        )


@pytest.mark.soc("950")
def test_mul_hi_all_supported_dtypes(a5_device, assert_simt_close):
    dtypes = ((32, torch.int32, True), (32, torch.uint32, False), (64, torch.int64, True), (64, torch.uint64, False))
    arguments = []
    goldens = []
    for bits, dtype, signed in dtypes:
        lower = -(1 << (bits - 1)) if signed else 0
        upper = (1 << (bits - int(signed))) - 1
        if signed:
            edges = [lower, lower + 1, -2, -1, 0, 1, upper - 1, upper]
        else:
            half = 1 << (bits - 1)
            edges = [0, 1, 2, half - 1, half, half + 1, upper - 1, upper]
        pairs = [(x, y) for x in edges for y in edges] * 2
        arguments.extend(
            torch.tensor(values, dtype=dtype).reshape(1, ELEMENTS).to(a5_device) for values in zip(*pairs)
        )
        goldens.append(torch.tensor([(x * y) >> bits for x, y in pairs], dtype=dtype).reshape(1, ELEMENTS))

    outputs = tuple(torch.empty((1, ELEMENTS), dtype=dtype, device=a5_device) for _, dtype, _ in dtypes)
    simt_mul_hi(*arguments, *outputs)
    torch.npu.synchronize()

    for output, expected in zip(outputs, goldens):
        assert_simt_close(output, expected)
