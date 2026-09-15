# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""All floating-point unary SIMT operations share one compiled kernel.

Each row is one operation's original input set, including infinities, signed zeros,
rounding ties, domain boundaries and large arguments. All three dtypes are checked
independently against CPU references; operations do not consume each other's output.
"""

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64
OPERATIONS = 14


def _round_away_from_zero(value):
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def unary_all_dtypes(
    src_fp16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.ceil(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.ceil(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.ceil(src_fp32[0, tid])
    out_fp16[1, tid] = pl.simt.floor(src_fp16[1, tid])
    out_bf16[1, tid] = pl.simt.floor(src_bf16[1, tid])
    out_fp32[1, tid] = pl.simt.floor(src_fp32[1, tid])
    out_fp16[2, tid] = pl.simt.trunc(src_fp16[2, tid])
    out_bf16[2, tid] = pl.simt.trunc(src_bf16[2, tid])
    out_fp32[2, tid] = pl.simt.trunc(src_fp32[2, tid])
    out_fp16[3, tid] = pl.simt.round(src_fp16[3, tid])
    out_bf16[3, tid] = pl.simt.round(src_bf16[3, tid])
    out_fp32[3, tid] = pl.simt.round(src_fp32[3, tid])
    out_fp16[4, tid] = pl.simt.rint(src_fp16[4, tid])
    out_bf16[4, tid] = pl.simt.rint(src_bf16[4, tid])
    out_fp32[4, tid] = pl.simt.rint(src_fp32[4, tid])
    out_fp16[5, tid] = pl.simt.sin(src_fp16[5, tid])
    out_bf16[5, tid] = pl.simt.sin(src_bf16[5, tid])
    out_fp32[5, tid] = pl.simt.sin(src_fp32[5, tid])
    out_fp16[6, tid] = pl.simt.cos(src_fp16[6, tid])
    out_bf16[6, tid] = pl.simt.cos(src_bf16[6, tid])
    out_fp32[6, tid] = pl.simt.cos(src_fp32[6, tid])
    out_fp16[7, tid] = pl.simt.exp(src_fp16[7, tid])
    out_bf16[7, tid] = pl.simt.exp(src_bf16[7, tid])
    out_fp32[7, tid] = pl.simt.exp(src_fp32[7, tid])
    out_fp16[8, tid] = pl.simt.exp2(src_fp16[8, tid])
    out_bf16[8, tid] = pl.simt.exp2(src_bf16[8, tid])
    out_fp32[8, tid] = pl.simt.exp2(src_fp32[8, tid])
    out_fp16[9, tid] = pl.simt.log(src_fp16[9, tid])
    out_bf16[9, tid] = pl.simt.log(src_bf16[9, tid])
    out_fp32[9, tid] = pl.simt.log(src_fp32[9, tid])
    out_fp16[10, tid] = pl.simt.log2(src_fp16[10, tid])
    out_bf16[10, tid] = pl.simt.log2(src_bf16[10, tid])
    out_fp32[10, tid] = pl.simt.log2(src_fp32[10, tid])
    out_fp16[11, tid] = pl.simt.sqrt(src_fp16[11, tid])
    out_bf16[11, tid] = pl.simt.sqrt(src_bf16[11, tid])
    out_fp32[11, tid] = pl.simt.sqrt(src_fp32[11, tid])
    out_fp16[12, tid] = pl.simt.rsqrt(src_fp16[12, tid])
    out_bf16[12, tid] = pl.simt.rsqrt(src_bf16[12, tid])
    out_fp32[12, tid] = pl.simt.rsqrt(src_fp32[12, tid])
    out_fp16[13, tid] = pl.simt.tanh(src_fp16[13, tid])
    out_bf16[13, tid] = pl.simt.tanh(src_bf16[13, tid])
    out_fp32[13, tid] = pl.simt.tanh(src_fp32[13, tid])


@pl.jit()
def simt_unary_all_dtypes(
    src_fp16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[OPERATIONS, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        unary_all_dtypes[ELEMENTS](src_fp16, src_bf16, src_fp32, out_fp16, out_bf16, out_fp32)


def _unary_cases():
    rounding = torch.tensor(
        [-2.5000002, -2.5, -2.4999998, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.4999998, 2.5, 2.5000002]
    ).repeat(6)[:ELEMENTS]
    for name, golden in (
        ("ceil", torch.ceil),
        ("floor", torch.floor),
        ("trunc", torch.trunc),
        ("round", _round_away_from_zero),
        ("rint", torch.round),
    ):
        yield name, rounding, golden

    angles = torch.linspace(-3.14159265, 3.14159265, ELEMENTS)
    angles[:10] = torch.tensor(
        [-0.0, 0.0, -3.14159265, 3.14159265, -314.159265, 314.159265, -10000.0, 10000.0, float("-inf"), float("inf")]
    )
    yield "sin", angles, torch.sin
    yield "cos", angles, torch.cos

    values = torch.linspace(-3.0, 3.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -104.0, -88.0, -1.0e-7, 0.0, 1.0e-7, 88.0, float("inf")])
    yield "exp", values, torch.exp

    values = torch.linspace(-3.0, 3.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -150.0, -126.0, -1.0e-7, 0.0, 1.0e-7, 127.0, float("inf")])
    yield "exp2", values, torch.exp2

    values = torch.linspace(0.125, 8.0, ELEMENTS)
    values[:8] = torch.tensor([-1.0, -0.0, 0.0, torch.finfo(torch.float32).tiny, 1.0e-7, 1.0, 2.0, float("inf")])
    yield "log", values, torch.log
    yield "log2", values, torch.log2

    values = torch.linspace(0.25, 16.0, ELEMENTS)
    values[:5] = torch.tensor([0.0, -0.0, torch.finfo(torch.float32).tiny, 1.0, float("inf")])
    yield "sqrt", values, torch.sqrt
    yield "rsqrt", values, torch.rsqrt

    values = torch.linspace(-4.0, 4.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -10.0, -1.0e-7, -0.0, 0.0, 1.0e-7, 10.0, float("inf")])
    yield "tanh", values, torch.tanh


@pytest.mark.soc("950")
def test_unary_operations_all_supported_dtypes(a5_device, assert_simt_close):
    cases = list(_unary_cases())
    values = torch.stack([values for _, values, _ in cases]).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=a5_device) for source in sources)
    simt_unary_all_dtypes(*(source.to(a5_device) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        for row, (name, _, golden) in enumerate(cases):
            expected = golden(source[row].float()).to(source.dtype)
            try:
                assert_simt_close(output[row], expected)
            except AssertionError as exc:
                raise AssertionError(f"{name}/{source.dtype}: {exc}") from exc
