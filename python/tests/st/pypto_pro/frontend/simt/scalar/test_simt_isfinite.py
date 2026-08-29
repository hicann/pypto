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
def classify(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    finite_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    finite_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
):
    tid = pl.simt.linear_thread_idx()
    finite_fp16[0, tid] = pl.simt.isfinite(src_fp16[0, tid])
    finite_fp32[0, tid] = pl.simt.isfinite(src_fp32[0, tid])


@pl.jit()
def simt_isfinite(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    finite_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    finite_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
):
    with pl.section_vector():
        pl.simt.launch(classify, threads=ELEMENTS, args=(src_fp16, src_fp32, finite_fp16, finite_fp32))


@pytest.mark.soc("950")
def test_isfinite_fp16_fp32(a5_device, assert_simt_close):
    values = [0.0, -0.0, 1.0, -1.0, 7.5, float("inf"), float("-inf"), float("nan")]
    repeat = ELEMENTS // len(values)
    source_fp16 = torch.tensor(values, dtype=torch.float16).repeat(repeat).reshape(1, ELEMENTS)
    source_fp32 = torch.tensor(values, dtype=torch.float32).repeat(repeat).reshape(1, ELEMENTS)
    finite_fp16 = torch.empty((1, ELEMENTS), dtype=torch.bool, device=a5_device)
    finite_fp32 = torch.empty((1, ELEMENTS), dtype=torch.bool, device=a5_device)
    simt_isfinite(source_fp16.to(a5_device), source_fp32.to(a5_device), finite_fp16, finite_fp32)
    torch.npu.synchronize()

    assert_simt_close(finite_fp16, torch.isfinite(source_fp16))
    assert_simt_close(finite_fp32, torch.isfinite(source_fp32))
