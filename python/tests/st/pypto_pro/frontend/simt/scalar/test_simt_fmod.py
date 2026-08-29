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
def fmod_values(
    lhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    out[0, tid] = pl.simt.fmod(lhs[0, tid], rhs[0, tid])


@pl.jit()
def simt_fmod_values(
    lhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(fmod_values, threads=ELEMENTS, args=(lhs, rhs, out))


@pytest.mark.soc("950")
def test_fmod_fp32(a5_device, assert_simt_close):
    pairs = [
        (5.5, 2.0),
        (5.5, -2.0),
        (-5.5, 2.0),
        (-5.5, -2.0),
        (4.0, 2.0),
        (-4.0, 2.0),
        (7.25, 3.0),
        (-7.25, 3.0),
    ] * (ELEMENTS // 8)
    lhs = torch.tensor([x for x, _ in pairs], dtype=torch.float32).reshape(1, ELEMENTS)
    rhs = torch.tensor([y for _, y in pairs], dtype=torch.float32).reshape(1, ELEMENTS)
    out = torch.empty_like(lhs, device=a5_device)
    simt_fmod_values(lhs.to(a5_device), rhs.to(a5_device), out)
    torch.npu.synchronize()

    expected = torch.fmod(lhs, rhs)
    assert_simt_close(out, expected)
