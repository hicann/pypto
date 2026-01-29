#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def add_op_range(b: torch.Tensor) -> torch.Tensor:
    b_shape = b.shape

    @pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU}
    )
    def add_kernel_range(
        b: pypto.Tensor(b_shape, pypto.DT_FP32)
    ) -> pypto.Tensor((16, 16), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(16, 16)
        c = b[0:16, 0:16]
        for i in range(1, 4):
            c = pypto.add(c, b[i * 16:(i + 1) * 16, i * 16:(i + 1) * 16])
        return c

    out_range = add_kernel_range(b)
    return out_range


def add_op_list(b: torch.Tensor) -> torch.Tensor:
    b_shape = b.shape
    out_shape = (16, 16)
    mode = pypto.RunMode.NPU

    @pypto.frontend.jit(
        runtime_options={"run_mode": mode}
    )
    def add_kernel_list(
        b: pypto.Tensor(b_shape, pypto.DT_FP32)
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(16, 16)
        c = b[0:16, 0:16]
        for i in [1, 2, 3]:
            c = pypto.add(c, b[i * 16:(i + 1) * 16, i * 16:(i + 1) * 16])

        return c

    out_list = add_kernel_list(b)
    return out_list


def test_range_list_iterate():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)

    b = torch.rand((64, 64), dtype=torch.float32, device=device_id)
    expected = b[:16, :16] + b[16:32, 16:32] + b[32:48, 32:48] + b[48:, 48:]

    out_range = add_op_range(b)
    out_list = add_op_list(b)

    assert_allclose(out_range.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    assert_allclose(out_list.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-3, atol=1e-3)