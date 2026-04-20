#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test pypto.frontend.jit kernel dynamic shape error behavior."""

import logging
import os

import pytest
import torch
import pypto

logging.basicConfig(level=logging.INFO, format="", force=True)
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))


# ------------------------------------------------------------------------------
# Kernel definitions
# ------------------------------------------------------------------------------

@pypto.frontend.jit
def kernel_with_dynamic(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    out[:] = pypto.exp(a)


@pypto.frontend.jit
def matmul_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    out[:] = pypto.matmul(a, b, out_dtype=pypto.DT_FP16)


@pypto.frontend.jit
def one_hot_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(4, 5, 32)
    out[:] = pypto.one_hot(a, 5)


@pypto.frontend.jit
def kernel_dynamic_reshape(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    sq = 128
    d = 64
    b = q.shape[0]
    q_reshaped = pypto.reshape(q, [b * sq, d])  # inplace=False, dynamic shape will raise an error.

    pypto.set_vec_tile_shapes(64, 64)
    for idx in pypto.loop(b):
        temp = pypto.view(q_reshaped, [sq, d], [idx * sq, 0])
        temp1 = pypto.exp(temp)
        pypto.assemble(temp1, [idx * sq, 0], out)


# ------------------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------------------

def test_kernel_dynamic_shape_error():
    """Test that dynamic shape (-1) in exp kernel operand causes CheckTensorDynamicShape error."""
    device = f"npu:{DEVICE_ID}"

    a = torch.ones(1, 8, dtype=torch.float32, device=device)
    out = torch.zeros(1, 8, dtype=torch.float32, device=device)

    with pytest.raises(Exception, match="has invalid shape value: -1"):
        kernel_with_dynamic(a, out)


def test_matmul_dynamic_shape_error():
    """Test that dynamic shape (-1) in matmul kernel operand causes CheckTensorDynamicShape error."""
    device = f"npu:{DEVICE_ID}"

    m, k, n = 128, 128, 128
    a_npu = torch.rand([m, k], dtype=torch.float16, device=device)
    b_npu = torch.rand([k, n], dtype=torch.float16, device=device)
    out_npu = torch.zeros([m, n], dtype=torch.float16, device=device)

    with pytest.raises(Exception, match="operand1 dim\\[0\\] = -1, must be > 0"):
        matmul_kernel(a_npu, b_npu, out_npu)


def test_one_hot_dynamic_shape_error():
    """Test that dynamic shape (-1) in one_hot kernel operand causes CheckTensorDynamicShape error."""
    device = f"npu:{DEVICE_ID}"

    input_npu = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    out_npu = torch.zeros([3, 5], dtype=torch.int32, device=device)

    with pytest.raises(Exception, match="has invalid shape value: -1"):
        one_hot_kernel(input_npu, out_npu)


def test_dynamic_reshape_error():
    """Test that dynamic reshape without inplace=True causes an error."""
    device = f"npu:{DEVICE_ID}"

    q = torch.ones(1, 128, 64, dtype=torch.float32, device=device)
    out = torch.zeros(128, 64, dtype=torch.float32, device=device)

    with pytest.raises(Exception, match="reshape\\(\\) requires integer shape when 'inplace=False'"):
        kernel_dynamic_reshape(q, out)


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.npu.set_device(DEVICE_ID)

    test_kernel_dynamic_shape_error()
    test_matmul_dynamic_shape_error()
    test_one_hot_dynamic_shape_error()
    test_dynamic_reshape_error()

    logging.info("All dynamic shape error tests passed.")
