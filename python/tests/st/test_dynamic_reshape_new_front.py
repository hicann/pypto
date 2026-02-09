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
"""
"""
import os
import pypto
import pytest
import numpy as np
import torch
from numpy.testing import assert_allclose
import torch_npu


def reshape_kernel_wrapper(shape):
    @pypto.frontend.jit()
    def reshape_kernel(in_tensor: pypto.Tensor(shape, pypto.DT_FP32)) -> pypto.Tensor(shape, pypto.DT_FP32):
        b = 3
        n1 = 64
        d = 64

        pypto.set_vec_tile_shapes(64, 64)

        tile_b = 1
        real_b = in_tensor.shape[0]
        loop_b_times = (real_b + tile_b - 1) // tile_b

        tmp = pypto.Tensor([real_b * n1, d], dtype=pypto.DT_FP32)
        for b_idx in pypto.loop(loop_b_times, name="b_loop", idx_name="b_idx"):
            in_2d = pypto.reshape(in_tensor, [real_b * n1, d], inplace=True)
            a0 = pypto.view(in_2d, [tile_b * n1, d], [b_idx * n1, 0])
            a1 = a0 + 1.0
            pypto.assemble(a1, [b_idx * n1, 0], tmp)
            result = pypto.reshape(tmp, [real_b, n1, d], inplace=True)

        return result

    return reshape_kernel


def test_reshape():
    b = 3
    n1 = 64
    d = 64
    shape = (b, n1, d)
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    # prepare data
    input_cpu = torch.rand((b, n1, d), dtype=torch.float32)
    input_npu = input_cpu.to(device=f'npu:{device_id}')

    result = reshape_kernel_wrapper(shape)(input_npu)
    torch_npu.npu.synchronize()

    output_cpu = result.cpu()

    ## golden
    output_golde = (input_cpu + 1)

    assert_allclose(np.array(output_cpu),
                    np.array(output_golde),
                    rtol=1e-3, atol=1e-3)


def reshape_infer_shape(x_shape: tuple):
    @pypto.frontend.jit()
    def kernel(x: pypto.Tensor(x_shape, pypto.DT_FP32),) -> (pypto.Tensor([4, 2, 4], pypto.DT_FP32)):
        pypto.set_vec_tile_shapes(4, 8)
        y = pypto.reshape(x, [x.shape[0], 2, -1])
        return y
    return kernel


def test_reshape_infer_shape():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    x = torch.rand((4, 8), dtype=torch.float32)
    x_tensor = x.npu()
    y_pypto = reshape_infer_shape(x_tensor.shape)(x_tensor)
    torch_npu.npu.synchronize()
    y = y_pypto.cpu()
    assert_allclose(np.array(y.flatten()), np.array(x.flatten()), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_reshape_infer_shape()