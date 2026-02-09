#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


@pypto.jit()
def reshape_kernel(in_tensor, out_tensor):
    b = 3
    n1 = 64
    d = 64

    pypto.set_vec_tile_shapes(1, 64, 64)

    tile_b = 1
    real_b = in_tensor.shape[0]
    loop_b_times = (real_b + tile_b - 1) // tile_b

    for b_idx in pypto.loop(loop_b_times, name="b_loop", idx_name="b_idx"):
        a0 = pypto.view(in_tensor, [tile_b, n1, d], [b_idx, 0, 0])
        a_ = pypto.reshape(a0, [tile_b * n1, -1])
        pypto.set_vec_tile_shapes(64, 64)
        a1 = pypto.add(a_, 1.0)
        a1_ = pypto.reshape(a1, [tile_b, n1, d])
        pypto.set_vec_tile_shapes(1, 64, 64)
        pypto.assemble(a1_, [b_idx, 0, 0], out_tensor)


def test_reshape():
    b = 3
    n1 = 64
    d = 64
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    # prepare data
    input_cpu = torch.rand((b, n1, d), dtype=torch.float32)
    output_cpu = torch.ones((b, n1, d), dtype=torch.float32)
    # def inputs and outputs
    input_npu = input_cpu.to(device=f'npu:{device_id}')
    output_npu = output_cpu.to(device=f'npu:{device_id}')
    pto_input = pypto.from_torch(input_npu, "IN", dynamic_axis=[0])
    # , dynamic_axis=[0]
    pto_output = pypto.from_torch(output_npu, "OUT", dynamic_axis=[0])

    # compute on npu
    reshape_kernel(pto_input, pto_output)
    torch_npu.npu.synchronize()

    output_cpu = output_npu.cpu()

    ## golden
    output_golde = (input_cpu + 1)

    assert_allclose(np.array(output_cpu),
                    np.array(output_golde),
                    rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_reshape()