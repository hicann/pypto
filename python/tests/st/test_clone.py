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

import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose

b = 3
s = 4
n1 = 64
d = 64


@pypto.jit(
    host_options={"only_codegen": True},
)
def kernel_func(in_tensor, out_tensor):
    pypto.set_vec_tile_shapes(1, 1, 64, 64)

    for b_idx in pypto.loop(b, name="b_loop", idx_name="b_idx"):
        for s_idx in pypto.loop(s, name="s_loop", idx_name="s_idx"):
            a0 = pypto.view(in_tensor, [1, 1, n1, d], [b_idx, s_idx, 0, 0])
            a1 = pypto.add(a0, 1.0)
            a2 = pypto.reshape(a1, [1, 1, n1 * d])
            a3 = a2.clone()
            pypto.assemble(a3, [b_idx, s_idx, 0], out_tensor)


def test_clone():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    # prepare data
    input_cpu = torch.rand((b, s, n1, d), dtype=torch.float32)
    output_cpu = torch.ones((b, s, n1*d), dtype=torch.float32)
    # def inputs and outputs
    input_npu = input_cpu.to(device=f'npu:{device_id}')
    output_npu = output_cpu.to(device=f'npu:{device_id}')
    pto_inputs = [pypto.from_torch(input_npu, "IN")]
    pto_outputs = [pypto.from_torch(output_npu, "OUT")]
    # compute on npu
    kernel_func(pto_inputs[0], pto_outputs[0])
    torch_npu.npu.synchronize()

    output_cpu = output_npu.cpu()

    ## golden
    output_golde = input_cpu.reshape((b, s, n1*d)) + 1

    assert_allclose(np.array(output_cpu),
                    np.array(output_golde),
                    rtol=1e-3, atol=1e-3)
