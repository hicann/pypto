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
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


@pypto.jit
def cust_dyn_func(a, b, c, tiling=None):
    pypto.set_vec_tile_shapes(32, 32)
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        c.move(pypto.add(a, b))


class Network(torch.nn.Module):
    def forward(self, data1, data2):
        add_01 = torch.add(data1, data2)

        inputs = [add_01, data2]
        outputs = [data2]
        pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        cust_dyn_func(pto_inputs[0], pto_inputs[1], pto_outputs[0], 32)

        data2 = torch.sub(data2, add_01)
        data2 = torch.add(data2, add_01)

        inputs = [data2, data1]
        outputs = [data2]
        pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        cust_dyn_func(pto_inputs[0], pto_inputs[1], pto_outputs[0], 32)
        return data2


def compute_golden(data1, data2):
    add_01 = torch.add(data1, data2)
    data2 = torch.add(add_01, data2)
    data2 = torch.sub(data2, add_01)
    data2 = torch.add(data2, add_01)
    data2 = torch.add(data2, data1)
    return data2


def test_select_experts():
    # 1. 设置参数
    input0 = torch.from_numpy(np.random.uniform(-5, 5, size=(256, 256))).to(torch.int32)
    input1 = torch.from_numpy(np.random.uniform(-5, 5, size=(256, 256))).to(torch.int32)
    # run golden
    golden_out = compute_golden(input0, input1)
    # run npu
    torch_npu.npu.set_device(int(os.environ.get('TILE_FWK_DEVICE_ID', 0)))
    input0 = input0.npu()
    input1 = input1.npu()
    npu_mode = Network().npu()

    assert not torch_npu.npu.is_current_stream_capturing()

    s = torch.npu.Stream()
    with torch.npu.stream(s):
        g = torch_npu.npu.NPUGraph()
        torch_npu.npu.empty_cache()
        assert not torch_npu.npu.is_current_stream_capturing()
        # 开始捕获
        g.capture_begin()
        for _ in range(1):
            npu_out = npu_mode(input0, input1)
        assert torch_npu.npu.is_current_stream_capturing()
        g.capture_end()
    torch_npu.npu.current_stream().wait_stream(s)
    # 执行
    g.replay()
    stream = torch_npu.npu.current_stream()
    stream.synchronize()
    g.reset()

    npu_out = npu_out.cpu().detach().numpy()
    golden_out = golden_out.cpu().detach().numpy()
    # 精度校验
    assert_allclose(npu_out, golden_out, rtol=5e-3, atol=5e-3)