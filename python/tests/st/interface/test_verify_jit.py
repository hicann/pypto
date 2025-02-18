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
import pytest
import torch
import torch_npu
import pypto

verify_options = {"enable_pass_verify": True,
                  "pass_verify_save_tensor": True,
                 }


@pypto.jit(verify_options=verify_options)
def add(a, b, c):
    for _ in pypto.loop(1):
        pypto.set_vec_tile_shapes(16, 16)
        c[:] = a + b


def test_verify_full_options():
    a = torch.ones((64, 64))
    b = torch.ones((64, 64))
    c = torch.zeros((64, 64))

    golden = a + b
    
    device_id = 0
    torch.npu.set_device(device_id)
    inputs = [a, b]
    outputs = [c]

    inputs = [a.to(f"npu:{device_id}"), a.to(f"npu:{device_id}")]
    outputs = [c.to(f"npu:{device_id}")]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]

    add(*pto_inputs, *pto_outputs)

    assert torch.allclose(outputs[0].cpu(), golden)
