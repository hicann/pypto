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

import os
import pypto

import torch
import torch_npu


@pypto.jit
def cust_dyn_func_add(a, b, c, tiling=None):
    pypto.set_vec_tile_shapes(tiling, tiling)
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        c.move(pypto.add(a, b))


@pypto.jit
def cust_dyn_func_sub(a, b, c, tiling=None):
    pypto.set_vec_tile_shapes(tiling, tiling)
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        c.move(pypto.sub(a, b))


def device_run(is_run_add):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    tiling = 32
    n, m = tiling * 1, tiling * 1

    # prepare data
    a_rawdata = torch.ones((n, m)) * 2
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    b_rawdata = torch.ones((n, m))
    b_data = b_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    c_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')

    # def inputs and outputs
    inputs = [a_data, b_data]
    outputs = [c_data]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    if is_run_add:
        cust_dyn_func_add(pto_inputs[0], pto_inputs[1], pto_outputs[0], tiling)
        torch_npu.npu.synchronize()

        golden = torch.ones((n, m)) * 3
        assert torch.allclose(golden.int(), c_data.cpu(), atol=1e-5)
    else:
        cust_dyn_func_sub(pto_inputs[0], pto_inputs[1], pto_outputs[0], tiling)
        torch_npu.npu.synchronize()

        golden = torch.ones((n, m))
        assert torch.allclose(golden.int(), c_data.cpu(), atol=1e-5)


def test_run_multi_jit():
    device_run(True)
    device_run(False)
    device_run(True)