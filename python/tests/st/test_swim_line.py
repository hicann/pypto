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
"""
profiling of aicpu pref  test for PyPTO
"""
import json
from typing import List, Dict
import contextlib
import os

import pypto
import pytest
import torch
import torch_npu
import pytest


@pypto.jit(debug_options=dict(runtime_debug_mode=1))
def matmul_add(in_tensor0, in_tensor1, in_tensor2, out_tensor):
    a = in_tensor0	 
    b = in_tensor1	 
    c = in_tensor2	 
    d = out_tensor
    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes(
        [tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


def test_device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 1

    a_rawdata = torch.tensor([[1] * k] * n)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')

    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.int32,
                             device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        inputs = [a_data, b_data, c_data]
        outputs = [d_data]
        pto_inputs = [pypto.from_torch(
            tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(
            tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        matmul_add(pto_inputs[0], pto_inputs[1], pto_inputs[2], pto_outputs[0])

    torch_npu.npu.synchronize()

    aicpu_json_path = pypto.pypto_impl.LogTopFolder() + "/aicpu_dev_pref.json"
    assert os.path.exists(aicpu_json_path), "Could not Get aicpu perf"
    
    with open(aicpu_json_path, 'r', encoding='utf-8') as f:
        core_list: List[Dict] = json.load(f)
        for core in core_list:
            tasks = core.get("tasks", [])
            assert len(tasks) > 0, "Could not Get aicpu perf"
    swim_lane_json_path = pypto.pypto_impl.LogTopFolder() + "/merged_swimlane.json"
    assert os.path.exists(swim_lane_json_path), "Could not Get swim lane"
    assert os.path.getsize(swim_lane_json_path) > 0, "Get swim lane is null"
