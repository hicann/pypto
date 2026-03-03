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
import json
from pathlib import Path

import pytest
import pypto
import torch
import torch_npu
from st.test_swim_line import matmul_add


def count_core_types(base_dir="./output"):
    # 获取日志存储目录
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    latest_dir = sorted(subdirs, reverse=True)

    file_path = str(latest_dir[0] / "tilefwk_L1_prof_data.json")
    for folder in latest_dir:
        target_path = folder / "tilefwk_L1_prof_data.json"
        if target_path.is_file():
            file_path = str(target_path)
            break

    # 读取json文件统计AIC和AIV使用的个数
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    data = json.loads(content)
    aic_count = 0
    aiv_count = 0
    for item in data:
        core_type = item.get('coreType')
        if core_type == 'AIC':
            aic_count += 1
        elif core_type == 'AIV':
            aiv_count += 1

    return aic_count, aiv_count


def kernel_func(device_id):
    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 2

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

    for idx in range(count):
        # get data and compare result
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)
    
    aic_count, aiv_count = count_core_types("./output")

    return aic_count, aiv_count


@pytest.mark.forked
def test_not_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    kernel_func(device_id)


@pytest.mark.forked
def test_rts_stream_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    stream1 = torch.npu.current_stream()
    torch.npu.set_stream_limit(stream1, 8, 27)
    
    aic_count, aiv_count = kernel_func(device_id)


@pytest.mark.forked
def test_rts_device_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch.npu.set_device_limit(device_id, 8, 16)

    aic_count, aiv_count = kernel_func(device_id)


@pytest.mark.forked
def test_rts_device_stream_control_cores():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    torch.npu.set_device_limit(device_id, 15, 27)

    stream1 = torch.npu.current_stream()
    torch.npu.set_stream_limit(stream1, 8, 30)

    aic_count, aiv_count = kernel_func(device_id)