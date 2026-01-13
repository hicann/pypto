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
import sys
import pypto
import pytest
import torch
import torch_npu


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../examples/models/deepseek_v32_exp/utils'))
from compare import compare


num, d, eps = 4, 512, 1e-6
num2 = (2 + num) * num


def gen_data(t=16):
    x_ori = torch.empty((t, num2), dtype=torch.bfloat16).uniform_(-1, 1)
    scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1)
    hc_base_ori = torch.empty((num2,), dtype=torch.float32).uniform_(-1, 1)

    base = hc_base_ori.reshape(1, num2)
    x = x_ori.to(torch.float32) - 0
    pre = x[:, :num] * scale[0] + base[:, :num]  # (t, 4)
    pre = x = 1 / (1 + pre) + eps   # (t, 4)
    res = pre.to(torch.bfloat16)
    
    return x_ori, scale, hc_base_ori, res


def sigmoid(x: pypto.Tensor) -> pypto.Tensor:
    ones = pypto.full(x.shape, 1.0, x.dtype, valid_shape=x.shape)
    x = pypto.div(ones, x + 1.0)
    return x


@pypto.jit
def kernel(x: pypto.Tensor, scale: pypto.Tensor, base_: pypto.Tensor, y: pypto.Tensor):
    pypto.set_debug_options(runtime_debug_mode=1)
    
    pypto.set_vec_tile_shapes(64, 64)
    pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])
    
    tile_t = 16
    real_t = x.shape[0]
    loop_t_times = (real_t + tile_t - 1) // tile_t
    
    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [real_t, num2], inplace=True)
        base = pypto.reshape(base_, [1, num2], inplace=True)
    
    for t_idx in pypto.loop(loop_t_times, name="t_loop", idx_name="t_idx"):
        x_view = pypto.view(x_2d, [tile_t, num * d], [t_idx * tile_t, 0])
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        rms_res = x_fp32
        pre = rms_res[:, :num] * (scale[0: 1].reshape([1, 1]).expand_clone([tile_t, 1])) + base[:, :num]  # (tile_t, 4)
        ones = pypto.full(pre.shape, 1.0, pre.dtype, valid_shape=pre.shape)
        pre = pypto.div(ones, pre + 1.0)
        y[t_idx * tile_t:, :] = pypto.cast(pre, pypto.DT_BF16)
        
        
def test_main(t=16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)
    
    x, scale, base, y_gd = gen_data(t)
    
    y = torch.zeros_like(y_gd).to(device=f'npu:{device_id}')
    
    in_outs = {
        x.to(device=f'npu:{device_id}'): [0],
        scale.to(device=f'npu:{device_id}'): None,
        base.to(device=f'npu:{device_id}'): None,
        y: [0]
    }
    
    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    kernel(*pto_in_outs)
    torch_npu.npu.synchronize()
    
    y = y .cpu()
    
    compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
    
    
if __name__ == "__main__":
    test_main(16)