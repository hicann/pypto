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
"""
import os
import pytest
import torch
import torch_npu
import pypto
import numpy as np

verify_options = {"enable_pass_verify": True,
                  "pass_verify_save_tensor": True,
                 }


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU},
                    verify_options=verify_options
                    )
def add_dyn_kernel(
        x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16), 
        y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16)):
    first_dim, second_dim = x.shape
    view_shape, tile_shape = (64, 64), (32, 32)

    first_view_shape, second_view_shape = view_shape
    for b_idx in pypto.loop(int(np.ceil(first_dim / view_shape[0])), name="LOOP_L0", idx_name="b_idx"):
        for s_idx in pypto.loop(int(np.ceil(second_dim / view_shape[1])), name="LOOP_L1", idx_name="s_idx"):
            tile_tensor_0 = pypto.view(
                x, view_shape,
                [b_idx * first_view_shape, s_idx * second_view_shape]
            )
            tile_tensor_1 = pypto.view(
                y, view_shape,
                [b_idx * first_view_shape, s_idx * second_view_shape]
            )
            pypto.set_vec_tile_shapes(*tile_shape)  # 32*32
            if b_idx < 2:
                res = ((tile_tensor_0 * (tile_tensor_0 + tile_tensor_1)) - tile_tensor_1) * tile_tensor_1
            else:
                res = tile_tensor_0
            pypto.assemble(
                res,
                [b_idx * first_view_shape, s_idx * second_view_shape],
                out,
            )
            del res, tile_tensor_0, tile_tensor_1


def test_verify_dyn():
    shape = [72, 144]

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))

    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)
    a = torch.rand(shape, dtype=torch.float16, device=device)
    b = torch.rand(shape, dtype=torch.float16, device=device)
    output_data = torch.zeros(shape, dtype=torch.float16, device=device)
    golden = ((a * (a + b)) - b) * b
    golden_cpu = golden.cpu()
    pypto.set_verify_golden_data(goldens=[None, None, golden_cpu])
    add_dyn_kernel(a, b, output_data)
    assert torch.allclose(output_data, golden)
