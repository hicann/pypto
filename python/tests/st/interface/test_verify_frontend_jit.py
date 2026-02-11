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


def create_add_dyn_kernel(shape: tuple, run_mode: str = "npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode},
                        verify_options=verify_options
                        )
    def add_dyn_kernel(
            x: pypto.Tensor(shape, pypto.DT_FP16), 
            y: pypto.Tensor(shape, pypto.DT_FP16)) -> pypto.Tensor(shape, pypto.DT_FP16):
        first_dim, second_dim = x.shape
        view_shape, tile_shape = (64, 64), (32, 32)

        first_view_shape, second_view_shape = view_shape
        out = pypto.Tensor(shape, pypto.DT_FP16)
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
        return out
    return add_dyn_kernel


def test_verify_dyn():
    shape = [72, 144]
    run_mode = "npu"

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    torch.npu.set_device(device_id)
    a = torch.rand(shape, dtype=torch.float16, device=device)
    b = torch.rand(shape, dtype=torch.float16, device=device)
    golden = ((a * (a + b)) - b) * b
    golden_cpu = golden.cpu()
    pypto.set_verify_golden_data(goldens=[None, None, golden_cpu])
    output_data = create_add_dyn_kernel(shape)(a, b)
    assert torch.allclose(output_data, golden)
