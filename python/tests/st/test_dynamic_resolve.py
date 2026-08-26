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

import logging
import os

import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

V64 = 64
V128 = 128


@pypto.frontend.jit()
def resolve_kernel(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    input_c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    for _ in pypto.loop(1, name="Step0"):
        tensor_list = []
        for j in range(V64):
            t = input_b[:, V128 * j:V128 * (j + 1)]
            mm = pypto.matmul(input_a, t, pypto.DT_FP32, b_trans=True)
            tensor_list.append(mm)
        mm_concat = pypto.concat(tensor_list, -1)
        output[:] = pypto.add(input_c, mm_concat)


@pytest.mark.soc("910")
@pypto.options(
    pass_options={
        "mg_copyin_upper_bound": 100 * 1024 * 1024,
        "pg_lower_bound": 1024,
        "cube_l1_reuse_setting": {-1: 32},
        "pg_parallel_lower_bound": 2,
        "vec_nbuffer_setting": {-1: 16},
    },
    vec_tile_shapes=[V64, V128],
    cube_tile_shapes=[[V64, V64], [V128, V128], [V128, V128]],
)
def test_dynamic_resolve():
    torch.npu.set_device(ST_DEVICE)

    input_a = torch.full((V64, V128), 2.0, dtype=torch.bfloat16, device=ST_DEVICE)

    input_b = torch.zeros((V128, V128 * V64), dtype=torch.bfloat16, device=ST_DEVICE)
    for col in range(V128 * V64):
        input_b[:, col] = float(col // V128)

    input_c = torch.full((V64, V128 * V64), 3.0, dtype=torch.float32, device=ST_DEVICE)
    output = torch.zeros((V64, V128 * V64), dtype=torch.float32, device=ST_DEVICE)

    golden = input_c.to(torch.float32)
    tensor_list = []
    for j in range(V64):
        t = input_b[:, V128 * j:V128 * (j + 1)].to(torch.float32)
        mm = input_a.to(torch.float32) @ t.T
        tensor_list.append(mm)
    mm_concat = torch.cat(tensor_list, dim=-1)
    golden = golden + mm_concat

    resolve_kernel(input_a, input_b, input_c, output)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), golden.cpu(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_dynamic_resolve()
    logging.info("test_dynamic_resolve passed!")
