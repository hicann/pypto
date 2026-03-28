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
import torch
import torch_npu
import pypto


@pypto.frontend.jit()
def loop_scope(a: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
               b: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32],
               result: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_INT32]):
    pypto.set_vec_tile_shapes(64, 64)

    for _ in pypto.loop(1, name="s0", idx_name="k"):
        pypto.set_vec_tile_shapes(32, 32)
        result.move(a + b)

    for _ in pypto.loop(1, name="s0", idx_name="k"):
        assert [64, 64] == pypto.get_vec_tile_shapes()
        result.move(result + b)


def test_loop_scope():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)

    # prepare data
    a_data = torch.ones((n, m), dtype=torch.int32, device=f'npu:{device_id}') * 2
    b_data = torch.ones((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    result = torch.zeros(shape, dtype=torch.int32, device=f'npu:{device_id}')

    loop_scope(a_data, b_data, result)
    torch_npu.npu.synchronize()

    golden = torch.ones((n, m), dtype=torch.int32) * 4
    assert torch.allclose(golden, result.cpu(), atol=1e-5)

if __name__ == "__main__":
    test_loop_scope()