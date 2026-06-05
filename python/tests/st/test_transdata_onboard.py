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
import pypto
from numpy.testing import assert_allclose
import torch


def test_transdata_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    group = 1
    intputShape = (1, 8, 1, 8)
    outputShape = (1, 1, 1, 8, 8)
    view_shape = (1, 8, 1, 8)
    tile_shape = (1, 8, 1, 8)
    pypto.runtime._device_init()

    input = pypto.tensor(intputShape, pypto.DT_INT32, "pypto_TENSOR_input")
    output = pypto.tensor(outputShape, pypto.DT_INT32, "pypto_TENSOR_output")

    with pypto.function("MAIN", input, output):
        for b_idx in pypto.loop(1, name="b0", idx_name="bidx"):
            view_tensor_a = pypto.view(input, view_shape,
                                        [0, 0, 0, 0],
                                        valid_shape=[pypto.symbolic_scalar(input.shape[0]),
                                            pypto.symbolic_scalar(input.shape[1]),
                                            pypto.symbolic_scalar(input.shape[2]),
                                            pypto.symbolic_scalar(input.shape[3]),
                                        ],
                                        )
            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1], tile_shape[2], tile_shape[3])
            view_tensor_b = pypto.transdata(view_tensor_a, 2, group)
            pypto.assemble(view_tensor_b, [0, 0, 0, 0, 0], output)
            del view_tensor_a
            del view_tensor_b
    assert isinstance(output, pypto.tensor)

    a_tensor = torch.randint(
        low=-10, high=10, size=intputShape, dtype=torch.int32)
    b_tensor = torch.zeros(outputShape, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = a_tensor.reshape(1,1,8,1,8).permute(0,1,3,4,2).contiguous()
    assert_allclose(b_tensor.flatten(), golden.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()

if __name__ == "__main__":
    test_transdata_onboard()
