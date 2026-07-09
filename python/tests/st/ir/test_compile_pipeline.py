# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""System tests for new IR compile pipeline JIT execution."""

import os

import pypto
import torch


@pypto.frontend.jit(new_ir=True,
    host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH})
def assemble_kernel(
    a: pypto.Tensor([pypto.DYN, 128]),
    out: pypto.Tensor([pypto.DYN, 128]),
):
    assert "a" == a.name
    assert "out" == out.name
    pypto.set_vec_tile_shapes(64, 128)
    aux_mat = pypto.tensor([129, 128], pypto.DT_FP32, name="aux_mat")
    pypto.assemble(pypto.full([129, 128], 0.0, pypto.DT_FP32), [0, 0], aux_mat)
    a = pypto.view(a, [129, 128], [0, 0])
    pypto.assemble(a, [0, 0], aux_mat)
    pypto.assemble(aux_mat + a, [0, 0], out)


def test_assemble_jit_compile():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    a_torch = torch.empty([129, 128], dtype=torch.float32, device=device)
    out_torch = torch.empty([129, 128], dtype=torch.float32, device=device)

    assemble_kernel(a_torch, out_torch)


if __name__ == "__main__":
    test_assemble_jit_compile()
