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
import os
import pypto
import torch


def compile_stage(comp_stage):
    @pypto.frontend.jit(
        host_options={"compile_stage": comp_stage},
        runtime_options={"run_mode": 0}
    )
    def compile_func(
        a: pypto.Tensor((4, 4), pypto.DT_FP32),
        b: pypto.Tensor((4, 4), pypto.DT_FP32),
        c: pypto.Tensor((4, 4), pypto.DT_FP32),
    ) -> None:
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b

    return compile_func


def test_all_compile_stages():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    a = torch.ones((4, 4), dtype=torch.float32, device=f"npu:{device_id}") * 2
    b = torch.ones((4, 4), dtype=torch.float32, device=f"npu:{device_id}") * 3
    c = torch.ones((4, 4), dtype=torch.float32, device=f"npu:{device_id}")
    compile_stage(pypto.CompStage.TENSOR_GRAPH)(a, b, c)
    compile_stage(pypto.CompStage.TILE_GRAPH)(a, b, c)
    compile_stage(pypto.CompStage.EXECUTE_GRAPH)(a, b, c)
    compile_stage(pypto.CompStage.CODEGEN_INSTRUCTION)(a, b, c)
    compile_stage(pypto.CompStage.CODEGEN_BINARY)(a, b, c)
    assert True


if __name__ == "__main__":
    test_all_compile_stages()
