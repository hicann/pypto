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
    @pypto.jit(
        host_options={"compile_stage": comp_stage},
        runtime_options={"run_mode": 1}
    )
    def compile_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b

    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    compile_func(input_a, input_b, output_c)


def test_all_compile_stages():
    compile_stage(pypto.CompStage.TENSOR_GRAPH)
    compile_stage(pypto.CompStage.TILE_GRAPH)
    compile_stage(pypto.CompStage.EXECUTE_GRAPH)
    compile_stage(pypto.CompStage.CODEGEN_INSTRUCTION)
    compile_stage(pypto.CompStage.CODEGEN_BINARY)
    assert True


if __name__ == "__main__":
    test_all_compile_stages()