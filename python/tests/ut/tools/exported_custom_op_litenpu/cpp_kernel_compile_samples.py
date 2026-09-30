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
"""Real-file kernel/infer samples following the ``create_*_kernel`` factory contract.

These live in a real module so ``inspect.getsource`` captures them (the compile
snippet generator and pypto's jit parser both need real source on disk). They are
NOT attached to an :class:`ExportedCustomOp` here; tests pass them to
``kernel_snippet.build_kernel_compile_snippet`` directly. The factory follows the
parameterized ``create_*_kernel(shape, dtype, soc_version)`` contract and references
only ``add_kernel_body`` / ``add_infer_tileshape`` and ``torch`` / ``pypto``.
"""

import torch

import pypto


def add_kernel_body(input0, input1):
    pypto.set_vec_tile_shapes(*add_infer_tileshape(input0, input1))
    return input0 + input1


def add_infer_tileshape(input0, input1):
    return (1, 4, 1, 64)


# Annotated like a real ``infer_shape`` / ``infer_dtype`` (the OpDef and the embedded
# snippet both parse the return annotation to derive the output arity).
def add_infer_shape(input0_shape: torch.Size, input1_shape: torch.Size) -> torch.Size:
    return input0_shape


def add_infer_dtype(input0_dtype: torch.dtype, input1_dtype: torch.dtype) -> torch.dtype:
    return input0_dtype


def create_add_kernel(shape, dtype, soc_version, run_mode=pypto.RunMode.SIM):
    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": run_mode},
    )
    def add_kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        output.move(add_kernel_body(input0, input1))

    return add_kernel
