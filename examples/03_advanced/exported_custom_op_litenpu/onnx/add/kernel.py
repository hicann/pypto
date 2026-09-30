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
"""Kernel for the add export demo — the compute the raw kernel needs to build.

The @pypto.frontend.jit kernel below IS the deployable kernel. Its tensor annotations are left
empty (`pypto.Tensor([...])`) so each tensor's shape and dtype are supplied at deploy/run instead of
being written by hand. This file holds only what building the raw kernel needs: the kernel body.
Shape/dtype inference (infer_shape/infer_dtype) lives next door in ``op.py``.
"""
import pypto


# the pypto compute a kernel author writes; JIT-built at deploy, run by the backend.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def add_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)
