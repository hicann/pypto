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
"""A direct top-level ``@pypto.frontend.jit`` kernel file whose module has ``from __future__
import annotations`` and N=3 tensor params. Deployed via ``JitCallableWrapper._get_signature`` (the
live-``__annotations__`` path), which round-trips ONLY because the emitted snippet carries NO
``__future__`` line — so the kernel's def-time annotations rebuild LIVE (``pypto.Tensor`` objects with
``.to_tensor``), not as strings.
"""
from __future__ import annotations

import torch

import pypto


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def pep563_direct_add_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)


def pep563_infer_shape(input0_shape: torch.Size, input1_shape: torch.Size) -> torch.Size:
    return input0_shape


def pep563_infer_dtype(input0_dtype: torch.dtype, input1_dtype: torch.dtype) -> torch.dtype:
    return input0_dtype
