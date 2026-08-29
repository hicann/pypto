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
"""Batch scale shape validation tests for pypto.scaled_mm."""

from contextlib import contextmanager

import pytest

import pypto
from pypto.error import PyptoError


@contextmanager
def npuarch(arch: str):
    try:
        old_arch = pypto.platform.npuarch
        pypto.platform.npuarch = arch
        yield
    finally:
        pypto.platform.npuarch = old_arch


@npuarch("DAV_3510")
def test_scaled_mm_3d_rejects_shared_scale():
    mat_a = pypto.tensor([2, 16, 128], pypto.DT_FP8E4M3, "mat_a")
    mat_b = pypto.tensor([2, 128, 32], pypto.DT_FP8E4M3, "mat_b")
    scale_a = pypto.tensor([16, 2, 2], pypto.DT_FP8E8M0, "scale_a")
    scale_b = pypto.tensor([2, 32, 2], pypto.DT_FP8E8M0, "scale_b")

    with pytest.raises(PyptoError, match=r"input_dim \+ 1 \(4\)"):
        pypto.scaled_mm(mat_a, mat_b, pypto.DT_FP16, scale_a, scale_b)


@npuarch("DAV_3510")
def test_scaled_mm_4d_accepts_paired_scale_batch_shape():
    mat_a = pypto.tensor([2, 1, 16, 128], pypto.DT_FP8E4M3, "mat_a")
    mat_b = pypto.tensor([1, 3, 128, 32], pypto.DT_FP8E4M3, "mat_b")
    scale_a = pypto.tensor([2, 1, 16, 2, 2], pypto.DT_FP8E8M0, "scale_a")
    scale_b = pypto.tensor([1, 3, 2, 32, 2], pypto.DT_FP8E8M0, "scale_b")

    with pypto.function("SCALED_MM_BATCH", mat_a, mat_b):
        pypto.set_cube_tile_shapes([32, 32],[128, 128],[32, 32])
        result = pypto.scaled_mm(mat_a, mat_b, pypto.DT_FP16, scale_a, scale_b)

    assert result.shape == [2, 3, 16, 32]


@npuarch("DAV_3510")
def test_scaled_mm_4d_rejects_output_batch_scale_shape():
    mat_a = pypto.tensor([2, 1, 16, 128], pypto.DT_FP8E4M3, "mat_a")
    mat_b = pypto.tensor([1, 3, 128, 32], pypto.DT_FP8E4M3, "mat_b")
    scale_a = pypto.tensor([2, 3, 16, 2, 2], pypto.DT_FP8E8M0, "scale_a")
    scale_b = pypto.tensor([1, 3, 2, 32, 2], pypto.DT_FP8E8M0, "scale_b")

    with pytest.raises(PyptoError, match="Scale batch dimension mismatch at axis 1"):
        pypto.scaled_mm(mat_a, mat_b, pypto.DT_FP16, scale_a, scale_b)
