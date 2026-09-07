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
""" """

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


def test_matrix_matmul():
    dtype = pypto.DT_FP32
    a = pypto.tensor((32, 64), dtype, "A")
    b = pypto.tensor((64, 32), dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, dtype)
        d = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [32, 32]

    assert isinstance(d, pypto.tensor)
    assert d.shape == [64, 64]


def test_matrix_batch_matmul():
    dtype = pypto.DT_FP32
    a = pypto.tensor((2, 64, 32), dtype, "A")
    b = pypto.tensor((2, 32, 64), dtype, "B")
    c = None

    with pypto.function("BATCH_MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, dtype)
        d = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [2, 64, 64]

    assert isinstance(d, pypto.tensor)
    assert d.shape == [2, 32, 32]


def test_matrix_matmul_with_syntactic_sugar():
    dtype = pypto.DT_FP16
    a = pypto.tensor((64, 32), dtype, "A")
    b = pypto.tensor((32, 64), dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = a @ b

    assert isinstance(c, pypto.tensor)
    assert c.dtype == pypto.DT_FP16
    assert c.shape == [64, 64]


def test_matrix_matmul_with_tensor_interface():
    input_dtype = pypto.DT_INT8
    out_dtype = pypto.DT_INT32
    a = pypto.tensor((3, 64, 32), input_dtype, "A")
    b = pypto.tensor((3, 32, 64), input_dtype, "B")
    c = None

    with pypto.function("BATCH_MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = a.matmul(b, out_dtype, a_trans=True, b_trans=True)

    assert isinstance(c, pypto.tensor)
    assert c.dtype == pypto.DT_INT32
    assert c.shape == [3, 32, 32]


@npuarch("DAV_3510")
def test_matrix_matmul_with_trans_mode_cast_rint():
    input_dtype = pypto.DT_FP32
    out_dtype = pypto.DT_FP32
    a = pypto.tensor((64, 32), input_dtype, "A")
    b = pypto.tensor((32, 64), input_dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, out_dtype, extend_params={"trans_mode": pypto.TransMode.CAST_RINT})

    assert isinstance(c, pypto.tensor)
    assert c.shape == [64, 64]


@npuarch("DAV_3510")
def test_matrix_matmul_with_trans_mode_cast_round():
    input_dtype = pypto.DT_FP32
    out_dtype = pypto.DT_FP32
    a = pypto.tensor((64, 32), input_dtype, "A")
    b = pypto.tensor((32, 64), input_dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, out_dtype, extend_params={"trans_mode": pypto.TransMode.CAST_ROUND})

    assert isinstance(c, pypto.tensor)
    assert c.shape == [64, 64]


@pytest.mark.parametrize("arch", ["DAV_1001", "DAV_2201"])
def test_matmul_rejects_hf8_input_on_a2a3(arch):
    a = pypto.tensor((16, 32), pypto.DT_HF8, "a")
    b = pypto.tensor((32, 64), pypto.DT_HF8, "b")

    with npuarch(arch):
        with pytest.raises(PyptoError, match=r"DT_HF8.*not supported on A2/A3 platforms"):
            pypto.matmul(a, b, pypto.DT_FP16)


@npuarch("DAV_3510")
def test_matmul_rejects_hf8_nz_input():
    a = pypto.tensor((16, 32), pypto.DT_HF8, "a", pypto.TileOpFormat.TILEOP_NZ)
    b = pypto.tensor((32, 64), pypto.DT_HF8, "b")

    with pytest.raises(PyptoError, match=r"Input tensor with DT_HF8 must use ND format"):
        pypto.matmul(a, b, pypto.DT_FP16)


@pytest.mark.parametrize("arch", ["DAV_1001", "DAV_2201"])
def test_matmul_rejects_trans_mode_on_a2a3(arch):
    a = pypto.tensor((16, 32), pypto.DT_FP32, "a")
    b = pypto.tensor((32, 64), pypto.DT_FP32, "b")
    extend_params = {'trans_mode': pypto.TransMode.CAST_RINT}

    with npuarch(arch):
        with pytest.raises(PyptoError, match=r"trans_mode.*not supported on A2/A3 platforms"):
            pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)


@npuarch("DAV_3510")
def test_scaled_mm_rejects_nz_scale():
    mat_a = pypto.tensor((16, 128), pypto.DT_FP8E4M3, "mat_a")
    mat_b = pypto.tensor((128, 32), pypto.DT_FP8E4M3, "mat_b")
    scale_a = pypto.tensor((16, 32, 32), pypto.DT_FP8E8M0, "scale_a", pypto.TileOpFormat.TILEOP_NZ)
    scale_b = pypto.tensor((2, 32, 2), pypto.DT_FP8E8M0, "scale_b")

    with pytest.raises(PyptoError, match=r"Scale tensor must use ND format"):
        pypto.scaled_mm(mat_a, mat_b, pypto.DT_FP16, scale_a, scale_b)


@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (pypto.DT_FP16, pypto.DT_BF16),
        (pypto.DT_FP16, pypto.DT_INT32),
        (pypto.DT_BF16, pypto.DT_FP16),
        (pypto.DT_INT8, pypto.DT_FP16),
    ],
)
def test_matmul_rejects_unsupported_out_dtype(in_dtype, out_dtype):
    a = pypto.tensor((16, 32), in_dtype, "a")
    b = pypto.tensor((32, 64), in_dtype, "b")

    with pytest.raises(PyptoError, match=r"Unsupported out_dtype"):
        pypto.matmul(a, b, out_dtype)


def test_matmul_rejects_unsupported_out_dtype_with_scale():
    a = pypto.tensor((16, 32), pypto.DT_FP16, "a")
    b = pypto.tensor((32, 64), pypto.DT_FP16, "b")
    extend_params = {'scale': 0.2}

    with pytest.raises(PyptoError, match=r"Unsupported out_dtype"):
        pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)


@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (pypto.DT_INT8, pypto.DT_FP16),
        (pypto.DT_INT8, pypto.DT_INT8),
        (pypto.DT_FP16, pypto.DT_INT8),
        (pypto.DT_BF16, pypto.DT_INT8),
        (pypto.DT_FP32, pypto.DT_INT8),
    ],
)
def test_matmul_accepts_quant_out_dtype_with_scale(in_dtype, out_dtype):
    a = pypto.tensor((16, 32), in_dtype, "a")
    b = pypto.tensor((32, 64), in_dtype, "b")
    extend_params = {'scale': 0.2}

    with pypto.function("MATMUL_DEQUANT", a, b):
        pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
        c = pypto.matmul(a, b, out_dtype, extend_params=extend_params)

    assert isinstance(c, pypto.tensor)
    assert c.dtype == out_dtype


def test_matmul_dequant_int8_to_fp16():
    a = pypto.tensor((16, 32), pypto.DT_INT8, "a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "b")

    with pypto.function("MATMUL_DEQUANT", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, pypto.DT_FP16, extend_params={'scale': 0.2})

    assert isinstance(c, pypto.tensor)
    assert c.shape == [16, 64]


@npuarch("DAV_3510")
def test_scaled_mm_rejects_unsupported_fp4_out_dtype():
    mat_a = pypto.tensor((16, 128), pypto.DT_FP4_E2M1, "mat_a")
    mat_b = pypto.tensor((128, 32), pypto.DT_FP4_E2M1, "mat_b")
    scale_a = pypto.tensor((16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    scale_b = pypto.tensor((2, 32, 2), pypto.DT_FP8E8M0, "scale_b")

    with pytest.raises(PyptoError, match=r"Unsupported out_dtype"):
        pypto.scaled_mm(mat_a, mat_b, pypto.DT_INT32, scale_a, scale_b)


def test_matmul_rejects_input_outside_combos():
    a = pypto.tensor((16, 32), pypto.DT_FP4_E2M1, "a")
    b = pypto.tensor((32, 64), pypto.DT_FP4_E2M1, "b")

    with pytest.raises(PyptoError, match=r"Unsupported input dtype combination"):
        pypto.matmul(a, b, pypto.DT_FP16)


def test_scaled_mm_rejects_input_outside_mx_combos():
    mat_a = pypto.tensor((16, 128), pypto.DT_FP16, "mat_a")
    mat_b = pypto.tensor((128, 32), pypto.DT_FP16, "mat_b")
    scale_a = pypto.tensor((16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    scale_b = pypto.tensor((2, 32, 2), pypto.DT_FP8E8M0, "scale_b")

    with pytest.raises(PyptoError, match=r"Unsupported input dtype combination"):
        pypto.scaled_mm(mat_a, mat_b, pypto.DT_FP16, scale_a, scale_b)
