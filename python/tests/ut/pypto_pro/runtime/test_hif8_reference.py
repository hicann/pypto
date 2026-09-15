# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep the fast ST golden encoder byte-identical to its independent scalar oracle."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def codec():
    path = Path(__file__).resolve().parents[3] / "st/pypto_pro/frontend/matmul/test_matmul_hf8_fp32.py"
    spec = importlib.util.spec_from_file_location("hif8_st_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_matches_scalar(codec, source):
    values = np.asarray(source, dtype=np.float32)
    expected = np.array([codec.encode_hif8_scalar(float(v)) for v in values.flat], dtype=np.uint8)
    actual = codec.quant_hif8(source)
    assert actual.dtype == np.uint8
    assert actual.shape == values.shape
    np.testing.assert_array_equal(actual, expected.reshape(values.shape))


def test_hif8_rounding_boundaries_and_float_bit_patterns(codec):
    values = np.unique(codec._HIF8_FINITE_VALS).astype(np.float32)
    midpoints = ((values[:-1].astype(np.float64) + values[1:]) / 2).astype(np.float32)
    rng = np.random.default_rng(20260915)
    random_bits = rng.integers(0, 2**32, size=4096, dtype=np.uint32).view(np.float32)
    _assert_matches_scalar(codec, np.concatenate([
        values,
        midpoints,
        np.nextafter(midpoints, np.float32(-np.inf)),
        np.nextafter(midpoints, np.float32(np.inf)),
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32),
        random_bits,
        rng.standard_normal(4096).astype(np.float32),
    ]))


def test_hif8_scalar_empty_and_noncontiguous_inputs(codec):
    for source in (
        0.75,
        [],
        np.empty((2, 0, 3), dtype=np.float32),
        np.arange(-100, 100, dtype=np.float32).reshape(10, 20).T[::2, ::3],
        [[-0.0625, 0.0625], [1.0, 49152.0]],
    ):
        _assert_matches_scalar(codec, source)
