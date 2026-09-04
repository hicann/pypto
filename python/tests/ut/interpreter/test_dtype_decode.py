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
"""UT for tools/verifier/dtype_decode.py: packed-dtype decoders (INT4/FP8/FP8E4M3/FP8E5M2/HF8/BF16).

References:
- C++ decoders: framework/src/interface/machine/device/tilefwk/aicore_print_base.h
- C++ UT expectations: framework/tests/ut/machine/src/test_aicore_print.cpp (Fp8DecodeTest)
- HF8 spec branches: Tiny(D=0000)/Small(D=0001)/Medium(D=001)/Large(D=01)/Huge(D=10)/Max(D=11)
"""

import os
import sys

import numpy as np
import pytest
import torch

_TOOLS_VERIFIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools", "verifier")
sys.path.insert(0, os.path.abspath(_TOOLS_VERIFIER))

from dtype_decode import _decode_fp8e5m2_value, _decode_hf8_value, decode_tensor_data  # noqa: E402

ALL_BYTES = np.arange(256, dtype=np.uint8)


def _torch_f8_ref(raw, torch_dtype):
    return torch.from_numpy(raw).view(torch_dtype).float().numpy()


def _assert_bitwise_equal(actual, expected):
    """NaN-aware bitwise comparison: non-NaN values must match bit-for-bit (keeps -0.0); NaN on both sides passes."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert actual.dtype == np.float32 and expected.dtype == np.float32
    nan_both = np.isnan(actual) & np.isnan(expected)
    assert not np.logical_xor(np.isnan(actual), np.isnan(expected)).any(), "NaN/non-NaN mismatch"
    diff = actual.view(np.uint32) != expected.view(np.uint32)
    assert np.logical_and(diff, ~nan_both).sum() == 0, (
        f"bitwise mismatch at: {np.flatnonzero(diff & ~nan_both)[:8]}"
    )


# ============================ INT4 ============================
class TestDecodeInt4:
    """One INT4 value per storage byte (low nibble), sign-extended to int8."""

    def test_nibble_values(self):
        out = decode_tensor_data(np.array([0xAB, 0x12, 0xF8], dtype=np.uint8), "INT4")
        assert out.dtype == np.int8
        assert list(out) == [-5, 2, -8]

    def test_boundary_values(self):
        out = decode_tensor_data(np.array([0x00, 0x77, 0x88, 0xF7, 0xFF], dtype=np.uint8), "INT4")
        # low nibble only: 0, 7, -8, 7, -1
        assert list(out) == [0, 7, -8, 7, -1]

    def test_full_range(self):
        raw = np.arange(256, dtype=np.uint8)
        out = decode_tensor_data(raw, "INT4")
        assert out.shape == (256,)
        expected = ((raw.astype(np.int16) & 0xF) ^ 0x8) - 0x8
        assert np.array_equal(out.astype(np.int16), expected)


# ============================ FP8 / FP8E4M3 (E4M3FN) ============================
class TestDecodeFp8E4M3:
    """FP8 and FP8E4M3 are the same E4M3FN format: no inf, only 0x7F/0xFF is NaN."""

    def test_zero(self):
        assert decode_tensor_data(np.array([0x00], dtype=np.uint8), "FP8")[0] == 0.0
        assert str(decode_tensor_data(np.array([0x80], dtype=np.uint8), "FP8")[0]) == "-0.0"

    def test_nan_encoding(self):
        for bits in (0x7F, 0xFF):
            assert np.isnan(decode_tensor_data(np.array([bits], dtype=np.uint8), "FP8")[0]), hex(bits)

    def test_max_finite(self):
        # exp=0b1111, mant=0b110 -> 448.0 (OCP float8_e4m3fn max)
        assert decode_tensor_data(np.array([0x7E], dtype=np.uint8), "FP8")[0] == 448.0

    def test_normal_values(self):
        # 1.0 = 2^0 x 1.0 -> E=7, M=0 -> 0x38; 2.0 -> 0x40; 0.5 -> 0x30
        out = decode_tensor_data(np.array([0x38, 0x40, 0x30], dtype=np.uint8), "FP8")
        assert out.tolist() == [1.0, 2.0, 0.5]

    def test_subnormal_values(self):
        # exp=0: mant * 2^-9
        out = decode_tensor_data(np.array([0x01, 0x07], dtype=np.uint8), "FP8")
        assert out.tolist() == [2.0**-9, 7 * 2.0**-9]

    def test_fp8e4m3_alias_matches_fp8(self):
        assert np.array_equal(
            decode_tensor_data(ALL_BYTES, "FP8E4M3"), decode_tensor_data(ALL_BYTES, "FP8"), equal_nan=True
        )

    def test_matches_torch_e4m3fn_all_256(self):
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch has no float8_e4m3fn")
        _assert_bitwise_equal(decode_tensor_data(ALL_BYTES, "FP8"), _torch_f8_ref(ALL_BYTES, torch.float8_e4m3fn))


# ============================ FP8E5M2 ============================
class TestDecodeFp8E5M2:
    """IEEE-style E5M2: has inf (0x7C/0xFC) and NaN (mant != 0 with exp=0b11111)."""

    def test_zero_and_negzero(self):
        assert _decode_fp8e5m2_value(0x00) == 0.0
        assert str(_decode_fp8e5m2_value(0x80)) == "-0.0"

    def test_inf(self):
        assert np.isinf(_decode_fp8e5m2_value(0x7C)) and _decode_fp8e5m2_value(0x7C) > 0
        assert np.isinf(_decode_fp8e5m2_value(0xFC)) and _decode_fp8e5m2_value(0xFC) < 0

    def test_nan(self):
        for bits in (0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF):
            assert np.isnan(_decode_fp8e5m2_value(bits)), hex(bits)

    def test_subnormal(self):
        # exp=0: mant * 2^-16
        assert _decode_fp8e5m2_value(0x01) == 2.0**-16
        assert _decode_fp8e5m2_value(0x02) == 2 * 2.0**-16
        assert _decode_fp8e5m2_value(0x03) == 3 * 2.0**-16
        assert _decode_fp8e5m2_value(0x81) == -2.0**-16
        assert _decode_fp8e5m2_value(0x83) == -(3 * 2.0**-16)

    def test_normal_values(self):
        # 0x3C=1.0 (E=15,M=0), 0x40=2.0, 0x44=4.0, 0x38=0.5
        assert _decode_fp8e5m2_value(0x3C) == 1.0
        assert _decode_fp8e5m2_value(0x40) == 2.0
        assert _decode_fp8e5m2_value(0x44) == 4.0
        assert _decode_fp8e5m2_value(0x38) == 0.5

    def test_matches_torch_e5m2_all_256(self):
        if not hasattr(torch, "float8_e5m2"):
            pytest.skip("torch has no float8_e5m2")
        _assert_bitwise_equal(decode_tensor_data(ALL_BYTES, "FP8E5M2"), _torch_f8_ref(ALL_BYTES, torch.float8_e5m2))


# ============================ HF8 ============================
class TestDecodeHf8:
    """HiFloat8: S(1) + D(prefix) + E(variable) + M(variable), 6 decode branches."""

    def test_tiny_branch(self):
        # D=0000: mv=0 -> +0 (sign=0) / NaN (sign=1); mv>0: 2^(mv-23)
        assert _decode_hf8_value(0x00) == 0.0
        assert np.isnan(_decode_hf8_value(0x80)) and not np.isinf(_decode_hf8_value(0x80))
        assert _decode_hf8_value(0x01) == 2.0**-22
        assert _decode_hf8_value(0x07) == 2.0**-16

    def test_small_branch(self):
        # D=0001: value = 1 + mv/8, mv in [0,7]
        assert _decode_hf8_value(0x08) == 1.0
        assert _decode_hf8_value(0x0F) == 1.875

    def test_medium_branch(self):
        # D=001: 1-bit eb, ev=+1/-1, value = 2^ev * (1 + mv/8)
        assert _decode_hf8_value(0x18) == 0.5  # ev=-1, mv=0
        assert _decode_hf8_value(0x10) == 2.0  # ev=+1, mv=0

    def test_large_branch(self):
        # D=01: 2-bit eb, ev in +-[2,3]
        assert _decode_hf8_value(0x20) == 4.0  # ev=+2, mv=0

    def test_huge_branch(self):
        # D=10: 3-bit eb, ev in +-[4,7]
        assert _decode_hf8_value(0x40) == 16.0  # ev=+4, mv=0

    def test_max_branch_and_inf(self):
        # D=11: 4-bit eb, ev in +-[8,15], 1-bit mv; S_11_0111_1 is inf
        assert np.isinf(_decode_hf8_value(0x6F)) and _decode_hf8_value(0x6F) > 0
        assert np.isinf(_decode_hf8_value(0xEF)) and _decode_hf8_value(0xEF) < 0
        # 0x6E (ev=15, mv=0) is max normal 32768, not inf
        assert not np.isinf(_decode_hf8_value(0x6E))
        assert _decode_hf8_value(0x6E) == 32768.0

    def test_sign_symmetry(self):
        # i in [1,128), skip inf(0x6F) / NaN(0x80) / -inf(0xEF)
        for i in range(1, 128):
            if i == 0x6F:
                continue
            pos = _decode_hf8_value(i)
            neg = _decode_hf8_value(i | 0x80)
            if (i | 0x80) in (0x80, 0xEF):
                continue
            assert pos > 0 and neg < 0, f"sign fail at {i:#04x}"
            assert abs(pos) == abs(neg), f"symmetry fail at {i:#04x}"

    def test_array_decode_uses_lut_consistently(self):
        lut = decode_tensor_data(ALL_BYTES, "HF8")
        assert lut.dtype == np.float32
        per_value = np.array([_decode_hf8_value(i) for i in range(256)], dtype=np.float32)
        assert np.array_equal(lut, per_value, equal_nan=True)


# ============================ BF16 ============================
class TestDecodeBf16:
    """BF16 read as uint16 raw, decode = bit-shift into fp32 exponent/mantissa layout."""

    def test_roundtrip_of_known_values(self):
        raw = np.array([0x0000, 0x3F80, 0xBF80, 0x7F7F, 0x3FC0], dtype=np.uint16)
        out = decode_tensor_data(raw, "BF16")
        assert out.dtype == np.float32
        # 0x7F7F is bf16 max finite, smaller than fp32 max
        assert np.allclose(out, [0.0, 1.0, -1.0, 3.3895313892515355e38, 1.5])

    def test_inf_and_nan(self):
        out = decode_tensor_data(np.array([0x7F80, 0xFF80, 0x7FC0], dtype=np.uint16), "BF16")
        assert np.isinf(out[0]) and out[0] > 0
        assert np.isinf(out[1]) and out[1] < 0
        assert np.isnan(out[2])

    def test_subnormal(self):
        # smallest bf16 subnormal = 2^-133
        out = decode_tensor_data(np.array([0x0001], dtype=np.uint16), "BF16")
        assert out[0] == np.float32(2.0**-133)

    def test_matches_torch_bf16_all_65536(self):
        raw = np.arange(65536, dtype=np.uint16)
        out = decode_tensor_data(raw, "BF16")
        ref = torch.from_numpy(raw.astype(np.int64)).to(torch.int32).numpy().astype(np.uint32)
        ref = (ref << 16).view(np.float32)
        _assert_bitwise_equal(out, ref)


# ============================ decode_tensor_data routing ============================
class TestDecodeTensorDataRouting:
    def test_unknown_dtype_passthrough(self):
        for dtype_name in ("INT32", "FP32", "DOUBLE", "UINT8", "BOOL", "UNKNOWN"):
            data = np.array([1, 2, 3], dtype=np.int32)
            assert decode_tensor_data(data, dtype_name) is data, dtype_name

    def test_result_dtypes(self):
        assert decode_tensor_data(np.array([0], dtype=np.uint8), "INT4").dtype == np.int8
        assert decode_tensor_data(np.array([0], dtype=np.uint8), "FP8").dtype == np.float32
        assert decode_tensor_data(np.array([0], dtype=np.uint8), "FP8E4M3").dtype == np.float32
        assert decode_tensor_data(np.array([0], dtype=np.uint8), "FP8E5M2").dtype == np.float32
        assert decode_tensor_data(np.array([0], dtype=np.uint8), "HF8").dtype == np.float32
        assert decode_tensor_data(np.array([0], dtype=np.uint16), "BF16").dtype == np.float32

    def test_empty_input(self):
        out = decode_tensor_data(np.array([], dtype=np.uint8), "INT4")
        assert out.shape == (0,) and out.dtype == np.int8
        for dtype_name in ("FP8", "FP8E4M3", "FP8E5M2", "HF8"):
            out = decode_tensor_data(np.array([], dtype=np.uint8), dtype_name)
            assert out.shape == (0,) and out.dtype == np.float32, dtype_name
        assert decode_tensor_data(np.array([], dtype=np.uint16), "BF16").shape == (0,)

    def test_shape_preserved(self):
        raw = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        out = decode_tensor_data(raw, "HF8")
        assert out.shape == (2, 3, 4)
