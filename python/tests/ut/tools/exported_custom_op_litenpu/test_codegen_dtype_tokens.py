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
"""Unit tests for the torch<->GE dtype mapping in ``exported_custom_op_litenpu.deploy.codegen``."""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

from exported_custom_op_litenpu.deploy.codegen import (
    _GE_DATA_TYPE_VALUE_TO_TORCH_BASE,
    _inline_dtype_conversion_src,
    _torch_dtype_to_ge_dtype,
)
import pytest

# The dtype bridge the built .so actually runs is the INLINE source the wrapper py::execs, so the
# tests below exercise that string rather than a module-level copy of it.
_BRIDGE: dict = {}
exec(_inline_dtype_conversion_src(), _BRIDGE)  # noqa: S102 - our own emitted bridge source
_ge_data_type_enum_value_to_torch_dtype = _BRIDGE["_ge_data_type_enum_value_to_torch_dtype"]


def test_torch_dtype_to_ge_dtype_mappings():
    # Floats
    assert _torch_dtype_to_ge_dtype("torch.float16") == "ge::DT_FLOAT16"
    assert _torch_dtype_to_ge_dtype("torch.float32") == "ge::DT_FLOAT"
    assert _torch_dtype_to_ge_dtype("torch.float64") == "ge::DT_DOUBLE"
    assert _torch_dtype_to_ge_dtype("torch.bfloat16") == "ge::DT_BF16"
    # Signed ints
    assert _torch_dtype_to_ge_dtype("torch.int8") == "ge::DT_INT8"
    assert _torch_dtype_to_ge_dtype("torch.int16") == "ge::DT_INT16"
    assert _torch_dtype_to_ge_dtype("torch.int32") == "ge::DT_INT32"
    assert _torch_dtype_to_ge_dtype("torch.int64") == "ge::DT_INT64"
    # Unsigned ints
    assert _torch_dtype_to_ge_dtype("torch.uint8") == "ge::DT_UINT8"
    assert _torch_dtype_to_ge_dtype("torch.uint16") == "ge::DT_UINT16"
    assert _torch_dtype_to_ge_dtype("torch.uint32") == "ge::DT_UINT32"
    assert _torch_dtype_to_ge_dtype("torch.uint64") == "ge::DT_UINT64"
    # Bool
    assert _torch_dtype_to_ge_dtype("torch.bool") == "ge::DT_BOOL"
    # Complex
    assert _torch_dtype_to_ge_dtype("torch.complex32") == "ge::DT_COMPLEX32"
    assert _torch_dtype_to_ge_dtype("torch.complex64") == "ge::DT_COMPLEX64"
    assert _torch_dtype_to_ge_dtype("torch.complex128") == "ge::DT_COMPLEX128"
    # Quantized storage dtypes
    assert _torch_dtype_to_ge_dtype("torch.qint8") == "ge::DT_QINT8"
    assert _torch_dtype_to_ge_dtype("torch.qint16") == "ge::DT_QINT16"
    assert _torch_dtype_to_ge_dtype("torch.qint32") == "ge::DT_QINT32"
    assert _torch_dtype_to_ge_dtype("torch.quint8") == "ge::DT_QUINT8"
    assert _torch_dtype_to_ge_dtype("torch.quint16") == "ge::DT_QUINT16"


@pytest.mark.parametrize("bad", ["float16", "fp16", "ge::DT_FLOAT16", "int"])
def test_torch_dtype_to_ge_dtype_rejects_non_torch_prefix(bad: str):
    with pytest.raises(ValueError, match="Unsupported dtype for GE mapping"):
        _torch_dtype_to_ge_dtype(bad)


def test_torch_dtype_to_ge_dtype_rejects_unsupported_torch_dtype():
    # e.g. float8: not mapped to ge::DT_HIFLOAT8 without a confirmed 1:1 semantics
    with pytest.raises(ValueError, match="Unsupported torch dtype for GE mapping"):
        _torch_dtype_to_ge_dtype("torch.float8_e4m3fn")


def test_ge_data_type_enum_value_to_torch_dtype_int16_example():
    import torch

    assert _ge_data_type_enum_value_to_torch_dtype(6) is torch.int16


@pytest.mark.parametrize(
    ("value", "basename"),
    sorted(_GE_DATA_TYPE_VALUE_TO_TORCH_BASE.items(), key=lambda x: x[0]),
)
def test_ge_data_type_enum_value_to_torch_dtype_matches_basename(value: int, basename: str):
    import torch

    if not hasattr(torch, basename):
        # Structural, not environmental: PyTorch exposes no 16-bit quantized dtype in any release (its own
        # NNAPI backend substitutes int16, see use_int16_for_qint16), so GE's DT_QINT16/DT_QUINT16 have no
        # torch counterpart to compare against. The hasattr guard stays so coverage switches on by itself if
        # a torch build ever adds them; the string-level mapping is asserted above without a live dtype.
        pytest.skip(f"torch has no {basename!r}: PyTorch exposes no 16-bit quantized dtype, so this GE enum "
                    f"value has no torch counterpart (structural, not a gap in this environment)")
    td = _ge_data_type_enum_value_to_torch_dtype(value)
    assert str(td) == f"torch.{basename}"


def test_ge_data_type_enum_value_to_torch_dtype_rejects_unmapped():
    with pytest.raises(ValueError, match="Unsupported ge::DataType enum value"):
        _ge_data_type_enum_value_to_torch_dtype(13)  # DT_STRING in gert_ge_minimal.hpp
