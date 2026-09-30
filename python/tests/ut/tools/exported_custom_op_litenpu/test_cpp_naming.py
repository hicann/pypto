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
"""Unit tests for the C++ codegen case-conversion helper (``exported_custom_op_litenpu.deploy.cpp_naming``)."""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

from exported_custom_op_litenpu.deploy.cpp_naming import camel_case_to_snake_case
import pytest


@pytest.mark.parametrize(
    ("pascal_or_camel", "expected_snake"),
    [
        ("Add", "add"),
        ("PyptoCustomOpAdd", "pypto_custom_op_add"),
        ("inferShape", "infer_shape"),
        ("XMLParser", "xml_parser"),
        ("a", "a"),
    ],
)
def test_camel_case_to_snake_case(pascal_or_camel: str, expected_snake: str):
    assert camel_case_to_snake_case(pascal_or_camel) == expected_snake
