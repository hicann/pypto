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
"""Unit tests for :class:`AttrSpec` -- construction, validation and JSON serialization.

``AttrSpec`` is imported through the package re-export rather than from ``common.attr_spec`` directly,
because that re-export line has no other test.
"""
import json

import pytest

from pypto.extensions.torch_custom_op_litenpu import AttrSpec
from pypto.extensions.torch_custom_op_litenpu.common.attr_spec import ATTR_TYPES


def test_attrspec_construct_serialize_roundtrip():
    s = AttrSpec("bias", "Int", 0)
    assert (s.name, s.type, s.default) == ("bias", "Int", 0)
    d = s.to_json_dict()
    assert d == {"name": "bias", "type": "Int", "default": 0}
    assert json.loads(json.dumps(d)) == d  # JSON-serializable


def test_attrspec_default_none_is_required():
    assert AttrSpec("k", "ListInt").default is None  # None => REQUIRED_ATTR downstream


def test_attrspec_rejects_unknown_type():
    with pytest.raises(ValueError, match="type"):
        AttrSpec("bias", "NotAType")
    assert set(ATTR_TYPES) == {"Int", "Float", "String", "ListInt"}


@pytest.mark.parametrize(
    "bad",
    ['say "hi"', "back\\slash", "line\nbreak", "carriage\rreturn", "tab\there", "nul\0here", "esc\x1bhere"],
)
def test_attrspec_rejects_string_default_with_special_chars(bad):
    # A String default is emitted verbatim inside a C++ string literal, so anything that would break or
    # silently shorten it is rejected at construction -- NUL would end the emitted C string early.
    with pytest.raises(ValueError, match="String default"):
        AttrSpec("mode", "String", bad)
    # A clean String default is accepted.
    assert AttrSpec("mode", "String", "sum").default == "sum"


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_attrspec_rejects_non_finite_float_default(bad):
    # repr(inf)/repr(nan) yield "inf"/"nan", which have no valid C++ float literal form, so a non-finite
    # Float default is rejected at construction.
    with pytest.raises(ValueError, match="finite"):
        AttrSpec("a", "Float", bad)
    # A finite Float default (incl. a whole number and 0.0) is accepted.
    assert AttrSpec("a", "Float", 1.0).default == 1.0
    assert AttrSpec("a", "Float", 0.0).default == 0.0


@pytest.mark.parametrize("bad", [2**63, -2**63 - 1, 10**20])
def test_attrspec_rejects_int_default_outside_int64_range(bad):
    with pytest.raises(ValueError, match="int64"):
        AttrSpec("k", "Int", bad)
    # Both int64 boundaries are representable and stay valid.
    assert AttrSpec("k", "Int", 2**63 - 1).default == 2**63 - 1
    assert AttrSpec("k", "Int", -2**63).default == -2**63


@pytest.mark.parametrize("bad", ["", "0bias", "my bias", "my-bias", "a.b", "bad name); boom(", 7, None])
def test_attrspec_rejects_a_name_that_is_not_an_identifier(bad):
    # The name is emitted verbatim as a C++ identifier in the REG_OP block and is the ONNX attribute key,
    # so anything that is not an identifier is rejected at construction rather than at code generation.
    with pytest.raises(ValueError, match="name"):
        AttrSpec(bad, "Int", 0)
    assert AttrSpec("_leading_underscore9", "Int", 0).name == "_leading_underscore9"


@pytest.mark.parametrize("declared,bad", [
    ("Int", "five"),        # a string is not an int
    ("Int", 2.7),           # int(2.7) would silently truncate to 2
    ("Int", True),          # bool is an int subclass; int(True) would emit 1
    ("String", 5),          # any object would be stringified into the C++ literal
    ("String", b"bytes"),   # bytes is not str; repr would leak the b'' prefix into the literal
    ("ListInt", 5),         # a scalar is not iterable where a brace-init list is emitted
    ("ListInt", [1, "x"]),  # a non-int element
    ("ListInt", [1, 2.5]),  # int(2.5) would silently truncate to 2
    ("ListInt", [True]),    # bool element, as above
    ("Float", True),        # bool, as above
    ("Float", "1.5"),       # a string is not a number
])
def test_attrspec_rejects_a_default_of_the_wrong_type(declared, bad):
    with pytest.raises(ValueError, match="default"):
        AttrSpec("a", declared, bad)


@pytest.mark.parametrize("declared,good", [
    ("Int", 0), ("Int", 7), ("Int", -3),
    ("Float", 1.0), ("Float", 0.01), ("Float", 1),  # an int is a lossless Float default
    ("String", "sum"), ("String", ""),
    ("ListInt", [4, 4]), ("ListInt", (4, 4)), ("ListInt", []),
])
def test_attrspec_accepts_a_default_of_the_declared_type(declared, good):
    # The counterpart to the rejection cases: every default shape the stack actually declares stays valid,
    # so the new type checks cannot be passing merely by rejecting everything.
    assert AttrSpec("a", declared, good).default == good
