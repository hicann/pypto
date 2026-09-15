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

"""Tests for parser-session TupleType classification and comparison."""

from pypto_pro import ir
from pypto_pro.language.parser._tuple_type_registry import (
    TupleTypeInfo,
    TupleTypeKind,
    TupleTypeRegistry,
)


def _tuple_type(*types: ir.Type) -> ir.TupleType:
    return ir.TupleType(list(types))


def test_tuple_type_registry_classifies_all_supported_kinds():
    debug_info = ir.IRDebugInfo()
    registry = TupleTypeRegistry(debug_info)
    scalar = ir.ScalarType(ir.DataType.INT64)
    plain = _tuple_type(scalar)
    named = _tuple_type(scalar)
    internal = _tuple_type(scalar, scalar, scalar)
    struct = _tuple_type(scalar)

    registry.register_named_tuple(named, ["value"])
    registry.register_named_tuple(internal, ["x", "y", "z"], name="dim3_context")
    registry.register_struct(struct, "Record", ["value"])

    assert registry.classify(plain) == TupleTypeInfo(TupleTypeKind.TUPLE)
    assert registry.classify(named) == TupleTypeInfo(TupleTypeKind.NAMED_TUPLE, fields=["value"])
    assert registry.classify(internal) == TupleTypeInfo(
        TupleTypeKind.NAMED_TUPLE,
        name="dim3_context",
        fields=["x", "y", "z"],
    )
    assert registry.classify(struct) == TupleTypeInfo(
        TupleTypeKind.STRUCT,
        name="Record",
        fields=["value"],
    )


def test_named_tuple_and_struct_metadata_update_ir_debug_info():
    debug_info = ir.IRDebugInfo()
    registry = TupleTypeRegistry(debug_info)
    scalar = ir.ScalarType(ir.DataType.INT64)
    named = _tuple_type(scalar)
    internal = _tuple_type(scalar)
    struct = _tuple_type(scalar)

    registry.register_named_tuple(named, ["value"])
    registry.register_named_tuple(internal, ["value"], name="internal")
    registry.register_struct(struct, "Record", ["value"])

    assert debug_info.get_tuple_type_info(named) == TupleTypeInfo(
        TupleTypeKind.NAMED_TUPLE, fields=["value"]
    )
    assert debug_info.get_tuple_type_info(internal) == TupleTypeInfo(
        TupleTypeKind.NAMED_TUPLE, name="internal", fields=["value"]
    )
    assert debug_info.get_tuple_type_info(struct) == TupleTypeInfo(
        TupleTypeKind.STRUCT, name="Record", fields=["value"]
    )


def test_tuple_type_comparison_checks_kind_name_and_fields():
    debug_info = ir.IRDebugInfo()
    registry = TupleTypeRegistry(debug_info)
    scalar = ir.ScalarType(ir.DataType.INT64)
    plain = _tuple_type(scalar)
    anonymous = _tuple_type(scalar)
    anonymous_same = _tuple_type(scalar)
    internal = _tuple_type(scalar)
    different_fields = _tuple_type(scalar)
    struct = _tuple_type(scalar)

    registry.register_named_tuple(anonymous, ["value"])
    registry.register_named_tuple(anonymous_same, ["value"])
    registry.register_named_tuple(internal, ["value"], name="internal")
    registry.register_named_tuple(different_fields, ["other"])
    registry.register_struct(struct, "Record", ["value"])

    assert registry.types_equal(anonymous, anonymous_same)
    assert not registry.types_equal(plain, anonymous)
    assert not registry.types_equal(anonymous, internal)
    assert not registry.types_equal(anonymous, different_fields)
    assert not registry.types_equal(anonymous, struct)


def test_tuple_type_comparison_checks_nested_tuple_metadata():
    debug_info = ir.IRDebugInfo()
    registry = TupleTypeRegistry(debug_info)
    scalar = ir.ScalarType(ir.DataType.INT64)
    named = _tuple_type(scalar)
    plain = _tuple_type(scalar)
    left = _tuple_type(named)
    right = _tuple_type(plain)
    registry.register_named_tuple(named, ["value"])

    assert ir.structural_equal(left, right, enable_auto_mapping=False)
    assert not registry.types_equal(left, right)
