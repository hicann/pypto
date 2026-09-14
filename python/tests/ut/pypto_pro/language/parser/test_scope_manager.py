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
"""Unit tests for ScopeManager."""

from pypto_pro import ir
from pypto_pro.language.parser._scope_manager import (
    ConstantState,
    ControlFlowInfo,
    JumpKind,
    PhiState,
    ScopeManager,
    SSAViolationError,
)
from pypto_pro.language.parser.diagnostics import ParserTypeError
import pytest


def test_initialization():
    """Test ScopeManager initializes correctly."""
    sm = ScopeManager()

    assert len(sm.scopes) == 1  # Global scope
    assert sm.current_scope_type() == "global"


def test_enter_exit_scope():
    """Test entering and exiting scopes."""
    sm = ScopeManager()

    sm.enter_scope("function")
    assert len(sm.scopes) == 2
    assert sm.current_scope_type() == "function"

    sm.exit_scope()
    assert len(sm.scopes) == 1
    assert sm.current_scope_type() == "global"


def test_nested_scopes():
    """Test nested scope management."""
    sm = ScopeManager()

    sm.enter_scope("function")
    sm.enter_scope("for")
    sm.enter_scope("if")

    assert len(sm.scopes) == 4  # global + function + for + if
    assert sm.current_scope_type() == "if"

    sm.exit_scope()
    assert sm.current_scope_type() == "for"

    sm.exit_scope()
    assert sm.current_scope_type() == "function"


def test_define_var():
    """Test defining variables in scope."""
    sm = ScopeManager()

    sm.enter_scope("function")
    sm.define_var("x", "value_x")

    assert sm.is_defined("x")
    assert sm.lookup_var("x") == "value_x"


def test_ssa_violation():
    """Test SSA violation detection when strict_ssa=True."""
    sm = ScopeManager(strict_ssa=True)

    sm.enter_scope("function")
    sm.define_var("x", "value1")

    # Trying to redefine should raise SSAViolationError
    with pytest.raises(SSAViolationError, match="already defined"):
        sm.define_var("x", "value2")


def test_allow_redef():
    """Test allowing redefinition for special cases."""
    sm = ScopeManager()

    sm.enter_scope("function")
    sm.define_var("x", "value1", allow_redef=True)
    sm.define_var("x", "value2", allow_redef=True)  # Should not raise

    assert sm.lookup_var("x") == "value2"


def test_variable_shadowing():
    """Test that variables in inner scopes shadow outer scopes."""
    sm = ScopeManager()

    sm.enter_scope("function")
    sm.define_var("x", "outer")

    sm.enter_scope("for")
    sm.define_var("x", "inner")

    # Inner scope should see inner value
    assert sm.lookup_var("x") == "inner"

    sm.exit_scope()
    # Outer scope should see outer value
    assert sm.lookup_var("x") == "outer"


def test_lookup_undefined_var():
    """Test looking up undefined variable returns None."""
    sm = ScopeManager()

    assert sm.lookup_var("undefined") is None
    assert not sm.is_defined("undefined")


def test_in_scope_type():
    """Test checking if in specific scope type."""
    sm = ScopeManager()

    assert sm.in_scope_type("global")
    assert not sm.in_scope_type("function")

    sm.enter_scope("function")
    assert sm.in_scope_type("function")
    assert sm.in_scope_type("global")  # Still in global too

    sm.enter_scope("for")
    assert sm.in_scope_type("for")
    assert sm.in_scope_type("function")


def test_exit_global_scope_error():
    """Test that exiting global scope raises error."""
    sm = ScopeManager()

    with pytest.raises(RuntimeError, match="Cannot exit global scope"):
        sm.exit_scope()


def test_scope_isolation():
    """Test that scope variables are properly isolated."""
    sm = ScopeManager()

    sm.enter_scope("function")
    sm.define_var("x", "func_var")

    sm.enter_scope("for")
    sm.define_var("y", "loop_var")

    # Both variables should be accessible in inner scope
    assert sm.is_defined("x")
    assert sm.is_defined("y")

    scope_vars = sm.exit_scope()

    # After exiting, loop variable should not be in function scope
    assert "y" in scope_vars
    assert sm.is_defined("x")
    # y is no longer accessible after exiting its scope
    assert not sm.is_defined("y")


def test_control_flow_info_contexts_restore_nested_state():
    sm = ScopeManager()
    outer_loop = ControlFlowInfo(("loop_value",))
    inner_loop = ControlFlowInfo(("nested_value",))
    branch = ControlFlowInfo(("branch_value",))

    with sm.change_loop_info(outer_loop):
        assert sm.loop_info is outer_loop
        with sm.change_if_info(branch), sm.change_loop_info(inner_loop):
            assert sm.if_info is branch
            assert sm.loop_info is inner_loop
        assert sm.if_info is None
        assert sm.loop_info is outer_loop

    assert sm.loop_info is None


def test_local_scope_keeps_first_jump_kind():
    sm = ScopeManager()
    sm.enter_scope("if")
    local = sm.current_scope

    local.set_jump(JumpKind.BREAK)
    local.set_jump(JumpKind.YIELD)

    assert local.jump_kind is JumpKind.BREAK


def test_phi_state_merges_type_and_equal_constants():
    span = ir.Span.unknown()
    state = PhiState()
    first_value = ir.ConstInt(1, ir.DataType.INT32, span)
    second_value = ir.ConstInt(1, ir.DataType.INT32, span)

    state.propagate(first_value)
    state.propagate(second_value)

    assert isinstance(state.ty, ir.ScalarType)
    assert state.ty.dtype == ir.DataType.INT32
    assert state.constant_state is ConstantState.MAY_BE_CONSTANT
    assert ir.structural_equal(state.constant_value, first_value, enable_auto_mapping=False)


def test_phi_state_merges_tuple_as_one_constant_value():
    span = ir.Span.unknown()
    state = PhiState()
    first_value = ir.MakeTuple(
        [
            ir.ConstInt(1, ir.DataType.INT32, span),
            ir.ConstInt(2, ir.DataType.INT32, span),
        ],
        span,
    )
    second_value = ir.MakeTuple(
        [
            ir.ConstInt(1, ir.DataType.INT32, span),
            ir.ConstInt(2, ir.DataType.INT32, span),
        ],
        span,
    )

    state.propagate(first_value)
    state.propagate(second_value)

    assert isinstance(state.ty, ir.TupleType)
    assert state.constant_state is ConstantState.MAY_BE_CONSTANT
    assert ir.structural_equal(state.constant_value, first_value, enable_auto_mapping=False)


def test_phi_state_does_not_keep_partially_equal_tuple_constant():
    span = ir.Span.unknown()
    state = PhiState()
    first_value = ir.MakeTuple(
        [
            ir.ConstInt(1, ir.DataType.INT32, span),
            ir.ConstInt(2, ir.DataType.INT32, span),
        ],
        span,
    )
    second_value = ir.MakeTuple(
        [
            ir.ConstInt(1, ir.DataType.INT32, span),
            ir.ConstInt(3, ir.DataType.INT32, span),
        ],
        span,
    )
    state.propagate(first_value)
    state.propagate(second_value)

    assert isinstance(state.ty, ir.TupleType)
    assert state.constant_state is ConstantState.NONCONSTANT
    assert state.constant_value is None


def test_phi_state_marks_mismatched_runtime_types_invalid():
    span = ir.Span.unknown()
    state = PhiState()
    first = ir.Var("first", ir.ScalarType(ir.DataType.INT32), span)
    second = ir.Var("second", ir.ScalarType(ir.DataType.INT64), span)

    state.propagate(first)
    state.propagate(second)

    assert isinstance(state.ty, ir.NoneType)


def test_phi_state_keeps_none_as_an_invalid_type():
    span = ir.Span.unknown()
    state = PhiState()
    none_value = ir.Var("None", ir.NoneType.get(), span)
    int_value = ir.Var("value", ir.ScalarType(ir.DataType.INT32), span)

    state.propagate(none_value)
    state.propagate(int_value)

    assert isinstance(state.ty, ir.NoneType)


def test_phi_state_eager_invalid_fails_only_after_a_concrete_type():
    span = ir.Span.unknown()
    invalid = ir.Var("invalid", ir.NoneType.get(), span)
    concrete = ir.Var("concrete", ir.ScalarType(ir.DataType.INT32), span)

    state = PhiState()
    state.propagate(invalid, fail_eagerly=True)
    state.propagate(concrete, fail_eagerly=True)
    assert isinstance(state.ty, ir.NoneType)

    concrete_first = PhiState()
    concrete_first.propagate(concrete)
    with pytest.raises(ParserTypeError, match="invalid type"):
        concrete_first.propagate(invalid, fail_eagerly=True)


def test_phi_state_can_start_nonconstant_for_loop_body():
    span = ir.Span.unknown()
    constant = ir.ConstInt(1, ir.DataType.INT32, span)
    state = PhiState(constant_state=ConstantState.NONCONSTANT)

    state.propagate(constant)

    assert state.constant_state is ConstantState.NONCONSTANT
    assert state.constant_value is None
