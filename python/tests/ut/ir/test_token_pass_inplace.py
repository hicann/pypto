# -*- coding: utf-8 -*-
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for token dependencies of inplace tensor operations."""


import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from .test_common import check_snapshot


def _run_passes(func, *args):
    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    pipeline = [
        ("infer_token_pass", ir.Pass.infer_token_pass()),
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("remove_redundant_token_pass", ir.Pass.remove_redundant_token_pass()),
    ]
    return compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)


def _tensor(shape, name):
    return pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name=name)


def _index_tensor(shape, name):
    return pypto.Tensor(shape=shape, dtype=pypto.DT_INT32, name=name)


INDEX_PUT_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %values, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %target_r = ADDS(%target) token();
    logical_tensor %target_0 = INDEX_PUT(%target, %values, %index) token(%target_r);
    logical_tensor %$2 = ADD(%$0, %target_0) token();
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %target_0, %index, %values, %out_1;
}
"""


def test_index_put_keeps_write_after_read_dependency():
    def foo(target, index, values, out):
        value = target + 1.0
        pypto.index_put_(target, (index,), values, False)
        pypto.assemble(value + target, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((8,), "index"),
        _tensor((8, 16), "values"),
        _tensor((16, 16), "out"),
    )
    check_snapshot(after_remove, INDEX_PUT_EXPECTED_IR)


INDEX_ADD_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %source, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %target_r = ADDS(%target) token();
    logical_tensor %target_0 = INDEX_ADD(%target, %source, %index) token(%target_r);
    logical_tensor %$2 = ADD(%$0, %target_0) token();
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %target_0, %index, %source, %out_1;
}
"""


def test_index_add_keeps_write_after_read_dependency():
    def foo(target, index, source, out):
        value = target + 1.0
        pypto.index_add_(target, 0, index, source)
        pypto.assemble(value + target, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((8,), "index"),
        _tensor((8, 16), "source"),
        _tensor((16, 16), "out"),
    )
    check_snapshot(after_remove, INDEX_ADD_EXPECTED_IR)


SCATTER_UPDATE_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %source, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %target_r = ADDS(%target) token();
    logical_tensor %$1 = INDEX_OUTCAST(%source, %index, %target) token(%target_r);
    logical_tensor %$2 = ADD(%$0, %$1) token();
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %target, %index, %source, %out_1;
}
"""


def test_scatter_update_keeps_write_after_read_dependency():
    def foo(target, index, source, out):
        value = target + 1.0
        updated = pypto.scatter_update(target, -2, index, source)
        pypto.assemble(value + updated, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((2, 2), "index"),
        _tensor((4, 16), "source"),
        _tensor((16, 16), "out"),
    )
    check_snapshot(after_remove, SCATTER_UPDATE_EXPECTED_IR)


AXPY_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %source, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %target_r = ADDS(%target) token();
    logical_tensor %target_0 = AXPY(%target, %source) token(%target_r);
    logical_tensor %$2 = ADD(%$0, %target_0) token();
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %target_0, %source, %out_1;
}
"""


def test_axpy_keeps_write_after_read_dependency():
    def foo(target, source, out):
        value = target + 1.0
        target.axpy_(source)
        pypto.assemble(value + target, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _tensor((16, 16), "source"),
        _tensor((16, 16), "out"),
    )
    check_snapshot(after_remove, AXPY_EXPECTED_IR)


RESHAPE_INPLACE_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %target_r = ADDS(%target) token();
    logical_tensor %$1 = RESHAPE(%target) token(%target_r);
    logical_tensor %$2 = ADD(%$0, %$1) token();
    logical_tensor %out_1 = ASSEMBLE(%$2) token();
    return %target, %out_1;
}
"""


def test_inplace_reshape_keeps_write_after_read_dependency():
    def foo(target, out):
        value = target + 1.0
        alias = pypto.reshape(target, [16, 16], inplace=True)
        pypto.assemble(value + alias, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _tensor((16, 16), "out"),
    )
    check_snapshot(after_remove, RESHAPE_INPLACE_EXPECTED_IR)


def test_inplace_scatter_keeps_write_after_read_dependency():
    def foo(target, index, source, out):
        value = target + 1.0
        target.scatter_(0, index, source)
        pypto.assemble(value + target, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((16, 16), "index"),
        _tensor((16, 16), "source"),
        _tensor((16, 16), "out"),
    )
    tensor_ops = [stmt for stmt in after_remove.body if isinstance(stmt, ir.TensorOpStmt)]
    adds = next(op for op in tensor_ops if op.opcode == "ADDS")
    scatter = next(op for op in tensor_ops if op.opcode == "SCATTER")
    assert adds.result_token[0] in scatter.tokens


ATOMIC_ADD_LOOP_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %source, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %target_3, token %target_3_w_for = for %loop_idx_5 inrange 0, 2, 1 iter {logical_tensor %target_1 = %target;token %target_3_w_for_iter = %None;} #parallel(false) #submit_before_loop(false) #_loop_conds(Unsupported) #_config_scope(Unsupported) #unroll_times(1) {
        logical_tensor %target_1, token %target_1_w = ATOMIC_RMW(%source) token();
        continue %target_1, %target_1_w;
    }
    logical_tensor %$1 = ADDS(%target_3) token(%target_3_w_for);
    logical_tensor %out_1 = ASSEMBLE(%$1) token();
    return %target_3, %source, %out_1;
}
"""


def test_atomic_add_target_is_carried_through_loop():
    def foo(target, source, out):
        for _ in pypto.loop(2):
            pypto.atomic_add(source, [0, 0], target)
        pypto.assemble(target + 1.0, [0, 0], out)

    after_remove = _run_passes(
        foo,
        _tensor((16, 16), "target"),
        _tensor((8, 16), "source"),
        _tensor((16, 16), "out"),
    )
    check_snapshot(after_remove, ATOMIC_ADD_LOOP_EXPECTED_IR)


CONTINUOUS_INPLACE_LATEST_ALIAS_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %source, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w, token %target_r = RESHAPE(%target) token();
    logical_tensor %$0_0, token %$0_0_w, token %$0_r, token %source_r, token %index_r = INDEX_ADD(%$0, %source, %index) token(%$0_w);
    logical_tensor %$2, token %$2_w, token %$0_0_r = ADDS(%$0_0) token(%$0_0_w);
    logical_tensor %out_1, token %out_1_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w);
    return %target, %index, %source, %out_1;
}
"""


def test_continuous_inplace_updates_latest_alias_state():
    def foo(target, index, source, out):
        alias = pypto.reshape(target, [16, 16], inplace=True)
        pypto.index_add_(alias, 0, index, source)
        pypto.assemble(alias + 1.0, [0, 0], out)

    after_infer = compile_new_ir(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((8,), "index"),
        _tensor((8, 16), "source"),
        _tensor((16, 16), "out"),
        pipeline=[("infer_token_pass", ir.Pass.infer_token_pass())],
        create_new_logical_tensor=True,
    )
    check_snapshot(after_infer, CONTINUOUS_INPLACE_LATEST_ALIAS_EXPECTED_IR)


A1_READ_ALIAS_WRITE_ORIGIN_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %values, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w, token %target_r = RESHAPE(%target) token();
    logical_tensor %$1, token %$1_w, token %$0_r = ADDS(%$0) token(%$0_w);
    logical_tensor %target_0, token %target_0_w, token %target_r_0, token %values_r, token %index_r = INDEX_PUT(%target, %values, %index) token(%$0_r);
    logical_tensor %out_1, token %out_1_w, token %$1_r = ASSEMBLE(%$1) token(%$1_w);
    return %target_0, %index, %values, %out_1;
}
"""


def test_read_alias_then_write_origin_keeps_war():
    """A-1: a write through the origin must wait for a read through its alias."""

    def foo(target, index, values, out):
        alias = pypto.reshape(target, [16, 16], inplace=True)
        value = alias + 1.0
        pypto.index_put_(target, (index,), values, False)
        pypto.assemble(value, [0, 0], out)

    after_infer = compile_new_ir(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((8,), "index"),
        _tensor((8, 16), "values"),
        _tensor((16, 16), "out"),
        pipeline=[("infer_token_pass", ir.Pass.infer_token_pass())],
        create_new_logical_tensor=True,
    )
    check_snapshot(after_infer, A1_READ_ALIAS_WRITE_ORIGIN_EXPECTED_IR)
    tensor_ops = [stmt for stmt in after_infer.body if isinstance(stmt, ir.TensorOpStmt)]
    adds = next(op for op in tensor_ops if op.opcode == "ADDS")
    index_put = next(op for op in tensor_ops if op.opcode == "INDEX_PUT")
    assert adds.result_token[-1] in index_put.tokens


A2_READ_ORIGIN_AFTER_ALIAS_WRITE_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %source, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w, token %target_r = RESHAPE(%target) token();
    logical_tensor %$0_0, token %$0_0_w, token %$0_r, token %source_r, token %index_r = INDEX_ADD(%$0, %source, %index) token(%$0_w);
    logical_tensor %$2, token %$2_w, token %target_r_0 = ADDS(%target) token(%$0_0_w);
    logical_tensor %out_1, token %out_1_w, token %$2_r = ASSEMBLE(%$2) token(%$2_w);
    return %target, %index, %source, %out_1;
}
"""


def test_read_origin_after_inplace_on_alias_keeps_raw():
    """A-2: a read through the origin must wait for a write through its alias."""

    def foo(target, index, source, out):

        alias = pypto.reshape(target, [16, 16], inplace=True)
        pypto.index_add_(alias, 0, index, source)
        pypto.assemble(target + 1.0, [0, 0], out)

    after_infer = compile_new_ir(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((8,), "index"),
        _tensor((8, 16), "source"),
        _tensor((16, 16), "out"),
        pipeline=[("infer_token_pass", ir.Pass.infer_token_pass())],
        create_new_logical_tensor=True,
    )
    check_snapshot(after_infer, A2_READ_ORIGIN_AFTER_ALIAS_WRITE_EXPECTED_IR)
    tensor_ops = [stmt for stmt in after_infer.body if isinstance(stmt, ir.TensorOpStmt)]
    index_add = next(op for op in tensor_ops if op.opcode == "INDEX_ADD")
    adds = next(op for op in tensor_ops if op.opcode == "ADDS")
    assert index_add.result_token[0] in adds.tokens


WRITE_ALIAS_THEN_WRITE_ORIGIN_EXPECTED_IR = """
function foo incast(logical_tensor %target, logical_tensor %index, logical_tensor %source, logical_tensor %values, logical_tensor %out) outcast() #type(Opaque) #entry(false) {
    logical_tensor %$0, token %$0_w, token %target_r = RESHAPE(%target) token();
    logical_tensor %$0_0, token %$0_0_w, token %$0_r, token %source_r, token %index_r = INDEX_ADD(%$0, %source, %index) token(%$0_w);
    logical_tensor %target_0, token %target_0_w, token %target_r_0, token %values_r, token %index_r = INDEX_PUT(%target, %values, %index) token(%$0_0_w);
    logical_tensor %$3, token %$3_w, token %target_0_r = ADDS(%target_0) token(%target_0_w);
    logical_tensor %out_1, token %out_1_w, token %$3_r = ASSEMBLE(%$3) token(%$3_w);
    return %target_0, %index, %source, %values, %out_1;
}
"""


def test_write_alias_then_write_origin_keeps_waw():
    """A write through the origin must wait for a previous write through its alias."""

    def foo(target, index, source, values, out):
        alias = pypto.reshape(target, [16, 16], inplace=True)
        pypto.index_add_(alias, 0, index, source)
        pypto.index_put_(target, (index,), values, False)
        pypto.assemble(target + 1.0, [0, 0], out)

    after_infer = compile_new_ir(
        foo,
        _tensor((16, 16), "target"),
        _index_tensor((8,), "index"),
        _tensor((8, 16), "source"),
        _tensor((8, 16), "values"),
        _tensor((16, 16), "out"),
        pipeline=[("infer_token_pass", ir.Pass.infer_token_pass())],
        create_new_logical_tensor=True,
    )
    check_snapshot(after_infer, WRITE_ALIAS_THEN_WRITE_ORIGIN_EXPECTED_IR)
    tensor_ops = [stmt for stmt in after_infer.body if isinstance(stmt, ir.TensorOpStmt)]
    index_add = next(op for op in tensor_ops if op.opcode == "INDEX_ADD")
    index_put = next(op for op in tensor_ops if op.opcode == "INDEX_PUT")
    assert index_add.result_token[0] in index_put.tokens
