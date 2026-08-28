# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for token dependencies of inplace tensor operations."""


from pathlib import Path

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from .test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent / "test_token_pass_inplace"


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


def test_inplace_reshapes_are_token_transparent():
    def foo(target):
        _alias_0 = pypto.reshape(target, [16, 32], inplace=True)
        _alias_1 = pypto.reshape(target, [8, 64], inplace=True)

    func = compile_new_ir(
        foo,
        _tensor((32, 16), "target"),
        pipeline=[("infer_token_pass", ir.Pass.infer_token_pass())],
        create_new_logical_tensor=True,
    )
    check_snapshot(func, _GOLDEN_DIR / "test_inplace_reshapes_are_token_transparent.pypto")


def test_assemble_writes_through_reshape_aliases_keep_waw():
    def foo(target, src_0, src_1):
        alias_0 = pypto.reshape(target, [16, 32], inplace=True)
        alias_1 = pypto.reshape(target, [8, 64], inplace=True)
        pypto.assemble(src_0, [0, 0], alias_0)
        pypto.assemble(src_1, [0, 0], alias_1)

    func = compile_new_ir(
        foo,
        _tensor((32, 16), "target"),
        _tensor((16, 32), "src_0"),
        _tensor((8, 64), "src_1"),
        pipeline=[("infer_token_pass", ir.Pass.infer_token_pass())],
        create_new_logical_tensor=True,
    )
    check_snapshot(func, _GOLDEN_DIR / "test_assemble_writes_through_reshape_aliases_keep_waw.pypto")


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
    check_snapshot(after_remove, _GOLDEN_DIR / "test_index_put_keeps_write_after_read_dependency.pypto")


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
    check_snapshot(after_remove, _GOLDEN_DIR / "test_index_add_keeps_write_after_read_dependency.pypto")


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
    check_snapshot(
        after_remove, _GOLDEN_DIR / "test_scatter_update_keeps_write_after_read_dependency.pypto")


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
    check_snapshot(after_remove, _GOLDEN_DIR / "test_axpy_keeps_write_after_read_dependency.pypto")


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
    check_snapshot(after_remove, _GOLDEN_DIR / "test_atomic_add_target_is_carried_through_loop.pypto")


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
    check_snapshot(after_infer, _GOLDEN_DIR / "test_read_alias_then_write_origin_keeps_war.pypto")
    tensor_ops = [stmt for stmt in after_infer.body if isinstance(stmt, ir.TensorOpStmt)]
    adds = next(op for op in tensor_ops if op.opcode == "ADDS")
    index_put = next(op for op in tensor_ops if op.opcode == "INDEX_PUT")
    assert adds.result_token[-1] in index_put.tokens


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
    check_snapshot(after_infer, _GOLDEN_DIR / "test_read_origin_after_inplace_on_alias_keeps_raw.pypto")


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
    check_snapshot(after_infer, _GOLDEN_DIR / "test_write_alias_then_write_origin_keeps_waw.pypto")
