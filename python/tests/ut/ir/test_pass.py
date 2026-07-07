# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Comprehensive tests for ir.Pass.aggressive_dce."""
import logging

import pypto
from pypto import pil, ir


def _collect_stmts(stmt, cls):
    """Recursively collect all stmts of a given IR type."""
    result = []
    if isinstance(stmt, cls):
        result.append(stmt)
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            result.extend(_collect_stmts(s, cls))
    if isinstance(stmt, ir.ForStmt):
        result.extend(_collect_stmts(stmt.body, cls))
    if isinstance(stmt, ir.IfStmt):
        result.extend(_collect_stmts(stmt.then_body, cls))
        if stmt.else_body is not None:
            result.extend(_collect_stmts(stmt.else_body, cls))
    if isinstance(stmt, ir.WhileStmt):
        result.extend(_collect_stmts(stmt.body, cls))
    return result


def _run_dce(func, *args):
    """Build a program from a compiled function and run aggressive DCE."""
    b = ir.IRBuilder()
    func = pil.compile(func, *args)
    prog = b.create_program([func], "main", ir.Span.unknown())
    dce = ir.Pass.aggressive_dce();
    canonical = ir.Pass.canonicalize()
    logging.info("\norigin: %s\n", prog)
    prog = canonical(prog)
    logging.info("\ncanonical: %s\n", prog)
    prog = dce(prog)
    logging.info("\ndce: %s\n", prog)
    prog = dce(canonical(prog))
    logging.info("\ndec canonical: %s\n", prog)

    return prog.functions[func.name]


# ---------- TensorOpStmt chain propagation ----------


def test_dce_tensor_chain_view_adds_assemble():
    """VIEW -> ADDS -> ASSEMBLE: all three must be preserved.

    The original bug: DCE only propagated liveness through AssignStmts,
    so TensorOpStmt chains were incorrectly treated as dead.
    """
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    assert 'VIEW' in opcodes, f"VIEW was incorrectly eliminated: {opcodes}"
    assert 'ADDS' in opcodes, f"ADDS was incorrectly eliminated: {opcodes}"
    assert 'ASSEMBLE' in opcodes, f"ASSEMBLE was incorrectly eliminated: {opcodes}"


def test_dce_dead_tensor_op_removed():
    """An unused TensorOpStmt should be removed by aggressive DCE."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            _ = ta + 2   # dead op: result unused
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    adds_count = sum(1 for op in tensor_ops if op.opcode == 'ADDS')
    assert adds_count == 1, f"Expected 1 ADDS, got {adds_count}"


def test_dce_long_tensor_chain():
    """A -> B -> C -> ASSEMBLE: entire chain must be preserved.

    Tests multi-step propagation through TensorOpStmts.
    """
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1   # step 1
            tc = tb + 1   # step 2
            y[i:, :] = tc  # ASSEMBLE

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"
    assert 'VIEW' in opcodes, f"VIEW was incorrectly eliminated: {opcodes}"
    assert 'ASSEMBLE' in opcodes, f"ASSEMBLE was incorrectly eliminated: {opcodes}"


def test_dce_branching_tensor_chain():
    """A -> B -> C -> ASSEMBLE and A -> D (dead): only A,B,C,ASSEMBLE kept."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1
            tc = tb + 1
            _ = ta + 99   # dead branch: result unused
            y[i:, :] = tc

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS (dead branch removed), got {adds_count} in {opcodes}"


# ---------- ContinueStmt / BreakStmt live-root ----------


def test_dce_continue_keeps_tensor_op():
    """A TensorOpStmt whose result is used by ContinueStmt must survive."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1
            y[i:, :] = tb
            # continue carries ta -- VIEW must be preserved
            if i > 0:
                continue

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    assert 'VIEW' in opcodes, f"VIEW eliminated but used by continue: {opcodes}"


def test_dce_break_keeps_tensor_op():
    """A TensorOpStmt whose result is used by BreakStmt must survive."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1
            y[i:, :] = tb
            if i > 2:
                break

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    assert 'VIEW' in opcodes, f"VIEW eliminated but used by break: {opcodes}"


# ---------- Nested control flow ----------


def test_dce_nested_for_dead_inner():
    """Dead tensor ops inside a for-loop body should be eliminated."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            _ = ta + 99   # dead: result unused
            y[i:, :] = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 1, f"Expected 1 ADDS, got {adds_count} in {opcodes}"


def test_dce_if_else_branches():
    """Dead code in if/else branches should be eliminated independently."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99   # dead in then-branch
                tb = ta + 1
                y[i:, :] = tb
            else:
                _ = ta + 88   # dead in else-branch
                tb = ta + 2
                y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    # Dead ADDS (ta+99, ta+88) removed; live ADDS (ta+1, ta+2) kept
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"


# ---------- Mixed AssignStmt + TensorOpStmt ----------


def test_dce_nested_if_else():
    """Dead code in nested if/else should be eliminated at every level."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99   # dead in then-branch
                if i == 0:
                    _ = ta + 77   # dead in nested then
                    tb = ta + 1
                else:
                    _ = ta + 66   # dead in nested else
                    tb = ta + 2
                y[i:, :] = tb
            else:
                _ = ta + 88   # dead in else-branch
                tb = ta + 3
                y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    # Dead ADDS (ta+99, ta+77, ta+66, ta+88) removed; live ADDS (ta+1, ta+2, ta+3) kept
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 3, f"Expected 3 ADDS, got {adds_count} in {opcodes}"


def test_dce_nested_loop_if_else():
    """Dead code in if/else inside nested loops should be eliminated."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i * 32:(i + 1) * 32, :]
            for j in pypto.loop(2):
                pypto.set_vec_tile_shapes(16, 16)
                tb = ta[j * 16:(j + 1) * 16, :]
                if j == 0:
                    _ = tb + 99   # dead in inner then
                    tc = tb + 1
                else:
                    _ = tb + 88   # dead in inner else
                    tc = tb + 2
                y[i * 32 + j * 16:i * 32 + (j + 1) * 16, :] = tc

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    # Each inner iteration has: VIEW, dead ADDS removed, live ADDS kept, ASSEMBLE
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"


def test_dce_if_else_with_break():
    """Dead code in if/else branch containing break should be eliminated."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99   # dead
                tb = ta + 1
                y[i:, :] = tb
            else:
                _ = ta + 88   # dead
                tb = ta + 2
                y[i:, :] = tb
                break

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"


def test_dce_if_else_with_continue():
    """Dead code in if/else branch containing continue should be eliminated."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            if i < 2:
                _ = ta + 99   # dead
                tb = ta + 1
                y[i:, :] = tb
                continue
            else:
                _ = ta + 88   # dead
                tb = ta + 2
                y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 2, f"Expected 2 ADDS, got {adds_count} in {opcodes}"


# ---------- Mixed AssignStmt + TensorOpStmt ----------


def test_dce_mixed_assign_and_tensor():
    """Dead scalar assignments and dead tensor ops should both be removed."""
    def foo(x, y, n):
        for i in pypto.loop(n):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i * 32:(i + 1) * 32, :]
            dead_scalar = i * 42   # dead scalar assign
            _ = dead_scalar
            _ = ta + 99            # dead tensor op
            tb = ta + 1
            y[i * 32:(i + 1) * 32, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y, 10)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    adds_count = sum(1 for c in opcodes if c == 'ADDS')
    assert adds_count == 1, f"Expected 1 ADDS (dead one removed), got {adds_count} in {opcodes}"


def test_dce_no_assemble_stmt():
    """Dead scalar assignments and dead tensor ops should both be removed."""
    def foo(x, y, n):
        for i in pypto.loop(n):
            pypto.set_vec_tile_shapes(32, 32)
            if i > 0:
                ta = x[i * 32:(i + 1) * 32, :]
            else:
                ta = x[i * 32:(i + 1) * 32, :]
            dead_scalar = i * 42   # dead scalar assign
            _ = dead_scalar
            _ = ta + 99            # dead tensor op
            tb = ta + 1

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y, 10)
    assert len(func.body.stmts) == 0, "Expected no stmts after DCE"
