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
    dce = ir.Pass.aggressive_dce()
    canonical = ir.Pass.canonicalize()
    logging.info("\norigin: %s\n", prog)
    prog = canonical(prog)
    logging.info("\ncanonical: %s\n", prog)
    prog = dce(prog)
    logging.info("\ndce: %s\n", prog)
    prog = dce(canonical(prog))
    logging.info("\ndce canonical: %s\n", prog)

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


def test_dce_keeps_tensor_op():
    """A TensorOpStmt whose result is used by ContinueStmt must survive."""
    def foo(x, y):
        for i in pypto.loop(x.shape[0] // 32):
            pypto.set_vec_tile_shapes(32, 32)
            ta = x[i:i + 32, :]
            tb = ta + 1
            y[i:, :] = tb

    x = pypto.Tensor((-1, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((-1, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)

    tensor_ops = _collect_stmts(func.body, ir.TensorOpStmt)
    opcodes = [op.opcode for op in tensor_ops]
    assert 'VIEW' in opcodes, f"VIEW eliminated but used by continue: {opcodes}"

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
    assert len(func.body.stmts) == 1, "Expected only return stmt after DCE"


def _run_merge(func, *args):
    """Compile a kernel and run canonicalize + dce + merge_stmts_into_if, stopping before lowering
    so the resulting if-tree (func.body) is inspectable. Mirrors compile_new_ir's first half."""
    b = ir.IRBuilder()
    func = pil.compile(func, *args)
    prog = b.create_program([func], "main", ir.Span.unknown())
    dce = ir.Pass.aggressive_dce()
    canonical = ir.Pass.canonicalize()
    merge = ir.Pass.merge_stmts_into_if()
    prog = dce(canonical(prog))
    prog = canonical(merge(prog))
    ir.Pass.run_verifier()(prog)
    logging.info("\nmerged:\n%s" % prog.functions[func.name].body)
    return prog.functions[func.name]


def test_merge_pass1():
    """A loop is a barrier (never duplicated into branches) and the pass recurses into its body."""
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            xv = pypto.view(x, [16, 16], [0, 0])
            yv = pypto.view(y, [16, 16], [0, 0])
            if i == 0:
                t = xv + yv
            else:
                t = xv - yv
            if i == 1:
                t2 = t + xv
            else:
                t2 = t - xv
            pypto.assemble(t2, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    # The loop must stay single (barrier), not be duplicated into every leaf.
    assert len(_collect_stmts(func.body, ir.ForStmt)) == 1
    # Conditions nested in source order inside the loop body (pass recurses into loop bodies).
    assert len(_collect_stmts(func.body, ir.IfStmt)) >= 2


def test_merge_pass2():
    """Under `i == 0` the nested `i == 1` is UNSAT, so its branch is pruned (not 2^N leaves)."""
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(2):
            if i == 0:
                if i == 1:          # impossible under i == 0
                    t = xv + yv      # DEAD -> must be pruned
                else:
                    t = xv - yv
            else:
                t = xv + yv
            pypto.assemble(t, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    # Only the outer `i == 0` if remains; the impossible nested `i == 1` collapsed (would be 2 ifs
    # without SAT pruning).
    assert len(_collect_stmts(func.body, ir.IfStmt)) == 1


def test_merge_pass3():
    """`i + 1 == 0` is impossible for i in {0, 1}: the solver models the loop range (i >= 0),
    proves the then-branch UNSAT, and prunes it -> the if collapses to the surviving else branch.
    """
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(2):
            if i + 1 == 0:          # UNSAT under the loop range i in {0, 1} -> then-branch pruned
                t = xv + yv
            else:
                t = xv - yv
            pypto.assemble(t, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    assert len(_collect_stmts(func.body, ir.IfStmt)) == 0


def test_merge_pass4():
    """A loop between two ifs splits the region: each side is tree-built independently, loop stays single."""
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(2):
            if i == 0:
                a = xv + yv
            else:
                a = xv - yv
            for j in pypto.loop(2):     # inner loop in the middle -> barrier (self-contained)
                pypto.assemble(a, [j * 16, 0], z)
            if i == 1:
                c = a + yv
            else:
                c = a - yv
            pypto.assemble(c, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    # Inner loop appears exactly once (not duplicated across the surrounding if branches).
    assert len(_collect_stmts(func.body, ir.ForStmt)) == 2  # outer + inner, each single


def test_merge_pass5():
    """An if that yields a value consumed after it: the consumer is sunk into both branches and the
    branch-local def is used in each (SSA preserved via return_vars + yield).
    """
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(2):
            if i == 0:
                t = xv + yv
            else:
                t = xv - yv
            r = t + xv              # consumer of the if's result `t`
            pypto.assemble(r, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    # The value-producing if is preserved...
    assert len(_collect_stmts(func.body, ir.IfStmt)) == 1
    # ...and the consumer `r = t + xv` is duplicated into both branches (one ADD per branch).
    adds = [op for op in _collect_stmts(func.body, ir.TensorOpStmt) if op.opcode == 'ADD']
    assert len(adds) >= 2, f"expected the consumer sunk into both branches, got {len(adds)} ADD"


def test_merge_pass6():
    """A nested if that exists only under one branch stays nested only under that branch's leaves."""
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(2):
            if i == 0:
                t = xv + yv
                if i == 1:          # nested only inside the i == 0 branch
                    t = t + yv
            else:
                t = xv - yv
            pypto.assemble(t, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    # Outer i == 0 plus the nested i == 1 under the else (where i != 0 keeps both); the nested if
    # under i == 0 was pruned (0 == 1 impossible). Inspect via the logged IR.
    assert len(_collect_stmts(func.body, ir.IfStmt)) == 1


def test_merge_pass7():
    """A nested if that exists only under one branch stays nested only under that branch's leaves."""
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(x.shape[0], unroll_list=[4]):
            if i == 0:
                t = xv + 1
            else:
                t = xv - 1
            t = t + 2
            if i == x.shape[0] - 1:
                t = xv + 1
            else:
                t = xv - 1
            pypto.assemble(t, [0, 0], z)

    x = pypto.Tensor([-1, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([-1, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([-1, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    assert len(_collect_stmts(func.body, ir.IfStmt)) == 6


def test_merge_pass8():
    """A nested if that exists only under one branch stays nested only under that branch's leaves."""
    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(x.shape[0], unroll_list=[4]):
            if pypto.is_loop_begin(i):
                t = xv + 1
            else:
                t = xv - 1
            t = t + 2
            if pypto.is_loop_end(i):
                t = xv + 1
            else:
                t = xv - 1
            pypto.assemble(t, [0, 0], z)

    x = pypto.Tensor([-1, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([-1, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([-1, 32], pypto.DT_FP32, 'z')
    func = _run_merge(foo, x, y, z)
    assert len(_collect_stmts(func.body, ir.IfStmt)) == 6


def test_forstmt_attrs_and_step_name_preserved():
    """ForStmt attrs and step-based path func naming must survive DCE/canonicalize/TransformStmts."""
    def foo(x, z):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(0, 4, name="LOOP_TEST", parallel=True, unroll_list=[2]):
            x_view = pypto.view(x, [16, 32], [i * 16, 0])
            pypto.assemble(x_view, [i * 16, 0], z)

    x = pypto.Tensor(shape=[64, 32], dtype=pypto.DT_FP32, name="x")
    z = pypto.Tensor(shape=[64, 32], dtype=pypto.DT_FP32, name="z")

    b = ir.IRBuilder()
    func = pil.compile(foo, x, z)
    prog = b.create_program([func], "main", ir.Span.unknown())
    prog = ir.Pass.canonicalize()(prog)
    prog = ir.Pass.aggressive_dce()(prog)
    prog = ir.Pass.create_root_functions()(prog)

    # 1. ForStmt exists (attrs preserved, no crash)
    dyn_func = prog.functions[func.name]
    for_stmts = _collect_stmts(dyn_func.body, ir.ForStmt)
    assert len(for_stmts) >= 1, "Expected at least one ForStmt after create_root_functions"

    # 2. path function name contains step value
    prog_str = str(prog)
    assert "_Unroll2" in prog_str, f"Expected '_Unroll2' (step=2) in program: {prog_str}"


def test_tensor_move():

    def foo(a, b):
        last = pypto.full(a.shape, 1.0, a.dtype)
        for _ in pypto.loop(10):
            a = last + 1
            last[:] = a + 1
            b[0:, 0:] = last
        b[:] = a + 1

    x = pypto.Tensor((32, 32), pypto.DT_FP32, 'x')
    y = pypto.Tensor((32, 32), pypto.DT_FP32, 'y')
    func = _run_dce(foo, x, y)
    for_stmt = func.body[1]
    assert isinstance(for_stmt, ir.ForStmt)
    assert "last" in [v.iterVar.name for v in for_stmt.iter_args]
    add_stmt = func.body[2]
    assert isinstance(add_stmt, ir.TensorOpStmt)
    assert add_stmt.opcode == "ADDS"
    assert add_stmt.result[0].name == 'b_0'
