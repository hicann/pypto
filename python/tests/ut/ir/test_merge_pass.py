# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Comprehensive tests for ir.Pass.merge_stmts_into_if."""

import logging

import pypto
from pypto import ir, pil


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
                if i == 1:  # impossible under i == 0
                    t = xv + yv  # DEAD -> must be pruned
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
            if i + 1 == 0:  # UNSAT under the loop range i in {0, 1} -> then-branch pruned
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
            for j in pypto.loop(2):  # inner loop in the middle -> barrier (self-contained)
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
            r = t + xv  # consumer of the if's result `t`
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
                if i == 1:  # nested only inside the i == 0 branch
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
        _yv = pypto.view(y, [16, 16], [0, 0])
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
        _yv = pypto.view(y, [16, 16], [0, 0])
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


def test_unroll_list():

    def foo(a, b, c):
        m, n = a.shape[0], a.shape[1]
        for x in pypto.loop(10):
            for y in pypto.loop(10):
                for z in pypto.loop(n):
                    cur_seq = m - (n - 1 - z)
                    k = (cur_seq + 32 - 1) // 32
                    for i in pypto.loop(k, unroll_list=[32]):
                        t = c + 1
                        if i == 0:
                            t = t + 1
                        else:
                            t = t + 2
                        if i == k - 1:
                            t = t + 3
                        else:
                            t = t + 4
                        b[:] = t
    x = pypto.Tensor((-1, -1), pypto.DT_FP32)
    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    _run_merge(foo, x, y, z)
