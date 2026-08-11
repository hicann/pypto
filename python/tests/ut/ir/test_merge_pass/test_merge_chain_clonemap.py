# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto
from pypto import ir, pil

from ..test_common import check_snapshot

IR = """
@ir.function
def foo(x@0: ir.Tensor, y@1: ir.Tensor, z@2: ir.Tensor):
    for loop_idx_20, (i, s0, t0, t1) in ir.range(0, 2, 1, init_values=(None, None, None, None), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (0<n):
            if (0<m):
                $0@3 = ADD(x@0, y@1)
                $3@5 = ADD($0@3, y@1)
                $6@7 = ADD($3@5, x@0)
                z@2 = ASSEMBLE($6@7, attrs=["toOffset": [0, 0]])
                $0_0, t1_0, $6_0, z = ir.yield_($0@3, $3@5, $6@7, z@2)
            else:
                $0_1@3 = ADD(x@0, y@1)
                $4@6 = SUB($0_1@3, y@1)
                $6_2@7 = ADD($4@6, x@0)
                z@2 = ASSEMBLE($6_2@7, attrs=["toOffset": [0, 0]])
                $0_0, t1_0, $6_0, z = ir.yield_($0_1@3, $4@6, $6_2@7, z@2)
            t0_0, t1_2, $6_1, z = ir.yield_($0_0@3, t1_0@5, $6_0@7, z@2)
        else:
            if (0<m):
                $1@4 = SUB(x@0, y@1)
                $3_0@5 = ADD($1@4, y@1)
                $6_4@7 = ADD($3_0@5, x@0)
                z@2 = ASSEMBLE($6_4@7, attrs=["toOffset": [0, 0]])
                $1_0, t1_3, $6_5, z = ir.yield_($1@4, $3_0@5, $6_4@7, z@2)
            else:
                $1_1@4 = SUB(x@0, y@1)
                $4_0@6 = SUB($1_1@4, y@1)
                $6_3@7 = ADD($4_0@6, x@0)
                z@2 = ASSEMBLE($6_3@7, attrs=["toOffset": [0, 0]])
                $1_0, t1_3, $6_5, z = ir.yield_($1_1@4, $4_0@6, $6_3@7, z@2)
            t0_0, t1_2, $6_1, z = ir.yield_($1_0@4, t1_3@5, $6_5@7, z@2)
        i_0, s0_0, t0_1, t1_1 = continue loop_idx_20, $6_1@7, t0_0@3, t1_2@5
    return x@0, y@1, z@2
"""  # noqa: E501


def test_merge_chain_clonemap():
    """Chained AppendIntoIfStmt produces a cloneMap chain (A->B->C->D)."""

    def foo(x, y, z):
        n = pypto.symbolic_scalar("n")
        m = pypto.symbolic_scalar("m")
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            if n > 0:
                t0 = x + y
            else:
                t0 = x - y
            if m > 0:
                t1 = t0 + y
            else:
                t1 = t0 - y
            s0 = t1 + x
            pypto.assemble(s0, [0, 0], z)

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([32, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([32, 32], pypto.DT_FP32, 'z')

    b = ir.IRBuilder()
    func = pil.compile(foo, x, y, z)
    prog = b.create_program([func], "main", ir.Span.unknown())
    merge = ir.Pass.merge_stmts_into_if()
    prog = merge(prog)

    func = prog.functions[func.name]

    # check_snapshot verifies the full IR string against golden.
    # Key difference from pre-fix IR: continue references $6_1@7 (outer returnVar)
    # instead of $6_0@7 (intermediate inner returnVar, not defined in any outer yield).
    check_snapshot(func, IR)
