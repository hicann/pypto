# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto

from ..test_common import check_snapshot, run_merge_pass

ir = """
@ir.function
def foo(x@0: ir.Tensor, y@1: ir.Tensor, z@2: ir.Tensor):
    View_x@3 = VIEW(x@0, attrs=["fromOffset": [0, 0], "toValidShape": [RUNTIME_Min(RUNTIME_Max(RUNTIME_GetInputShapeDim(ARG_x,0), 0), 16), 16]])
    for loop_idx_37 in ir.range(0, (((RUNTIME_GetInputShapeDim(ARG_x,0)-0)/4)*4), 4, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 4}):
        if (loop_idx_37==0):
            if (RUNTIME_GetInputShapeDim(ARG_x,0)<=(loop_idx_37+4)):
                $0@5 = ADDS(View_x@3)
                $5@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5@9, attrs=["toOffset": [0, 0]])
                $8@11 = SUBS(View_x@3)
                $12@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12@14, attrs=["toOffset": [0, 0]])
                $15@16 = SUBS(View_x@3)
                $19@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19@19, attrs=["toOffset": [0, 0]])
                $22@21 = SUBS(View_x@3)
                $25@23 = ADDS(View_x@3)
                z@2 = ASSEMBLE($25@23, attrs=["toOffset": [0, 0]])
                ir.yield_()
            else:
                $0_1@5 = ADDS(View_x@3)
                $5_2@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5_2@9, attrs=["toOffset": [0, 0]])
                $8_2@11 = SUBS(View_x@3)
                $12_2@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12_2@14, attrs=["toOffset": [0, 0]])
                $15_2@16 = SUBS(View_x@3)
                $19_2@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19_2@19, attrs=["toOffset": [0, 0]])
                $22_2@21 = SUBS(View_x@3)
                $26@24 = SUBS(View_x@3)
                z@2 = ASSEMBLE($26@24, attrs=["toOffset": [0, 0]])
                ir.yield_()
            ir.yield_()
        else:
            if (RUNTIME_GetInputShapeDim(ARG_x,0)<=(loop_idx_37+4)):
                $1@6 = SUBS(View_x@3)
                $5_5@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5_5@9, attrs=["toOffset": [0, 0]])
                $8_5@11 = SUBS(View_x@3)
                $12_5@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12_5@14, attrs=["toOffset": [0, 0]])
                $15_5@16 = SUBS(View_x@3)
                $19_5@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19_5@19, attrs=["toOffset": [0, 0]])
                $22_5@21 = SUBS(View_x@3)
                $25_0@23 = ADDS(View_x@3)
                z@2 = ASSEMBLE($25_0@23, attrs=["toOffset": [0, 0]])
                ir.yield_()
            else:
                $1_1@6 = SUBS(View_x@3)
                $5_4@9 = SUBS(View_x@3)
                z@2 = ASSEMBLE($5_4@9, attrs=["toOffset": [0, 0]])
                $8_4@11 = SUBS(View_x@3)
                $12_4@14 = SUBS(View_x@3)
                z@2 = ASSEMBLE($12_4@14, attrs=["toOffset": [0, 0]])
                $15_4@16 = SUBS(View_x@3)
                $19_4@19 = SUBS(View_x@3)
                z@2 = ASSEMBLE($19_4@19, attrs=["toOffset": [0, 0]])
                $22_4@21 = SUBS(View_x@3)
                $26_0@24 = SUBS(View_x@3)
                z@2 = ASSEMBLE($26_0@24, attrs=["toOffset": [0, 0]])
                ir.yield_()
            ir.yield_()
        continue
    for loop_idx_37_0 in ir.range((((RUNTIME_GetInputShapeDim(ARG_x,0)-0)/4)*4), RUNTIME_GetInputShapeDim(ARG_x,0), 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_37_0==0):
            if (RUNTIME_GetInputShapeDim(ARG_x,0)<=(loop_idx_37_0+1)):
                $30@25 = ADDS(View_x@3)
                $34@28 = ADDS(View_x@3)
                z@2 = ASSEMBLE($34@28, attrs=["toOffset": [0, 0]])
                ir.yield_()
            else:
                $30_1@25 = ADDS(View_x@3)
                $35@29 = SUBS(View_x@3)
                z@2 = ASSEMBLE($35@29, attrs=["toOffset": [0, 0]])
                ir.yield_()
            ir.yield_()
        else:
            if (RUNTIME_GetInputShapeDim(ARG_x,0)<=(loop_idx_37_0+1)):
                $31@26 = SUBS(View_x@3)
                $34_0@28 = ADDS(View_x@3)
                z@2 = ASSEMBLE($34_0@28, attrs=["toOffset": [0, 0]])
                ir.yield_()
            else:
                $31_1@26 = SUBS(View_x@3)
                $35_0@29 = SUBS(View_x@3)
                z@2 = ASSEMBLE($35_0@29, attrs=["toOffset": [0, 0]])
                ir.yield_()
            ir.yield_()
        continue
    return x@0, y@1, z@2
"""  # noqa: E501


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
    func = run_merge_pass(foo, x, y, z)
    check_snapshot(func, ir)
