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

IR = """
@ir.function
def foo(x@0: ir.Tensor, y@1: ir.Tensor, z@2: ir.Tensor):
    for loop_idx_10 in ir.range(0, 2, 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_10==0):
            View_x@3 = VIEW(x@0, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
            View_y@4 = VIEW(y@1, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
            $0@5 = ADD(View_x@3, View_y@4)
            $4@8 = SUB($0@5, View_x@3)
            z@2 = ASSEMBLE($4@8, attrs=["toOffset": [0, 0]])
            ir.yield_()
        else:
            if (loop_idx_10==1):
                View_x_8@3 = VIEW(x@0, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
                View_y_8@4 = VIEW(y@1, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
                $1@6 = SUB(View_x_8@3, View_y_8@4)
                $3@7 = ADD($1@6, View_x_8@3)
                z@2 = ASSEMBLE($3@7, attrs=["toOffset": [0, 0]])
                ir.yield_()
            else:
                View_x_6@3 = VIEW(x@0, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
                View_y_6@4 = VIEW(y@1, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
                $1_1@6 = SUB(View_x_6@3, View_y_6@4)
                $4_0@8 = SUB($1_1@6, View_x_6@3)
                z@2 = ASSEMBLE($4_0@8, attrs=["toOffset": [0, 0]])
                ir.yield_()
            ir.yield_()
        continue
    return x@0, y@1, z@2
"""


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
    func = run_merge_pass(foo, x, y, z)
    check_snapshot(func, IR)
