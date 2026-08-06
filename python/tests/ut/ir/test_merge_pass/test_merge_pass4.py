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
    View_x@3 = VIEW(x@0, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
    View_y@4 = VIEW(y@1, attrs=["fromOffset": [0, 0], "toValidShape": [16, 16]])
    for loop_idx_32 in ir.range(0, 2, 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_32==0):
            $0@5 = ADD(View_x@3, View_y@4)
            a_0 = ir.yield_($0@5)
        else:
            $1@6 = SUB(View_x@3, View_y@4)
            a_0 = ir.yield_($1@6)
        for loop_idx_50 in ir.range(0, 2, 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
            z@2 = ASSEMBLE(a_0@5, attrs=["toOffset": [(loop_idx_50*16), 0]])
            continue
        if (loop_idx_32==1):
            $3@7 = ADD(a_0@5, View_y@4)
            z@2 = ASSEMBLE($3@7, attrs=["toOffset": [0, 0]])
            ir.yield_()
        else:
            $4@8 = SUB(a_0@5, View_y@4)
            z@2 = ASSEMBLE($4@8, attrs=["toOffset": [0, 0]])
            ir.yield_()
        continue
    return x@0, y@1, z@2
"""


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
    func = run_merge_pass(foo, x, y, z)
    check_snapshot(func, ir)
