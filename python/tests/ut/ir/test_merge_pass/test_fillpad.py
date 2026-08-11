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
def foo(a@0: ir.Tensor, b@1: ir.Tensor):
    for loop_idx_12, (b_1,) in ir.range(0, (((n-0)/2)*2), 2, init_values=(b@1,), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 2}):
        if ((loop_idx_12+1)==(n-1)):
            $3@4 = VEC_DUP()
            b_5@1 = ADDS($3@4)
            b_7 = ir.yield_(b_5@1)
        else:
            b_7 = ir.yield_(b_1@1)
        b_9 = continue b_7@1
    for loop_idx_12_0, (b_11,) in ir.range((((n-0)/2)*2), n, 1, init_values=(b_9@1,), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_12_0==(n-1)):
            $8@6 = VEC_DUP()
            b_12@1 = ADDS($8@6)
            b_14 = ir.yield_(b_12@1)
        else:
            b_14 = ir.yield_(b_11@1)
        b_16 = continue b_14@1
    return a@0, b_16@1
""" # noqa: E501


def test_fillpad():
    """A local defined only inside one branch (x) must not become a loop carry."""

    def foo(a, b):
        n = pypto.symbolic_scalar("n")
        for i in pypto.loop(n, unroll_list=[2]):
            if i == n-1:
                x = pypto.full([32, 32], 0, pypto.DT_FP32)
                b[:] = x + 1

    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = run_merge_pass(foo, y, z)
    check_snapshot(func, IR)
