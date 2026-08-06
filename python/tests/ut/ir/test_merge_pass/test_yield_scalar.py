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
def foo(x@0: ir.Tensor, y@1: ir.Tensor):
    for loop_idx_5 in ir.range(0, 10, 1, attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_5==0):
            y@1 = ASSEMBLE(x@0, attrs=["toOffset": [32, 32]])
            ir.yield_()
        else:
            y@1 = ASSEMBLE(x@0, attrs=["toOffset": [64, 64]])
            ir.yield_()
        continue
    return x@0, y@1
"""

def test_yield_symbolic_scalar():
    def foo(x, y):
        for i in pypto.loop(10):
            if i == 0:
                off = 0
            else:
                off = 32
            y[off + 32:, off + 32:] = x

    a = pypto.Tensor((64, 32), pypto.DT_FP32, 'a')
    out = pypto.Tensor((32, 32), pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, a, out)
    check_snapshot(func, IR)
