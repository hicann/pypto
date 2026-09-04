# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import pypto

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent

IR1 = _GOLDEN_DIR / "test_valid_shape1.pypto"

def test_valid_shape1():

    def foo(a: pypto.Tensor, b: pypto.Tensor,):
        tile = 32
        n = pypto.ceildiv(a.shape[0], tile)
        for x in pypto.loop(n):
            if x == n - 1:
                t = pypto.view(a, [32, 32], [x*32, 0], valid_shape=[a.shape[0] - x*32, 32])
            else:
                t = pypto.view(a, [32, 32], [x*32, 0], valid_shape=[32, 32])
            t += 1
            pypto.assemble(t, [x*32, 0], b)

    x = pypto.Tensor((-1, 32), pypto.DT_FP32)
    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = run_merge_pass(foo, x, y)
    check_snapshot(func, IR1)
