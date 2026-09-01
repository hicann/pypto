# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from pathlib import Path

import pypto

from .test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent

IR = _GOLDEN_DIR / "test_drop_none_iterargs.pypto"


def online_softmax_like_kernel(a, b):
    """Abstract of flash_attention_varlen_forward_950 inner k_tile loop:
    - a real carried accumulator `acc` (as mi/li/oi_update) crossing iterations,
    - branch-local temporaries (pij/m/l/o1/o2 as pij/mi_new/li_new/out_fp32/out_bf16)
      defined and consumed only inside begin/else/end branches.
    """
    n = pypto.symbolic_scalar("n")
    for x in pypto.loop(10):
        for y in pypto.loop(10):
            acc = pypto.tensor([16, 16], pypto.DT_FP32, "acc")
            for i in pypto.loop(n, unroll_list=[2]):
                if i == 0:
                    pij = pypto.mul(a, a)
                    acc[:] = pypto.add(pij, 1)
                else:
                    m = pypto.add(acc, 2)
                    li = pypto.sub(acc, 2)
                    acc[:] = pypto.add(m, li)
                    if i + 1 >= n:
                        o1 = pypto.div(acc, li, precision_type=pypto.PrecisionType.INTRINSIC)
                        o2 = pypto.cast(o1, pypto.DT_BF16)
                        b[:] = pypto.cast(o2, pypto.DT_FP32)
                    else:
                        acc[:] = pypto.add(acc, 1)


def test_drop_none_iterargs_snapshot():
    a = pypto.Tensor((16, 16), pypto.DT_FP32, "a")
    b = pypto.Tensor((16, 16), pypto.DT_FP32, "b")
    func = run_merge_pass(online_softmax_like_kernel, a, b)
    check_snapshot(func, IR)
