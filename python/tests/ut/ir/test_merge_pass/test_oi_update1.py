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
def kernel(a@0: ir.Tensor, b@1: ir.Tensor):
    oi_update@2 = TENSOR_ALLOC()
    oi_update1@3 = TENSOR_ALLOC()
    for loop_idx_31, (oi_update_2, oi_update_3) in ir.range(0, 10, 1, init_values=(oi_update@2, oi_update1@3), attrs={"parallel": False, "submit_before_loop": False, "unroll_times": 1}):
        if (loop_idx_31==0):
            oi_update_4@2 = ADDS(a@0)
            oi_update_22@3 = ADDS(a@0)
            oi_update_43, oi_update_44 = ir.yield_(oi_update_4@2, oi_update_22@3)
        else:
            if (10<=(loop_idx_31+1)):
                oi_update_10@2 = ADD(a@0, oi_update_2@2)
                b@1 = ASSEMBLE(oi_update_10@2, attrs=["toOffset": [0, 0]])
                oi_update_27@3 = ADD(a@0, oi_update_3@3)
                b@1 = ASSEMBLE(oi_update_27@3, attrs=["toOffset": [0, 0]])
                oi_update_49, oi_update_50 = ir.yield_(oi_update_2@2, oi_update_3@3)
            else:
                oi_update_54@2 = ADD(a@0, oi_update_2@2)
                oi_update_46@3 = ADD(a@0, oi_update_3@3)
                oi_update_49, oi_update_50 = ir.yield_(oi_update_54@2, oi_update_46@3)
            oi_update_43, oi_update_44 = ir.yield_(oi_update_49@2, oi_update_50@3)
        oi_update_40, oi_update_41 = continue oi_update_43@2, oi_update_44@3
    return a@0, b@1
"""  # noqa: E501


def test_oi_update1():

    def kernel(a, b):
        pypto.set_vec_tile_shapes(32, 32)
        oi_update_0 = pypto.tensor([32, 32], pypto.DT_FP32, "oi_update")
        oi_update_1 = pypto.tensor([32, 32], pypto.DT_FP32, "oi_update1")

        for k_tile_idx in pypto.loop(10, name="k_tile_loop"):
            for h_s_idx in range(2):
                if h_s_idx == 0:
                    oi_update = oi_update_0
                else:
                    oi_update = oi_update_1

                if pypto.is_loop_begin(k_tile_idx):
                    if pypto.is_loop_end(k_tile_idx):
                        pass
                    else:
                        oij = a + 1
                        oi_update[:] = oij
                else:
                    oi_tmp = a + oi_update
                    if pypto.is_loop_end(k_tile_idx):
                        pypto.assemble(oi_tmp, [0, 0], b)
                        pass
                    else:
                        oi_update[:] = oi_tmp


    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = run_merge_pass(kernel, y, z)
    check_snapshot(func, IR)
