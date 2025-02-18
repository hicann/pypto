#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import pypto
import pytest
import torch


def test_verify_dynamic_ops_assemble():
    s = 32
    n = 2
    m = 1
    t0 = pypto.tensor((n * s, m * s), pypto.DT_FP32)
    t1 = pypto.tensor((n * s, m * s), pypto.DT_FP32)
    out = pypto.tensor((n * s, m * s), pypto.DT_FP32)
    t0_data = torch.ones((n * s, m * s))
    t1_data = torch.ones((n * s, m * s)) * 2
    out_data = torch.zeros((n * s, m * s))
    golden = torch.ones((n * s, m * s)) * 3
    pypto.set_verify_options(
        enable_pass_verify=True
    )
    pypto.set_verify_golden_data([t0_data, t1_data, out_data], [t0_data, t1_data, golden])
    with pypto.function("main", t0, t1, out):
        pypto.set_vec_tile_shapes(8, 8)
        for idx in pypto.loop(10):
            pypto.pass_verify_print(t0)
            t0a = pypto.view(t0, [s, s], [0, 0])
            t0b = pypto.view(t0, [s, s], [s, 0])
            t1a = pypto.view(t1, [s, s], [0, 0])
            t1b = pypto.view(t1, [s, s], [s, 0])
            t2a = t0a + t1a
            t2b = t0b + t1b
            pypto.assemble(t2a, [0, 0], out)
            pypto.assemble(t2b, [s, 0], out)
            pypto.pass_verify_save(out, "tensor_out_idx$idx", idx=idx)
