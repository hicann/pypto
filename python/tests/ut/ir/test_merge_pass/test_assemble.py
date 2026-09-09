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
from pypto.pil.compile_pipeline import compile_new_ir

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent

IR = _GOLDEN_DIR / "test_assemble_local_tensor.pypto"

def test_assemble_local_tensor():
    def foo(a, b, y):
        for i in pypto.loop(10):
            if i == 0:
                x = a + 1
            else:
                x = b + 1
            t = pypto.Tensor((64, 32), pypto.DT_FP32, 't')
            y = pypto.full([32, 32], 1.0, pypto.DT_FP32)
            t[0:, :] = x
            t[32:, :] = y
            y[i * 64:, :] = t + 1

    a = pypto.Tensor((32, 32), pypto.DT_FP32, 'a')
    b = pypto.Tensor((32, 32), pypto.DT_FP32, 'b')
    out = pypto.Tensor((-1, 32), pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, a, b, out)
    check_snapshot(func, IR)

def test_assemble_local_tensor_full_pipeline():
    """The else branch's ASSEMBLE clones share the branch's own buffer rawtensors after
    merge_stmts; keep them through the full lowering (remove_redundant_token +
    create_root_functions), which run_merge_pass stops before."""
    def foo(a, b, y):
        pypto.set_vec_tile_shapes(32, 32)
        for i in pypto.loop(10):
            if i == 0:
                x = a + 1
            else:
                x = b + 1
            t = pypto.Tensor((64, 32), pypto.DT_FP32, 't')
            y = pypto.full([32, 32], 1.0, pypto.DT_FP32)
            t[0:, :] = x
            t[32:, :] = y
            y[i * 64:, :] = t + 1

    a = pypto.Tensor((32, 32), pypto.DT_FP32, 'a')
    b = pypto.Tensor((32, 32), pypto.DT_FP32, 'b')
    out = pypto.Tensor((-1, 32), pypto.DT_FP32, 'out')
    func = compile_new_ir(foo, a, b, out, create_new_logical_tensor=True)
    assert func.name == 'foo'
