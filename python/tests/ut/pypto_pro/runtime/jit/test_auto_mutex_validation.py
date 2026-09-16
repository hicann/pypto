#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate auto_mutex at the public JIT entry point before lazy compilation."""

import pypto_pro.language as pl
import pytest


@pytest.mark.parametrize(
    "auto_mutex",
    ["invalid", "True", "False", "true", "false", "", None, 0, 1, 0.0, 1.0, [], {}, object()],
)
def test_jit_rejects_non_boolean_auto_mutex(auto_mutex):
    with pytest.raises(TypeError, match="auto_mutex must be a bool, got "):
        pl.jit(auto_mutex=auto_mutex)


@pytest.mark.parametrize("auto_mutex", ["False", 1])
def test_jit_direct_call_rejects_non_boolean_auto_mutex(auto_mutex):
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32]):
        return

    with pytest.raises(TypeError, match="auto_mutex must be a bool, got "):
        pl.jit(kernel, auto_mutex=auto_mutex)


@pytest.mark.parametrize("auto_mutex", [True, False])
@pytest.mark.parametrize("direct", [True, False])
def test_jit_accepts_boolean_auto_mutex(auto_mutex, direct):
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32]):
        return

    decorated = pl.jit(kernel, auto_mutex=auto_mutex) if direct else pl.jit(auto_mutex=auto_mutex)(kernel)
    assert callable(decorated[1])


@pytest.mark.parametrize("direct", [True, False])
def test_jit_accepts_default_auto_mutex(direct):
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32]):
        return

    decorated = pl.jit(kernel) if direct else pl.jit()(kernel)
    assert callable(decorated[1])
