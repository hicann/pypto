#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

import pypto


@pytest.mark.parametrize("divisor", [0, -1])
def test_assume_divisible_rejects_non_positive_divisor(divisor):
    with pytest.raises(ValueError, match="divisor must be positive"):
        pypto.experimental.assume_divisible(pypto.SymbolicScalar("vm"), divisor)


def test_assume_divisible_accepts_divisible_constant():
    result = pypto.experimental.assume_divisible(256, 128)

    assert result.is_concrete()
    assert result.concrete() == 256


def test_assume_divisible_rejects_non_divisible_constant():
    with pytest.raises(ValueError, match="constant 130 is not divisible by 128"):
        pypto.experimental.assume_divisible(130, 128)


def test_assume_divisible_returns_symbolic_input_unchanged():
    vm = pypto.SymbolicScalar("vm")

    assert pypto.experimental.assume_divisible(vm, 128) is vm
