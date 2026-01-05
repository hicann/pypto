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
""" """
import torch
import pytest

import pypto
import pypto._controller as controller


def test_static_function():
    dtype = pypto.DT_FP16
    shape = (8, 8)
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_b")
    c = pypto.tensor(shape, dtype, "tensor_c")

    with pypto.function("ADD", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        c[:] = pypto.add(a, b)

    print(controller.dump())
    # Replace True with False to see graph
    assert isinstance(c, pypto.tensor)


@pytest.mark.skip(reason="Case is no longer maintained")
def test_empty_function():
    dtype = pypto.DT_FP16
    a = pypto.tensor((8, 8), dtype, "tensor_a")
    fnc_name = "name"

    with pypto.function(fnc_name, a):
        pypto.set_vec_tile_shapes(8, 8)

    assert True


@pypto.jit
def test_dyn_function():
    dtype = pypto.DT_FP16
    shape = (8, 8)
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_b")
    c = pypto.tensor(shape, dtype, "tensor_c")

    pypto.set_vec_tile_shapes(8, 8)
    c[:] = pypto.add(a, b)

    print(controller.dump())


def test_mix_assemble_and_common_op():
    x = pypto.from_torch(torch.ones(32, 32, dtype=torch.float32))
    y = pypto.from_torch(torch.empty(32, 32, dtype=torch.float32))

    with pytest.raises(RuntimeError, match=".*mix assemble and common operation.*"):
        with pypto.function("cycle", [x, y]):
            pypto.set_vec_tile_shapes(16, 16)
            a = pypto.zeros([32, 32])
            b = a[:16, :16]
            a[16:, 16:] = b.exp()
            y[:] = x + a


def test_cycle_detection():
    x = pypto.from_torch(torch.ones(32, 32, dtype=torch.float32))
    y = pypto.from_torch(torch.empty(32, 32, dtype=torch.float32))

    with pytest.raises(RuntimeError, match=".*cycle detected.*"):
        with pypto.function("cycle", [x, y]):
            pypto.set_vec_tile_shapes(16, 16)
            b = x[:16, :16]
            x[16:, 16:] = b.exp()
            y[:] = x + x


def test_nested_function():
    x = pypto.from_torch(torch.ones(32, 32, dtype=torch.float32))
    y = pypto.from_torch(torch.empty(32, 32, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="function nested is not allowed"):
        with pypto.function("graph", [x, y]):
            pypto.set_vec_tile_shapes(16, 16)
            a = pypto.zeros([32, 32])
            c = pypto.zeros([16, 16])

            with pypto.function("graph_0"):
                b = a[:16, :16]
                c = b.exp()


def test_move_from_scalar():
    y = pypto.from_torch(torch.empty(32, 32, dtype=torch.float32))

    with pytest.raises(TypeError, match="'int' type cannot be moved to Tensor"):
        with pypto.function("graph", [y]):
            y[:] = 0

