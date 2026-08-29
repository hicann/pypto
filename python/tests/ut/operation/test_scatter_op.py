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
import pypto

B, S = 4, 5
IDX0, IDX1 = 2, 5


def make_case(input_dtype=pypto.DT_FP32, index_dtype=pypto.DT_INT64, dim=0, src_dtype=pypto.DT_FP32):
    """Build input/index/src tensors for a 2-dim scatter case on the given axis."""
    index_shape = [IDX0, IDX1] if dim == 0 else [B, 2]
    src_shape = list(index_shape)
    x = pypto.tensor([B, S], input_dtype, "x")
    index = pypto.tensor(index_shape, index_dtype, "index")
    src = pypto.tensor(src_shape, src_dtype, "src")
    return x, index, src


def test_scatter_inplace_scalar_src():
    x, index, _ = make_case()
    with pypto.function("SCATTER_INPLACE_SCALAR", x, index):
        pypto.set_vec_tile_shapes(4, 16)
        res = pypto.scatter_(x, 0, index, 2.0)
    assert res.base() is x.base()
    assert x.shape == [B, S]
    assert x.dtype == pypto.DT_FP32


def test_scatter_inplace_int_scalar_src():
    x, index, _ = make_case(input_dtype=pypto.DT_INT32)
    with pypto.function("SCATTER_INPLACE_INT_SCALAR", x, index):
        pypto.set_vec_tile_shapes(4, 16)
        res = pypto.scatter_(x, 0, index, 3)
    assert res.base() is x.base()
    assert x.dtype == pypto.DT_INT32


def test_scatter_inplace_tensor_src():
    x, index, src = make_case()
    with pypto.function("SCATTER_INPLACE_TENSOR", x, index, src):
        pypto.set_vec_tile_shapes(4, 16)
        res = pypto.scatter_(x, 0, index, src)
    assert res.base() is x.base()
    assert x.shape == [B, S]


def test_scatter_out_of_place_returns_new_base():
    x, index, _ = make_case()
    with pypto.function("SCATTER_OUT_OF_PLACE", x, index):
        pypto.set_vec_tile_shapes(4, 16)
        res = pypto.scatter(x, 0, index, 2.0)
    assert isinstance(res, pypto.tensor)
    assert res.base() is not x.base()
    assert res.shape == [B, S]
    assert res.dtype == pypto.DT_FP32
