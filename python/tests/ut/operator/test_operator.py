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
from test_base import BaseTest

dtype = pypto.DT_FP16
shape = (64, 64)
tiles = (32, 32)


class TestOperator(BaseTest):

    def test_sin(self):
        A = pypto.tensor(shape, dtype, 'A')
        C = pypto.tensor(shape, dtype, 'C')
        with pypto.function("sign", A, C):
            pypto.set_vec_tile_shapes(*tiles)
            C[:] = pypto.sin(A)

    def test_cos(self):
        A = pypto.tensor(shape, dtype, 'A')
        C = pypto.tensor(shape, dtype, 'C')
        with pypto.function("cos", A, C):
            pypto.set_vec_tile_shapes(*tiles)
            C[:] = pypto.cos(A)

    def test_sigmoid(self):
        A = pypto.tensor(shape, dtype, 'A')
        C = pypto.tensor(shape, dtype, 'C')
        with pypto.function("sigmoid", A, C):
            pypto.set_vec_tile_shapes(*tiles)
            C[:] = pypto.sigmoid(A)

    def test_softmax(self):
        A = pypto.tensor(shape, dtype, 'A')
        C = pypto.tensor(shape, dtype, 'C')
        with pypto.function("softmax", A, C):
            pypto.set_vec_tile_shapes(*tiles)
            C[:] = pypto.softmax(A, dim=-1)
