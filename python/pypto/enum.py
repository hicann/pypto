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

from . import pypto_impl


def _enum_repr(self):
    return f"{self.__class__.__name__}.{self.name}"  # remove pypto_impl. prefix


DataType = pypto_impl.DataType
TileOpFormat = pypto_impl.TileOpFormat
CachePolicy = pypto_impl.CachePolicy
ReduceMode = pypto_impl.ReduceMode
CastMode = pypto_impl.CastMode
OpType = pypto_impl.OpType
OutType = pypto_impl.OutType
ReLuType = pypto_impl.ReLuType
ScatterMode = pypto_impl.ScatterMode

DataType.__repr__ = _enum_repr
TileOpFormat.__repr__ = _enum_repr
CachePolicy.__repr__ = _enum_repr
ReduceMode.__repr__ = _enum_repr
CastMode.__repr__ = _enum_repr
OpType.__repr__ = _enum_repr
OutType.__repr__ = _enum_repr

DT_INT4 = pypto_impl.DataType.DT_INT4
DT_INT8 = pypto_impl.DataType.DT_INT8
DT_INT16 = pypto_impl.DataType.DT_INT16
DT_INT32 = pypto_impl.DataType.DT_INT32
DT_INT64 = pypto_impl.DataType.DT_INT64
DT_FP8 = pypto_impl.DataType.DT_FP8
DT_FP16 = pypto_impl.DataType.DT_FP16
DT_FP32 = pypto_impl.DataType.DT_FP32
DT_BF16 = pypto_impl.DataType.DT_BF16
DT_HF4 = pypto_impl.DataType.DT_HF4
DT_HF8 = pypto_impl.DataType.DT_HF8
DT_FP8E4M3 = pypto_impl.DataType.DT_FP8E4M3
DT_FP8E5M2 = pypto_impl.DataType.DT_FP8E5M2
DT_FP8E8M0 = pypto_impl.DataType.DT_FP8E8M0
DT_UINT8 = pypto_impl.DataType.DT_UINT8
DT_UINT16 = pypto_impl.DataType.DT_UINT16
DT_UINT32 = pypto_impl.DataType.DT_UINT32
DT_UINT64 = pypto_impl.DataType.DT_UINT64
DT_BOOL = pypto_impl.DataType.DT_BOOL
DT_DOUBLE = pypto_impl.DataType.DT_DOUBLE
DT_BOTTOM = pypto_impl.DataType.DT_BOTTOM
