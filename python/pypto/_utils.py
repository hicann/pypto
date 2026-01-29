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
import sys
import math
import functools
from typing import Sequence, Union, List

from . import pypto_impl
from .enum import DataType
from .symbolic_scalar import SymbolicScalar, SymInt


def to_sym(value) -> pypto_impl.SymbolicScalar:
    if isinstance(value, int):
        return pypto_impl.SymbolicScalar(value)
    if isinstance(value, pypto_impl.SymbolicScalar):
        return value
    if isinstance(value, SymbolicScalar):
        return value.base()
    raise ValueError("Invalid value type")


def to_syms(value: Union[Sequence[int], Sequence[SymbolicScalar]]) -> List[pypto_impl.SymbolicScalar]:
    return [to_sym(v) for v in value]


def ceildiv(cur_s: SymInt, block_size: SymInt) -> SymInt:
    result = math.ceil(float(cur_s) / float(block_size))
    return SymInt(result) 


# only outer takes effect void avoid tensor.py hide source_location of user code
_source_location_depth = 0


def set_source_location(level: int = 1):
    global _source_location_depth
    if _source_location_depth == 0:
        frame = sys._getframe(level + 1)
        pypto_impl.SetLocation(frame.f_code.co_filename, frame.f_lineno, "")
    _source_location_depth += 1


def clear_source_location():
    global _source_location_depth
    _source_location_depth -= 1
    if _source_location_depth == 0:
        pypto_impl.ClearLocation()


def source_location(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        set_source_location()
        out = func(*args, **kwargs)
        clear_source_location()
        return out
    return wrapper


def bytes_of(dtype: DataType) -> int:
    """ return the number of bytes of the current datatype

    Parameters
    ----------
    dtype: pypto.DataType
        datatype to be determined the number of bytes

    Returns
    -------
    int: the size of bytes the datatype contains

    Examples
    --------
    >>> print(pypto.bytes_of(pypto.DT_FP32))
        4
    """
    # implementation
    return pypto_impl.BytesOf(dtype)
