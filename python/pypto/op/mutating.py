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
"""PyPTO"""
from typing import Optional, Union, List

from .. import pypto_impl
from ..enum import CastMode, DataType
from .._op_wrapper import op_wrapper
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


@op_wrapper
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Returns a tensor that is a transposed version of `input`. The given dimensions `dim0` and `dim1` are swapped.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.

    Returns
    -------
    Tensor
        A new tensor that is a transposed version of `input`.

    Raises
    ------
    RuntimeError
        If dim0 or dim1 is greater than or equal to the input dimension.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    out = pypto.transpose(x, 0, 1)

    Input x:    [[ 1.0028 -0.9893 0.5809],
                 [-0.1669 0.7299  0.4942]]
    Output out: [[ 1.0028 -0.1669],
                 [-0.9893 0.7299],
                 [ 0.5809 0.4942]]
    """
    return pypto_impl.Transpose(input, [dim0, dim1])


@op_wrapper
def cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE) -> Tensor:
    """Casting the operand to the specified type.

    Parameters
    ----------
    operand : Tensor
        The input tensor.
    dtype : DataType
        The desired type.

    Returns
    -------
    Tensor
        Return a tensor after type cast, if this is already of the correct type, no copy is performed and the
        original object is returned

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2], pypto.DT_FP32)
    y = pypto.cast(x, pypto.DT_FP16)

    Input  x: [2.0, 3.0] x.dtype: pypto.DT_FP32
    Output y: [2.0, 3.0] y.dtype: pypto.DT_FP16
    """
    if dtype == input.dtype and mode == CastMode.CAST_NONE:
        return input
    else:
        return pypto_impl.Cast(input, dtype, mode)


@op_wrapper
def expand_clone(
    input: Tensor,
    shape: List[int],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
) -> Tensor:
    """
    Broadcast the input tensor along the axis where it is uniquely to 1 to match shape.A deep copy will be performed,
    and a new tensor that actually occupies memory will be returned.

    Parameters
    ----------
    input : Tensor
        The input tensor will be broadcasted.
    shape : List[int]
        Target shape.
    valid_shape : List[int] | List[SymbolicScalar]]
        Keyword argument, used for dynamic graph, represent the actual shapes at runtime.
        They can be ommitted in static graph.
    
    Examples
    --------
    x = pypto.tensor([1,3], pypto.DT_INT32)
    y = pypto.expand_clone(x, [3,4])

    Input  x: [[1], [2], [3]]
    Output y: [[ 1,  1,  1,  1],
               [ 2,  2,  2,  2],
               [ 3,  3,  3,  3]]
    
    """
    if valid_shape is None:
        valid_shape = []
    return pypto_impl.Expand(input, shape, valid_shape)
