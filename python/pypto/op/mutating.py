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
from ..enum import CastMode, DataType, SaturationMode
from .._op_wrapper import op_wrapper
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


@op_wrapper
def permute(input_tensor: Tensor, dims: List[int]) -> Tensor:
    """Returns a tensor that is a permuted version of `input`. The dimensions are reordered according to `dims`.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor.
    dims : List[int]
        The desired ordering of dimensions. Must be a permutation of all dimensions indices (0 to input.dim() - 1).

    Returns
    -------
    Tensor
        A new tensor with the same data as `input` but with dimensions reordered as specified.

    Raises
    ------
    RuntimeError
        If `dims` is not a valid permutation of input dimensions.
        If the length of `dims` does not match the input tensor's dimension.

    Examples
    --------
    x = pypto.tensor([2, 3, 4], pypto.DT_FP32)
    out = pypto.permute(x, [2, 0, 1])

    Input x:    shape [2, 3, 4]
    Output out: shape [4, 2, 3]  # data permuted accordingly

    # For 2D transpose (equivalent to transpose(x, 0, 1)):
    y = pypto.permute(x, [1, 0])
    """
    return pypto_impl.Permute(input_tensor, dims)


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
def transdata(input: Tensor, format: int, group: int = 1) -> Tensor:
    """Perform tensor data format conversion.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    format : int
        The desired tensor format.
    group : int
        The number of groups

    Returns
    -------
    Tensor
        A new tensor with the specified format.

    Examples
    --------
    group = 1
    x = pypto.tensor([1, 16, 1, 8], pypto.DT_INT32)
    out = pypto.transdata(x, format, group)

    Input x:    [[[[  0,   1,   2,   3,   4,   5,   6,   7],
                 [  8,   9,  10,  11,  12,  13,  14,  15]],

                [[ 16,  17,  18,  19,  20,  21,  22,  23],
                 [ 24,  25,  26,  27,  28,  29,  30,  31]],

                [[ 32,  33,  34,  35,  36,  37,  38,  39],
                 [ 40,  41,  42,  43,  44,  45,  46,  47]],

                [[ 48,  49,  50,  51,  52,  53,  54,  55],
                 [ 56,  57,  58,  59,  60,  61,  62,  63]],

                [[ 64,  65,  66,  67,  68,  69,  70,  71],
                 [ 72,  73,  74,  75,  76,  77,  78,  79]],

                [[ 80,  81,  82,  83,  84,  85,  86,  87],
                 [ 88,  89,  90,  91,  92,  93,  94,  95]],

                [[ 96,  97,  98,  99, 100, 101, 102, 103],
                 [104, 105, 106, 107, 108, 109, 110, 111]],

                [[112, 113, 114, 115, 116, 117, 118, 119],
                 [120, 121, 122, 123, 124, 125, 126, 127]]]]
    Output out: [[[[[  0,  64],
                  [  1,  65],
                  [  2,  66],
                  [  3,  67],
                  [  4,  68],
                  [  5,  69],
                  [  6,  70],
                  [  7,  71]],

                 [[  8,  72],
                  [  9,  73],
                  [ 10,  74],
                  [ 11,  75],
                  [ 12,  76],
                  [ 13,  77],
                  [ 14,  78],
                  [ 15,  79]],

                 [[ 16,  80],
                  [ 17,  81],
                  [ 18,  82],
                  [ 19,  83],
                  [ 20,  84],
                  [ 21,  85],
                  [ 22,  86],
                  [ 23,  87]],

                 [[ 24,  88],
                  [ 25,  89],
                  [ 26,  90],
                  [ 27,  91],
                  [ 28,  92],
                  [ 29,  93],
                  [ 30,  94],
                  [ 31,  95]],

                 [[ 32,  96],
                  [ 33,  97],
                  [ 34,  98],
                  [ 35,  99],
                  [ 36, 100],
                  [ 37, 101],
                  [ 38, 102],
                  [ 39, 103]],

                 [[ 40, 104],
                  [ 41, 105],
                  [ 42, 106],
                  [ 43, 107],
                  [ 44, 108],
                  [ 45, 109],
                  [ 46, 110],
                  [ 47, 111]],

                 [[ 48, 112],
                  [ 49, 113],
                  [ 50, 114],
                  [ 51, 115],
                  [ 52, 116],
                  [ 53, 117],
                  [ 54, 118],
                  [ 55, 119]],

                 [[ 56, 120],
                  [ 57, 121],
                  [ 58, 122],
                  [ 59, 123],
                  [ 60, 124],
                  [ 61, 125],
                  [ 62, 126],
                  [ 63, 127]]]]]
    """
    return pypto_impl.TransData(input, format, group)


@op_wrapper
def cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE,
         satmode: SaturationMode = SaturationMode.OFF) -> Tensor:
    """Casting the operand to the specified type.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dtype : DataType
        The desired type.
    mode : CastMode, optional
        The rounding mode for the cast operation. Default is CAST_NONE.
    satmode : SaturationMode, optional
        The saturation mode for float to integer conversions.
        Default is OFF (truncation behavior). Use ON for saturation (clamping).

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

    # With saturation mode
    x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
    y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
    # Values will be clamped to [-128, 127] range

    Input  x: [2.0, 3.0] x.dtype: pypto.DT_FP32
    Output y: [2.0, 3.0] y.dtype: pypto.DT_FP16
    """
    if dtype == input.dtype and mode == CastMode.CAST_NONE and satmode == SaturationMode.OFF:
        return input
    else:
        return pypto_impl.Cast(input, dtype, mode, satmode)


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
