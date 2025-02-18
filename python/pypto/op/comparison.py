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
from typing import Optional, Tuple, Union
from .._element import Element
from .. import pypto_impl
from ..enum import OpType, OutType
from .._op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def greater(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.greater(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False False True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.GT, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.GT, OutType.BOOL)


@op_wrapper
def gt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.gt(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False False True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.GT, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.GT, OutType.BOOL)


@op_wrapper
def ge(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is greater_equal than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.ge(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False True True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.GE, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.GE, OutType.BOOL)


@op_wrapper
def eq(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is equal other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.eq(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [False True False]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.EQ, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.EQ, OutType.BOOL)


@op_wrapper
def ne(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is not equal other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.ne(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [True False True]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.NE, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.NE, OutType.BOOL)


@op_wrapper
def lt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is less than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.lt(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [True False False]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.LT, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.LT, OutType.BOOL)


@op_wrapper
def le(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor:
    """Performs element-wise comparison between `input` and `other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor or a scalar value for comparison.

    Returns
    -------
    Tensor
        A boolean tensor that is True where input is less equal than other and False elsewhere.
        BOOL tensor with same shape as inputs

    Raises
    ------
    TypeError
        If `other` is neither a Tensor nor a scalar number.


    Examples
    --------
    a = pypto.tensor([3], pypto.DT_FP32)
    b = pypto.tensor([3], pypto.DT_FP32)
    out = pypto.le(a, b)

    Input a:    [1.0 2.0 3.0]
    Input b:    [2.0 2.0 2.0]
    Output out: [True True False]

    """
    if isinstance(other, float):
        # Tensor vs Scalar comparison
        return pypto_impl.Compare(input, pypto_impl.Element(input.dtype, other), OpType.LE, OutType.BOOL)
    return pypto_impl.Compare(input, other, OpType.LE, OutType.BOOL)


@op_wrapper
def topk(
    input: Tensor, k: int, dim: Optional[int] = None, largest: bool = True
) -> Tuple[Tensor, Tensor]:
    """Returns the k largest elements of the given input tensor along a given dimension.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    k : int
        The k in "top-k".
    dim : int, optional
        The dimension to sort along, if dim is not given, the last dimension of the input is chosen.
    largest : bool
        Controling whether to return the elements in sorted order, if largest is False then the k smallest
        elements are returned.

    Returns
    -------
    tuple
        A tuple of (values, indices) is returned with the values and indices of the largest k elements of
        each row of the input tensor in the given dimension dim.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.topk(x, 2, -1, True)

    Input x:     [[1.0 2.0 3.0],
                  [1.0 2.0 3.0]]
    Output y[0]: [[3.0 2.0],
                  [3.0 2.0]]
    Output y[1]: [[2 1],
                  [2 1]]
    """

    return pypto_impl.TopK(input, k, (-1 if dim is None else dim), largest)
