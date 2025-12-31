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
from typing import Optional, Union

from .. import pypto_impl
from .._element import Element
from .._op_wrapper import op_wrapper
from ..tensor import Tensor


@op_wrapper
def add(
    input: Tensor, other: Union[Tensor, float], *, alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise addition of `input` and `other`.

    This function calculates the formula: `out = input + alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be added.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise sum.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_FP32)
    b = pypto.tensor([1, 3], pypto.DT_FP32)
    out = pypto.add(a, b)

    Input a:    [[1.0 2.0 3.0]]
    Input b:    [[2.0 3.0 4.0]]
    Output out: [[3.0 5.0 7.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Add(input, other)
        else:
            return pypto_impl.Add(
                input, pypto_impl.Mul(other, pypto_impl.Element(input.dtype, alpha))
            )
    else:
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Add(input, pypto_impl.Element(input.dtype, other))
        else:
            if not isinstance(other, (int, float)):
                raise TypeError(f"alpha must be int or float, but got {type(other)}.")
            return pypto_impl.Add(input, pypto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def sub(
    input: Tensor, other: Union[Tensor, float], *, alpha: Union[int, float] = 1
) -> Tensor:
    """Computes the element-wise subtraction of `input` and `other`.

    This function calculates the formula: `out = input - alpha * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be subtracted.
    alpha : float, optional, keyword-only
        A scaling factor for the `other` input. Default is 1.0.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise subtraction.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 3], pypto.DT_FP32)
    out1 = pypto.sub(a, b)

    Input x:      [[9.0 9.0 9.0],
                   [9.0 9.0 9.0]]
    Input y:      [[1.0 2.0 3.0],
                   [1.0 2.0 3.0]]
    Output out1 : [[8.0 7.0 6.0],
                   [8.0 7.0 6.0]]

    # Using a scalar and alpha
    c = pypto.sub(x, 2.0, alpha=3) # Computes x - 2 * 3

    Output c:[[3.0 3.0 3.0],
              [3.0 3.0 3.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Sub(input, other)
        else:
            return pypto_impl.Sub(
                input, pypto_impl.Mul(other, pypto_impl.Element(input.dtype, alpha))
            )
    else:
        if alpha == 1 or alpha == 1.0:
            return pypto_impl.Sub(input, pypto_impl.Element(input.dtype, other))
        else:
            if not isinstance(other, (int, float)):
                raise TypeError(f"alpha must be int or float, but got {type(other)}.")
            return pypto_impl.Sub(input, pypto_impl.Element(input.dtype, other * alpha))


@op_wrapper
def mul(input: Tensor, other: Union[Tensor, float]) -> Tensor:
    """Computes the element-wise multiplication of `input` and `other`.

    This function calculates the formula: `out = input * other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to be multiplied.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise multiplication.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 3], pypto.DT_FP32)
    z = pypto.mul(a, b)

    Input x:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Input y:[[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
    Output z:[[1.0 4.0 9.0],
              [1.0 4.0 9.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Mul(input, other)
    else:
        return pypto_impl.Mul(input, pypto_impl.Element(input.dtype, other))


@op_wrapper
def div(input: Tensor, other: Union[Tensor, float]) -> Tensor:
    """Computes the element-wise division of `input` and `other`.

    This function calculates the formula: `out = input / other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to divide.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise division.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    See Also
    --------
    sub : The inverse operation, element-wise subtraction.
    mul : Element-wise multiplication.

    Examples
    --------
    a = pypto.tensor([1, 3], pypto.DT_FP32)
    b = pypto.tensor([1, 3], pypto.DT_FP32)
    out = pypto.div(a, b)

    Input a:    [[2.0 4.0 6.0]]
    Input b:    [[2.0 2.0 2.0]]
    Output out: [[1.0 2.0 3.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Div(input, other)
    else:
        return pypto_impl.Div(input, pypto_impl.Element(input.dtype, other))


@op_wrapper
def pow(input: Tensor, other: Union[int, float]) -> Tensor:
    """Computes the element-wise power of `input` raised to `other`.

    This function calculates the formula: `out = input ** other`.

    Parameters
    ----------
    input : Tensor
        The base input tensor.
    other : Number
        The exponent to which each element in `input` will be raised.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise power operation results.

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    a = 2
    y = pypto.pow(x, a)

    Input x:[[1.0 2.0],
             [3.0 4.0]]
    Output y:[[1.0  4.0],
              [9.0 16.0]]
    """
    if not isinstance(other, (int, float)):
        raise TypeError(f"other must be int or float, but got {type(other)}.")
    return pypto_impl.Pow(input, pypto_impl.Element(input.dtype, other))


@op_wrapper
def exp(input: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function calculates the formula: `out = e ** input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.exp(x)

    Input x: [0.0    1.0    2.0]
    Output y:[1.0000 2.7183 7.3891]
    """
    return pypto_impl.Exp(input)


@op_wrapper
def abs(a: Tensor) -> Tensor:
    """
    Computes the absolute value of each element in input.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise absolute.

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_INT32)
    y = pypto.abs(x)

    Input x:  [-1, -2, 3]
    Output y: [ 1,  2, 3]
    """
    return pypto_impl.Abs(a)


@op_wrapper
def reciprocal(a: Tensor) -> Tensor:
    """
    Returns a new tensor with the reciprocal of the elements of input
    
    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise reciprocal.

    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.reciprocal(x)

    Input x:  [-0.4595, -2.1219, -1.4314,  0.7298]
    Output y: [-2.1763, -0.4713, -0.6986,  1.3702]
    """
    return pypto_impl.Reciprocal(a)


@op_wrapper
def logical_not(input: Tensor) -> Tensor:
    """
    Computes the element-wise logical NOT of 'input'

    This function calculates the formula: 'out = input == 0? True : False'.

    Parameters
    ----------
    input : Tensor
        The input tensor

    Returns
    -------
    Tensor
        A tensor of bool with the same shape as input

    Examples
    --------
    a = pypto.tensor([5], pypto.DT_INT32)
    out = pypto.logical_not(a)

    Input a:    [0 1 2 3 4]
    Output out: [True False False False False]

    """
    return pypto_impl.LogicalNot(input)


@op_wrapper
def logical_and(input: Tensor, other: Tensor) -> Tensor:
    """Computes the element-wise logical AND of `input` and `other`.

    This function calculates the formula: `out = input && other`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor. Should be broadcastable to the shape of `input`.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise logical AND operation results.

    Examples
    --------
    x = pypto.tensor([True, False], pypto.DT_BOOL)
    y = pypto.tensor([True, True], pypto.DT_BOOL)
    z = pypto.logical_and(x, y)

    Input x: [True, False]
    Input y: [True, True]
    Output z: [True, False]

    # 支持广播
    x = pypto.tensor([[True, False], [False, True]], pypto.DT_BOOL)
    y = pypto.tensor([True, False], pypto.DT_BOOL)
    z = pypto.logical_and(x, y)

    Input x:  [[True, False], [False, True]]
    Input y:  [True, False]
    Output z: [[True, False], [False, False]]
    """
    return pypto_impl.LogicalAnd(input, other)


@op_wrapper
def rsqrt(input: Tensor) -> Tensor:
    """Computes the element-wise reciprocal of the square-root of `input`

    This function calculates the formula: `out = 1 / sqrt(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor with the reciprocal of the square-root of each of the element of input.

    Raises
    ------
    TODO

    See Also
    --------
    sqrt : square-root of each of the element

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.rsqrt(x)

    Input x: [[1.0  4.0],
              [16.0 9.0]]
    Output y:[[1.0  0.5],
              [0.25 0.33333]]
    """
    return pypto_impl.Rsqrt(input)


@op_wrapper
def sqrt(input: Tensor) -> Tensor:
    """Computes the element-wise squareroot of `input`.

    This function calculates the formula: `out = √input`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise squareroot.

    See Also
    --------
    exp : Element-wise exponential function.

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.sqrt(x)

    Input x:  [1.0 4.0 9.0 16.0 25.0]
    Output y: [1.0 2.0 3.0 4.0  5.0]
    """
    return pypto_impl.Sqrt(input)


@op_wrapper
def neg(a: Tensor) -> Tensor:
    """
    Returns a new tensor with the negative of the elements of input.
    
    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise neg.

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.neg(x)

    Input x: [ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940]
    Output y:[-0.0090,  0.2262,  0.0682,  0.2866, -0.3940]
    """
    return pypto_impl.Neg(a)


@op_wrapper
def log(input: Tensor) -> Tensor:
    """Computes the element-wise log of `input`.

    This function calculates the formula: `out = log(input)`.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise log.

    See Also
    -------
    sqrt : Element-wise square-root

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.log(x)

    Input x: [1.0     2.0    3.0]
    Output y:[0.0000 0.6931 1.0986]
    """

    return pypto_impl.Log(input, pypto_impl.LogBaseType.LOG_E)


@op_wrapper
def clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None
):
    """
    Make the values in `input` greater than `min` and less than `max`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    min : Tensor or Element
        The minimum value.
    max: Tensor or Element
        The maximum value

    Returns
    -------
    Tensor
        A new tensor containing the values greater than `min` and less than `max`.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_INT32)
    min = 1
    max = 3
    out = pypto.clip(x, min, max)

    Input x:    [[0 2 4], [3, 4, 6]]
    Output out: [[1 2 3], [3, 3, 3]]
    """
    if min is None and max is None:
        return input

    element_types = (pypto_impl.Element, int, float)
    is_element_mode = isinstance(min, element_types) or isinstance(max, element_types)
    default = (
        pypto_impl.Tensor()
        if not is_element_mode
        else pypto_impl.Element(pypto_impl.DataType.DT_BOTTOM, 0)
    )
    min = min or default
    max = max or default

    if not isinstance(min, pypto_impl.Element) and isinstance(min, element_types):
        min = pypto_impl.Element(input.GetDataType(), min)

    if not isinstance(max, pypto_impl.Element) and isinstance(min, element_types):
        max = pypto_impl.Element(input.GetDataType(), max)

    return pypto_impl.Clip(input, min, max)


@op_wrapper
def cumsum(
    input: Tensor,
    dim: int
) -> Tensor:
    """
    This function returns the cumulative sum over a given axis.
    Parameters
    ---------
    input: Tensor
        tensor to be calculated.
    dim : int
        specified dimension.
    out: Tensor
        The tensor after calculating the cumulative sum.
    Examples
    ---------
    x = pypto.tensor([2, 3], pypto.data_type.DT_INT32) 
    y = pypto.tensor([2, 3], pypto.data_type.DT_INT32) 
    dim = 0
    out = pypto.cumsum(x, dim)
    Input  x : [[0 1 2],
                [3 4 5]]
    Output out:[[0 1 2],
                [3 5 7]]
    """
    return pypto_impl.cumsum(input, dim)