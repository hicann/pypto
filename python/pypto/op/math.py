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
from ..enum import DataType
from ..symbolic_scalar import SymbolicScalar, SymInt


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
def fmod(input: Tensor, other: Union[Tensor, float]) -> Tensor:
    """Computes the element-wise modulus of `input` and `other`.

    This function calculates the formula: `out = input % other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to modulo operation.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise modulus.

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
    out = pypto.fmod(a, b)

    Input a:    [[2.0 5.0 9.0]]
    Input b:    [[2.0 2.0 2.0]]
    Output out: [[0.0 1.0 1.0]]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Fmod(input, other)
    else:
        return pypto_impl.Fmod(input, pypto_impl.Element(input.dtype, other))


@op_wrapper
def bitwise_and(self: Tensor, other: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise AND of `self` and `other`.

    This function calculates the formula: `out = self & other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    self : Tensor
        The first input tensor.
    other : Tensor or int
        The second input tensor or an integer scalar.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise bitwise AND result.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    a = pypto.tensor([0x1234, 0x5678], pypto.DT_INT16)
    b = pypto.tensor([0x0F0F, 0xF0F0], pypto.DT_INT16)
    out = pypto.bitwise_and(a, b)

    Input a:    [5, 3]
    Input b:    [3, 1]
    Output out: [1, 1]
    """
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.BitwiseAnd(self, other)
    else:
        if not isinstance(other, int):
            raise TypeError(f"Scalar operand for bitwise_and must be an integer, but got {type(other)}.")
        return pypto_impl.BitwiseAnd(self, pypto_impl.Element(self.dtype, other))


@op_wrapper
def bitwise_or(input1: Tensor, input2: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise OR of `input1` and `input2`.

    This function calculates the formula: `out = input1 | input2`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input1 : Tensor
        The first input tensor.
    input2 : Tensor or int
        The second input tensor or an integer scalar.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise bitwise OR result.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    a = pypto.tensor([5, 3], pypto.DT_INT16)
    b = pypto.tensor([3, 1], pypto.DT_INT16)
    out = pypto.bitwise_or(a, b)

    Input a:    [5, 3]
    Input b:    [3, 1]
    Output out: [7, 3]
    """
    if isinstance(input2, pypto_impl.Tensor):
        return pypto_impl.BitwiseOr(input1, input2)
    else:
        if not isinstance(input2, int):
            raise TypeError(f"Scalar operand for bitwise_or must be an integer, but got {type(input2)}.")
        return pypto_impl.BitwiseOr(input1, pypto_impl.Element(input1.dtype, input2))


@op_wrapper
def bitwise_xor(first: Tensor, second: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise XOR of `first` and `second`.

    This function calculates the formula: `out = first ^ second`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    first : Tensor
        The first input tensor.
    second : Tensor or int
        The second input tensor or an integer scalar.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise bitwise XOR result.

    Raises
    ------
    RuntimeError
        If the two tensors are not broadcastable to a common shape.

    Examples
    --------
    a = pypto.tensor([5, 3], pypto.DT_INT16)
    b = pypto.tensor([3, 1], pypto.DT_INT16)
    out = pypto.bitwise_xor(a, b)

    Input a:    [5, 3]   # binary: [101, 011]
    Input b:    [3, 1]   # binary: [011, 001]
    Output out: [6, 2]   # binary: [110, 010]
    """
    if isinstance(second, pypto_impl.Tensor):
        return pypto_impl.BitwiseXor(first, second)
    else:
        if not isinstance(second, int):
            raise TypeError(f"Scalar operand for bitwise_xor must be an integer, but got {type(second)}.")
        return pypto_impl.BitwiseXor(first, pypto_impl.Element(first.dtype, second))


@op_wrapper
def pow(input: Tensor, other: Union[Tensor, int, float]) -> Tensor:
    """Computes the element-wise power of `input` raised to `other`.

    This function calculates the formula: `out = input ** other`.

    Parameters
    ----------
    input : Tensor
        The base input tensor.
    other : Tensor or Number
        The exponent to which each element in `input` will be raised.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise power operation results.

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    a = 2
    b = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.pow(x, a)
    z = pypto.pow(x, b)

    Input x:[[ 1.0 2.0],
             [-3.0 4.0]]
          b:[[2.0 2.0],
             [1.0 1.0]]
    Output y:[[1.0  4.0],
              [9.0 16.0]]
           z:[[ 1.0 4.0],
              [-3.0 4.0]]
    """
    if not isinstance(other, (pypto_impl.Tensor, int, float)):
        raise TypeError(f"other must be Tensor, int or float but got {type(other)}.")
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Pow(input, other)
    if isinstance(other, int):
        return pypto_impl.Pow(input, pypto_impl.Element(DataType.DT_INT32, other))
    return pypto_impl.Pow(input, pypto_impl.Element(DataType.DT_FP32, other))


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
def sign(a: Tensor) -> Tensor:
    """Computes the element-wise exponential of `input`.

    This function return a tensor with the signs of the elements of input.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise exponential.

    Examples
    --------
    x = pypto.tensor([3], pypto.DT_FP32)
    y = pypto.sign(x)

    Input x: [-5.0    0.0    5.0    10.0]
    Output y:[-1.    0.    1.    1.]
    """
    return pypto_impl.Sign(a)


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
def relu(a: Tensor) -> Tensor:
    """
    Returns a new tensor with the rectified linear unit function applied element-wise.
    
    The function is defined as:
    y = max(0, x)
    
    Parameters
    ----------
    input : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise relu result.

    Examples
    --------
    x = pypto.tensor([-1.0, 2.0, 0.0, -0.5], pypto.DT_FP32)
    y = pypto.relu(x)

    Input x:  [-1.0, 2.0, 0.0, -0.5]
    Output y: [ 0.0, 2.0, 0.0,  0.0]
    """
    return pypto_impl.Relu(a)


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
def round(input: Tensor, decimals: int = 0) -> Tensor:
    """Rounds elements of `input` to the nearest number of decimal places.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    decimals : int
        Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions to the left of the decimal point.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise round.

    Examples
    --------
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.round(x, decimals=1)

    Input x: [[1.21, 2.35], [3.65, 4.76]]
    Output y: [[1.2, 2.4], [3.6, 4.8]]
    """
    return pypto_impl.Round(input, decimals)


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
def ceil(input: Tensor) -> Tensor:
    """Computes the element-wise ceiling of `input` (upward rounding to the nearest integer)

    This function calculates the formula: `out = ceil(input)`.
    The ceiling of a number is the smallest integer greater than or equal to the number.

    Parameters
    ----------
    input : Tensor
        The input tensor containing numerical values to be ceiling-rounded.

    Returns
    -------
    Tensor
        A new tensor with the ceiling value of each element of the input tensor.

    Raises
    ------
    TODO

    See Also
    --------
    ceil : ceil rounding (upward to the nearest integer)

    Examples
    --------
    x = pypto.tensor([2.1, -2.1, 5.0, 3.9], pypto.DT_FP32)
    y = pypto.ceil(x)

    Input x: [[1.2  4.7],
              [-1.1  9.0]]
    Output y:[[2.0  5.0],
              [-1.0  9.0]]
    """
    return pypto_impl.Ceil(input)


@op_wrapper
def floor(input: Tensor) -> Tensor:
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
    floor : floor rounding (downward to the nearest integer)

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.floor(x)

    Input x:  [1.2 4.2 9.8 6.9 25.5]
    Output y: [1.0 4.0 9.0 6.0  25.0]
    """
    return pypto_impl.Floor(input)


@op_wrapper
def trunc(input: Tensor) -> Tensor:
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
    trunc : trunc rounding (towards zero to the nearest integer)

    Examples
    --------
    x = pypto.tensor([5], pypto.DT_FP32)
    y = pypto.trunc(x)

    Input x:  [1.3 4.2 9.8 16.4 25.8]
    Output y: [1.0 4.0 10.0 16.0 26.0]
    """
    return pypto_impl.Trunc(input)


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
    if min is None:
        min = default
    if max is None:
        max = default

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
    dim = 0
    out = pypto.cumsum(x, dim)
    Input  x : [[0 1 2],
                [3 4 5]]
    Output out:[[0 1 2],
                [3 5 7]]
    """
    return pypto_impl.cumsum(input, dim)


@op_wrapper
def bitwise_right_shift(
    input: Union[Tensor, int], other: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise right shift of `input` and `other`.

    This function calculates the formula: `out = input >> other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to bitwise right shift.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise right shift.

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
    a = pypto.tensor([1, 3], pypto.DT_INT16)
    b = pypto.tensor([1, 3], pypto.DT_INT16)
    out = pypto.bitwise_right_shift(a, b)

    Input a:    [[1 2 3]]
    Input b:    [[1 1 1]]
    Output out: [[0 1 1]]
    """
    if isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        return pypto_impl.BitwiseRightShift(input, other)
    elif isinstance(input, pypto_impl.Tensor):
        return pypto_impl.BitwiseRightShift(input, pypto_impl.Element(input.dtype, other))
    else:
        return pypto_impl.BitwiseRightShift(pypto_impl.Element(other.dtype, input), other)


@op_wrapper
def bitwise_left_shift(
    input: Union[Tensor, int], other: Union[Tensor, int]) -> Tensor:
    """Computes the element-wise bitwise left shift of `input` and `other`.

    This function calculates the formula: `out = input << other`.
    It supports broadcasting between the input tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor or Number
        The second input tensor or a scalar to bitwise left shift.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise left shift.

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
    a = pypto.tensor([1, 3], pypto.DT_INT16)
    b = pypto.tensor([1, 3], pypto.DT_INT16)
    out = pypto.bitwise_left_shift(a, b)

    Input a:    [[1 2 3]]
    Input b:    [[1 1 1]]
    Output out: [[2 4 6]]
    """
    if isinstance(input, pypto_impl.Tensor) and isinstance(other, pypto_impl.Tensor):
        return pypto_impl.BitwiseLeftShift(input, other)
    elif isinstance(input, pypto_impl.Tensor):
        return pypto_impl.BitwiseLeftShift(input, pypto_impl.Element(input.dtype, other))
    else:
        return pypto_impl.BitwiseLeftShift(pypto_impl.Element(other.dtype, input), other)


@op_wrapper
def bitwise_not(self: Tensor) -> Tensor:
    """
    Computes the element-wise bitwise NOT of 'self'

    This function calculates the formula: 'out = ~self'.
    For each element in the self tensor, performs a bitwise NOT operation.

    Parameters
    ----------
    self : Tensor
        The input tensor (should be of integer type)

    Returns
    -------
    Tensor
        A tensor with the same shape and dtype as input

    Examples
    --------
    a = pypto.tensor([0, 1, 2, 3, 4], pypto.DT_INT32)
    out = pypto.bitwise_not(a)

    Self a:    [0 1 2 3 4]  (in binary: [000, 001, 010, 011, 100])
    Output out: [-1 -2 -3 -4 -5]  (in binary: [111, 110, 101, 100, 011])

    """
    return pypto_impl.BitwiseNot(self)


@op_wrapper
def triu(
    input: Tensor,
    diagonal: SymInt = 0
) -> Tensor:
    """
    Return the upper traingular part of a matrix or a banch of matrices `input`, the other elements of 
    the result are set to 0.
    Parameters
    ---------
    input: Tensor
        The tensor to be calculated.
    diagonal : SymInt
        The diagonal to consider.
    out: Tensor
        The tensor after calculation.
    Examples
    ---------
    x = pypto.tensor([3, 3], pypto.data_type.DT_INT32) 
    diagonal = 0
    out = pypto.triu(x, diagonal)
    Input  x : [[1 2 3],
                [4 5 6],
                [7 8 9]]
    Output out:[[1 2 3],
                [0 5 6],
                [0 0 9]]
    """
    if isinstance(diagonal, int):
        diagonal = SymbolicScalar(diagonal).base()
    return pypto_impl.TriU(input, diagonal)


@op_wrapper
def tril(
    input: Tensor,
    diagonal: SymInt = 0
) -> Tensor:
    """
    Return the lower traingular part of a matrix or a banch of matrices `input`, the other elements of
    the result are set to 0.
    Parameters
    ---------
    input: Tensor
        The tensor to be calculated.
    diagonal : SymInt
        The diagonal to consider.
    out: Tensor
        The tensor after calculation.
    Examples
    ---------
    x = pypto.tensor([3, 3], pypto.data_type.DT_INT32) 
    diagonal = 0
    out = pypto.tril(x, diagonal)
    Input  x : [[1 2 3],
                [4 5 6],
                [7 8 9]]
    Output out:[[1 0 0],
                [4 5 0],
                [7 8 9]]
    """
    if isinstance(diagonal, int):
        diagonal = SymbolicScalar(diagonal).base()
    return pypto_impl.TriL(input, diagonal)


@op_wrapper
def copysign(input: Tensor, other: Tensor) -> Tensor:
    """
    Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
    Parameters
    ---------
    input: Tensor
        The tensor of magnitudes.
    other : Tensor
         The tensor that contains value(s) whose signbit(s) are applied to the magnitudes in input.
    out: Tensor
        The output tensor.
    Examples
    ---------
    x = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
    y = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
    out = pypto.copysign(x, y)
    Input  x : [[1 -2  3],
                [4  5 -6],
                [-7 8  9]]
    Input  y : [[-1 6 -8],
                [1 -1  0],
                [7 -8  9]]
    Output out:[[-1 2 -3],
                [4 -5  6],
                [7 -8  9]]
    """
    return pypto_impl.CopySign(input, other)


@op_wrapper
def cbrt(self: Tensor) -> Tensor:
    """
    Computes the element-wise cube root of 'self'

    This function calculates the formula: 'out = self^(1/3)'.
    For each element in the self tensor, performs a cube root operation.

    Parameters
    ----------
    self : Tensor
        The input tensor

    Returns
    -------
    Tensor
        A tensor with the same shape and dtype as input

    Examples
    --------
    x = pypto.tensor([1, 2], pypto.DT_FP32)
    out = pypto.cbrt(x)

    Input  x:[[8, -8]]
    Output y:[[2, -2]]
    """
    return copysign(pow(abs(self), 1.0 / 3.0), self)
