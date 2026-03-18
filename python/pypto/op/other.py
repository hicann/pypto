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
from typing import Union, Sequence, List

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..tensor import Tensor
from .._element import Element


@op_wrapper
def where(
    condition: Tensor, input: Union[Tensor, float, Element], other: Union[Tensor, float, Element]
) -> Tensor:
    """
    Return a tensor of elements selected from either `input` or `other`, depending on `condition`.

    This function implements element-wise selection:
    'out[i] = input[i] if condition[i] else other[i]'.
    It supports broadcasting among `condition`, `input`, and `other`.

    Parameters
    ----------
    condition : Tensor of bool
        A boolean tensor indicating which elements to select from `input` (True) or `other` (False).
    input : Tensor or Number
        A tensor or scalar value to be selected where `condition` is True.
    other : Tensor or Number
        A tensor or scalar value to be selected where `condition` is False.

    Returns
    -------
    Tensor
        A tensor with the same shape as the broadcasted `condition`, containing elements
        from `input` where `condition` is True, and from `other` otherwise.
        The data type is determined by type promotion rules between `input` and `other`.

    Raises
    ------
    RuntimeError
        If `condition`, `input`, and `other` cannot be broadcasted to a common shape.
    TypeError
        If `condition` is not a boolean tensor.

    See Also
    --------
    logical_not : Computes element-wise logical NOT.
    add : Element-wise addition with optional scaling.

    Examples
    --------
    cond = pypto.tensor([4], pypto.DT_BOOL)
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.tensor([4], pypto.DT_FP32)
    out1 = pypto.where(cond, x, y)

    Input cond:  [True False True False]
    Input x:     [1.0  2.0  3.0  4.0]
    Input y:     [10.0 20.0 30.0 40.0]
    Output out1: [1.0  20.0 3.0  40.0]

    # Using scalar inputs
    out2 = pypto.where(cond, 1.0, 0.0)

    Output out2: [1.0 0.0 1.0 0.0]

    # Broadcasting example
    cond = pypto.tensor([2, 2], pypto.DT_BOOL)
    x = pypto.tensor([1, 2], pypto.DT_FP32)  # Will be broadcasted
    y = 0.0
    out3 = pypto.where(cond, x, y)

    Input cond:  [[True False], [False True]]
    Input x:     [1.0 2.0]
    Input y:     0.0

    Output out3: [[1.0 0.0],
                  [0.0 2.0]])
    """
    if isinstance(input, pypto_impl.Tensor) or isinstance(input, pypto_impl.Element):
        input_base = input
    else:
        input_base = pypto_impl.Element(pypto_impl.DT_FP32, input)

    if isinstance(other, pypto_impl.Tensor) or isinstance(other, pypto_impl.Element):
        other_base = other
    else:
        other_base = pypto_impl.Element(pypto_impl.DT_FP32, other)
    return pypto_impl.Where(condition, input_base, other_base)


@op_wrapper
def pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0) -> Tensor:
    """
    Pads the input tensor.

    Padding size:
    The padding size by which to pad some dimensions of input are described starting from
    the last dimension and moving forward.
    For pad has format (d_last_dim, d_last_dim-1, ..., d_last_dim-k).
    Current implementation supports padding the last 2 dimensions (Right and Bottom) with 
    constant values.

    Parameters
    ----------
    input : Tensor
        The input tensor to be padded.
    pad : tuple or list of int
        m-elements tuple, where m/2 <= input dimensions and m is even.
        Format is (pad_left, pad_right, pad_top, pad_bottom, ...).
        Note: Currently only supports pad_left=0 and pad_top=0 (Right/Bottom padding only).
    mode : str, optional
        'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        Note: Currently only 'constant' is supported.
    value : float, optional
        fill value for 'constant' padding. Default: 0.0
        Note: Currently only 'inf' '-inf' '0.0' is supported.

    Returns
    -------
    Tensor
        Padded tensor.

    Raises
    ------
    TypeError
        If input is not a Tensor.
        If pad is not a sequence of integers.
    ValueError
        If pad length is not even.

    Examples
    --------
    t4d = pypto.tensor([1, 1, 2, 2], pypto.DT_FP32)
    # Pad last dim by (0, 1) -> Right pad 1
    # Pad 2nd to last dim by (0, 1) -> Bottom pad 1
    p1 = (0, 1, 0, 1) 
    out = pypto.pad(t4d, p1, "constant", 0.0)

    Input t4d: [[[[0.0, 1.0],
                [2.0, 3.0]]]]
    
    Output out: [[[[0.0, 1.0, 0.0],
                [2.0, 3.0, 0.0],
                [0.0, 0.0, 0.0]]]]

    """

    pad_list = list(pad)
    return pypto_impl.Pad(input, pad_list, mode, float(value))


@op_wrapper
def fillpad(input: Tensor, mode: str = "constant", value: float = 0.0) -> Tensor:
    """
    Fills the padding region of the input tensor with a constant value.

    Unlike `pad`, this function does not change the tensor shape. It fills the
    padding region (the area beyond valid_shape) with the specified value.

    Parameters
    ----------
    input : Tensor
        The input tensor with padding region to be filled.
    mode : str, optional
        'constant'. Default: 'constant'
        Note: Currently only 'constant' is supported.
    value : float, optional
        fill value for 'constant' padding. Default: 0.0
        Note: Currently only 'inf' '-inf' '0.0' is supported.

    Returns
    -------
    Tensor
        Tensor with padding region filled.

    Raises
    ------
    TypeError
        If input is not a Tensor.

    Examples
    --------
    # Create a tensor with valid_shape smaller than actual shape
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    # Assume valid_shape is [2, 2], the rest [2:4, 2:4] is padding region
    out = pypto.fillpad(a, "constant", "-inf")

    Input a: [[1.0, 2.0, 0.0, 0.0],
              [3.0, 4.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0]]
    
    Output out: [[1.0, 2.0, -inf, -inf],
                 [3.0, 4.0, -inf, -inf],
                 [-inf, -inf, -inf, -inf],
                 [-inf, -inf, -inf, -inf]]

    """

    return pypto_impl.FillPad(input, mode, float(value))


@op_wrapper
def one_hot(input: Tensor, num_classes: int) -> Tensor:
    """
    Converts a tensor of indices to one-hot encoded tensor.

    Parameters
    ----------
    input : Tensor
        LongTensor containing class indices of any shape (*)
    num_classes : int
        Total number of classes.

    Returns
    -------
    Tensor
        One-hot encoded tensor(LongTensor) of shape (*, num_classes) where:
        - 1 is placed at the index specified by input value
        - 0 is placed everywhere else

    Examples
    --------
    a = pypto.tensor([3], pypto.DT_INT32)
    out = pypto.one_hot(a, 5)

    Input a:    [0 2 4]
    Input num_classes:  5
    Output out: [[1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1]]

    """
    if not isinstance(input, pypto_impl.Tensor):
        raise TypeError("input must be a `Tensor`")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an `int`")
    if num_classes == -1:
        raise RuntimeError("num_classes must be specified")
    if num_classes <= 0:
        raise RuntimeError("num_classes must be a positive integer")
    return pypto_impl.OneHot(input, num_classes)


@op_wrapper
def expand_exp_dif(input: Tensor, other: Tensor) -> Tensor:
    """Computes the exp dif of `input` and `other`.

    This function calculates the formula: `out = e ** (input - other)`.

    Parameters
    ----------
    input : Tensor
        The first input tensor.
    other : Tensor
        The second input tensor.

    Returns
    -------
    Tensor
        A new tensor containing the element-wise expand exp dif.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([1, 3], pypto.DT_FP32)
    z = pypto.expand_exp_dif(x, y)

    Input x:      [[1, 2, 3], [4, 5, 6]]
    Input y:      [[1, 2, 3]]
    Output z :    [[ 1.      ,  1.      ,  1.      ],
                   [20.085537, 20.085537, 20.085537]]

    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.tensor([2, 1], pypto.DT_FP32)
    z = pypto.expand_exp_dif(x, y)

    Input x:      [[1, 2, 3], [4, 5, 6]]
    Input x:      [[1], [2]]
    Output z :    [[ 1.       ,  2.718282 ,  7.3890557],
                   [ 7.3890557, 20.085537 , 54.59815  ]]
    """
    return pypto_impl.ExpandExpDif(input, other)
