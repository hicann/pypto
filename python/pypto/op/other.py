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
from typing import Union

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
