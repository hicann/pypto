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
import pypto
from pypto import Tensor

__all__ = ["sin", "cos", "sigmoid", "softmax", "rms_norm"]


def sin(input: Tensor) -> Tensor:
    """Return a tensor containing the element-wise sine values of input.

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements
        are the sine values of the corresponding elements in the input tensor.


    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.sin(x)

    Input x:[-0.5461,  0.1347, -2.7266, -0.2746]
    Output y:[-0.5194,  0.1343, -0.4032, -0.2711]
    """
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    number2048 = 2048.0
    one_over_n = 1.0 / 2048.0
    inv_half_pi = 0.63661975
    pi0 = 1.5708008
    pi1 = -0.0000044535846
    pi2 = -8.706138e-10
    f_025 = 0.25
    f_05 = 0.5
    f_4 = 4.0
    f_1 = 1.0
    f_nega_1 = -1.0
    f_nega_2 = -2.0

    x_scaled = pypto.mul(input, one_over_n)
    x_over_pi = pypto.mul(x_scaled, inv_half_pi)
    n = pypto.cast(x_over_pi, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(x_over_pi, one_over_n)
    n0 = pypto.cast(n0, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(n0, number2048)

    n1 = pypto.sub(n, n0)

    fix = pypto.mul(n0, pi0)
    x_fix = pypto.sub(x_scaled, fix)
    fix = pypto.mul(n1, pi0)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi1)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi1)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi2)
    x_fix = pypto.sub(x_fix, fix)

    pi_02 = 1.5703125
    pi_12 = 0.0004837513

    remain_x = pypto.mul(x_fix, number2048)
    temp = pypto.mul(remain_x, inv_half_pi)
    n2 = pypto.cast(temp, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)

    n0 = pypto.mul(n0, number2048)
    n1 = pypto.mul(n1, number2048)
    fix = pypto.mul(n0, pi_02)
    x_fix = pypto.sub(input, fix)
    fix = pypto.mul(n1, pi_02)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_12)
    x_fix = pypto.sub(x_fix, fix)

    pi_22 = 0.000000075495336
    fix = pypto.mul(n2, pi_02)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_12)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_22)
    x_fix = pypto.sub(x_fix, fix)

    pi_32 = 2.5579538e-12
    fix = pypto.mul(n2, pi_12)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_22)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_32)
    x_fix = pypto.sub(x_fix, fix)

    pi_42 = 5.389786e-15
    fix = pypto.mul(n2, pi_22)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_32)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_42)
    x_fix = pypto.sub(x_fix, fix)

    pi_52 = 5.166901e-19
    fix = pypto.mul(n2, pi_32)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_42)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_52)
    x_fix = pypto.sub(x_fix, fix)

    pi_62 = 3.281839e-22
    fix = pypto.mul(n2, pi_42)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_52)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_62)
    x_fix = pypto.sub(x_fix, fix)

    fix = pypto.mul(n2, pi_52)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_62)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n2, pi_62)
    x_fix = pypto.sub(x_fix, fix)

    half_n2 = pypto.mul(n2, f_05)
    half4_n2 = pypto.mul(n2, f_025)
    n_half2 = pypto.cast(half_n2, pypto.DT_FP32, pypto.CastMode.CAST_FLOOR)
    n_half4 = pypto.cast(half4_n2, pypto.DT_FP32, pypto.CastMode.CAST_FLOOR)

    k1 = pypto.mul(n_half2, f_nega_2)
    k2 = pypto.mul(n_half4, f_4)
    sign = pypto.add(k1, k2)
    sign = pypto.add(sign, f_1)

    ifcos = pypto.add(n2, k1)
    ifsin = pypto.mul(ifcos, f_nega_1)
    ifsin = pypto.add(ifsin, f_1)

    scoef4 = 0.0000027183114939898219064
    scoef3 = -0.000198393348360966317347
    scoef2 = 0.0083333293858894631756
    scoef1 = -0.166666666416265235595
    x_pow = pypto.mul(x_fix, x_fix)
    sin_poly = pypto.mul(x_pow, scoef4)
    sin_poly = pypto.add(sin_poly, scoef3)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, scoef2)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, scoef1)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, f_1)
    sin_poly = pypto.mul(x_fix, sin_poly)

    ccoef4 = 0.0000243904487962774090654
    ccoef3 = -0.00138867637746099294692
    ccoef2 = 0.0416666233237390631894
    ccoef1 = -0.499999997251031003120
    cos_poly = pypto.mul(x_pow, ccoef4)
    cos_poly = pypto.add(cos_poly, ccoef3)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, ccoef2)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, ccoef1)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, f_1)

    temp1 = pypto.mul(sin_poly, ifsin)
    cos_poly = pypto.mul(cos_poly, ifcos)
    res = pypto.add(temp1, cos_poly)
    res = pypto.mul(res, sign)

    if dtype != res.dtype:
        res = pypto.cast(res, dtype)
    return res


def cos(input: Tensor) -> Tensor:
    """Return a tensor containing the element-wise cosine values of input.

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements are
        the cosine values of the corresponding elements in the input tensor.


    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.cos(x)

    Input x:[0.0000, 0.7854, 1.5708, 2.3562]
    Output y:[1.0000, 0.7071, 0.0000, -0.7071]
    """
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    number2048 = 2048.0
    one_over_n = 1.0 / 2048.0
    inv_half_pi = 0.63661975

    pi0 = 1.5708008
    pi1 = -0.0000044535846
    pi2 = -8.706138e-10
    f_025 = 0.25
    f_05 = 0.5
    f_4 = 4.0
    f_1 = 1.0
    f_nega_1 = -1.0
    f_nega_2 = -2.0

    x_scaled = pypto.mul(input, one_over_n)
    x_over_pi = pypto.mul(x_scaled, inv_half_pi)
    n = pypto.cast(x_over_pi, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(x_over_pi, one_over_n)
    n0 = pypto.cast(n0, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(n0, number2048)
    n1 = pypto.sub(n, n0)

    fix = pypto.mul(n0, pi0)
    x_fix = pypto.sub(x_scaled, fix)
    fix = pypto.mul(n1, pi0)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi1)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi1)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi2)
    x_fix = pypto.sub(x_fix, fix)

    pi_02 = 1.5703125
    pi_12 = 0.0004837513

    remain_x = pypto.mul(x_fix, number2048)
    temp = pypto.mul(remain_x, inv_half_pi)
    n2 = pypto.cast(temp, pypto.DT_FP32, pypto.CastMode.CAST_ROUND)
    n0 = pypto.mul(n0, number2048)
    n1 = pypto.mul(n1, number2048)
    fix = pypto.mul(n0, pi_02)
    x_fix = pypto.sub(input, fix)
    fix = pypto.mul(n1, pi_02)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_12)
    x_fix = pypto.sub(x_fix, fix)

    pi_22 = 0.000000075495336
    fix = pypto.mul(n2, pi_02)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_12)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_22)
    x_fix = pypto.sub(x_fix, fix)

    pi_32 = 2.5579538e-12
    fix = pypto.mul(n2, pi_12)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_22)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_32)
    x_fix = pypto.sub(x_fix, fix)

    pi_42 = 5.389786e-15
    fix = pypto.mul(n2, pi_22)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_32)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_42)
    x_fix = pypto.sub(x_fix, fix)

    pi_52 = 5.166901e-19
    fix = pypto.mul(n2, pi_32)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_42)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_52)
    x_fix = pypto.sub(x_fix, fix)

    pi_62 = 3.281839e-22
    fix = pypto.mul(n2, pi_42)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_52)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n0, pi_62)
    x_fix = pypto.sub(x_fix, fix)

    fix = pypto.mul(n2, pi_52)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n1, pi_62)
    x_fix = pypto.sub(x_fix, fix)
    fix = pypto.mul(n2, pi_62)
    x_fix = pypto.sub(x_fix, fix)

    n2 = pypto.add(n2, f_1)
    half_n2 = pypto.mul(n2, f_05)
    half4_n2 = pypto.mul(n2, f_025)
    n_half2 = pypto.cast(half_n2, pypto.DT_FP32, pypto.CastMode.CAST_FLOOR)
    n_half4 = pypto.cast(half4_n2, pypto.DT_FP32, pypto.CastMode.CAST_FLOOR)

    k1 = pypto.mul(n_half2, f_nega_2)
    k2 = pypto.mul(n_half4, f_4)
    sign = pypto.add(k1, k2)
    sign = pypto.add(sign, f_1)

    ifcos = pypto.add(n2, k1)
    ifsin = pypto.mul(ifcos, f_nega_1)
    ifsin = pypto.add(ifsin, f_1)

    scoef4 = 0.0000027183114939898219064
    scoef3 = -0.000198393348360966317347
    scoef2 = 0.0083333293858894631756
    scoef1 = -0.166666666416265235595
    x_pow = pypto.mul(x_fix, x_fix)
    sin_poly = pypto.mul(x_pow, scoef4)
    sin_poly = pypto.add(sin_poly, scoef3)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, scoef2)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, scoef1)
    sin_poly = pypto.mul(x_pow, sin_poly)
    sin_poly = pypto.add(sin_poly, f_1)
    sin_poly = pypto.mul(x_fix, sin_poly)

    ccoef4 = 0.0000243904487962774090654
    ccoef3 = -0.00138867637746099294692
    ccoef2 = 0.0416666233237390631894
    ccoef1 = -0.499999997251031003120
    cos_poly = pypto.mul(x_pow, ccoef4)
    cos_poly = pypto.add(cos_poly, ccoef3)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, ccoef2)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, ccoef1)
    cos_poly = pypto.mul(x_pow, cos_poly)
    cos_poly = pypto.add(cos_poly, f_1)

    temp1 = pypto.mul(sin_poly, ifsin)
    cos_poly = pypto.mul(cos_poly, ifcos)
    res = pypto.add(temp1, cos_poly)
    res = pypto.mul(res, sign)

    if res.dtype != dtype:
        res = pypto.cast(res, dtype)
    return res


def sigmoid(input: Tensor) -> Tensor:
    """ Return a tensor containing the element-wise sigmoid values of input.
        The sigmoid function is a common activation function in machine learning,
        defined mathematically as: sigmoid(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).

    Returns
    -------
    Tensor
        A tensor with the same shape and data type as the input, whose elements are
        the results of the input elements mapped to the interval (0, 1) via the sigmoid function.

    Examples
    --------
    x = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.sigmoid(x)

    Input x:[-3.0, 0.0, 2.0, 5.0]
    Output y:[0.0474, 0.5000, 0.8808, 0.9933]
    """
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    f_1 = 1.0
    f_nega_1 = -1.0

    exp_res = pypto.exp(pypto.mul(input, f_nega_1))
    res = pypto.add(exp_res, f_1)
    ones = pypto.full(res.shape, 1.0, pypto.DT_FP32, valid_shape=res.shape)
    res = pypto.div(ones, res)

    if dtype != pypto.DT_FP32:
        res = pypto.cast(res, dtype)
    return res


def softmax(input: Tensor, dim: int) -> Tensor:
    """ Return a tensor obtained by applying the softmax activation function to the input.
        Mathematically, for an input tensor x along a specified dimension dim,
        the softmax of element x_i is computed as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in dimension dim

    Parameters
    ----------
    input: Tensor
        The input tensor to compute.
        The supported data type is DT_FP32.
        Empty tensors are not supported, and the shape size must not exceed 2147483647 (i.e., INT32_MAX).
    dim: int
        Specify the dimension for normalization.
        Negative indices are supported (e.g., -1 indicates the last dimension).
        It must be within the range of [-input.dim, input.dim - 1].

    Returns
    -------
    Tensor
        A tensor with the same shape as the input, where the sum of elements along the
        specified dimension is 1, and the data type is determined by dtype or the input type.

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_FP32)
    y = pypto.softmax(x, -1)

    Input x:[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    Output y:[[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
    """
    dtype = input.dtype
    input = pypto.cast(input, pypto.DT_FP32)

    rowmax = pypto.amax(input, dim, True)
    sub_res = pypto.sub(input, rowmax)
    exp_res = pypto.exp(sub_res)
    esum = pypto.sum(exp_res, dim, True)
    output = pypto.div(exp_res, esum)

    if dtype != pypto.DT_FP32:
        output = pypto.cast(output, dtype)
    return output


def rms_norm(input: Tensor, gamma: Tensor = None, epsilon: float = 1e-6) -> Tensor:
    """
    Root Mean Square LayerNorm (RMSNorm) along the last dimension.
    If `gamma` is provided, applies an element-wise scale on the last dim.

    Parameters
    ----------
    input : Tensor
        Input tensor. Any shape (..., C).
    gamma : Tensor | None
        Optional scale of shape (C,).
    epsilon : float
        Numerical stability constant (default: 1e-6).

    Returns
    -------
    Tensor
        Same shape as `input`, cast back to the original dtype.

    Examples
    --------
    x = pypto.tensor([2, 4], pypto.DT_FP32)
    gamma = pypto.tensor([4], pypto.DT_FP32)
    y = pypto.rms_norm(x, gamma)

    Input x: [[1, 2, 3, 4],
              [5, 6, 7, 8]]
          gamma: [1, 1, 1, 1]
    Output y: [[0.3651, 0.7302, 1.0954, 1.4605], 
               [0.7580, 0.9097, 1.0613, 1.2129]]
    """
    in_dtype = input.dtype
    x = pypto.cast(input, pypto.DT_FP32)

    n = x.shape[-1]

    y = pypto.sqrt(pypto.sum(x * x * (1.0 / n), -1, keepdim=True)) + epsilon

    ones = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = x * ones / y

    if gamma is not None:
        rank = input.dim
        shape = [1] * rank
        shape[-1] = gamma.shape[0]
        g = pypto.cast(pypto.reshape(gamma, shape), pypto.DT_FP32)
        y *= g

    if in_dtype != pypto.DT_FP32:
        y = pypto.cast(y, in_dtype)
    return y
