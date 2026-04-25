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
__all__ = ["sigmoid", "softmax", "rms_norm", "stateless_random_uniform_v2", "stateless_random_normal_v2"]

import pypto
from pypto import Tensor


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
    res = pypto.div(ones, res, pypto.DivAlgorithm.INTRINSIC)

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
    output = pypto.div(exp_res, esum, pypto.DivAlgorithm.INTRINSIC)

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

    y = pypto.sqrt(pypto.sum(x * x * (1.0 / n), -1, keepdim=True) + epsilon)

    ones = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(x * ones, y, pypto.DivAlgorithm.INTRINSIC)

    if gamma is not None:
        rank = input.dim
        shape = [1] * rank
        shape[-1] = gamma.shape[0]
        g = pypto.cast(pypto.reshape(gamma, shape), pypto.DT_FP32)
        y *= g

    if in_dtype != pypto.DT_FP32:
        y = pypto.cast(y, in_dtype)
    return y


def stateless_random_uniform_v2(shape, key, counter, alg, dtype) -> Tensor:
    """Return a tensor containing uniformly distributed random values.

    Parameters
    ----------
    shape: List[int]
        The shape of the output tensor.
    key: List[int]
        The key for random number generation.
    counter: List[int]
        The counter for random number generation.
    alg: List[int]
        The algorithm to use for random number generation, support 1(Philox) and 3(auto_select, select Philox).
    dtype: DataType
        The data type of the output tensor.
        The supported data types are DT_FP32, DT_FP16, and DT_BF16.

    Returns
    -------
    Tensor
        A tensor with the specified shape and data type, containing
        uniformly distributed random values in the range [0, 1).


    Examples
    --------
    shape = [4, 4]
    key = [1234]
    counter = [0, 1]
    alg = [1]

    y = pypto.stateless_random_uniform_v2(shape, key, counter, alg, dtype)

    Output y:[[0.1689806  0.9725481  0.90036285 0.16582811]
              [0.1454581  0.48029935 0.02495587 0.99239147]
              [0.02835405 0.10649502 0.45283175 0.87260246]
              [0.6877538  0.24809706 0.95886254 0.24039495]]
    """
    if len(shape) < 1 or len(shape) > 4:
        raise ValueError(f"output shape dim should be in [1, 4], but got {len(shape)}.")

    if len(key) != 1:
        raise ValueError(f"input key number should be 1, but got {len(key)}.")

    if len(counter) != 2:
        raise ValueError(f"input counter number should be 2, but got {len(counter)}.")

    if len(alg) != 1:
        raise ValueError(f"input alg number should be 1, but got {len(alg)}.")

    alg = alg[0]
    if alg != 1 and alg != 3:
        raise ValueError(f"alg only support Philox.")

    tile_shapes = pypto.get_vec_tile_shapes()
    tile_shape_one_dim = 1
    for dim_num in tile_shapes:
        tile_shape_one_dim *= dim_num
    pypto.set_vec_tile_shapes(tile_shape_one_dim)

    shape_one_dim = 1
    for dim_num in shape:
        shape_one_dim *= dim_num

    counter0, counter1 = counter
    uniform_res = pypto.uniform(key[0], counter0, counter1, [shape_one_dim], rounds=10, dtype=dtype)
    pypto.set_vec_tile_shapes(*tile_shapes)
    return pypto.reshape(uniform_res, shape)


def stateless_random_normal_v2(shape, key, counter, alg, dtype) -> Tensor:
    """Return a tensor containing normally distributed random values.

    Parameters
    ----------
    shape: List[int]
        The shape of the output tensor.
    key: List
        The key for random number generation.
    counter: List[int]
        The counter for random number generation.
    alg: List[int]
        The algorithm to use for random number generation, support 1(Philox) and 3(auto_select, select Philox).
    dtype: DataType
        The data type of the output tensor.
        The supported data types are DT_FP32, DT_FP16, and DT_BF16.

    Returns
    -------
    Tensor
        A tensor with the specified shape and data type, containing
        uniformly distributed random values in the range [0, 1).


    Examples
    --------
    shape = [4, 4]
    key = [1234]
    counter = [0, 1]
    alg = [1]

    y = pypto.stateless_random_normal_v2(shape, key, counter, alg, dtype)

    Output y:[[-0.32364845  1.8577391   0.39556974  0.2311697 ]
              [ 0.24243996 -1.9485782  -0.12983137  2.7137496 ]
              [ 1.6558666   2.0938187  -0.90338254  0.8765667 ]
              [ 0.86518306  0.01034508  0.2893259   0.01748212]]
    """
    def box_muller(input: Tensor) -> Tensor:
        input = pypto.reshape(input, [-1])
        tensor_len = input.shape[0]
        u1_index = pypto.arange(0, tensor_len, 2)
        u2_index = pypto.add(u1_index, 1)
        u1 = pypto.gather(input, 0, u1_index)
        u2 = pypto.gather(input, 0, u2_index)

        eps = 1.0e-7
        m_pi = 3.14159265358979323846
        u1 = pypto.maximum(u1, eps)
        v1 = pypto.mul(u2, 2.0 * m_pi)
        v2 = pypto.sqrt(pypto.mul(pypto.log(u1), -2.0))
        f0 = pypto.sin(v1)
        f1 = pypto.cos(v1)
        f2 = pypto.mul(f0, v2)
        f3 = pypto.mul(f1, v2)
        f4 = pypto.zeros([tensor_len], dtype=pypto.DT_FP32)

        pypto.set_vec_tile_shapes(tensor_len)
        pypto.scatter_(f4, 0, u1_index, f2, reduce='add')

        if tensor_len <= 1:
            return f4

        if tensor_len % 2 != 0:
            u2_index = pypto.arange(1, tensor_len, 2)
        pypto.scatter_(f4, 0, u2_index, f3, reduce='add')
        return f4

    tile_shapes = pypto.get_vec_tile_shapes()
    uniform_res = pypto.stateless_random_uniform_v2(shape, key, counter, alg, dtype=pypto.DT_FP32)
    normal_res = box_muller(uniform_res)

    if dtype != pypto.DT_FP32:
        normal_res = pypto.cast(normal_res, dtype)
    pypto.set_vec_tile_shapes(*tile_shapes)
    return pypto.reshape(normal_res, shape)
