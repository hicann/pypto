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
from ..enum import ScatterMode
from .._op_wrapper import op_wrapper
from ..tensor import Tensor
from .._element import Element


@op_wrapper
def index_add_(
    input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1
    ) -> Tensor:
    """
    Accumulate the elements of `alpha` times `source` into `input` tensor by
    adding to the indices in the order given in `index`.

    For a 3-D tensor this function specified output as:
    input[index[i], :, :] += alpha * source[i, :, :]  # if dim == 0
    input[:, index[i], :] += alpha * source[:, i, :]  # if dim == 1
    input[:, :, index[i]] += alpha * source[:, :, i]  # if dim == 2

    Parameters
    ----------
    input : Tensor
        Source tensor that needs to be added in place.
    dim : int
        Dimension along which to index. Negative indexing is supported.
    index : Tensor
        Indices of `source` to select from, should have dtype either int64
        or int32 and the dimension must be 1.The length of `index` must have
        the same size as the `dim` th dimension of `source`.
    source : Tensor
        The tensor containing values to add. The dimth dimension of
        `source` must have the same size as the length of `index`, and
        all other dimensions must match `self`, or an error will be raised.

    Keyword Arguments:
    ----------
    alpha : Number
        The scalar multiplier for `source`.

    Returns
    -------
    Tensor
        A new tensor sharing the same storage with the `input` tensor.

    Raises
    ------
    RuntimeError
        If any value in `index` is outside the inclusive range
        [0, source.shape[dim]-1].

    Examples
    --------
    x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
    source = pypto.tensor([3, 3], pypto.DT_INT32)        # shape (3, 3)
    index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
    dim = 0

    # use alpha
    y = pypto.index_add_(x, dim, index, source, alpha=1)

    # not use alpha
    y = pypto.index_add_(x, dim, index, source)

    Input x:   [[0 0 0],
                [0 0 0]]
    source:    [[1 1 1],
                [1 1 1],
                [1 1 1]]
    index:      [0 1 0]

    Output y:  [[2 2 2],
                [1 1 1]]               # shape (2, 3)
    """
    if alpha == 1 or alpha == 1.0:
        return pypto_impl.IndexAdd_(input, source, index, dim)
    else:
        return pypto_impl.IndexAdd_(input, source, index, dim, pypto_impl.Element(input.dtype, alpha))


@op_wrapper
def index_add(
    input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1
    ) -> Tensor:
    """
    The out-of-place version of index_add_()
    """
    if alpha == 1 or alpha == 1.0:
        return pypto_impl.IndexAdd(input, source, index, dim)
    else:
        return pypto_impl.IndexAdd(input, source, index, dim, pypto_impl.Element(input.dtype, alpha))


@op_wrapper
def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Gather elements from `input` along `dim` according to `index`.

    This function specified output for a 3-D tensor:
    output[i][j][k] = input[index[i][j][k]][j][k] # if dim == 0
    output[i][j][k] = input[i][index[i][j][k]][k] # if dim == 1
    output[i][j][k] = input[i][j][index[i][j][k]] # if dim == 2

    Parameters
    ----------
    input : Tensor
        Source tensor from which to gather values.
    index : Tensor
        Integer tensor containing the subscripts to pick along `dim`. It must have
        the same numbers of dimensions as `input`. And it is also required that
        index.shape[d] <= input.shape[d] for all dimensions d != dim.
    dim : int
        Dimension in `input` along which to gather. Negative indexing is supported.

    Returns
    -------
    Tensor
        A new tensor, with the same dtype as `input` and the same shape as `index`.

    Raises
    ------
    IndexError
        If any value in `index` is outside the inclusive range
        [0, input.shape[dim]-1].
    RuntimeError
        If the broadcast shape of `index` against `input` is incompatible.

    Examples
    --------
    x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)

    index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
    dim = 0
    y = pypto.gather(x, dim, index)

    Input x:  [[0 1 2 3 4],
               [5 6 7 8 9],
               [10 11 12 13 14]]
    index:    [[0 1 2 0],
               [1 2 0 1],
               [2 2 1 0]]

    Output y: [[0 6 12 3],
               [5 11 2 8],
               [10 11 7 3]]               # shape (3, 4)

    """

    return pypto_impl.GatherElements(input, index, dim)


@op_wrapper
def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Gathers slices from `param` along a single axis `dim` using `indices`.
    param ∈ {S₀xS₁x…xS_{n-1}}, indices ∈ {I₀xI₁x…xI_{m-1}}.

    Output shape
    out.shape = (S₀,…,S_{dim-1}, I₀,…,I_{m-1}, S_{dim+1},…,S_{n-1}).
    That is, the dimension `S_dim` in `param` is replaced by the full shape of `indices`, 
    while all other dimensions of `param` are preserved.

    For any multi-indices
    i = (i₀,…,i_{m-1}), t = (t₀,…,t_{n-2}),
    out[i, t] = param[t₀,…,t_{dim-1}, indices[i], t_{dim},…,t_{n-2}].
    Parameters
    ----------
    input : Tensor
    2-4-D tensor of shape (S0, S1, …, Sn-1) that provides the source values to gather from.

    index : Tensor (integer type)
    1-2-D integer tensor of shape (I0, I1); every entry must satisfy 0 ≤ value < S_dim.

    dim : int
    int axis in the range -n ≤ dim < n along which to gather; negative values are interpreted as dim + n.

    Examples
    --------
    x = pypto.tensor([3, 4], pypto.DT_FP32)
    indices = pypto.tensor([2,], pypto.DT_INT32)
    out0 = pypto.index_select(x, 0, indices)
    out1 = pypto.index_select(x, 1, indices)

    Input x:       [[ 0.1427,  0.0231, -0.5414, -1.0009],
                    [-0.4664,  0.2647, -0.1228, -1.1068],
                    [-1.1734, -0.6571,  0.7230, -0.6004]]
    Input indices:  [0, 2]
    Output out1 :  [[ 0.1427,  0.0231, -0.5414, -1.0009],
                    [-1.1734, -0.6571,  0.7230, -0.6004]]
    Output out2 :  [[ 0.1427, -0.5414],
                    [-0.4664, -0.1228],
                    [-1.1734,  0.7230]]
    """

    return pypto_impl.index_select(input, dim, index)


@op_wrapper
def scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Write all values from the tensor 'src' into 'input' at the indices specified in the 'index' tensor.

    This function calculates the formula:
    For dim2,
    input[index[i][j]][:] = src[i][:]
    For dim4,
    input[index[i][j]][index[i][j]][0][:] = src[i][j][0][:]

    Parameters
    ----------
    input : Tensor
        The input tensor to be chenged.
    dim : int
        The axis along which to index.
    index : Tensor
        The indices of elements to scatter.
    src : Tensor
        The source elements to scatter.

    Returns
    -------
    Tensor
        A new tensor containing the elements of input after scatter.

    Raises
    ------
    RuntimeError
        If the dimension of 'index' is not equal 2.
        If the dimension of 'input' and 'src' is not equal 2 or 4.
        If the value of 'index' is not less than the blockSize of 'input'.

    See Also
    --------
    gather : The inverse operation, gather values along an axis specified by dim.

    Examples
    --------
    # dim2
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    o = pypto.scatter_update(x, -2, y, z)

    Input x:[[0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0],
             [0 0 0]]
    Input y:[[1 2],
             [4 5]]
    Input z:[[1 2 3],
             [4 5 6],
             [7 8 9],
             [10 11 12]]
    Output o:[[0 0 0],
              [1 2 3],
              [4 5 6],
              [0 0 0],
              [7 8 9],
              [10 11 12],
              [0 0 0],
              [0 0 0]])

    #dim4
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    o = pypto.scatter_update(x, -2, y, z)

    Input x:[[
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
             ],
             [
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
                [[0 0 0]],
             ]]
    Input y:[[1 8],
             [4 10]]
    Input z:[[
                [[1 2 3]],
                [[4 5 6]],
             ],
             [
                [[7 8 9]],
                [[10 11 12]],
             ]]
    Output o:[[
                [[0 0 0]],
                [[1 2 3]],
                [[0 0 0]],
                [[0 0 0]],
                [[7 8 9]],
                [[0 0 0]],
             ],
             [
                [[0 0 0]],
                [[0 0 0]],
                [[4 5 6]],
                [[0 0 0]],
                [[10 11 12]],
                [[0 0 0]],
             ]]
    """
    if dim != -2:
        raise ValueError("scatter currection only support the case where dim = -2.")
    dims = input.Dim()
    if dims == 4:
        chunk_size = input.GetShape()[1]
    elif dims == 2:
        chunk_size = 1
    else:
        raise ValueError("dim must be 2 or 4")

    return pypto_impl.ScatterUpdate(input, index, src, -2, "PA_BSND", chunk_size)


def get_scatter_mode(reduce: str):
    if reduce is None:
        return ScatterMode.NONE
    elif reduce == 'add':
        return ScatterMode.ADD
    elif reduce == 'multiply':
        return ScatterMode.MULTIPLY
    else:
        raise ValueError("scatter reduce only support 'add', 'multiply'")


@op_wrapper
def scatter_(input: Tensor, dim: int, index: Tensor, src: Union[float, Element], *, reduce: str = None) -> Tensor:
    """Write all values from the value 'src' into 'input' at the indices specified in the 'index' tensor.

    This function calculates the formula:
    For a 3-D tensor, 'input' is update as:
    self[index[i][j][k]][j][k] = src  # if dim == 0
    self[i][index[i][j][k]][k] = src  # if dim == 1
    self[i][j][index[i][j][k]] = src  # if dim == 2

    Parameters
    ----------
    input : Tensor
        The input tensor to be chenged.
    dim : int
        The axis along which to index.
    index : Tensor
        The indices of elements to scatter.
    src : float
        The scalar value to scatter.

    Returns
    -------
    Tensor
        A new tensor containing the elements of input after scatter.

    Raises
    ------
    RuntimeError
        If the dimension of 'index' is not equal to the dimension of 'input'.
        If the index.size(d) > input.size(d)
        If the value of 'input[i][j][k]' is bigger than the shape size of the dimension of 'input'.

    See Also
    --------
    gather : The inverse operation, gather values along an axis specified by dim.

    Examples
    --------
    # dim2 and src is scalar
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)

    Input x:  [[0 0 0 0 0],
               [0 0 0 0 0],
               [0 0 0 0 0]]
    Input y:  [[1 2],
               [0 1]]
    Output o:  [[2.0 0   0 0 0],
                [2.0 2.0 0 0 0],
                [0   2.0 0 0 0]]
    """
    scatter_mode = get_scatter_mode(reduce)
    if isinstance(src, float):
        return pypto_impl.Scatter_(input, index, pypto_impl.Element(input.dtype, src), dim, scatter_mode)
    return pypto_impl.Scatter_(input, index, src, dim, scatter_mode)


@op_wrapper
def scatter(input: Tensor, dim: int, index: Tensor, src: Union[float, Element], *, reduce: str = None) -> Tensor:
    """Out-of-place version of 'scatter_'."""
    scatter_mode = get_scatter_mode(reduce)
    if isinstance(src, float):
        return pypto_impl.Scatter(input, index, pypto_impl.Element(input.dtype, src), dim, scatter_mode)
    return pypto_impl.Scatter(input, index, src, dim, scatter_mode)
