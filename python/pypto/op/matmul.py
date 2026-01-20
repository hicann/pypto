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
import struct
from typing import Any, Type

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import DataType
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


@op_wrapper
def matmul(
    input,
    mat2,
    out_dtype,
    *,
    a_trans=False,
    b_trans=False,
    c_matrix_nz=False,
    extend_params=None
) -> Tensor:
    """
    Performs matrix multiplication with support for batched operations, broadcasting, transposition, and extended
    features like bias addition and dequantization.

    Supports two primary computation modes:
    1.  Standard matrix multiplication of two matrices.
    2.  Batched matrix-matrix multiplication.

    `input` and `mat2` can be 2-D, 3-D, or 4-D tensors, each containing the same number of matrices.
    - If `input` is (n x k) and `mat2` is (k x m), output is (n x m).
    - If `input` is (b x n x k) and `mat2` is (b x k x m), output is (b x n x m).

    Note: Broadcasting is supported for 3-D or 4-D tensors.
    Example: If `input` is (1 x n x k) and `mat2` is (b x k x m), output will be (b x n x m).

    Parameters
    ----------
    input : Tensor
        The left operand matrix.
    mat2 : Tensor
        The right operand matrix.
    out_dtype : dtype
        The data type of the output tensor.

    Keyword Arguments
    ----------
    a_trans : bool, default=False
        If True, transpose the left matrix (`input`) before multiplication.
    b_trans : bool, default=False
        If True, transpose the right matrix (`mat2`) before multiplication.
    c_matrix_nz : bool, default=False
        If True, output the result matrix in NZ (non-zero) format.
    extend_params : dict, optional
        A dictionary specifying extended computation features:
        - 'bias_tensor': Tensor
            Adds a learnable bias to the output: C = A @ B + bias.
        - 'scale': float
            For dequantization: C = DEQF16(ReLU(A @ B)) * scale.
        - 'scale_tensor': Tensor
            For dequantization with a per-channel scale: C = DEQF16(ReLU(A @ B)) * scale_tensor.
        - 'relu_type': ReLuType
            Type of ReLU activation to apply before dequantization (e.g., ReLuType.RELU).

    Returns
    -------
    Tensor
        A new tensor containing the matrix multiplication result.

    Raises
    ------
    RuntimeError
        If input dimensions are invalid (<2-D or >4-D), or if matrix dimensions are incompatible.

    Examples
    --------
    # Standard matrix multiplication
    a = pypto.tensor((16, 32), pypto.DT_BF16, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_BF16, "tensor_b")
    pypto.matmul(a, b, pypto.DT_BF16)

    # Batched matrix multiplication
    a = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
    b = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
    pypto.matmul(a, b, pypto.DT_FP16)

    # Batched multiplication with broadcasting
    a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
    b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
    pypto.matmul(a, b, pypto.DT_FP32)

    # With bias addition
    a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
    bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
    extend_params = {'bias_tensor': bias}
    pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)

    # With dequantization (scale)
    a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
    extend_params = {'scale': 0.2}
    pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)

    # With dequantization (scale & ReLU)
    a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
    extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
    pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)

    # With dequantization (scale_tensor & ReLU)
    a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
    scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
    extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
    pypto.matmul(a, b, pypto.DT_BF16, extend_params=extend_params)
    """
    __validate_inputs(input, mat2, out_dtype, [a_trans, b_trans, c_matrix_nz, extend_params])
    if input.Dim() == 2:
        if extend_params is not None:
            extend_params = pypto_impl.MatmulExtendParam(
                **__convert_matmul_extend_params(extend_params)
            )
            return pypto_impl.Matmul(
                out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz, extend_params
            )
        else:
            return pypto_impl.Matmul(
                out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz
            )
    else:
        return pypto_impl.BatchMatmul(
            out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz
        )


def __validate_type(value: Any, expect_type: Type, arg_name: str = "input") -> None:
    if value is None:
        return
    if not isinstance(value, expect_type):
        raise TypeError(
            f"Argument '{arg_name}' must be of type {expect_type.__name__}, but got {type(value).__name__}."
        )


def __get_valid_shape(tensor):
    return [SymbolicScalar.from_base(n) for n in tensor.GetValidShape()]


def __validate_shape(input_tensor1: Tensor, input_tensor2: Tensor, a_trans: bool, b_trans: bool) -> None:
    input_dim = input_tensor1.Dim()
    mat2_dim = input_tensor2.Dim()
    if input_dim != mat2_dim or input_dim not in {2, 3, 4}:
        raise RuntimeError(
            "Tensor dimension mismatch. Expect input_dim == mat2_dim and both in [2, 3, 4], "
            f"got input_dim: {input_dim}, mat2_dim: {mat2_dim}."
        )

    input_valid_shape = __get_valid_shape(input_tensor1)
    mat2_valid_shape = __get_valid_shape(input_tensor2)
    m_dim, ka_dim = (input_valid_shape[-2], input_valid_shape[-1]) if not a_trans else \
        (input_valid_shape[-1], input_valid_shape[-2])
    kb_dim, n_dim = (mat2_valid_shape[-2], mat2_valid_shape[-1]) if not b_trans else \
        (mat2_valid_shape[-1], mat2_valid_shape[-2])
    if ka_dim.is_concrete() and kb_dim.is_concrete() and ka_dim != kb_dim:
        raise RuntimeError(
            "K-dimension valid shape mismatch. "
            f"Got input valid shape: {input_valid_shape}, mat2 valid shape: {mat2_valid_shape}, "
            f"a_trans: {a_trans}, b_trans: {b_trans}."
        )


def __validate_inputs(input_tensor1, input_tensor2, out_dtype, optional_param) -> None:
    a_trans, b_trans, is_out_nz, extend_params = optional_param
    __validate_type(out_dtype, DataType, "out_dtype")
    __validate_type(a_trans, bool, "a_trans")
    __validate_type(b_trans, bool, "b_trans")
    __validate_type(is_out_nz, bool, "is_out_nz")
    __validate_type(extend_params, dict, "extend_params")
    __validate_shape(input_tensor1, input_tensor2, a_trans, b_trans)

    if is_out_nz:
        raise ValueError("Output tensor do not support NZ currently.")
    input1_valid = input_tensor1.GetDataType() == pypto_impl.DataType.DT_FP32 \
        and input_tensor1.Format() == pypto_impl.TileOpFormat.TILEOP_NZ
    input2_valid = input_tensor2.GetDataType() == pypto_impl.DataType.DT_FP32 \
        and input_tensor2.Format() == pypto_impl.TileOpFormat.TILEOP_NZ
    if input1_valid or input2_valid:
        raise ValueError("Input tensor with DT_FP32 must use ND format, NZ format is not support currently.")
    if input_tensor1.GetDataType() != input_tensor2.GetDataType():
        raise ValueError("All input tensors must have the same data type")
    if input_tensor1.Dim() != 2 and extend_params is not None:
        raise RuntimeError(
            "extend_params is not supported for batched matrix multiplication."
        )


def __convert_matmul_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pypto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pypto_impl.Tensor())
    extend_params.setdefault('relu_type', pypto_impl.ReLuType.NO_RELU)
    extend_params.setdefault('scale', 0.0)
    return extend_params
