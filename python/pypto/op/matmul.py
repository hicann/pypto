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
""" """
import struct

from .. import pypto_impl
from .._op_wrapper import op_wrapper
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
    input_dim = input.Dim()
    mat2_dim = mat2.Dim()
    check_data_valid(input, mat2, c_matrix_nz)
    if input_dim == mat2_dim == 2:
        if (extend_params is None) or (not extend_params):
            return pypto_impl.Matmul(
                out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz
            )
        else:
            extend_params = pypto_impl.MatmulExtendParam(
                **convert_matmul_extend_params(extend_params)
            )
            return pypto_impl.Matmul(
                out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz, extend_params
            )
    elif (input_dim == mat2_dim == 3) or (input_dim == mat2_dim == 4):
        return pypto_impl.BatchMatmul(
            out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz
        )
    else:
        raise RuntimeError(
            "input dim and mat dim must equals, which only support 2-D/3-D/4-D currently"
        )


def check_data_valid(input_tensor1, input_tensor2, is_out_nz):
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


def convert_matmul_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pypto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pypto_impl.Tensor())
    extend_params.setdefault('relu_type', pypto_impl.ReLuType.NO_RELU)
    extend_params.setdefault('scale', 0.0)
    return extend_params
