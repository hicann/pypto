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

from typing import List, Optional
from functools import wraps

from .enum import DataType, TileOpFormat
from .tensor import Tensor
from .pypto_impl import ir


def _count_calls(func):
    count = 0

    @wraps(func)
    def wrapper(tensor, name: str = "", *, dynamic_axis: Optional[List[int]] = None,
                tensor_format: Optional[TileOpFormat] = None):
        nonlocal count
        count += 1
        if name == "":
            name = f"TENSOR_{count}"
        return func(tensor, name, dynamic_axis, tensor_format)

    return wrapper


@_count_calls
def from_torch(tensor, name: str = "", dynamic_axis: Optional[List[int]] = None,
               tensor_format: Optional[TileOpFormat] = None):
    """
    convert the input into a PyPTO Tensor

    Parameters
    ----------
    tensor: object
        The input tensor to be converted. Currently, supports PyTorch tensors.
    name: str
        The name of the resulting PyPTO Tensor.
    dynamic_axis: List[int]
        Specifies which axes of the tensor should be marked as dynamic.

    Returns
    -------
    Tensor
        A PyPTO Tensor object containing the following properties:
        - shape: The dimensions of the tensor.
        - name: The specified name of the tensor.
        - data_ptr: The memory address of the tensor data.
        - format: The format of the tensor (e.g., TILEOP_ND or  TILEOP_NZ).

    Examples
    --------
    >>> x= torch.randn(2, 3)
    >>> x_pto = pypto.from_torch(x)
    >>> print(x_pto.shape)
    [2, 3]
    >>> y = torch.randn(2, 3)
    >>> y_pto = pypto.from_torch(y, "input_tensor", dynamic_axis=[0])
    >>> print(y_pto.shape)
    [SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_input_tensor,0)), 3]
    >>> y = torch.randn(2, 3)
    >>> y_pto = pypto.from_torch(y, "input_tensor", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
    >>> print(y_pto.format)
    TileOpFormat.TILEOP_NZ
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("input type is not currently supported.")

    if not tensor.is_contiguous():
        raise RuntimeError("not all tensors are contiguous")

    if tensor_format is None:
        tensor_format = TileOpFormat.TILEOP_ND
        if tensor.device.type == "npu":
            import torch_npu

            if torch_npu.get_npu_format(tensor) == 29:
                tensor_format = TileOpFormat.TILEOP_NZ

    dtype = _dtype_from(tensor.dtype)
    if tensor.dim() == 0:
        return Tensor(
            shape=tuple([1]),
            dtype=dtype,
            name=name,
            data_ptr=tensor.data_ptr(),
            format=tensor_format,
            device=tensor.device,
        )
    dyn_shape = list(tensor.shape)
    if dynamic_axis is not None:
        for axis in dynamic_axis:
            dyn_shape[axis] = -1
    return Tensor(
        shape=dyn_shape,
        dtype=dtype,
        name=name,
        data_ptr=tensor.data_ptr(),
        format=tensor_format,
        device=tensor.device,
        ori_shape=list(tensor.shape),
    )


_dtype_dict = {
    "torch.float16": DataType.DT_FP16,
    "torch.bfloat16": DataType.DT_BF16,
    "torch.float32": DataType.DT_FP32,
    "torch.float64": DataType.DT_DOUBLE,
    "torch.int8": DataType.DT_INT8,
    "torch.uint8": DataType.DT_UINT8,
    "torch.int16": DataType.DT_INT16,
    "torch.uint16": DataType.DT_UINT16,
    "torch.int32": DataType.DT_INT32,
    "torch.uint32": DataType.DT_UINT32,
    "torch.int64": DataType.DT_INT64,
    "torch.uint64": DataType.DT_UINT64,
    "torch.bool": DataType.DT_BOOL,
}


def _dtype_from(dtype: str) -> DataType:
    pto_dtype = _dtype_dict.get(dtype.__str__())
    if pto_dtype is None:
        raise ValueError(f"Input torch.dtype is not supported. Got {dtype}")
    return pto_dtype


def _torch_dtype_from(dtype: DataType) -> "torch.dtype":
    """
    convert the input into a torch.dtype

    Parameters
    ----------
    dtype: DataType
        The input pypto.DataType to be converted.

    Returns
    -------
    torch.dtype
        The torch.dtype string.
    """
    import torch

    _torch_dtype_dict = {
        DataType.DT_FP16: torch.float16,
        DataType.DT_BF16: torch.bfloat16,
        DataType.DT_FP32: torch.float32,
        DataType.DT_DOUBLE: torch.float64,
        DataType.DT_INT8: torch.int8,
        DataType.DT_UINT8: torch.uint8,
        DataType.DT_INT16: torch.int16,
        DataType.DT_UINT16: torch.uint16,
        DataType.DT_INT32: torch.int32,
        DataType.DT_UINT32: torch.uint32,
        DataType.DT_INT64: torch.int64,
        DataType.DT_UINT64: torch.uint64,
        DataType.DT_BOOL: torch.bool,
    }

    torch_dtype = _torch_dtype_dict.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Input pypto.DataType is not supported. Got {dtype}")
    return torch_dtype


def _gen_pto_tensor(input_tensors):
    import torch
    
    torch_tensors = []
    pto_tensors = []
    for t in input_tensors:
        torch_dtype = _torch_dtype_from(t.dtype)
        tshape = t.shape if all([isinstance(s, int) for s in t.shape]) else t.ori_shape
        torch_tensor = torch.zeros(tshape, dtype=torch_dtype)
        pto_tensor = Tensor(shape=tshape,
                            dtype=t.dtype,
                            name=t.name,
                            data_ptr=torch_tensor.data_ptr(),
                            format=t.format,
                            device=torch_tensor.device,
                            ori_shape=tshape)

        torch_tensors.append(torch_tensor)
        pto_tensors.append(pto_tensor)
    return pto_tensors, torch_tensors


def ir_from_tensor(pto_tensor: Tensor):
    tensor_shape = []
    dim_num = 0
    for ele in pto_tensor.shape:
        dim = ir.Scalar(ir.DataType.int32, ele, "dim_" + str(dim_num))
        tensor_shape.append(dim)
        dim_num = dim_num + 1
    return ir.Tensor(tensor_shape, ir.DataType(int(pto_tensor.dtype)), \
        "_MACRO_" + pto_tensor.name, ir.Format((int(pto_tensor.format))))
