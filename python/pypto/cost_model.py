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
"""
"""
from typing import List, overload

import pypto
import torch

from . import pypto_impl
from .converter import _dtype_from, from_torch

__all__ = [
    "_cost_model_run_once_data_from_host",
]


def _pto_to_tensor_data(tensors: List[pypto.Tensor]) -> List[pypto_impl.DeviceTensorData]:
    datas = []
    for t in tensors:
        if t.ori_shape is None:
            raise RuntimeError("The ori_shape of the tensor is not specified.")
        data = pypto_impl.DeviceTensorData(
            t.dtype,
            t.data_ptr,
            list(t.ori_shape),
        )
        datas.append(data)
    return datas


def _cost_model_run_once_data_from_host(inputs: List[pypto.Tensor], outputs: List[pypto.Tensor]):
    isDevice = False
    for t in inputs:
        if t.device != torch.device("cpu"):
            isDevice = True
            break

    input_datas = _pto_to_tensor_data(inputs)
    output_datas = _pto_to_tensor_data(outputs)

    if isDevice:
        input_datas = [pypto_impl.CopyToHost(t) for t in input_datas]
        output_datas = [pypto_impl.CopyToHost(t) for t in output_datas]

    pypto_impl.CostModelRunOnceDataFromHost(input_datas, output_datas)
