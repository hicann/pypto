#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
from enum import IntEnum
from typing import List

import pypto
import torch

from . import pypto_impl
from .converter import from_torch

__all__ = [
    "_device_init",
    "_device_fini",
    "_device_run_once_data_from_host",
    "_device_synchronize",
    "verify",
    "set_verify_data",
]

_device_init = pypto_impl.DeviceInit
_device_fini = pypto_impl.DeviceFini


class RunMode(IntEnum):
    NPU = 0
    SIM = 1


class _CachedVerifyData:

    def __init__(self):
        self._data = []
        self._ori_data = []

    def reset(self):
        self._data = []
        self._ori_data = []

    def set_data(self, goldens, ori_goldens):
        self._data = goldens
        self._ori_data = ori_goldens

    def get_data(self):
        return self._data


_pto_verify_datas = _CachedVerifyData()


def _current_stream():
    npu = getattr(torch, 'npu', None)
    if npu:
        return npu.current_stream().npu_stream
    else:
        return 0


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


def _device_run_once_data_from_host(*args):
    for i, t in enumerate(args):
        if not isinstance(t, pypto.Tensor):
            raise TypeError(f"Expected a pypto.Tensor at inputs[{i}], but got {type(t).__name__}.")
    pypto_impl.DeviceRunOnceDataFromHost(
        _pto_to_tensor_data(args), [])


def _device_synchronize():
    pypto_impl.OperatorDeviceSynchronize(_current_stream())


def verify(func, inputs, outputs, goldens, *args,
           codegen_options=None,
           host_options=None,
           pass_options=None,
           verify_options=None, **kwargs):
    """
    Verify the tensor graph of the function.

    Args:
        func: The function to verify.
        inputs: The input tensors.
        outputs: The output tensors.
        goldens: The golden tensors.
        *args: The extra arguments for func.
        verify_options: dict
            see :func:`set_verify_options`.
        codegen_options: dict
            see :func:`set_codegen_options`.
        host_options: dict
            see :func:`set_host_options`.
        pass_options: dict
            see :func:`set_pass_options`.
        **kwargs: The extra keyword arguments for func.
    Returns:
        None
    """
    pypto_impl.DeviceInit()

    if pass_options is None:
        pass_options = {}
    pypto.set_pass_options(**pass_options)

    if verify_options is None:
        verify_options = {"enable_pass_verify": True}
    pypto.set_verify_options(**verify_options)

    pypto_impl.SetVerifyData(_pto_to_tensor_data(inputs),
                             _pto_to_tensor_data(outputs),
                             _pto_to_tensor_data(goldens))

    inputs = [from_torch(t, f"IN_{idx}") for idx, t in enumerate(inputs)]
    outputs = [from_torch(t, f"OUT_{idx}") for idx, t in enumerate(outputs)]
    handler = pypto_impl.OperatorBegin()
    func(inputs, outputs, *args, **kwargs)
    pypto_impl.OperatorEnd(handler)


def _check_tensor_on_cpu(tensor, kind: str, index: int):
    """Raise if tensor (torch.Tensor or pypto.Tensor) is on device; must be CPU-side."""
    if isinstance(tensor, torch.Tensor):
        if tensor.device.type != 'cpu':
            raise ValueError(
                f"set_verify_golden_data: {kind} must be on CPU, but {kind}[{index}] "
                f"(torch.Tensor) is on device '{tensor.device}'. Please call .cpu() before passing."
            )
        return
    if isinstance(tensor, pypto.Tensor) and tensor.device is not None:
        if getattr(tensor.device, 'type', None) != 'cpu':
            raise ValueError(
                f"set_verify_golden_data: {kind} must be on CPU, but {kind}[{index}] "
                f"(pypto.Tensor) is on device '{tensor.device}'. Please use CPU data, e.g. pypto.from_torch(t.cpu())."
            )


def set_verify_golden_data(in_out_tensors=None, goldens=None):
    from .enum import DT_FP16
    pto_goldens = []
    if goldens:
        for idx, golden in enumerate(goldens):
            if golden is None:
                data = pypto_impl.DeviceTensorData(DT_FP16, 0, [0, 0])
                pto_goldens.append(data)
                continue
            _check_tensor_on_cpu(golden, "goldens", idx)
            if not isinstance(golden, pypto.Tensor):
                t = pypto.from_torch(golden)
            else:
                t = golden

            data = pypto_impl.DeviceTensorData(
                t.dtype,
                t.data_ptr,
                list(t.ori_shape),
            )
            pto_goldens.append(data)
        _pto_verify_datas.set_data(pto_goldens, goldens)

    if in_out_tensors:
        pto_in_out = []
        for idx, t in enumerate(in_out_tensors):
            _check_tensor_on_cpu(t, "in_out_tensors", idx)
            pto_in_out.append(t if isinstance(t, pypto.Tensor)
                              else pypto.from_torch(t))

        pypto_impl.SetVerifyData(_pto_to_tensor_data(pto_in_out),
                                 [], pto_goldens)
