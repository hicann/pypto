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
import os
from contextlib import contextmanager
from enum import IntEnum
from typing import List, overload

import pypto

from . import pypto_impl
from .converter import _gen_pto_tensor, from_torch

__all__ = [
    "_device_init",
    "_device_fini",
    "_device_run_once_data_from_host",
    "_device_synchronize",
    "jit",
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

    def reset(self):
        self._data = []

    def set_data(self, goldens):
        self._data = goldens

    def get_data(self):
        return self._data


_pto_verify_datas = _CachedVerifyData()


def _current_stream():
    import torch
    return torch.npu.current_stream().npu_stream


def _is_current_stream_capturing():
    import torch
    return torch.npu.is_current_stream_capturing()


@contextmanager
def _change_device(device):
    import torch
    ori_device = torch.npu.current_device()
    try:
        if device.index != ori_device:
            torch.npu.set_device(device.index)
        yield
    finally:
        if device.index != ori_device:
            torch.npu.set_device(ori_device)


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


class _ArgType:

    def __init__(self):
        self.argtype = []
        self.hash = 0

    def __eq__(self, other: '_ArgType') -> bool:
        return (self.hash == other.hash) and (self.argtype == other.argtype)

    def __hash__(self):
        return self.hash

    def __str__(self):
        return str(self.argtype)

    def append(self, shape, dtype):
        self.argtype.append((tuple(shape), dtype))
        self.hash = self.hash ^ hash(self.argtype[-1])


class _ControlflowShape:

    def __init__(self, shapes=None):
        self.shapes = [list(shape) for shape in shapes]
        self.hash = hash(tuple([tuple(shape) for shape in shapes]))

    def __eq__(self, other: '_ControlflowShape') -> bool:
        return (self.hash == other.hash) and (self.shapes == other.shapes)

    def __hash__(self):
        return self.hash

    def __str__(self):
        return str(self.shapes)


class _JIT:
    def __init__(self, dyn_func, codegen_options=None, host_options=None,
                 pass_options=None, runtime_options=None, verify_options=None,
                 debug_options=None, infer_controlflow_shape=None):
        self.dyn_func = dyn_func
        self.codegen_options = codegen_options
        self.host_options = host_options
        self.pass_options = pass_options
        self.runtime_options = runtime_options or {}
        self.verify_options = verify_options
        self.debug_options = debug_options
        self.infer_controlflow_shape = infer_controlflow_shape
        self.kernel_cache = {}
        self.controlflow_cache = {}

        # if infer cache shape supported, also use full cache mode
        if self.infer_controlflow_shape:
            # set to max cfgcache size 100000000
            self.runtime_options['stitch_cfgcache_size'] = 100000000

    def __call__(self, *args, **kwargs):
        if len(args) < 1:
            raise ValueError("at least one tensor is required")

        # all tensor must be on same device
        tensors = [item for item in args if isinstance(item, pypto.Tensor)]
        device = None
        for t in tensors:
            if device is None:
                device = t.device
            elif device != t.device:
                raise RuntimeError("not all tensors are on the same device")

        # if not set npu mode if cann available else cpu mode
        run_mode = self.set_run_mode()
        # kernel ptoto type
        argtype = self.get_argtype(tensors)
        # shape for build control_flow_cache
        cfshape = self.get_controlflow_shape(tensors)
        # kernel launch args
        start_args = _pto_to_tensor_data(tensors)

        with pypto.options("jit_scope"):
            self._set_config_option()
            self.kernel_warmup(device, tensors, argtype, *args, **kwargs)
            kernel, ctrcache = self.get_cached_kernel(device, tensors, argtype, cfshape, *args, **kwargs)
            if run_mode == RunMode.NPU:
                self.run_npu(device, kernel, ctrcache, start_args)
            else:
                self.run_cpu(kernel, tensors)

    @staticmethod
    def run_npu(device, kernel, ctrl_cache, start_args):
        import torch
        with _change_device(device):
            if device.type == 'npu':
                workspace_size = pypto_impl.GetWorkSpaceSize(kernel, start_args, [])
                workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
                pypto_impl.OperatorDeviceRunOnceDataFromDevice(kernel,
                    start_args, [], _current_stream(), workspace_tensor.data_ptr(), ctrl_cache)
            else:
                pypto_impl.DeviceRunOnceDataFromHost(start_args, [])

    @staticmethod
    def run_cpu(kernel, tensors):
        # call cost_model interface
        from .cost_model import _cost_model_run_once_data_from_host
        _cost_model_run_once_data_from_host(tensors, [])

    @staticmethod
    def get_argtype(tensors: List[pypto.Tensor]):
        argtype = _ArgType()
        for tensor in tensors:
            shape = [dim if isinstance(
                dim, int) else -1 for dim in tensor.shape]
            argtype.append(shape, tensor.dtype)
        return argtype

    @staticmethod
    def verify_end():
        _pto_verify_datas.reset()

    def get_controlflow_shape(self, tensors: List[pypto.Tensor]):
        if self.infer_controlflow_shape:
            inferred = self.infer_controlflow_shape(
                *[t.ori_shape for t in tensors])
            return _ControlflowShape(inferred) if inferred else None
        elif self.runtime_options.get('stitch_cfgcache_size', 0):
            return _ControlflowShape([t.ori_shape for t in tensors])
        else:
            return None

    def verify_begin(self, tensors):
        if isinstance(self.verify_options, dict) and self.verify_options.get("enable_pass_verify"):
            host_pto_tensors, _ = _gen_pto_tensor(tensors)
            host_pto_t_datas = _pto_to_tensor_data(host_pto_tensors)
            for i, dev_tensor in enumerate(_pto_to_tensor_data(tensors)):
                pypto_impl.CopyToHost(dev_tensor, host_pto_t_datas[i])
            pypto_impl.SetVerifyData(
                host_pto_t_datas, [], _pto_verify_datas.get_data())

    def compile(self, tensors, *args, **kwargs):
        pypto_impl.DeviceInit()
        # config is reset DeviceInit
        self._set_config_option()
        # flowverify begin
        self.verify_begin(tensors)

        handler = pypto_impl.OperatorBegin()
        with pypto.function(self.dyn_func.__name__, *tensors) as rlf:
            for _ in rlf:
                self.dyn_func(*args, **kwargs)
            del rlf
        pypto_impl.OperatorEnd(handler)

        # flowverify begin
        self.verify_end()
        # suspicious code?
        pypto_impl.ResetLog(pypto_impl.LogTopFolder())
        return handler

    def set_run_mode(self):
        is_cann_enable = bool(os.environ.get("ASCEND_HOME_PATH"))
        if "run_mode" in self.runtime_options:
            run_mode = RunMode(self.runtime_options["run_mode"])
        else:
            run_mode = RunMode.NPU if is_cann_enable else RunMode.SIM
        if run_mode == RunMode.NPU and not is_cann_enable:
            raise RuntimeError(
                "Please source cann environment while run mode is NPU.")
        self.runtime_options["run_mode"] = int(run_mode)
        return RunMode(run_mode)

    def kernel_warmup(self, device, tensors: List[pypto.Tensor], argtype, *args, **kwargs):
        if self.infer_controlflow_shape and not self.kernel_cache:
            for shape in self.infer_controlflow_shape():
                cfshape = _ControlflowShape(shape)
                self.get_cached_kernel(device, tensors, argtype, cfshape, *args, **kwargs)

    def get_cached_kernel(self, device, tensors: List[pypto.Tensor], argtype: _ArgType,
                          cfshape: _ControlflowShape, *args, **kwargs):
        kernel = self.kernel_cache.get(argtype)
        if kernel is None:
            kernel = self.compile(tensors, *args, **kwargs)
            self.kernel_cache[argtype] = kernel

        if not cfshape:
            return kernel, 0

        cfcache = None
        if device.type == 'npu':
            cfdata = [pypto_impl.DeviceTensorData(t.dtype, 0, shape) for t, shape in zip(tensors, cfshape.shapes)]
            cfcache = pypto_impl.BuildCache(kernel, cfdata, [], _is_current_stream_capturing())

        return kernel, cfcache

    def _set_config_option(self):
        if isinstance(self.codegen_options, dict):
            pypto.set_codegen_options(**self.codegen_options)

        if isinstance(self.host_options, dict):
            pypto.set_host_options(**self.host_options)

        if isinstance(self.pass_options, dict):
            pypto.set_pass_options(**self.pass_options)

        if isinstance(self.runtime_options, dict):
            pypto.set_runtime_options(**self.runtime_options)

        if isinstance(self.verify_options, dict):
            pypto.set_verify_options(**self.verify_options)

        if isinstance(self.debug_options, dict):
            pypto.set_debug_options(**self.debug_options)


@overload
def jit(dyn_func=None):
    ...


@overload
def jit(
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None,
        verify_options=None,
        debug_options=None,
        infer_controlflow_shape=None
):
    ...


def jit(dyn_func=None,
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None,
        verify_options=None,
        debug_options=None,
        infer_controlflow_shape=None):

    def decorator(func):
        return _JIT(func,
                    codegen_options=codegen_options,
                    host_options=host_options,
                    pass_options=pass_options,
                    runtime_options=runtime_options,
                    verify_options=verify_options,
                    debug_options=debug_options,
                    infer_controlflow_shape=infer_controlflow_shape)

    if dyn_func is not None:
        return _JIT(dyn_func)
    else:
        return decorator


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


def set_verify_golden_data(in_out_tensors=None, goldens=None):
    from .enum import DT_FP16
    pto_goldens = []
    if goldens:
        for golden in goldens:
            if golden is None:
                data = pypto_impl.DeviceTensorData(DT_FP16, 0, [0, 0])
                pto_goldens.append(data)
                continue
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
        _pto_verify_datas.set_data(pto_goldens)

    if in_out_tensors:
        pto_in_out = []
        for t in in_out_tensors:
            pto_in_out.append(t if isinstance(t, pypto.Tensor)
                              else pypto.from_torch(t))

        pypto_impl.SetVerifyData(_pto_to_tensor_data(pto_in_out),
                                 [], pto_goldens)
