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
import os

from enum import Enum
from . import pypto_impl
from .converter import _dtype_from, from_torch, _gen_pto_tensor

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


class RunMode(Enum):
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


def _set_device(device: int):
    import torch
    torch.npu.set_device(device)


def _current_device() -> int:
    import torch
    return torch.npu.current_device()


def _current_stream():
    import torch
    return torch.npu.current_stream().npu_stream


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
    in_out_tensors = [item for item in args]
    for i, inp in enumerate(in_out_tensors):
        if not isinstance(inp, pypto.Tensor):
            raise TypeError(
                f"Expected pypto.Tensor at inputs[{i}], "f"but got {type(inp).__name__}. "
                "Use from_torch() to convert torch.Tensor to pypto.Tensor."
            )
    pypto_impl.DeviceRunOnceDataFromHost(
        _pto_to_tensor_data(in_out_tensors), [])


def _compute_tensor_hash(tensors, tensor_data):
    if (len(tensors) != len(tensor_data)):
        raise RuntimeError("The number of tensors does not match the number of tensor_data.")
    hash_list = []
    for tensor, t_data in zip(tensors, tensor_data):
        shape = tuple([dim if isinstance(dim, int) else -1 for dim in tensor.shape])
        real_shape = tuple(t_data.GetShape())
        dtype = tensor.dtype
        hash_list.append(tuple([shape, real_shape, dtype]))
    comupted_hash = tuple(hash_list)
    return comupted_hash


class _JIT:
    def __init__(self, dyn_func, codegen_options=None, host_options=None,
                 pass_options=None, runtime_options=None, verify_options=None, debug_options=None):
        self.dyn_func = dyn_func
        self._is_compiled: bool = False
        self._handler = None
        self._handler_cache = {}
        self.codegen_options = codegen_options
        self.host_options = host_options
        self.pass_options = pass_options
        self.runtime_options = runtime_options
        self.verify_options = verify_options
        self.debug_options = debug_options

    def compile(self, *args, **kwargs):
        pypto_impl.DeviceInit()
        # config is reset DeviceInit
        self._set_config_option()
        in_out_tensors = [item for item in args if isinstance(item, pypto.Tensor)]

        if isinstance(self.verify_options, dict) and self.verify_options.get("enable_pass_verify"):
            host_pto_tensors, _ = _gen_pto_tensor(in_out_tensors)
            host_pto_t_datas = _pto_to_tensor_data(host_pto_tensors)
            for i, dev_tensor in enumerate(_pto_to_tensor_data(in_out_tensors)):
                pypto_impl.CopyToHost(dev_tensor, host_pto_t_datas[i])
            pypto_impl.SetVerifyData(host_pto_t_datas, [], _pto_verify_datas.get_data())

        handler = pypto_impl.OperatorBegin()
        self._set_config_option()
        with pypto.function(self.dyn_func.__name__, *in_out_tensors) as rlf:
            for _ in rlf:
                self.dyn_func(*args, **kwargs)
            del rlf
        pypto_impl.OperatorEnd(handler)

        _pto_verify_datas.reset()

        self._handler = handler
        self._is_compiled = True

    def run(self, in_tensor_data, out_tensor_data, device):
        import torch
        assert self._handler is not None
        workspace_size = pypto_impl.GetWorkSpaceSize(self._handler, in_tensor_data, out_tensor_data)
        workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        pypto_impl.OperatorDeviceRunOnceDataFromDevice(
            self._handler,
            in_tensor_data,
            out_tensor_data,
            _current_stream(),
            workspace_tensor.data_ptr())

    def run_with_npu(self, inputs, outputs, device):
        if device.type == 'cpu':
            _device_run_once_data_from_host(*inputs, *outputs)
        elif device.type == 'npu':
            import torch_npu
            in_tensor_data = _pto_to_tensor_data(inputs)
            out_tensor_data = _pto_to_tensor_data(outputs)
            ori_device = _current_device()
            if device and device.index != ori_device:
                _set_device(device.index)
                self.run(in_tensor_data, out_tensor_data, device)
                _set_device(ori_device)
            else:
                self.run(in_tensor_data, out_tensor_data, device)

    def run_with_cpu(self, in_tensor_data, out_tensor_data):
        # call cost_model interface
        from .cost_model import _cost_model_run_once_data_from_host
        _cost_model_run_once_data_from_host(in_tensor_data, out_tensor_data)
        return


    def dispatch_with_run_mode(self, in_tensor_data, out_tensor_data, device):
        cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))
        run_mode = pypto.get_runtime_options().get('run_mode', 0)
        if run_mode == 0:
            if cann_is_configed == False:
                raise RuntimeError("Please source cann environment while run mode is NPU.")
            self.run_with_npu(in_tensor_data, out_tensor_data, device)
        else:
            self.run_with_cpu(in_tensor_data, out_tensor_data)

    def set_run_mode(self):
        if self.runtime_options is None:
            self.runtime_options = {}

        run_mode = self.runtime_options.get("run_mode", None)
        if run_mode is not None:
            if run_mode not in [RunMode.NPU, RunMode.SIM, 0, 1]:
                raise RuntimeError("Invalid run mode, run mode must be RunMode.NPU or RunMode.SIM.")
            else:
                if isinstance(run_mode, RunMode):
                    self.runtime_options.update({"run_mode": run_mode.value})
                return

        cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))
        if cann_is_configed:
            self.runtime_options.update({"run_mode": RunMode.NPU.value})
        else:
            self.runtime_options.update({"run_mode": RunMode.SIM.value})

    def __call__(self, *args, **kwargs):
        if len(args) < 1:
            raise ValueError("at least one tensor is required")
        device = None
        in_out_tensors = [item for item in args if isinstance(item, pypto.Tensor)]

        for t in in_out_tensors:
            if device is None:
                device = t.device
            elif device != t.device:
                raise RuntimeError("not all tensors are on the same device")

        # Convert tensors to tensor data before compile, as compile turns tensor shapes into symbolic scalars.
        in_out_tensors_data = _pto_to_tensor_data(in_out_tensors)
        input_hash = _compute_tensor_hash(in_out_tensors, in_out_tensors_data)

        self.set_run_mode()
        with pypto.options("jit_scope"):
            self._set_config_option()
            if not self._is_compiled or not self._hit_cache(input_hash):
                self.compile(*args, **kwargs)
                self._handler_cache[input_hash] = self._handler
                pypto_impl.BuildCache(self._handler, in_out_tensors_data, [])
            else:
                pypto_impl.ResetLog()
                self._handler = self._handler_cache.get(input_hash)
            # dispatch run mode based on ASCEND_HOME_PATH or run_mode
            '''
              if run_mode is not config, use ASCEND_HOME_PATH
              when ASCEND_HONE_PATH is config, run on with npu
              when ASCEND_HONE_PATH is not config, run on with simulator

              if run_mode is configed, use run_mode
              if run_mode is npu , check env, than run with differnet tensor type (support cpu or npu)
              if run_mode is simulator, dont check env, change all tensor to cpu, and run
            '''
            self.dispatch_with_run_mode(in_out_tensors, [], device)

    @property
    def handler(self):
        return self._handler


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

    def _hit_cache(self, input_hash):
        if self._handler is None or len(self._handler_cache) == 0:
            return False
        return self._handler_cache.get(input_hash) is not None


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
        debug_options=None
):
    ...


def jit(dyn_func=None,
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None,
        verify_options=None,
        debug_options=None):

    def decorator(func):
        return _JIT(func,
                   codegen_options=codegen_options,
                   host_options=host_options,
                   pass_options=pass_options,
                   runtime_options=runtime_options,
                   verify_options=verify_options,
                   debug_options=debug_options)

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

    if host_options is None:
        host_options = {"only_codegen": True}
    pypto.set_host_options(**host_options)

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
            pto_in_out.append(t if isinstance(t, pypto.Tensor) else pypto.from_torch(t))

        pypto_impl.SetVerifyData(_pto_to_tensor_data(pto_in_out),
                                 [], pto_goldens)
