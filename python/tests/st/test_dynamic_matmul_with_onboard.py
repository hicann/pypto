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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import pypto
from numpy.testing import assert_allclose
import torch_npu

FP32 = np.float32
FP16 = np.float16
INT32 = np.int32
INT8 = np.int8
UINT64 = np.uint64
UINT32 = np.uint32


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: np.dtype
    out_dtype: np.dtype
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None



def test_matmul_bf16_with_n_split():
    input_config = ShapeConfig([16, 32, 512], [128, 128], [128, 128], [128, 128], [-1, 100], FP16, FP32,
                               True, True, False, False, False)
    dynamic_matmul_onboard_util(input_config)


def test_matmul_bf16_nd_with_no_split():
    input_config = ShapeConfig([127, 255, 511], [128, 128], [128, 128], [128, 128], [-1, -1], FP16, FP32, True, True,
                               False, False, False)
    dynamic_matmul_onboard_util(input_config)


def test_matmul_bias_with_no_split():
    input_config = ShapeConfig([127, 255, 511], [128, 128], [128, 128], [128, 128], [-1, -1], FP16, FP32, True, True,
                               False, False, False)
    extend_params = ExtendParams([1, 511], FP16)
    dynamic_matmul_onboard_util(input_config, extend_params)


def dynamic_matmul_onboard_util(input_config: ShapeConfig, extend_params: Optional[ExtendParams] = None):
    # onboard prepare
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()
    pypto.set_cube_tile_shapes(input_config.m_tile_shape, input_config.k_tile_shape, input_config.n_tile_shape)

    tensor_a = create_tensor(input_config.in_dtype, input_config.ori_shape, "tensor_a", input_config.a_format_nz,
                             input_config.a_trans)
    tensor_b = create_tensor(input_config.in_dtype, input_config.ori_shape, "tensor_b", input_config.b_format_nz,
                             input_config.b_trans)
    tensor_c = create_tensor(input_config.out_dtype, input_config.ori_shape, "tensor_c", input_config.c_format_nz)

    view_m = input_config.view_shape[0]
    view_n = input_config.view_shape[1]
    if extend_params is None:
        if view_m > 0 and view_n > 0:
            split_m_n_axis(tensor_a, tensor_b, tensor_c, input_config)
        elif view_m > 0:
            split_m_axis(tensor_a, tensor_b, tensor_c, input_config)
        elif view_n > 0:
            split_n_axis(tensor_a, tensor_b, tensor_c, input_config)
        else:
            no_split_m_n(tensor_a, tensor_b, tensor_c, input_config)
    else:
        no_split_m_n_with_extend_param(tensor_a, tensor_b, tensor_c, input_config, extend_params)

    # gen golden
    a_data, b_data, c_data, bias, c_device_data = gen_matmul_golden_data(input_config, extend_params)

    pto_a_tensor = pypto.from_torch(a_data, "a_data")
    pto_b_tensor = pypto.from_torch(b_data, "b_data")
    pto_c_device_tensor = pypto.from_torch(c_device_data, "c_device_data")
    # onboard execute
    if extend_params is None:
        pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_device_tensor)
    if bias is not None:
        pto_bias_tensor = pypto.from_torch(bias, "bias_data")
        pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_bias_tensor,
                                                        pto_c_device_tensor)

    # compare golden with onboard data
    assert_allclose(c_data, c_device_data, rtol=0.001, atol=0.001)

    # onboard finish--clean env
    pypto.runtime._device_fini()


def no_split_m_n(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    valid_shape_a = [shape_a[0], shape_a[1]]
    valid_shape_b = [shape_b[0], shape_b[1]]
    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    with pypto.function("test_no_split", tensor_a, tensor_b, tensor_c):
        for idx in pypto.loop(1, name="loop", idx_name="idx"):
            dyn_a = pypto.view(tensor_a, shape_a, [idx, 0], valid_shape=valid_shape_a)
            dyn_b = pypto.view(tensor_b, shape_b, [0, 0], valid_shape=valid_shape_b)
            tensor_c.move(pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans,
                            b_trans=input_config.b_trans, c_matrix_nz=input_config.c_format_nz))


def no_split_m_n_with_extend_param(tensor_a, tensor_b, tensor_c, input_config, extend_params):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    tensor_bias = pypto.tensor()
    dyn_extend_params = {}
    tensor_bias = create_tensor(extend_params.bias_dtype, extend_params.bias_shape, "tensor_bias", False)
    valid_shape_a = [shape_a[0], shape_a[1]]
    valid_shape_b = [shape_b[0], shape_b[1]]

    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    with pypto.function("test_no_split", tensor_a, tensor_b, tensor_bias, tensor_c):
        for idx in pypto.loop(1, name="loop", idx_name="idx"):
            dyn_a = pypto.view(tensor_a, shape_a, [idx, 0], valid_shape=valid_shape_a)
            dyn_b = pypto.view(tensor_b, shape_b, [0, 0], valid_shape=valid_shape_b)
            shape_bias = tensor_bias.shape
            valid_shape_bias = [shape_bias[0], shape_bias[1]]
            dyn_bias = pypto.view(tensor_bias, shape_bias, [0, 0], valid_shape=valid_shape_bias)
            dyn_extend_params['bias_tensor'] = dyn_bias
            tensor_c.move(pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans,
                        b_trans=input_config.b_trans, c_matrix_nz=input_config.c_format_nz,
                        extend_params=dyn_extend_params))


def split_m_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.shape
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans

    m_axis = shape_a[1] if a_trans else shape_a[0]
    loop_end = ceil_div_util(m_axis, view_shape[0])

    with pypto.function("test_m_split", tensor_a, tensor_b, tensor_c):
        for m_idx in pypto.loop(0, loop_end, 1, name="m_loop", idx_name="m_idx"):
            matmul_split_m_utils(tensor_a, tensor_b, tensor_c, input_config, m_idx)


def matmul_split_m_utils(tensor_a, tensor_b, tensor_c, input_config, m_idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans
    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    if a_trans:
        dyn_a = pypto.view(tensor_a, [shape_a[0], view_shape[0]],
                    [0, m_idx * view_shape[0]],
                    valid_shape=[shape_a[0], (shape_a[1] - m_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))])
    else:
        dyn_a = pypto.view(tensor_a, [view_shape[0], shape_a[1]],
                    [m_idx * view_shape[0], 0],
                    valid_shape=[(shape_a[0] - m_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])), shape_a[1]])

    dyn_b = pypto.view(tensor_b, shape_b, [0, 0], valid_shape=[shape_b[0], shape_b[1]])
    res = pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                            c_matrix_nz=input_config.c_format_nz)

    pypto.assemble(res, [m_idx * view_shape[0], 0], tensor_c)


def split_n_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans

    n_axis = shape_b[0] if b_trans else shape_b[1]
    loop_end = ceil_div_util(n_axis, view_shape[1])

    with pypto.function("test_n_split", tensor_a, tensor_b, tensor_c):
        for n_idx in pypto.loop(0, loop_end, 1, name="n_loop", idx_name="n_idx"):
            matmul_split_n_utils(tensor_a, tensor_b, tensor_c, input_config, n_idx)



def matmul_split_n_utils(tensor_a, tensor_b, tensor_c, input_config, n_idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans
    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)
    dyn_a = pypto.view(tensor_a, shape_a, [0, 0], valid_shape=[shape_a[0], shape_a[1]])
    if b_trans:
        dyn_b = pypto.view(tensor_b, [view_shape[1], shape_b[1]],
        [n_idx * view_shape[1], 0],
        valid_shape=[(shape_b[0] - n_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])), shape_b[1]])
    else:
        dyn_b = pypto.view(tensor_b, [shape_b[0], view_shape[1]],
        [0, n_idx * view_shape[1]],
        valid_shape=[(shape_b[0], shape_b[1] - n_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1]))])

    res = pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                c_matrix_nz=input_config.c_format_nz)

    pypto.assemble(res, [0, n_idx * view_shape[1]], tensor_c)



def split_m_n_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.shape
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans

    m_axis = shape_a[1] if a_trans else shape_a[0]
    m_loop_end = ceil_div_util(m_axis, view_shape[0])

    with pypto.function("test_m_n_split", tensor_a, tensor_b, tensor_c):
        for m_idx in pypto.loop(0, m_loop_end, 1, name="m_loop", idx_name="m_idx"):
            matmul_split_m_n_util(tensor_a, tensor_b, tensor_c, input_config, m_idx)


def matmul_split_m_n_util(tensor_a, tensor_b, tensor_c, input_config, m_idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans
    a_trans = input_config.a_trans

    n_axis = shape_b[0] if b_trans else shape_b[1]
    n_loop_end = ceil_div_util(n_axis, view_shape[1])

    dtype = convert_np_dtype_to_pto_dtype(input_config.out_dtype)

    for n_idx in pypto.loop(0, n_loop_end, 1, name="n_loop", idx_name="n_idx"):
        if a_trans:
            dyn_a = pypto.view(tensor_a, [shape_a[0], view_shape[0]],
                        [0, m_idx * view_shape[0]],
                        valid_shape=[shape_a[0], (shape_a[1] - m_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0]))])
        else:
            dyn_a = pypto.view(tensor_a, [view_shape[0], shape_a[1]],
                        [m_idx * view_shape[0], 0],
                        valid_shape=[(shape_a[0] - m_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])), shape_a[1]])
        if not b_trans:
            dyn_b = pypto.view(tensor_b, [shape_b[0], view_shape[1]],
                        [0, n_idx * view_shape[1]],
                        valid_shape=[(shape_b[0], shape_b[1] - n_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1]))])
        else:
            dyn_b = pypto.view(tensor_b, [view_shape[1], shape_b[1]],
                        [n_idx * view_shape[1], 0],
                        valid_shape=[(shape_b[0] - n_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])), shape_b[1]])
        res = pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                            c_matrix_nz=input_config.c_format_nz)

        pypto.assemble(res, [m_idx * view_shape[0], n_idx * view_shape[1]], tensor_c)


def convert_np_dtype_to_pto_dtype(dtype):
    if dtype == INT8:
        return pypto.DT_INT8
    elif dtype == FP16:
        return pypto.DT_FP16
    elif dtype == INT32:
        return pypto.DT_INT32
    elif dtype == FP32:
        return pypto.DT_FP32
    elif dtype == UINT64:
        return pypto.DT_UINT64
    else:
        assert False, "pypto dtype not found in matmul"


def create_tensor(dtype, ori_shape, tensor_name, format_nz, transposed=None):
    if tensor_name in ["tensor_bias", "tensor_scale"]:
        shape = (ori_shape[0], ori_shape[1])
        return pypto.tensor(shape, convert_np_dtype_to_pto_dtype(dtype), tensor_name)
    m = ori_shape[0]
    k = ori_shape[1]
    n = ori_shape[2]
    if tensor_name == "tensor_c":
        shape = (m, n)
    elif tensor_name == "tensor_b":
        shape = (n, k) if transposed else (k, n)
    elif tensor_name == "tensor_a":
        shape = (k, m) if transposed else (m, k)
    else:
        assert False, "tensor name not found in matmul"
    if format_nz:
        return pypto.tensor(shape, convert_np_dtype_to_pto_dtype(dtype), tensor_name, pypto.TileOpFormat.TILEOP_NZ)
    else:
        return pypto.tensor(shape, convert_np_dtype_to_pto_dtype(dtype), tensor_name)


def ceil_div_util(a, b):
    return a if b == 0 else (a + b - 1) // b


def nd_trans_to_fractal_nz(data: np.ndarray):
    ori_shape = data.shape
    ori_m, ori_n = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_padding = ((0, 0),) * len(batch_ori)
    if data.dtype == INT8:
        m0, n0 = 16, 32
    elif data.dtype == FP16 or data.dtype == INT32:
        m0, n0 = 16, 16
    elif data.dtype == FP32:
        m0, n0 = 16, 8

    m_data = ceil_div_util(ori_m, m0)
    n_data = ceil_div_util(ori_n, n0)
    padding_m = m_data * m0 - ori_m
    padding_n = n_data * n0 - ori_n
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant")
    offset = len(data.shape) - 2
    array_trans = [x for x in range(offset)] + [x + offset for x in [2, 0, 1, 3]]
    data = data.reshape(batch_ori + (m_data, m0, n_data, n0)).transpose(*array_trans)
    return data


def gen_matmul_golden_data(input_config: ShapeConfig, extend_params: Optional[ExtendParams] = None):
    ori_shape = input_config.ori_shape
    shape_a = [ori_shape[0], ori_shape[1]]
    shape_b = [ori_shape[1], ori_shape[2]]
    shape_bias = []
    bais_tensor = None
    if extend_params is not None and extend_params.bias_shape:
        shape_bias = [extend_params.bias_shape[0], extend_params.bias_shape[1]]

    if input_config.in_dtype == INT8:
        a = np.random.randint(-4, 5, shape_a).astype(INT8)
        b = np.random.randint(-4, 5, shape_b).astype(INT8)
        c = np.matmul(a.astype(INT32), b.astype(INT32)).astype(INT32)
    elif input_config.in_dtype in [FP16, FP32]:
        a = np.random.uniform(-1, 1, shape_a).astype(input_config.in_dtype)
        b = np.random.uniform(-1, 1, shape_b).astype(input_config.in_dtype)
        if extend_params is not None and extend_params.bias_shape:
            bias = np.random.uniform(-1, 1, shape_bias).astype(extend_params.bias_dtype)
            c = (np.matmul(a.astype(FP32), b.astype(FP32)) + bias.astype(FP32)).astype(input_config.out_dtype)
            bais_tensor = torch.from_numpy(bias.copy())
        if extend_params is None:
            c = np.matmul(a.astype(FP32), b.astype(FP32)).astype(input_config.out_dtype)
    else:
        assert False, "golden dtype not found"

    if input_config.a_trans:
        a = a.transpose(1, 0)
    if input_config.a_format_nz:
        a = nd_trans_to_fractal_nz(a)

    if input_config.b_trans:
        b = b.transpose(1, 0)
    if input_config.b_format_nz:
        b = nd_trans_to_fractal_nz(b)

    if input_config.c_format_nz:
        c = nd_trans_to_fractal_nz(c)

    a_tensor = torch.from_numpy(a.copy())
    b_tensor = torch.from_numpy(b.copy())
    c_tensor = torch.from_numpy(c.copy())
    c_device_tensor = torch.zeros_like(c_tensor)

    return a_tensor, b_tensor, c_tensor, bais_tensor, c_device_tensor
