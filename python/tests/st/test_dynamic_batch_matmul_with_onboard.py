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
from dataclasses import dataclass

import numpy as np
import torch
import pypto
from numpy.testing import assert_allclose
import torch_npu


@dataclass
class BatchMatmulShapeConfig:
    in_dtype: np.dtype
    out_dtype: np.dtype
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False


FP32 = np.float32
FP16 = np.float16
INT32 = np.int32
INT8 = np.int8


def test_batch_matmul_fp16_with_no_split():
    input_config = BatchMatmulShapeConfig(FP16, FP32, [3, 64, 128, 512], [128, 128], [128, 128], [128, 128],
                                             [-1, -1, -1], True, False, False, False, False)
    dynamic_batch_matmul_onboard_util(input_config)


def test_batch_matmul_bf16_with_m_split():
    input_config = BatchMatmulShapeConfig(FP16, FP32, [2, 16, 512, 128], [128, 128], [128, 128], [128, 128],
                                            [-1, -1, -1], True, True, False, True, False)
    dynamic_batch_matmul_onboard_util(input_config)


def dynamic_batch_matmul_onboard_util(input_config: BatchMatmulShapeConfig):
    # onboard prepare
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()
    pypto.set_cube_tile_shapes(input_config.m_tile_shape, input_config.k_tile_shape, input_config.n_tile_shape)

    tensor_a = batch_matmu_create_tensor(input_config.in_dtype, input_config.ori_shape, "tensor_a",
                                            input_config.a_format_nz, input_config.a_trans)
    tensor_b = batch_matmu_create_tensor(input_config.in_dtype, input_config.ori_shape, "tensor_b",
                                            input_config.b_format_nz, input_config.b_trans)
    tensor_c = batch_matmu_create_tensor(input_config.out_dtype, input_config.ori_shape, "tensor_c",
                                            input_config.c_format_nz)

    view_m = input_config.view_shape[1]
    view_n = input_config.view_shape[2]

    if view_m > 0 and view_n > 0:
        split_m_n_axis(tensor_a, tensor_b, tensor_c, input_config)
    elif view_m > 0:
        split_m_axis(tensor_a, tensor_b, tensor_c, input_config)
    elif view_n > 0:
        split_n_axis(tensor_a, tensor_b, tensor_c, input_config)
    else:
        no_split_m_n(tensor_a, tensor_b, tensor_c, input_config)

    # gen golden
    a_data, b_data, c_data, c_device_data = gen_batch_matmul_golden_data(input_config)

    pto_a_tensor = pypto.from_torch(a_data, "a_data")
    pto_b_tensor = pypto.from_torch(b_data, "b_data")
    pto_c_device_tensor = pypto.from_torch(c_device_data, "c_device_data")
    # onboard execute
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_device_tensor)

    # compare golden with onboard data
    assert_allclose(c_data, c_device_data, rtol=0.001, atol=0.001)

    # onboard finish--clean env
    pypto.runtime._device_fini()


def no_split_m_n(tensor_a, tensor_b, tensor_c, input_config):
    with pypto.function("test_no_split", tensor_a, tensor_b, tensor_c):
        for idx in pypto.loop(1, name="loop", idx_name="idx"):
            batch_matmul_no_split_util(tensor_a, tensor_b, tensor_c, input_config, idx)


def batch_matmul_no_split_util(tensor_a, tensor_b, tensor_c, input_config, idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    valid_shape_a = [shape_a[0], shape_a[1], shape_a[2]]
    valid_shape_b = [shape_b[0], shape_b[1], shape_b[2]]
    dtype = batch_matmul_convert_dtype(input_config.out_dtype)
    if input_config.a_format_nz or input_config.b_format_nz or input_config.c_format_nz:
        pypto.set_matrix_size([input_config.ori_shape[1], input_config.ori_shape[2], input_config.ori_shape[3]])
    dyn_a = pypto.view(tensor_a, shape_a, [0, idx, 0], valid_shape=valid_shape_a)
    dyn_b = pypto.view(tensor_b, shape_b, [0, 0, 0], valid_shape=valid_shape_b)
    tensor_c.move(pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                c_matrix_nz=input_config.c_format_nz))


def split_m_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.shape
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans

    m_axis = shape_a[2] if a_trans else shape_a[1]
    loop_end = ceil_div(m_axis, view_shape[1])

    with pypto.function("test_m_split", tensor_a, tensor_b, tensor_c):
        for m_idx in pypto.loop(0, loop_end, 1, name="m_loop", idx_name="m_idx"):
            batch_matmul_split_m_util(tensor_a, tensor_b, tensor_c, input_config, m_idx)


def batch_matmul_split_m_util(tensor_a, tensor_b, tensor_c, input_config, m_idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans
    if input_config.a_format_nz or input_config.b_format_nz or input_config.c_format_nz:
        pypto.set_matrix_size([input_config.ori_shape[1], input_config.ori_shape[2], input_config.ori_shape[3]])
    dtype = batch_matmul_convert_dtype(input_config.out_dtype)
    if a_trans:
        dyn_a = pypto.view(tensor_a, [shape_a[0], shape_a[1], view_shape[1]],
                    [0, 0, m_idx * view_shape[0]],
                    valid_shape=[shape_a[0], shape_a[1], (shape_a[2] - m_idx *
                    view_shape[1]).min(pypto.symbolic_scalar(view_shape[1]))])
    else:
        dyn_a = pypto.view(tensor_a, [shape_a[0], view_shape[1], shape_a[2]],
                    [0, 0, m_idx * view_shape[0]],
                    valid_shape=[shape_a[0], (shape_a[1] - m_idx *
                    view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])), shape_a[2]])

    dyn_b = pypto.view(tensor_b, shape_b, [0, 0, 0],
                    valid_shape=[shape_b[0], shape_b[1], shape_b[2]])
    res = pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                            c_matrix_nz=input_config.c_format_nz)

    pypto.assemble(res, [0, m_idx * view_shape[1], 0], tensor_c)


def split_n_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans

    n_axis = shape_b[1] if b_trans else shape_b[2]
    loop_end = ceil_div(n_axis, view_shape[2])

    dtype = batch_matmul_convert_dtype(input_config.out_dtype)
    with pypto.function("test_n_split", tensor_a, tensor_b, tensor_c):
        for n_idx in pypto.loop(0, loop_end, 1, name="n_loop", idx_name="n_idx"):
            batch_matmul_split_n_utils(tensor_a, tensor_b, tensor_c, input_config, n_idx)


def batch_matmul_split_n_utils(tensor_a, tensor_b, tensor_c, input_config, n_idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans
    dtype = batch_matmul_convert_dtype(input_config.out_dtype)
    if input_config.a_format_nz or input_config.b_format_nz or input_config.c_format_nz:
        pypto.set_matrix_size([input_config.ori_shape[1], input_config.ori_shape[2], input_config.ori_shape[3]])
    dyn_a = pypto.view(tensor_a, shape_a, [0, 0, 0],
                     valid_shape=[shape_a[0], shape_a[1], shape_a[2]])
    if b_trans:
        dyn_b = pypto.view(tensor_b, [shape_b[0], view_shape[2], shape_b[2]],
        [0, n_idx * view_shape[1], 0],
        valid_shape=[shape_b[0], (shape_b[1] - n_idx * view_shape[2]).min(pypto.symbolic_scalar(view_shape[2])),
            shape_b[2]])
    else:
        dyn_b = pypto.view(tensor_b, [shape_b[0], shape_b[1], view_shape[2]],
        [0, 0, n_idx * view_shape[1]],
        valid_shape=[(shape_b[0], shape_b[1],
            shape_b[2] - n_idx * view_shape[2]).min(pypto.symbolic_scalar(view_shape[2]))])

    res = pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                c_matrix_nz=input_config.c_format_nz)

    pypto.assemble(res, [0, 0, n_idx * view_shape[2]], tensor_c)


def split_m_n_axis(tensor_a, tensor_b, tensor_c, input_config):
    shape_a = tensor_a.shape
    view_shape = input_config.view_shape
    a_trans = input_config.a_trans

    m_axis = shape_a[2] if a_trans else shape_a[1]
    m_loop_end = ceil_div(m_axis, view_shape[1])

    dtype = batch_matmul_convert_dtype(input_config.out_dtype)
    with pypto.function("test_batch_m_n_split", tensor_a, tensor_b, tensor_c):
        for m_idx in pypto.loop(0, m_loop_end, 1, name="m_loop", idx_name="m_idx"):
            batch_matmul_split_m_n_utils(tensor_a, tensor_b, tensor_c, input_config, m_idx)


def batch_matmul_split_m_n_utils(tensor_a, tensor_b, tensor_c, input_config, m_idx):
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    view_shape = input_config.view_shape
    b_trans = input_config.b_trans
    a_trans = input_config.a_trans

    n_axis = shape_b[1] if b_trans else shape_b[2]
    n_loop_end = ceil_div(n_axis, view_shape[2])

    dtype = batch_matmul_convert_dtype(input_config.out_dtype)
    for n_idx in pypto.loop(0, n_loop_end, 1, name="n_loop", idx_name="n_idx"):
        if input_config.a_format_nz or input_config.b_format_nz or input_config.c_format_nz:
            pypto.set_matrix_size([input_config.ori_shape[1], input_config.ori_shape[2], input_config.ori_shape[3]])
        if not a_trans:
            dyn_a = pypto.view(tensor_a, [shape_a[0], view_shape[1], shape_a[2]],
            [0, 0, m_idx * view_shape[0]],
            valid_shape=[shape_a[0], (shape_a[1] -
            m_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])), shape_a[2]])
        else:
            dyn_a = pypto.view(tensor_a, [shape_a[0], shape_a[1], view_shape[1]],
            [0, 0, m_idx * view_shape[0]],
            valid_shape=[shape_a[0], shape_a[1], (shape_a[2] -
            m_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1]))])
        if not b_trans:
            dyn_b = pypto.view(tensor_b, [shape_b[0], shape_b[1], view_shape[2]],
            [0, 0, n_idx * view_shape[1]],
            valid_shape=[(shape_b[0], shape_b[1], shape_b[2] -
            n_idx * view_shape[2]).min(pypto.symbolic_scalar(view_shape[2]))])
        else:
            dyn_b = pypto.view(tensor_b, [shape_b[0], view_shape[2], shape_b[2]],
            [0, n_idx * view_shape[1], 0],
            valid_shape=[shape_b[0], (shape_b[1] -
            n_idx * view_shape[2]).min(pypto.symbolic_scalar(view_shape[2])), shape_b[2]])

        res = pypto.matmul(dyn_a, dyn_b, dtype, a_trans=input_config.a_trans, b_trans=input_config.b_trans,
                                c_matrix_nz=input_config.c_format_nz)

        pypto.assemble(res, [0, m_idx * view_shape[1], n_idx * view_shape[2]], tensor_c)


def batch_matmul_convert_dtype(dtype):
    if dtype == INT8:
        return pypto.DT_INT8
    elif dtype == INT32:
        return pypto.DT_INT32
    elif dtype == FP32:
        return pypto.DT_FP32
    elif dtype == FP16:
        return pypto.DT_FP16
    else:
        assert False, "pypto dtype not found in batch_matmul"


def batch_matmu_create_tensor(dtype, ori_shape, tensor_name, format_nz, transposed=None):
    b = ori_shape[0]
    m = ori_shape[1]
    k = ori_shape[2]
    n = ori_shape[3]
    if tensor_name == "tensor_c":
        shape = (b, m, n)
    elif tensor_name == "tensor_b":
        shape = (b, n, k) if transposed else (b, k, n)
    elif tensor_name == "tensor_a":
        shape = (b, k, m) if transposed else (b, m, k)
    else:
        assert False, "tensor name not found in batch_matmul"
    if format_nz:
        return pypto.tensor(shape, batch_matmul_convert_dtype(dtype), tensor_name, pypto.TileOpFormat.TILEOP_NZ)
    else:
        return pypto.tensor(shape, batch_matmul_convert_dtype(dtype), tensor_name)


def ceil_div(a, b):
    return a if b == 0 else (a + b - 1) // b


def nd_to_fractal_nz(data: np.ndarray):
    ori_shape = data.shape
    batch_ori = ori_shape[:-2]
    m_ori, n_ori = ori_shape[-2:]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == FP16 or data.dtype == INT32:
        m0, n0 = 16, 16
    elif data.dtype == FP32:
        m0, n0 = 16, 8
    elif data.dtype == INT8:
        m0, n0 = 16, 32
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_n = n1 * n0 - n_ori
    padding_m = m1 * m0 - m_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), "constant")
    offset = len(data.shape) - 2
    base_list = [2, 0, 1, 3]
    array_trans = [x for x in range(offset)] + [x + offset for x in base_list]
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def gen_batch_matmul_golden_data(input_config: BatchMatmulShapeConfig):
    ori_shape = input_config.ori_shape
    shape_a = [ori_shape[0], ori_shape[1], ori_shape[2]]
    shape_b = [ori_shape[0], ori_shape[2], ori_shape[3]]

    if input_config.in_dtype == INT8:
        a_data = np.random.randint(-4, 5, shape_a).astype(INT8)
        b_data = np.random.randint(-4, 5, shape_b).astype(INT8)
        c_data = np.matmul(a_data.astype(INT32), b_data.astype(INT32)).astype(INT32)
    elif input_config.in_dtype in [FP16, FP32]:
        a_data = np.random.uniform(-1, 1, shape_a).astype(input_config.in_dtype)
        b_data = np.random.uniform(-1, 1, shape_b).astype(input_config.in_dtype)
        c_data = np.matmul(a_data.astype(FP32), b_data.astype(FP32)).astype(input_config.out_dtype)
    else:
        raise ValueError("golden dtype not found")

    if input_config.a_trans:
        a_data = a_data.transpose(0, 2, 1)
    if input_config.a_format_nz:
        a_data = nd_to_fractal_nz(a_data)

    if input_config.b_trans:
        b_data = b_data.transpose(0, 2, 1)
    if input_config.b_format_nz:
        b_data = nd_to_fractal_nz(b_data)

    if input_config.c_format_nz:
        c_data = nd_to_fractal_nz(c_data)

    a_tensor = torch.from_numpy(a_data.copy())
    b_tensor = torch.from_numpy(b_data.copy())
    c_tensor = torch.from_numpy(c_data.copy())
    c_device_tensor = torch.zeros_like(c_tensor)

    return a_tensor, b_tensor, c_tensor, c_device_tensor
