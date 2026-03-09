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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pypto
import pytest
from numpy.testing import assert_allclose


K_BLOCK_SIZE_64 = 64
K_BLOCK_SIZE_32 = 32
SHAPE_DIM_2 = 2


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    scale_a_trans: bool = False
    scale_b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False


def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = (0,) * batch_num
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        data = F.pad(data, (batch_padding + (0, padding_n, 0, padding_m)), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        data = F.pad(data, (batch_padding + (0, padding_n, 0, 0)), "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data


def convert_pypto_dtype_to_torch(dtype):
    if dtype == pypto.DataType.DT_FP8E4M3:
        return torch.float8_e4m3fn
    elif dtype == pypto.DataType.DT_FP8E5M2:
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported pypto DataType: {dtype}")


def create_scaled_mm_kernel_with_mn_split(tile_config: ShapeConfig):
    m, k, n = tile_config.ori_shape
    m_view, n_view = tile_config.view_shape
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = [k, m] if tile_config.a_trans else [m, k]
    b_shape = [n, k] if tile_config.b_trans else [k, n]
    bias_shape = [1, n]
    scale_a_shape = [k // K_BLOCK_SIZE_64, m, SHAPE_DIM_2] if tile_config.scale_a_trans else \
                    [m, k // K_BLOCK_SIZE_64, SHAPE_DIM_2]
    scale_b_shape = [n, k // K_BLOCK_SIZE_64, SHAPE_DIM_2] if tile_config.scale_b_trans else \
                    [k // K_BLOCK_SIZE_64, n, SHAPE_DIM_2]
    out_shape = [m, n]

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def scaled_mm_pto(a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        a_scale: pypto.Tensor(scale_a_shape, pypto.DT_FP8E8M0),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
        b_scale: pypto.Tensor(scale_b_shape, pypto.DT_FP8E8M0),
        bias: pypto.Tensor(bias_shape, pypto.DT_FP32)) -> pypto.Tensor(out_shape, tile_config.out_dtype):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape)
        m_loop = (m + m_view - 1) // m_view
        n_loop = (n + n_view - 1) // n_view
        out_tensor = pypto.Tensor([m, n], tile_config.out_dtype)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                #Get the view tensor of mat_a
                if tile_config.a_trans:
                    a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view]
                else:
                    a_view = a_tensor[m_idx * m_view: m_idx * m_view + m_view, :]
                #Get the view tensor of scale_a
                if tile_config.scale_a_trans:
                    scale_a_view = a_scale[:, m_idx * m_view: m_idx * m_view + m_view, :]
                else:
                    scale_a_view = a_scale[m_idx * m_view: m_idx * m_view + m_view, :, :]
                #Get the view tensor of mat_b
                if tile_config.b_trans:
                    b_view = b_tensor[n_idx * n_view: n_idx * n_view + n_view, :]
                else:
                    b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view]
                #Get the view tensor of scale_b
                if tile_config.scale_b_trans:
                    scale_b_view = b_scale[n_idx * n_view: n_idx * n_view + n_view, :, :]
                else:
                    scale_b_view = b_scale[:, n_idx * n_view: n_idx * n_view + n_view, :]
                #Get the view tensor of bias
                bias_view = bias[:, n_idx * n_view: n_idx * n_view + n_view]
                extend_params = {'bias_tensor': bias_view}
                out_view = pypto.scaled_mm(a_view, b_view, tile_config.out_dtype, scale_a_view, scale_b_view,
                                        extend_params=extend_params, a_trans=tile_config.a_trans,
                                        b_trans=tile_config.b_trans, scale_a_trans=tile_config.scale_a_trans,
                                        scale_b_trans=tile_config.scale_b_trans)
                out_tensor[m_idx * m_view: m_idx * m_view + m_view,
                        n_idx * n_view: n_idx * n_view + n_view] = out_view
        return out_tensor
    return scaled_mm_pto


def create_scale_mm_with_bias(tile_config: ShapeConfig):
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    a_shape = [k, m] if tile_config.a_trans else [m, k]
    b_shape = [n, k] if tile_config.b_trans else [k, n]
    scale_a_shape = [k // K_BLOCK_SIZE_64, m, SHAPE_DIM_2] if tile_config.scale_a_trans else \
                    [m, k // K_BLOCK_SIZE_64, SHAPE_DIM_2]
    scale_b_shape = [n, k // K_BLOCK_SIZE_64, SHAPE_DIM_2] if tile_config.scale_b_trans else \
                    [k // K_BLOCK_SIZE_64, n, SHAPE_DIM_2]
    bias_shape = [1, n]
    bias = torch.randn(bias_shape, dtype=torch.float32).uniform_(-3, 3)

    torch_in_dtype = convert_pypto_dtype_to_torch(tile_config.in_dtype)
    mat_a = torch.randn(a_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_a = torch.randn(scale_a_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    mat_b = torch.randn(b_shape, dtype=torch.float32).uniform_(-3, 3).to(torch_in_dtype)
    scale_b = torch.randn(scale_b_shape, dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    scale_a_tmp = scale_a.view(m, k // K_BLOCK_SIZE_32) if not tile_config.scale_a_trans else \
        torch.transpose(scale_a, -2, -1).reshape(k // K_BLOCK_SIZE_32, m).T
    scale_b_tmp = torch.transpose(scale_b, -2, -1).reshape(k // K_BLOCK_SIZE_32, n) if \
        not tile_config.scale_b_trans else scale_b.view(n, k // K_BLOCK_SIZE_32).T
    scale_a_tmp = np.repeat(scale_a_tmp.to(torch.float32), 32, axis=1)
    scale_b_tmp = np.repeat(scale_b_tmp.to(torch.float32), 32, axis=0)
    mat_a_tmp = mat_a.to(torch.float32).T if tile_config.a_trans else mat_a.to(torch.float32)
    mat_a_tmp = mat_a_tmp * scale_a_tmp.to(torch.float32)
    mat_b_tmp = mat_b.to(torch.float32).T if tile_config.b_trans else mat_b.to(torch.float32)
    mat_b_tmp = scale_b_tmp.to(torch.float32) * mat_b_tmp
    bias_tmp = np.repeat(bias, m, axis=0)
    golden = torch.matmul(mat_a_tmp.to(torch.float32), mat_b_tmp.to(torch.float32)) + bias_tmp
    if tile_config.a_format_nz:
        mat_a = trans_nd_to_fractal_nz(mat_a, True)
    if tile_config.b_format_nz:
        mat_b = trans_nd_to_fractal_nz(mat_b, True)
    out = create_scaled_mm_kernel_with_mn_split(tile_config)(mat_a.npu(), scale_a.npu(), mat_b.npu(), scale_b.npu(),
                                                            bias.npu())
    assert torch.allclose(out.cpu().to(torch.float32), golden, rtol=1e-3, atol=1e-3), "结果精度不匹配"


@pytest.mark.soc("950")
def test_scaled_mm_with_bias():
    tile_config = ShapeConfig([385, 192, 96], [64, 64], [64, 256], [256, 256], [192, 32], pypto.DataType.DT_FP8E4M3,
                              pypto.DataType.DT_FP16, False, True, True, False, False, True)
    create_scale_mm_with_bias(tile_config)

