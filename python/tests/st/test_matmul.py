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
import os
import numpy as np
import torch
import pypto
import pytest
from numpy.testing import assert_allclose
import torch.nn.functional as F

FP32 = pypto.DT_FP32
FP16 = pypto.DT_FP16
INT32 = pypto.DT_INT32
INT8 = pypto.DT_INT8
UINT64 = pypto.DT_UINT64
UINT32 = pypto.DT_UINT32


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
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    mdl_flag: bool = False
    gm_acc: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None


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
    

def create_mm_kernel_with_mn_split(tile_config):
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    m_view = tile_config.view_shape[0]
    n_view = tile_config.view_shape[1]
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = [k, m] if tile_config.a_trans else [m, k]
    b_shape = [n, k] if tile_config.b_trans else [k, n]
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )
    def matmul_kernel(
        a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
    ) -> pypto.Tensor([m, n], tile_config.out_dtype):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape,
                                   enable_multi_data_load=tile_config.mdl_flag, enable_split_k=tile_config.gm_acc)
        m_loop = (m + m_view - 1) // m_view
        n_loop = (n + n_view - 1) // n_view
        out_tensor = pypto.Tensor([m, n], tile_config.out_dtype)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                if tile_config.a_trans:
                    a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view]
                else:
                    a_view = a_tensor[m_idx * m_view: m_idx * m_view + m_view, :]
                if tile_config.b_trans:
                    b_view = b_tensor[n_idx * n_view: n_idx * n_view + n_view, :]
                else:
                    b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view]
                out_view = pypto.matmul(a_view, b_view, a_trans=tile_config.a_trans, b_trans=tile_config.b_trans,
                                        out_dtype=tile_config.out_dtype)
                out_tensor[m_idx * m_view: m_idx * m_view + m_view, n_idx * n_view: n_idx * n_view + n_view] = out_view
        return out_tensor
    return matmul_kernel


def create_bmm_kernel_with_no_mn_split(tile_config):
    b = tile_config.ori_shape[0]
    m = tile_config.ori_shape[1]
    k = tile_config.ori_shape[2]
    n = tile_config.ori_shape[3]
    a_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b_format = pypto.TileOpFormat.TILEOP_NZ if tile_config.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a_shape = [b, k, m] if tile_config.a_trans else [b, m, k]
    b_shape = [b, n, k] if tile_config.b_trans else [b, k, n]
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )
    def matmul_kernel(
        a_tensor: pypto.Tensor(a_shape, tile_config.in_dtype, format=a_format),
        b_tensor: pypto.Tensor(b_shape, tile_config.in_dtype, format=b_format),
    ) -> pypto.Tensor([b, m, n], tile_config.out_dtype):
        pypto.set_cube_tile_shapes(tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape, 
                                   enable_multi_data_load=tile_config.mdl_flag, enable_split_k=tile_config.gm_acc)
        out_tensor = pypto.Tensor([b, m, n], tile_config.out_dtype)
        out_tensor = pypto.matmul(a_tensor, b_tensor, a_trans=tile_config.a_trans, b_trans=tile_config.b_trans,
                                    out_dtype=tile_config.out_dtype)
        return out_tensor
    return matmul_kernel


def create_l0c2l1_kernel(tile_config1, tile_config2, extend_config):
    m1 = tile_config1.ori_shape[0]
    k1 = tile_config1.ori_shape[1]
    n1 = tile_config1.ori_shape[2]
    m2 = tile_config1.ori_shape[2]
    k2 = tile_config2.ori_shape[2]
    a1_format = pypto.TileOpFormat.TILEOP_NZ if tile_config1.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    b1_format = pypto.TileOpFormat.TILEOP_NZ if tile_config1.b_format_nz else pypto.TileOpFormat.TILEOP_ND
    a2_format = pypto.TileOpFormat.TILEOP_NZ if tile_config2.a_format_nz else pypto.TileOpFormat.TILEOP_ND
    a1_shape = [k1, m1] if tile_config1.a_trans else [m1, k1]
    b1_shape = [n1, k1] if tile_config1.b_trans else [k1, n1]
    a2_shape = [k2, m2] if tile_config2.b_trans else [m2, k2]

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
    )
    def matmul_kernel_l0c2l1(
        a1_tensor: pypto.Tensor(a1_shape, tile_config1.in_dtype, format=a1_format),
        b1_tensor: pypto.Tensor(b1_shape, tile_config1.in_dtype, format=b1_format),
        a2_tensor: pypto.Tensor(a2_shape, tile_config2.in_dtype, format=a2_format),
        scale_tensor: pypto.Tensor([1, n1], extend_config.scale_dtype),
    ) -> pypto.Tensor([m1, k2], tile_config2.out_dtype):
        mm2_c = pypto.Tensor([m1, k2], tile_config2.out_dtype)
        params = {}
        pypto.set_cube_tile_shapes(tile_config1.m_tile_shape, tile_config1.k_tile_shape, tile_config1.n_tile_shape,
                                    enable_multi_data_load=tile_config1.mdl_flag, enable_split_k=tile_config1.gm_acc)
        if extend_config.scale_shape is not None:
            if extend_config.relu_type == 1:
                params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
            else:
                params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.NO_RELU}
        elif extend_config.scale is not None:
            if extend_config.relu_type == 1:
                params = {'scale_tensor': extend_config.scale, 'relu_type': pypto.ReLuType.RELU}
            else:
                params = {'scale_tensor': extend_config.scale, 'relu_type': pypto.ReLuType.NO_RELU}
        mm1_c = pypto.matmul(a1_tensor, b1_tensor, a_trans=tile_config1.a_trans, b_trans=tile_config1.b_trans,
                                out_dtype=tile_config1.out_dtype, extend_params=params)
        pypto.set_cube_tile_shapes(tile_config2.m_tile_shape, tile_config2.k_tile_shape, tile_config2.n_tile_shape,
                                    enable_multi_data_load=tile_config2.mdl_flag, enable_split_k=tile_config2.gm_acc)
        mm2_c = pypto.matmul(mm1_c, a2_tensor, a_trans=tile_config2.a_trans, b_trans=tile_config2.b_trans,
                                out_dtype=tile_config2.out_dtype)
        return mm2_c
    return matmul_kernel_l0c2l1


@pytest.mark.soc("950", "910")
def test_mm_with_mn_split():
    m = 69
    k = 99
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    tile_config = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP16,
                                FP32, True, True, False, False, False, False, False)
    a1_tensor = torch.rand([k, m], dtype=torch.float16)
    b1_tensor = torch.rand([n, k], dtype=torch.float16)
    golden = torch.matmul(a1_tensor.to(torch.float32).T, b1_tensor.to(torch.float32).T)
    c1 = create_mm_kernel_with_mn_split(tile_config)(a1_tensor.npu(), b1_tensor.npu())
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
def test_mm_with_mn_split_nz():
    m = 64
    k = 128
    n = 128
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 128
    n_view = 256
    tile_config = ShapeConfig([m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [m_view, n_view], FP16,
                                FP32, True, True, True, True, False, False, False)
    
    a1_tensor = torch.rand([k, m], dtype=torch.float16)
    b1_tensor = torch.rand([n, k], dtype=torch.float16)

    a1_tensor_nz = trans_nd_to_fractal_nz(a1_tensor).view(k, m) if tile_config.a_format_nz else a1_tensor
    b1_tensor_nz = trans_nd_to_fractal_nz(b1_tensor).view(n, k) if tile_config.b_format_nz else b1_tensor

    golden = torch.matmul(a1_tensor.to(torch.float32).T, b1_tensor.to(torch.float32).T)
    c1 = create_mm_kernel_with_mn_split(tile_config)(a1_tensor_nz.npu(), b1_tensor_nz.npu())
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
def test_bmm_with_mn_split():
    b = 3
    m = 63
    k = 127
    n = 129
    tile_m = 64
    tile_k = 64
    tile_n = 64
    tile_config = ShapeConfig([b, m, k, n], [tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], [-1, -1], FP16, FP32,
                                True, False, False, False, False, False, False)
    a1_tensor = torch.rand([b, k, m], dtype=torch.float16)
    b1_tensor = torch.rand([b, k, n], dtype=torch.float16)
    golden = torch.matmul(a1_tensor.to(torch.float32).transpose(-2, -1), b1_tensor.to(torch.float32))
    c1 = create_bmm_kernel_with_no_mn_split(tile_config)(a1_tensor.npu(), b1_tensor.npu())
    assert torch.allclose(c1.cpu().to(torch.float32), golden, atol=1e-3, rtol=1e-3)