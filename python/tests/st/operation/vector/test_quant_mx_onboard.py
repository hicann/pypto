#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import os

import numpy as np
import pypto
import pytest
import torch


pytestmark = pytest.mark.soc("950")

_FP8_E4M3_TARGET_MAX_POW2 = 8
_FP8_E4M3_MAX_POS = 448.0
_FP8_E4M3_MIN_NORMAL = 2 ** -6
_FP8_E4M3_EXP_BIAS = 7
_FP8_E4M3_MBITS = 3
_F32_EXP_BIAS = 127
_F32_MBITS = 23
_E8M0_EXPONENT_BIAS = 127
_QUANT_MX_GROUP_COLS = 32
_QUANT_MX_SCALE_GROUP_COLS = 64


def _compute_shared_exponents(max_abs: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(max_abs)
    bits = max_abs.astype(np.float32, copy=False).view(np.int32)
    fp_exponent = ((bits >> _F32_MBITS) & 0xFF).astype(np.int32)
    biased = np.clip(fp_exponent - _FP8_E4M3_TARGET_MAX_POW2,
                     0, 254).astype(np.uint8)
    biased[nan_mask] = 0xFF
    return biased


def _compute_scalings_from_exponents(e8m0: np.ndarray) -> np.ndarray:
    e8m0_i32 = e8m0.astype(np.int32)
    scale_exp = np.int32(254) - e8m0_i32
    result = (scale_exp << _F32_MBITS).astype(np.int32).view(np.float32)
    result[scale_exp == 0] = np.float32(math.ldexp(1.0, -_E8M0_EXPONENT_BIAS))
    result[e8m0 == 0xFF] = np.float32(np.nan)
    return result


def _encode_e4m3_fn_vectorized(values: np.ndarray) -> np.ndarray:
    shift = _F32_MBITS - _FP8_E4M3_MBITS
    magic_adder = np.int32((1 << (shift - 1)) - 1)
    denorm_exp = (_F32_EXP_BIAS - _FP8_E4M3_EXP_BIAS) + shift + 1
    denorm_mask_int = np.int32(denorm_exp << _F32_MBITS)
    denorm_mask_float = np.array(
        denorm_mask_int, dtype=np.int32).view(np.float32)
    val_to_add = np.int32(
        ((_FP8_E4M3_EXP_BIAS - _F32_EXP_BIAS) << _F32_MBITS) + int(magic_adder))

    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.int32)
    sign = ((bits >> 24) & np.int32(0x80)).astype(np.uint8)
    abs_bits = bits & np.int32(0x7FFFFFFF)
    abs_val = abs_bits.view(np.float32).copy()

    nan_mask = np.isnan(values)
    saturate_mask = abs_val >= np.float32(_FP8_E4M3_MAX_POS)
    denormal_mask = (~saturate_mask) & (
        abs_val < np.float32(_FP8_E4M3_MIN_NORMAL)) & (~nan_mask)
    normal_mask = (~saturate_mask) & (~denormal_mask) & (~nan_mask)

    denorm_result = (
        abs_val + denorm_mask_float).view(np.int32) - denorm_mask_int
    denorm_result = denorm_result.astype(np.uint8)

    mant_odd = ((abs_bits >> np.int32(shift)) & np.int32(1)).astype(np.int32)
    normal_result = abs_bits + val_to_add + mant_odd
    normal_result = ((normal_result >> np.int32(shift))
                     & np.int32(0x7F)).astype(np.uint8)

    result = np.where(saturate_mask, np.uint8(0x7E), np.uint8(0))
    result = np.where(denormal_mask, denorm_result, result)
    result = np.where(normal_mask, normal_result, result)
    result = np.where(nan_mask, np.uint8(0x7F), result)
    return (result | sign).astype(np.uint8)


def _quant_mx_golden_bytes(input_tensor: torch.Tensor):
    x = input_tensor.cpu().numpy().astype(np.float32, copy=False)
    cols = x.shape[-1]
    rows = x.size // cols
    group_cols = (cols + _QUANT_MX_GROUP_COLS - 1) // _QUANT_MX_GROUP_COLS
    scale_group_cols = (cols + _QUANT_MX_SCALE_GROUP_COLS -
                        1) // _QUANT_MX_SCALE_GROUP_COLS

    x_flat = x.reshape(rows, cols)
    padded_cols = group_cols * _QUANT_MX_GROUP_COLS
    x_padded = np.zeros((rows, padded_cols), dtype=np.float32)
    x_padded[:, :cols] = x_flat
    x_grouped = x_padded.reshape(rows, group_cols, _QUANT_MX_GROUP_COLS)

    max_abs = np.max(np.abs(x_grouped), axis=2).astype(np.float32)
    e8m0 = _compute_shared_exponents(max_abs)
    group_scaling = _compute_scalings_from_exponents(e8m0)
    quant_grouped = _encode_e4m3_fn_vectorized(
        x_grouped * group_scaling[:, :, np.newaxis])

    quant = quant_grouped.reshape(rows, padded_cols)[:, :cols].reshape(x.shape)
    scale_shape = list(x.shape[:-1]) + [scale_group_cols, 2]
    scale = np.zeros(scale_shape, dtype=np.uint8)
    scale.reshape(rows, scale_group_cols *
                  2)[:, :group_cols] = e8m0.reshape(rows, group_cols)
    return torch.from_numpy(quant.copy()), torch.from_numpy(scale.copy())


@pytest.mark.soc("950")
def test_quant_mx_fp32_2d_onboard():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    input_shape = [4, 256]
    view_shape = [2, 256]
    tile_shape = [1, 256]
    scale_shape = [4, 4, 2]

    input_data = torch.linspace(-7.5, 7.5, steps=math.prod(input_shape),
                                dtype=torch.float32).reshape(input_shape)
    quant_output = torch.zeros(input_shape, dtype=torch.float8_e4m3fn)
    scale_output = torch.zeros(scale_shape, dtype=torch.float8_e8m0fnu)
    golden_quant_bytes, golden_scale_bytes = _quant_mx_golden_bytes(input_data)

    input_tensor = pypto.tensor(input_shape, pypto.DT_FP32, "PTO_TENSOR_input")
    quant_tensor = pypto.tensor(
        input_shape, pypto.DT_FP8E4M3, "PTO_TENSOR_quant")
    scale_tensor = pypto.tensor(
        scale_shape, pypto.DT_FP8E8M0, "PTO_TENSOR_scale")

    pypto.runtime._device_init()
    try:
        with pypto.function("MAIN", input_tensor, quant_tensor, scale_tensor):
            for row_idx in pypto.loop(input_shape[0] // view_shape[0], name="LOOP_ROW", idx_name="row_idx"):
                pypto.set_vec_tile_shapes(*tile_shape)
                input_offset = [row_idx * view_shape[0], 0]
                scale_offset = [row_idx * view_shape[0], 0, 0]
                input_view = pypto.view(input_tensor, view_shape, input_offset)
                quant_view, scale_view = pypto.quant_mx(
                    input_view,
                    pypto.DT_FP8E4M3,
                    pypto.ROUND_DOWN,
                    -1,
                    True,
                )
                pypto.assemble(quant_view, input_offset, quant_tensor)
                pypto.assemble(scale_view, scale_offset, scale_tensor)

        pto_input = pypto.from_torch(input_data, "input")
        pto_quant = pypto.from_torch(quant_output, "quant")
        pto_scale = pypto.from_torch(scale_output, "scale")
        pypto.runtime._device_run_once_data_from_host(
            pto_input, pto_quant, pto_scale)
    finally:
        pypto.runtime._device_fini()

    assert torch.equal(quant_output.view(torch.uint8), golden_quant_bytes)
    assert torch.equal(scale_output.view(torch.uint8), golden_scale_bytes)
