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

import pytest
import torch

import pypto

DTYPE_CASES = [
    ("fp32", torch.float32, pypto.DT_FP32),
    ("fp16", torch.float16, pypto.DT_FP16),
    ("bf16", torch.bfloat16, pypto.DT_BF16),
    ("int32", torch.int32, pypto.DT_INT32),
    ("int16", torch.int16, pypto.DT_INT16),
    ("uint16", torch.uint16, pypto.DT_UINT16),
    ("uint32", torch.uint32, pypto.DT_UINT32),
]

SPLIT_CASES = [
    ("rank1_axis0", (8192,), (257,), (63,), 0),
    ("rank2_axis0", (4097, 37), (259, 31), (64, 13), 0),
    ("rank2_axis1", (37, 4097), (29, 259), (11, 64), 1),
    ("rank3_axis0", (1025, 19, 17), (67, 13, 11), (17, 7, 5), 0),
    ("rank3_axis1", (19, 1025, 17), (13, 67, 11), (7, 17, 5), 1),
    ("rank3_axis2", (19, 17, 1025), (13, 11, 67), (7, 5, 17), 2),
    ("rank4_axis0", (1025, 5, 7, 9), (67, 4, 6, 8), (17, 3, 4, 5), 0),
    ("rank4_axis1", (5, 1025, 7, 9), (4, 67, 6, 8), (3, 17, 4, 5), 1),
    ("rank4_axis2", (5, 7, 1025, 9), (4, 6, 67, 8), (3, 4, 17, 5), 2),
    ("rank4_axis3", (5, 7, 9, 1025), (4, 6, 8, 67), (3, 4, 5, 17), 3),
]


def _make_source(shape, dtype):
    values = torch.arange(math.prod(shape), dtype=torch.int64).remainder(251)
    if dtype.is_floating_point:
        values = values - 125
    return values.to(dtype).reshape(shape)


def _make_indices(shape, axis_size, dtype):
    values = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    return ((values * 251 + 17) % axis_size).to(dtype)


def _gather_golden(source, axis, indices):
    gather_indices = indices.to(torch.int64)
    if source.dtype in (torch.uint16, torch.uint32):
        return torch.gather(source.to(torch.int64), axis, gather_indices).to(source.dtype)
    return torch.gather(source, axis, gather_indices)


def _run_gm_source_case(case_name, source_shape, indices_shape, tile_shape, axis, torch_dtype, pypto_dtype,
                        index_torch_dtype=torch.int32, index_pypto_dtype=pypto.DT_INT32):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    source = _make_source(source_shape, torch_dtype)
    indices = _make_indices(indices_shape, source_shape[axis], index_torch_dtype)
    indices_input = indices.to(torch.int32) if index_pypto_dtype == pypto.DT_INT64 else indices
    indices_input_dtype = pypto.DT_INT32 if index_pypto_dtype == pypto.DT_INT64 else index_pypto_dtype
    output = torch.zeros(indices_shape, dtype=torch_dtype)
    expected = _gather_golden(source, axis, indices)

    pypto.runtime._device_init()
    try:
        pto_source = pypto.tensor(source_shape, pypto_dtype, f"source_{case_name}")
        pto_indices = pypto.tensor(indices_shape, indices_input_dtype, f"indices_{case_name}")
        pto_output = pypto.tensor(indices_shape, pypto_dtype, f"output_{case_name}")

        with pypto.function(f"GATHER_ELEMENTS_GM_UB_{case_name.upper()}", pto_source, pto_indices, pto_output):
            for loop_index in pypto.loop(1, name=f"gather_{case_name}_loop",
                                         idx_name=f"gather_{case_name}_loop_index"):
                indices_offset = [loop_index * indices_shape[0]] + [0] * (len(indices_shape) - 1)
                source_view = pto_source
                indices_view = pto_indices

                pypto.set_vec_tile_shapes(*tile_shape)
                indices_ub = indices_view
                if index_pypto_dtype == pypto.DT_INT64:
                    indices_ub = pypto.cast(indices_ub, pypto.DT_INT64)
                pypto.set_vec_tile_shapes(*tile_shape)
                result = pypto.gather(source_view, axis, indices_ub)
                pypto.assemble(result, indices_offset, pto_output)

        pypto.runtime._device_run_once_data_from_host(
            pypto.from_torch(source, f"source_{case_name}"),
            pypto.from_torch(indices_input, f"indices_{case_name}"),
            pypto.from_torch(output, f"output_{case_name}"),
        )
        torch.npu.synchronize(device_id)
        assert torch.equal(output, expected)
    finally:
        pypto.runtime._device_fini()


@pytest.mark.soc("950")
@pytest.mark.parametrize("case_name,source_shape,indices_shape,tile_shape,axis", SPLIT_CASES)
def test_gather_elements_from_gm_simt_splits_onboard(case_name, source_shape, indices_shape, tile_shape, axis):
    _run_gm_source_case(case_name, source_shape, indices_shape, tile_shape, axis, torch.int32, pypto.DT_INT32)


@pytest.mark.soc("950")
@pytest.mark.parametrize("dtype_name,torch_dtype,pypto_dtype", DTYPE_CASES)
def test_gather_elements_from_gm_simt_dtypes_onboard(dtype_name, torch_dtype, pypto_dtype):
    _run_gm_source_case(dtype_name, (17, 4097), (13, 67), (7, 17), 1, torch_dtype, pypto_dtype)


@pytest.mark.soc("950")
def test_gather_elements_from_gm_simt_int64_indices_onboard():
    _run_gm_source_case("int64_indices", (17, 4097), (13, 67), (7, 17), 1, torch.int32, pypto.DT_INT32,
                        torch.int64, pypto.DT_INT64)


@pytest.mark.soc("950")
def test_gather_elements_from_ub_regression_onboard():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    shape = (16, 64)
    source = torch.arange(1, 1 + shape[0] * shape[1], dtype=torch.int32).reshape(shape)
    indices = torch.arange(shape[1], dtype=torch.int32).reshape(1, -1).repeat(shape[0], 1)
    output = torch.zeros(shape, dtype=torch.int32)
    expected = torch.gather(source, 1, indices.to(torch.int64))

    pypto.runtime._device_init()
    try:
        pto_source = pypto.tensor(shape, pypto.DT_INT32, "source_ub_regression")
        pto_indices = pypto.tensor(shape, pypto.DT_INT32, "indices_ub_regression")
        pto_output = pypto.tensor(shape, pypto.DT_INT32, "output_ub_regression")

        with pypto.function("GATHER_ELEMENTS_UB_REGRESSION", pto_source, pto_indices, pto_output):
            for loop_index in pypto.loop(1, name="gather_ub_regression_loop",
                                         idx_name="gather_ub_regression_loop_index"):
                source_view = pypto.view(pto_source, list(shape), [loop_index * shape[0], 0])
                indices_view = pypto.view(pto_indices, list(shape), [loop_index * shape[0], 0])
                pypto.set_vec_tile_shapes(*shape)
                source_ub = pypto.add(source_view, 0)
                indices_ub = pypto.add(indices_view, 0)
                result = pypto.gather(source_ub, 1, indices_ub)
                pypto.assemble(result, [loop_index * shape[0], 0], pto_output)

        pypto.runtime._device_run_once_data_from_host(
            pypto.from_torch(source, "source_ub_regression"),
            pypto.from_torch(indices, "indices_ub_regression"),
            pypto.from_torch(output, "output_ub_regression"),
        )
        torch.npu.synchronize(device_id)
        assert torch.equal(output, expected)
    finally:
        pypto.runtime._device_fini()
