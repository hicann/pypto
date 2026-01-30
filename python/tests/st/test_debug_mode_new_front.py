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

import os
import json
import pypto
import torch
import torch_npu


def add_wrapper(shape, tiling=None):
    @pypto.frontend.jit(
    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1}
    )
    def add(a: pypto.Tensor(shape, pypto.DT_INT32),
            b: pypto.Tensor(shape, pypto.DT_INT32)) -> pypto.Tensor(shape, pypto.DT_INT32):
        pypto.set_vec_tile_shapes(tiling, tiling)
        c = a + b
        return c
    return add


def sub_wrapper(shape, tiling=None):
    @pypto.frontend.jit(
    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1}
    )
    def sub(a: pypto.Tensor(shape, pypto.DT_INT32),
            b: pypto.Tensor(shape, pypto.DT_INT32)) -> pypto.Tensor(shape, pypto.DT_INT32):
        pypto.set_vec_tile_shapes(tiling, tiling)
        c = a - b
        return c
    return sub


def safe_json_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data, None
    except FileNotFoundError:
        return None, "File not found"
    except json.JSONDecodeError as e:
        return None, f"Invalid json format: {e}"
    except PermissionError:
        return None, "Permission Erro"
    except Exception as e:
        return None, f"Load json fail, unknow error: {e}"


def get_out_put_path():
    out_path = "./output"
    if os.path.exists(out_path):
        subdirs = [os.path.join(out_path, d) for d in os.listdir(out_path)
                if os.path.isdir(os.path.join(out_path, d))]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getctime)
            return latest_dir
    return None


def device_run(is_run_add, device_id):
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)

    # prepare data
    a_data = torch.ones((n, m), dtype=torch.int32, device=f'npu:{device_id}') * 2
    b_data = torch.ones((n, m), dtype=torch.int32, device=f'npu:{device_id}')

    if is_run_add:
        result = add_wrapper(shape, tiling)(a_data, b_data)
        torch_npu.npu.synchronize()

        golden = torch.ones((n, m), dtype=torch.int32) * 3
        assert torch.allclose(golden, result.cpu(), atol=1e-5)

    else:
        result = sub_wrapper(shape, tiling)(a_data, b_data)
        torch_npu.npu.synchronize()

        golden = torch.ones((n, m), dtype=torch.int32)
        assert torch.allclose(golden, result.cpu(), atol=1e-5)

    output_path = get_out_put_path()
    assert output_path


def test_debug_mode():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    device_run(True, device_id)
    device_run(False, device_id)
