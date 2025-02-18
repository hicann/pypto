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

import os
import pypto
import numpy as np
import torch
import torch_npu


def op_hidden_loop_with_if(t0, t1, out, tiling=None):
    """
    实现隐藏循环带条件分支的逻辑
    参考C++案例: DynamicBasicTest.HiddenLoopWithIf
    """

    if tiling:
        pypto.set_vec_tile_shapes(tiling, tiling)

    CONDITION_THRESHOLD = 8
    k = 0
    # for _ in pypto.loop(1, name="L0", idx_name="i"):
    # for _ in pypto.loop(1, name="L01", idx_name="j"):
    out.move(pypto.add(t0, t1))

    if pypto.cond(k < CONDITION_THRESHOLD):
        for _ in pypto.loop(2, name="L02", idx_name="idx3"):
            t0.move(pypto.add(t0, t1))
            out.move(pypto.add(t0, out))
    else:
        for _ in pypto.loop(2, name="L03", idx_name="idx4"):
            t0.move(pypto.add(t0, t1))
            out.move(pypto.sub(t0, out))


@pypto.jit
def cust_hidden_loop_func(t0, t1, out, tiling=None):
    """
    包装函数，供设备运行使用
    """
    op_hidden_loop_with_if(t0, t1, out, tiling)


def test_hidden_loop_with_if_jit_function():
    """
    使用jit装饰器测试
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = 1, 1
    s = 32
    shape = (n * s, m * s)

    t0_data = torch.full(shape, 11.0, dtype=torch.float32, device=f"npu:{device_id}")
    t1_data = torch.full(shape, 20.0, dtype=torch.float32, device=f"npu:{device_id}")
    out_data = torch.zeros(shape, dtype=torch.float32, device=f"npu:{device_id}")

    cust_hidden_loop_func(pypto.from_torch(t0_data, "IN_t0"), \
        pypto.from_torch(t1_data, "IN_t1"), \
        pypto.from_torch(out_data, "OUT"), tiling=tiling)

    # 同步设备
    torch_npu.npu.synchronize()

    # 获取结果并验证
    out_cpu = out_data.cpu()
    golden = torch.full(shape, 113.0, dtype=torch.float32)

    if torch.allclose(golden, out_cpu, atol=1e-5):
        return True
    else:
        return False


def test_hidden_loop_with_if_multiple_shapes():
    """
    测试不同形状
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    test_cases = [
        {"tiling": 16, "n": 1, "m": 1, "s": 32},
        {"tiling": 8, "n": 2, "m": 2, "s": 16},
        {"tiling": 32, "n": 1, "m": 2, "s": 16},
    ]

    all_passed = True

    for i, config in enumerate(test_cases):
        tiling = config["tiling"]
        n, m, s = config["n"], config["m"], config["s"]
        shape = (n * s, m * s)

        t0_data = torch.full(
            shape, 11.0, dtype=torch.float32, device=f"npu:{device_id}"
        )
        t1_data = torch.full(
            shape, 20.0, dtype=torch.float32, device=f"npu:{device_id}"
        )
        out_data = torch.zeros(shape, dtype=torch.float32, device=f"npu:{device_id}")

        pto_input0 = pypto.from_torch(t0_data, f"IN_t0_case{i}")
        pto_input1 = pypto.from_torch(t1_data, f"IN_t1_case{i}")
        pto_outputs = pypto.from_torch(out_data, f"OUT_case{i}")

        cust_hidden_loop_func(pto_input0, pto_input1, pto_outputs, tiling=tiling)

        torch_npu.npu.synchronize()

        out_cpu = out_data.cpu()
        golden = torch.full(shape, 113.0, dtype=torch.float32)

        passed = torch.allclose(golden, out_cpu, atol=1e-5)
        all_passed = all_passed and passed
    return all_passed


def op_hidden_loop_mix_loops(t0, t1, t2, t3, t4, out, tiling=None):
    if tiling:
        pypto.set_vec_tile_shapes(tiling, tiling)

    CONDITION_THRESHOLD = 8
    k0 = 0
    # for _ in pypto.loop(1, name="L0", idx_name="i"):
    # for _ in pypto.loop(1, name="L01", idx_name="j"):
    if pypto.cond(k0 < CONDITION_THRESHOLD):
        t0_temp = pypto.add(t1, t1)
    else:
        t0_temp = pypto.add(t2, t2)
    t0_temp.move(pypto.add(t0_temp, 1.0))

    for _ in pypto.loop(2, name="L02", idx_name="k"):
        t3.move(pypto.mul(t3, t2))

    # for _ in pypto.loop(1, name="L03", idx_name="l"):
    out.move(pypto.sub(t3, t0_temp))

    for _ in pypto.loop(2, name="L04", idx_name="h"):
        t0_temp.move(pypto.mul(t0_temp, t2))

    # for _ in pypto.loop(1, name="L05", idx_name="q"):
    out.move(pypto.add(out, t0_temp))


@pypto.jit
def cust_hidden_loop_mix_func(t0, t1, t2, t3, t4, out, tiling=None):
    op_hidden_loop_mix_loops(t0, t1, t2, t3, t4, out, tiling)


def test_hidden_loop_mix_loops_jit_function():
    """
    使用jit装饰器测试
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = 1, 1
    s = 32
    shape = (n * s, m * s)

    t0_data = torch.full(shape, 11.0, dtype=torch.float32, device=f"npu:{device_id}")
    t1_data = torch.full(shape, 20.0, dtype=torch.float32, device=f"npu:{device_id}")
    t2_data = torch.full(shape, 30.0, dtype=torch.float32, device=f"npu:{device_id}")
    t3_data = torch.full(shape, 40.0, dtype=torch.float32, device=f"npu:{device_id}")
    t4_data = torch.full(shape, 50.0, dtype=torch.float32, device=f"npu:{device_id}")
    out_data = torch.zeros(shape, dtype=torch.float32, device=f"npu:{device_id}")

    cust_hidden_loop_mix_func( \
        pypto.from_torch(t0_data, "IN_t0"), \
        pypto.from_torch(t1_data, "IN_t1"), \
        pypto.from_torch(t2_data, "IN_t2"), \
        pypto.from_torch(t3_data, "IN_t3"), \
        pypto.from_torch(t4_data, "IN_t4"), \
        pypto.from_torch(out_data, "OUT"), tiling=tiling)

    # 同步设备
    torch_npu.npu.synchronize()

    # 获取结果并验证
    out_cpu = out_data.cpu()
    golden = torch.full(shape, 72859.0, dtype=torch.float32)

    if torch.allclose(golden, out_cpu, atol=1e-5):
        return True
    else:
        return False
