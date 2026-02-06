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
import contextlib
import os

import pypto
import pytest
import torch
import torch_npu
import pytest


@pytest.mark.skip(reason="There is a probability of failure")
def test_device_run_data_from_host_numpy():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 16
    n, m, k = tiling * 1, tiling * 1, tiling * 1

    pypto.runtime._device_init()

    a = pypto.tensor((n, m, k), pypto.DT_FP32, "PTO_TENSOR_a")
    b = pypto.tensor((n, m, k), pypto.DT_FP32, "PTO_TENSOR_b")

    pypto.set_vec_tile_shapes(tiling, tiling, tiling)
    with pypto.function("MAIN", a, b):
        for idx in pypto.loop(10, name="s0", idx_name="idx"):
            if pypto.cond(idx == 0):
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)

    a_tensor = torch.rand(n, m, k, dtype=torch.float32) * 2 - 1
    b_tensor = torch.zeros(n, m, k, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = 11 * a_tensor

    assert torch.allclose(golden, b_tensor, atol=1e-5)
    pypto.runtime._device_fini()


def test_device_run_data_from_host_torch():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 8
    n, m = tiling * 1, tiling * 1

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_FP32, "PTO_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_FP32, "PTO_TENSOR_b")

    pypto.set_vec_tile_shapes(tiling, tiling)
    with pypto.function("MAIN", a, b):
        for k in pypto.loop(10, name="s0", idx_name="k"):
            if pypto.cond(k == 0):
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)

    a_tensor = torch.rand(n, m, dtype=torch.float32)
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    golden = 11 * a_tensor

    assert torch.allclose(golden, b_tensor, atol=1e-5)
    pypto.runtime._device_fini()


def test_device_run_data_from_host():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    pypto.runtime._device_init()

    a = pypto.tensor((n, m), pypto.DT_INT32, "PTO_TENSOR_a")
    b = pypto.tensor((n, m), pypto.DT_INT32, "PTO_TENSOR_b")

    pypto.set_vec_tile_shapes(tiling, tiling)
    with pypto.function("MAIN", a, b):
        for k in pypto.loop(10, name="s0", idx_name="k"):
            if pypto.cond(k == 0):
                b.move(pypto.add(a, a))
            else:
                b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)

    a_tensor = torch.arange(n * m, dtype=torch.int32).reshape(n, m)
    b_tensor = torch.zeros(n, m, dtype=torch.int32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)
    golden = 11 * a_tensor

    assert torch.equal(golden, b_tensor)
    pypto.runtime._device_fini()


# def dynamic function
@pypto.jit
def cust_dyn_func(in_tensor, out_tensor, tiling=None):
    a = in_tensor
    b = out_tensor
    pypto.set_vec_tile_shapes(tiling, tiling)
    for k in pypto.loop(10, name="s0", idx_name="k"):
        if pypto.cond(k == 0):
            b.move(pypto.add(a, a))
        else:
            b.move(pypto.add(a, b))
    assert isinstance(b, pypto.tensor)


def test_device_run_data_from_device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32
    n, m = tiling * 1, tiling * 1

    # prepare data
    a_rawdata = torch.tensor(
        [[k * 100 + v for v in range(m)] for k in range(n)])
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    b_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    # def inputs and outputs
    inputs = [a_data]
    outputs = [b_data]
    pto_inputs = [pypto.from_torch(
        tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(
        tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    cust_dyn_func(pto_inputs[0], pto_outputs[0], tiling)

    torch_npu.npu.synchronize()
    # get data and compare result
    a_data_cpu = a_data.cpu()
    b_data_cpu = b_data.cpu()
    # verify
    a_data_list = [c for r in a_data_cpu.tolist() for c in r]
    b_data_list = [c for r in b_data_cpu.tolist() for c in r]
    assert b_data_list == [v * 11 for v in a_data_list]

    c_rawdata = torch.tensor(
        [[k * 1000 + v for v in range(m)] for k in range(n)])
    c_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    pto_inputs = [pypto.from_torch(c_data, f"IN")]
    pto_outputs = [pypto.from_torch(d_data, f"OUT")]
    cust_dyn_func(pto_inputs[0], pto_outputs[0])
    c_data_list = [c for r in c_data.cpu().tolist() for c in r]
    d_data_list = [c for r in d_data.cpu().tolist() for c in r]
    assert d_data_list == [v * 11 for v in c_data_list]


# def dynamic function
@pypto.jit
def matmul_add(in_tensor0, in_tensor1, in_tensor2, out_tensor, m, k, n, tiling=None):
    a = in_tensor0
    b = in_tensor1
    c = in_tensor2
    d = out_tensor
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes(
        [tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        d.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


def test_device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 16

    a_rawdata = torch.tensor([[1] * k] * n)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')

    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.int32,
                             device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        inputs = [a_data, b_data, c_data]
        outputs = [d_data]
        pto_inputs = [pypto.from_torch(
            tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(
            tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        matmul_add(pto_inputs[0], pto_inputs[1], pto_inputs[2],
                   pto_outputs[0], m, k, n, tiling=tiling)

    torch_npu.npu.synchronize()

    for idx in range(count):
        # get data and compare result
        d_data_inlist = [c for r in d_data_list[idx].cpu().tolist() for c in r]
        assert d_data_inlist == [k + idx] * len(d_data_inlist)


class InferControlflowShape:

    def __init__(self):
        s = 32
        self.b_vec = [1024, 128, 64, 32]
        self.cache_shapes = {b: [(b, s), (b, s), (b, s)] for b in self.b_vec}

    def __call__(self, *args):
        if not args:
            return list(self.cache_shapes.values())
        for b in self.b_vec:
            if args[0][0] >= b:
                return self.cache_shapes[b]
        raise ValueError(f"invalid shape {args[0]}")


@pypto.jit(
    infer_controlflow_shape=InferControlflowShape(),
    runtime_options={
        "triple_stream_sched": True,
        "stitch_cfgcache_size": 1024 * 1024,
    }
)
def infer_shape_kenrel(a, b, c, eps):
    assert eps == 1.0
    pypto.set_vec_tile_shapes(16, 16)
    for i in pypto.loop(0, a.shape[0], 32):
        ta = a[i: i + 32, :]
        tb = b[i: i + 32, :]
        c[i:, 0:] = ta + tb


@pypto.jit(
    runtime_options={
        "stitch_cfgcache_size": 1024 * 1024,
    }
)
def infer_shape_kenrel1(a, b, c, eps):
    assert eps == 1.0
    pypto.set_vec_tile_shapes(16, 16)
    for i in pypto.loop(0, a.shape[0], 32):
        ta = a[i: i + 32, :]
        tb = b[i: i + 32, :]
        c[i:, 0:] = ta + tb


def infer_shape_common(device, s=1, n=2, mix=False):
    for bs in [2048, 1024, 512, 256, 128, 64, 32]:
        a = [torch.randn((bs, 32 * s), device=device) for _ in range(n)]
        b = [torch.randn((bs, 32 * s), device=device) for _ in range(n)]
        c = [torch.zeros_like(a[0], device=device) for _ in range(n)]

        for i in range(n):
            ta = pypto.from_torch(a[i], dynamic_axis=[0])
            tb = pypto.from_torch(b[i], dynamic_axis=[0])
            tc = pypto.from_torch(c[i], dynamic_axis=[0])
            if mix:
                if i % 3 == 0:
                    infer_shape_kenrel(ta, tb, tc, 1.0)
                else:
                    infer_shape_kenrel1(ta, tb, tc, 1.0)
            else:
                infer_shape_kenrel(ta, tb, tc, 1.0)


@contextlib.contextmanager
def aclgraph_enable():
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        g = torch_npu.npu.NPUGraph()
        torch_npu.npu.empty_cache()
        assert not torch_npu.npu.is_current_stream_capturing()
        g.capture_begin()
        yield
        assert torch_npu.npu.is_current_stream_capturing()
        g.capture_end()
    torch_npu.npu.current_stream().wait_stream(s)
    # 执行
    for _ in range(10):
        g.replay()
        stream = torch_npu.npu.current_stream()
        stream.synchronize()
    g.reset()


class TestTripleStream:

    def __init__(self):
        self.device = None

    def setup_method(self):
        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)
        self.device = f'npu:{device_id}'

    def test_aclgraph(self):
        with aclgraph_enable():
            infer_shape_common(self.device)

    def test_two_kernel(self):
        # one jit multi kernel
        infer_shape_common(self.device, s=2)
        infer_shape_common(self.device, s=1)

    @pytest.mark.skip(reason="mix two and triple stream sched not support")
    def test_mix_unify_split_stream(self):
        infer_shape_common(self.device, mix=True)
