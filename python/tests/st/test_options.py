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
import torch
import torch_npu
import pypto

shape = [4, 4]
dtype = pypto.DT_FP32


# func scope 0
@pypto.options(vec_tile_shapes=[64, 128])
def layer_norm_func():
    x = pypto.tensor([1, 1], pypto.DT_FP32)
    pypto.add(x, 1.0)
    assert [64, 128] == get_options("vec_tile_shapes")
    assert 1048 == get_options("pass.mg_copyin_upper_bound") # 当前scope未设置，依然是上层scope值


# jit scope 1
@pypto.jit(host_options={"only_codegen": True},
        pass_options={"mg_copyin_upper_bound": 1048},
        )
def set_scope_options(a, c, tiling=None):
    assert 1048 == get_options("pass.mg_copyin_upper_bound")

    # 原接口依然有效，同时修改当前scope中的配置
    pypto.set_vec_tile_shapes(32, 32)
    pypto.set_cube_tile_shapes([16, 16], [32, 32], [64, 64])
    assert [32, 32] == get_options("vec_tile_shapes")
    assert check_cube_tile_shapes([16, 16], [32, 32], [64, 64], False)

    for _ in pypto.loop(1, name="s0", idx_name="k"):
        c.move(pypto.add(a, 1.0))
        assert True == get_options("host.only_codegen")

        # 隐式 scope
        pypto.set_options(pass_options={"pg_upper_bound": 1024})
        assert 1024 == get_options("pass.pg_upper_bound")
        assert 1048 == get_options("pass.mg_copyin_upper_bound") # 当前scope未设置，依然是上层scope值

        # func scope
        layer_norm_func()

        # 显式 scope
        with pypto.options("scope2",
                            host_options={"only_codegen": True},
                            pass_options={"cube_l1_reuse_mode": 0,
                                          "pg_upper_bound": 100,
                                          "cube_nbuffer_setting": {3: 4}},
                            vec_tile_shapes=[64, 64],
                            matrix_size=[64, 32],
                            cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True]
                            ): # scope 3
            assert 0 == get_options("pass.cube_l1_reuse_mode")
            assert 100 == get_options("pass.pg_upper_bound")
            assert {3: 4} == get_options("pass.cube_nbuffer_setting")
            assert [64, 64] == get_options("vec_tile_shapes")
            assert [64, 32] == get_options("matrix_size")
            assert check_cube_tile_shapes([16, 16], [256, 512, 128], [128, 128], True)
            assert 1048 == get_options("pass.mg_copyin_upper_bound") # 当前scope未设置，依然是上层scope值
            print(pypto.get_options_tree())

        assert 1024 == get_options("pass.pg_upper_bound")
        assert [32, 32] == get_options("vec_tile_shapes")


def check_cube_tile_shapes(expected_m, expected_k, expected_n, expected_enable_multi_data_load=False, 
                        expected_enable_split_k=False):
    """Check if cube_tile_shapes matches expected values"""
    cube_tile = get_options("cube_tile_shapes")
    # Expand k to 3 elements if needed
    if len(expected_k) == 2:
        expected_k = [expected_k[0], expected_k[1], expected_k[1]]
    return (list(cube_tile.m) == expected_m and
            list(cube_tile.k) == expected_k and
            list(cube_tile.n) == expected_n and
            cube_tile.enableMultiDataLoad == expected_enable_multi_data_load and
            cube_tile.enableSplitK == expected_enable_split_k)


def get_options(key):
    scope = pypto.get_current_scope()
    return scope.get_options_prefix(key)


def test_scope():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    tiling = 32
    n, m = tiling * 1, tiling * 1
    # prepare data
    a_rawdata = torch.ones((n, m))
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
    c_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
    # def inputs and outputs
    inputs = [a_data]
    outputs = [c_data]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    set_scope_options(*pto_inputs, *pto_outputs, tiling)
    torch_npu.npu.synchronize()
    golden = torch.ones((n, m)) * 2
    assert torch.allclose(golden.int(), c_data.cpu(), atol=1e-5)


@pypto.jit
def loop_scope(a, b, c, tiling=None):
    pypto.set_vec_tile_shapes(tiling * 2, tiling * 2)
    for _ in pypto.loop(1, name="s0", idx_name="k"):
        pypto.set_vec_tile_shapes(tiling, tiling)
        c.move(pypto.add(a, b))

    for _ in pypto.loop(1, name="s0", idx_name="k"):
        assert [tiling * 2, tiling * 2] == pypto.get_vec_tile_shapes()
        c.move(pypto.add(c, b))


def test_loop_scope():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    tiling = 32
    n, m = tiling * 1, tiling * 1

    # prepare data
    a_rawdata = torch.ones((n, m)) * 2
    a_data = a_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    b_rawdata = torch.ones((n, m))
    b_data = b_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')

    c_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')

    # def inputs and outputs
    inputs = [a_data, b_data]
    outputs = [c_data]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]

    loop_scope(pto_inputs[0], pto_inputs[1], pto_outputs[0], tiling)
    torch_npu.npu.synchronize()

    golden = torch.ones((n, m)) * 4
    assert torch.allclose(golden.int(), c_data.cpu(), atol=1e-5)

if __name__ == "__main__":
    test_scope()