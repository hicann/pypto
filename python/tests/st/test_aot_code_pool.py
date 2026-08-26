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

import logging
import os

import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

KERNEL_COUNT = 20
TILE = 32
N = TILE * 2


def _expected_result(kernel_index):
    fill_a = 1
    fill_b = 2 + kernel_index
    value = fill_a + fill_b
    for _ in range(kernel_index % 3):
        value += fill_b
    return value


@pypto.frontend.jit()
def aot_kernel_0(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_0"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_1(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_1"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_2(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_2"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_3(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_3"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_4(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_4"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_5(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_5"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_6(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_6"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_7(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_7"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_8(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_8"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_9(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_9"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_10(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_10"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_11(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_11"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_12(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_12"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_13(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_13"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_14(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_14"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_15(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_15"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_16(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_16"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_17(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_17"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_18(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_18"):
        acc = pypto.add(input_a, input_b)
        output[:] = acc


@pypto.frontend.jit()
def aot_kernel_19(
    input_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    input_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    output: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for _ in pypto.loop(1, name="aot_loop_19"):
        acc = pypto.add(input_a, input_b)
        acc = pypto.add(acc, input_b)
        output[:] = acc


KERNEL_FNS = [
    aot_kernel_0, aot_kernel_1, aot_kernel_2, aot_kernel_3, aot_kernel_4,
    aot_kernel_5, aot_kernel_6, aot_kernel_7, aot_kernel_8, aot_kernel_9,
    aot_kernel_10, aot_kernel_11, aot_kernel_12, aot_kernel_13, aot_kernel_14,
    aot_kernel_15, aot_kernel_16, aot_kernel_17, aot_kernel_18, aot_kernel_19,
]


@pytest.mark.soc("910")
def test_aot_code_pool_exceed_capacity():
    torch.npu.set_device(ST_DEVICE)

    for kernel_index in range(KERNEL_COUNT):
        fill_a = 1
        fill_b = 2 + kernel_index
        input_a = torch.full((N, N), fill_a, dtype=torch.int32, device=ST_DEVICE)
        input_b = torch.full((N, N), fill_b, dtype=torch.int32, device=ST_DEVICE)
        output = torch.zeros((N, N), dtype=torch.int32, device=ST_DEVICE)

        KERNEL_FNS[kernel_index](input_a, input_b, output)
        torch.npu.synchronize()

        golden = torch.full((N, N), _expected_result(kernel_index), dtype=torch.int32)
        torch.testing.assert_close(output.cpu(), golden, atol=0, rtol=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_aot_code_pool_exceed_capacity()
    logging.info("test_aot_code_pool_exceed_capacity passed!")
