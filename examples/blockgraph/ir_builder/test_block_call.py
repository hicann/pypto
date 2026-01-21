#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 - 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
BlockCall demo
"""
import torch
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.block_call import BlockCallHelper
import pypto


def test_block_function(args):
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)
    tile_shape = [128, 128]
    sig = ir.FunctionSignature(args)
    func = block.create_function("test_div", ir.FunctionKind.Block, sig)
    with block.function_scope(func):
        constant0 = block.const(0, "const_0")
        res_loop_x = block.tile(tile_shape, ir.DataType.float, "outputX")
        res_loop_y = block.tile(tile_shape, ir.DataType.float, "outputY")
        block.div(res_loop_x, res_loop_y, out=res_loop_x)
        block.create_return([constant0])
    return func


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def test_block_call(x, y, o):
    loop_count = 4
    for idx in pypto.loop(0, loop_count, 1, name="LOOP_L0", idx_name="idx"):
        o = BlockCallHelper.call(test_block_function, [x, y], [o], [idx])[0]


if __name__ == "__main__":
    shape = [512, 512]

    x = torch.rand(shape, dtype=torch.float32, device=f'cpu')
    y = torch.rand(shape, dtype=torch.float32, device=f'cpu')
    o = torch.rand(shape, dtype=torch.float32, device=f'cpu')
    pto_x = pypto.from_torch(x)
    pto_y = pypto.from_torch(y)
    pto_o = pypto.from_torch(o)
    test_block_call(pto_x, pto_y, pto_o)