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
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.block_call import BlockCallHelper
import torch
import pypto


def _get_common_test_shape():
    """Helper function to create common test shape variables used across test functions."""
    tile_shape = [128, 128]
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    return tile_shape, batch, constant128, tensor_shape


def block_function(input_args, output_args):
    """
    1:1 ported from `test_control_flow` in `test_ir_binding.py`
    just hide `builder` and `ctx` behind ``BlockBuilderHelper
    """

    # ===== Module =====
    block = BlockBuilderHelper()

    # ===== Signature =====
    sig = ir.FunctionSignature()

    # tensor<[batch, 128], float32>
    # Passing None to Scalar indicates a symbolic/non-immediate value
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()

    input_x = ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND)
    input_y = ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND)
    scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
    scale2 = ir.Scalar(ir.DataType.float, None, "scale2")

    result_x = ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND)
    result_y = ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND)

    sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
    sig.returns = [ir.Scalar(ir.DataType.int32, None)]

    # ===== Function =====
    func = block.create_function("test_control", ir.FunctionKind.Block, sig)

    # Enter function body scope
    with block.function_scope(func):
        # for i = 0 to batch step 1
        i = block.scalar(ir.DataType.int32, "i")
        constant0 = block.const(0, "const_0")
        constant1 = block.const(1, "const_1")
        fs = block.for_node(i, constant0, batch, constant1, unroll=4)
        with block.for_scope(fs):

            res_loop_x = block.tile(tile_shape, ir.DataType.float, "outputX")
            # Note: create_op used as CreateBinaryOp placeholder
            add_op_x = block.adds(res_loop_x, scale1, out=res_loop_x)

            res_loop_y = block.tile(tile_shape, ir.DataType.float, "outputY")
            add_op_y = block.adds(res_loop_y, scale2, out=res_loop_y)

            # if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
            ifs = block.if_node(i)

            # --- IF THEN ---
            with block.if_then_scope(ifs):
                res_if_x = block.tile(tile_shape, ir.DataType.float, "outputX")
                mul_op_x = block.muls(res_loop_x, scale1, out=res_if_x)

                # test compound remove value (Assuming remove_var exists in binding)
                then_comp = ifs.then_stmts()
                assert then_comp.vars()["outputX"] == res_if_x

            # --- IF ELSE ---
            with block.if_else_scope(ifs):
                res_if_y = block.tile(tile_shape, ir.DataType.float, "outputY")
                mul_op_y = block.muls(res_loop_y, scale2, out=res_if_y)

            block.exit_if(ifs)

            # Check yields
            then_yield = then_comp.stmts()[-1]
            then_yield_set = set(then_yield.values())
            assert then_yield_set == {res_if_x, res_loop_y}

            else_comp = ifs.else_stmts()
            else_yield = else_comp.stmts()[-1]
            else_yield_set = set(else_yield.values())
            assert else_yield_set == {res_loop_x, res_if_y}
            assert else_yield.values()[0] == res_if_y
            assert else_yield.values()[1] == res_loop_x

        # Check for-yield results
        # Accessing the second statement in the for compound
        ifs_in_for = fs.stmts().stmts()[1]
        # assert set(ifs_in_for.results()) == set(fs.yield().values())

        block.create_return([constant0])
    return func


def test_block_call_1():
    shape = [512, 512]
    a = torch.rand(shape, dtype=torch.float32, device=f'cpu')
    b = torch.rand(shape, dtype=torch.float32, device=f'cpu')
    c = torch.rand(shape, dtype=torch.float32, device=f'cpu')
    pto_a = pypto.from_torch(a)
    pto_b = pypto.from_torch(b)
    pto_c = pypto.from_torch(c)
    with pypto.function("test_block_call", pto_a, pto_b, pto_c):
        loop_count = 4
        for idx in pypto.loop(0, loop_count, 1, name="LOOP_L0", idx_name="idx"):
            pto_c = BlockCallHelper.call(block_function, [pto_a, pto_b], [pto_c], [idx])[0]
