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
"""
"""
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper


def test_control_flow():
    """
    1:1 ported from `test_control_flow` in `test_ir_binding.py`
    just hide `builder` and `ctx` behind ``BlockBuilderHelper
    """

    # ===== Module =====
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    # ===== Signature =====
    sig = ir.FunctionSignature()

    # tensor<[batch, 128], float32>
    # Passing None to Scalar indicates a symbolic/non-immediate value
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    input_x = ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND)
    input_y = ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND)
    scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
    scale2 = ir.Scalar(ir.DataType.float, None, "scale2")

    result_x = ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND)
    result_y = ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND)

    sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
    sig.returns = [ir.Scalar(ir.DataType.int32, None)]

    # ===== Function =====
    func = block.create_function("test_control", ir.FunctionKind.ControlFlow, sig)
    module.add_function(func)
    module.entry = func  # NOTE: now runs until here

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

    # TODO: assert IR module structure
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")


def test_control_flow_closure():
    """
    Rearrange `test_control_flow_rearrange` to a more functional style.
    Further ast transforms will work on `create_function` level, not module level.
    """
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    # NOTE: `block` helper and shape parameter `tensor_shape`, `tile_shape`, `batch` are passed via closure
    # NOTE: use `metadata` kwarg so we reserve positional args for input arguments in pre-transformed ast
    def create_function(metadata=None):
        input_x = ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND)
        input_y = ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND)
        scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
        scale2 = ir.Scalar(ir.DataType.float, None, "scale2")
        result_x = ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND)
        result_y = ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND)

        sig = ir.FunctionSignature()
        sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
        sig.returns = [ir.Scalar(ir.DataType.int32, None)]

        assert isinstance(metadata, dict)
        func = block.create_function(metadata["name"], metadata["function_kind"], sig)
        with block.function_scope(func):
            # for i = 0 to batch step 1
            i = block.scalar(ir.DataType.int32, "i")
            constant0 = block.const(0, "const_0")
            constant1 = block.const(1, "const_1")
            fs = block.for_node(i, constant0, batch, constant1, unroll=4)
            with block.for_scope(fs):
                res_loop_x = block.tile(tile_shape, ir.DataType.float, "outputX")
                add_op_x = block.adds(res_loop_x, scale1, out=res_loop_x)

                res_loop_y = block.tile(tile_shape, ir.DataType.float, "outputY")
                add_op_y = block.adds(res_loop_y, scale2, out=res_loop_y)

                ifs = block.if_node(i)
                with block.if_then_scope(ifs):
                    res_if_x = block.tile(tile_shape, ir.DataType.float, "outputX")
                    mul_op_x = block.muls(res_loop_x, scale1, out=res_if_x)

                with block.if_else_scope(ifs):
                    res_if_y = block.tile(tile_shape, ir.DataType.float, "outputY")
                    mul_op_y = block.muls(res_loop_y, scale2, out=res_if_y)

                block.exit_if(ifs)

            block.create_return([constant0])

        return func

    metadata = dict(name="test_control", function_kind=ir.FunctionKind.ControlFlow)
    func = create_function(metadata=metadata)
    module.add_function(func)
    module.entry = func

    # TODO: assert IR module structure
    print(f"Module: {module}\nEntry: {module.entry}\nFunctions: {module.functions}")


def test_unary_operations():
    """Test all unary operations: exp, neg, rsqrt, sqrt, logicalnot, reciprocal, abs, ln"""
    module = ir.module("test_unary")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    # Setup
    tile_shape = [128, 128]
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_unary", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        # Create input tile inside function scope
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")

        # Test all unary operations
        res_exp = block.tile(tile_shape, ir.DataType.float, "res_exp")
        block.exp(input_tile, out=res_exp)

        res_neg = block.tile(tile_shape, ir.DataType.float, "res_neg")
        block.neg(input_tile, out=res_neg)

        res_rsqrt = block.tile(tile_shape, ir.DataType.float, "res_rsqrt")
        block.rsqrt(input_tile, out=res_rsqrt)

        res_sqrt = block.tile(tile_shape, ir.DataType.float, "res_sqrt")
        block.sqrt(input_tile, out=res_sqrt)

        res_logicalnot = block.tile(tile_shape, ir.DataType.float, "res_logicalnot")
        block.logicalnot(input_tile, out=res_logicalnot)

        res_reciprocal = block.tile(tile_shape, ir.DataType.float, "res_reciprocal")
        block.reciprocal(input_tile, out=res_reciprocal)

        res_abs = block.tile(tile_shape, ir.DataType.float, "res_abs")
        block.abs(input_tile, out=res_abs)

        res_ln = block.tile(tile_shape, ir.DataType.float, "res_ln")
        block.ln(input_tile, out=res_ln)

        block.create_return([res_ln])

    print(f"Unary operations test completed: {module}")


def test_binary_operations():
    """Test all binary operations: sub, mul, div, min, max"""
    module = ir.module("test_binary")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    # Setup
    tile_shape = [128, 128]
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_binary", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        # Create input tiles inside function scope
        input_x = block.tile(tile_shape, ir.DataType.float, "input_x")
        input_y = block.tile(tile_shape, ir.DataType.float, "input_y")

        # Test all binary operations
        res_sub = block.tile(tile_shape, ir.DataType.float, "res_sub")
        block.sub(input_x, input_y, out=res_sub)

        res_mul = block.tile(tile_shape, ir.DataType.float, "res_mul")
        block.mul(input_x, input_y, out=res_mul)

        res_div = block.tile(tile_shape, ir.DataType.float, "res_div")
        block.div(input_x, input_y, out=res_div)

        res_min = block.tile(tile_shape, ir.DataType.float, "res_min")
        block.min(input_x, input_y, out=res_min)

        res_max = block.tile(tile_shape, ir.DataType.float, "res_max")
        block.max(input_x, input_y, out=res_max)

        block.create_return([res_max])

    print(f"Binary operations test completed: {module}")


def test_binary_scalar_mix_operations():
    """Test all binary scalar mix operations: adds, subs, muls, divs, mins, maxs"""
    module = ir.module("test_binary_scalar")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    # Setup
    tile_shape = [128, 128]
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    scale = ir.Scalar(ir.DataType.float, None, "scale")
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor, scale]
    sig.returns = [output_tensor]

    func = block.create_function("test_binary_scalar", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        # Create input tile inside function scope
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")

        # Test all binary scalar mix operations
        res_adds = block.tile(tile_shape, ir.DataType.float, "res_adds")
        block.adds(input_tile, scale, out=res_adds)

        res_subs = block.tile(tile_shape, ir.DataType.float, "res_subs")
        block.subs(input_tile, scale, out=res_subs)

        res_muls = block.tile(tile_shape, ir.DataType.float, "res_muls")
        block.muls(input_tile, scale, out=res_muls)

        res_divs = block.tile(tile_shape, ir.DataType.float, "res_divs")
        block.divs(input_tile, scale, out=res_divs)

        res_mins = block.tile(tile_shape, ir.DataType.float, "res_mins")
        block.mins(input_tile, scale, out=res_mins)

        res_maxs = block.tile(tile_shape, ir.DataType.float, "res_maxs")
        block.maxs(input_tile, scale, out=res_maxs)

        block.create_return([res_maxs])

    print(f"Binary scalar mix operations test completed: {module}")


if __name__ == "__main__":
    test_control_flow()
    test_control_flow_closure()
    test_unary_operations()
    test_binary_operations()
    test_binary_scalar_mix_operations()
