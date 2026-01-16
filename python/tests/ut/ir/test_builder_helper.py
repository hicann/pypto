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


def _get_common_test_shape():
    """Helper function to create common test shape variables used across test functions."""
    tile_shape = [128, 128]
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    return tile_shape, batch, constant128, tensor_shape


def test_control_flow():
    """
    1:1 ported from `test_control_flow` in `test_ir_binding.py`
    just hide `builder` and `ctx` behind ``BlockBuilderHelper
    """

    # ===== Module =====
    module = ir.module("main")
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
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()

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
    module = ir.module("test_unary")
    block = BlockBuilderHelper()

    # Setup
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
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
        res_assign = block.tile(tile_shape, ir.DataType.float, "res_assign")
        block.assign(input_tile, out=res_assign)

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

        res_compact = block.tile(tile_shape, ir.DataType.float, "res_compact")
        block.compact(input_tile, out=res_compact)

        block.create_return([res_compact])


def test_binary_operations():
    module = ir.module("test_binary")
    block = BlockBuilderHelper()

    # Setup
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
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
        res_add = block.tile(tile_shape, ir.DataType.float, "res_add")
        block.add(input_x, input_y, out=res_add)

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

        res_s_add = block.tile(tile_shape, ir.DataType.float, "res_s_add")
        block.s_add(input_x, input_y, out=res_s_add)

        res_s_sub = block.tile(tile_shape, ir.DataType.float, "res_s_sub")
        block.s_sub(input_x, input_y, out=res_s_sub)

        res_s_mul = block.tile(tile_shape, ir.DataType.float, "res_s_mul")
        block.s_mul(input_x, input_y, out=res_s_mul)

        res_s_div = block.tile(tile_shape, ir.DataType.float, "res_s_div")
        block.s_div(input_x, input_y, out=res_s_div)

        res_s_min = block.tile(tile_shape, ir.DataType.float, "res_s_min")
        block.s_min(input_x, input_y, out=res_s_min)

        res_s_max = block.tile(tile_shape, ir.DataType.float, "res_s_max")
        block.s_max(input_x, input_y, out=res_s_max)

        res_pad = block.tile(tile_shape, ir.DataType.float, "res_pad")
        block.pad(input_x, input_y, out=res_pad)

        block.create_return([res_pad])


def test_binary_scalar_mix_operations():
    module = ir.module("test_binary_scalar")
    block = BlockBuilderHelper()

    # Setup
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
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

        res_s_adds = block.tile(tile_shape, ir.DataType.float, "res_s_adds")
        block.s_adds(input_tile, scale, out=res_s_adds)

        res_s_subs = block.tile(tile_shape, ir.DataType.float, "res_s_subs")
        block.s_subs(input_tile, scale, out=res_s_subs)

        res_s_muls = block.tile(tile_shape, ir.DataType.float, "res_s_muls")
        block.s_muls(input_tile, scale, out=res_s_muls)

        res_s_divs = block.tile(tile_shape, ir.DataType.float, "res_s_divs")
        block.s_divs(input_tile, scale, out=res_s_divs)

        res_s_mins = block.tile(tile_shape, ir.DataType.float, "res_s_mins")
        block.s_mins(input_tile, scale, out=res_s_mins)

        res_s_maxs = block.tile(tile_shape, ir.DataType.float, "res_s_maxs")
        block.s_maxs(input_tile, scale, out=res_s_maxs)

        block.create_return([res_s_maxs])


def test_unary_with_temp_operations():
    module = ir.module("test_unary_with_temp")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_unary_with_temp", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        res_logicalnot = block.tile(tile_shape, ir.DataType.float, "res_logicalnot")
        temp_tensor = block.tile(tile_shape, ir.DataType.float, "temp_tensor")
        block.logicalnot_with_temp(input_tile, res_logicalnot, temp_tensor)

        block.create_return([res_logicalnot])


def test_range_operations():
    module = ir.module("test_range")
    block = BlockBuilderHelper()

    _, _, constant128, _ = _get_common_test_shape()
    tile_shape = [128]
    tensor_shape = [constant128]
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.int32, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = []
    sig.returns = [output_tensor]

    func = block.create_function("test_range", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        start = block.const(0, "start")
        step = block.const(1, "step")
        size = block.const(128, "size")
        output_tile = block.tile(tile_shape, ir.DataType.int32, "output_tile")
        block.range_op(start, step, size, output_tile)

        block.create_return([output_tile])


def test_vec_dup_operations():
    module = ir.module("test_vec_dup")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = []
    sig.returns = [output_tensor]

    func = block.create_function("test_vec_dup", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        value = block.const(1.0, "value")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        block.vec_dup(value, output_tile)

        block.create_return([output_tile])


def test_pow_operations():
    module = ir.module("test_pow")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_pow", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        exponent = block.const(2.0, "exponent")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        block.pow(input_tile, exponent, output_tile)

        block.create_return([output_tile])


def test_gather_operations():
    module = ir.module("test_gather")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    indices_tensor = ir.Tensor(tensor_shape, ir.DataType.int32, "indices", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor, indices_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_gather", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        indices_tile = block.tile(tile_shape, ir.DataType.int32, "indices_tile")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        block.gather(input_tile, indices_tile, output_tile)
        block.gather_extended(input_tile, indices_tile, output_tile)
        block.gather_element(input_tile, indices_tile, output_tile)

        block.create_return([output_tile])


def test_reduce_operations():
    module = ir.module("test_reduce")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor([batch], ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_reduce", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        output_tile = block.tile([128], ir.DataType.float, "output_tile")
        block.reduce(input_tile, output_tile)
        block.reduce_minline(input_tile, output_tile)

        temp_tensor = block.tile(tile_shape, ir.DataType.float, "temp_tensor")
        block.reduce_with_temp(input_tile, output_tile, temp_tensor)
        block.reduce_min_with_temp(input_tile, output_tile, temp_tensor)
        block.reduce_sum_with_temp(input_tile, output_tile, temp_tensor)
        block.reduce_sumline_with_temp(input_tile, output_tile, temp_tensor)

        block.create_return([output_tile])


def test_broadcast_operations():
    module = ir.module("test_broadcast")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_broadcast", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_x = block.tile(tile_shape, ir.DataType.float, "input_x")
        input_y = block.tile(tile_shape, ir.DataType.float, "input_y")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        temp_tensor = block.tile(tile_shape, ir.DataType.float, "temp_tensor")
        block.broadcast_with_temp(input_x, input_y, output_tile, temp_tensor)
        block.broadcast_minimum(input_x, input_y, output_tile, temp_tensor)
        block.broadcast_pairmax(input_x, input_y, output_tile, temp_tensor)
        block.broadcast_pairmin(input_x, input_y, output_tile, temp_tensor)
        block.broadcast_pairsum(input_x, input_y, output_tile, temp_tensor)

        block.create_return([output_tile])


def test_cast_operations():
    module = ir.module("test_cast")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.int32, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_cast", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        output_tile = block.tile(tile_shape, ir.DataType.int32, "output_tile")
        block.cast(input_tile, output_tile)

        block.create_return([output_tile])


def test_where_operations():
    module = ir.module("test_where")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    condition_tensor = ir.Tensor(tensor_shape, ir.DataType.bool, "condition", ir.Format.ND)
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    other_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "other", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [condition_tensor, input_tensor, other_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_where", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        condition_tile = block.tile(tile_shape, ir.DataType.bool, "condition_tile")
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        other_tile = block.tile(tile_shape, ir.DataType.float, "other_tile")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        temp_tensor = block.tile(tile_shape, ir.DataType.float, "temp_tensor")

        block.where_tt(condition_tile, input_tile, other_tile, output_tile, temp_tensor)

        scalar_other = block.const(0.0, "scalar_other")
        block.where_ts(condition_tile, input_tile, scalar_other, output_tile, temp_tensor)

        scalar_input = block.const(1.0, "scalar_input")
        block.where_st(condition_tile, scalar_input, other_tile, output_tile, temp_tensor)

        block.where_ss(condition_tile, scalar_input, scalar_other, output_tile, temp_tensor)

        block.create_return([output_tile])


def test_compare_operations():
    module = ir.module("test_compare")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.bool, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_compare", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_x = block.tile(tile_shape, ir.DataType.float, "input_x")
        input_y = block.tile(tile_shape, ir.DataType.float, "input_y")
        output_tile = block.tile(tile_shape, ir.DataType.bool, "output_tile")
        temp_tensor = block.tile(tile_shape, ir.DataType.float, "temp_tensor")

        block.compare(input_x, input_y, output_tile, temp_tensor)

        scalar_y = block.const(0.0, "scalar_y")
        block.compare_scalar(input_x, scalar_y, output_tile, temp_tensor)

        block.create_return([output_tile])


def test_matmul_operations():
    module = ir.module("test_matmul")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_matmul", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        offsets = [block.const(0, "offset0"), block.const(0, "offset1")]

        block.matmul_extract(input_tile, offsets, output_tile)
        block.matmul_extract_l0b(input_tile, offsets, output_tile)
        block.matmul_extract_l0at(input_tile, offsets, output_tile)
        block.matmul_extract_l0bt(input_tile, offsets, output_tile)
        block.matmul_extract_l0c_to_l1(input_tile, offsets, output_tile)

        lhs_tile = block.tile(tile_shape, ir.DataType.float, "lhs_tile")
        rhs_tile = block.tile(tile_shape, ir.DataType.float, "rhs_tile")
        block.matmul_mmad(lhs_tile, rhs_tile, output_tile)
        block.matmul_acc(lhs_tile, rhs_tile, output_tile)

        block.create_return([output_tile])


def test_matmul_load_store_operations():
    module = ir.module("test_matmul_load_store")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_matmul_load_store", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        # Create tensor using tensor method
        tensor_shape_scalars = [batch, constant128]
        input_tensor_value = block.tensor(tensor_shape_scalars, ir.DataType.float, "input_tensor")
        output_tensor_value = block.tensor(tensor_shape_scalars, ir.DataType.float, "output_tensor")

        # Create tiles for matmul operations
        l1_tile = block.tile(tile_shape, ir.DataType.float, "l1_tile")
        l0c_tile = block.tile(tile_shape, ir.DataType.float, "l0c_tile")

        # Create offset scalars
        offset0 = block.const(0, "offset0")
        offset1 = block.const(0, "offset1")
        offsets = [offset0, offset1]

        # Test matmul load: tensor -> tile (L1 copy in)
        block.matmul_load(input_tensor_value, offsets, l1_tile)

        # Test matmul store: tile -> tensor (L0C copy out)
        block.matmul_store(l0c_tile, offsets, output_tensor_value)

        block.create_return([output_tensor_value])


def test_transpose_operations():
    module = ir.module("test_transpose")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_transpose", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        block.transpose_movein(input_tile, output_tile)
        block.transpose_moveout(input_tile, output_tile)

        block.create_return([output_tile])


def test_copy_operations():
    module = ir.module("test_copy")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_copy", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        block.copy_in(input_tile, output_tile)
        block.copy_out(input_tile, output_tile)

        block.create_return([output_tile])


def test_ub_copy_operations():
    module = ir.module("test_ub_copy")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_ub_copy", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        # Create tensor using tensor method
        tensor_shape_scalars = [batch, constant128]
        input_tensor_value = block.tensor(tensor_shape_scalars, ir.DataType.float, "input_tensor")
        output_tensor_value = block.tensor(tensor_shape_scalars, ir.DataType.float, "output_tensor")

        # Create tile for UB operations
        ub_tile = block.tile(tile_shape, ir.DataType.float, "ub_tile")

        # Create offset scalars
        offset0 = block.const(0, "offset0")
        offset1 = block.const(0, "offset1")
        offsets = [offset0, offset1]

        # Test UB copy in: tensor -> tile
        block.ub_copy_in(input_tensor_value, offsets, ub_tile)

        # Test UB copy out: tile -> tensor
        block.ub_copy_out(ub_tile, offsets, output_tensor_value)

        block.create_return([output_tensor_value])


def test_any_data_copy_operations():
    module = ir.module("test_any_data_copy")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_any_data_copy", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_tile = block.tile(tile_shape, ir.DataType.float, "input_tile")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        block.vld(input_tile, output_tile)
        block.vst(input_tile, output_tile)

        block.create_return([output_tile])


def test_binary_with_temp_operations():
    module = ir.module("test_binary_with_temp")
    block = BlockBuilderHelper()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = block.create_function("test_binary_with_temp", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    with block.function_scope(func):
        input_x = block.tile(tile_shape, ir.DataType.float, "input_x")
        input_y = block.tile(tile_shape, ir.DataType.float, "input_y")
        output_tile = block.tile(tile_shape, ir.DataType.float, "output_tile")
        temp_tensor = block.tile(tile_shape, ir.DataType.float, "temp_tensor")
        block.logicaland(input_x, input_y, output_tile, temp_tensor)

        block.create_return([output_tile])

