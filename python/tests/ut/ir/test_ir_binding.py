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
from pypto.pypto_impl import ir


def _get_common_test_shape():
    """Helper function to create common test shape variables used across test functions."""
    tile_shape = [128, 128]
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    return tile_shape, batch, constant128, tensor_shape


def test_dtype():
    dtypes = [
        (ir.DataType.bool, "bool", 8, 1, False),
        (ir.DataType.uint8, "uint8", 8, 1, False),
        (ir.DataType.uint16, "uint16", 16, 2, False),
        (ir.DataType.uint32, "uint32", 32, 4, False),
        (ir.DataType.uint64, "uint64", 64, 8, False),
        (ir.DataType.int8, "int8", 8, 1, False),
        (ir.DataType.int16, "int16", 16, 2, False),
        (ir.DataType.int32, "int32", 32, 4, False),
        (ir.DataType.int64, "int64", 64, 8, False),
        (ir.DataType.float8_e4m3fn, "float8_e4m3fn", 8, 1, True),
        (ir.DataType.float8_e5m2, "float8_e5m2", 8, 1, True),
        (ir.DataType.float, "float32", 32, 4, True),
        (ir.DataType.float32, "float32", 32, 4, True),
        (ir.DataType.float64, "float64", 64, 8, True),
        (ir.DataType.double, "float64", 64, 8, True),
    ]
    for (dtype, name, bit_cnt, byte_cnt, is_fp) in dtypes:
        assert str(dtype) == f"DataType.{name}"
        assert dtype.bits() == bit_cnt
        assert dtype.bytes() == byte_cnt
        assert dtype.is_float() == is_fp


def test_enum():
    ir_enums = {
        "ObjectType": [
            ("Program", ir.ObjectType.Program),
            ("Function", ir.ObjectType.Function),
            ("Statement", ir.ObjectType.Statement),
            ("Operation", ir.ObjectType.Operation),
            ("Value", ir.ObjectType.Value),
            ("Memory", ir.ObjectType.Memory),
        ],
        "FunctionKind": [
            ("ControlFlow", ir.FunctionKind.ControlFlow),
            ("DataFlow", ir.FunctionKind.DataFlow),
            ("Block", ir.FunctionKind.Block),
        ],
        "StatementKind": [
            ("Compound", ir.StatementKind.Compound),
            ("Op", ir.StatementKind.Op),
            ("For", ir.StatementKind.For),
            ("If", ir.StatementKind.If),
            ("Yield", ir.StatementKind.Yield),
            ("Call", ir.StatementKind.Call),
            ("Return", ir.StatementKind.Return),
        ],
        "ValueKind": [
            ("Scalar", ir.ValueKind.Scalar),
            ("Tensor", ir.ValueKind.Tensor),
            ("Tile", ir.ValueKind.Tile),
        ],
        "MemSpaceKind": [
            ("DDR", ir.MemSpaceKind.DDR),
            ("L2", ir.MemSpaceKind.L2),
            ("UB", ir.MemSpaceKind.UB),
            ("L1", ir.MemSpaceKind.L1),
            ("L0A", ir.MemSpaceKind.L0A),
            ("L0B", ir.MemSpaceKind.L0B),
            ("L0C", ir.MemSpaceKind.L0C),
            ("REG", ir.MemSpaceKind.REG),
            ("SHMEM", ir.MemSpaceKind.SHMEM),
        ],
        "Format": [
            ("ND", ir.Format.ND),
            ("NZ", ir.Format.NZ),
        ],
    }
    for (kind, items) in ir_enums.items():
        for name, enum in items:
            assert str(enum) == f"{kind}.{name}"


def test_control_flow():
    # ===== Module =====
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

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
    func = builder.create_function("test_control", ir.FunctionKind.ControlFlow, sig)
    module.add_function(func)
    module.entry = func  # NOTE: now runs until here

    # Enter function body scope
    builder.enter_function(ctx, func)

    # for i = 0 to batch step 1
    i = builder.create_scalar(ctx, ir.DataType.int32, "i")
    constant0 = builder.create_const(ctx, 0, "const_0")
    constant1 = builder.create_const(ctx, 1, "const_1")
    fs = builder.create_for(ctx, i, constant0, batch, constant1)

    # Test for attribute
    fs.properties()["unroll"] = "4"

    builder.enter_for(ctx, fs)

    res_loop_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputX")
    # Note: create_op used as CreateBinaryOp placeholder
    add_op_x = builder.create_binary_scalar_op(
        ir.Opcode.OP_ADDS, res_loop_x, scale1, res_loop_x
    )
    # add_op_x setup would occur here (opcode, inputs, outputs)
    builder.emit(ctx, add_op_x)

    res_loop_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputY")
    add_op_y = builder.create_binary_scalar_op(
        ir.Opcode.OP_ADDS, res_loop_y, scale2, res_loop_y
    )
    builder.emit(ctx, add_op_y)

    # if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
    ifs = builder.create_if(ctx, i)

    # --- IF THEN ---
    builder.enter_if_then(ctx, ifs)
    res_if_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputX")
    mul_op_x = builder.create_binary_scalar_op(
        ir.Opcode.OP_MULS, res_loop_x, scale1, res_if_x
    )
    builder.emit(ctx, mul_op_x)

    # test compound remove value (Assuming remove_var exists in binding)
    then_comp = ifs.then_stmts()
    assert then_comp.vars()["outputX"] == res_if_x

    ctx.pop_scope()

    # --- IF ELSE ---
    builder.enter_if_else(ctx, ifs)
    res_if_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputY")
    mul_op_y = builder.create_binary_scalar_op(
        ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y
    )
    builder.emit(ctx, mul_op_y)

    ctx.pop_scope()

    builder.exit_if(ctx, ifs)

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

    ctx.pop_scope()  # for-body
    builder.exit_for(ctx, fs)

    # Check for-yield results
    # Accessing the second statement in the for compound
    ifs_in_for = fs.stmts().stmts()[1]
    # assert set(ifs_in_for.results()) == set(fs.yield().values())

    builder.create_return(ctx, [constant0])

    ctx.pop_scope()  # function-body


def test_unary_operations():
    module = ir.module("test_unary")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    # Setup
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_unary", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    # Create input tile inside function scope
    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")

    # Test all unary operations
    unary_ops = [
        (ir.Opcode.OP_ASSIGN, "res_assign"),
        (ir.Opcode.OP_EXP, "res_exp"),
        (ir.Opcode.OP_NEG, "res_neg"),
        (ir.Opcode.OP_RSQRT, "res_rsqrt"),
        (ir.Opcode.OP_SQRT, "res_sqrt"),
        (ir.Opcode.OP_RECIPROCAL, "res_reciprocal"),
        (ir.Opcode.OP_ABS, "res_abs"),
        (ir.Opcode.OP_LN, "res_ln"),
        (ir.Opcode.OP_COMPACT, "res_compact"),
    ]
    for opcode, res_name in unary_ops:
        res = builder.create_tile(ctx, tile_shape, ir.DataType.float, res_name)
        op = builder.create_unary_op(opcode, input_tile, res)
        builder.emit(ctx, op)
    res_compact = res  # Keep reference for return statement

    builder.create_return(ctx, [res_compact])
    ctx.pop_scope()


def test_binary_operations():
    module = ir.module("test_binary")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    # Setup
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_binary", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    # Create input tiles inside function scope
    input_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_x")
    input_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_y")

    # Test all binary operations
    binary_ops = [
        (ir.Opcode.OP_ADD, "res_add"),
        (ir.Opcode.OP_SUB, "res_sub"),
        (ir.Opcode.OP_MUL, "res_mul"),
        (ir.Opcode.OP_DIV, "res_div"),
        (ir.Opcode.OP_MIN, "res_min"),
        (ir.Opcode.OP_MAX, "res_max"),
        (ir.Opcode.OP_S_ADD, "res_s_add"),
        (ir.Opcode.OP_S_SUB, "res_s_sub"),
        (ir.Opcode.OP_S_MUL, "res_s_mul"),
        (ir.Opcode.OP_S_DIV, "res_s_div"),
        (ir.Opcode.OP_S_MIN, "res_s_min"),
        (ir.Opcode.OP_S_MAX, "res_s_max"),
        (ir.Opcode.OP_PAD, "res_pad"),
    ]
    for opcode, res_name in binary_ops:
        res = builder.create_tile(ctx, tile_shape, ir.DataType.float, res_name)
        op = builder.create_binary_op(opcode, input_x, input_y, res)
        builder.emit(ctx, op)
    res_pad = res  # Keep reference for return statement

    builder.create_return(ctx, [res_pad])
    ctx.pop_scope()


def test_binary_scalar_mix_operations():
    module = ir.module("test_binary_scalar")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    # Setup
    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    scale = ir.Scalar(ir.DataType.float, None, "scale")
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor, scale]
    sig.returns = [output_tensor]

    func = builder.create_function("test_binary_scalar", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    # Create input tile inside function scope
    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")

    # Test all binary scalar mix operations
    binary_scalar_ops = [
        (ir.Opcode.OP_ADDS, "res_adds"),
        (ir.Opcode.OP_SUBS, "res_subs"),
        (ir.Opcode.OP_MULS, "res_muls"),
        (ir.Opcode.OP_DIVS, "res_divs"),
        (ir.Opcode.OP_MINS, "res_mins"),
        (ir.Opcode.OP_MAXS, "res_maxs"),
        (ir.Opcode.OP_S_ADDS, "res_s_adds"),
        (ir.Opcode.OP_S_SUBS, "res_s_subs"),
        (ir.Opcode.OP_S_MULS, "res_s_muls"),
        (ir.Opcode.OP_S_DIVS, "res_s_divs"),
        (ir.Opcode.OP_S_MINS, "res_s_mins"),
        (ir.Opcode.OP_S_MAXS, "res_s_maxs"),
    ]
    for opcode, res_name in binary_scalar_ops:
        res = builder.create_tile(ctx, tile_shape, ir.DataType.float, res_name)
        op = builder.create_binary_scalar_op(opcode, input_tile, scale, res)
        builder.emit(ctx, op)
    res_s_maxs = res  # Keep reference for return statement

    builder.create_return(ctx, [res_s_maxs])
    ctx.pop_scope()


def test_unary_with_temp_operations():
    module = ir.module("test_unary_with_temp")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_unary_with_temp", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    res_logicalnot = builder.create_tile(ctx, tile_shape, ir.DataType.float, "res_logicalnot")
    temp_tensor = builder.create_tile(ctx, tile_shape, ir.DataType.float, "temp_tensor")
    op_logicalnot = builder.create_unary_with_temp_op(
        ir.Opcode.OP_LOGICALNOT, input_tile, res_logicalnot, temp_tensor
    )
    builder.emit(ctx, op_logicalnot)

    builder.create_return(ctx, [res_logicalnot])
    ctx.pop_scope()


def test_range_operations():
    module = ir.module("test_range")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    _, _, constant128, _ = _get_common_test_shape()
    tile_shape = [128]
    tensor_shape = [constant128]
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.int32, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = []
    sig.returns = [output_tensor]

    func = builder.create_function("test_range", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    start = builder.create_const(ctx, 0, "start")
    step = builder.create_const(ctx, 1, "step")
    size = builder.create_const(ctx, 128, "size")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.int32, "output_tile")
    op_range = builder.create_range_op(ir.Opcode.OP_RANGE, start, step, size, output_tile)
    builder.emit(ctx, op_range)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_vec_dup_operations():
    module = ir.module("test_vec_dup")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = []
    sig.returns = [output_tensor]

    func = builder.create_function("test_vec_dup", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    value = builder.create_const(ctx, 1.0, "value")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    op_vec_dup = builder.create_vec_dup_op(ir.Opcode.OP_VEC_DUP, value, output_tile)
    builder.emit(ctx, op_vec_dup)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_pow_operations():
    module = ir.module("test_pow")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_pow", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    exponent = builder.create_const(ctx, 2.0, "exponent")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    op_pow = builder.create_pow_op(ir.Opcode.OP_POW, input_tile, exponent, output_tile)
    builder.emit(ctx, op_pow)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_gather_operations():
    module = ir.module("test_gather")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    indices_tensor = ir.Tensor(tensor_shape, ir.DataType.int32, "indices", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor, indices_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_gather", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    indices_tile = builder.create_tile(ctx, tile_shape, ir.DataType.int32, "indices_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    op_gather = builder.create_gather_op(ir.Opcode.OP_GATHER, input_tile, indices_tile, output_tile)
    builder.emit(ctx, op_gather)

    op_gather_from_ub = builder.create_gather_extended_op(
        ir.Opcode.OP_GATHER_FROM_UB, input_tile, indices_tile, output_tile
    )
    builder.emit(ctx, op_gather_from_ub)

    op_gather_element = builder.create_gather_extended_op(
        ir.Opcode.OP_GATHER_ELEMENT, input_tile, indices_tile, output_tile
    )
    builder.emit(ctx, op_gather_element)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_reduce_operations():
    module = ir.module("test_reduce")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor([batch], ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_reduce", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    output_tile = builder.create_tile(ctx, [128], ir.DataType.float, "output_tile")
    op_reduce_max = builder.create_reduce_op(ir.Opcode.OP_ROWMAXLINE, input_tile, output_tile)
    builder.emit(ctx, op_reduce_max)

    op_reduce_min = builder.create_reduce_op(ir.Opcode.OP_ROWMINLINE, input_tile, output_tile)
    builder.emit(ctx, op_reduce_min)

    temp_tensor = builder.create_tile(ctx, tile_shape, ir.DataType.float, "temp_tensor")
    op_reduce_max_temp = builder.create_reduce_with_temp_op(
        ir.Opcode.OP_ROWMAX_SINGLE, input_tile, output_tile, temp_tensor
    )
    builder.emit(ctx, op_reduce_max_temp)

    op_reduce_min_temp = builder.create_reduce_with_temp_op(
        ir.Opcode.OP_ROWMIN_SINGLE, input_tile, output_tile, temp_tensor
    )
    builder.emit(ctx, op_reduce_min_temp)

    op_reduce_sum_temp = builder.create_reduce_with_temp_op(
        ir.Opcode.OP_ROWSUM_SINGLE, input_tile, output_tile, temp_tensor
    )
    builder.emit(ctx, op_reduce_sum_temp)

    op_reduce_sumline_temp = builder.create_reduce_with_temp_op(
        ir.Opcode.OP_ROWSUMLINE, input_tile, output_tile, temp_tensor
    )
    builder.emit(ctx, op_reduce_sumline_temp)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_broadcast_operations():
    module = ir.module("test_broadcast")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_broadcast", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_x")
    input_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_y")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    temp_tensor = builder.create_tile(ctx, tile_shape, ir.DataType.float, "temp_tensor")
    
    op_maximum = builder.create_broadcast_with_temp_op(
        ir.Opcode.OP_MAXIMUM, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_maximum)

    op_minimum = builder.create_broadcast_with_temp_op(
        ir.Opcode.OP_MINIMUM, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_minimum)

    op_pairmax = builder.create_broadcast_with_temp_op(
        ir.Opcode.OP_PAIRMAX, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_pairmax)

    op_pairmin = builder.create_broadcast_with_temp_op(
        ir.Opcode.OP_PAIRMIN, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_pairmin)

    op_pairsum = builder.create_broadcast_with_temp_op(
        ir.Opcode.OP_PAIRSUM, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_pairsum)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_cast_operations():
    module = ir.module("test_cast")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.int32, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_cast", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.int32, "output_tile")
    op_cast = builder.create_cast_op(ir.Opcode.OP_CAST, input_tile, output_tile)
    builder.emit(ctx, op_cast)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_where_operations():
    module = ir.module("test_where")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    condition_tensor = ir.Tensor(tensor_shape, ir.DataType.bool, "condition", ir.Format.ND)
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    other_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "other", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [condition_tensor, input_tensor, other_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_where", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    condition_tile = builder.create_tile(ctx, tile_shape, ir.DataType.bool, "condition_tile")
    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    other_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "other_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    temp_tensor = builder.create_tile(ctx, tile_shape, ir.DataType.float, "temp_tensor")

    op_where_tt = builder.create_ternary_op(
        ir.Opcode.OP_WHERE_TT, condition_tile, input_tile, other_tile, output_tile, temp_tensor
    )
    builder.emit(ctx, op_where_tt)

    scalar_other = builder.create_const(ctx, 0.0, "scalar_other")
    op_where_ts = builder.create_where_ts_op(
        ir.Opcode.OP_WHERE_TS, condition_tile, input_tile, scalar_other, output_tile, temp_tensor
    )
    builder.emit(ctx, op_where_ts)

    scalar_input = builder.create_const(ctx, 1.0, "scalar_input")
    op_where_st = builder.create_where_st_op(
        ir.Opcode.OP_WHERE_ST, condition_tile, scalar_input, other_tile, output_tile, temp_tensor
    )
    builder.emit(ctx, op_where_st)

    op_where_ss = builder.create_where_ss_op(
        ir.Opcode.OP_WHERE_SS, condition_tile, scalar_input, scalar_other, output_tile, temp_tensor
    )
    builder.emit(ctx, op_where_ss)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_compare_operations():
    module = ir.module("test_compare")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.bool, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_compare", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_x")
    input_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_y")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.bool, "output_tile")
    temp_tensor = builder.create_tile(ctx, tile_shape, ir.DataType.float, "temp_tensor")

    op_compare = builder.create_compare_op(
        ir.Opcode.OP_CMP, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_compare)

    scalar_y = builder.create_const(ctx, 0.0, "scalar_y")
    op_compare_scalar = builder.create_compare_scalar_op(
        ir.Opcode.OP_CMPS, input_x, scalar_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_compare_scalar)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_matmul_operations():
    module = ir.module("test_matmul")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_matmul", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    offsets = [builder.create_const(ctx, 0, "offset0"), builder.create_const(ctx, 0, "offset1")]

    op_l1_to_l0a = builder.create_matmul_extract_op(
        ir.Opcode.OP_L1_TO_L0A, input_tile, offsets, output_tile
    )
    builder.emit(ctx, op_l1_to_l0a)

    op_l1_to_l0b = builder.create_matmul_extract_op(
        ir.Opcode.OP_L1_TO_L0B, input_tile, offsets, output_tile
    )
    builder.emit(ctx, op_l1_to_l0b)

    op_l1_to_l0at = builder.create_matmul_extract_op(
        ir.Opcode.OP_L1_TO_L0_AT, input_tile, offsets, output_tile
    )
    builder.emit(ctx, op_l1_to_l0at)

    op_l1_to_l0bt = builder.create_matmul_extract_op(
        ir.Opcode.OP_L1_TO_L0_BT, input_tile, offsets, output_tile
    )
    builder.emit(ctx, op_l1_to_l0bt)

    op_l0c_to_l1 = builder.create_matmul_extract_op(
        ir.Opcode.OP_L0C_TO_L1, input_tile, offsets, output_tile
    )
    builder.emit(ctx, op_l0c_to_l1)

    lhs_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "lhs_tile")
    rhs_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "rhs_tile")
    op_matmul_mmad = builder.create_matmul_mmad_op(
        ir.Opcode.OP_A_MUL_B, lhs_tile, rhs_tile, output_tile
    )
    builder.emit(ctx, op_matmul_mmad)

    op_matmul_acc = builder.create_matmul_acc_op(
        ir.Opcode.OP_A_MULACC_B, lhs_tile, rhs_tile, output_tile
    )
    builder.emit(ctx, op_matmul_acc)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_transpose_operations():
    module = ir.module("test_transpose")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_transpose", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")

    op_transpose_movein = builder.create_transpose_op(
        ir.Opcode.OP_TRANSPOSE_MOVEIN, input_tile, output_tile
    )
    builder.emit(ctx, op_transpose_movein)

    op_transpose_moveout = builder.create_transpose_op(
        ir.Opcode.OP_TRANSPOSE_MOVEOUT, input_tile, output_tile
    )
    builder.emit(ctx, op_transpose_moveout)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_copy_operations():
    module = ir.module("test_copy")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_copy", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")

    op_copy_in = builder.create_copy_in_out_op(
        ir.Opcode.OP_COPY_IN, input_tile, output_tile
    )
    builder.emit(ctx, op_copy_in)

    op_copy_out = builder.create_copy_in_out_op(
        ir.Opcode.OP_COPY_OUT, input_tile, output_tile
    )
    builder.emit(ctx, op_copy_out)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_any_data_copy_operations():
    module = ir.module("test_any_data_copy")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_any_data_copy", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_tile")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")

    op_vld = builder.create_any_data_copy_op(
        ir.Opcode.OP_VLD, input_tile, output_tile
    )
    builder.emit(ctx, op_vld)

    op_vst = builder.create_any_data_copy_op(
        ir.Opcode.OP_VST, input_tile, output_tile
    )
    builder.emit(ctx, op_vst)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_binary_with_temp_operations():
    module = ir.module("test_binary_with_temp")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_x_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_x", ir.Format.ND)
    input_y_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input_y", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_x_tensor, input_y_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_binary_with_temp", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    input_x = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_x")
    input_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "input_y")
    output_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "output_tile")
    temp_tensor = builder.create_tile(ctx, tile_shape, ir.DataType.float, "temp_tensor")

    op_logicaland = builder.create_binary_with_temp_op(
        ir.Opcode.OP_LOGICALAND, input_x, input_y, output_tile, temp_tensor
    )
    builder.emit(ctx, op_logicaland)

    builder.create_return(ctx, [output_tile])
    ctx.pop_scope()


def test_ub_copy_operations():
    module = ir.module("test_ub_copy")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_ub_copy", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    tensor_shape_scalars = [batch, constant128]
    input_tensor_value = builder.create_tensor(ctx, tensor_shape_scalars, ir.DataType.float, "input_tensor")
    output_tensor_value = builder.create_tensor(ctx, tensor_shape_scalars, ir.DataType.float, "output_tensor")

    ub_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "ub_tile")

    offset0 = builder.create_const(ctx, 0, "offset0")
    offset1 = builder.create_const(ctx, 0, "offset1")
    offsets = [offset0, offset1]

    # Test UB copy in: tensor -> tile
    op_ub_copy_in = builder.create_ub_copy_in_op(
        ir.Opcode.OP_UB_COPY_IN, input_tensor_value, offsets, ub_tile
    )
    builder.emit(ctx, op_ub_copy_in)

    # Test UB copy out: tile -> tensor
    op_ub_copy_out = builder.create_ub_copy_out_op(
        ir.Opcode.OP_UB_COPY_OUT, ub_tile, offsets, output_tensor_value
    )
    builder.emit(ctx, op_ub_copy_out)

    builder.create_return(ctx, [output_tensor_value])
    ctx.pop_scope()


def test_matmul_load_store_operations():
    module = ir.module("test_matmul_load_store")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()

    tile_shape, batch, constant128, tensor_shape = _get_common_test_shape()
    input_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "input", ir.Format.ND)
    output_tensor = ir.Tensor(tensor_shape, ir.DataType.float, "output", ir.Format.ND)

    sig = ir.FunctionSignature()
    sig.arguments = [input_tensor]
    sig.returns = [output_tensor]

    func = builder.create_function("test_matmul_load_store", ir.FunctionKind.DataFlow, sig)
    module.add_function(func)
    module.entry = func

    builder.enter_function(ctx, func)

    tensor_shape_scalars = [batch, constant128]
    input_tensor_value = builder.create_tensor(ctx, tensor_shape_scalars, ir.DataType.float, "input_tensor")
    output_tensor_value = builder.create_tensor(ctx, tensor_shape_scalars, ir.DataType.float, "output_tensor")

    l1_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "l1_tile")
    l0c_tile = builder.create_tile(ctx, tile_shape, ir.DataType.float, "l0c_tile")

    offset0 = builder.create_const(ctx, 0, "offset0")
    offset1 = builder.create_const(ctx, 0, "offset1")
    offsets = [offset0, offset1]

    # Test matmul load: tensor -> tile (L1 copy in)
    op_matmul_load = builder.create_matmul_load_op(
        ir.Opcode.OP_L1_COPY_IN, input_tensor_value, offsets, l1_tile
    )
    builder.emit(ctx, op_matmul_load)

    # Test matmul store: tile -> tensor (L0C copy out)
    op_matmul_store = builder.create_matmul_store_op(
        ir.Opcode.OP_L0C_COPY_OUT, l0c_tile, offsets, output_tensor_value
    )
    builder.emit(ctx, op_matmul_store)

    builder.create_return(ctx, [output_tensor_value])
    ctx.pop_scope()
