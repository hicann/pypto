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


def test_dtype():
    dtypes = [
        (ir.DataType.bool, 'bool', 8, 1, False),
        (ir.DataType.uint8, 'uint8', 8, 1, False),
        (ir.DataType.uint16, 'uint16', 16, 2, False),
        (ir.DataType.uint32, 'uint32', 32, 4, False),
        (ir.DataType.uint64, 'uint64', 64, 8, False),
        (ir.DataType.int8, 'int8', 8, 1, False),
        (ir.DataType.int16, 'int16', 16, 2, False),
        (ir.DataType.int32, 'int32', 32, 4, False),
        (ir.DataType.int64, 'int64', 64, 8, False),
        (ir.DataType.float8_e4m3fn, 'float8_e4m3fn', 8, 1, True),
        (ir.DataType.float8_e5m2, 'float8_e5m2', 8, 1, True),
        (ir.DataType.float, 'float32', 32, 4, True),
        (ir.DataType.float32, 'float32', 32, 4, True),
        (ir.DataType.float64, 'float64', 64, 8, True),
        (ir.DataType.double, 'float64', 64, 8, True),
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
    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    input_x = ir.Tensor(tensor_shape, ir.DataType.float,
                        "inputX", ir.Format.ND)
    input_y = ir.Tensor(tensor_shape, ir.DataType.float,
                        "inputY", ir.Format.ND)
    scale1 = ir.Scalar(ir.DataType.float, None, "scale1")
    scale2 = ir.Scalar(ir.DataType.float, None, "scale2")

    result_x = ir.Tensor(tensor_shape, ir.DataType.float,
                         "outputX", ir.Format.ND)
    result_y = ir.Tensor(tensor_shape, ir.DataType.float,
                         "outputY", ir.Format.ND)

    sig.arguments = [input_x, input_y, scale1, scale2, result_x, result_y]
    sig.returns = [ir.Scalar(ir.DataType.int32, None)]

    # ===== Function =====
    func = builder.create_function(
        "test_control", ir.FunctionKind.ControlFlow, sig)
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

    res_loop_x = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputX")
    # Note: create_op used as CreateBinaryOp placeholder
    add_op_x = builder.create_binary_scalar_op(
        ir.Opcode.OP_ADDS, res_loop_x, scale1, res_loop_x)
    # add_op_x setup would occur here (opcode, inputs, outputs)
    builder.emit(ctx, add_op_x)

    res_loop_y = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputY")
    add_op_y = builder.create_binary_scalar_op(
        ir.Opcode.OP_ADDS, res_loop_y, scale2, res_loop_y)
    builder.emit(ctx, add_op_y)

    # if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
    ifs = builder.create_if(ctx, i)

    # --- IF THEN ---
    builder.enter_if_then(ctx, ifs)
    res_if_x = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputX")
    mul_op_x = builder.create_binary_scalar_op(
        ir.Opcode.OP_MULS, res_loop_x, scale1, res_if_x)
    builder.emit(ctx, mul_op_x)

    # test compound remove value (Assuming remove_var exists in binding)
    then_comp = ifs.then_stmts()
    assert then_comp.vars()["outputX"] == res_if_x

    ctx.pop_scope()

    # --- IF ELSE ---
    builder.enter_if_else(ctx, ifs)
    res_if_y = builder.create_tile(
        ctx, tile_shape, ir.DataType.float, "outputY")
    mul_op_y = builder.create_binary_scalar_op(
        ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y)
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
