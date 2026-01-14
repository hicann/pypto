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
import os
import ast
import inspect
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.ast_mutator import AstMutator, ast_to_ir


def test_ast_to_ir_explicit():
    module = ir.module("main")
    builder = ir.IrBuilder()
    ctx = ir.IrBuilderContext()
    block = BlockBuilderHelper(builder, ctx)

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    # NOTE: `block` helper and shape parameter `tensor_shape`, `tile_shape`, `batch` are passed via closure
    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        input_y: ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND),
        scale1: ir.Scalar(ir.DataType.float, None, "scale1"),
        scale2: ir.Scalar(ir.DataType.float, None, "scale2"),
        result_x: ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND),
        result_y: ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        # NOTE: original low-level example does not use input_x/y to compute result_x/y
        #       will fix accordingly after the low-level example is fixed
        constant0 = block.const(0, "const_0")
        constant1 = block.const(1, "const_1")
        for i in block.loop(constant0, batch, constant1, unroll=4):
            res_loop_x = block.tile(tile_shape, ir.DataType.float, "outputX")
            block.adds(res_loop_x, scale1, out=res_loop_x)

            res_loop_y = block.tile(tile_shape, ir.DataType.float, "outputY")
            block.adds(res_loop_y, scale2, out=res_loop_y)

            if i:
                res_if_x = block.tile(tile_shape, ir.DataType.float, "outputX")
                block.muls(res_loop_x, scale1, out=res_if_x)

            else:
                res_if_y = block.tile(tile_shape, ir.DataType.float, "outputY")
                block.muls(res_loop_y, scale2, out=res_if_y)

        return (constant0,)

    # Get transformed AST (without executing)
    transformed_ast = AstMutator.mutate_ast(my_kernel)
    module_ast = ast.Module(body=[transformed_ast], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    print("unparsed:\n", ast.unparse(module_ast))

    code = compile(module_ast, filename="<ast>", mode="exec")

    # The transformed function needs access to:
    exec_namespace = {
        **globals(),  # Include global imports (ir, BlockBuilderHelper, etc.)
        **locals(),  # Include local vars `tensor_shape`, `tile_shape`, `batch`, `block`
    }
    exec(code, exec_namespace)
    my_kernel_transformed = exec_namespace["my_kernel"]
    metadata = {"name": "my_kernel", "function_kind": ir.FunctionKind.ControlFlow}
    func_ir = my_kernel_transformed(metadata)
    module.add_function(func_ir)
    module.entry = func_ir

    assert isinstance(func_ir, ir.Function)
    print("obtained ir.Function from ast!")
    # TODO: assert more information in `ir.Function` structure


def test_ast_to_ir_decorator():
    """
    Test AST transformation without executing the transformed code.
    Checks that the transformed AST contains expected patterns.
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
    @ast_to_ir(
        metadata=dict(name="my_kernel", function_kind=ir.FunctionKind.ControlFlow),
        closure_vars=dict(
            tensor_shape=tensor_shape, tile_shape=tile_shape, batch=batch, block=block
        ),
    )
    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        input_y: ir.Tensor(tensor_shape, ir.DataType.float, "inputY", ir.Format.ND),
        scale1: ir.Scalar(ir.DataType.float, None, "scale1"),
        scale2: ir.Scalar(ir.DataType.float, None, "scale2"),
        result_x: ir.Tensor(tensor_shape, ir.DataType.float, "outputX", ir.Format.ND),
        result_y: ir.Tensor(tensor_shape, ir.DataType.float, "outputY", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        # NOTE: original low-level example does not use input_x/y to compute result_x/y
        #       will fix accordingly after the low-level example is fixed
        constant0 = block.const(0, "const_0")
        constant1 = block.const(1, "const_1")
        for i in block.loop(constant0, batch, constant1, unroll=4):
            res_loop_x = block.tile(tile_shape, ir.DataType.float, "outputX")
            block.adds(res_loop_x, scale1, out=res_loop_x)

            res_loop_y = block.tile(tile_shape, ir.DataType.float, "outputY")
            block.adds(res_loop_y, scale2, out=res_loop_y)

            if i:
                res_if_x = block.tile(tile_shape, ir.DataType.float, "outputX")
                block.muls(res_loop_x, scale1, out=res_if_x)

            else:
                res_if_y = block.tile(tile_shape, ir.DataType.float, "outputY")
                block.muls(res_loop_y, scale2, out=res_if_y)

        return (constant0,)

    module.add_function(my_kernel)
    module.entry = my_kernel
    assert isinstance(my_kernel, ir.Function)
    print("obtained ir.Function from ast!")
    # TODO: assert more information in `ir.Function` structure


if __name__ == "__main__":
    test_ast_to_ir_explicit()
    test_ast_to_ir_decorator()
