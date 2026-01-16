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
from pypto.pypto_impl import ir
from pypto.blockgraph.builder_helper import BlockBuilderHelper
from pypto.blockgraph.ast_mutator import AstMutator

# Global switch to control dumping transformed source code
DUMP_TRANSFORMED_SOURCE = False  # set to True to understand parser behavior
DUMP_DIR = "./temp"


def _get_dump_path(test_name: str) -> str:
    """Get dump path for transformed source code."""
    if not DUMP_TRANSFORMED_SOURCE:
        return None
    os.makedirs(DUMP_DIR, exist_ok=True)
    return os.path.join(DUMP_DIR, f"{test_name}_transformed.py")


def _verify_transformed_ast(transformed_ast: ast.FunctionDef, func_name: str):
    """
    Verify that the transformed AST contains expected patterns.

    Args:
        transformed_ast: The transformed AST function node
        func_name: The name of the original function
    """
    # Verify it's a FunctionDef node
    assert isinstance(transformed_ast, ast.FunctionDef), "Expected FunctionDef node"

    # Unparse and check for expected patterns
    source = ast.unparse(transformed_ast)

    # Check for expected transformations
    assert "block.create_function" in source, "Expected block.create_function call"
    assert (
        "block.function_scope" in source
    ), "Expected block.function_scope context manager"
    assert "block.create_return" in source, "Expected block.create_return call"

    # Check that metadata parameter is present
    assert "metadata" in source, "Expected metadata parameter"

    # Verify the function signature accepts metadata parameter
    assert (
        f"def {func_name}(metadata):" in source or f"def {func_name}(metadata" in source
    ), f"Function should accept metadata parameter"

    # Verify key transformations are present (positive checks)
    # The presence of these patterns confirms the transformation worked
    assert "FunctionSignature" in source, "Should create FunctionSignature"
    assert "sig.arguments" in source, "Should set signature arguments"
    assert "sig.returns" in source, "Should set signature returns"

    # Check for control flow transformations if present in the source
    # (These are optional depending on the test case)
    if "block.for_scope" in source:
        assert "block.for_node" in source, "Expected block.for_node call"
        assert "block.scalar" in source, "Expected block.scalar call for loop variable"

    if "block.if_then_scope" in source:
        assert "block.if_node" in source, "Expected block.if_node call"
        assert "block.exit_if" in source, "Expected block.exit_if call"

    if "block.if_else_scope" in source:
        assert (
            "block.if_then_scope" in source
        ), "Expected block.if_then_scope when else is present"


def test_ast_transform():
    """
    Test AST transformation without executing the transformed code.
    Checks that the transformed AST contains expected patterns.
    """
    block = BlockBuilderHelper()

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
    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_ast_transform")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Additional checks specific to this test (for loop and if statement)
    source = ast.unparse(transformed_ast)
    assert "block.for_scope" in source, "Expected block.for_scope context manager"
    assert (
        "block.if_then_scope" in source
    ), "Expected block.if_then_scope context manager"
    assert (
        "block.if_else_scope" in source
    ), "Expected block.if_else_scope context manager"


def test_nested_for_loops():
    """Test nested for loops transformation."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        constant1 = block.const(1, "const_1")
        for i in block.loop(constant0, batch, constant1):
            for j in block.loop(constant0, constant128, constant1):
                res = block.tile(tile_shape, ir.DataType.float, "output")
                block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_nested_for_loops")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Additional checks specific to nested for loops
    source = ast.unparse(transformed_ast)
    assert "block.for_scope" in source, "Expected block.for_scope context manager"
    # Should have multiple for_scope calls for nested loops
    assert source.count("block.for_scope") >= 2, "Expected nested for loops"


def test_nested_if_statements():
    """Test nested if statements transformation."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond1: ir.Scalar(ir.DataType.bool, None, "cond1"),
        cond2: ir.Scalar(ir.DataType.bool, None, "cond2"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        if cond1:
            res1 = block.tile(tile_shape, ir.DataType.float, "output1")
            if cond2:
                res2 = block.tile(tile_shape, ir.DataType.float, "output2")
                block.adds(res1, res2, out=res1)
            else:
                block.adds(res1, input_x, out=res1)
        else:
            res3 = block.tile(tile_shape, ir.DataType.float, "output3")
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_nested_if_statements")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Additional checks specific to nested if statements
    source = ast.unparse(transformed_ast)
    assert (
        "block.if_then_scope" in source
    ), "Expected block.if_then_scope context manager"
    assert (
        "block.if_else_scope" in source
    ), "Expected block.if_else_scope context manager"
    # Should have multiple if_then_scope calls for nested ifs
    assert source.count("block.if_then_scope") >= 2, "Expected nested if statements"


def test_for_with_nested_if():
    """Test for loop with nested if statement."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        constant1 = block.const(1, "const_1")
        for i in block.loop(constant0, batch, constant1):
            res = block.tile(tile_shape, ir.DataType.float, "output")
            if i:
                block.adds(res, input_x, out=res)
            else:
                block.muls(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_for_with_nested_if")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Additional checks specific to for loop with nested if
    source = ast.unparse(transformed_ast)
    assert "block.for_scope" in source, "Expected block.for_scope context manager"
    assert (
        "block.if_then_scope" in source
    ), "Expected block.if_then_scope context manager"
    assert (
        "block.if_else_scope" in source
    ), "Expected block.if_else_scope context manager"


def test_if_with_nested_for():
    """Test if statement with nested for loop."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond: ir.Scalar(ir.DataType.bool, None, "cond"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        constant1 = block.const(1, "const_1")
        if cond:
            for i in block.loop(constant0, batch, constant1):
                res = block.tile(tile_shape, ir.DataType.float, "output")
                block.adds(res, input_x, out=res)
        else:
            res = block.tile(tile_shape, ir.DataType.float, "output")
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_if_with_nested_for")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Additional checks specific to if statement with nested for
    source = ast.unparse(transformed_ast)
    assert (
        "block.if_then_scope" in source
    ), "Expected block.if_then_scope context manager"
    assert (
        "block.if_else_scope" in source
    ), "Expected block.if_else_scope context manager"
    assert "block.for_scope" in source, "Expected block.for_scope context manager"


def test_if_without_else():
    """Test if statement without else block."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond: ir.Scalar(ir.DataType.bool, None, "cond"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        res = block.tile(tile_shape, ir.DataType.float, "output")
        if cond:
            block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_if_without_else")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Additional checks specific to if without else
    source = ast.unparse(transformed_ast)
    assert (
        "block.if_then_scope" in source
    ), "Expected block.if_then_scope context manager"
    # Should not have if_else_scope when there's no else clause
    # (The helper function checks for if_else_scope only if it exists in source)


def test_if_with_python_bool_constant():
    """Test if statement with Python bool constant (True/False) is not transformed."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        res = block.tile(tile_shape, ir.DataType.float, "output")
        if True:
            block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_if_with_python_bool_constant")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Check that Python bool constant is NOT transformed
    source = ast.unparse(transformed_ast)
    # Should NOT have block.if_then_scope for Python bool constant
    assert (
        "block.if_then_scope" not in source
    ), "Python bool constant should not be transformed"
    assert (
        "block.if_node" not in source
    ), "Python bool constant should not be transformed"
    # Should still have the original if True pattern
    assert "if True:" in source, "Original if True should remain"


def test_if_with_python_bool_variable():
    """Test if statement with Python bool variable is not transformed."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        res = block.tile(tile_shape, ir.DataType.float, "output")
        x = True  # Python bool variable
        if x:
            block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel, dump_source=_get_dump_path("test_if_with_python_bool_variable")
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Check that Python bool variable is NOT transformed
    source = ast.unparse(transformed_ast)
    # Should NOT have block.if_then_scope for Python bool variable
    assert (
        "block.if_then_scope" not in source
    ), "Python bool variable should not be transformed"
    assert (
        "block.if_node" not in source
    ), "Python bool variable should not be transformed"
    # Should still have the original if x pattern
    assert "if x:" in source, "Original if x should remain"


def test_if_with_ir_scalar_still_transformed():
    """Test that if statement with ir.Scalar condition is still transformed."""
    block = BlockBuilderHelper()

    batch = ir.Scalar(ir.DataType.int32, None, "batch")
    constant128 = ir.Scalar(ir.DataType.int64, 128, "const_128")
    tensor_shape = [batch, constant128]
    tile_shape = [128, 128]

    def my_kernel(
        input_x: ir.Tensor(tensor_shape, ir.DataType.float, "inputX", ir.Format.ND),
        cond: ir.Scalar(ir.DataType.bool, None, "cond"),
    ) -> (ir.Scalar(ir.DataType.int32, None),):
        constant0 = block.const(0, "const_0")
        res = block.tile(tile_shape, ir.DataType.float, "output")
        if cond:
            block.adds(res, input_x, out=res)
        return (constant0,)

    transformed_ast = AstMutator.mutate_ast(
        my_kernel,
        dump_source=_get_dump_path("test_if_with_ir_scalar_still_transformed"),
    )

    # Verify transformed AST
    _verify_transformed_ast(transformed_ast, "my_kernel")

    # Check that ir.Scalar condition IS transformed
    source = ast.unparse(transformed_ast)
    assert "block.if_then_scope" in source, "ir.Scalar condition should be transformed"
    assert "block.if_node" in source, "ir.Scalar condition should be transformed"


if __name__ == "__main__":
    tests = [
        ("test_ast_transform", test_ast_transform),
        ("test_nested_for_loops", test_nested_for_loops),
        ("test_nested_if_statements", test_nested_if_statements),
        ("test_for_with_nested_if", test_for_with_nested_if),
        ("test_if_with_nested_for", test_if_with_nested_for),
        ("test_if_without_else", test_if_without_else),
        ("test_if_with_python_bool_constant", test_if_with_python_bool_constant),
        ("test_if_with_python_bool_variable", test_if_with_python_bool_variable),
        (
            "test_if_with_ir_scalar_still_transformed",
            test_if_with_ir_scalar_still_transformed,
        ),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"Running {test_name}...")
            test_func()
            print(f"✓ {test_name} passed\n")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}\n")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")
    if failed > 0:
        exit(1)
