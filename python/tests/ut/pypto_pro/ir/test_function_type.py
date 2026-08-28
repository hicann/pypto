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
"""Tests for function type attribute feature."""

import inspect

from pypto_pro import DataType, ir
from pypto_pro.ir import IRBuilder
import pypto_pro.language as pl


def test_ir_builder_with_function_type():
    """Test IR Builder with function type parameter."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Build function with InCore type
    with ib.function("aicore_kernel", span=span, func_type=ir.FunctionType.InCore) as f:
        y = f.param("y", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(y, span=span)

    func2 = f.get_result()
    assert func2.name == "aicore_kernel"
    assert func2.func_type == ir.FunctionType.InCore


def test_function_type_python_print():
    """Test that function type is correctly printed in Python syntax."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Ordinary functions print without a frontend decorator.
    with ib.function("default_func", span=span) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_opaque = f.get_result()
    printed = ir.python_print(func_opaque, "pl")
    assert printed.startswith("def default_func")
    assert "@pl.function" not in printed
    assert "type=" not in printed

    # FunctionType remains IR metadata and is not serialized as a removed decorator.
    with ib.function("kernel", span=span, func_type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_incore = f.get_result()
    printed_incore = ir.python_print(func_incore, "pl")
    assert printed_incore.startswith("def kernel")
    assert "@pl.function" not in printed_incore
    assert "type=" not in printed_incore


def test_jit_kernel_uses_opaque_function_type():
    """pl.jit captures only Opaque kernel entries."""
    assert "func_type" not in inspect.signature(pl.jit).parameters

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[4], pl.DT_INT64]):
        _test_result = x

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    kernel = program.get_function(kernel.__name__)

    assert kernel.name == "kernel"
    assert kernel.func_type == ir.FunctionType.Opaque


def test_function_type_language_export():
    """Test that FunctionType is exported from language module."""

    assert hasattr(pl, "FunctionType")
    assert pl.FunctionType.Opaque
    assert pl.FunctionType.InCore
