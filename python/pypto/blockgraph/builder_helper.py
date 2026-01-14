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
from contextlib import contextmanager

from pypto.pypto_impl import ir


class BlockBuilderHelper:
    def __init__(self, builder, ctx):
        self.builder = builder
        self.ctx = ctx

    # ===== Scope Management (Context Managers) =====

    @contextmanager
    def function_scope(self, func):
        self.builder.enter_function(self.ctx, func)
        try:
            yield
        finally:
            self.ctx.pop_scope()

    @contextmanager
    def for_scope(self, loop_node):
        self.builder.enter_for(self.ctx, loop_node)
        try:
            yield
        finally:
            self.ctx.pop_scope()
            self.builder.exit_for(self.ctx, loop_node)

    @contextmanager
    def if_then_scope(self, if_node):
        self.builder.enter_if_then(self.ctx, if_node)
        try:
            yield
        finally:
            self.ctx.pop_scope()

    @contextmanager
    def if_else_scope(self, if_node):
        self.builder.enter_if_else(self.ctx, if_node)
        try:
            yield
        finally:
            self.ctx.pop_scope()

    # NOTE: a pair of `if`` and `else` only call `builder.exit_if` once,
    #       thus do not wrap `exit`, like `for_scope`

    # ===== Control Flow Statements =====

    def for_node(self, var, start, end, step, **kwargs):
        fs = self.builder.create_for(self.ctx, var, start, end, step)
        if kwargs:
            props = fs.properties()
            for key, value in kwargs.items():
                props[key] = str(value)
        return fs

    def if_node(self, cond):
        return self.builder.create_if(self.ctx, cond)

    def exit_if(self, if_node):
        return self.builder.exit_if(self.ctx, if_node)

    # ===== Function Creation =====

    def create_function(self, name, kind, sig):
        return self.builder.create_function(name, kind, sig)

    def create_return(self, values):
        return self.builder.create_return(self.ctx, values)

    # ===== Value Creation =====

    def scalar(self, dtype, name):
        return self.builder.create_scalar(self.ctx, dtype, name)

    def const(self, value, name):
        return self.builder.create_const(self.ctx, value, name)

    def tile(self, shape, dtype, name):
        return self.builder.create_tile(self.ctx, shape, dtype, name)

    # ===== Operations =====

    # Unary operations (Tile -> Tile)
    def exp(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_EXP, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def neg(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_NEG, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def rsqrt(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_RSQRT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def sqrt(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_SQRT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def logicalnot(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_LOGICALNOT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def reciprocal(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_RECIPROCAL, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def abs(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_ABS, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def ln(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_LN, a, out)
        self.builder.emit(self.ctx, op)
        return op

    # Binary operations (Tile, Tile -> Tile)
    def sub(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_SUB, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def mul(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_MUL, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def div(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_DIV, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def min(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_MIN, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def max(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_MAX, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    # Binary scalar mix operations (Tile, Scalar -> Tile)
    def adds(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def subs(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_SUBS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def muls(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_MULS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def divs(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_DIVS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def mins(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_MINS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def maxs(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_MAXS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op
