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
    def __init__(self, builder=None, ctx=None):
        if builder is None:
            builder = ir.IrBuilder()
        if ctx is None:
            ctx = ir.IrBuilderContext()
        self.builder = builder
        self.ctx = ctx
        self.last_func = None

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
        self.last_func = self.builder.create_function(name, kind, sig)
        return self.last_func

    def create_return(self, values):
        return self.builder.create_return(self.ctx, values)

    # ===== Value Creation =====

    def scalar(self, dtype, name):
        return self.builder.create_scalar(self.ctx, dtype, name)

    def const(self, value, name):
        return self.builder.create_const(self.ctx, value, name)

    def tile(self, shape, dtype, name):
        return self.builder.create_tile(self.ctx, shape, dtype, name)

    def tensor(self, shape, dtype, name):
        return self.builder.create_tensor(self.ctx, shape, dtype, name)

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
    
    def sign(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_SIGN, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def rsqrt(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_RSQRT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def reciprocal(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_RECIPROCAL, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def relu(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_RELU, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def round(self, a, b, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_ROUND, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def sqrt(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_SQRT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def ceil(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_CEIL, a, out)
        self.builder.emit(self.ctx, op)
        return op
    
    def floor(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_FLOOR, a, out)
        self.builder.emit(self.ctx, op)
        return op
    
    def floor(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_FLOOR, a, out)
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

    def assign(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_ASSIGN, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def compact(self, a, out):
        op = self.builder.create_unary_op(ir.Opcode.OP_COMPACT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    # Binary operations (Tile, Tile -> Tile)
    def add(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_ADD, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

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

    def hypot(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_HYPOT, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def fmod(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_MOD, a, b, out)
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

    def s_add(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_S_ADD, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_sub(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_S_SUB, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_mul(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_S_MUL, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_div(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_S_DIV, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_min(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_S_MIN, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_max(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_S_MAX, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def pad(self, a, b, out):
        op = self.builder.create_binary_op(ir.Opcode.OP_PAD, a, b, out)
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

    def s_adds(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_S_ADDS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_subs(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_S_SUBS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_muls(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_S_MULS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_divs(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_S_DIVS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_mins(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_S_MINS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    def s_maxs(self, a, b, out):
        op = self.builder.create_binary_scalar_op(ir.Opcode.OP_S_MAXS, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    # Binary operations with temp tensor
    def logicaland(self, a, b, out, temp):
        op = self.builder.create_binary_with_temp_op(ir.Opcode.OP_LOGICALAND, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    # Unary operations with temp tensor
    def logicalnot_with_temp(self, a, out, temp):
        op = self.builder.create_unary_with_temp_op(ir.Opcode.OP_LOGICALNOT, a, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    # Range and vector operations
    def range_op(self, start, step, size, out):
        op = self.builder.create_range_op(ir.Opcode.OP_RANGE, start, step, size, out)
        self.builder.emit(self.ctx, op)
        return op

    def vec_dup(self, value, out):
        op = self.builder.create_vec_dup_op(ir.Opcode.OP_VEC_DUP, value, out)
        self.builder.emit(self.ctx, op)
        return op

    def pow(self, a, b, out):
        op = self.builder.create_pow_op(ir.Opcode.OP_POW, a, b, out)
        self.builder.emit(self.ctx, op)
        return op

    # Gather operations
    def gather(self, lhs, rhs, out):
        op = self.builder.create_gather_op(ir.Opcode.OP_GATHER, lhs, rhs, out)
        self.builder.emit(self.ctx, op)
        return op

    def gather_extended(self, lhs, rhs, out):
        op = self.builder.create_gather_extended_op(ir.Opcode.OP_GATHER_FROM_UB, lhs, rhs, out)
        self.builder.emit(self.ctx, op)
        return op

    def gather_element(self, lhs, rhs, out):
        op = self.builder.create_gather_extended_op(ir.Opcode.OP_GATHER_ELEMENT, lhs, rhs, out)
        self.builder.emit(self.ctx, op)
        return op

    # Scatter operations
    def scatter_elements(self, src0, src1, scatter, out):
        op = self.builder.create_scatter_elements_op(ir.Opcode.OP_SCATTER_ELEMENT, src0, src1, scatter, out)
        self.builder.emit(self.ctx, op)
        return op

    def scatter(self, src0, src1, src2, out):
        op = self.builder.create_scatter_op(ir.Opcode.OP_SCATTER, src0, src1, src2, out)
        self.builder.emit(self.ctx, op)
        return op

    # Reduce operations
    def reduce(self, a, out):
        op = self.builder.create_reduce_op(ir.Opcode.OP_ROWMAXLINE, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def reduce_minline(self, a, out):
        op = self.builder.create_reduce_op(ir.Opcode.OP_ROWMINLINE, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def reduce_with_temp(self, a, out, temp):
        op = self.builder.create_reduce_with_temp_op(ir.Opcode.OP_ROWMAX_SINGLE, a, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def reduce_min_with_temp(self, a, out, temp):
        op = self.builder.create_reduce_with_temp_op(ir.Opcode.OP_ROWMIN_SINGLE, a, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def reduce_sum_with_temp(self, a, out, temp):
        op = self.builder.create_reduce_with_temp_op(ir.Opcode.OP_ROWSUM_SINGLE, a, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def reduce_sumline_with_temp(self, a, out, temp):
        op = self.builder.create_reduce_with_temp_op(ir.Opcode.OP_ROWSUMLINE, a, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    # Broadcast operations
    def broadcast_with_temp(self, a, b, out, temp):
        op = self.builder.create_broadcast_with_temp_op(ir.Opcode.OP_MAXIMUM, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def broadcast_minimum(self, a, b, out, temp):
        op = self.builder.create_broadcast_with_temp_op(ir.Opcode.OP_MINIMUM, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def broadcast_pairmax(self, a, b, out, temp):
        op = self.builder.create_broadcast_with_temp_op(ir.Opcode.OP_PAIRMAX, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def broadcast_pairmin(self, a, b, out, temp):
        op = self.builder.create_broadcast_with_temp_op(ir.Opcode.OP_PAIRMIN, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def broadcast_pairsum(self, a, b, out, temp):
        op = self.builder.create_broadcast_with_temp_op(ir.Opcode.OP_PAIRSUM, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    # Cast operation
    def cast(self, a, out):
        op = self.builder.create_cast_op(ir.Opcode.OP_CAST, a, out)
        self.builder.emit(self.ctx, op)
        return op

    # Where/Ternary operations
    def where_tt(self, condition, input_val, other, out, temp):
        op = self.builder.create_ternary_op(ir.Opcode.OP_WHERE_TT, condition, input_val, other, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def where_ts(self, condition, input_val, other, out, temp):
        op = self.builder.create_where_ts_op(ir.Opcode.OP_WHERE_TS, condition, input_val, other, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def where_st(self, condition, input_val, other, out, temp):
        op = self.builder.create_where_st_op(ir.Opcode.OP_WHERE_ST, condition, input_val, other, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def where_ss(self, condition, input_val, other, out, temp):
        op = self.builder.create_where_ss_op(ir.Opcode.OP_WHERE_SS, condition, input_val, other, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    # Compare operations
    def compare(self, a, b, out, temp):
        op = self.builder.create_compare_op(ir.Opcode.OP_CMP, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    def compare_scalar(self, a, b, out, temp):
        op = self.builder.create_compare_scalar_op(ir.Opcode.OP_CMPS, a, b, out, temp)
        self.builder.emit(self.ctx, op)
        return op

    # Matmul operations
    def matmul_extract(self, input_val, offsets, out):
        op = self.builder.create_matmul_extract_op(ir.Opcode.OP_L1_TO_L0A, input_val, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_extract_l0b(self, input_val, offsets, out):
        op = self.builder.create_matmul_extract_op(ir.Opcode.OP_L1_TO_L0B, input_val, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_extract_l0at(self, input_val, offsets, out):
        op = self.builder.create_matmul_extract_op(ir.Opcode.OP_L1_TO_L0_AT, input_val, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_extract_l0bt(self, input_val, offsets, out):
        op = self.builder.create_matmul_extract_op(ir.Opcode.OP_L1_TO_L0_BT, input_val, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_extract_l0c_to_l1(self, input_val, offsets, out):
        op = self.builder.create_matmul_extract_op(ir.Opcode.OP_L0C_TO_L1, input_val, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_mmad(self, lhs, rhs, out):
        op = self.builder.create_matmul_mmad_op(ir.Opcode.OP_A_MUL_B, lhs, rhs, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_acc(self, lhs, rhs, out):
        op = self.builder.create_matmul_acc_op(ir.Opcode.OP_A_MULACC_B, lhs, rhs, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_load(self, input_tensor, offsets, out):
        op = self.builder.create_matmul_load_op(ir.Opcode.OP_L1_COPY_IN, input_tensor, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    def matmul_store(self, input_tile, offsets, out):
        op = self.builder.create_matmul_store_op(ir.Opcode.OP_L0C_COPY_OUT, input_tile, offsets, out)
        self.builder.emit(self.ctx, op)
        return op

    # Transpose operations
    def transpose_movein(self, a, out):
        op = self.builder.create_transpose_op(ir.Opcode.OP_TRANSPOSE_MOVEIN, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def transpose_moveout(self, a, out):
        op = self.builder.create_transpose_op(ir.Opcode.OP_TRANSPOSE_MOVEOUT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    # Copy operations
    def copy_in(self, a, out):
        op = self.builder.create_copy_in_out_op(ir.Opcode.OP_COPY_IN, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def copy_out(self, a, out):
        op = self.builder.create_copy_in_out_op(ir.Opcode.OP_COPY_OUT, a, out)
        self.builder.emit(self.ctx, op)
        return op

    # UB copy operations (Tensor <-> Tile with offsets)
    def ub_copy_in(self, src_tensor, offsets, dst_tile):
        idx = 0
        args = self.last_func.get_sig().arguments
        for ele in args:
            if ele.id == src_tensor.id:
                break
            idx = idx + 1
        constant_idx = self.const(idx, "ub_copy_in_idx" + str(idx))
        gm_addr = self.scalar(ir.DataType.uint64, src_tensor.name + "Addr")
        self.call_scalar(constant_idx, out=gm_addr, call_type="GET_TENSOR_ADDR")

        op = self.builder.create_ub_copy_in_op(ir.Opcode.OP_UB_COPY_IN, src_tensor, offsets, dst_tile)
        self.builder.emit(self.ctx, op)
        return op

    def ub_copy_out(self, src_tile, offsets, dst_tensor):
        idx = 0
        args = self.last_func.get_sig().arguments
        for ele in args:
            if ele.id == dst_tensor.id:
                break
            idx = idx + 1
        constant_idx = self.const(idx, "ub_copy_out_idx" + str(idx))
        gm_addr = self.scalar(ir.DataType.uint64, dst_tensor.name + "Addr")
        self.call_scalar(constant_idx, out=gm_addr, call_type="GET_TENSOR_ADDR")

        op = self.builder.create_ub_copy_out_op(ir.Opcode.OP_UB_COPY_OUT, src_tile, offsets, dst_tensor)
        self.builder.emit(self.ctx, op)
        return op

    # AnyDataCopy operations
    def vld(self, a, out):
        op = self.builder.create_any_data_copy_op(ir.Opcode.OP_VLD, a, out)
        self.builder.emit(self.ctx, op)
        return op

    def vst(self, a, out):
        op = self.builder.create_any_data_copy_op(ir.Opcode.OP_VST, a, out)
        self.builder.emit(self.ctx, op)
        return op

    # Call operations
    def call_scalar(self, *args, out, call_type, is_void=False):
        num_args = len(args)

        if not (1 <= num_args <= 5):
            raise ValueError(f"call_scalar supports 1-5 arguments, but {num_args} were provided.")

        if is_void:
            opcode_name = f"OP_SCALAR_CALL_{num_args}_RETVOID"
        else:
            opcode_name = f"OP_SCALAR_CALL_{num_args}"

        opcode = getattr(ir.Opcode, opcode_name)
        method_name = f"create_call_{num_args}_scalar_op"
        builder_method = getattr(self.builder, method_name)

        op = builder_method(opcode, *args, out, call_type)
        self.builder.emit(self.ctx, op)
        return op