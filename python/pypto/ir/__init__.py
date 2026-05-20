#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) PyPTO Contributors.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


__all__ = [
    "IRBuilder",
    "IRNode",
    "Span",
    "Type",
    "Expr",
    "Stmt",
    "Var",
    "MemRef",
    "Function",
    "Program",

    "UnknownType",
    "ScalarType",
    "TensorType",
    "TupleType",
    "PtrType",
    "TokenType",
    "LogicalTensorType",
    "DataType",

    "FunctionType",
    "TensorLayout",
    "MemorySpace",
    "PipeType",
    "CoreType",
    "ConstInt",

    "ConstFloat",
    "ConstBool",
    "Call",
    "MakeTuple",
    "TupleGetItem",

    # BinaryExpr
    "Add",
    "Sub",
    "Mul",
    "FloorDiv",
    "FloorMod",
    "FloatDiv",
    "Min",
    "Max",
    "Pow",
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "And",
    "Or",
    "Xor",
    "BitAnd",
    "BitOr",
    "BitXor",
    "BitShiftLeft",
    "BitShiftRight",

    # UnaryExpr
    "Abs",
    "Neg",
    "Not",
    "BitNot",
    "Cast",

    # stmt
    "IterArg",
    "AssignStmt",
    "IfStmt",
    "YieldStmt",
    "ReturnStmt",
    "ForStmt",
    "WhileStmt",
    "SeqStmts",
    "EvalStmt",
    "BreakStmt",
    "ContinueStmt",
]

# --- Type classes ---
from ..pypto_impl.ir import (
    Type,
    UnknownType,
    ScalarType,
    TensorType,
    TupleType,
    PtrType,
    TokenType,
    LogicalTensorType,
)

# --- Data types ---
from ..pypto_impl.ir import DataType, Span

# --- Enums ---
from ..pypto_impl.ir import (
    FunctionType,
    TensorLayout,
    MemorySpace,
    PipeType,
    CoreType,
)

from ..pypto_impl.ir import IRBuilder

# --- Expression base & leaf classes ---
from ..pypto_impl.ir import (
    IRNode,
    Expr,
    Stmt,
    MemRef,
    Var,
    ConstInt,
    ConstFloat,
    ConstBool,
    Call,
    MakeTuple,
    TupleGetItemExpr as TupleGetItem,
)

# --- Binary expression ops ---
from ..pypto_impl.ir import (
    Add,
    Sub,
    Mul,
    FloorDiv,
    FloorMod,
    FloatDiv,
    Min,
    Max,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
    BitAnd,
    BitOr,
    BitXor,
    BitShiftLeft,
    BitShiftRight,
)

# --- Unary expression ops ---
from ..pypto_impl.ir import (
    Abs,
    Neg,
    Not,
    BitNot,
    Cast,
)

# --- Statement classes ---
from ..pypto_impl.ir import (
    IterArg,
    AssignStmt,
    IfStmt,
    YieldStmt,
    ReturnStmt,
    ForStmt,
    WhileStmt,
    SeqStmts,
    EvalStmt,
    BreakStmt,
    ContinueStmt,
    TensorOpStmt,
)

# --- Function / Program ---
from ..pypto_impl.ir import Function, Program

# --- DataType static instances ---
BOOL = DataType.BOOL
INT4 = DataType.INT4
INT8 = DataType.INT8
INT16 = DataType.INT16
INT32 = DataType.INT32
INT64 = DataType.INT64
UINT4 = DataType.UINT4
UINT8 = DataType.UINT8
UINT16 = DataType.UINT16
UINT32 = DataType.UINT32
UINT64 = DataType.UINT64
FP4 = DataType.FP4
FP8E4M3FN = DataType.FP8E4M3FN
FP8E5M2 = DataType.FP8E5M2
FP16 = DataType.FP16
FP32 = DataType.FP32
FP64 = DataType.FP64
BF16 = DataType.BF16
FP64 = DataType.FP64
HF4 = DataType.HF4
HF8 = DataType.HF8
INDEX = DataType.INDEX
