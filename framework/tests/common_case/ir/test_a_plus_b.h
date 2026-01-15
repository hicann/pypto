/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_a_plus_b.cpp
 * \brief
 */

#pragma once

#include "ir/opcode.h"
#include "ir/serializer.h"

#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"

namespace pto {

ProgramModulePtr CreateAdd(const std::string &funcName = "func") {
    auto prog = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    auto createConst = [&](int n) {
        return builder.CreateConst(ctx, int64_t(n), "const_" + std::to_string(n));
    };

    FunctionSignature sig;

    // ===== Function =====
    auto func = builder.CreateFunction(funcName, FunctionKind::Block, sig);
    prog->AddFunction(func);
    prog->SetProgramEntry(func);
        // 进入函数体作用域
    builder.EnterFunctionBody(ctx, func);

    auto param = builder.CreateScalar(ctx, DataType::INT64, "param");

    auto pipeV = builder.CreateScalar(ctx, DataType::INT32, "PIPE_V");
    auto pipeMTE2 = builder.CreateScalar(ctx, DataType::INT32, "PIPE_MTE2");
    auto pipeMTE3 = builder.CreateScalar(ctx, DataType::INT32, "PIPE_MTE3");
    auto eventID0 = builder.CreateScalar(ctx, DataType::INT32, "EVENT_ID0");
    auto voidValue = builder.CreateScalar(ctx, DataType::INT32, "_");

    auto sym_72_dim_0 = builder.CreateScalar(ctx, DataType::UINT64, "sym_72_dim_0");
    auto sym_72_dim_1 = builder.CreateScalar(ctx, DataType::UINT64, "sym_72_dim_1");
    auto sym_876_dim_0 = builder.CreateScalar(ctx, DataType::UINT64, "sym_876_dim_0");
    auto sym_876_dim_1 = builder.CreateScalar(ctx, DataType::UINT64, "sym_876_dim_1");

    auto gmt5Addr = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_gmt5Addr");
    auto gmt5Rawshape_0 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_gmt5Rawshape_0");
    auto gmt5Rawshape_1 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_gmt5Rawshape_1");

    auto gmt1Addr = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_gmt1Addr");
    auto gmt1Rawshape_0 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_gmt1Rawshape_0");
    auto gmt1Rawshape_1 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_gmt1Rawshape_1");

    auto copyInOffset_0 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_copyInOffset_0");
    auto copyInOffset_1 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_copyInOffset_1");

    auto copyOutOffset_0 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_copyOutOffset_0");
    auto copyOutOffset_1 = builder.CreateScalar(ctx, DataType::UINT64, "_MACRO_copyOutOffset_1");

    auto gmt5 = builder.CreateTensor(ctx, {gmt5Rawshape_0, gmt5Rawshape_1}, DataType::FP32, "_MACRO_gmt5");
    auto gmt1 = builder.CreateTensor(ctx, {gmt1Rawshape_0, gmt1Rawshape_1}, DataType::FP32, "_MACRO_gmt1");
    auto ubt0 = builder.CreateTile(ctx, {32, 32}, DataType::FP32, "ubt0");
    ubt0->SetValidShapes({sym_72_dim_0, sym_72_dim_1});

    auto ubt0mem = std::make_shared<Memory>(0x1000, MemSpaceKind::UB);
    ubt0mem->SetAddr(0x0);
    ubt0->SetMemory(ubt0mem);

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, param, createConst(1), createConst(10), gmt5Addr,
        "GET_PARAM_ADDR"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(1), createConst(10), createConst(2), createConst(0), gmt5Rawshape_0,
        "GET_PARAM_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(1), createConst(10), createConst(2), createConst(1), gmt5Rawshape_1,
        "GET_PARAM_RAWSHAPE_BY_IDX"));

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, param, createConst(0), createConst(1), gmt1Addr,
        "GET_PARAM_ADDR"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(0), createConst(1), createConst(2), createConst(0), gmt1Rawshape_0,
        "GET_PARAM_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(0), createConst(1), createConst(2), createConst(1), gmt1Rawshape_1,
        "GET_PARAM_RAWSHAPE_BY_IDX"));

    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(1), createConst(32), createConst(2), createConst(1), createConst(0), sym_72_dim_0,
        "RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST"));
    builder.Emit(ctx, builder.CreateUnaryScalarOp(Opcode::OP_SCALAR_ASSIGN, sym_72_dim_0, sym_72_dim_1));
    builder.Emit(ctx, builder.CreateUnaryScalarOp(Opcode::OP_SCALAR_ASSIGN, sym_72_dim_0, sym_876_dim_0));
    builder.Emit(ctx, builder.CreateUnaryScalarOp(Opcode::OP_SCALAR_ASSIGN, sym_72_dim_0, sym_876_dim_1));

    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(0), createConst(0), createConst(2), createConst(1), createConst(0), copyInOffset_0,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(0), createConst(0), createConst(2), createConst(1), createConst(1), copyInOffset_1,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));

    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(1), createConst(0), createConst(2), createConst(10), createConst(0), copyOutOffset_0,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(1), createConst(0), createConst(2), createConst(10), createConst(1), copyOutOffset_1,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));

    builder.Emit(ctx, builder.CreateUBCopyInOp(Opcode::OP_UB_COPY_IN, gmt1, {copyInOffset_0, copyInOffset_1}, ubt0));

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeMTE2, pipeV, eventID0, voidValue, "set_flag"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeMTE2, pipeV, eventID0, voidValue, "wait_flag"));

    builder.Emit(ctx, builder.CreateBinaryOp(Opcode::OP_ADD, ubt0, ubt0, ubt0));

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeV, pipeMTE3, eventID0, voidValue, "set_flag"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeV, pipeMTE3, eventID0, voidValue, "wait_flag"));

    builder.Emit(ctx, builder.CreateUBCopyOutOp(Opcode::OP_UB_COPY_OUT, ubt0, {copyOutOffset_0, copyOutOffset_1}, gmt5));

    builder.CreateReturn(ctx, {createConst(0)});
    ctx.PopScope(); // function-body
    return prog;
}

}
