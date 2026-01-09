/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_operation.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include <vector>

#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/value.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"

namespace pto{

TEST(IRTEST, TestTensorOperation){
    // ===== Program module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // ===== Function signature =====
    FunctionSignature sig;

    // Input tensor: tensor<[B, 128], f32>
    auto B = std::make_shared<ScalarValue>(DataType::INT32, "B", ScalarValueKind::Symbolic);
    std::vector<uint64_t> tileShape = { 128, 128 };
    auto inputTensor =
        std::make_shared<TileValue>(tileShape, DataType::FP32, "input");

    sig.arguments.push_back(inputTensor);

    // Output tensor
    auto outputTensor =
        std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    sig.arguments.push_back(outputTensor);

    // ===== Function =====
    auto func = builder.CreateFunction("test_all_ops", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    builder.EnterFunctionBody(ctx, func);

    auto c2 = builder.CreateConst(ctx, 2.0, "c2");
    auto c3 = builder.CreateConst(ctx, 3.0, "c3");

    // tensorAdd = add(input, c2)
    auto tensorAdd = builder.CreateTile(ctx, tileShape, DataType::FP32, "tensorAdd");
    auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, inputTensor, c2, tensorAdd);
    builder.Emit(ctx, addOp);

    // tensorSub = sub(tensorAdd, c3)
    auto tensorSub = builder.CreateTile(ctx, tileShape, DataType::FP32, "tensorSub");
    auto subOp = builder.CreateBinaryScalarMixOp(Opcode::OP_SUBS, tensorAdd, c3, tensorSub);
    builder.Emit(ctx, subOp);

    // tensorMul = mul(tensorSub, c2)
    auto tensorMul = builder.CreateTile(ctx, tileShape, DataType::FP32, "tensorMul");
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, tensorSub, c2, tensorMul);
    builder.Emit(ctx, mulOp);

    // tensorDiv = div(tensorMul, c2)
    auto tensorDiv = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto divOp = builder.CreateBinaryScalarMixOp(Opcode::OP_DIVS, tensorMul, c2, tensorDiv);
    builder.Emit(ctx, divOp);

    // return tensorDiv
    builder.CreateReturn(ctx, { });

    ctx.PopScope();

    ASSERT_EQ(func->GetCompound()->FindValue("output"), tensorDiv);

    std::cout << *module << std::endl;
}

};