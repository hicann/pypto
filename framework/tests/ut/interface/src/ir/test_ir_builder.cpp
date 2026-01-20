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
 * \file test_ir_builder.cpp
 * \brief
 */

#include <cstdint>
#include <memory>
#include <unordered_set>
#include "gtest/gtest.h"
#include "ir/block_call.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;
namespace pto{

TEST(IRTEST, TestBuilder) {
    // ===== Module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    std::vector<int64_t> tileShape = { 128, 128 };

    auto inputTensor  = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    inputTensor->Attributes()["io"] = "in";
    auto scale1       = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    scale1->Attributes()["io"] = "in";
    auto result = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    result->Attributes()["io"] = "out";

    sig.arguments = { inputTensor, scale1, result };

    // ===== Function =====
    auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig);
    module->AddFunction(func);
    module->SetProgramEntry(func);

    // enter func scope + create an initial block as insertion point
    builder.EnterFunctionBody(ctx, func);

    // mul1_res = mul(input, scale1)
    auto mulVal1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "mul1_res");
    auto mulOp1 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTensor, scale1, mulVal1);
    builder.Emit(ctx, mulOp1);

    auto pi = builder.CreateConst(ctx, 3.14, "const_pi");

    // mul2_res = mul(mul1_res, pi)
    auto mulVal2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto mulOp2 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, mulVal1, pi, mulVal2);
    builder.Emit(ctx, mulOp2);

    builder.CreateReturn(ctx, { });

    ASSERT_EQ(ctx.func, func);
    ASSERT_EQ(ctx.compound, func->GetCompound());
    ASSERT_EQ(ctx.activeOpStmt, func->GetCompound()->GetStatement(0));

    ctx.PopScope();

    ASSERT_EQ(ctx.func, nullptr);
    ASSERT_EQ(ctx.compound, nullptr);
    ASSERT_EQ(ctx.activeOpStmt, nullptr);

    // ===== Program attributes =====
    module->Attributes()["arch"] = "\"PTOv2\"";
    module->Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module->Attributes()["enable_debug"] = "true";

    std::cout << *module << std::endl;
}

TEST(IRTEST, TestControlFlow) {
    // ===== Module =====
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    // ===== Signature =====
    FunctionSignature sig;

    // tensor<[b, 128], fp32>
    auto batch = std::make_shared<ScalarValue>(DataType::INT32, "batch", ScalarValueKind::Symbolic);
    auto constant128 = std::make_shared<ScalarValue>(int64_t(128), "const_128");
    std::vector<ScalarValuePtr> tensorShape = { batch, constant128 };

    std::vector<int64_t> tileShape = { 128, 128 };

    auto inputX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputX");
    inputX->Attributes()["io"] = "in";
    auto inputY = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputY");
    inputY->Attributes()["io"] = "in";
    auto scale1 = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    scale1->Attributes()["io"] = "in";
    auto scale2 = std::make_shared<ScalarValue>(DataType::FP32, "scale2", ScalarValueKind::Symbolic);
    scale2->Attributes()["io"] = "in";

    auto resultX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputX");
    resultX->Attributes()["io"] = "out";
    auto resultY = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputY");
    resultY->Attributes()["io"] = "out";

    sig.arguments = { inputX, inputY, scale1, scale2, resultX, resultY };

    sig.results.push_back(std::make_shared<ScalarValue>(DataType::INT32));

    // ===== Function =====
    auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig);
    module->AddFunction(func);
    module->SetProgramEntry(func);
        // 进入函数体作用域
    builder.EnterFunctionBody(ctx, func);

    // for i = 0 to batch step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);

    // test for attribute
    fs->Attributes()["unroll"] = "4";

    builder.EnterForBody(ctx, fs);

    // 目前没有 view / assemble，直接在整张 tensor 上做计算
    // outputX = add(inputX, scale1)
    auto resLoopX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
    auto addOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, resLoopX, scale1, resLoopX);
    builder.Emit(ctx, addOpX);

    // outputY = add(inputY, scale2)
    auto resLoopY = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputY");
    auto addOpY = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, resLoopY, scale2, resLoopY);
    builder.Emit(ctx, addOpY);

    // if i then outputX = mul(outputX, scale1) else outputY = mul(outputY, scale2)
    auto ifs = builder.CreateIfStmt(ctx, i);

    builder.EnterIfThen(ctx, ifs);

    auto resIfX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
    auto mulOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, resLoopX, scale1, resIfX);
    builder.Emit(ctx, mulOpX);

    // test compound remove value
    ifs->GetThenCompound()->RemoveValue(resIfX);
    ASSERT_EQ(ifs->GetThenCompound()->FindValue("outputX"), resLoopX);
    ifs->GetThenCompound()->SetEnvVar("outputX", resIfX);

    ctx.PopScope();


    builder.EnterIfElse(ctx, ifs);

    auto resIfY = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputY");
    auto mulOpY = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, resLoopY, scale2, resIfY);
    builder.Emit(ctx, mulOpY);

    ctx.PopScope();

    builder.ExitIfStatement(ctx, ifs);

    // check if then and else yield
    auto thenYield = std::dynamic_pointer_cast<YieldStatement>(ifs->GetThenCompound()->GetStatement(ifs->GetThenCompound()->GetStatementsNum() - 1));
    std::unordered_set<ValuePtr> thenYieldSet(thenYield->Values().begin(), thenYield->Values().end());
    std::unordered_set<ValuePtr> thenYieldSetGolden{resIfX, resLoopY};
    ASSERT_EQ(thenYieldSet, thenYieldSetGolden);

    auto elseYield = std::dynamic_pointer_cast<YieldStatement>(ifs->GetElseCompound()->GetStatement(ifs->GetElseCompound()->GetStatementsNum() - 1));
    std::unordered_set<ValuePtr> elseYieldSet(elseYield->Values().begin(), elseYield->Values().end());
    std::unordered_set<ValuePtr> elseYieldSetGolden{resLoopX, resIfY};
    ASSERT_EQ(elseYieldSet, elseYieldSetGolden);
    ASSERT_NE(elseYield, nullptr);
    ASSERT_GE(elseYield->Values().size(), 2);
    ASSERT_EQ(elseYield->Values()[0], resIfY);
    ASSERT_EQ(elseYield->Values()[1], resLoopX);

    ctx.PopScope(); // for-body

    builder.ExitForStatement(ctx, fs);

    // check for yield of for-statement: for 的结果应等于 if 的结果
    auto ifsInFor = std::dynamic_pointer_cast<IfStatement>(fs->GetCompound()->GetStatement(1));
    auto ifResults = ifsInFor->Results();
    auto forYields = fs->Yield()->Values();
    std::unordered_set<ValuePtr> ifResultSet(ifResults.begin(), ifResults.end());
    std::unordered_set<ValuePtr> forYieldSet(forYields.begin(), forYields.end());
    ASSERT_EQ(ifResultSet, forYieldSet);

    // return
    builder.CreateReturn(ctx, {constant0});

    ctx.PopScope(); // function-body

    std::cout << *module << std::endl;
}

std::shared_ptr<Function> TestBlockFunction(
    const std::vector<TensorValuePtr> &inputArgs,
    const std::vector<TensorValuePtr> &outputArgs,
    [[maybe_unused]]const std::vector<ScalarValuePtr> &indices) 
{
    IRBuilderContext ctx;
    IRBuilder builder;
    FunctionSignature sig = FunctionSignature(inputArgs, outputArgs);
    sig.results.push_back(std::make_shared<ScalarValue>(DataType::INT32));

    auto func = builder.CreateFunction("test_all_ops", FunctionKind::Block, sig);
    builder.EnterFunctionBody(ctx, func);
    // STUB LeafFuncAttribute
    auto blockFunc = std::dynamic_pointer_cast<pto::BlockFunction>(func);
    auto leafFuncAttr = std::make_shared<npu::tile_fwk::LeafFuncAttribute>();
    leafFuncAttr->coreType = CoreType::AIV;
    blockFunc->SetLeafFuncAttribute(leafFuncAttr);
    std::vector<int64_t> shape = {64, 64};
    // tensorAdd = add(input[0], input[1])
    auto tileAdd = builder.CreateTile(ctx, shape, DataType::FP32, "tensorAdd");
    auto tileAdd1 = builder.CreateTile(ctx, shape, DataType::FP32, "tensorAdd1");
    auto tileAdd2 = builder.CreateTile(ctx, shape, DataType::FP32, "tensorAdd2");
    auto addOp = builder.CreateBinaryOp(Opcode::OP_ADD, tileAdd1, tileAdd2, tileAdd);
    builder.Emit(ctx, addOp);

    auto tileOut = builder.CreateTile(ctx, shape, DataType::FP32, "tensorAdd2");
    auto divOp = builder.CreateBinaryOp(Opcode::OP_DIV, tileAdd1, tileAdd, tileOut);
    builder.Emit(ctx, divOp);

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    return func;
}

TEST(IRTEST, TestDynControlFlow)
{
    constexpr int LOOP_ITERATION = 8;
    std::vector<int64_t> shape = {512, 64 * LOOP_ITERATION};
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestNewIR";
    FUNCTION(funcName, {inputA, inputB}, {output}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, npu::tile_fwk::LoopRange(LOOP_ITERATION)) {
            CallBlock(TestBlockFunction, {inputA, inputB}, {output}, {i});
        }
    }
}

std::shared_ptr<Function> TestBlockFunctionCoa1(
    const std::vector<TensorValuePtr> &inputArgs,
    const std::vector<TensorValuePtr> &outputArgs,
    [[maybe_unused]]const std::vector<ScalarValuePtr> &indices)
{
    IRBuilderContext ctx;
    IRBuilder builder;

    auto createConst = [&](int n) {
        return builder.CreateConst(ctx, int64_t(n), "const_" + std::to_string(n));
    };

    FunctionSignature sig = FunctionSignature(inputArgs, outputArgs);
    sig.results.push_back(std::make_shared<ScalarValue>(DataType::INT32));

    auto func = builder.CreateFunction("test_coa1", FunctionKind::Block, sig);
    builder.EnterFunctionBody(ctx, func);
    // STUB LeafFuncAttribute
    auto blockFunc = std::dynamic_pointer_cast<pto::BlockFunction>(func);
    auto leafFuncAttr = std::make_shared<npu::tile_fwk::LeafFuncAttribute>();
    leafFuncAttr->coreType = CoreType::AIV;
    blockFunc->SetLeafFuncAttribute(leafFuncAttr);

    auto tensorAAddr = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorAAddr");
    auto tensorBAddr = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorBAddr");
    auto tensorCAddr = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorCAddr");
    auto scalarJ = builder.CreateScalar(ctx, DataType::INT64, "_SCALAR_J");
    auto scalarI = builder.CreateScalar(ctx, DataType::INT64, "_SCALAR_I");
    auto tensorARawShape_0 = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorARawShape_0");
    auto tensorBRawShape_1 = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorBRawShape_1");

    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(0), tensorAAddr,
        "GET_TENSOR_ADDR"));
    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(1), tensorBAddr,
        "GET_TENSOR_ADDR"));
    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(2), tensorCAddr,
        "GET_TENSOR_ADDR"));
    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(0), scalarJ,
        "GET_COA"));
    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(1), scalarI,
        "GET_COA"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, createConst(0), createConst(2), createConst(0), tensorARawShape_0,
        "GET_TENSOR_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, createConst(1), createConst(2), createConst(1), tensorBRawShape_1,
        "GET_TENSOR_RAWSHAPE_BY_IDX"));

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    return func;
}

std::shared_ptr<Function> TestBlockFunctionCoa2(
    const std::vector<TensorValuePtr> &inputArgs,
    const std::vector<TensorValuePtr> &outputArgs,
    [[maybe_unused]]const std::vector<ScalarValuePtr> &indices)
{
    IRBuilderContext ctx;
    IRBuilder builder;

    auto createConst = [&](int n) {
        return builder.CreateConst(ctx, int64_t(n), "const_" + std::to_string(n));
    };

    FunctionSignature sig = FunctionSignature(inputArgs, outputArgs);
    sig.results.push_back(std::make_shared<ScalarValue>(DataType::INT32));

    auto func = builder.CreateFunction("test_coa2", FunctionKind::Block, sig);
    builder.EnterFunctionBody(ctx, func);
    // STUB LeafFuncAttribute
    auto blockFunc = std::dynamic_pointer_cast<pto::BlockFunction>(func);
    auto leafFuncAttr = std::make_shared<npu::tile_fwk::LeafFuncAttribute>();
    leafFuncAttr->coreType = CoreType::AIV;
    blockFunc->SetLeafFuncAttribute(leafFuncAttr);

    auto tensorBAddr = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorBAddr");
    auto tensorCAddr = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorCAddr");
    auto scalarJ = builder.CreateScalar(ctx, DataType::INT64, "_SCALAR_J");
    auto tensorBRawShape_0 = builder.CreateScalar(ctx, DataType::UINT64, "_MARCRO_tensorBRawShape_0");

    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(0), tensorBAddr,
        "GET_TENSOR_ADDR"));
    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(1), tensorCAddr,
        "GET_TENSOR_ADDR"));
    builder.Emit(ctx, builder.CreateCall1ScalarOp(
        Opcode::OP_SCALAR_CALL_1, createConst(0), scalarJ,
        "GET_COA"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, createConst(0), createConst(2), createConst(0), tensorBRawShape_0,
        "GET_TENSOR_RAWSHAPE_BY_IDX"));

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    return func;
}

TEST(IRTEST, TestDynControlFlowCoa)
{
    constexpr int LOOP_ITERATION = 8;
    std::vector<int64_t> shape = {512, 64 * LOOP_ITERATION};
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestNewIRCoa";
    FUNCTION(funcName, {inputA, inputB}, {output}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, npu::tile_fwk::LoopRange(LOOP_ITERATION)) {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, npu::tile_fwk::LoopRange(LOOP_ITERATION)) {
                CallBlock(TestBlockFunctionCoa1, {inputA, inputB}, {output}, {j, i});
                CallBlock(TestBlockFunctionCoa2, {inputB}, {output}, {j});
            }
        }
    }
}
} // namespace pto
