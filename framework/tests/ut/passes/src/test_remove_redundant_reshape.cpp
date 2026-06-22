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
 * \file test_remove_redundant_reshape.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include <fstream>
#include <vector>
#include <string>
#define private public
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"

using namespace npu::tile_fwk;

class RemoveRedundantReshapeTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ReshapeTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

protected:
    static Operation* FindOpByOpcode(const OperationsViewer& ops, Opcode opcode)
    {
        for (auto& op : ops) {
            if (op.GetOpcode() == opcode) {
                return const_cast<Operation*>(&op);
            }
        }
        return nullptr;
    }

    static bool IsOpRemoved(const OperationsViewer& ops, int opmagic)
    {
        for (const auto& op : ops) {
            if (op.opmagic == opmagic) {
                return false;
            }
        }
        return true;
    }

    static Operation* FindOpByMagic(const OperationsViewer& ops, int opmagic)
    {
        for (auto& op : ops) {
            if (op.opmagic == opmagic) {
                return const_cast<Operation*>(&op);
            }
        }
        return nullptr;
    }
};

TEST_F(RemoveRedundantReshapeTest, TestReshapeChain)
{
    // Define Tensor shapes
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};
    std::vector<int64_t> shape3{1, 128, 1024};
    // Create Tensors
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_B(DT_FP32, shape3, "out_tensor_B");

    // Initialize PassManager
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy(
        "ReshapeTestStrategy", {
                                   {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                               });
    ConfigManager::Instance();

    // Create and configure the function
    FUNCTION("ReshapeChainFunction")
    {
        Tensor out_tensor_A = Reshape(in_tensor, shape2);
        out_tensor_B = Reshape(out_tensor_A, shape3);
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReshapeChainFunction");

    auto updated_operations = currentFunction->Operations();

    EXPECT_EQ(updated_operations.size(), 3)
        << "After the Pass, there should be 3 operations (View + Assemble + Reshape)";
    Operation* view_op = FindOpByOpcode(updated_operations, Opcode::OP_VIEW);
    Operation* assemble_op = FindOpByOpcode(updated_operations, Opcode::OP_ASSEMBLE);
    Operation* reshape_op = FindOpByOpcode(updated_operations, Opcode::OP_RESHAPE);
    ASSERT_NE(view_op, nullptr) << "View operation should be kept";
    ASSERT_NE(assemble_op, nullptr) << "Assemble operation should be kept";
    ASSERT_NE(reshape_op, nullptr) << "Reshape operation should be kept";

    EXPECT_EQ(view_op->GetIOperands()[0]->shape, shape1)
        << "The input shape of View should be the same as shape1";
    EXPECT_EQ(view_op->GetOOperands()[0]->shape, shape1)
        << "Without matmul, view/reshape/assemble reorder should be skipped";
    EXPECT_EQ(reshape_op->GetIOperands()[0]->shape, shape1)
        << "Without matmul, final Reshape should consume the original shape";
    EXPECT_EQ(reshape_op->GetOOperands()[0]->shape, shape3)
        << "The output shape of final Reshape should be the same as shape3";
}

TEST_F(RemoveRedundantReshapeTest, TestReplaceInput)
{
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};
    std::vector<int64_t> shape3{1, 128, 1024};

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_B(DT_FP32, shape2, "out_tensor_B");
    Tensor out_tensor_C(DT_FP32, shape3, "out_tensor_C");

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy(
        "ReshapeTestStrategy", {{"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE}});

    TileShape::Current().SetVecTile({1, 64, 64});
    int64_t first_reshape_magic = -1;
    FUNCTION("ReplaceInputFunction")
    {
        Tensor out_tensor_A = Reshape(in_tensor, shape2);
        out_tensor_B = Reshape(out_tensor_A, shape3);
        Tensor add_tensor(DT_FP32, shape2, "add_tensor");
        out_tensor_C = Add(out_tensor_A, add_tensor);

        auto operations = Program::GetInstance().GetCurrentFunction()->Operations();
        for (const auto& op : operations) {
            if (op.GetOpcodeStr() == "RESHAPE") {
                first_reshape_magic = op.opmagic;
                break;
            }
        }
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReplaceInputFunction");
    auto updated_operations = currentFunction->Operations();

    EXPECT_EQ(updated_operations.size(), 6)
        << "Without matmul, view/reshape/assemble reorder should be skipped";
    EXPECT_FALSE(IsOpRemoved(updated_operations, first_reshape_magic))
        << "Without matmul, the first Reshape operation should be kept";

    Operation* add_op = FindOpByOpcode(updated_operations, Opcode::OP_ADD);
    Operation* view_op = FindOpByOpcode(updated_operations, Opcode::OP_VIEW);
    ASSERT_NE(add_op, nullptr) << "Add operation should be present";
    ASSERT_NE(view_op, nullptr) << "View operation should be present";
    EXPECT_EQ(add_op->GetIOperands()[0]->shape, shape2)
        << "The input shape of Add should be the same as shape2";
}

/*
 * View->Reshape reorder with MatMul present.
 * Before: input{32,64} -> view -> middle{16,64} -> reshape -> output{1024}
 * After:  input{32,64} -> reshape -> newMid{2048} -> view -> output{1024}
 * A separate OP_A_MUL_B is present to trigger HasTargetMatmulOp.
 */
TEST_F(RemoveRedundantReshapeTest, TestViewReshapeReorderWithMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestViewReshapeReorder", "TestViewReshapeReorder", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> inputShape = {32, 64};
    std::vector<int64_t> middleShape = {16, 64};
    std::vector<int64_t> outputShape = {1024};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input}, {middle});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    int viewMagic = viewOp.GetOpMagic();

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {middle}, {output});
    int reshapeMagic = reshapeOp.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->inCasts_.push_back(matmulA);
    currFunctionPtr->inCasts_.push_back(matmulB);
    currFunctionPtr->outCasts_.push_back(output);
    currFunctionPtr->outCasts_.push_back(matmulC);

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto ops = currFunctionPtr->Operations();
    EXPECT_TRUE(IsOpRemoved(ops, viewMagic)) << "Original view should be deleted after reorder";
    EXPECT_TRUE(IsOpRemoved(ops, reshapeMagic)) << "Original reshape should be deleted after reorder";

    int viewCount = 0, reshapeCount = 0, matmulCount = 0;
    Operation* newReshape = nullptr;
    Operation* newView = nullptr;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_VIEW) { viewCount++; newView = const_cast<Operation*>(&op); }
        else if (op.GetOpcode() == Opcode::OP_RESHAPE) { reshapeCount++; newReshape = const_cast<Operation*>(&op); }
        else if (op.GetOpcode() == Opcode::OP_A_MUL_B) { matmulCount++; }
    }
    EXPECT_EQ(reshapeCount, 1) << "Should have exactly one new reshape after reorder";
    EXPECT_EQ(viewCount, 1) << "Should have exactly one new view after reorder";
    EXPECT_EQ(matmulCount, 1) << "MatMul should be preserved";

    ASSERT_NE(newReshape, nullptr);
    EXPECT_EQ(newReshape->GetIOperands()[0], input) << "New reshape should consume the original view input";
    ASSERT_NE(newView, nullptr);
    EXPECT_EQ(newView->GetOOperands()[0], output) << "New view should produce the original reshape output";
}

/*
 * Reshape->Assemble reorder with MatMul present.
 * Before: input{2048} -> reshape -> middle{32,64} -> assemble -> output{32,64}
 * After:  input{2048} -> assemble -> newAsmOut{2048} -> reshape -> output{32,64}
 */
TEST_F(RemoveRedundantReshapeTest, TestReshapeAssembleReorderWithMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestReshapeAssembleReorder", "TestReshapeAssembleReorder", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> inputShape = {2048};
    std::vector<int64_t> middleShape = {32, 64};
    std::vector<int64_t> outputShape = {32, 64};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {input}, {middle});
    int reshapeMagic = reshapeOp.GetOpMagic();

    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {middle}, {output});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    int assembleMagic = assembleOp.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->inCasts_.push_back(matmulA);
    currFunctionPtr->inCasts_.push_back(matmulB);
    currFunctionPtr->outCasts_.push_back(output);
    currFunctionPtr->outCasts_.push_back(matmulC);

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto ops = currFunctionPtr->Operations();
    EXPECT_TRUE(IsOpRemoved(ops, reshapeMagic)) << "Original reshape should be deleted after reorder";
    EXPECT_TRUE(IsOpRemoved(ops, assembleMagic)) << "Original assemble should be deleted after reorder";

    int assembleCount = 0, reshapeCount = 0, matmulCount = 0;
    Operation* newAssemble = nullptr;
    Operation* newReshape = nullptr;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) { assembleCount++; newAssemble = const_cast<Operation*>(&op); }
        else if (op.GetOpcode() == Opcode::OP_RESHAPE) { reshapeCount++; newReshape = const_cast<Operation*>(&op); }
        else if (op.GetOpcode() == Opcode::OP_A_MUL_B) { matmulCount++; }
    }
    EXPECT_EQ(assembleCount, 1) << "Should have exactly one new assemble after reorder";
    EXPECT_EQ(reshapeCount, 1) << "Should have exactly one new reshape after reorder";
    EXPECT_EQ(matmulCount, 1) << "MatMul should be preserved";

    ASSERT_NE(newAssemble, nullptr);
    EXPECT_EQ(newAssemble->GetIOperands()[0], input) << "New assemble should consume the original reshape input";
    ASSERT_NE(newReshape, nullptr);
    EXPECT_EQ(newReshape->GetOOperands()[0], output) << "New reshape should produce the original assemble output";
}

struct FanoutGraphInfo {
    std::shared_ptr<Function> func;
    LogicalTensorPtr input;
    int viewMagic;
    int reshapeMagic;
    int fanoutView1Magic;
    int fanoutView2Magic;
};

static FanoutGraphInfo BuildViewReshapeFanoutGraph()
{
    FanoutGraphInfo info;
    info.func = std::make_shared<Function>(
        Program::GetInstance(), "TestViewReshapeFanout", "TestViewReshapeFanout", nullptr);

    std::vector<int64_t> inputShape = {4, 32};
    std::vector<int64_t> middleShape = {4, 16};
    std::vector<int64_t> reshapeOutShape = {64};
    std::vector<int64_t> fanoutShape = {32};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto reshapeOut = IRBuilder().CreateTensorVar(DT_FP32, reshapeOutShape, CreateTestConstIntVector(reshapeOutShape));
    auto fanout1 = IRBuilder().CreateTensorVar(DT_FP32, fanoutShape, CreateTestConstIntVector(fanoutShape));
    auto fanout2 = IRBuilder().CreateTensorVar(DT_FP32, fanoutShape, CreateTestConstIntVector(fanoutShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    info.input = input;

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_VIEW, {input}, {middle});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 8}));
    info.viewMagic = viewOp.GetOpMagic();

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_RESHAPE, {middle}, {reshapeOut});
    info.reshapeMagic = reshapeOp.GetOpMagic();

    auto& fanoutView1 = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_VIEW, {reshapeOut}, {fanout1});
    fanoutView1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0}));
    info.fanoutView1Magic = fanoutView1.GetOpMagic();

    auto& fanoutView2 = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_VIEW, {reshapeOut}, {fanout2});
    fanoutView2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32}));
    info.fanoutView2Magic = fanoutView2.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    info.func->inCasts_.push_back(input);
    info.func->inCasts_.push_back(matmulA);
    info.func->inCasts_.push_back(matmulB);
    info.func->outCasts_.push_back(fanout1);
    info.func->outCasts_.push_back(fanout2);
    info.func->outCasts_.push_back(matmulC);
    return info;
}

/*
 * View->Reshape fanout reorder with MatMul present.
 * Before: input{4,32} -> view(offset={0,8}) -> middle{4,16} -> reshape -> reshapeOut{64}
 *         reshapeOut -> fanoutView1(offset={0})  -> fanout1{32}
 *         reshapeOut -> fanoutView2(offset={32}) -> fanout2{32}
 * After:  input{4,32} -> reshape -> newMid{128}
 *         newMid -> fanoutView1' -> fanout1{32}
 *         newMid -> fanoutView2' -> fanout2{32}
 * Non-contiguous collapsed groups trigger the fanout path.
 */
TEST_F(RemoveRedundantReshapeTest, TestViewReshapeFanoutWithMatmul)
{
    auto info = BuildViewReshapeFanoutGraph();
    ASSERT_NE(info.func, nullptr);

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*info.func), SUCCESS);

    auto ops = info.func->Operations();
    EXPECT_TRUE(IsOpRemoved(ops, info.viewMagic)) << "Original view should be deleted after fanout reorder";
    EXPECT_TRUE(IsOpRemoved(ops, info.reshapeMagic)) << "Original reshape should be deleted after fanout reorder";
    EXPECT_TRUE(IsOpRemoved(ops, info.fanoutView1Magic)) << "Original fanout view1 should be deleted";
    EXPECT_TRUE(IsOpRemoved(ops, info.fanoutView2Magic)) << "Original fanout view2 should be deleted";

    int viewCount = 0, reshapeCount = 0, matmulCount = 0;
    Operation* newReshape = nullptr;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_VIEW) { viewCount++; }
        else if (op.GetOpcode() == Opcode::OP_RESHAPE) { reshapeCount++; newReshape = const_cast<Operation*>(&op); }
        else if (op.GetOpcode() == Opcode::OP_A_MUL_B) { matmulCount++; }
    }
    EXPECT_EQ(reshapeCount, 1) << "Should have exactly one new reshape after fanout reorder";
    EXPECT_EQ(viewCount, 2) << "Should have two new fanout views after reorder";
    EXPECT_EQ(matmulCount, 1) << "MatMul should be preserved";

    ASSERT_NE(newReshape, nullptr);
    EXPECT_EQ(newReshape->GetIOperands()[0], info.input) << "New reshape should consume the original view input";
}
