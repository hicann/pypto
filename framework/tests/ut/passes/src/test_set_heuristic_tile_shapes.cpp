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
 * \file test_set_heuristic_tile_shapes.cpp
 * \brief Unit test for SetHeuristicTileShapes.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/set_heuristic_tile_shapes.h"
#include "computational_graph_builder.h"
#include "ut_json/ut_json_tool.h"
#include "interface/tensor/irbuilder.h"

namespace npu::tile_fwk {

class TestSetHeuristicTileShapes : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(TestSetHeuristicTileShapes, TestCube)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestSetHeuristicTileShapes", "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inputAShape = {64, 128};
    std::vector<int64_t> inputBShape = {128, 64};
    std::vector<int64_t> outputCShape = {64, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputAShape, CreateTestConstIntVector(inputAShape));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputBShape, CreateTestConstIntVector(inputBShape));
    auto outputC = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputCShape, CreateTestConstIntVector(outputCShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB}, {outputC});

    currFunctionPtr->inCasts_.push_back(inputA);
    currFunctionPtr->inCasts_.push_back(inputB);
    currFunctionPtr->outCasts_.push_back(outputC);

    // Run the pass
    SetHeuristicTileShapes setHeuristicTileShapes;
    auto status = setHeuristicTileShapes.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(TestSetHeuristicTileShapes, TestVector)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestSetHeuristicTileShapes", "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inputShape = {32, 8, 8};
    std::vector<int64_t> reshapeShape = {32, 64};
    std::vector<int64_t> rowmaxShape = {64, 1};
    std::vector<int64_t> outputShape = {1, 64};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, rowmaxShape, CreateTestConstIntVector(rowmaxShape));
    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, rowmaxShape, CreateTestConstIntVector(rowmaxShape));
    auto ubTensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {ubTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ROWMAX, {ubTensor3}, {ubTensor4});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    std::vector<int> shapeVal = {1, 0};
    std::string shapeKey = OP_ATTR_PREFIX + "shape";
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_TRANSPOSE_VNCHWCONV, {ubTensor5}, {ubTensor6},
        [&shapeKey, &shapeVal](Operation& op) {
            op.SetAttribute(shapeKey, shapeVal);
        });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor6}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    // Run the pass
    SetHeuristicTileShapes setHeuristicTileShapes;
    auto status = setHeuristicTileShapes.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(TestSetHeuristicTileShapes, TestSemanticLabel)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestSetHeuristicTileShapes", "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inputAShape = {64, 128};
    std::vector<int64_t> inputBShape = {128, 64};
    std::vector<int64_t> outputCShape = {64, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputAShape, CreateTestConstIntVector(inputAShape));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputBShape, CreateTestConstIntVector(inputBShape));
    auto outputC = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputCShape, CreateTestConstIntVector(outputCShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB}, {outputC});

    std::shared_ptr<SemanticLabel> label = std::make_shared<SemanticLabel>("test", "test", 10);
    std::cout << currFunctionPtr->GetSortedOperations().size() << std::endl;

    for (auto& op : currFunctionPtr->GetSortedOperations()) {
        op->SetSemanticLabel(label);
    }

    currFunctionPtr->inCasts_.push_back(inputA);
    currFunctionPtr->inCasts_.push_back(inputB);
    currFunctionPtr->outCasts_.push_back(outputC);

    // Run the pass
    SetHeuristicTileShapes setHeuristicTileShapes;
    auto status = setHeuristicTileShapes.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(TestSetHeuristicTileShapes, TestPythonJsonGeneration)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestSetHeuristicTileShapes", "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inputAShape = {64, 128};
    std::vector<int64_t> inputBShape = {128, 64};
    std::vector<int64_t> outputCShape = {64, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputAShape, CreateTestConstIntVector(inputAShape));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputBShape, CreateTestConstIntVector(inputBShape));
    auto outputC = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputCShape, CreateTestConstIntVector(outputCShape));

    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);
    ir::Span::SetCurrent(ir::Span("noexist.cpp", 1, 0));
    auto& add_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB}, {outputC});
    add_op.tileShape_.SetCubeTile({64, 64}, {64, 64}, {64, 64});

    currFunctionPtr->inCasts_.push_back(inputA);
    currFunctionPtr->inCasts_.push_back(inputB);
    currFunctionPtr->outCasts_.push_back(outputC);

    // Run the pass
    SetHeuristicTileShapes setHeuristicTileShapes;
    auto status = setHeuristicTileShapes.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
}

} // namespace npu::tile_fwk
