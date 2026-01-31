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

namespace npu::tile_fwk {

class TestSetHeuristicTileShapes : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override {}
};

TEST_F(TestSetHeuristicTileShapes, TestCube) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSetHeuristicTileShapes", "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    // Prepare the graph
    std::vector<int64_t> inputAShape = {64, 128};
    std::vector<int64_t> inputBShape = {128, 64};
    std::vector<int64_t> outputCShape = {64, 64};

    auto inputA = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inputAShape);
    auto inputB = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inputBShape);
    auto outputC = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outputCShape);

    currFunctionPtr->AddOperation(Opcode::OP_A_MUL_B, {inputA, inputB}, {outputC});

    currFunctionPtr->inCasts_.push_back(inputA);
    currFunctionPtr->inCasts_.push_back(inputB);
    currFunctionPtr->outCasts_.push_back(outputC);

    // Run the pass
    SetHeuristicTileShapes setHeuristicTileShapes;
    auto status = setHeuristicTileShapes.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
}

TEST_F(TestSetHeuristicTileShapes, TestVector) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSetHeuristicTileShapes", "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    // Prepare the graph
    std::vector<int64_t> inputShape = {32, 8, 8};
    std::vector<int64_t> reshapeShape = {32, 64};
    std::vector<int64_t> rowmaxShape = {64, 1};
    std::vector<int64_t> outputShape = {1, 64};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inputShape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inputShape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, rowmaxShape);
    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, rowmaxShape);
    auto ubTensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outputShape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outputShape);

    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {ubTensor3});
    currFunctionPtr->AddOperation(Opcode::OP_ROWMAX, {ubTensor3}, {ubTensor4});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    auto &transpose = currFunctionPtr->AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {ubTensor5}, {ubTensor6});
    transpose.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{1, 0});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor6}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    // Run the pass
    SetHeuristicTileShapes setHeuristicTileShapes;
    auto status = setHeuristicTileShapes.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
}

} // namespace acend
