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
 * \file test_pass_check.cpp
 * \brief Unit test for pass checkers.
 */

#include "gtest/gtest.h"
#include <algorithm>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/inner/tile_shape.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "passes/pass_check/checker.h"

using namespace npu::tile_fwk;
using namespace std;

class PassCheckTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(PassCheckTest, TestCheckCompletenessWithoutIncast) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    exp_op1.UpdateSubgraphID(0);

    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
    EXPECT_EQ(checker.PublicCheck(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithNullIncast) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    currFunctionPtr->inCasts_.push_back(nullptr);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithIncastHasNoConsumer) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->inCasts_.push_back(incast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithoutOutcast) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PassCheckTest", "PassCheckTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp_op0 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast1}, {tensor1});
    exp_op0.UpdateSubgraphID(0);

    currFunctionPtr->inCasts_.push_back(incast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithNullOutcast) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {tensor1});
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(nullptr);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompletenessWithOutcastHasNoProducer) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {tensor1});
    currFunctionPtr->inCasts_.push_back(incast1);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestCheckGraphLoop) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast1}, {tensor1});
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {tensor2, outcast1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor2}, {tensor1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckGraphLoop(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestPublicCheck) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast1}, {tensor1});
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.PublicCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestCheckDynAttrForView) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {outcast1});
    auto &tensorOffset = incast1->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset()));

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckDynAttrForViewWithoutToDynValidShape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {16, 16};

    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {outcast1});
    Offset offset = {0, 128};
    std::vector<SymbolicScalar> dynOffset = {0, 128};
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, dynOffset));

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckToDynOffsetForAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast1}, {outcast1});
    auto &tensorOffset = incast1->GetTensorOffset();
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(tensorOffset.GetOffset()));

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckToDynOffsetForAssemble(*currFunctionPtr), FAILED);
}