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
 * \file test_duplicate_op.cpp
 * \brief Unit test for DuplicateOp pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_optimization/duplicate_op.h"

#define private public
namespace npu {
namespace tile_fwk{
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t KNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumSix = 6u;
static const uint16_t kNumSeven = 7u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumEleven = 11u;
static const uint16_t kNumForteen = 14u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

class TestDuplicateOpPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "DuplicateOpTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

/*
TESTDuplicateViewSingleConsumer
inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast{1,8,16}

inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor});
    auto &tensorOffset = inCast->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), ubTensor->GetDynValidShape()));
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(viewNum, kNumOne);
}

/*
TESTDuplicateViewThreeConsumer
inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast1{1,8,16}
                                    ->view->outCast2{8,16}
                                    ->sqrt->outCast3{1,8,16}
inCast{8,16}->view->ubTensor{1,8,16}->view->outCast2{8,16}
            ->view->viewTensor1{1,8,16}->exp->outCast1{1,8,16}                        
            ->view->viewTensor2{1,8,16}->sqrt->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor});
    auto &tensorOffset = inCast->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), ubTensor->GetDynValidShape()));
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor}, {outCast2});
    auto &tensorOffset1 = ubTensor->GetTensorOffset();
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), outCast2->GetDynValidShape()));
    auto &sqrtOp = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast3});
    auto &expOp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast1});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast1);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    // 旧的两个 + 新的两个
    EXPECT_EQ(viewNum, kNumFour);
    EXPECT_EQ(expOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(expOp.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(viewOp1.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(viewOp1.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(sqrtOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(sqrtOp.GetInputOperand(kSizeZero), ubTensor);
}

/*
TESTDuplicateViewAlternativeConsumer
inCast{8,16}->view->ubTensor1{1,8,16}
            ->view->ubTensor2{1,8,16}
ubTensor1+ubTensor1->div->outCast1{1,8,16}
ubTensor1+ubTensor2->div->outCast2{1,8,16}
ubTensor2+ubTensor2->div->outCast3{1,8,16}

inCast{8,16}->view->ubTensor1'{1,8,16}
            ->view->ubTensor2'{1,8,16}
            ->view->ubTensor3'{1,8,16}
            ->view->ubTensor4'{1,8,16}
            ->view->ubTensor5'{1,8,16}
            ->view->ubTensor6'{1,8,16}
ubTensor1'+ubTensor2'->div->outCast1{1,8,16}
ubTensor3'+ubTensor4'->div->outCast2{1,8,16}
ubTensor5'+ubTensor6'->div->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor1});
    auto &tensorOffset = inCast->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), ubTensor1->GetDynValidShape()));
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor2});
    auto &tensorOffset1 = inCast->GetTensorOffset();
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor2->GetDynValidShape()));
    auto &div1 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor1}, {outCast1});
    auto &div2 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor2}, {outCast2});
    auto &div3 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor2, ubTensor2}, {outCast3});
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast1);
    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
    uint32_t viewNum = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(viewNum, kNumFour);
    EXPECT_EQ(div1.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div2.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div3.GetInputOperandSize(), kNumTwo);
    auto div1Input1 = div1.GetInputOperand(kSizeZero);
    auto div1Input2 = div1.GetInputOperand(kSizeOne);
    EXPECT_NE(div1Input1, ubTensor1);
    EXPECT_EQ(div1Input1, div1Input2);
    auto div2Input1 = div2.GetInputOperand(kSizeZero);
    auto div2Input2 = div2.GetInputOperand(kSizeOne);
    EXPECT_NE(div2Input2, ubTensor2);
    EXPECT_NE(div2Input1, div2Input2);
    auto div3Input1 = div3.GetInputOperand(kSizeZero);
    auto div3Input2 = div3.GetInputOperand(kSizeOne);
    EXPECT_NE(div3Input2, ubTensor2);
    EXPECT_EQ(div3Input1, div3Input2);
}

/*
incast    ->view  -> tensor1 -> exp  -> outcast1
                             -> exp  -> outcast2
incast    ->view  -> tensor1 -> exp  -> outcast1
                             -> exp  -> outcast2
*/
TEST_F(TestDuplicateOpPass, TestDupViewL1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDupViewL1", "TestDupViewL1", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {tensor1});
    auto &expOp1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    auto &expOp2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    (void) expOp1;
    (void) expOp2;
    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, 
                                                       MEM_L1, 
                                                       std::vector<SymbolicScalar>(), 
                                                       std::vector<SymbolicScalar>());
    viewOp.SetOpAttribute(viewAttr);
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    DuplicateOp duplicateoppass;
    duplicateoppass.RunOnFunction(*currFunctionPtr);
    int viewNum = 0;
    auto opList = currFunctionPtr->Operations();
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewNum++;
        }
    }
    EXPECT_EQ(viewNum, 1);
}

/*
TESTDuplicateGatherinSingleConsumer
inCast{8,16}->Gatherin->ubTensor{8,16}->exp->outCast{8,16}

inCast{8,16}->Gatherin->ubTensor{8,16}->exp->outCast{8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
    }
    EXPECT_EQ(gatherinNum, kNumOne);
}

/*
TESTDuplicateGatherinThreeConsumer
inCast{8,16}->Gatherin->ubTensor{1,8,16}->exp->outCast1{1,8,16}
                                    ->view->outCast2{8,16}
                                    ->sqrt->outCast3{1,8,16}
inCast{8,16}->Gatherin->ubTensor{1,8,16}->view->outCast2{8,16}
            ->Gatherin->GatherinTensor1{1,8,16}->exp->outCast1{1,8,16}
            ->Gatherin->GatherinTensor2{1,8,16}->sqrt->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t j = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &gatherinOp = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor});
    gatherinOp.SetAttribute(OpAttributeKey::startOffset,j);
    auto &expOp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast1});
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor}, {outCast2});
    auto &tensorOffset1 = ubTensor->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), outCast2->GetDynValidShape()));
    auto &sqrtOp = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinnum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinnum;
        }
    }
    // 旧的两个 + 新的两个
    EXPECT_EQ(gatherinnum, KNumThree);
    EXPECT_EQ(expOp.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(viewOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(viewOp.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(sqrtOp.GetInputOperandSize(), kNumOne);
    EXPECT_NE(sqrtOp.GetInputOperand(kSizeZero), ubTensor);
}

/*
TESTDuplicateViewAlternativeConsumer
inCast{8,16}->gatherin->ubTensor1{1,8,16}
            ->gatherin->ubTensor2{1,8,16}
ubTensor1+ubTensor1->div->outCast1{1,8,16}
ubTensor1+ubTensor2->div->outCast2{1,8,16}
ubTensor2+ubTensor2->div->outCast3{1,8,16}

inCast{8,16}->gatherin->ubTensor1'{1,8,16}
            ->gatherin->ubTensor2'{1,8,16}
            ->gatherin->ubTensor3'{1,8,16}
            ->gatherin->ubTensor4'{1,8,16}
ubTensor1'+ubTensor1'->div->outCast1{1,8,16}
ubTensor2'+ubTensor3'->div->outCast2{1,8,16}
ubTensor4'+ubTensor4'->div->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    int64_t j = 1;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &gatherinOp = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor1});
    gatherinOp.SetAttribute(OpAttributeKey::startOffset, i);
    auto &gatherinOp1 = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor2});
    gatherinOp1.SetAttribute(OpAttributeKey::startOffset, j);
    auto &divOp1 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor1}, {outCast1});
    auto &divOp2 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor2}, {outCast2});
    auto &divOp3 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor2, ubTensor2}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
    }
    EXPECT_EQ(gatherinNum, kNumFour);
    EXPECT_EQ(divOp1.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(divOp2.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(divOp3.GetInputOperandSize(), kNumTwo);

    auto div1Input1 = divOp1.GetInputOperand(kSizeZero);
    auto div1Input2 = divOp1.GetInputOperand(kSizeOne);
    EXPECT_EQ(div1Input2, ubTensor1);
    EXPECT_EQ(div1Input1, div1Input2);

    auto div2Input1 = divOp2.GetInputOperand(kSizeZero);
    auto div2Input2 = divOp2.GetInputOperand(kSizeOne);
    EXPECT_NE(div2Input1, ubTensor1);
    EXPECT_NE(div2Input1, div2Input2);

    auto div3Input1 = divOp3.GetInputOperand(kSizeZero);
    auto div3Input2 = divOp3.GetInputOperand(kSizeOne);
    EXPECT_NE(div3Input2, ubTensor2);
    EXPECT_EQ(div3Input1, div3Input2);
}

/*
TESTDuplicateGatherinConsumerGatherin(ERROR)
inCast{8,16}->Gatherin->ubTensor{1,8,16}->Gatherin->outCast{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTDuplicateGatherinViewnNormal
inCast{8,16}->Gatherin->ubTensor1{8,16}->exp->outCast1{8,16}
                      ->view->ubtensor2 ->exp->outcast2
                                        ->view->outcast3
inCast{8,16}->Gatherin->ubTensor1{8,16}->exp->outCast1{8,16}
            ->Gatherin->ubTensor1'{8,16}->view->ubtensor2 ->exp->outcast2
            ->Gatherin->ubTensor1'{8,16}->view->ubtensor2 ->view->outcast3
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &gatherin = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor1});
    gatherin.SetAttribute(OpAttributeKey::startOffset, i);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {outCast1});
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor1}, {ubTensor2});
    auto &tensorOffset1 = ubTensor1->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor2->GetDynValidShape()));
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {outCast3});
    auto &tensorOffset2 = ubTensor2->GetTensorOffset();
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset2.GetOffset(), tensorOffset2.GetDynOffset(), outCast3->GetDynValidShape()));
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    uint32_t viewNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
    }
    EXPECT_EQ(gatherinNum, KNumThree);
    EXPECT_EQ(viewNum, KNumThree);
}

/*
TESTDuplicateGatherinView(special1)
inCast{8,16}->Gatherin->ubtensor0 ->view->ubtensor1 ->gatherIn->ubtensor2->sqrt->outcast1
                                                             ->exp->outcast2
inCast{8,16}->Gatherin->ubtensor0 ->view->ubtensor1 ->gatherIn->ubtensor2->sqrt->outcast1
            ->Gatherin->ubtensor0'->view->ubtensor1'->gatherIn->ubtensor2'->exp->outcast2
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    int64_t j = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &gatherin = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor0});
    gatherin.SetAttribute(OpAttributeKey::startOffset, i);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor0}, {ubTensor1});
    auto &tensorOffset1 = ubTensor0->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor1->GetDynValidShape()));
    auto &gatherin1 = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {ubTensor1}, {ubTensor2});
    gatherin1.SetAttribute(OpAttributeKey::startOffset, j);
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast2);
    
    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    uint32_t viewNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
    }
    EXPECT_EQ(gatherinNum, kNumFour);
    EXPECT_EQ(viewNum, kNumTwo);
}

/*
TESTDuplicateGatherinView(special2)
inCast{8,16}->View->ubtensor0 ->GatherIn->ubtensor1 ->View->ubtensor2->sqrt->outcast1
                                                                      ->exp->outcast2
inCast{8,16}->View->ubtensor0 ->GatherIn->ubtensor1 ->View->ubtensor2->sqrt->outcast1
            ->View->ubtensor0' ->GatherIn->ubtensor1' ->View->ubtensor2'->exp->outcast2
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor0});
    auto &tensorOffset1 = inCast->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor0->GetDynValidShape()));
    auto &gatherin = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {ubTensor0}, {ubTensor1});
    gatherin.SetAttribute(OpAttributeKey::startOffset, i);
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor1}, {ubTensor2});
    auto &tensorOffset2 = ubTensor1->GetTensorOffset();
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset2.GetDynOffset(), ubTensor2->GetDynValidShape()));

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast1});
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->inCasts_.push_back(inCast);

    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t gatherinNum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinNum;
        }
    }
    EXPECT_EQ(viewNum, kNumFour);
    EXPECT_EQ(gatherinNum, kNumTwo);
}

/*
incast    ->view  -> tensor1 -> sqrt  -> tensor1 -> gatherIn -> tensor3 -> exp -> Outcast2
                             -> sqrt  -> Outcast1                       -> exp -> Outcast3
incast    ->view  -> tensor1 -> sqrt  -> tensor1 -> gatherIn -> tensor3 -> exp -> Outcast2
                             -> sqrt  -> Outcast1   gatherIn -> tensor4 -> exp -> Outcast3
*/
TEST_F(TestDuplicateOpPass, TestCheck1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t i = 0;
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor3}, {outcast2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor3}, {outcast3});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {tensor1}, {tensor2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {tensor1}, {outcast1});
    auto &gatherinOp = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {tensor2}, {tensor3});
    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, 
                                                       MEM_L1, 
                                                       std::vector<SymbolicScalar>(), 
                                                       std::vector<SymbolicScalar>());
    gatherinOp.SetAttribute(OpAttributeKey::startOffset, i);
    viewOp.SetOpAttribute(viewAttr);
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);
    DuplicateOp duplicateoppass;
    EXPECT_EQ(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t gatherinNum = kNumZero;
    auto opList = currFunctionPtr->Operations();
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewNum++;
        }
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            gatherinNum++;
        }
    }
    EXPECT_EQ(gatherinNum, kNumTwo);
    EXPECT_EQ(viewNum, kNumOne);
}

/*
incast    ->gatherIn  -> tensor1 -> sqrt  -> tensor2 -> view -> tensor3 -> exp -> Outcast2
                      -> sqrt  -> Outcast1                               -> exp -> Outcast3
incast    ->gatherIn  -> tensor1 -> sqrt  -> tensor2 -> view -> tensor3 -> exp -> Outcast2
            gatherIn  -> tensor1'-> sqrt  -> Outcast1   view -> tensor4 -> exp -> Outcast3
*/
TEST_F(TestDuplicateOpPass, TestCheck2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t i = 0;
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &gatherinOp = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {incast}, {tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor3}, {outcast2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor3}, {outcast3});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {tensor1}, {tensor2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {tensor1}, {outcast1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, 
                                                       MEM_VECTOR_REG, 
                                                       std::vector<SymbolicScalar>(), 
                                                       std::vector<SymbolicScalar>());
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor2}, {tensor3});
    viewOp.SetOpAttribute(viewAttr);
    gatherinOp.SetAttribute(OpAttributeKey::startOffset, i);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);
    currFunctionPtr->inCasts_.push_back(incast);
    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t gatherinNum = kNumZero;
    uint32_t viewNum = kNumZero;
    auto opList = currFunctionPtr->Operations();
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            gatherinNum++;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewNum++;
        }
    }
    EXPECT_EQ(gatherinNum, kNumTwo);
    EXPECT_EQ(viewNum, kNumTwo);
}

/* ERROR
incast    ->view  -> tensor1 -> exp - >output1 
*/
TEST_F(TestDuplicateOpPass, TestCheck3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    currFunctionPtr->inCasts_.push_back(incast);
    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
}

/* ERROR
incast    ->view  -> tensor1 -> exp - >output1 
                             -> exp - >output2
*/
TEST_F(TestDuplicateOpPass, TestCheck4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostCheck", "TestPostCheck", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {tensor1});
    auto &tensorOffset = incast->GetTensorOffset();
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), tensor1->GetDynValidShape()));
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    DuplicateOp duplicateoppass;
    EXPECT_NE(duplicateoppass.PostCheck(*currFunctionPtr), SUCCESS);
}
}
}