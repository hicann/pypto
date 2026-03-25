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
#include "passes/pass_check/pre_graph_checker.h"

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

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast1}, {tensor1});
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
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

TEST_F(PassCheckTest, TestCheckLocalTensor) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCheckLocalTensor", "TestCheckLocalTensor", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto localTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    currFunctionPtr->AddOperation(Opcode::OP_ADD, {incast1, localTensor}, {outcast1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckLocalTensor(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestPreGraphCheckerAssembleViewReshapeInvalidIO) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PreGraphCheckerTest", "PreGraphCheckerTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1, incast2}, {outcast1});
    currFunctionPtr->Operations().back().UpdateSubgraphID(0);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);

    PreGraphProcessChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestPreGraphCheckerTensorNotInSubgraph) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PreGraphCheckerTest2", "PreGraphCheckerTest2", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_ADD, {incast1}, {tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_ADD, {tensor1}, {outcast1});
    currFunctionPtr->Operations()[0].UpdateSubgraphID(0);
    currFunctionPtr->Operations()[1].UpdateSubgraphID(1);
    incast1->subGraphID = 0;
    outcast1->subGraphID = 1;
    tensor1->subGraphID = NOT_IN_SUBGRAPH;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->SetTotalSubGraphCount(2);

    PreGraphProcessChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(checker.DoPostCheck(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckConsumerProducer_ProducerIsNull) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckConsumerProducer_ProducerIsNull", "TestCheckConsumerProducer_ProducerIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    tensor->GetProducers().insert(nullptr);

    Checker checker;
    EXPECT_EQ(checker.CheckConsumerProducer(tensor), FAILED);
}

TEST_F(PassCheckTest, TestCheckConsumerProducer_ConsumerIsNull) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckConsumerProducer_ConsumerIsNull", "TestCheckConsumerProducer_ConsumerIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    tensor->GetConsumers().insert(nullptr);

    Checker checker;
    EXPECT_EQ(checker.CheckConsumerProducer(tensor), FAILED);
}

TEST_F(PassCheckTest, TestCheckOpIOValid_OutputIsNull) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckOpIOValid_OutputIsNull", "TestCheckOpIOValid_OutputIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    
    auto& addOp = currFunctionPtr->AddOperation(Opcode::OP_ADD, {incast1}, {outcast1});
    addOp.oOperand[0] = nullptr; 
    (void)addOp;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckOpIOValid(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompleteness_IncastIsNull) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckCompleteness_IncastIsNull", "TestCheckCompleteness_IncastIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    
    currFunctionPtr->inCasts_.push_back(nullptr);
    currFunctionPtr->outCasts_.push_back(outcast1);

    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompleteness_OutcastEmpty) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckCompleteness_OutcastEmpty", "TestCheckCompleteness_OutcastEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    
    currFunctionPtr->inCasts_.push_back(incast1);

    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckCompleteness_OutcastIsNull) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckCompleteness_OutcastIsNull", "TestCheckCompleteness_OutcastIsNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(nullptr);

    Checker checker;
    EXPECT_EQ(checker.CheckCompleteness(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckGraphLoop_HasLoop) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckGraphLoop_HasLoop", "TestCheckGraphLoop_HasLoop", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto& op1 = currFunctionPtr->AddOperation(Opcode::OP_ADD, {incast1, tensor2}, {tensor1});
    auto& op2 = currFunctionPtr->AddOperation(Opcode::OP_ADD, {tensor1}, {tensor2});
    (void)op1;
    (void)op2;

    currFunctionPtr->inCasts_.push_back(incast1);
    Checker checker;
    EXPECT_EQ(checker.CheckGraphLoop(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckDynAttrForView_FromDynOffsetEmpty) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckDynAttrForView_FromDynOffsetEmpty", "TestCheckDynAttrForView_FromDynOffsetEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    auto& viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {outcast1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(Offset(0, 0));
    viewAttr->GetFromDynOffset().clear(); 
    viewOp.SetOpAttribute(viewAttr);
    (void)viewAttr;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckDynAttrForView_ToDynValidShapeEmpty) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckDynAttrForView_ToDynValidShapeEmpty", "TestCheckDynAttrForView_ToDynValidShapeEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    
    auto& viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {outcast1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(Offset(0, 0));
    viewAttr->GetToDynValidShape().clear(); 
    viewOp.SetOpAttribute(viewAttr);
    (void)viewAttr;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckDynAttrForView(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckToDynOffsetForAssemble_ToDynOffsetEmpty) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckToDynOffsetForAssemble_ToDynOffsetEmpty", "TestCheckToDynOffsetForAssemble_ToDynOffsetEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    
    auto& assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast1}, {outcast1});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(Offset(0, 0));
    assembleAttr->GetToDynOffset().clear(); 
    assembleOp.SetOpAttribute(assembleAttr);
    (void)assembleAttr;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckToDynOffsetForAssemble(*currFunctionPtr), FAILED);
}

TEST_F(PassCheckTest, TestCheckLocalTensor_LocalInput) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCheckLocalTensor_LocalInput", "TestCheckLocalTensor_LocalInput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};

    auto incast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::INCAST);
    auto localTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    currFunctionPtr->AddOperation(Opcode::OP_ADD, {incast1, localTensor}, {outcast1});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    Checker checker;
    EXPECT_EQ(checker.CheckLocalTensor(*currFunctionPtr), FAILED);
}

// Some computation graphs may not include incast because they contain operations such as VEC_DUP.
TEST_F(PassCheckTest, TestPublicCheck_IncastEmpty) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPublicCheck_IncastEmpty", "TestPublicCheck_IncastEmpty", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {32, 32};
    auto outcast1 = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "", NodeType::OUTCAST);
    
    currFunctionPtr->outCasts_.push_back(outcast1);

    Checker checker;
    EXPECT_EQ(checker.PublicCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(PassCheckTest, TestDefaultCheckItems) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDefaultCheckItems", "TestDefaultCheckItems", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Checker checker;
    EXPECT_EQ(checker.DoDefaultEnabledPreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(checker.DoDefaultEnabledPostCheck(*currFunctionPtr), SUCCESS);
}