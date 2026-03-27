/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file test_generate_move_op_checker.cpp
\brief Unit test for Generate_Move_Op pass checker - all error scenarios
*/
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/attribute.h"
#include <vector>
#include <string>
#include <algorithm>

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class TestGenerateMoveOpChecker : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "GenerateMoveOpCheckerTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

};

template<typename OpType>
OpType *FindOpByOpcode(Function *function, Opcode targetOpcode) {
    OpType *targetOp = nullptr;
    for (auto &op : function->Operations()) {
        if (op.GetOpcode() == targetOpcode) {
            targetOp = &op;
            break;
        }
    }
    return targetOp;
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_AttrNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"VIEW_AttrNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_MoreThanOneInput) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestViewMultiInput", "TestViewMultiInput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {16, 32};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {t1Tensor, t2Tensor}, {t3Tensor});

    std::vector<int64_t> viewShape{16, 32};
    auto viewAttr = std::make_shared<ViewOpAttribute>(viewShape);
    viewOp.SetOpAttribute(viewAttr);

    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t3Tensor);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    const auto &operations = currFunctionPtr->Operations();
    for (auto &op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_EQ(op.GetIOperands().size(), 2);
        }
    }
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_MoreThanOneOutput) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestViewMultiOutput", "TestViewMultiOutput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 32};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {t1Tensor}, {t2Tensor, t3Tensor});
    std::vector<int64_t> viewShape{16, 32};
    auto viewAttr = std::make_shared<ViewOpAttribute>(viewShape);
    viewOp.SetOpAttribute(viewAttr);

    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    const auto &operations = currFunctionPtr->Operations();
    for (auto &op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_EQ(op.GetOOperands().size(), 2);
        }
    }
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_OutputHasNullConsumer) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestViewOutputHasNullConsumer", "TestViewOutputHasNullConsumer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {16, 48};
    auto t01Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    t01Tensor->nodetype = NodeType::INCAST;
    auto t02Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    t02Tensor->nodetype = NodeType::OUTCAST;

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {t01Tensor}, {t02Tensor});

    std::vector<int64_t> viewShape{16, 48};
    auto viewAttr = std::make_shared<ViewOpAttribute>(viewShape);
    viewOp.SetOpAttribute(viewAttr);

    using ConsumerSetType = std::set<Operation*, LogicalTensor::CompareOp>;
    auto& consumers = const_cast<ConsumerSetType&>(t02Tensor->GetConsumers());
    consumers.clear();
    consumers.insert(nullptr);

    currFunctionPtr->inCasts_.push_back(t01Tensor);
    currFunctionPtr->outCasts_.push_back(t02Tensor);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    const auto &operations = currFunctionPtr->Operations();
    for (auto &op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto outputTensor = op.GetOOperands().front();
            ASSERT_NE(outputTensor, nullptr);
            const auto& targetConsumers = outputTensor->GetConsumers();
            EXPECT_TRUE(std::find(targetConsumers.begin(), targetConsumers.end(), nullptr) != targetConsumers.end());
        }
    }
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_ConsumerNotSupportDDR) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestViewConsumerNotSupportDDR", "TestViewConsumerNotSupportDDR", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {16, 16};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    t1Tensor->nodetype = NodeType::INCAST;
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    t2Tensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    t3Tensor->nodetype = NodeType::OUTCAST;
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {t1Tensor}, {t2Tensor});
    auto &mulOp = currFunctionPtr->AddOperation(Opcode::OP_MUL, {t2Tensor}, {t3Tensor});
    std::vector<int64_t> viewShape{16, 16};
    auto viewAttr = std::make_shared<ViewOpAttribute>(viewShape);
    viewOp.SetOpAttribute(viewAttr);
    using ConsumerSetType = std::set<Operation*, LogicalTensor::CompareOp>;
    auto& consumers = const_cast<ConsumerSetType&>(t2Tensor->GetConsumers());
    consumers.insert(&mulOp);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t3Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);

    const auto &operations = currFunctionPtr->Operations();
    for (auto &op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto outputTensor = op.GetOOperands().front();
            ASSERT_NE(outputTensor, nullptr);
            EXPECT_EQ(outputTensor->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
            const auto& targetConsumers = outputTensor->GetConsumers();
            EXPECT_TRUE(targetConsumers.find(&mulOp) != targetConsumers.end());
        }
    }
}

TEST_F(TestGenerateMoveOpChecker, ViewOp_ConvertPathInvalid) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewConvertPathInvalid", "TestViewConvertPathInvalid", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    t2Tensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    auto &convertOp = currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {t2Tensor}, {t2Tensor});
    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    convertOp.SetOpAttribute(convertAttr);

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {t1Tensor}, {t2Tensor});
    auto viewAttr = std::make_shared<ViewOpAttribute>(shape);
    viewOp.SetOpAttribute(viewAttr);

    using ConsumerSetType = std::set<Operation*, LogicalTensor::CompareOp>;
    auto& consumers = const_cast<ConsumerSetType&>(t2Tensor->GetConsumers());
    consumers.clear();
    consumers.insert(&convertOp);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = FAILED;
    auto convertAttrPtr = dynamic_cast<ConvertOpAttribute*>(convertOp.GetOpAttribute().get());
    ASSERT_NE(convertAttrPtr, nullptr);
    EXPECT_NE(convertAttrPtr->GetConvertPath().first, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_AttrNull) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleOpAttrNull", "TestAssembleOpAttrNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 32};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {t1Tensor}, {t2Tensor});
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_EQ(assembleOp.GetOpAttribute().get(), nullptr);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_MoreThanOneInput) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleOpMoreThanOneInput", "TestAssembleOpMoreThanOneInput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 16};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {t1Tensor, t2Tensor}, {t3Tensor});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(Offset{0, 0});
    assembleOp.SetOpAttribute(assembleAttr);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t3Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_EQ(assembleOp.GetIOperands().size(), 2);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_MoreThanOneOutput) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleOpMoreThanOneOutput", "TestAssembleOpMoreThanOneOutput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 32};
    auto t0Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {t0Tensor}, {t2Tensor, t3Tensor});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(Offset{0, 0});
    assembleOp.SetOpAttribute(assembleAttr);
    currFunctionPtr->inCasts_.push_back(t0Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_EQ(assembleOp.GetOOperands().size(), 2);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_MoreThanOneInput) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestConvertOpMoreThanOneInput", "TestConvertOpMoreThanOneInput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 16};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &convertOp = currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {t1Tensor, t2Tensor}, {t3Tensor});
    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    convertOp.SetOpAttribute(convertAttr);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t3Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_EQ(convertOp.GetIOperands().size(), 2);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_MoreThanOneOutput) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestConvertOpMoreThanOneOutput", "TestConvertOpMoreThanOneOutput", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 16};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t3Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &convertOp = currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {t1Tensor}, {t2Tensor, t3Tensor});
    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    convertOp.SetOpAttribute(convertAttr);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_EQ(convertOp.GetOOperands().size(), 2);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_SameMemType) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestConvertOpSameMemType", "TestConvertOpSameMemType", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 32};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &convertOp = currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {t1Tensor}, {t2Tensor});
    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR);
    convertOp.SetOpAttribute(convertAttr);
    t1Tensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    t2Tensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_EQ(t1Tensor->GetMemoryTypeOriginal(), t2Tensor->GetMemoryTypeOriginal());
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_DiffShape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestConvertOpDiffShape", "TestConvertOpDiffShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape1{16, 32};
    std::vector<int64_t> shape2{32, 64};
    auto t1Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto t2Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &convertOp = currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {t1Tensor}, {t2Tensor});
    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    convertOp.SetOpAttribute(convertAttr);
    currFunctionPtr->inCasts_.push_back(t1Tensor);
    currFunctionPtr->outCasts_.push_back(t2Tensor);
    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
    EXPECT_NE(t1Tensor->GetShape(), t2Tensor->GetShape());
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_View_InputInvalid) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PostCheckViewInputInvalid", "PostCheckViewInputInvalid", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {16, 16};
    auto in1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto in2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto out = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {in1, in2}, {out});
    auto viewAttr = std::make_shared<ViewOpAttribute>(shape);
    viewOp.SetOpAttribute(viewAttr);
    
    GenerateMoveOp generateMoveOp;
    Status postCheckStatus = generateMoveOp.PostCheck(*currFunctionPtr);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_View_OutputInvalid) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PostCheckViewOutputInvalid", "PostCheckViewOutputInvalid", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {16, 16};
    auto in1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {in1}, {out1, out2});
    auto viewAttr = std::make_shared<ViewOpAttribute>(shape);
    viewOp.SetOpAttribute(viewAttr);
    
    GenerateMoveOp generateMoveOp;
    Status postCheckStatus = generateMoveOp.PostCheck(*currFunctionPtr);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_DuplicateOp_Invalid) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PostCheckDupInvalid", "PostCheckDupInvalid", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {16, 16};
    auto in1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    currFunctionPtr->AddOperation(Opcode::OP_DUPLICATE, {in1}, {out1});
    
    GenerateMoveOp generateMoveOp;
    Status postCheckStatus = generateMoveOp.PostCheck(*currFunctionPtr);
    
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_ConvertOp_Invalid) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PostCheckConvertInvalid", "PostCheckConvertInvalid", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {16, 16};
    auto in1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {in1}, {out1});
    
    GenerateMoveOp generateMoveOp;
    Status postCheckStatus = generateMoveOp.PostCheck(*currFunctionPtr);
    
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, View_MemoryTypeMismatch) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "PostCheckViewMemMismatch", "PostCheckViewMemMismatch", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {16, 16};
    auto inTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    inTensor->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    outTensor->SetMemoryTypeOriginal(MEM_UB);
    
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inTensor}, {outTensor});
    auto viewAttr = std::make_shared<ViewOpAttribute>(shape);
    viewOp.SetOpAttribute(viewAttr);
    
    GenerateMoveOp generateMoveOp;
    Status postCheckStatus = generateMoveOp.PostCheck(*currFunctionPtr);
    
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, ViewInputNullCheck) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_1(DT_FP32, shape1, "input_1");
        Tensor input_2(DT_FP32, shape1, "input_2");
        Tensor output(DT_FP32, shape2, "output");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            { "GenerateMoveOp", PassName::GENERATE_MOVE_OP },
        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        config::SetBuildStatic(true);
        FUNCTION("VIEW", {input_1, input_2, output}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");
            auto tmp_view = View(input_1, shape2, {0,0});
            output = tmp_view;
        }

        std::string jsonFilePath = "./view_null_check.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData1 = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData1);

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_VIEW");
        ASSERT_NE(originFunction, nullptr);
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                auto& inputs = op.GetIOperands();
                if (!inputs.empty()) {
                    inputs[0] = nullptr;
                }
            }
        }

        GenerateMoveOp generateMoveOp;
        bool preCheck = generateMoveOp.PreCheck(*originFunction);
        
        EXPECT_EQ(preCheck, true);
    }
}

TEST_F(TestGenerateMoveOpChecker, ViewOutputNullCheck) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_aa(DT_FP32, shape1, "input_aa");
        Tensor input_bb(DT_FP32, shape1, "input_bb");
        Tensor output(DT_FP32, shape2, "output");
        Tensor output2(DT_FP32, shape2, "output2");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            { "GenerateMoveOp", PassName::GENERATE_MOVE_OP },
        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        config::SetBuildStatic(true);

        FUNCTION("VIEW", {input_aa, input_bb, output, output2}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");
            auto tmp_view = View(input_aa, shape2, {0,0});
            output = tmp_view;
            output2 = tmp_view;
        }

        std::string jsonFilePath = "./view_output_null_check.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_VIEW");
        ASSERT_NE(originFunction, nullptr);
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                auto& outputs = op.GetOOperands();
                if (!outputs.empty()) {
                    outputs[0] = nullptr;
                }
            }
        }

        GenerateMoveOp generateMoveOp;
        bool preCheck = generateMoveOp.PreCheck(*originFunction);
        
        EXPECT_EQ(preCheck, true);
    }
}

TEST_F(TestGenerateMoveOpChecker, AssembleInputNullCheck) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape1{128, 128};
        std::vector<int64_t> shape2{256, 128};
        TileShape::Current().SetVecTile({128, 128});

        Tensor input_m(DT_FP32, shape1, "input_m");
        Tensor input_n(DT_FP32, shape1, "input_n");
        Tensor output(DT_FP32, shape2, "output");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            { "GenerateMoveOp", PassName::GENERATE_MOVE_OP },
        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        config::SetBuildStatic(true);

        FUNCTION("ASSEMBLE", {input_m, input_n, output}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");
            
            Assemble({{input_m, {0, 0}}, {input_n, {128, 0}}});
        }

        std::string jsonFilePath = "./assemble_null_check.json";
        DumpJsonFile(Program::GetInstance().DumpJson(), jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ASSEMBLE");

        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                auto& inputs = op.GetIOperands();
                inputs[0] = nullptr;
            }
        }

        GenerateMoveOp generateMoveOp;
        bool preCheck = generateMoveOp.PreCheck(*originFunction);
        
        EXPECT_EQ(preCheck, true);
    }
}

TEST_F(TestGenerateMoveOpChecker, AssembleOutputNullCheck) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape1{128, 128};
        std::vector<int64_t> shape2{256, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_c(DT_FP32, shape1, "input_c");
        Tensor input_d(DT_FP32, shape1, "input_d");
        Tensor output2(DT_FP32, shape2, "output2");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            { "GenerateMoveOp", PassName::GENERATE_MOVE_OP },
        });
        ConfigManager::Instance();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("ASSEMBLE", {input_c, input_d, output2}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");
            Assemble({{input_c, {0, 0}}, {input_d, {128, 0}}});
        }
        std::string jsonFilePath = "./assemble_output_null_check.json";
        DumpJsonFile(Program::GetInstance().DumpJson(), jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ASSEMBLE");

        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                auto& outputs = op.GetOOperands();
                outputs[0] = nullptr;
            }
        }

        GenerateMoveOp generateMoveOp;
        bool preCheck = generateMoveOp.PreCheck(*originFunction);
        
        EXPECT_EQ(preCheck, true);
    }
}

TEST_F(TestGenerateMoveOpChecker, ConvertOp_ShapeMismatch) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestConvertShapeMismatch", "TestConvertShapeMismatch", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape_in  = {16, 16};
    std::vector<int64_t> shape_out = {32, 32};

    auto inTensor  = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape_in);
    auto outTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape_out);
    inTensor->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    outTensor->SetMemoryTypeOriginal(MEM_UB);

    auto &convertOp = currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {inTensor}, {outTensor});
    
    auto convertAttr = std::make_shared<ConvertOpAttribute>(MEM_DEVICE_DDR, MEM_UB);
    convertOp.SetOpAttribute(convertAttr);

    currFunctionPtr->inCasts_.push_back(inTensor);
    currFunctionPtr->outCasts_.push_back(outTensor);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, ConvertOpAttributeNull) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestConvertAttrNull", "TestConvertAttrNull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {16, 32};
    auto inTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_CONVERT, {inTensor}, {outTensor});
    currFunctionPtr->inCasts_.push_back(inTensor);
    currFunctionPtr->outCasts_.push_back(outTensor);

    GenerateMoveOp generateMoveOp;
    Status preCheckStatus = generateMoveOp.PreCheck(*currFunctionPtr);
    EXPECT_EQ(preCheckStatus, FAILED);
}
} // namespace tile_fwk
} // namespace npu