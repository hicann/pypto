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
 * \file test_infermemoryconflict.cpp
 * \brief Unit test for InferMemoryConflict pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#define private public
#include "passes/tensor_graph_pass/infer_memory_conflict.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

const int NUM_ZERO = 0;
const int NUM_ONE = 1;
const int NUM_2 = 2;
const int NUM_3 = 3;
const int NUM_4 = 4;
const int NUM_6 = 6;
const int NUM_8 = 8;
const int NUM_10 = 10;
const int NUM_11 = 11;
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_127 = 127;
const int NUM_128 = 128;
const int NUM_129 = 129;

class InferMemoryConflictTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "InferMemoryTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(InferMemoryConflictTest, TestInit) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_2, NUM_4};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset1, shape1);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset2, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->inCasts_.push_back(input1);
    currFunctionPtr->inCasts_.push_back(input2);
    currFunctionPtr->outCasts_.push_back(output);

    auto &assembleOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp1.SetOpAttribute(assembleAttr1);

    auto &assembleOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp2.SetOpAttribute(assembleAttr2);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.memoryInfo.size(), NUM_3);
    EXPECT_NE(pass.memoryInfo.find(input1), pass.memoryInfo.end());
    EXPECT_NE(pass.memoryInfo.find(input2), pass.memoryInfo.end());
    EXPECT_NE(pass.memoryInfo.find(output), pass.memoryInfo.end());
    EXPECT_EQ(pass.memoryInfo[input1], input1);
    EXPECT_EQ(pass.memoryInfo[input2], input2);
    EXPECT_EQ(pass.memoryInfo[output], output);
}

/*
Case 1:
input->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> shape = {NUM_2, NUM_4};


    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);

    currFunctionPtr->inCasts_.push_back(input);

    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape);

    currFunctionPtr->outCasts_.push_back(output);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T1}, {T2});
    
    auto &assembleOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp1.SetOpAttribute(assembleAttr1);
    
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset);
    viewOp.SetOpAttribute(viewAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&assembleOp1), pass.preregcopys.end());
}

/*
Case 3:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_2};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape2);
    
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output");

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);
    
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);
    auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T1}, {T2});

    InferMemoryConflict pass;
    auto passStatus = pass.Init(*currFunctionPtr);
    passStatus = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(passStatus, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&reshapeOp), pass.preregcopys.end());
}

/*
Case 4:
input->view->T->assemble->output(same memoryid)
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_2, NUM_2};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape);
    auto T = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 0;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset);
    viewOp1.SetOpAttribute(viewAttr1);
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
}

/*
Case 5:
T2->
input->index_outcast->T1->assemble->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};
    std::vector<int64_t> shape1 = {NUM_ONE, NUM_2};

    auto tensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);

    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);

    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape0);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape0);
    
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape0);
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    currFunctionPtr->outCasts_.push_back(output);
    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {tensor0, tensor2, input}, {tensor1});
   
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor1}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
}

/*
Case 6:
T2->
input->index_outcast->T1->reshape->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestForwardPropagation5) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape1 = {NUM_ONE, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};

    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    auto logicalTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto logicalTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape0);
    auto logicalTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape0);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape0);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {logicalTensor0, logicalTensor2, input}, {logicalTensor1});
   
    auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {logicalTensor1}, {output});
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&reshapeOp), pass.preregcopys.end());
}

/*
Case 1:
input1->view->T1->exp->T2->assemble->output
                         ->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_4};
    std::vector<int64_t> shape2 = {NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset1, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor3 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor3, offset2, shape2);
    
    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor3->SetSymbol("output2");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor3->memoryId = 2;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset1);
    viewOp.SetOpAttribute(viewAttr);

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {T1}, {T2});
    
    auto &assembleOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp1.SetOpAttribute(assembleAttr1);

    auto &assembleOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output2});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp2.SetOpAttribute(assembleAttr2);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&assembleOp2), pass.postregcopys.end());
}

/*
Case 2:
input1->view->T1->exp->T2->reshape->T3->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_2, NUM_2, NUM_2};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape2);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {T1}, {T2});
    
    auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T2}, {T3});
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T3}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);

    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.postregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.postregcopys.find(&reshapeOp), pass.postregcopys.end());
}

/*
Case 3:
input->view->T1->exp->T2->assemble->output1
                        ->assemble->output2(same memoryId)
                        ->assemble->output3(same symbol)
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_6};
    std::vector<int64_t> shape2 = {NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};
    std::vector<int64_t> offset3 = {NUM_ZERO, NUM_4};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset1, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor3 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor3, offset2, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor4 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output3 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor4, offset3, shape2);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor3->SetSymbol("output2");
    ddrRawTensor4->SetSymbol("output1");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor3->memoryId = 1;
    ddrRawTensor4->memoryId = 2;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);
    currFunctionPtr->outCasts_.push_back(output3);

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {T1}, {T2});
    
    auto &assembleOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp2.SetOpAttribute(assembleAttr2);

    auto &assembleOp3 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output2});
    auto assembleAttr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp3.SetOpAttribute(assembleAttr3);

    auto &assembleOp4 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output3});
    auto assembleAttr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset3);
    assembleOp4.SetOpAttribute(assembleAttr4);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);
}

/*
Case 4:
T2->
input->index_outcast->T1->assemble->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape1 = {1, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};

    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    auto T0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape0);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape0);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape0);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);

    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {T0, T2, input}, {T1});
   
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T1}, {output1});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr1);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ZERO);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);
}

/*
Case 5:
T2->
input->index_outcast->T1->reshape->output
T0->
*/
TEST_F(InferMemoryConflictTest, TestBackwardPropagation5) {
    auto currFunctionPtr1 = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr1 != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_4, NUM_4};
    std::vector<int64_t> shape1 = {1, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};
    std::vector<int64_t> shape3 = {NUM_ONE, NUM_4, NUM_4};
    std::vector<int64_t> shape4 = {NUM_2, NUM_ONE, NUM_4};

    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    auto T0 = std::make_shared<LogicalTensor>(*currFunctionPtr1, DT_FP32, shape2);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr1, DT_FP32, shape1);
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr1, ddrRawTensor1, offset1, shape2);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr1, DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr1, ddrRawTensor2, offset2, shape4);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr1->inCasts_.push_back(input);
    currFunctionPtr1->outCasts_.push_back(output);

    currFunctionPtr1->AddOperation(Opcode::OP_INDEX_OUTCAST, {T0, T2, input}, {T1});
   
    auto &reshapeOp1 = currFunctionPtr1->AddOperation(Opcode::OP_RESHAPE, {T1}, {output});
    
    InferMemoryConflict testPass;
    auto status = testPass.Init(*currFunctionPtr1);
    status = testPass.BackwardPropagation(*currFunctionPtr1);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(testPass.postregcopys.size(), NUM_ONE);
    EXPECT_NE(testPass.postregcopys.find(&reshapeOp1), testPass.postregcopys.end());
}

/*
Case 1:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBothPropagation1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_2, NUM_4};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape);

    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape);
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto &viewOperation = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset);
    viewOperation.SetOpAttribute(viewAttr);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T1}, {T2});
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ZERO);
    EXPECT_NE(pass.preregcopys.find(&assembleOp), pass.preregcopys.end());
}

/*
Case 2:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestBothPropagation2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_2, NUM_4};
    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_ONE, NUM_2};
    std::vector<int64_t> shape3 = {NUM_2, NUM_2, NUM_2};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape2);
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->SetSymbol("input1");

    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);

    auto &viewOp2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr2 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp2.SetOpAttribute(viewAttr2);

    auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T1}, {T2});
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);
    
    InferMemoryConflict pass;
    auto status = pass.Init(*currFunctionPtr);
    status = pass.ForwardPropagation(*currFunctionPtr);
    status = pass.BackwardPropagation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.preregcopys.size(), NUM_ONE);
    EXPECT_EQ(pass.postregcopys.size(), NUM_ONE);
    EXPECT_NE(pass.preregcopys.find(&reshapeOp), pass.preregcopys.end());
    EXPECT_NE(pass.postregcopys.find(&reshapeOp), pass.postregcopys.end());
}

/*
Case 2:
input1->view->T1->reshape->T2->assemble->output
*/
TEST_F(InferMemoryConflictTest, TestInsertCopys) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_32};
    std::vector<int64_t> shape2 = {NUM_ONE, NUM_2, NUM_32};
    std::vector<int64_t> offset3 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset3, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    
    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output");
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape2);
    ddrRawTensor1->memoryId = 0;
    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);
    ddrRawTensor2->memoryId = 1;

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset3);
    viewOp1.SetOpAttribute(viewAttr1);

    auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T1}, {T2});
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp.SetOpAttribute(assembleAttr);
    
    InferMemoryConflict pass;
    pass.preregcopys.insert(&reshapeOp);
    pass.postregcopys.insert(&reshapeOp);
    auto status = pass.InsertCopys(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy1 = nullptr;
    Operation* copy2 = nullptr;
    for (auto &op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            if (*(op->GetOOperands().begin()) == T2) {
                copy2 = op;
                cnt += 1;
            } else {
                copy1 = op;
                cnt += 10;
            }
        }
    }
    EXPECT_EQ(cnt, NUM_11);
    EXPECT_NE(copy1, nullptr);
    EXPECT_EQ(*(copy1->GetIOperands().begin()), T1);
    auto newTensorOut1 = *(copy1->GetOOperands().begin());
    EXPECT_EQ(*(newTensorOut1->GetConsumers().begin()), &reshapeOp);
    EXPECT_NE(copy2, nullptr);
    auto newTensorIn2 = *(copy2->GetIOperands().begin());
    EXPECT_EQ(*(newTensorIn2->GetProducers().begin()), &reshapeOp);
}

/*
STest1
input1->view->T1->reshape->T2->assemble->output
单链，存在地址冲突
*/
TEST_F(InferMemoryConflictTest, STest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {NUM_129, NUM_127};
    std::vector<int64_t> offset = {NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset, shape);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset, shape);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset);
    viewOp.SetOpAttribute(viewAttr);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T1}, {T2});
    
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);
    
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy = nullptr;
    for (auto &op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(*(copy->GetIOperands().begin()), T2);
    auto newTensorOut1 = *(copy->GetOOperands().begin());
    EXPECT_EQ(copy->GetTileShape().GetVecTile().size(), NUM_2);
    std::vector<int64_t> expectShape = {NUM_128, NUM_128};
    EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, expectShape);
    EXPECT_EQ(*(newTensorOut1->GetConsumers().begin()), &assembleOp);
}

/*
STest2
input1->view->T1->index_outcast->T2->reshape->T3->exp->output
单链，存在reshape
*/
TEST_F(InferMemoryConflictTest, STest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape0 = {NUM_32, NUM_32, NUM_128};
    std::vector<int64_t> shape1 = {NUM_32, NUM_32, NUM_64};
    std::vector<int64_t> shape2 = {NUM_ONE, NUM_32, NUM_32, NUM_64};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_ZERO, NUM_ZERO, NUM_ZERO};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape0);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape2);
    
    ddrRawTensor1->SetSymbol("input1");
    ddrRawTensor2->SetSymbol("output");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {T4, T5, T1}, {T2});

    auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {T2}, {T3});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {T3}, {output});
    
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy = nullptr;
    for (auto &op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(*(copy->GetIOperands().begin()), T2);
    EXPECT_EQ(copy->GetTileShape().GetVecTile().size(), NUM_3);
    std::vector<int64_t> expectShape = {NUM_8, NUM_32, NUM_64};
    EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, expectShape);
    auto newTensorOut = *(copy->GetOOperands().begin());
    EXPECT_EQ(*(newTensorOut->GetConsumers().begin()), &reshapeOp);
}

/*
STest3
input1->view->T1->exp->T2->assemble->output
                         ->assemble->output
同一tensor assemble输出到不同outcast
*/
TEST_F(InferMemoryConflictTest, STest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape1 = {NUM_2, NUM_2};
    std::vector<int64_t> shape2 = {NUM_2, NUM_4};
    std::vector<int64_t> offset1 = {NUM_ZERO, NUM_ZERO};
    std::vector<int64_t> offset2 = {NUM_ZERO, NUM_2};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto T1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto T2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset1, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor3 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor3, offset2, shape2);
    
    ddrRawTensor1->SetSymbol("input");
    ddrRawTensor2->SetSymbol("output1");
    ddrRawTensor3->SetSymbol("output2");
    ddrRawTensor1->memoryId = 0;
    ddrRawTensor2->memoryId = 1;
    ddrRawTensor3->memoryId = 2;

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {input}, {T1});
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    viewOp1.SetOpAttribute(viewAttr1);

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {T1}, {T2});

    auto &assembleOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output1});
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assembleOp1.SetOpAttribute(assembleAttr1);
    
    auto &assembleOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {T2}, {output2});
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assembleOp2.SetOpAttribute(assembleAttr2);
    
    InferMemoryConflict pass;
    auto status = pass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    int cnt = 0;
    Operation* copy = nullptr;
    for (auto &op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
            cnt += 1;
        }
    }
    EXPECT_EQ(cnt, NUM_ONE);
    EXPECT_NE(copy, nullptr);
    EXPECT_EQ(*(copy->GetIOperands().begin()), T2);
    EXPECT_EQ(copy->GetTileShape().GetVecTile().size(), NUM_2);
    std::vector<int64_t> expectShape = {NUM_2, NUM_32};
    EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, expectShape);
    auto newTensorOut = *(copy->GetOOperands().begin());
    EXPECT_EQ(*(newTensorOut->GetConsumers().begin()), &assembleOp2);
}

/*
STest4
view->reshape->matmul
优化场景不插入 registery copy
*/
TEST_F(InferMemoryConflictTest, STest4) {
    PassManager &passManager = PassManager::Instance();
    Tensor in0(DT_FP32, Shape{3, 128, 64}, "in0");
    Tensor in1(DT_FP32, Shape{3, 64, 256}, "in1");
    Tensor out(DT_FP32, Shape{128, 256}, "out");
    TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
    FUNCTION("InferMemoryConflictTest") {
        auto a = View(in0, Shape{1, 128, 64}, {0, 0, 0});
        auto b = View(in1, Shape{1, 64, 256}, {0, 0, 0});
        auto a0 = Reshape(a, Shape{128, 64});
        auto b0 = Reshape(b, Shape{64, 256});
        out = Matrix::Matmul(DataType::DT_FP32, a0, b0, false, false);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_InferMemoryConflictTest");
    int cnt = 0;
    for (auto e : func->Operations().DuplicatedOpList()) {
        if (e->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    passManager.RegisterStrategy("InferMemoryConflictTestStrategy", {
        {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "InferMemoryConflictTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    for (auto e : func->Operations().DuplicatedOpList()) {
        if (e->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
}

} // namespace tile_fwk
} // namespace npu
