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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#define private public
#include "passes/tile_graph_pass/graph_optimization/remove_redundant_op.h"

namespace npu {
namespace tile_fwk{
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeSeven = 7UL;
static const size_t kSizeEight = 8UL;
static const size_t kSizeTen = 10UL;
static const size_t kSizeEleven = 11UL;
static const size_t kSizeThirteen = 13UL;
static const size_t kSizeForteen = 14UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;
static const uint16_t kNumExpEight = 256u;

class TestRemoveRedundantOpPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        TileShape::Current().SetVecTile({64, 64});
    }
    void TearDown() override {}
};

/*
TESTRemoveDummyExpand
inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                    ->sqrt->outCast2{8,16}
                                    ->reciprocal->outCast3{8,16}
inCast{8,16}->exp->outCast1
            ->sqrt->outCast2
            ->reciprocal->outCast3
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_EXPAND, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_RECIPROCAL, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t expand_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            ++expand_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(expand_num, kNumZero);
}

/*
TESTRemoveDummyRegCopy
inCast{8,16}->regcopy->ubTensor1{16,8}->regcopy->ubTensor2{16,8}->exp->outCast1{16,8}
inCast{8,16}->regcopy->ubTensor1{16,8}->exp->outCast1{16,8}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &regcopy = currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t regcopy_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            EXPECT_EQ(op.GetOpMagic(), regcopy.GetOpMagic());
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
            ++regcopy_num;
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), ubTensor1);
        }
    }
    EXPECT_EQ(regcopy_num, kNumOne);
}

/*
TESTRemoveDummyAssembleDDRSpecialCase(WARNING CASE)
inCast{8,16}->exp(any legal op)->ddrTensor1{8,16}  ->exp->outCast3{8,16}
                                    ->assemble->outCast1{8,16}
                                    ->assemble->outCast2{8,16}

inCast{8,16}->exp->outCast1/outCast2{8,16}->exp->outCast3{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape,
        TileOpFormat::TILEOP_ND, "outCast1", NodeType::OUTCAST);
    outCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape,
        TileOpFormat::TILEOP_ND, "outCast2", NodeType::OUTCAST);
    outCast2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &exp1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast2});
    auto &exp2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(exp1.GetOutputOperandSize(), kSizeOne);
    EXPECT_EQ(exp2.GetInputOperandSize(), kSizeOne);
}

/*
TESTRemoveDummyView(WARNING CASE)
inCast{8,16}->exp->ddrTensor1{8,16}->exp->ubTensor2{8,16}->view->ubTensor3{8,16}->exp->outCast2{8,16}
                                  ->view->outCast1{8,16}                       ->reciprocal->outCast3{8,16}
                                                                               ->sqrt->outCast4{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor1}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {ubTensor3});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor3}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_RECIPROCAL, {ubTensor3}, {outCast3});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor3}, {outCast4});
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast4);
    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTRemoveAssemble1
inCast{8,16}->view->ddrTensor{8,16}->assemble->outCast{1,8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest6) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ddrTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ddrTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ddrTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTRemoveDummyRegCopy
inCast{8,16}/{a0,16}->regcopy->ubTensor1{8,16}/{a1,16}->regcopy->ubTensor2{16,8}/{a1,16}->exp->outCast1{16,8}
inCast{8,16}/{a0,16}->regcopy->ubTensor1{8,16}/{a1,16}->exp->outCast1{16,8}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest7) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    outCast->SetMemoryTypeBoth(MemoryType::MEM_UB);

    auto &regcopy = currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t regcopy_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            EXPECT_EQ(op.GetOpMagic(), regcopy.GetOpMagic());
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
            ++regcopy_num;
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), ubTensor1);
        }
    }
    EXPECT_EQ(regcopy_num, kNumOne);
}

/*
TESTRemoveAssembleDDR2
inCast{8,16}->view->ubTensor1{8,16}->assemble->outCast1{8,16}
all delete
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest10) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "outCast", NodeType::OUTCAST);
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &view = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    auto &assemble = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(view_num, kNumZero);
}

/*
TESTRemoveAssembleDDR3
inCast1{8,16}->view->ubTensor1{16,16}->assemble->outCast1{16,16}
inCast2{8,16}->view->
all delete
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest11) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumEight, kNumZero};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2, TileOpFormat::TILEOP_ND, "outCast", NodeType::OUTCAST);
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &view1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {ubTensor});
    view1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    auto &view2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast2}, {ubTensor});
    view2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2));
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(view_num, kNumTwo);
}

/*
TESTPostExpand(DynValidShape not same)
inCast{8,16}->sqrt->ubTensor1{8,16}->expand->ubTensor2{8,16}->exp->outCast1{8,16}
inCast{8,16}->sqrt->ubTensor1{8,16}->expand->ubTensor2{8,16}->exp->outCast1{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest12) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    std::vector<SymbolicScalar> dynValidShape1;
    std::vector<SymbolicScalar> dynValidShape2;
    dynValidShape1.push_back(SymbolicScalar("Tensor1"));
    dynValidShape2.push_back(SymbolicScalar("Tensor2"));
    ubTensor1->UpdateDynValidShape(dynValidShape1);
    ubTensor2->UpdateDynValidShape(dynValidShape2);
    currFunctionPtr->AddOperation(Opcode::OP_EXPAND, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t expand_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            ++expand_num;
        }
    }
    EXPECT_EQ(expand_num, kNumOne);
}

/*
view->exp(end assemble)->view(end assemble)->expand(end assemble)->exp(end assemble)
                                                                 ->exp(end assemble)

exp(end assemble*3) ->exp(end assemble)
                    ->exp(end assemble)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest1) {
    //Define the shape of the Tensors
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};

    PassManager &passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape, "view");
    Tensor expand(DT_FP32, shape, "expand");
    Tensor output1(DT_FP32, shape, "output1");
    Tensor output2(DT_FP32, shape, "output2");

    FUNCTION("STCase1") {
        exp = Exp(input);
        view = View(exp, shape, {kNumZero, kNumZero});
        expand = Expand(view, shape);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeEleven);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        { "AssignMemoryType",  PassName::ASSIGN_MEMORY_TYPE},
        {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int view_num = kNumZero;
    int expand_num = kNumZero;
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
            expand_num++;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(expand_num, kNumZero);
}

/*
view->exp(end assemble)->view(end assemble)->expand(end assemble)->exp(end assemble)
                                                                 ->exp(end assemble)

exp(end assemble)->view(end assemble)->expand(end assemble) ->exp(end assemble)
                                                            ->exp(end assemble)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest2) {
    //Define the shape of the Tensors
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape2 = {kNumExpFour, 1};
    std::vector<int64_t> shape3 = {kNumExpFour, kNumExpEight};

    PassManager &passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape2, "view");
    Tensor expand(DT_FP32, shape3, "expand");
    Tensor output1(DT_FP32, shape3, "output1");
    Tensor output2(DT_FP32, shape3, "output2");

    FUNCTION("STCase2") {
        exp = Exp(input);
        view = View(exp, shape2, {kNumZero, kNumZero});
        expand = Expand(view, shape3);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");
    
    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        { "AssignMemoryType",  PassName::ASSIGN_MEMORY_TYPE},
        {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int view_num = kNumZero;
    int expand_num = kNumZero;
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
            expand_num++;
        }
    }
    EXPECT_EQ(view_num, kNumTwo);
    EXPECT_EQ(expand_num, kNumOne);
}

/*
view{64,64} ->exp{64,64} ->assemble{64, 64}
view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64} ->assemble{64, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest3) {
    //Define the shape of the Tensors
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpSix};

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
        {  "ExpandFunction",    PassName::EXPAND_FUNCTION},
        {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
    });

    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("STCase3") {
        TileShape::Current().SetVecTile(tile_shape);
        output = Exp(input);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase3");
    int assemble_before = kNumZero;
    for (const auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_before++;
        }
    }
    EXPECT_EQ(assemble_before, kNumThree);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy"), SUCCESS);
    
    // ================== Verify the effect of the Pass ==================
    int assemble_after = kNumZero;
    for (const auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_after++;
        }
    }
    EXPECT_EQ(assemble_after, kNumTwo);
    EXPECT_NE(assemble_after, assemble_before);
}
void RemoveRedundantL1DataMoveGraph (std::shared_ptr<Function> &currFunctionPtr) {
    std::shared_ptr<LogicalTensor> input_cast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> input_cast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64,16});
    std::shared_ptr<LogicalTensor> input_cast1_view = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> input_cast2_view = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64,16});
    input_cast1_view ->SetMemoryTypeBoth(MEM_L1);
    input_cast2_view ->SetMemoryTypeBoth(MEM_L1);
    std::shared_ptr<LogicalTensor> op_view_L1_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> op_view_L1_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64,16});
    op_view_L1_out1 ->SetMemoryTypeBoth(MEM_L1);
    op_view_L1_out2 ->SetMemoryTypeBoth(MEM_L1);
    std::shared_ptr<LogicalTensor> view_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> view_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> view_out3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> view_out4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> l0a_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> l0a_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> l0b_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> l0b_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> a_mul_b_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> a_mul_b_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    auto &head_view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_cast1}, {input_cast1_view});
    std::vector<int> newoffset{0,0};
    auto viewAttribute =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    head_view_op1.SetOpAttribute(viewAttribute);

    auto &head_view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_cast2}, {input_cast2_view});
    head_view_op2.SetOpAttribute(viewAttribute);

    auto &view_L1_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_cast1_view}, {op_view_L1_out1});
    view_L1_op1.SetOpAttribute(viewAttribute);
    auto &view_L1_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_cast2_view}, {op_view_L1_out2});
    view_L1_op2.SetOpAttribute(viewAttribute);

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out1}, {view_out1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out1}, {view_out2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 32}));
    auto &view_op3 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out2}, {view_out3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &view_op4 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out2}, {view_out4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 0}));

    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0A, {view_out1}, {l0a_out1});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0A, {view_out2}, {l0a_out2});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0B, {view_out3}, {l0b_out1});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0B, {view_out4}, {l0b_out2});

    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {l0a_out1,l0b_out1}, {a_mul_b_out1});
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {l0a_out2, l0b_out2}, {a_mul_b_out2});

    currFunctionPtr->inCasts_.push_back(input_cast1);
    currFunctionPtr->inCasts_.push_back(input_cast2);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out1);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out2);
}
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpL1DataMove) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "RemoveRedundantOpL1DataMove", "RemoveRedundantOpL1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("RemoveRedundantOpL1DataMove", currFunctionPtr);

    RemoveRedundantL1DataMoveGraph(currFunctionPtr);

    //验证构图
    int view_count = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        }
    }
    EXPECT_EQ(view_count,8);

    std::stringstream ssBefore;
    ssBefore << "Before_RemoveRedundantOp";

    // Call the pass
    RemoveRedundantOp removeRedundantOp;
    removeRedundantOp.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/removeRedundant_L1DataMove_before.json");
    removeRedundantOp.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/removeRedundant_L1DataMove_after.json");
    removeRedundantOp.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_RemoveRedundantOp";

    // Validate the results
    int view_count_after_pass = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count_after_pass++;
        }
    }
    EXPECT_EQ(view_count_after_pass,6);
}

/*
RemoveReshapeChain
inCast{8,16}->reshape->ubTensor1{16,8}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
inCast{8,16}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest13) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &reshape1 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor1});
    auto &reshape2 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    const auto &operations = currFunctionPtr->Operations();
    uint32_t reshape_num = kNumZero;
    for (auto &op : operations) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(reshape2.GetOpMagic(), op.GetOpMagic());
            EXPECT_EQ(reshape2.GetInputOperand(kSizeZero), inCast);
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), ubTensor2);
        }
    }
    EXPECT_EQ(operations.Contains(reshape1), false);
    EXPECT_EQ(reshape_num, kNumOne);
}

/*
RemoveSameReshape
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast{8,16}
inCast{8,16}->sqrt->outCast{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest14) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
RemoveReshapeChainSeveralConsumer(WARNING CASE)
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast1{8,16}
                                    ->exp->outCast2{8,16}
                                    ->reshape->outCast3{16,8}
inCast{8,16}->sqrt->outCast1{8,16}
            ->exp->outCast2{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest15) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast1});
    auto &exp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);

    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(exp.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(exp.GetInputOperand(kSizeZero), inCast);
        }
    }
}

/*
RemoveReshapeChainSeveralConsumer
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
                                      ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
            ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest16) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &reshape1 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {outCast1});
    auto &reshape2 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast2});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantOp removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        }
    }
    EXPECT_EQ(reshape1.GetInputOperand(kSizeZero), inCast);
    EXPECT_EQ(reshape2.GetInputOperand(kSizeZero), inCast);
    EXPECT_EQ(reshape_num, kNumTwo);
}

/*
TESTRemoveIterative
inCast{8,16}->view->ubTensor1{8,16}->reshape->ubTensor2{8,16}->assemble->outCast1{8,16}
all delete
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest17) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, TileOpFormat::TILEOP_ND, "outCast", NodeType::OUTCAST);
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &view = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor1});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &assemble = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    uint32_t view_num = kNumZero;
    uint32_t reshape_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(view_num, kNumZero);
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
TestRemoveAssembleSpecialCase
inCast{8,16}->exp->ddrTensor1{8,16} ->assemble-> outCast{8,16}
            ->exp->ddrTensor1{8,16} ->assemble->

inCast{8,16}->exp->ddrTensor1{8,16} ->assemble-> outCast{8,16}
            ->exp->ddrTensor1{8,16} ->assemble->
*/
TEST_F(TestRemoveRedundantOpPass, TestRemoveMoreAssembleSpecialCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape,
        TileOpFormat::TILEOP_ND, "outCast", NodeType::OUTCAST);
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor1}, {ubTensor3});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor2}, {ubTensor3});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor3}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assembleNum = kNumZero;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumTwo);
}

/*
TestRemoveAssembleDynSpecialCase
inCast{8,16}->exp->Tensor1{8,16} ->Reshape->Tensor2{8,16} ->assemble-> outCast{8,16}
            
inCast{8,16}->exp->Tensor1{8,16} ->Reshape->Tensor2{16,8} ->assemble-> outCast{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, TestRemoveMoreAssembleDynSpecialCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumEight};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1,
        TileOpFormat::TILEOP_ND, "outCast", NodeType::OUTCAST);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    ubTensor1->UpdateDynValidShape({SymbolicScalar("Reshape_0_Dim_0"), SymbolicScalar("Reshape_0_Dim_1")});
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    ubTensor2->UpdateDynValidShape({SymbolicScalar("Reshape_0_Dim_0"), SymbolicScalar("Reshape_0_Dim_1")});

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    RemoveRedundantOp RemoveRedundantOpPass;

    EXPECT_EQ(RemoveRedundantOpPass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(currFunctionPtr->GetOutcast()[0]->GetDynValidShape()[0].Dump(), SymbolicScalar("Reshape_0_Dim_0").Dump());
    EXPECT_EQ(currFunctionPtr->GetOutcast()[0]->GetDynValidShape()[1].Dump(), SymbolicScalar("Reshape_0_Dim_1").Dump());
    EXPECT_EQ(viewNum, kNumZero);
    EXPECT_EQ(assembleNum, kNumZero);
}

/*
TestGenerateViewSpecialCase
inCast1{8,16}->view->Tensor1{4,16}->assemble->outCast{16,16}
             ->view->Tensor2{4,16}->assemble->
inCast2{8,16}->mul->Tenosr3{8,16}->assemble->
inCast3{8,16}  

inCast1{8,16}->view->Tensor1{4,16}->assemble->outCast{16,16}
             ->view->Tensor2{4,16}->assemble->
inCast2{8,16}->mul->Tenosr3{8,16}->assemble->
inCast3{8,16}    
*/
TEST_F(TestRemoveRedundantOpPass, TestGenerateViewSpecialCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumFour, kNumZero};
    std::vector<int64_t> offset3 = {kNumEight, kNumZero};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto inCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1,
        TileOpFormat::TILEOP_ND, "outCast", NodeType::OUTCAST);
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor3->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {ubTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    auto &viewOp2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {ubTensor2});
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2));
    currFunctionPtr->AddOperation(Opcode::OP_MUL, {inCast2, inCast3}, {ubTensor3});
    auto &assembleOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor1}, {outCast});
    assembleOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1));
    auto &assembleOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast});
    assembleOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset2));
    auto &assembleOp3 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor3}, {outCast});
    assembleOp3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset3));

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->inCasts_.push_back(inCast3);
    currFunctionPtr->outCasts_.push_back(outCast);
    
    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(viewNum, kNumTwo);
    EXPECT_EQ(assembleNum, kNumThree);
}
}
}
