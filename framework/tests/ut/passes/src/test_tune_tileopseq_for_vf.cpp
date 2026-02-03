/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file test_tune_tileop_for_vf.cpp
* \brief Unit test for TuneTileOpSeqForVF.
*/
#include <gtest/gtest.h>
#include "tilefwk/platform.h"
#include "passes/block_graph_pass/tune_tileopseq_for_vf.h"
#define private public

namespace npu {
namespace tile_fwk {
constexpr int TT_NUM10 = 10;
constexpr int TT_NUM20 = 20;
constexpr int TT_NUM30 = 30;
constexpr int TT_NUM40 = 40;
constexpr int TT_NUM50 = 50;
constexpr int TT_NUM60 = 60;
constexpr int TT_NUM70 = 70;
constexpr int TT_NUM80 = 80;
constexpr int TT_NUM90 = 90;
constexpr int TT_NUM100 = 100;
constexpr int TT_NUM16 = 16;
constexpr int TT_NUM5 = 5;

class TuneTileopseqForVFTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

void BuildGraphForTest(std::shared_ptr<Function> currFunctionPtr, std::vector<Operation *> &opListPtr) {
    std::vector<int64_t> shape = {TT_NUM16, TT_NUM16};
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = TT_NUM10;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->memoryrange.start = TT_NUM10;
    tensor2->memoryrange.end = TT_NUM20;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->memoryrange.start = TT_NUM20;
    tensor3->memoryrange.end = TT_NUM30;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->memoryrange.start = TT_NUM30;
    tensor4->memoryrange.end = TT_NUM40;
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor5->memoryrange.start = TT_NUM40;
    tensor5->memoryrange.end = TT_NUM50;
    auto tensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor6->memoryrange.start = TT_NUM50;
    tensor6->memoryrange.end = TT_NUM60;
    auto tensor7 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor7->memoryrange.start = TT_NUM60;
    tensor7->memoryrange.end = TT_NUM70;
    auto tensor8 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor8->memoryrange.start = TT_NUM70;
    tensor8->memoryrange.end = TT_NUM80;
    auto tensor9 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor9->memoryrange.start = TT_NUM80;
    tensor9->memoryrange.end = TT_NUM90;
    auto tensor10 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor10->memoryrange.start = TT_NUM90;
    tensor10->memoryrange.end = TT_NUM100;
    auto &vecop1 = currFunctionPtr->AddRawOperation(Opcode::OP_EXP, {tensor1}, {tensor2});
    opListPtr.emplace_back(&vecop1);
    auto &vecop2 = currFunctionPtr->AddRawOperation(Opcode::OP_SQRT, {tensor3}, {tensor4});
    opListPtr.emplace_back(&vecop2);
    auto &vecop3 = currFunctionPtr->AddRawOperation(Opcode::OP_RECIPROCAL, {tensor5}, {tensor6});
    opListPtr.emplace_back(&vecop3);
    auto &op1 = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_MOVEIN, {tensor7}, {tensor8});
    opListPtr.emplace_back(&op1);
    auto &op2 = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {tensor6}, {tensor9});
    opListPtr.emplace_back(&op2);
    auto &vecop4 = currFunctionPtr->AddRawOperation(Opcode::OP_EXPAND, {tensor8}, {tensor10});
    opListPtr.emplace_back(&vecop4);
}

TEST_F(TuneTileopseqForVFTest, TestMergeForTuneTileop) {
    // Build Graph
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDep", "TestFindDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDepLeaf", "TestFindDepLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation *> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto &op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_TRANSPOSE_MOVEIN);
    EXPECT_EQ(tuneTileop.opList_[TT_NUM5]->GetOpcode(), Opcode::OP_TRANSPOSE_MOVEOUT);
}

TEST_F(TuneTileopseqForVFTest, TestNotMergeForTuneTileop) {
    // Build Graph
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestTuneTileop", "TestTuneTileop", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestTuneTileopLeaf", "TestTuneTileopLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation *> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    opListPtr[3]->GetIOperands()[0]->memoryrange.start = TT_NUM50;
    opListPtr[3]->GetIOperands()[0]->memoryrange.end = TT_NUM60;
    opListPtr[3]->GetOOperands()[0]->memoryrange.start = TT_NUM60;
    opListPtr[3]->GetOOperands()[0]->memoryrange.end = TT_NUM70;
    opListPtr[4]->GetIOperands()[0]->memoryrange.start = TT_NUM60;
    opListPtr[4]->GetIOperands()[0]->memoryrange.end = TT_NUM70;
    opListPtr[4]->GetOOperands()[0]->memoryrange.start = TT_NUM70;
    opListPtr[4]->GetOOperands()[0]->memoryrange.end = TT_NUM80; 
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto &op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(tuneTileop.opList_[TT_NUM5]->GetOpcode(), Opcode::OP_EXPAND);
}

TEST_F(TuneTileopseqForVFTest, TestMainProcess) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainProcess", "TestMainProcess", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainProcessLeaf", "TestMainProcessLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MULACC_B, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_SRC, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_DST, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_COPY_UB, {input}, {output});
    TuneTileOpSeqForVF tuneSync;
    tuneSync.RunOnFunction(*rootFuncPtr.get());
    auto it = rootFuncPtr->rootFunc_->programs_.begin();
    auto funcPtr = it->second;
    std::vector<Operation *> opList(funcPtr->Operations(false).DuplicatedOpList());
    EXPECT_EQ(opList.size(), TT_NUM5);
}

} // namespace tile_fwk
} // namespace npu

#undef private