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
* \file test_tune_sync_for_vf.cpp
* \brief Unit test for TuneSyncForVF.
*/
#include <gtest/gtest.h>
#include "tilefwk/platform.h"
#include "passes/block_graph_pass/tune_sync_for_vf.h"
#define private public

namespace npu {
namespace tile_fwk {
constexpr int TS_NUM3 = 3;
constexpr int TS_NUM4 = 4;
constexpr int TS_NUM5 = 5;
constexpr int TS_NUM10 = 10;
constexpr int TS_NUM20 = 20;
constexpr int TS_NUM30 = 30;
constexpr int TS_NUM40 = 40;
constexpr int TS_NUM50 = 50;
constexpr int TS_NUM60 = 60;
constexpr int TS_NUM16 = 16;
constexpr int TS_NUM15 = 15;
constexpr int TS_NUM28 = 28;
constexpr int TS_NUM42 = 42;

class TuneSyncForVFTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPassGlobalConfig(KEY_ENABLE_VF, true);
    }
    void TearDown() override {}
};

void BuildGraphForTest(std::shared_ptr<Function> &currFunctionPtr, std::vector<Operation *> &opListPtr) {
    std::vector<int64_t> shape = {TS_NUM16, TS_NUM16};
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = TS_NUM10;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->memoryrange.start = TS_NUM10;
    tensor2->memoryrange.end = TS_NUM20;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->memoryrange.start = TS_NUM20;
    tensor3->memoryrange.end = TS_NUM30;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->memoryrange.start = TS_NUM30;
    tensor4->memoryrange.end = TS_NUM40;
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor5->memoryrange.start = TS_NUM40;
    tensor5->memoryrange.end = TS_NUM50;
    auto tensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor6->memoryrange.start = TS_NUM50;
    tensor6->memoryrange.end = TS_NUM60;

    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    auto &op1 = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_MOVEIN, {tensor1}, {tensor2});
    op1.cycleStart = 0;
    op1.cycleEnd = op1.cycleStart + op1.GetLatency();
    opListPtr.emplace_back(&op1);
    auto &setflag1 = currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_SRC, {input}, {output});
    setflag1.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, 0};
    opListPtr.emplace_back(&setflag1);
    auto &vecop1 = currFunctionPtr->AddRawOperation(Opcode::OP_SQRT, {tensor3}, {tensor4});
    vecop1.cycleStart = TS_NUM15;
    vecop1.cycleEnd = vecop1.cycleStart + vecop1.GetLatency();
    opListPtr.emplace_back(&vecop1);
    auto &setflag2 = currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_SRC, {input}, {output});
    setflag2.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE3, CoreType::AIV, CoreType::AIV, 0};
    opListPtr.emplace_back(&setflag2);
    auto &waitflag1 = currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_DST, {input}, {output});
    waitflag1.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, 0};
    opListPtr.emplace_back(&waitflag1);
    auto &vecop2 = currFunctionPtr->AddRawOperation(Opcode::OP_RECIPROCAL, {tensor2}, {tensor5});
    vecop2.cycleStart = TS_NUM28;
    vecop2.cycleEnd = vecop2.cycleStart + vecop2.GetLatency();
    opListPtr.emplace_back(&vecop2);
    auto &waitflag2 = currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_DST, {input}, {output});
    waitflag2.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE3, CoreType::AIV, CoreType::AIV, 0};
    opListPtr.emplace_back(&waitflag2);
    auto &op2 = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {tensor4}, {tensor6});
    op2.cycleStart = TS_NUM42;
    op2.cycleEnd = op2.cycleStart + op2.GetLatency();
    opListPtr.emplace_back(&op2);
    currFunctionPtr->setOpMap.emplace(&setflag1, &vecop2);
    currFunctionPtr->setOpMap.emplace(&setflag2, &op2);
    currFunctionPtr->waitOpMap.emplace(&waitflag1, &op1);
    currFunctionPtr->waitOpMap.emplace(&waitflag2, &vecop1);
    std::vector<Operation *> oriOpList{&op1, &vecop1, &vecop2, &op2};
    currFunctionPtr->oriOpList = oriOpList;
    currFunctionPtr->pipeEndTime[PipeType::PIPE_MTE2] = op1.cycleEnd;
    currFunctionPtr->pipeEndTime[PipeType::PIPE_V] = vecop2.cycleEnd;
    currFunctionPtr->pipeEndTime[PipeType::PIPE_MTE3] = op2.cycleEnd;
}

TEST_F(TuneSyncForVFTest, TestTuneSyncForVF) {
    // Build Graph
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestTuneSync", "TestTuneSync", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestTuneSyncLeaf", "TestTuneSyncLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation *> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    TuneSyncForVF tuneSync;
    tuneSync.opList_ = opListPtr;
    for (auto &op : tuneSync.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneSync.GenPipeOpMap(currFunctionPtr.get());
    tuneSync.ChangeOpSeq(currFunctionPtr.get(), false);
    EXPECT_EQ(tuneSync.opList_[TS_NUM3]->GetOpcode(), Opcode::OP_SQRT);
    EXPECT_EQ(tuneSync.opList_[TS_NUM4]->GetOpcode(), Opcode::OP_RECIPROCAL);
}

TEST_F(TuneSyncForVFTest, TestMainProcess) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainSchedule", "TestMainSchedule", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainScheduleLeaf", "TestMainScheduleLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MULACC_B, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_SRC, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_SYNC_DST, {input}, {output});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_COPY_UB, {input}, {output});
    TuneSyncForVF tuneSync;
    tuneSync.RunOnFunction(*rootFuncPtr.get());
    auto it = rootFuncPtr->rootFunc_->programs_.begin();
    auto funcPtr = it->second;
    std::vector<Operation *> opList(funcPtr->Operations(false).DuplicatedOpList());
    EXPECT_EQ(opList.size(), TS_NUM5);
}

TEST_F(TuneSyncForVFTest, TestSkip) {
    config::SetPassGlobalConfig(KEY_ENABLE_VF, false);
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestMainSchedule", "TestMainSchedule", nullptr);
    TuneSyncForVF tuneSync;
    EXPECT_EQ(tuneSync.RunOnFunction(*rootFuncPtr.get()), SUCCESS);
}

} // namespace tile_fwk
} // namespace npu

#undef private