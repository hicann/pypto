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
 * \file test_n_buffer_merge.cpp
 * \brief Unit test for n_buffer_merge pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk{

class NBufferMergeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

        config::SetPassOption(VEC_NBUFFER_MODE, 2);
        config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{{-1, 2}});
        Platform::Instance().ObtainPlatformInfo();
    }

    void TearDown() override {}

    Status TestNBufferMergeWithDifferentVecBufferMap(std::map<int64_t, int64_t> vecNBufferSetting);
};

TEST_F(NBufferMergeTest, TestNBufferMerge) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->subGraphID = subGraphID0;
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast2->subGraphID = subGraphID1;
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->subGraphID = subGraphID0;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->subGraphID = subGraphID0;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->subGraphID = subGraphID1;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->subGraphID = subGraphID1;

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    auto &copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);
    auto &copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    NBufferMerge nPass;
    Pass &nbufferPass = nPass;
    nbufferPass.PreCheck(*currFunctionPtr);
    nbufferPass.Run(*currFunctionPtr, "", "");
    nbufferPass.PostCheck(*currFunctionPtr);
}

TEST_F(NBufferMergeTest, TestMulityInputOutputMode3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    const int vecParallelNum = 2;
    const int manualMode = 3;
    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &mul_op1 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor1});
    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    mul_op1.UpdateSubgraphID(0);
    exp_op1.UpdateSubgraphID(0);
    auto &mul_op2 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor2});
    auto &exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor2}, {outcast2});
    mul_op2.UpdateSubgraphID(1);
    exp_op2.UpdateSubgraphID(1);
    auto &mul_op3 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor3});
    auto &exp_op3 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor3}, {outcast3});
    int subGraphId2 = 2;
    mul_op3.UpdateSubgraphID(subGraphId2);
    exp_op3.UpdateSubgraphID(subGraphId2);
    auto &mul_op4 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor4});
    auto &exp_op4 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor4}, {outcast4});
    int subGraphId3 = 3;
    mul_op4.UpdateSubgraphID(subGraphId3);
    exp_op4.UpdateSubgraphID(subGraphId3);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);
    currFunctionPtr->outCasts_.push_back(outcast4);

    // Call the pass
    NBufferMerge NBM;
    currFunctionPtr->paramConfigs_.mgVecParallelLb = vecParallelNum;
    currFunctionPtr->paramConfigs_.vecNBuffermode = manualMode;
    currFunctionPtr->DumpJsonFile("./config/pass/json/nBufferMerge_mulity_input_before.json");
    size_t subGraphCount = 4;
    currFunctionPtr->SetTotalSubGraphCount(subGraphCount);
    EXPECT_EQ(NBM.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetTotalSubGraphCount(), 2);
    currFunctionPtr->DumpJsonFile("./config/pass/json/nBufferMerge_mulity_input_after.json");
}

TEST_F(NBufferMergeTest, TestMode4) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int result = 11;
    const int manualMode = 4;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.vecNBuffermode = manualMode;
    function->paramConfigs_.vecNBufferSetting = {{-1, 4}, {1, 2}};
    function->paramConfigs_.mgVecParallelLb = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(NBufferMergeTest, TestInvalidMode) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int noneMode = 999;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.vecNBuffermode = noneMode;
    function->paramConfigs_.mgVecParallelLb = vecParallelNum;
    function->SetTotalSubGraphCount(1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), FAILED);
}
}
}