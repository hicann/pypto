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
namespace tile_fwk {

class NBufferMergeTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    Status TestNBufferMergeWithDifferentVecBufferSetting(std::map<int64_t, int64_t> vecNBufferSetting);
};

TEST_F(NBufferMergeTest, TestNBufferMerge)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto& copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    auto& copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    auto& copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);
    auto& copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    NBufferMerge nPass;
    Pass& nbufferPass = nPass;
    nbufferPass.PreCheck(*currFunctionPtr);
    nbufferPass.Run(*currFunctionPtr, "", "");
    nbufferPass.PostCheck(*currFunctionPtr);
}

TEST_F(NBufferMergeTest, TestMulityInputOutputMode3)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    const int vecParallelNum = 2;
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
    auto& mul_op1 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor1});
    auto& exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    mul_op1.UpdateSubgraphID(0);
    exp_op1.UpdateSubgraphID(0);
    auto& mul_op2 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor2});
    auto& exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor2}, {outcast2});
    mul_op2.UpdateSubgraphID(1);
    exp_op2.UpdateSubgraphID(1);
    auto& mul_op3 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor3});
    auto& exp_op3 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor3}, {outcast3});
    int subGraphId2 = 2;
    mul_op3.UpdateSubgraphID(subGraphId2);
    exp_op3.UpdateSubgraphID(subGraphId2);
    auto& mul_op4 = currFunctionPtr->AddOperation(Opcode::OP_MUL, {incast1, incast2}, {tensor4});
    auto& exp_op4 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor4}, {outcast4});
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
    currFunctionPtr->paramConfigs_.vecNBufferSetting = {{-2, 0}};
    currFunctionPtr->DumpJsonFile("./config/pass/json/nBufferMerge_mulity_input_before.json");
    size_t subGraphCount = 4;
    currFunctionPtr->SetTotalSubGraphCount(subGraphCount);
    EXPECT_EQ(NBM.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetTotalSubGraphCount(), 2);
    currFunctionPtr->DumpJsonFile("./config/pass/json/nBufferMerge_mulity_input_after.json");
}

TEST_F(NBufferMergeTest, TestMode4)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int result = 11;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(
            G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}),
            true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{
            {"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{
            {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.vecNBufferSetting = {{-2, 1}, {-1, 4}, {1, 2}};
    function->paramConfigs_.mgVecParallelLb = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(NBufferMergeTest, TestNoMergeMode)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.vecNBufferSetting = {{-1, 1}};
    function->paramConfigs_.mgVecParallelLb = vecParallelNum;
    function->SetTotalSubGraphCount(1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
}

TEST_F(NBufferMergeTest, TestInvalidMode)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-2, 4}, {1, 2}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingKeyMoreThanMaxValue_Tolerated)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 4}, {100, 2}}), SUCCESS);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingValueLessThanMinValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 4}, {1, 0}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingValueMoreThanMaxValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 4}, {1, INT64_MAX}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndNoDefaultValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{0, 2}}), SUCCESS);
}

Status NBufferMergeTest::TestNBufferMergeWithDifferentVecBufferSetting(std::map<int64_t, int64_t> vecNBufferSetting)
{
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 3;
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 10;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(
            G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}),
            true);
        std::vector<std::vector<std::string>> iOperands{
            {"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{
            {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.vecNBufferSetting = vecNBufferSetting;
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    return NBM.RunOnFunction(*function);
}

Function* BuildFunctionWithSubgraphs(
    ComputationalGraphBuilder& G, const std::vector<int64_t>& tileShape, int subGraphNum)
{
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    for (int i = 1; i <= subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(
            G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1_" + strID, "tensor2_" + strID, "tensor3_" + strID}),
            true);
        std::vector<std::vector<std::string>> iOperands{
            {"incast1"}, {"tensor1_" + strID}, {"tensor2_" + strID}, {"tensor3_" + strID}};
        std::vector<std::vector<std::string>> oOperands{
            {"tensor1_" + strID}, {"tensor2_" + strID}, {"tensor3_" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    return G.GetFunction();
}

TEST_F(NBufferMergeTest, TestSemanticLabelSetting)
{
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 3;
    const int subGraphNum = 4;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, tileShape, subGraphNum);

    // Set semantic label for subgraph 1 and 2
    auto vecLabel = std::make_shared<SemanticLabel>("VecLabel", __FILE__, __LINE__);
    for (int i = 1; i <= 2; i++) {
        std::string strID = std::to_string(i);
        G.GetOp("ABS_" + strID)->SetSemanticLabel(vecLabel);
        G.GetOp("EXP_" + strID)->SetSemanticLabel(vecLabel);
    }

    // Default merge=4, but "VecLabel" override to 1 (no merge for labeled subgraphs)
    function->paramConfigs_.vecNBufferSetting = {{-1, 4}};
    function->paramConfigs_.vecNBufferSettingByLabel = {{"VecLabel", 1}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
}

// ===== ByFunc Integration Tests =====
TEST_F(NBufferMergeTest, ByFuncMergeDefaultTwo)
{
    // Build 8 AIV subgraphs (same structure, same hash) + 1 COPY_IN subgraph = 9 total
    // ByFunc DEFAULT=2 → all AIV merge in groups of 2 → ceil(8/2)=4 AIV groups + 1 COPY_IN = 5
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 48;
    const int subGraphNum = 8;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    for (int i = 1; i <= subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"t1_" + strID, "t2_" + strID, "t3_" + strID}), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_ABS}, {{"incast1"}}, {{"t1_" + strID}}, {"ABS_" + strID}, true), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_EXP}, {{"t1_" + strID}}, {{"t2_" + strID}}, {"EXP_" + strID}, true), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_ADDS}, {{"t2_" + strID}}, {{"t3_" + strID}}, {"ADDS_" + strID}, true), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_ASSEMBLE}, {{"t3_" + strID}}, {{"outcast"}}, {"ASM_" + strID}, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASM_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();

    function->paramConfigs_.vecNBufferSettingByFunc = {{"DEFAULT", 2}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 5);
}

TEST_F(NBufferMergeTest, ByFuncMergeFuncSpecificOverride)
{
    // BuildFunctionWithSubgraphs: 1 COPY_IN (AIC, hashOrder=0) + 8 AIV subgraphs (hashOrder=1)
    // DEFAULT=4 → AIV ceil(8/4)=2 groups, override func{magic}_1=2 → ceil(8/2)=4 groups
    // Expected with override: 4 AIV groups + 1 COPY_IN = 5
    const int mgVecParallelLb = 48;
    const int subGraphNum = 8;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, {16, 16}, subGraphNum);

    int fm = function->GetFuncMagic();
    function->paramConfigs_.vecNBufferSettingByFunc = {
        {"DEFAULT", 4},
        {"func" + std::to_string(fm) + "_1", 2}
    };
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 5);
}

TEST_F(NBufferMergeTest, ByFuncMergeDefaultOneNoMerge)
{
    // ByFunc DEFAULT:1 with size==1 → noMerge mode, graph untouched.
    const int mgVecParallelLb = 48;
    const int subGraphNum = 8;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, {16, 16}, subGraphNum);

    function->paramConfigs_.vecNBufferSettingByFunc = {{"DEFAULT", 1}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum + 1);
}

} // namespace tile_fwk
} // namespace npu
