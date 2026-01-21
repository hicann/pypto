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
 * \file test_common_operation_eliminate.cpp
 * \brief Unit test for common_operation_eliminate pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/common_operation_eliminate.h"
#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {

class CommonOperationEliminateTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
    }
    void TearDown() override {}
};

TEST_F(CommonOperationEliminateTest, EliminateRedundantOps) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2","t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiInputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiOutputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2", "t3"}, {"t4", "t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"RowMax1", "RowMax2", "MUL1", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6", "t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantCascadeOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "EXP1", "EXP2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
    std::shared_ptr<LogicalTensor> tensorPtr = G.GetTensor("t1");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetConsumers().size(), 1);
    tensorPtr = G.GetTensor("t6");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetProducers().size(), 1);
}

TEST_F(CommonOperationEliminateTest, IgnoreSingleInputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t3"}, {"t2", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t3"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreMultiInputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t4"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentAttr) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ADDS1", "ADDS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ADDS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 1.0));
    opPtr = G.GetOp("ADDS2");
    EXPECT_NE(opPtr, nullptr);
    const double value2 = 2.0;
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, value2));
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS", "EXP", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentSubgraph) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ABS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(0);
    opPtr = G.GetOp("ABS2");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(1);
    opPtr = G.GetOp("MUL");
    EXPECT_NE(opPtr, nullptr);
    const int subgraphID2 = 2;
    opPtr->UpdateSubgraphID(subgraphID2);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    const int subgraphNum = 3;
    function->SetTotalSubGraphCount(subgraphNum);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreSpecialOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_MUL,
                                Opcode::OP_L1_TO_FIX, Opcode::OP_L1_TO_FIX, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}, {"t1"}, {"t1"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"VIEW1", "VIEW2", "MUL1", "COPY1", "COPY2", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4", "t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 6;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}
} // namespace tile_fwk
} // namespace npu