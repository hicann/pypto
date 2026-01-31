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
 * \file test_reduce_copy.cpp
 * \brief Unit test for ReduceCopy pass.
 */

#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/platform.h"
#include "interface/function/function.h"
#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class ReduceCopyTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ReduceCopyTestStrategy");
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override {}
};

void BuildMatmulAddBranch(ComputationalGraphBuilder &G, int brId, std::vector<std::string>& incasts,
        std::vector<std::string> &outcasts) {
    std::vector<int64_t> tileShape{16,16};
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tRA" + br, "tRB" + br, "tL1A" + br, "tL1B" + br, "tA" + br, "tB" + br, 
        "tC" + br, "tUB" + br};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A,Opcode::OP_L1_TO_L0B, 
        Opcode::OP_A_MUL_B, Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"tRA" + br}, {"tRB" + br}, {"tL1A" + br}, {"tL1B" + br}, 
        {"tA"  + br, "tB"  + br}, {"tC"  + br}};
    std::vector<std::vector<std::string>> ooperands{{"tL1A" + br}, {"tL1B"  + br}, {"tA" + br}, {"tB" + br},
        {"tC"  + br}, {"tUB"  + br}};
    std::vector<std::string> opNames{"view" + br, "view2" + br, "toA" + br, "toB" + br, "matmul" + br, "convert" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tRA" + br);
    incasts.push_back("tRB" + br);
    const int Num100 = 100;
    for (auto opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(brId);
        G.GetOp(opName)->UpdateLatency(Num100);
    }
    const int Num2 = 2;
    for (int k = 0; k < Num2; k++) {
        std::string brv1 = std::to_string(brId + 1 + k);
        std::vector<std::string> tensorNamesV1{"add1" + brv1, "add2" + brv1, "out" + brv1};
        std::vector<Opcode> opCodesV1{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> ioperandsV1{{"tUB"  + br},   {"add1" + brv1}, {"add2" + brv1},};
        std::vector<std::vector<std::string>> ooperandsV1{{"add1" + brv1}, {"add2" + brv1}, {"out" + brv1},};
        std::vector<std::string> opNamesV1{"add1" + brv1, "add2" + brv1, "assemble" + brv1};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNamesV1), true);
        EXPECT_EQ(G.AddOps(opCodesV1, ioperandsV1, ooperandsV1, opNamesV1, true), true);
        outcasts.push_back("out" + brv1);
        for (auto opName : opNamesV1) {
            G.GetOp(opName)->UpdateSubgraphID(brId + 1 + k);
            G.GetOp(opName)->UpdateLatency(Num100);
        }
    }
}

void BuildMatmulAddsGraph(ComputationalGraphBuilder &G) {
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    const int Num3 = 3;
    BuildMatmulAddBranch(G, 0, incasts, outcasts);
    BuildMatmulAddBranch(G, Num3, incasts, outcasts);
    Function *function = G.GetFunction();
    const int Num6 = 6;
    function->SetTotalSubGraphCount(Num6);
    EXPECT_EQ(G.SetInCast(incasts), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

TEST_F(ReduceCopyTest, TestCase0) {
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function *function = G.GetFunction();
    ReduceCopyRunner runner;
    const double lowerBound = 0.1;
    const double upperBound = 10.0;
    runner.mergeThresholds = {{lowerBound, upperBound}};
    EXPECT_EQ(runner.ReduceCopy(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

void BuildConnectMatmul(ComputationalGraphBuilder &G, int brId, std::vector<std::string>& incasts,
        std::vector<std::string> &outcasts) {
    std::vector<int64_t> tileShape{16,16};
    const int Num100 = 100;
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tRA" + br, "tRB" + br, "tL1A" + br, "tL1B" + br, "tA" + br, "tB" + br, 
        "tC" + br, "tGM" + br};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A,Opcode::OP_L1_TO_L0B, 
        Opcode::OP_A_MUL_B, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"tRA" + br}, {"tRB" + br}, {"tL1A" + br}, {"tL1B" + br}, 
        {"tA"  + br, "tB"  + br}, {"tC"  + br}};
    std::vector<std::vector<std::string>> ooperands{{"tL1A" + br}, {"tL1B"  + br}, {"tA" + br}, {"tB" + br},
        {"tC"  + br}, {"tGM"  + br}};
    std::vector<std::string> opNames{"view" + br, "view2" + br, "toA" + br, "toB" + br, "matmul" + br, "convert" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tRA" + br);
    incasts.push_back("tRB" + br);
    for (auto opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(brId);
        G.GetOp(opName)->UpdateLatency(Num100);
    }
    std::string br2 = std::to_string(brId + 1);
    std::vector<std::string> tensorNames2{"tRB" + br2, "tL1A" + br2, "tL1B" + br2, "tA" + br2, "tB" + br2, 
        "tC" + br2, "tGM" + br2};
    std::vector<Opcode> opCodes2{Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A,Opcode::OP_L1_TO_L0B, 
        Opcode::OP_A_MUL_B, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands2{{"tGM" + br}, {"tRB" + br2}, {"tL1A" + br2}, {"tL1B" + br2}, 
        {"tA"  + br2, "tB"  + br2}, {"tC"  + br2}};
    std::vector<std::vector<std::string>> ooperands2{{"tL1A" + br2}, {"tL1B"  + br2}, {"tA" + br2}, {"tB" + br2},
        {"tC"  + br2}, {"tGM"  + br2}};
    std::vector<std::string> opNames2{"view" + br2, "view2" + br2, "toA" + br2, "toB" + br2, "matmul" + br2, "convert" + br2};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames2), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands2, ooperands2, opNames2, true), true);
    incasts.push_back("tRB" + br2);
    outcasts.push_back("tGM" + br2);
    for (auto opName : opNames2) {
        G.GetOp(opName)->UpdateSubgraphID(brId + 1);
        G.GetOp(opName)->UpdateLatency(Num100);
    }
}

void BuildConnectVector(ComputationalGraphBuilder &G, int brId, std::vector<std::string>& incasts,
        std::vector<std::string> &outcasts) {
    std::vector<int64_t> tileShape{16,16};
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tin" + br, "tadds1" + br, "tout" + br};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS};
    std::vector<std::vector<std::string>> ioperands{{"tin" + br}, {"tadds1" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tadds1" + br}, {"tout" + br}};
    std::vector<std::string> opNames{"adds1" + br, "adds2" + br}; 
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tin" + br);
    outcasts.push_back("tout" + br);
    const int Num100 = 100;
    G.GetOp("adds1" + br)->UpdateSubgraphID(brId);
    G.GetOp("adds1" + br)->UpdateLatency(Num100);
    G.GetOp("adds2" + br)->UpdateSubgraphID(brId + 1);
    G.GetOp("adds2" + br)->UpdateLatency(Num100);
}

void BuildConnect(ComputationalGraphBuilder &G) {
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    const int Num2 = 2;
    BuildConnectMatmul(G, 0, incasts, outcasts);
    BuildConnectVector(G, Num2, incasts, outcasts);
    Function *function = G.GetFunction();
    const int Num4 = 4;
    function->SetTotalSubGraphCount(Num4);
    EXPECT_EQ(G.SetInCast(incasts), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

TEST_F(ReduceCopyTest, TestCase1) {
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function *function = G.GetFunction();
    ReduceCopyRunner runner;
    const double lowerBound = 0.1;
    const double upperBound = 10.0;
    runner.mergeThresholds = {{lowerBound, upperBound}};
    EXPECT_EQ(runner.ReduceCopy(*function), SUCCESS);
    const int Num4 = 4;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num4);
}

}
}