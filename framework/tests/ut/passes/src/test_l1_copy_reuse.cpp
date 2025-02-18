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
 * \file test_l1_copy_reuse.cpp
 * \brief Unit test for L1CopyInReuseMerge pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/l1_copy_reuse.h"

using namespace npu::tile_fwk;
namespace npu {
namespace tile_fwk {
class L1CopyInReuseTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

        config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
        config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{0,2}});
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0,1}});
    }

    void TearDown() override {}
};

TEST_F(L1CopyInReuseTest, TwoCopyIn) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestL1CopyInReuse", "TestL1CopyInReuse", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->tensor->rawmagic = 1;
    incast1->memoryTypeToBe_ = MEM_DEVICE_DDR;
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast2->tensor->rawmagic = 1;
    incast2->memoryTypeOriginal_ = MEM_DEVICE_DDR;
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->memoryTypeOriginal_ = MEM_L1;
    tensor1->tensor->rawmagic = 2;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->memoryTypeOriginal_ = MEM_L1;
    tensor3->tensor->rawmagic = 3;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    copy_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
    auto &copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_L1_TO_L0A, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);

    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {incast2});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.UpdateSubgraphID(subGraphID1);
    auto &alloc_op1 = currFunctionPtr->AddOperation(Opcode::OP_L1_ALLOC, {}, {tensor3});
    alloc_op1.UpdateSubgraphID(subGraphID1);
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    incast2->AddConsumer(copy_op2);
    copy_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
    auto &copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_L1_TO_L0A, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    L1CopyInReuseMerge pass;
    pass.PreCheck(*currFunctionPtr);
    pass.RunOnFunction(*currFunctionPtr);
    pass.PostCheck(*currFunctionPtr);
}

void InitGraphBuilder(ComputationalGraphBuilder &G, std::vector<int64_t> tileShape, const int subGraphNum) {
    auto shapeImme = OpImmediate::Specified(tileShape);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_VIEW}, {{"incast0"}}, {{"incast1"}}, {"view"}, true), true);
    G.GetOp("view")->UpdateSubgraphID(0);
    G.GetTensor("incast1")->tensor->rawmagic = 1;
    G.GetTensor("incast1")->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_VIEW, Opcode::OP_EXP};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"VIEW_" + strID, "EXP_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("VIEW_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("VIEW_" + strID)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, 
                                                    MEM_L1, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>()));
        G.GetTensor("tensor" + strID)->SetMemoryTypeOriginal(MEM_L1);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
}

TEST_F(L1CopyInReuseTest, TestInvalidOp) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensorL1"}), true);
    G.GetTensor("tensorL1")->SetMemoryTypeOriginal(MEM_L1);
    EXPECT_EQ(G.AddOps({Opcode::OP_GATHER_IN_L1}, {{"incast1"}}, {{"tensorL1"}}, {"gather_in_l1"}, true), true);
    G.GetOp("gather_in_l1")->UpdateSubgraphID(1);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
}


TEST_F(L1CopyInReuseTest, TestNormal) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int result = 5;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoL1Num) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int cube_nbuffer = 2;
    const int sg_cube_parallel_num = 4;
    const int result = 11;
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoL1Map) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int result = 5;
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoBufferMap) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int result = 5;
    const int subGraphNum = 20;
    auto shapeImme = OpImmediate::Specified(tileShape);
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestNoParam) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int result = 20;
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestInvalidL1Num) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int cube_nbuffer = 4;
    const int l1_reuse_num = -1;
    const int sg_cube_parallel_num = 4;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}

TEST_F(L1CopyInReuseTest, TestInvalidL1Map) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{-2, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    function->paramConfigs_.cubeL1ReuseSetting = {{0, -3}};
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
    function->paramConfigs_.cubeL1ReuseSetting = {{-1, 2}};
    function->paramConfigs_.cubeNBufferSetting = {{0, -5}};
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}

// 健康检查用例:静态图和非静态图
TEST_F(L1CopyInReuseTest, TestHealthReport) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int result = 5;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);

    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);

    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);

    nlohmann::json report;
    const int maxFaninOpsResult = 29;
    const int maxFanoutOpsResult = 1;
    const int totalOpCount = 30;
    const int peakMemoryUsage = 512;
    const int copyDataCount = 0;
    // 计算operation节点信息
    CalcOperatorInfo(*function, report);
    EXPECT_EQ(report["totalOpCount"], totalOpCount);
    EXPECT_EQ(report["peakMemory"]["peakMemoryUsage"], peakMemoryUsage);
    EXPECT_EQ(report["copyDataCount"], copyDataCount);

    // 构建operation节点图
    std::vector<std::vector<int>> inMap; // magic到magic的映射，in - parent, out - child
    std::vector<std::vector<int>> outMap;
    std::vector<bool> actualMagic;
    GetOpConnectionMap(*function, inMap, outMap, actualMagic);

    // 计算图信息
    CalcGraphMetrics(inMap, outMap, actualMagic, report);
    EXPECT_EQ(report["maxFaninOps"].size(), maxFaninOpsResult);
    EXPECT_EQ(report["maxFanoutOps"].size(), maxFanoutOpsResult);

    // 计算operation节点信息，静态图下部分字段不计算
    nlohmann::json reportNull;
    function->SetFunctionType(FunctionType::DYNAMIC);
    EXPECT_EQ(LCRM.RunOnFunction(*function), SUCCESS);
    CalcOperatorInfo(*function, reportNull);
    EXPECT_EQ(reportNull["peakMemory"], nullptr);
    EXPECT_EQ(reportNull["copyDataCount"], nullptr);
}

TEST_F(L1CopyInReuseTest, TestGeneralizationL1CopyIn) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int cube_nbuffer = 4;
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int result = 5;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast0", "outcast1"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_VIEW}, {{"incast0"}}, {{"incast1"}}, {"view"}, true), true);
    G.GetOp("view")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    G.GetTensor("incast1")->tensor->rawmagic = 1;
    G.GetTensor("incast1")->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_CONVERT, Opcode::OP_MUL};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor" + strID}, {"outcast0", "outcast1"}};
        std::vector<std::string> opNames{"CONVERT_" + strID, "MUL_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("CONVERT_" + strID)->UpdateSubgraphID(i);
        G.GetOp("MUL_" + strID)->UpdateSubgraphID(i);
        G.GetOp("CONVERT_" + strID)->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MEM_DEVICE_DDR, MEM_L1));
        G.GetTensor("tensor" + strID)->SetMemoryTypeOriginal(MEM_L1);
    }

    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast0", "outcast1"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.cubeNBufferSetting = {{1, 2}, {-1, cube_nbuffer}};
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("myStrategy", {
        {"L1CopyInReuseMerge", "L1CopyInReuseMerge"},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *function, "myStrategy");
    // L1CopyInReuseMerge LCRM;
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(L1CopyInReuseTest, TestTensorReuseFailed) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    auto shapeImme = OpImmediate::Specified(tileShape);
    const int l1_reuse_num = 2;
    const int sg_cube_parallel_num = 4;
    const int subGraphNum = 20;
    InitGraphBuilder(G, tileShape, subGraphNum);
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor_before" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_EXP};
        std::vector<std::vector<std::string>> iOperands{{"tensor_before" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor" + strID}};
        std::vector<std::string> opNames{"EXP_BEFORE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("EXP_BEFORE_" + strID)->UpdateSubgraphID(i);
        G.GetTensor("tensor_before" + strID)->SetMemoryTypeOriginal(MEM_L1);
    }
    G.GetTensor("tensor_before1")->tensor->datatype = DataType::DT_FP16;
    G.GetTensor("tensor1")->tensor->datatype = DataType::DT_FP16;
    Function *function = G.GetFunction();
    function->paramConfigs_.l1ReuseNum = l1_reuse_num;
    function->paramConfigs_.cubeL1ReuseSetting = {{1, 2}};
    function->paramConfigs_.sgCubeParallelNum = sg_cube_parallel_num;
    function->SetTotalSubGraphCount(subGraphNum);
    L1CopyInReuseMerge LCRM;
    EXPECT_EQ(LCRM.RunOnFunction(*function), FAILED);
}
}
}