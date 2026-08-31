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
 * \file test_vf_fusion_cluster_identify.cpp
 * \brief Unit test for VFFusionClusterIdentify.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "computational_graph_builder.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "passes/block_graph_pass/vf_fusion_cluster_identify.h"
#include "passes/pass_interface/pass.h"

#include "passes/pass_mgr/pass_manager.h"
#include "passes/pass_mgr/pass_registry.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

class VFFusionClusterIdentifyTestAccessor {
public:
    static void MarkVfFusionCluster(VFFusionClusterIdentify& pass, Operation& op) { pass.SetVfFusionCluster(op, true); }

    static bool VerifyScheduleTopology(VFFusionClusterIdentify& pass, Function& function,
                                       const std::vector<Operation*>& schedule)
    {
        auto graph = pass.BuildGraph(function);
        return pass.VerifyScheduleTopology(graph, schedule);
    }

    static std::vector<Operation*> BuildDfsPriorityTopoOps(VFFusionClusterIdentify& pass, Function& function)
    {
        auto graph = pass.BuildGraph(function);
        std::vector<size_t> topoOrder;
        if (!pass.BuildDfsPriorityTopoOrder(graph, topoOrder)) {
            return {};
        }
        std::vector<Operation*> result;
        result.reserve(topoOrder.size());
        for (size_t opIndex : topoOrder) {
            if (opIndex >= graph.ops.size()) {
                return {};
            }
            result.emplace_back(graph.ops[opIndex]);
        }
        return result;
    }
};

namespace {
constexpr int VF_CLUSTER_ID_START_FOR_TEST = 200000000;
constexpr int VF_CLUSTER_SIZE_LIMIT_FOR_TEST = 32;
constexpr int USER_ATOMIC_SCOPE_ID_FOR_TEST = 8;

std::shared_ptr<Function> CreateRootFunction(const std::string& name)
{
    auto rootFunc = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    rootFunc->rootFunc_ = rootFunc.get();
    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);
    return rootFunc;
}

std::shared_ptr<Function> CreateLeafFunction(Function& rootFunc, const std::string& name)
{
    auto leafFunc = std::make_shared<Function>(Program::GetInstance(), name, name, &rootFunc);
    leafFunc->rootFunc_ = &rootFunc;
    leafFunc->SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc->SetFunctionType(FunctionType::STATIC);
    rootFunc.rootFunc_->programs_.emplace(leafFunc->GetFuncMagic(), leafFunc.get());
    return leafFunc;
}

std::vector<int> GetOpMagics(Function& function)
{
    std::vector<int> result;
    for (auto* op : function.Operations(false).DuplicatedOpList()) {
        result.emplace_back(op->GetOpMagic());
    }
    return result;
}

std::vector<int> GetOpMagics(const std::vector<Operation*>& ops)
{
    std::vector<int> result;
    for (auto* op : ops) {
        result.emplace_back(op->GetOpMagic());
    }
    return result;
}

} // namespace

TEST(AncestorBitsTest, PropagatesAncestorsAcrossBranchJoinAndDescendant)
{
    AncestorBits ancestorBits;
    const std::vector<std::vector<size_t>> producers = {
        {},     // 0: left source
        {},     // 1: right source
        {0},    // 2: left branch
        {1},    // 3: right branch
        {2, 3}, // 4: branch join
        {4},    // 5: downstream consumer
        {1},    // 6: independent side branch
    };
    const std::vector<size_t> topoOrder = {0, 1, 2, 3, 4, 5, 6};

    ancestorBits.Build(producers, topoOrder);

    EXPECT_EQ(ancestorBits.Size(), producers.size());
    for (size_t ancestorIndex : {0UL, 1UL, 2UL, 3UL, 4UL}) {
        EXPECT_TRUE(ancestorBits.IsAncestor(5, ancestorIndex));
    }
    EXPECT_TRUE(ancestorBits.IsAncestor(4, 0));
    EXPECT_TRUE(ancestorBits.IsAncestor(4, 1));
    EXPECT_TRUE(ancestorBits.IsAncestor(6, 1));
    EXPECT_FALSE(ancestorBits.IsAncestor(2, 1));
    EXPECT_FALSE(ancestorBits.IsAncestor(6, 0));
    EXPECT_FALSE(ancestorBits.IsAncestor(4, 5));
    EXPECT_FALSE(ancestorBits.IsAncestor(5, 7));
}

TEST(AncestorBitsTest, TracksAncestorsAcrossBitsetWordBoundary)
{
    constexpr size_t nodeCount = 130;
    AncestorBits ancestorBits;
    std::vector<std::vector<size_t>> producers(nodeCount);
    std::vector<size_t> topoOrder;
    topoOrder.reserve(nodeCount);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
        topoOrder.emplace_back(nodeIndex);
        if (nodeIndex > 0) {
            producers[nodeIndex].emplace_back(nodeIndex - 1);
        }
    }

    ancestorBits.Build(producers, topoOrder);

    EXPECT_EQ(ancestorBits.Size(), nodeCount);
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 0));
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 63));
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 64));
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 128));
    EXPECT_TRUE(ancestorBits.IsAncestor(64, 63));
    EXPECT_FALSE(ancestorBits.IsAncestor(64, 64));
    EXPECT_FALSE(ancestorBits.IsAncestor(63, 64));
    EXPECT_FALSE(ancestorBits.IsAncestor(nodeCount - 1, nodeCount));
}

class VFFusionClusterIdentifyTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPassGlobalConfig(KEY_ENABLE_VF, true);
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    }

    void TearDown() override {}
};

TEST_F(VFFusionClusterIdentifyTest, AssignsOneClusterToDependentPipeVChain)
{
    auto rootFunc = CreateRootFunction("TestVFFusionChainRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionChainLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    auto* copyOut = graph.GetOp("CopyOut");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);
    ASSERT_NE(copyOut, nullptr);
    const auto originalOrder = GetOpMagics(*leafFunc);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(copyOut->GetAtomicScopeId(), -1);
    EXPECT_EQ(GetOpMagics(*leafFunc), originalOrder);
}

TEST_F(VFFusionClusterIdentifyTest, AssignsDistinctClusterIdsAcrossLeafFunctions)
{
    auto rootFunc = CreateRootFunction("TestVFFusionDistinctIdRoot");
    auto firstLeafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDistinctIdFirstLeaf");
    auto secondLeafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDistinctIdSecondLeaf");

    ComputationalGraphBuilder firstGraph(firstLeafFunc.get());
    ASSERT_TRUE(firstGraph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2"}));
    ASSERT_TRUE(firstGraph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "FirstExp"));
    ASSERT_TRUE(firstGraph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "FirstSqrt"));
    auto* firstExp = firstGraph.GetOp("FirstExp");
    auto* firstSqrt = firstGraph.GetOp("FirstSqrt");
    ASSERT_NE(firstExp, nullptr);
    ASSERT_NE(firstSqrt, nullptr);

    ComputationalGraphBuilder secondGraph(secondLeafFunc.get());
    ASSERT_TRUE(secondGraph.AddTensors(DataType::DT_FP32, {16, 16}, {"b0", "b1", "b2"}));
    ASSERT_TRUE(secondGraph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "SecondExp"));
    ASSERT_TRUE(secondGraph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "SecondSqrt"));
    auto* secondExp = secondGraph.GetOp("SecondExp");
    auto* secondSqrt = secondGraph.GetOp("SecondSqrt");
    ASSERT_NE(secondExp, nullptr);
    ASSERT_NE(secondSqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    const int firstClusterId = firstLeafFunc->GetFuncMagic() < secondLeafFunc->GetFuncMagic() ?
                                   firstExp->GetAtomicScopeId() :
                                   secondExp->GetAtomicScopeId();
    const int secondClusterId = firstLeafFunc->GetFuncMagic() < secondLeafFunc->GetFuncMagic() ?
                                    secondExp->GetAtomicScopeId() :
                                    firstExp->GetAtomicScopeId();
    EXPECT_EQ(firstClusterId, VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(secondClusterId, VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_NE(firstClusterId, secondClusterId);
    EXPECT_EQ(firstExp->GetAtomicScopeId(), firstSqrt->GetAtomicScopeId());
    EXPECT_EQ(secondExp->GetAtomicScopeId(), secondSqrt->GetAtomicScopeId());
}

TEST_F(VFFusionClusterIdentifyTest, KeepsUserAtomicScopeSeparateFromGeneratedClusterIds)
{
    auto rootFunc = CreateRootFunction("TestVFFusionUserScopeSeparateRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionUserScopeSeparateLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"u0", "u1", "a0", "a1", "a2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"u0"}, {"u1"}, "UserExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    auto* userExp = graph.GetOp("UserExp");
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    ASSERT_NE(userExp, nullptr);
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    constexpr int userScopeId = USER_ATOMIC_SCOPE_ID_FOR_TEST;
    userExp->SetAtomicScopeId(userScopeId);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(userExp->GetAtomicScopeId(), userScopeId);
    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_NE(userExp->GetAtomicScopeId(), aExp->GetAtomicScopeId());
}

TEST_F(VFFusionClusterIdentifyTest, MovesUnrelatedMiddleOpOutOfClusterWindow)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMoveMiddleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMoveMiddleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "a3", "b0", "b1", "b2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"a2"}, {"a3"}, "ARsqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"a3"}, {"out"}, "CopyOut"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* aRsqrt = graph.GetOp("ARsqrt");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(aRsqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(GetOpMagics(*leafFunc),
              std::vector<int>({aExp->GetOpMagic(), aSqrt->GetOpMagic(), aRsqrt->GetOpMagic(),
                                graph.GetOp("CopyOut")->GetOpMagic(), bExp->GetOpMagic(), bSqrt->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, BuildsDfsPriorityTopoOrderIndependentOfInitialSchedule)
{
    auto rootFunc = CreateRootFunction("TestVFFusionDfsOrderRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDfsOrderLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "a3", "a4", "b0", "b1"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"a2"}, {"a3"}, "Consumer"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a3"}, {"a4"}, "Descendant"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* consumer = graph.GetOp("Consumer");
    auto* descendant = graph.GetOp("Descendant");
    auto* bExp = graph.GetOp("BExp");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(consumer, nullptr);
    ASSERT_NE(descendant, nullptr);
    ASSERT_NE(bExp, nullptr);

    leafFunc->ScheduleBy({bExp, aExp, descendant, aSqrt, consumer}, true);

    VFFusionClusterIdentify pass;
    auto topoOps = VFFusionClusterIdentifyTestAccessor::BuildDfsPriorityTopoOps(pass, *leafFunc);

    EXPECT_EQ(GetOpMagics(*leafFunc),
              std::vector<int>({bExp->GetOpMagic(), aExp->GetOpMagic(), descendant->GetOpMagic(), aSqrt->GetOpMagic(),
                                consumer->GetOpMagic()}));
    EXPECT_EQ(GetOpMagics(topoOps), std::vector<int>({aExp->GetOpMagic(), aSqrt->GetOpMagic(), consumer->GetOpMagic(),
                                                      descendant->GetOpMagic(), bExp->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, RejectsTopologicalScheduleWithNonContiguousVfCluster)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNonContiguousClusterRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonContiguousClusterLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(
        DataType::DT_FP32, {16, 16},
        {"clusterInput0", "clusterOutput0", "middleInput", "middleOutput", "clusterInput1", "clusterOutput1"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"clusterInput0"}, {"clusterOutput0"}, "ClusterExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"middleInput"}, {"middleOutput"}, "MiddleNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"clusterInput1"}, {"clusterOutput1"}, "ClusterSqrt"));
    auto* clusterExp = graph.GetOp("ClusterExp");
    auto* middleNeg = graph.GetOp("MiddleNeg");
    auto* clusterSqrt = graph.GetOp("ClusterSqrt");
    ASSERT_NE(clusterExp, nullptr);
    ASSERT_NE(middleNeg, nullptr);
    ASSERT_NE(clusterSqrt, nullptr);

    constexpr int clusterId = VF_CLUSTER_ID_START_FOR_TEST;
    clusterExp->SetAtomicScopeId(clusterId);
    clusterSqrt->SetAtomicScopeId(clusterId);
    VFFusionClusterIdentify pass;
    VFFusionClusterIdentifyTestAccessor::MarkVfFusionCluster(pass, *clusterExp);
    VFFusionClusterIdentifyTestAccessor::MarkVfFusionCluster(pass, *clusterSqrt);

    const std::vector<Operation*> schedule = {clusterExp, middleNeg, clusterSqrt};
    EXPECT_FALSE(VFFusionClusterIdentifyTestAccessor::VerifyScheduleTopology(pass, *leafFunc, schedule));
}

TEST_F(VFFusionClusterIdentifyTest, DissolvesSingleOpClusters)
{
    auto rootFunc = CreateRootFunction("TestVFFusionSingleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionSingleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t1"}, {"t2"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    ASSERT_NE(exp, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesConsumerWithMultipleInputClusters)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMultiInputRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMultiInputLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "b0", "b1", "b2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "b2"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MergesConsumerOnlyWithLeafClusterThroughNonVfSideBranch)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNonVfSideBranchRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonVfSideBranchLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(
        DataType::DT_FP32, {16, 16},
        {"input", "mainExpOut", "mainSqrtOut", "mainRsqrtOut", "sideNegOut", "sideAbsOut", "sideExpOut", "joinOut"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"input"}, {"mainExpOut"}, "MainExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"mainExpOut"}, {"mainSqrtOut"}, "MainSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"mainSqrtOut"}, {"mainRsqrtOut"}, "MainRsqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"mainExpOut"}, {"sideNegOut"}, "SideNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"sideNegOut"}, {"sideAbsOut"}, "SideAbs"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"sideAbsOut"}, {"sideExpOut"}, "SideExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"mainRsqrtOut", "sideExpOut"}, {"joinOut"}, "JoinAdd"));
    auto* mainExp = graph.GetOp("MainExp");
    auto* mainSqrt = graph.GetOp("MainSqrt");
    auto* mainRsqrt = graph.GetOp("MainRsqrt");
    auto* sideNeg = graph.GetOp("SideNeg");
    auto* sideAbs = graph.GetOp("SideAbs");
    auto* sideExp = graph.GetOp("SideExp");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(mainExp, nullptr);
    ASSERT_NE(mainSqrt, nullptr);
    ASSERT_NE(mainRsqrt, nullptr);
    ASSERT_NE(sideNeg, nullptr);
    ASSERT_NE(sideAbs, nullptr);
    ASSERT_NE(sideExp, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(mainExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mainSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mainRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(sideNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(sideAbs->GetAtomicScopeId(), -1);
    EXPECT_EQ(sideExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_NE(mainRsqrt->GetAtomicScopeId(), joinAdd->GetAtomicScopeId());
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenNonVfBarrierWouldSplitCluster)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNonVfBarrierRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonVfBarrierLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(
        graph.AddTensors(DataType::DT_FP32, {16, 16},
                         {"branchInput", "branchExpOut", "branchSqrtOut", "branchBarrierOut", "blockedJoinOut"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"branchInput"}, {"branchExpOut"}, "BranchExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"branchExpOut"}, {"branchBarrierOut"}, "BranchBarrierNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"branchExpOut"}, {"branchSqrtOut"}, "BranchSqrt"));
    ASSERT_TRUE(
        graph.AddOp(Opcode::OP_ADD, {"branchSqrtOut", "branchBarrierOut"}, {"blockedJoinOut"}, "BlockedJoinAdd"));
    auto* branchExp = graph.GetOp("BranchExp");
    auto* branchBarrierNeg = graph.GetOp("BranchBarrierNeg");
    auto* branchSqrt = graph.GetOp("BranchSqrt");
    auto* blockedJoinAdd = graph.GetOp("BlockedJoinAdd");
    ASSERT_NE(branchExp, nullptr);
    ASSERT_NE(branchBarrierNeg, nullptr);
    ASSERT_NE(branchSqrt, nullptr);
    ASSERT_NE(blockedJoinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(branchExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(branchSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(branchBarrierNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(blockedJoinAdd->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesOnlyInputClustersThatFitSizeLimit)
{
    auto rootFunc = CreateRootFunction("TestVFFusionPartialSizeRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionPartialSizeLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    std::vector<std::string> tensorNames;
    for (int i = 0; i <= static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        tensorNames.emplace_back("a" + std::to_string(i));
    }
    tensorNames.insert(tensorNames.end(), {"b0", "b1", "b2", "out"});
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a" + std::to_string(i)}, {"a" + std::to_string(i + 1)},
                                "AExp" + std::to_string(i)));
    }
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(
        graph.AddOp(Opcode::OP_ADD, {"a" + std::to_string(VF_CLUSTER_SIZE_LIMIT_FOR_TEST), "b2"}, {"out"}, "JoinAdd"));
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        auto* op = graph.GetOp("AExp" + std::to_string(i));
        ASSERT_NE(op, nullptr);
    }
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        EXPECT_EQ(graph.GetOp("AExp" + std::to_string(i))->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    }
    EXPECT_EQ(bExp->GetAtomicScopeId(), bSqrt->GetAtomicScopeId());
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), bExp->GetAtomicScopeId());
    EXPECT_NE(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MovesMiddleOpAfterConsumerWhenItDependsOnCluster)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMoveAfterRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMoveAfterLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "side", "sideOut", "a3", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "side"}, {"sideOut"}, "MiddleAdd"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"a2"}, {"a3"}, "ARsqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"a3"}, {"out"}, "CopyOut"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* middleAdd = graph.GetOp("MiddleAdd");
    auto* aRsqrt = graph.GetOp("ARsqrt");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(middleAdd, nullptr);
    ASSERT_NE(aRsqrt, nullptr);
    middleAdd->SetAtomicScopeId(USER_ATOMIC_SCOPE_ID_FOR_TEST);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(middleAdd->GetAtomicScopeId(), USER_ATOMIC_SCOPE_ID_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MovesMiddleOpBeforeClusterWhenConsumerDependsOnIt)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMoveBeforeRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMoveBeforeLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "side0", "side1", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"side0"}, {"side1"}, "SideNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "side1"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* sideNeg = graph.GetOp("SideNeg");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(sideNeg, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(sideNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(GetOpMagics(*leafFunc), std::vector<int>({sideNeg->GetOpMagic(), aExp->GetOpMagic(), aSqrt->GetOpMagic(),
                                                        joinAdd->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenMiddleOpMustStayBetweenClusterAndConsumer)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMiddleBarrierRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMiddleBarrierLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "mid", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"a2"}, {"mid"}, "MiddleNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "mid"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* middleNeg = graph.GetOp("MiddleNeg");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(middleNeg, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(middleNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), -1);
    EXPECT_EQ(GetOpMagics(*leafFunc), std::vector<int>({aExp->GetOpMagic(), aSqrt->GetOpMagic(),
                                                        middleNeg->GetOpMagic(), joinAdd->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, MergesPastUserScopedSideInput)
{
    auto rootFunc = CreateRootFunction("TestVFFusionRejectSideInputRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionRejectSideInputLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "u0", "u1", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"u0"}, {"u1"}, "UserSide"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "u1"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* userSide = graph.GetOp("UserSide");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(userSide, nullptr);
    ASSERT_NE(joinAdd, nullptr);
    userSide->SetAtomicScopeId(USER_ATOMIC_SCOPE_ID_FOR_TEST);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(userSide->GetAtomicScopeId(), USER_ATOMIC_SCOPE_ID_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, RespectsClusterSizeLimit)
{
    auto rootFunc = CreateRootFunction("TestVFFusionSizeLimitRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionSizeLimitLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    std::vector<std::string> tensorNames;
    for (int i = 0; i <= static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST + 1); i++) {
        tensorNames.emplace_back("t" + std::to_string(i));
    }
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST + 1); i++) {
        ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t" + std::to_string(i)}, {"t" + std::to_string(i + 1)},
                                "Exp" + std::to_string(i)));
    }
    std::vector<Operation*> ops;
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST + 1); i++) {
        auto* op = graph.GetOp("Exp" + std::to_string(i));
        ASSERT_NE(op, nullptr);
        ops.emplace_back(op);
    }

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    for (size_t i = 0; i < VF_CLUSTER_SIZE_LIMIT_FOR_TEST; i++) {
        EXPECT_EQ(ops[i]->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    }
    EXPECT_EQ(ops.back()->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, DoesNothingWhenVfDisabled)
{
    config::SetPassGlobalConfig(KEY_ENABLE_VF, false);
    auto rootFunc = CreateRootFunction("TestVFFusionDisabledRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDisabledLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, DoesNothingOnNonA5Arch)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    auto rootFunc = CreateRootFunction("TestVFFusionNonA5Root");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonA5Leaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, FailsOnCyclicOperationDependencies)
{
    auto rootFunc = CreateRootFunction("TestVFFusionCycleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionCycleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "First"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Second"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t2"}, {"t1"}, "Third"));

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), FAILED);
}

TEST_F(VFFusionClusterIdentifyTest, MergesExpandFollowedByElementwise)
{
    auto rootFunc = CreateRootFunction("TestExpandEleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestExpandEleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXPAND, {"t0"}, {"t1"}, "Expand"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t1"}, {"t2"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* expand = graph.GetOp("Expand");
    auto* exp = graph.GetOp("Exp");
    ASSERT_NE(expand, nullptr);
    ASSERT_NE(exp, nullptr);
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(expand->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MergesElementwiseFollowedByExpand)
{
    auto rootFunc = CreateRootFunction("TestEleExpandRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestEleExpandLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16, 16}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16, 16}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "Expand"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    auto* expand = graph.GetOp("Expand");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(expand, nullptr);
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Producer (Exp) is not a reduce op, so merge is allowed regardless of shape difference.
    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(expand->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MergesElementwiseFollowedByReduce)
{
    auto rootFunc = CreateRootFunction("TestEleReduceRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestEleReduceLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"t1"}, {"t2"}, "Reduce"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    auto* reduce = graph.GetOp("Reduce");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(reduce, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(reduce->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenReduceFollowedByElementwise)
{
    auto rootFunc = CreateRootFunction("TestReduceEleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestReduceEleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"t0"}, {"t1"}, "Reduce"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t1"}, {"t2"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* reduce = graph.GetOp("Reduce");
    auto* exp = graph.GetOp("Exp");
    ASSERT_NE(reduce, nullptr);
    ASSERT_NE(exp, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Reduce producer check rejects merge: exp's producer is a reduce op.
    // Both ops end up without a cluster: exp can't merge, and reduce becomes a
    // singleton cluster that gets dissolved by DissolveSingletonClusters.
    EXPECT_EQ(reduce->GetAtomicScopeId(), -1);
    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
}

} // namespace npu::tile_fwk
