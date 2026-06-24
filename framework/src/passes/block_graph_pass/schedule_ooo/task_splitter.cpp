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
 * \file task_splitter.cpp
 * \brief TaskSplitter: DAG task graph splitting, clustering, and merging logic.
 */

#include "task_splitter.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <queue>
#include <stack>

#ifndef MODULE_NAME
#define MODULE_NAME "CoreAssign"
#endif

namespace npu::tile_fwk {

// Alloc op需要与其同级的op处于同一个子图中, Convert的alloc应跟随其后op
void TaskSplitter::BuildSameLayerConnectionWithBack()
{
    for (size_t i = 0; i < opList_.size(); i++) {
        if (ALLOC_OPCODE.count(opList_[i]->GetOpcode()) == 0) {
            continue;
        }
        ScheduleCoreType srcCoreType = opCoreTypes_[i];
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Found alloc op %s[%d].", opList_[i]->GetOpcodeStr().c_str(),
            opList_[i]->GetOpMagic());
        bool withBack = true;
        for (auto& oop : opList_[i]->GetOOperands()) {
            for (auto& sameLayerOpPtr : oop->GetProducers()) {
                int dstOpMagic = sameLayerOpPtr->GetOpMagic();
                if (opMagicToIdx_.count(dstOpMagic) == 0) {
                    continue;
                }
                if (opCoreTypes_[opMagicToIdx_[dstOpMagic]] != srcCoreType ||
                    opList_[i]->GetOpMagic() == sameLayerOpPtr->GetOpMagic()) {
                    continue;
                }
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "-- add %s[%d] to same layer connection because of the alloc op.",
                    sameLayerOpPtr->GetOpcodeStr().c_str(), sameLayerOpPtr->GetOpMagic());
                sameLayerConnection_.push_back({i, opMagicToIdx_[sameLayerOpPtr->GetOpMagic()]});
                withBack = false;
            }
        }
        if (withBack) {
            for (auto& oop : opList_[i]->GetOOperands()) {
                auto& consumers = oop->GetConsumers();
                if (consumers.empty()) {
                    continue;
                }
                auto& sameLayerOpPtr = *consumers.begin();
                int dstOpMagic = sameLayerOpPtr->GetOpMagic();
                if (opMagicToIdx_.count(dstOpMagic) == 0) {
                    continue;
                }
                if (opCoreTypes_[opMagicToIdx_[dstOpMagic]] != srcCoreType) {
                    continue;
                }
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "-- add %s[%d] to same layer connection because of the alloc op.",
                    sameLayerOpPtr->GetOpcodeStr().c_str(), sameLayerOpPtr->GetOpMagic());
                sameLayerConnection_.push_back({i, opMagicToIdx_[sameLayerOpPtr->GetOpMagic()]});
            }
        }
    }
}

// Alloc op需要与其同级的op处于同一个子图中, Convert的alloc应跟随其前op
void TaskSplitter::BuildSameLayerConnectionWithFront()
{
    for (size_t i = 0; i < opList_.size(); i++) {
        if (ALLOC_OPCODE.count(opList_[i]->GetOpcode()) == 0) {
            continue;
        }
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Found alloc op %s[%d].", opList_[i]->GetOpcodeStr().c_str(),
            opList_[i]->GetOpMagic());
        for (auto& oop : opList_[i]->GetOOperands()) {
            for (auto& sameLayerOpPtr : oop->GetProducers()) {
                int dstOpMagic = sameLayerOpPtr->GetOpMagic();
                if (opMagicToIdx_.count(dstOpMagic) == 0) {
                    continue;
                }
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "-- add %s[%d] to same layer connection because of the alloc op.",
                    sameLayerOpPtr->GetOpcodeStr().c_str(), sameLayerOpPtr->GetOpMagic());
                sameLayerConnection_.push_back({i, opMagicToIdx_[sameLayerOpPtr->GetOpMagic()]});
                opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[dstOpMagic]];
            }
        }
    }
}

// 构建op的coreType和连接图
void TaskSplitter::BuildOpGraph()
{
    int opNum = static_cast<int>(opList_.size());
    opCoreTypes_.resize(opNum);
    opMagicToIdx_.clear();
    for (int i = 0; i < opNum; i++) {
        opMagicToIdx_[opList_[i]->GetOpMagic()] = i;
        opCoreTypes_[i] = OpcodeManager::Inst().GetCoreType(opList_[i]->GetOpcode()) == OpCoreType::AIC ?
                              ScheduleCoreType::AIC :
                              ScheduleCoreType::AIV;
    }
    for (int i = 0; i < opNum; i++) {
        if (opList_[i]->GetOpcode() == Opcode::OP_COPY_IN) {
            auto nextOp = *opList_[i]->ConsumerOpsOrdered().begin();
            opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[nextOp->GetOpMagic()]];
        } else if (opList_[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            auto prevOp = *opList_[i]->ProducerOpsOrdered().begin();
            opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[prevOp->GetOpMagic()]];
        }
        if (opList_[i]->HasAttribute(OpAttributeKey::isCube)) {
            bool isCube = opList_[i]->GetBoolAttribute(OpAttributeKey::isCube);
            opCoreTypes_[i] = isCube ? ScheduleCoreType::AIC : ScheduleCoreType::AIV;
        }
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Mark %s[%d] as %s core type.", opList_[i]->GetOpcodeStr().c_str(),
            opList_[i]->GetOpMagic(), opCoreTypes_[i] == ScheduleCoreType::AIC ? "AIC" : "AIV");
    }
    APASS_LOG_INFO_F(Elements::Operation, "Mark core type finished.");
    opInGraph_.resize(opNum);
    opOutGraph_.resize(opNum);
    for (int i = 0; i < opNum; i++) {
        for (auto consumerOp : opList_[i]->ConsumerOpsOrdered()) {
            if (opMagicToIdx_.count(consumerOp->GetOpMagic()) == 0) {
                continue;
            }
            int nextOpIdx = opMagicToIdx_[consumerOp->GetOpMagic()];
            opOutGraph_[i].insert(nextOpIdx);
            opInGraph_[nextOpIdx].insert(i);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "Build op connection graph finished.");
}

// 记录成环的 Cluster 对
void TaskSplitter::RecordCycledClusters(
    const std::vector<ScheduleCoreType>& clusterCoreTypes, const std::vector<std::vector<int>>& sccResult)
{
    cycledTaskNodePairs_.clear();
    for (int sccId = 0; sccId < static_cast<int>(sccResult.size()); sccId++) {
        if (sccResult[sccId].size() <= 1) {
            continue;
        }
        // 这个 SCC 有成环。找出包含的 AIC 和 AIV cluster
        bool hasAIC = false;
        bool hasAIV = false;
        for (int clusterId : sccResult[sccId]) {
            if (clusterCoreTypes[clusterId] == ScheduleCoreType::AIC) {
                hasAIC = true;
            } else {
                hasAIV = true;
            }
        }

        // 输出日志：记录成环的 Cluster
        std::string sccInfo = "SCC " + std::to_string(sccId) + " has cycle with " +
                              std::to_string(sccResult[sccId].size()) + " clusters: [";
        for (size_t j = 0; j < sccResult[sccId].size(); j++) {
            int cid = sccResult[sccId][j];
            sccInfo +=
                "(cluster=" + std::to_string(cid) + ", core=" + ScheduleCoreTypeToString(clusterCoreTypes[cid]) + ")";
            if (j + 1 < sccResult[sccId].size()) {
                sccInfo += ", ";
            }
        }
        sccInfo += "]";
        APASS_LOG_WARN_F(Elements::Operation, "%s", sccInfo.c_str());

        // 如果同时包含 AIC 和 AIV，FlattenSCC 后会拆成两个新 Cluster,CombineSCC 会消除 SCC 内部边,故记录下这对关系
        if (hasAIC && hasAIV) {
            APASS_LOG_WARN_F(
                Elements::Operation,
                "SCC %d contains both AIC and AIV clusters, "
                "potential cycle after FlattenSCC between the resulting AIC-group and AIV-group taskNodes.",
                sccId);
            // 标记：这个 SCC 包含的所有 cluster ID，后续在 CombineSCC 映射后
            // 会被映射到新的 taskNode ID。我们在 SplitGraph 结束后再进行映射。
            cycledSCCClusters_.push_back(sccResult[sccId]);
        }
    }

    if (cycledSCCClusters_.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "No AIC-AIV mixed cycles detected in SCC results.");
    } else {
        APASS_LOG_WARN_F(
            Elements::Operation,
            "Detected %zu SCC(s) with AIC-AIV mixed cycles, will record for post-schedule reorder.",
            cycledSCCClusters_.size());
    }
}
// mix子图切分主函数
void TaskSplitter::SplitGraph(const std::vector<Operation*>& opList)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start to split mix graph with op num %zu.", opList.size());
    opList_ = opList;
    BuildOpGraph();
    BuildSameLayerConnectionWithBack();
    std::vector<int> clusterIds;
    std::vector<ScheduleCoreType> clusterCoreTypes;
    int clusterNum = BuildCluster(clusterIds, clusterCoreTypes);
    APASS_LOG_INFO_F(Elements::Operation, "Find clusters finished.");
    std::vector<std::set<int>> inGraph;
    std::vector<std::set<int>> outGraph;
    BuildInOutGraph(inGraph, outGraph, clusterIds, clusterNum);
    std::vector<std::vector<int>> sccResult;
    StrongConnectionComponentFinder sccFinder;
    sccFinder.Find(inGraph, outGraph, sccResult);
    // 这两个新 Cluster 之间的"成环关系"
    RecordCycledClusters(clusterCoreTypes, sccResult);
    CombineSCC(clusterIds, clusterCoreTypes, inGraph, outGraph, sccResult);
    APASS_LOG_INFO_F(Elements::Operation, "Find strongly connected components finished.");
    opIdxToTaskId_.swap(clusterIds);
    inGraph_.swap(inGraph);
    outGraph_.swap(outGraph);
    taskCoreTypes_ = std::vector<ScheduleCoreType>(inGraph_.size(), ScheduleCoreType::AIV);
    taskIdToOps_.clear();
    taskIdToOps_.resize(inGraph_.size());
    for (size_t i = 0; i < opList_.size(); i++) {
        int currTaskId = opIdxToTaskId_[i];
        taskIdToOps_[currTaskId].push_back(i);
        if (opCoreTypes_[i] == ScheduleCoreType::AIC) {
            taskCoreTypes_[currTaskId] = ScheduleCoreType::AIC;
        }
    }
    taskGraph_ = BuildTaskGraph();
    ComputeTaskLevelBranches();
    APASS_LOG_INFO_F(Elements::Operation, "Build the task graph finished.");
}

// 将强连通分量展开，避免成环
inline int FlattenSCC(
    std::vector<ScheduleCoreType>& clusterCoreTypes, std::vector<std::vector<int>>& sccResult,
    std::unordered_map<int, int>& oldClusterIdToSCCId, std::unordered_map<int, std::vector<int>>& sccIdToNewClusters,
    std::unordered_map<int, int>& oldClusterToNewCluster)
{
    int currNewClusterIdx = 0;
    for (int sccId = 0; sccId < static_cast<int>(sccResult.size()); sccId++) {
        for (int clusterId : sccResult[sccId]) {
            oldClusterIdToSCCId[clusterId] = sccId;
        }
        if (sccResult[sccId].size() == 0) {
            continue;
        }
        if (sccResult[sccId].size() == 1) {
            sccIdToNewClusters[sccId].push_back(currNewClusterIdx);
            oldClusterToNewCluster[sccResult[sccId][0]] = currNewClusterIdx;
            currNewClusterIdx++;
            continue;
        }
        std::vector<int> AICclusters;
        std::vector<int> AIVclusters;
        for (int clusterId : sccResult[sccId]) {
            if (clusterCoreTypes[clusterId] == ScheduleCoreType::AIC) {
                AICclusters.push_back(clusterId);
            } else {
                AIVclusters.push_back(clusterId);
            }
        }
        if (AICclusters.size() > 0) {
            for (int aicIds : AICclusters) {
                oldClusterToNewCluster[aicIds] = currNewClusterIdx;
            }
            sccIdToNewClusters[sccId].push_back(currNewClusterIdx);
            currNewClusterIdx++;
        }
        if (AIVclusters.size() > 0) {
            for (int aivIds : AIVclusters) {
                oldClusterToNewCluster[aivIds] = currNewClusterIdx;
            }
            sccIdToNewClusters[sccId].push_back(currNewClusterIdx);
            currNewClusterIdx++;
        }
    }
    return currNewClusterIdx;
}

// 将 cycledSCCClusters_ 中记录的旧 Cluster ID 映射为新 TaskNode ID, 形成 cycledTaskNodePairs_
void TaskSplitter::RecordIDMap(
    std::unordered_map<int, int>& oldClusterToNewCluster, std::vector<ScheduleCoreType>& clusterCoreTypes)
{
    for (auto& oldClusters : cycledSCCClusters_) {
        std::set<int> aicNewIds;
        std::set<int> aivNewIds;
        for (int oldCid : oldClusters) {
            int newCid = oldClusterToNewCluster[oldCid];
            if (clusterCoreTypes[oldCid] == ScheduleCoreType::AIC) {
                aicNewIds.insert(newCid);
            } else {
                aivNewIds.insert(newCid);
            }
        }
        // 每个 AIC 新 ID 和 AIV 新 ID 之间都是成环对
        for (int aicId : aicNewIds) {
            for (int aivId : aivNewIds) {
                cycledTaskNodePairs_.push_back({aicId, aivId});
            }
        }
    }
}

// 将强连通分量展开，并构建新的连接图
void TaskSplitter::CombineSCC(
    std::vector<int>& clusterIds, std::vector<ScheduleCoreType>& clusterCoreTypes, std::vector<std::set<int>>& inGraph,
    std::vector<std::set<int>>& outGraph, std::vector<std::vector<int>>& sccResult)
{
    std::unordered_map<int, int> oldClusterIdToSCCId;
    std::unordered_map<int, std::vector<int>> sccIdToNewClusters;
    std::unordered_map<int, int> oldClusterToNewCluster;
    int newClusterNum =
        FlattenSCC(clusterCoreTypes, sccResult, oldClusterIdToSCCId, sccIdToNewClusters, oldClusterToNewCluster);
    APASS_LOG_INFO_F(
        Elements::Operation, "Cluster num after flatten strongly connected components is %d.", newClusterNum);
    RecordIDMap(oldClusterToNewCluster, clusterCoreTypes);
    std::set<std::pair<int, int>> sccConnection;
    for (size_t oldIdx = 0; oldIdx < inGraph.size(); oldIdx++) {
        int currSCC = oldClusterIdToSCCId[oldIdx];
        for (int prevIdx : inGraph[oldIdx]) {
            int prevSCC = oldClusterIdToSCCId[prevIdx];
            if (currSCC == prevSCC) {
                continue;
            }
            sccConnection.insert({prevSCC, currSCC});
        }
    }
    std::vector<std::set<int>> newInGraph(newClusterNum);
    std::vector<std::set<int>> newOutGraph(newClusterNum);
    for (auto pr : sccConnection) {
        for (int prevNewCluster : sccIdToNewClusters[pr.first]) {
            for (int currNewCluster : sccIdToNewClusters[pr.second]) {
                newInGraph[currNewCluster].insert(prevNewCluster);
                newOutGraph[prevNewCluster].insert(currNewCluster);
            }
        }
    }
    inGraph.swap(newInGraph);
    outGraph.swap(newOutGraph);
    for (size_t i = 0; i < clusterIds.size(); i++) {
        clusterIds[i] = oldClusterToNewCluster[clusterIds[i]];
    }
}

// 获得有向图中所有强连通分量
void StrongConnectionComponentFinder::Find(
    std::vector<std::set<int>>& inGraph, std::vector<std::set<int>>& outGraph, std::vector<std::vector<int>>& sccResult)
{
    sccResult.clear();
    index_ = 0;
    dfn_.clear();
    dfn_.resize(inGraph.size(), 0);
    low_.resize(inGraph.size());
    instack_.clear();
    instack_.resize(inGraph.size(), false);
    visited_.clear();
    stack_.clear();
    APASS_LOG_INFO_F(Elements::Operation, "Start finding strongly connected components using TarJan Algorithm.");
    for (int i = 0; i < static_cast<int>(inGraph.size()); i++) {
        if (dfn_[i] == 0) {
            TarJanAlg(i, outGraph, sccResult);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "TarJan Algorithm finished.");
}

// 递归使用TarJan算法获得强连通分量
void StrongConnectionComponentFinder::TarJanAlg(
    int idx, std::vector<std::set<int>>& outGraph, std::vector<std::vector<int>>& sccResult)
{
    index_++;
    dfn_[idx] = index_;
    low_[idx] = index_;
    stack_.push_back(idx);
    instack_[idx] = true;
    for (int nextIdx : outGraph[idx]) {
        if (dfn_[nextIdx] == 0) {
            TarJanAlg(nextIdx, outGraph, sccResult);
            low_[idx] = std::min(low_[idx], low_[nextIdx]);
        } else if (instack_[nextIdx]) {
            low_[idx] = std::min(low_[idx], dfn_[nextIdx]);
        }
    }
    if (dfn_[idx] == low_[idx]) {
        sccResult.push_back({});
        int currSCCidx = static_cast<int>(sccResult.size()) - 1;
        int stackTop = 0;
        do {
            stackTop = stack_.back();
            stack_.pop_back();
            instack_[stackTop] = false;
            sccResult[currSCCidx].push_back(stackTop);
        } while (stackTop != idx);
    }
}

// 获得taskNode的连接图
void TaskSplitter::BuildInOutGraph(
    std::vector<std::set<int>>& inGraph, std::vector<std::set<int>>& outGraph, std::vector<int>& clusterIds,
    int clusterNum)
{
    inGraph.clear();
    inGraph.resize(clusterNum);
    outGraph.clear();
    outGraph.resize(clusterNum);
    int opNum = static_cast<int>(opList_.size());
    for (int i = 0; i < opNum; i++) {
        int currTaskIdx = clusterIds[i];
        for (auto consumerOp : opList_[i]->ConsumerOpsOrdered()) {
            if (opMagicToIdx_.count(consumerOp->GetOpMagic()) == 0) {
                continue;
            }
            int nextOpIdx = opMagicToIdx_[consumerOp->GetOpMagic()];
            int nextTaskIdx = clusterIds[nextOpIdx];
            if (currTaskIdx == nextTaskIdx) {
                continue;
            }
            outGraph[currTaskIdx].insert(nextTaskIdx);
            inGraph[nextTaskIdx].insert(currTaskIdx);
        }
    }
}

// 建立TaskGraph
TaskGraph TaskSplitter::BuildTaskGraph()
{
    TaskGraph s = TaskGraph();
    for (int taskId = 0; taskId < static_cast<int>(taskIdToOps_.size()); taskId++) {
        s.AddTask(std::to_string(taskId), taskCoreTypes_[taskId], 0);
        for (auto opIdx : taskIdToOps_[taskId]) {
            s.tasks[taskId].opList_.push_back(opList_[opIdx]);
            s.tasks[taskId].latency += opList_[opIdx]->GetLatency();
        }
    }
    for (int taskId = 0; taskId < static_cast<int>(outGraph_.size()); taskId++) {
        for (auto nextTaskId : outGraph_[taskId]) {
            s.AddDependency(taskId, nextTaskId);
        }
    }
    return s;
}

// 判断op的ioperand为AIC类型且ooperand为AIV类型
inline bool IsFromAICToAIV(Operation* op)
{
    const std::unordered_set<MemoryType> AICmem{
        MemoryType::MEM_L0C, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B};
    const std::unordered_set<MemoryType> AIVmem{MemoryType::MEM_UB};
    for (auto iop : op->GetIOperands()) {
        if (AICmem.count(iop->GetMemoryTypeToBe()) == 0) {
            return false;
        }
    }
    for (auto oop : op->GetOOperands()) {
        if (AIVmem.count(oop->GetMemoryTypeToBe()) == 0) {
            return false;
        }
    }
    APASS_LOG_DEBUG_F(
        Elements::Operation, "op %s[%d] is from AIC to AIV.", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return true;
}

// 判断op的ioperand为AIV类型且ooperand为AIC类型
inline bool IsFromAIVToAIC(Operation* op)
{
    const std::unordered_set<MemoryType> AICmem{
        MemoryType::MEM_L0C, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B};
    const std::unordered_set<MemoryType> AIVmem{MemoryType::MEM_UB};
    for (auto iop : op->GetIOperands()) {
        if (AIVmem.count(iop->GetMemoryTypeToBe()) == 0) {
            return false;
        }
    }
    for (auto oop : op->GetOOperands()) {
        if (AICmem.count(oop->GetMemoryTypeToBe()) == 0) {
            return false;
        }
    }
    APASS_LOG_DEBUG_F(
        Elements::Operation, "op %s[%d] is from AIV to AIC.", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return true;
}

// 反向DFS查找产出指定MemoryType tensor的前驱op
void TaskSplitter::ReverseDFSFindByOutputMemType(
    int opIdx, MemoryType targetMemType, std::vector<int>& result, std::vector<bool>& visited)
{
    if (visited[opIdx]) {
        return;
    }
    visited[opIdx] = true;
    for (auto& oop : opList_[opIdx]->GetOOperands()) {
        if (oop->GetMemoryTypeToBe() == targetMemType) {
            result.push_back(opIdx);
            return;
        }
    }
    for (auto& iop : opList_[opIdx]->GetIOperands()) {
        for (auto& producerOp : iop->GetProducers()) {
            if (opMagicToIdx_.count(producerOp->GetOpMagic()) == 0) {
                continue;
            }
            int producerIdx = opMagicToIdx_[producerOp->GetOpMagic()];
            ReverseDFSFindByOutputMemType(producerIdx, targetMemType, result, visited);
        }
    }
}

// 判断 op 的后接 tensor 为 L1 且存在多个消费者
inline bool TaskSplitter::IsL1MultiConsumerSkip(size_t idx) const
{
    if (opList_[idx]->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
        opOutGraph_[idx].size() > 1) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Skip union op: %s[%d]", opList_[idx]->GetOpcodeStr().c_str(),
            opList_[idx]->GetOpMagic());
        return true;
    }
    return false;
}

// 判断 op 是否为 L1_TO_L0 操作
inline bool TaskSplitter::IsL1ToL0Op(int opIdx) const
{
    return opList_[opIdx]->GetOpcodeStr().find("L1_TO_L0") != std::string::npos;
}

// Union 同核操作：仅 union cube(AIC) op，vector op 不参与切图合并
void TaskSplitter::UnionSameCoreOps(DSUWithOrder& dsu)
{
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        if (opCoreTypes_[idx] != ScheduleCoreType::AIC) {
            continue;
        }
        bool skip = IsL1MultiConsumerSkip(idx);
        for (int nextOpIdx : opOutGraph_[idx]) {
            if (opCoreTypes_[nextOpIdx] != ScheduleCoreType::AIC) {
                continue;
            }
            if (!skip || !IsL1ToL0Op(nextOpIdx)) {
                dsu.Union(idx, nextOpIdx);
            }
        }
    }
}

// 构建 vector op 的连通分支信息（用于 ScheduleOneTask 中约束同一分支必须在同一 AIV 核上）
void TaskSplitter::BuildVecConnectedComponents(DSUWithOrder& vecDsu)
{
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        if (opCoreTypes_[idx] != ScheduleCoreType::AIV) {
            continue;
        }
        for (int nextOpIdx : opOutGraph_[idx]) {
            if (opCoreTypes_[nextOpIdx] == ScheduleCoreType::AIV) {
                vecDsu.Union(idx, nextOpIdx);
            }
        }
    }
    vecBranchId_.resize(opList_.size(), -1);
    int branchIdx = 0;
    std::unordered_map<int, int> rootToBranch;
    for (size_t idx = 0; idx < opList_.size(); idx++) {
        if (opCoreTypes_[idx] != ScheduleCoreType::AIV) {
            continue;
        }
        int root = vecDsu.Find(idx);
        if (rootToBranch.count(root) == 0) {
            rootToBranch[root] = branchIdx++;
        }
        vecBranchId_[idx] = rootToBranch[root];
    }
    APASS_LOG_INFO_F(Elements::Operation, "Built %d vector connected components.", branchIdx);
}

// Union 同层连接：仅 union 同核类型的 alloc 关联
void TaskSplitter::UnionSameLayerConnections(DSUWithOrder& dsu)
{
    for (auto pr : sameLayerConnection_) {
        if (opCoreTypes_[pr.first] == opCoreTypes_[pr.second]) {
            dsu.Union(pr.first, pr.second);
        }
    }
}

// Union Combine 操作：参考 SuperNodeGraphBuilder 的 Assemble/CopyIn/CopyOut 合并逻辑
// 注意：仅在两侧 op 的 core type 相同时才合并，避免 cube 与 vec cluster 误合并
void TaskSplitter::UnionCombineOps(DSUWithOrder& dsu)
{
    auto sameCore = [this](size_t a, int b) { return opCoreTypes_[a] == opCoreTypes_[b]; };
    for (size_t i = 0; i < opList_.size(); i++) {
        Operation* op = opList_[i];
        Opcode opcode = op->GetOpcode();
        OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(opcode);
        // AssembleCombine: assemble 和其输入绑定
        if (opcode == Opcode::OP_ASSEMBLE && !opInGraph_[i].empty()) {
            int inNode = *opInGraph_[i].begin();
            if (sameCore(i, inNode)) {
                dsu.Union(i, inNode);
            }
            continue;
        }

        // CopyOutCombine: 所有 UB MOVE_OUT 操作与其输入绑定
        if (opCoreTypes_[i] != ScheduleCoreType::AIV) {
            continue;
        }
        if (calcType == OpCalcType::MOVE_OUT || opcode == Opcode::OP_UB_COPY_ND2NZ || opcode == Opcode::OP_UB_COPY_L1) {
            for (int inNode : opInGraph_[i]) {
                if (sameCore(i, inNode)) {
                    dsu.Union(inNode, i);
                }
            }
            continue;
        }
        // CopyInCombine: 所有 UB MOVE_IN/MOVE_LOCAL 操作与其第一个输出绑定
        if ((calcType == OpCalcType::MOVE_IN || calcType == OpCalcType::MOVE_LOCAL) && !opOutGraph_[i].empty()) {
            int outNode = *opOutGraph_[i].begin();
            if (sameCore(i, outNode)) {
                dsu.Union(i, outNode);
            }
        }
    }
}

// 获取 DSU 中的 cluster ID 映射和 cluster 的核类型
int TaskSplitter::CollectClusters(
    DSUWithOrder& dsu, std::unordered_map<int, int>& rootToCluster, std::vector<ScheduleCoreType>& coreTypes) const
{
    int currIdx = 0;
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        int rootId = dsu.Find(idx);
        if (rootToCluster.count(rootId) == 0) {
            rootToCluster[rootId] = currIdx;
            coreTypes.push_back(opCoreTypes_[idx]);
            currIdx++;
        }
    }
    return currIdx;
}

// 获取每个 op 所在的临时 cluster ID
void TaskSplitter::GetOpClusterIds(
    DSUWithOrder& dsu, const std::unordered_map<int, int>& rootToCluster, std::vector<int>& opCluster) const
{
    opCluster.resize(opOutGraph_.size());
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        opCluster[idx] = rootToCluster.at(dsu.Find(idx));
    }
}

// 按 vec cluster 依赖的 cube cluster 集合分组，组内按连通性合并
void TaskSplitter::UnionVecClustersByDep(DSUWithOrder& dsu)
{
    std::unordered_map<int, int> rootToCluster;
    std::vector<ScheduleCoreType> tmpCoreTypes;
    // 收集连通分量
    int clusterNum = CollectClusters(dsu, rootToCluster, tmpCoreTypes);
    std::vector<int> opClusterVec;
    GetOpClusterIds(dsu, rootToCluster, opClusterVec);
    // 构建 cluster 间依赖图并计算传递闭包
    std::vector<std::set<int>> clusterIn(clusterNum);
    BuildClusterDepGraph(opClusterVec, clusterNum, clusterIn);

    std::vector<std::set<int>> clusterOut(clusterNum);
    BuildClusterInOutGraph(clusterIn, clusterNum, clusterOut);

    std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>> vecGroups;
    GroupAIVClustersByDep(clusterIn, clusterOut, tmpCoreTypes, clusterNum, vecGroups);

    for (auto& [key, vecClusters] : vecGroups) {
        (void)key;
        MergeVecClusterGroup(dsu, opClusterVec, vecClusters);
    }
}

// 构建 cluster 之间的依赖图（入边），并计算传递闭包
void TaskSplitter::BuildClusterDepGraph(
    const std::vector<int>& opCluster, int clusterNum, std::vector<std::set<int>>& clusterIn) const
{
    // Step 1: 构建直接依赖图
    for (size_t i = 0; i < opOutGraph_.size(); i++) {
        int src = opCluster[i];
        for (int next : opOutGraph_[i]) {
            int dst = opCluster[next];
            if (src != dst) {
                clusterIn[dst].insert(src);
            }
        }
    }

    // Step 2: 计算传递闭包（Floyd-Warshall 算法）
    // reach[i][j] = true 表示 cluster i 可达 cluster j
    std::vector<std::vector<bool>> reach(clusterNum, std::vector<bool>(clusterNum, false));
    for (int i = 0; i < clusterNum; i++) {
        reach[i][i] = true; // 自反性
        for (int j : clusterIn[i]) {
            reach[j][i] = true; // j -> i 有直接边
        }
    }

    // Floyd-Warshall 传递闭包
    for (int k = 0; k < clusterNum; k++) {
        for (int i = 0; i < clusterNum; i++) {
            for (int j = 0; j < clusterNum; j++) {
                if (reach[i][k] && reach[k][j]) {
                    reach[i][j] = true;
                }
            }
        }
    }

    // Step 3: 根据传递闭包重建 clusterIn（包含间接依赖）
    std::vector<std::set<int>> clusterInTransitive(clusterNum);
    for (int i = 0; i < clusterNum; i++) {
        for (int j = 0; j < clusterNum; j++) {
            if (i != j && reach[j][i]) {
                clusterInTransitive[i].insert(j);
            }
        }
    }
    clusterIn.swap(clusterInTransitive);
}

void TaskSplitter::BuildClusterInOutGraph(
    const std::vector<std::set<int>>& clusterIn, int clusterNum, std::vector<std::set<int>>& clusterOut) const
{
    for (int i = 0; i < clusterNum; i++) {
        for (int j : clusterIn[i]) {
            clusterOut[j].insert(i);
        }
    }
}

void TaskSplitter::GroupAIVClustersByDep(
    const std::vector<std::set<int>>& clusterIn, const std::vector<std::set<int>>& clusterOut,
    const std::vector<ScheduleCoreType>& tmpCoreTypes, int clusterNum,
    std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>& vecGroups) const
{
    for (int c = 0; c < clusterNum; c++) {
        if (tmpCoreTypes[c] != ScheduleCoreType::AIV) {
            continue;
        }

        std::set<int> aicInDeps, aicOutDeps;
        for (int dep : clusterIn[c]) {
            if (tmpCoreTypes[dep] == ScheduleCoreType::AIC) {
                aicInDeps.insert(dep);
            }
        }
        for (int succ : clusterOut[c]) {
            if (tmpCoreTypes[succ] == ScheduleCoreType::AIC) {
                aicOutDeps.insert(succ);
            }
        }

        auto key = std::make_pair(aicInDeps, aicOutDeps);
        vecGroups[key].push_back(c);
    }
}

void TaskSplitter::MergeVecClusterGroup(
    DSUWithOrder& dsu, const std::vector<int>& opCluster, const std::vector<int>& vecClusters)
{
    if (vecClusters.size() <= 1) {
        return;
    }
    std::unordered_set<int> clusterSet(vecClusters.begin(), vecClusters.end());
    for (size_t i = 0; i < opOutGraph_.size(); i++) {
        if (clusterSet.count(opCluster[i]) == 0) {
            continue;
        }
        for (int next : opOutGraph_[i]) {
            if (clusterSet.count(opCluster[next]) > 0 && opCluster[i] != opCluster[next]) {
                dsu.Union(static_cast<int>(i), static_cast<int>(next));
            }
        }
    }
}

void TaskSplitter::GroupAICClustersByDep(
    const std::vector<std::set<int>>& clusterIn, const std::vector<std::set<int>>& clusterOut,
    const std::vector<ScheduleCoreType>& tmpCoreTypes, int clusterNum,
    std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>& cubeGroups) const
{
    for (int c = 0; c < clusterNum; c++) {
        if (tmpCoreTypes[c] != ScheduleCoreType::AIC) {
            continue;
        }

        std::set<int> aivInDeps, aivOutDeps;
        for (int dep : clusterIn[c]) {
            if (tmpCoreTypes[dep] == ScheduleCoreType::AIV) {
                aivInDeps.insert(dep);
            }
        }
        for (int succ : clusterOut[c]) {
            if (tmpCoreTypes[succ] == ScheduleCoreType::AIV) {
                aivOutDeps.insert(succ);
            }
        }

        if (aivInDeps.empty() && aivOutDeps.empty()) {
            continue;
        }

        auto key = std::make_pair(aivInDeps, aivOutDeps);
        cubeGroups[key].push_back(c);
    }
}

void TaskSplitter::MergeCubeClusterGroup(
    DSUWithOrder& dsu, const std::vector<int>& opCluster, const std::vector<int>& cubeClusters)
{
    if (cubeClusters.size() <= 1) {
        return;
    }

    int baseCluster = cubeClusters[0];
    for (size_t i = 1; i < cubeClusters.size(); i++) {
        int otherCluster = cubeClusters[i];
        int baseRep = -1, otherRep = -1;
        for (size_t idx = 0; idx < opCluster.size(); idx++) {
            if (opCluster[idx] == baseCluster && baseRep < 0) {
                baseRep = static_cast<int>(idx);
            }
            if (opCluster[idx] == otherCluster && otherRep < 0) {
                otherRep = static_cast<int>(idx);
            }
            if (baseRep >= 0 && otherRep >= 0) {
                break;
            }
        }
        if (baseRep >= 0 && otherRep >= 0) {
            dsu.Union(otherRep, baseRep);
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Merge cube cluster %d into cluster %d (same bidirectional AIV dependency pattern).", otherCluster,
                baseCluster);
        }
    }
}

// Union AIC->AIV 跨核操作（保留，用于非重构路径兼容）
void TaskSplitter::UnionCrossCoreAICToAIV(DSUWithOrder& dsu)
{
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        if (IsFromAICToAIV(opList_[idx])) {
            for (int nextOpIdx : opOutGraph_[idx]) {
                dsu.Union(nextOpIdx, *opOutGraph_[idx].begin());
            }
        }
    }
}

// Union L0C 输入到 L1_COPY_IN
void TaskSplitter::UnionL0CToL1CopyIn(DSUWithOrder& dsu)
{
    // 对输入tensor为L0C的非alloc op，反向DFS找L1_COPY_IN，未与L1_TO_L0 union的则union到当前L0C集合
    for (size_t idx = 0; idx < opList_.size(); idx++) {
        if (opList_[idx]->GetIOperands().size() == 0 ||
            opList_[idx]->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
            continue;
        }
        std::vector<int> l1CopyInOps;
        std::vector<bool> visited(opList_.size(), false);
        ReverseDFSFindByOutputMemType(idx, MemoryType::MEM_L1, l1CopyInOps, visited);
        for (auto& l1CopyInOpIdx : l1CopyInOps) {
            bool alreadyUnionedWithL1ToL0 = false;
            for (auto& consumer : opList_[l1CopyInOpIdx]->ConsumerOpsOrdered()) {
                if (consumer->GetOpcodeStr().find("L1_TO_L0") == std::string::npos) {
                    continue;
                }
                if (opMagicToIdx_.count(consumer->GetOpMagic()) == 0) {
                    continue;
                }
                int consumerIdx = opMagicToIdx_[consumer->GetOpMagic()];
                if (dsu.Find(l1CopyInOpIdx) == dsu.Find(consumerIdx)) {
                    alreadyUnionedWithL1ToL0 = true;
                    break;
                }
            }
            if (!alreadyUnionedWithL1ToL0 && opCoreTypes_[idx] == opCoreTypes_[l1CopyInOpIdx]) {
                dsu.Union(l1CopyInOpIdx, idx);
            }
        }
    }
}

// 碎子图合并：拓扑排序并估计 cycle，将小 vec cluster 合并到相邻较小 cluster
void TaskSplitter::MergeSmallVecClusters(DSUWithOrder& dsu)
{
    std::unordered_map<int, int> rootToCluster;
    std::vector<ScheduleCoreType> tmpCoreTypes;
    int clusterNum = CollectClusters(dsu, rootToCluster, tmpCoreTypes);
    std::vector<int> opCluster;
    GetOpClusterIds(dsu, rootToCluster, opCluster);
    // 构建 cluster 间有向图
    std::vector<std::set<int>> clIn(clusterNum), clOut(clusterNum);
    BuildClusterGraph(opCluster, clusterNum, clIn, clOut);
    // 估算每个 cluster 的 cycle
    std::vector<int> clusterCycle(clusterNum, 0);
    EstimateClusterCycles(opCluster, clusterNum, clusterCycle);
    // 拓扑排序
    std::vector<int> topoOrder = TopoSortClusters(clusterNum, clIn, clOut);
    // 迭代合并
    RunSmallClusterMerge(dsu, topoOrder, tmpCoreTypes, clusterCycle, clIn, clOut, clusterNum);
}

// 按 AIC cluster 对 AIV cluster 的双向依赖关系分组，合并同组 AIC cluster
// 与 UnionVecClustersByDep 类似，但：
// 1. 针对 AIC cluster（而非 vec cluster）
// 2. 使用双向依赖匹配（incoming + outgoing）
// 3. 依赖关系基于传递闭包（而非直接依赖）
// 4. 不要求组内连通性，只要双向依赖集合相同即可合并
void TaskSplitter::UnionCubeClustersByDep(DSUWithOrder& dsu)
{
    std::unordered_map<int, int> rootToCluster;
    std::vector<ScheduleCoreType> tmpCoreTypes;
    int clusterNum = CollectClusters(dsu, rootToCluster, tmpCoreTypes);
    std::vector<int> opCluster;
    GetOpClusterIds(dsu, rootToCluster, opCluster);
    std::vector<std::set<int>> clusterIn(clusterNum);
    BuildClusterDepGraph(opCluster, clusterNum, clusterIn);

    std::vector<std::set<int>> clusterOut(clusterNum);
    BuildClusterInOutGraph(clusterIn, clusterNum, clusterOut);

    std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>> cubeGroups;
    GroupAICClustersByDep(clusterIn, clusterOut, tmpCoreTypes, clusterNum, cubeGroups);

    for (auto& [key, cubeClusters] : cubeGroups) {
        (void)key;
        MergeCubeClusterGroup(dsu, opCluster, cubeClusters);
    }
}

// 构建 cluster 间完整有向图
void TaskSplitter::BuildClusterGraph(
    const std::vector<int>& opCluster, int /* clusterNum */, std::vector<std::set<int>>& clIn,
    std::vector<std::set<int>>& clOut) const
{
    for (size_t i = 0; i < opOutGraph_.size(); i++) {
        int src = opCluster[i];
        for (int next : opOutGraph_[i]) {
            int dst = opCluster[next];
            if (src != dst) {
                clOut[src].insert(dst);
                clIn[dst].insert(src);
            }
        }
    }
}

// 估算每个 cluster 的总 cycle
void TaskSplitter::EstimateClusterCycles(
    const std::vector<int>& opCluster, int /* clusterNum */, std::vector<int>& clusterCycle) const
{
    for (size_t i = 0; i < opList_.size(); i++) {
        clusterCycle[opCluster[i]] += opList_[i]->GetLatency();
    }
}

// cluster 拓扑排序（Kahn 算法）
std::vector<int> TaskSplitter::TopoSortClusters(
    int clusterNum, const std::vector<std::set<int>>& clIn, const std::vector<std::set<int>>& clOut) const
{
    std::vector<int> inDeg(clusterNum, 0);
    for (int c = 0; c < clusterNum; c++) {
        inDeg[c] = static_cast<int>(clIn[c].size());
    }
    std::queue<int> q;
    for (int c = 0; c < clusterNum; c++) {
        if (inDeg[c] == 0) {
            q.push(c);
        }
    }
    std::vector<int> order;
    order.reserve(clusterNum);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : clOut[u]) {
            if (--inDeg[v] == 0) {
                q.push(v);
            }
        }
    }
    return order;
}

// 找到小 vec cluster 的最佳合并目标（in 中编号最大或 out 中编号最小中 cycle 较小者）
// 如果候选目标都是 cube cluster，则不合并
int TaskSplitter::FindMergeTarget(
    int clusterId, const std::vector<ScheduleCoreType>& coreTypes, const std::vector<int>& cycle,
    const std::vector<std::set<int>>& clIn, const std::vector<std::set<int>>& clOut) const
{
    int targetIn = -1;
    int targetOut = -1;
    // in 中编号最大的 cluster
    for (int dep : clIn[clusterId]) {
        if (targetIn < 0 || dep > targetIn) {
            targetIn = dep;
        }
    }
    // out 中编号最小的 cluster
    for (int succ : clOut[clusterId]) {
        if (targetOut < 0 || succ < targetOut) {
            targetOut = succ;
        }
    }
    // 两侧候选都是 cube，不适合合并
    bool inIsCube = (targetIn >= 0 && coreTypes[targetIn] == ScheduleCoreType::AIC);
    bool outIsCube = (targetOut >= 0 && coreTypes[targetOut] == ScheduleCoreType::AIC);
    if (inIsCube && outIsCube) {
        return -1;
    }
    if (targetIn < 0 && targetOut < 0) {
        return -1;
    }
    // 仅一侧有候选时选非 cube 侧；若该侧是 cube 则无可合并
    if (targetIn < 0) {
        return outIsCube ? -1 : targetOut;
    }
    if (targetOut < 0) {
        return inIsCube ? -1 : targetIn;
    }
    // 优先选非 cube 侧
    if (inIsCube) {
        return targetOut;
    }
    if (outIsCube) {
        return targetIn;
    }
    return (cycle[targetIn] <= cycle[targetOut]) ? targetIn : targetOut;
}

// 合并两个 cluster：将 src 中的所有 op union 到 dst 的代表 op
void TaskSplitter::UnionTwoClusters(DSUWithOrder& dsu, const std::vector<int>& opCluster, int srcCluster, int dstCluster)
{
    int dstRep = -1;
    for (size_t i = 0; i < opCluster.size(); i++) {
        if (opCluster[i] == dstCluster) {
            dstRep = static_cast<int>(i);
            break;
        }
    }
    if (dstRep < 0) {
        return;
    }
    for (size_t i = 0; i < opCluster.size(); i++) {
        if (static_cast<int>(opCluster[i]) == srcCluster) {
            dsu.Union(i, dstRep);
        }
    }
}

// 更新 cluster 图：合并 src 到 dst 后更新边和 cycle
void TaskSplitter::UpdateClusterGraphAfterMerge(
    int src, int dst, std::vector<int>& cycle, std::vector<std::set<int>>& clIn,
    std::vector<std::set<int>>& clOut) const
{
    cycle[dst] += cycle[src];
    // 将 src 的 in/out 边转移给 dst
    for (int dep : clIn[src]) {
        if (dep != dst) {
            clOut[dep].erase(src);
            clOut[dep].insert(dst);
            clIn[dst].insert(dep);
        }
    }
    for (int succ : clOut[src]) {
        if (succ != dst) {
            clIn[succ].erase(src);
            clIn[succ].insert(dst);
            clOut[dst].insert(succ);
        }
    }
    // 清除 dst<->src 之间的边
    clIn[dst].erase(src);
    clOut[dst].erase(src);
    // 清空 src
    clIn[src].clear();
    clOut[src].clear();
    cycle[src] = 0;
}

// 执行碎子图迭代合并
void TaskSplitter::RunSmallClusterMerge(
    DSUWithOrder& dsu, const std::vector<int>& topoOrder, std::vector<ScheduleCoreType>& coreTypes,
    std::vector<int>& cycle, std::vector<std::set<int>>& clIn, std::vector<std::set<int>>& clOut, int clusterNum)
{
    std::vector<bool> removed(clusterNum, false);
    for (int iter = 0; iter < OOO_SMALL_CLUSTER_MERGE_MAX_ITER; iter++) {
        bool merged = false;
        // 需要重新获取 opCluster 因为 dsu 可能变化
        std::unordered_map<int, int> rootToCluster2;
        std::vector<ScheduleCoreType> tmpTypes2;
        CollectClusters(dsu, rootToCluster2, tmpTypes2);
        std::vector<int> opCluster2;
        GetOpClusterIds(dsu, rootToCluster2, opCluster2);
        for (int c : topoOrder) {
            if (removed[c] || coreTypes[c] != ScheduleCoreType::AIV) {
                continue;
            }
            if (cycle[c] >= OOO_CYCLE_LB) {
                continue;
            }
            int target = FindMergeTarget(c, coreTypes, cycle, clIn, clOut);
            if (target < 0) {
                continue;
            }
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Merge small vec cluster %d (cycle=%d) into cluster %d (cycle=%d).", c, cycle[c],
                target, cycle[target]);
            UnionTwoClusters(dsu, opCluster2, c, target);
            UpdateClusterGraphAfterMerge(c, target, cycle, clIn, clOut);
            removed[c] = true;
            merged = true;
        }
        if (!merged) {
            break;
        }
    }
}

// 根据op的CoreType构建连通集（重构后的主流程）
int TaskSplitter::BuildCluster(std::vector<int>& clusterIds, std::vector<ScheduleCoreType>& clusterCoreTypes)
{
    DSUWithOrder dsu(opList_.size());
    // Step 0: 构建 vector op 连通分支信息（仅用于 ScheduleOneTask 核选择约束，不影响切图）
    DSUWithOrder vecDsu(opList_.size());
    BuildVecConnectedComponents(vecDsu);
    // Step 1: 仅 union cube op
    UnionSameCoreOps(dsu);
    // Step 2: union 同层 alloc 连接
    UnionSameLayerConnections(dsu);
    // Step 3: Assemble/CopyIn/CopyOut 合并（同核类型）
    UnionCombineOps(dsu);
    // Step 4: vec cluster 按依赖的 cube cluster 分组后连通性合并
    UnionVecClustersByDep(dsu);
    // Step 5: L0C->L1 COPY_IN 合并 + 碎子图合并
    UnionL0CToL1CopyIn(dsu);
    MergeSmallVecClusters(dsu);
    // Step 6: 按 AIC cluster 对 AIV cluster 的双向依赖关系分组并合并
    UnionCubeClustersByDep(dsu);

    // Step 7: 映射为顺序 cluster ID
    clusterIds.resize(opOutGraph_.size());
    clusterCoreTypes.clear();
    int currIdx = 0;
    std::unordered_map<int, int> rootIdToClusterId;
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        int rootId = dsu.Find(idx);
        if (rootIdToClusterId.count(rootId) == 0) {
            rootIdToClusterId[rootId] = currIdx;
            clusterCoreTypes.push_back(opCoreTypes_[idx]);
            currIdx++;
        }
        clusterIds[idx] = rootIdToClusterId[rootId];
    }
    return currIdx;
}

// 判断currTask与currOldTasks都无依赖关系
inline bool NoDepDetached(const std::vector<int>& currOldTasks, int currTaskId, DAGReachableJudger& judger)
{
    for (int oldTaskId : currOldTasks) {
        if (judger.IsReachable(oldTaskId, currTaskId)) {
            return false;
        }
    }
    return true;
}

// 根据taskNode在模拟泳道图中的位置和可达性，返回合并后的taskNode列表
std::vector<std::vector<int>> TaskSplitter::FindMergeableTaskNodes()
{
    std::vector<std::vector<int>> newTaskToOldTasks;
    std::unordered_map<TargetCoreType, std::vector<int>> targetTypeToTasks{
        {TargetCoreType::AIC, {}}, {TargetCoreType::AIV0, {}}, {TargetCoreType::AIV1, {}}};
    DAGReachableJudger reachableJudger;
    reachableJudger.Build(inGraph_, outGraph_);
    for (int i = 0; i < static_cast<int>(taskGraph_.tasks.size()); i++) {
        targetTypeToTasks[taskGraph_.tasks[i].targetCoreType].push_back(i);
    }
    for (auto& tasksPair : targetTypeToTasks) {
        std::sort(tasksPair.second.begin(), tasksPair.second.end(), [this](int i, int j) {
            return taskGraph_.tasks[i].startTime < taskGraph_.tasks[j].startTime;
        });
        std::vector<int> oldTasks;
        for (int currTaskId : tasksPair.second) {
            if (oldTasks.empty() || NoDepDetached(oldTasks, currTaskId, reachableJudger)) {
                oldTasks.push_back(currTaskId);
            } else {
                newTaskToOldTasks.push_back(oldTasks);
                oldTasks.clear();
                oldTasks.push_back(currTaskId);
            }
        }
        if (!oldTasks.empty()) {
            newTaskToOldTasks.push_back(oldTasks);
        }
    }
    return newTaskToOldTasks;
}

// 根据taskNode在模拟泳道图中的位置和可达性，创建合并后的taskGraph
void TaskSplitter::MergeTask()
{
    std::vector<std::vector<int>> newTaskToOldTasks = FindMergeableTaskNodes();
    std::vector<int> oldTaskToNewTask(taskGraph_.tasks.size());
    TaskGraph s;
    s.makespan = taskGraph_.makespan;
    for (size_t newTaskIdx = 0; newTaskIdx < newTaskToOldTasks.size(); newTaskIdx++) {
        int sampleOldTaskId = newTaskToOldTasks[newTaskIdx][0];
        int sampleOldTaskIdEnd = newTaskToOldTasks[newTaskIdx].back();
        int newTaskId = s.AddTask(std::to_string(newTaskIdx), taskGraph_.tasks[sampleOldTaskId].coreType, 0);
        s.tasks[newTaskId].targetCoreType = taskGraph_.tasks[sampleOldTaskId].targetCoreType;
        s.tasks[newTaskId].startTime = taskGraph_.tasks[sampleOldTaskId].startTime;
        s.tasks[newTaskId].endTime = taskGraph_.tasks[sampleOldTaskIdEnd].endTime;
        for (int oldTaskId : newTaskToOldTasks[newTaskIdx]) {
            oldTaskToNewTask[oldTaskId] = newTaskIdx;
            s.tasks[newTaskId].latency += taskGraph_.tasks[oldTaskId].latency;
            s.tasks[newTaskId].opList_.insert(
                s.tasks[newTaskId].opList_.end(), taskGraph_.tasks[oldTaskId].opList_.begin(),
                taskGraph_.tasks[oldTaskId].opList_.end());
        }
    }
    for (int oldTaskId = 0; oldTaskId < static_cast<int>(taskGraph_.tasks.size()); oldTaskId++) {
        int currNewTaskId = oldTaskToNewTask[oldTaskId];
        for (int nextOldTaskId : taskGraph_.tasks[oldTaskId].outTasks) {
            s.AddDependency(currNewTaskId, oldTaskToNewTask[nextOldTaskId]);
        }
    }
    taskGraph_ = s;
}

// 将属于同一个TargetCoreType的taskNode合并成一个taskNode
void TaskSplitter::MergeTaskByTargetCoreType()
{
    std::unordered_map<TargetCoreType, std::vector<int>> targetTypeToTasks{
        {TargetCoreType::AIC, {}}, {TargetCoreType::AIV0, {}}, {TargetCoreType::AIV1, {}}};
    std::unordered_map<TargetCoreType, ScheduleCoreType> targetTypeToScheduleType{
        {TargetCoreType::AIC, ScheduleCoreType::AIC},
        {TargetCoreType::AIV0, ScheduleCoreType::AIV},
        {TargetCoreType::AIV1, ScheduleCoreType::AIV}};
    for (int i = 0; i < static_cast<int>(taskGraph_.tasks.size()); i++) {
        targetTypeToTasks[taskGraph_.tasks[i].targetCoreType].push_back(i);
    }
    TaskGraph s;
    int newTaskIdx = 0;
    for (auto& tasksPair : targetTypeToTasks) {
        if (tasksPair.second.size() == 0) {
            continue;
        }
        std::sort(tasksPair.second.begin(), tasksPair.second.end(), [this](int i, int j) {
            return taskGraph_.tasks[i].startTime < taskGraph_.tasks[j].startTime;
        });
        int newTaskId = s.AddTask(std::to_string(newTaskIdx), targetTypeToScheduleType[tasksPair.first], 0);
        newTaskIdx++;
        s.tasks[newTaskId].targetCoreType = tasksPair.first;
        s.tasks[newTaskId].startTime = taskGraph_.tasks[tasksPair.second[0]].startTime;
        s.tasks[newTaskId].endTime = taskGraph_.tasks[tasksPair.second.back()].endTime;
        for (int oldTaskId : tasksPair.second) {
            s.tasks[newTaskId].latency += taskGraph_.tasks[oldTaskId].latency;
            s.tasks[newTaskId].opList_.insert(
                s.tasks[newTaskId].opList_.end(), taskGraph_.tasks[oldTaskId].opList_.begin(),
                taskGraph_.tasks[oldTaskId].opList_.end());
        }
    }
    taskGraph_ = s;
}

void TaskSplitter::ComputeTaskLevelBranches()
{
    int numTasks = static_cast<int>(taskGraph_.tasks.size());
    std::vector<int> parent(numTasks);
    for (int i = 0; i < numTasks; i++)
        parent[i] = i;
    std::function<int(int)> findRoot = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = findRoot(a);
        b = findRoot(b);
        if (a != b)
            parent[a] = b;
    };
    for (int i = 0; i < numTasks; i++) {
        if (taskCoreTypes_[i] != ScheduleCoreType::AIV)
            continue;
        for (int s : taskGraph_.tasks[i].outTasks) {
            if (taskCoreTypes_[s] == ScheduleCoreType::AIV) {
                unite(i, s);
            }
        }
    }
    int branchIdx = 0;
    std::unordered_map<int, int> rootToBranch;
    for (int i = 0; i < numTasks; i++) {
        if (taskCoreTypes_[i] != ScheduleCoreType::AIV)
            continue;
        int root = findRoot(i);
        if (rootToBranch.count(root) == 0) {
            rootToBranch[root] = branchIdx++;
        }
        taskGraph_.tasks[i].vecBranchId = rootToBranch[root];
    }
    APASS_LOG_INFO_F(Elements::Operation, "Built %d task-level vector branches.", branchIdx);
}

// 根据划分结果标记op的AIVCore与internalSubgraphID
void TaskSplitter::MarkInternalSubgraphID()
{
    std::unordered_map<TargetCoreType, AIVCore> targetMap{
        {TargetCoreType::AIC, AIVCore::UNSPECIFIED},
        {TargetCoreType::UNKNOWN, AIVCore::UNSPECIFIED},
        {TargetCoreType::AIV0, AIVCore::AIV0},
        {TargetCoreType::AIV1, AIVCore::AIV1}};
    std::unordered_map<TargetCoreType, int> subGraphIdMap{
        {TargetCoreType::AIC, NEGATIVE_ONE},
        {TargetCoreType::AIV0, NEGATIVE_ONE},
        {TargetCoreType::AIV1, NEGATIVE_ONE},
        {TargetCoreType::UNKNOWN, NEGATIVE_ONE}};
    int id = 0;
    for (auto& task : taskGraph_.tasks) {
        if (task.targetCoreType == TargetCoreType::UNKNOWN) {
            APASS_LOG_ERROR_F(Elements::Operation, "task %d coreType is unknow", task.idx);
        }
        AIVCore targetType = targetMap[task.targetCoreType];
        if (subGraphIdMap[task.targetCoreType] == NEGATIVE_ONE) {
            subGraphIdMap[task.targetCoreType] = id++;
        }
        for (auto opPtr : task.opList_) {
            opPtr->SetAIVCore(targetType);
        }
    }
    for (auto& task : taskGraph_.tasks) {
        auto subGraphId = subGraphIdMap[task.targetCoreType];
        for (auto opPtr : task.opList_) {
            opPtr->UpdateInternalSubgraphID(subGraphId);
        }
    }
}

// 将多个taskNode的opList在保持内部顺序的前提下合并成符合拓扑序的一个opList
std::vector<Operation*> TaskSplitter::GetMergedOperations()
{
    std::priority_queue<
        std::pair<int, Operation*>, std::vector<std::pair<int, Operation*>>, std::greater<std::pair<int, Operation*>>>
        pQueue;
    std::unordered_map<Operation*, int> opPriority;
    std::unordered_map<Operation*, int> inLinkNum;
    std::vector<Operation*> topoSeq;
    for (auto& task : taskGraph_.tasks) {
        for (size_t opIdx = 0; opIdx < task.opList_.size(); opIdx++) {
            Operation* opPtr = task.opList_[opIdx];
            opPriority[opPtr] = opIdx;
            inLinkNum[opPtr] = opPtr->ProducerOpsOrdered().size();
            if (inLinkNum[opPtr] == 0) {
                pQueue.push({opIdx, opPtr});
            }
        }
    }
    while (pQueue.size() > 0) {
        auto ele = pQueue.top();
        pQueue.pop();
        topoSeq.push_back(ele.second);
        for (auto& nextOpPtr : ele.second->ConsumerOpsOrdered()) {
            inLinkNum[nextOpPtr]--;
            if (inLinkNum[nextOpPtr] == 0) {
                pQueue.push({opPriority[nextOpPtr], nextOpPtr});
            }
        }
    }
    return topoSeq;
}

DSUWithOrder::DSUWithOrder(int num)
{
    parent.resize(num);
    for (int i = 0; i < num; i++) {
        parent[i] = i;
    }
}

int DSUWithOrder::Find(int i)
{
    if (parent[i] == i) {
        return i;
    }
    parent[i] = Find(parent[i]);
    return parent[i];
}

void DSUWithOrder::Union(int i, int j)
{
    int rootI = Find(i);
    int rootJ = Find(j);
    if (rootI == rootJ) {
        return;
    }
    if (rootI < rootJ) {
        parent[rootJ] = rootI;
    } else {
        parent[rootI] = rootJ;
    }
}

// 根据连接图计算传递闭包
void DAGReachableJudger::Build(const std::vector<std::set<int>>& inGraph, const std::vector<std::set<int>>& outGraph)
{
    int nodeNum = static_cast<int>(inGraph.size());
    APASS_LOG_DEBUG_F(Elements::Operation, "Build DAG reachable judger with node num %d.", nodeNum);
    const int bitPerBlock = 32;
    int blockNum = (nodeNum + bitPerBlock - 1) / bitPerBlock;
    reachableSet.resize(nodeNum);
    for (int i = 0; i < nodeNum; i++) {
        reachableSet[i].resize(blockNum, 0);
    }
    std::vector<bool> finishedTasks(inGraph.size(), false);
    std::vector<int> taskStack;
    for (size_t i = 0; i < inGraph.size(); i++) {
        taskStack.push_back(i);
    }
    while (taskStack.size() > 0) {
        int taskId = taskStack.back();
        taskStack.pop_back();
        if (finishedTasks[taskId]) {
            continue;
        }
        std::vector<int> notReadyNextTaskIds;
        for (int nextTaskId : outGraph[taskId]) {
            if (!finishedTasks[nextTaskId]) {
                notReadyNextTaskIds.push_back(nextTaskId);
            }
        }
        if (notReadyNextTaskIds.size() > 0) {
            taskStack.push_back(taskId);
            taskStack.insert(taskStack.end(), notReadyNextTaskIds.begin(), notReadyNextTaskIds.end());
            continue;
        }
        for (int nextTaskId : outGraph[taskId]) {
            SetReachable(taskId, nextTaskId);
            MergeReachable(taskId, nextTaskId);
        }
        finishedTasks[taskId] = true;
    }
}

// 设定从src到dst可达
void DAGReachableJudger::SetReachable(const int src, const int dst)
{
    const int bitPerBlock = 32;
    size_t index = dst / bitPerBlock;
    size_t offset = dst % bitPerBlock;
    if (reachableSet[src].size() < index + 1) {
        reachableSet[src].resize(index + 1, 0);
    }
    reachableSet[src][index] |= (1U << offset);
}

// 设定从src可以到达dst可达的所有节点
void DAGReachableJudger::MergeReachable(int src, int dst)
{
    if (reachableSet[src].size() < reachableSet[dst].size()) {
        reachableSet[src].resize(reachableSet[dst].size(), 0);
    }
    for (size_t i = 0; i < reachableSet[dst].size(); i++) {
        reachableSet[src][i] |= reachableSet[dst][i];
    }
}

// 判断有向无环图中是否存在从src到dst的路径
bool DAGReachableJudger::IsReachable(int src, int dst)
{
    const int bitPerBlock = 32;
    size_t index = dst / bitPerBlock;
    size_t offset = dst % bitPerBlock;
    return (reachableSet[src][index] & (1U << offset)) != 0;
}

} // namespace npu::tile_fwk
