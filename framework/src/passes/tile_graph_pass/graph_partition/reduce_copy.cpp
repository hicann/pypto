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
 * \file reduce_copy.cpp
 * \brief
 */

#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <sstream>

#define MODULE_NAME "ReduceCopy"

namespace npu::tile_fwk {

static std::string IntVecToString(const std::vector<int>& vec)
{
    std::ostringstream ss;
    ss << "[";
    for (size_t idx = 0; idx < vec.size(); idx++) {
        ss << vec[idx];
        if (idx != vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

static bool IsValidMergeGroup(const std::vector<int>& mergeGroup, const std::unordered_set<int>& noMergeSubgraph)
{
    for (int subidx : mergeGroup) {
        if (noMergeSubgraph.count(subidx) > 0) {
            return false;
        }
    }
    return true;
}

Status ReduceCopyMerge::RunOnFunction(Function& function)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip ReduceCopy Pass.");
        return SUCCESS;
    }
    size_t subgraphNumBefore = function.GetTotalSubGraphCount();
    MergeInput mergeInput;
    mergeInput.maxLatency = 1e7;
    mergeInput.aivRatio = {1e-6, 1e6};
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph Info before ReduceCopy Pass:");
    BuildGraph(function, mergeInput);
    MarkNoMergeSubgraph(function);
    BuildMergeGroup(function, mergeInput);
    CombineForkSubgraph(function, mergeInput);
    MixGraphMerger merger;
    bool enableAutoMix = (function.paramConfigs_.autoMixPartition != 0);
    APASS_LOG_INFO_F(Elements::Operation, "Enable auto CV mix partition: %s", enableAutoMix ? "True" : "False");
    merger.enableAutoMix = enableAutoMix;
    MergeOutput output = merger.Merge(mergeInput);
    function.SetTotalSubGraphCount(output.numSubgraphUpdated);
    for (auto& op : function.Operations()) {
        int src = op.GetSubgraphID();
        if (src >= static_cast<int>(output.subgraphIdUpdated.size())) {
            APASS_LOG_ERROR_F(Elements::Operation, "Current op subgraphID not in ReduceCopy subgraph update record.");
            return FAILED;
        }
        if (op.HasAttr(OpAttributeKey::isCube) && (op.GetBoolAttribute(OpAttributeKey::isCube) == false)) {
            op.SetAttr(OpAttributeKey::reduceCopyPreSubgraphId, static_cast<int64_t>(src));
        }
        op.UpdateSubgraphID(output.subgraphIdUpdated[src]);
    }
    MergeInput mergeInputTmp;
    APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph Info after ReduceCopy Pass:");
    BuildGraph(function, mergeInputTmp);
    if (enableAutoMix && function.GetTotalSubGraphCount() == subgraphNumBefore) {
        APASS_LOG_WARN_F(Elements::Operation, 
            "CV mix merging was not performed, since fusion would degrade performance or may cause loop on this computation graph.");
    }
    return SUCCESS;
}

void ReduceCopyMerge::CombineForkSubgraph(Function& function, MergeInput& mergeInput)
{
    std::unordered_map<int, std::unordered_set<int>> forkSubgraphsMap;
    for (auto& op : function.Operations()) {
        std::unordered_set<int> forkSubgraphIds;
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            for (auto& prod : op.ProducerOps()) {
                forkSubgraphIds.insert(prod->GetSubgraphID());
            }
        }
        if (forkSubgraphIds.size() <= 1) {
            continue;
        }
        for (auto& idx : forkSubgraphIds) {
            for (auto& idy : forkSubgraphIds) {
                if (idx != idy) {
                    forkSubgraphsMap[idx].insert(idy);
                }
            }
        }
    }
    for (int i = 0; i < static_cast<int>(mergeInput.mergeGroup.size()); i++) {
        size_t originalSize = mergeInput.mergeGroup[i].size();
        std::set<int> combinedGroup(mergeInput.mergeGroup[i].begin(), mergeInput.mergeGroup[i].end());
        for (auto& subgraphId : mergeInput.mergeGroup[i]) {
            if (forkSubgraphsMap.count(subgraphId) > 0) {
                combinedGroup.insert(forkSubgraphsMap[subgraphId].begin(), forkSubgraphsMap[subgraphId].end());
            }
        }
        if (combinedGroup.size() != originalSize) {
            mergeInput.mergeGroup[i].clear();
            mergeInput.mergeGroup[i].insert(mergeInput.mergeGroup[i].end(), combinedGroup.begin(), combinedGroup.end());
            mergeInput.isValidMergeGroup[i] = IsValidMergeGroup(mergeInput.mergeGroup[i], noMergeSubgraph);
            APASS_LOG_DEBUG_F(Elements::Operation, "merge group %d after combine fork: %s, isEnforce: %s, isValid: %s.",
                i, IntVecToString(mergeInput.mergeGroup[i]).c_str(),
                mergeInput.isEnforceMergeGroup[i] ? "True" : "False",
                mergeInput.isValidMergeGroup[i] ? "True" : "False");
        }
    }
}

static void MarkCrossSubgraph(Function& function, std::unordered_set<int>& noMergeSubgraph)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        int src = op.GetSubgraphID();
        bool hasSameConsumerId = false;
        bool hasOtherConsumerId = false;
        for (auto& consumerOp : op.ConsumerOps()) {
            if (consumerOp->GetSubgraphID() == src) {
                hasSameConsumerId = true;
            } else {
                hasOtherConsumerId = true;
            }
        }
        if (hasSameConsumerId && hasOtherConsumerId) {
            noMergeSubgraph.insert(src);
            APASS_LOG_DEBUG_F(Elements::Operation,
                "Subgraph %d is not mergeable because it has inner tensor used by other subgraph.", src);
        }
        bool hasSameProducerId = false;
        bool hasOtherProducerId = false;
        for (auto& producerOp : op.ProducerOps()) {
            if (producerOp->GetSubgraphID() == src) {
                hasSameProducerId = true;
            } else {
                hasOtherProducerId = true;
            }
        }
        if (hasSameProducerId && hasOtherProducerId) {
            noMergeSubgraph.insert(src);
            APASS_LOG_DEBUG_F(Elements::Operation,
                "Subgraph %d is not mergeable because it has inner tensor used by other subgraph.", src);
        }
    }
}

Status ReduceCopyMerge::MarkNoMergeSubgraph(Function& function)
{
    noMergeSubgraph.clear();
    noMergeSubgraphEnforce.clear();
    int subgraphNum = function.GetTotalSubGraphCount();
    std::vector<int> subgraphOpNum(subgraphNum, 0);
    std::vector<bool> subgraphHasReshape(subgraphNum, false);
    std::vector<bool> subgraphHasInnerDDR(subgraphNum, false);
    for (auto& op : function.Operations()) {
        int src = op.GetSubgraphID();
        // noMergeSubgraphEnforce
        if (OpcodeManager::Inst().GetCoreType(op.GetOpcode()) == OpCoreType::AICPU) {
            noMergeSubgraphEnforce.insert(src);
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d is not mergeable because it has AICPU op.", src);
        }
        // noMergeSubgraph
        subgraphOpNum[src] += 1;
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            subgraphHasReshape[src] = true;
        }
        for (auto& operand : op.GetOOperands()) {
            if (operand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            for (auto& consumer : operand->GetConsumers()) {
                if (consumer->GetSubgraphID() == src) {
                    subgraphHasInnerDDR[src] = true;
                    break;
                }
            }
        }
    }
    MarkCrossSubgraph(function, noMergeSubgraph);
    for (int i = 0; i < subgraphNum; i++) {
        if (subgraphOpNum[i] == 1 && subgraphHasReshape[i] == true) {
            noMergeSubgraph.insert(i);
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d is not mergeable because it only has Reshape.", i);
        } else if (subgraphHasInnerDDR[i]) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d has inner DDR tensor.", i);
        }
    }
    return SUCCESS;
}

Status ReduceCopyMerge::BuildGraph(Function& function, MergeInput& mergeInput)
{
    int subgraphNum = function.GetTotalSubGraphCount();
    mergeInput.numSubgraph = subgraphNum;
    mergeInput.subgraphAICLatency.clear();
    mergeInput.subgraphAICLatency.resize(subgraphNum, 0);
    mergeInput.subgraphAIVLatency.clear();
    mergeInput.subgraphAIVLatency.resize(subgraphNum, 0);
    mergeInput.subGraphOutGraph.clear();
    mergeInput.subGraphOutGraph.resize(subgraphNum);
    mergeInput.subGraphInGraph.clear();
    mergeInput.subGraphInGraph.resize(subgraphNum);
    for (auto& op : function.Operations()) {
        int src = op.GetSubgraphID();
        int opLatency = op.GetLatency();
        if (op.HasAttr(OpAttributeKey::isCube) && op.GetBoolAttribute(OpAttributeKey::isCube)) {
            mergeInput.subgraphAICLatency[src] += opLatency;
        } else {
            mergeInput.subgraphAIVLatency[src] += opLatency;
        }
        for (auto& consumer : op.ConsumerOps()) {
            int dst = consumer->GetSubgraphID();
            if (src != dst) {
                mergeInput.subGraphOutGraph[src].insert(dst);
                mergeInput.subGraphInGraph[dst].insert(src);
            }
        }
    }
    for (int i = 0; i < subgraphNum; i++) {
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph %d : AIC Latency %d, AIV Latency %d.", i,
                         mergeInput.subgraphAICLatency[i], mergeInput.subgraphAIVLatency[i]);
    }
    return SUCCESS;
}

bool ReduceCopyMerge::IsEnforceMergeBoundary(LogicalTensorPtr& tensor)
{
    std::unordered_set<int> boundaryScopeIds;
    for (auto& op : tensor->GetProducers()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Boundary tensor %d has produce %d with scopeInfoCvFuseId %d.",
                          tensor->GetMagic(), op->GetOpMagic(), op->GetCvFuseId());
        if (op->GetCvFuseId() == -1) {
            return false;
        }
        boundaryScopeIds.insert(op->GetCvFuseId());
    }
    for (auto& op : tensor->GetConsumers()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Boundary tensor %d has consumer %d with scopeInfoCvFuseId %d.",
                          tensor->GetMagic(), op->GetOpMagic(), op->GetCvFuseId());
        if (op->GetCvFuseId() == -1) {
            return false;
        }
        boundaryScopeIds.insert(op->GetCvFuseId());
    }
    if (boundaryScopeIds.size() == 1) {
        return true;
    }
    return false;
}

void ReduceCopyMerge::UpdateBoundaryTensorSize(LogicalTensorPtr& tensor, int tensorSize)
{
    for (auto& op : tensor->GetProducers()) {
        subgraphOutputSize[op->GetSubgraphID()] += tensorSize;
        subgraphToOutputTensors[op->GetSubgraphID()].push_back(tensor->GetMagic());
    }
    for (auto& op : tensor->GetConsumers()) {
        subgraphInputSize[op->GetSubgraphID()] += tensorSize;
        subgraphToInputTensors[op->GetSubgraphID()].push_back(tensor->GetMagic());
    }
}

void ReduceCopyMerge::RecordBoundaryTensorInfo(
    LogicalTensorPtr& tensor, MergeInput& mergeInput, const std::set<int>& connectGraphs)
{
    BoundaryTensorInfo tensorInfo;
    tensorInfo.tensorMagic = tensor->GetMagic();
    std::set<int> producerSubgraphs;
    std::set<int> consumerSubgraphs;
    for (auto& op : tensor->GetProducers()) {
        producerSubgraphs.insert(op->GetSubgraphID());
    }
    for (auto& op : tensor->GetConsumers()) {
        consumerSubgraphs.insert(op->GetSubgraphID());
    }
    tensorInfo.producerSubgraphs.assign(producerSubgraphs.begin(), producerSubgraphs.end());
    tensorInfo.consumerSubgraphs.assign(consumerSubgraphs.begin(), consumerSubgraphs.end());
    int tensorId = static_cast<int>(mergeInput.boundaryTensors.size());
    mergeInput.boundaryTensors.push_back(tensorInfo);
    for (int subgraphId : connectGraphs) {
        mergeInput.subgraphToBoundaryTensorIds[subgraphId].push_back(tensorId);
    }
}

void ReduceCopyMerge::UpdateConnectRecord(Function& function, MergeInput& mergeInput)
{
    for (auto tensor : GraphUtils::GetAllTensors(function)) {
        int tensorSize = tensor->MemorySize();
        if (tensor->GetProducers().size() == 0 || tensor->GetConsumers().size() == 0) {
            continue;
        }
        std::set<int> connectGraphs;
        for (auto& op : tensor->GetProducers()) {
            connectGraphs.insert(op->GetSubgraphID());
        }
        for (auto& op : tensor->GetConsumers()) {
            connectGraphs.insert(op->GetSubgraphID());
        }
        if (connectGraphs.size() <= 1) {
            continue;
        }
        int tensorMagic = tensor->GetMagic();
        UpdateBoundaryTensorSize(tensor, tensorSize);
        RecordBoundaryTensorInfo(tensor, mergeInput, connectGraphs);
        std::vector<int> mergeGroup(connectGraphs.begin(), connectGraphs.end());
        tensorToMergeGroup[tensorMagic] = mergeGroup;
        APASS_LOG_DEBUG_F(Elements::Operation, "Found boundary tensor %d of subgraphs %s.",
            tensor->GetMagic(), IntVecToString(mergeGroup).c_str());
        mergeGroupToPriority[mergeGroup] += tensorSize;
        if (IsEnforceMergeBoundary(tensor)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "----boundary tensor %d is marked as enforced.", tensor->GetMagic());
            enforceMergeGroup.insert(mergeGroup);
        }
    }
}

Status ReduceCopyMerge::BuildMergeGroup(Function& function, MergeInput& mergeInput)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Build merge group before mix subgraph merge start.");
    subgraphInputSize.resize(mergeInput.numSubgraph, 0);
    subgraphOutputSize.resize(mergeInput.numSubgraph, 0);
    mergeInput.boundaryTensors.clear();
    mergeInput.subgraphToBoundaryTensorIds.clear();
    mergeInput.subgraphToBoundaryTensorIds.resize(mergeInput.numSubgraph);
    UpdateConnectRecord(function, mergeInput);
    std::multimap<int, std::vector<int>> sortedMergeGroup;
    for (auto& pair : mergeGroupToPriority) {
        std::vector<int> boundaryGroup = pair.first;
        std::sort(boundaryGroup.begin(), boundaryGroup.end());
        sortedMergeGroup.insert({pair.second, boundaryGroup});
    }
    for (int subIdx = 0; subIdx < mergeInput.numSubgraph; subIdx++) {
        if (subgraphToOutputTensors.count(subIdx) == 0) {
            continue;
        }
        std::set<int> localGroup;
        for (auto localBoundary : subgraphToOutputTensors[subIdx]) {
            localGroup.insert(tensorToMergeGroup[localBoundary].begin(), tensorToMergeGroup[localBoundary].end());
        }
        if (localGroup.size() > 1) {
            std::vector<int> group(localGroup.begin(), localGroup.end());
            sortedMergeGroup.insert({subgraphOutputSize[subIdx], group});
        }
    }
    for (int subIdx = 0; subIdx < mergeInput.numSubgraph; subIdx++) {
        if (subgraphToInputTensors.count(subIdx) == 0) {
            continue;
        }
        std::set<int> localGroup;
        for (auto localBoundary : subgraphToInputTensors[subIdx]) {
            localGroup.insert(tensorToMergeGroup[localBoundary].begin(), tensorToMergeGroup[localBoundary].end());
        }
        if (localGroup.size() > 1) {
            std::vector<int> group(localGroup.begin(), localGroup.end());
            sortedMergeGroup.insert({subgraphInputSize[subIdx], group});
        }
    }
    UpdateMergeInput(mergeInput, sortedMergeGroup);
    return SUCCESS;
}

void ReduceCopyMerge::UpdateMergeInput(MergeInput& mergeInput, std::multimap<int, std::vector<int>>& sortedMergeGroup)
{
    int groupIdx = 0;
    std::set<std::vector<int>> visitedMergeGroup;
    for (auto it = sortedMergeGroup.rbegin(); it != sortedMergeGroup.rend(); it++) {
        if (visitedMergeGroup.count(it->second) > 0 ||
            !IsValidMergeGroup(it->second, noMergeSubgraphEnforce)) {
            continue;
        }
        visitedMergeGroup.insert(it->second);
        mergeInput.mergeGroup.push_back(it->second);
        bool isEnforce = enforceMergeGroup.count(it->second) > 0 ? true : false;
        mergeInput.isEnforceMergeGroup.push_back(isEnforce);
        mergeInput.isValidMergeGroup.push_back(IsValidMergeGroup(it->second, noMergeSubgraph));
        APASS_LOG_DEBUG_F(Elements::Operation, "merge group %d: %s, isEnforce: %s.",
            groupIdx, IntVecToString(it->second).c_str(), isEnforce ? "True" : "False");
        groupIdx++;
    }
}

Status ReduceCopyMerge::PostCheck(Function& function)
{
    (void)function;
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for ReduceCopy.");
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip PostCheck for ReduceCopy Pass.");
        return SUCCESS;
    }
    return SUCCESS;
}

static bool ValidateInput(const MergeInput& input)
{
    if (input.numSubgraph < 0 || input.maxLatency < 0) {
        return false;
    }
    int n = input.numSubgraph;
    if ((int)input.subgraphAICLatency.size() != n || (int)input.subgraphAIVLatency.size() != n ||
        (int)input.subGraphOutGraph.size() != n) {
        return false;
    }
    if (input.mergeGroup.size() != input.isEnforceMergeGroup.size()) {
        return false;
    }
    if (!input.subgraphToBoundaryTensorIds.empty() &&
        static_cast<int>(input.subgraphToBoundaryTensorIds.size()) != n) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (input.subgraphAICLatency[i] < 0 || input.subgraphAIVLatency[i] < 0) {
            return false;
        }
    }
    return true;
}

void MixGraphMerger::Initialize(const MergeInput& input)
{
    mInput = input;
    mParent.resize(input.numSubgraph);
    mRank.resize(input.numSubgraph, 0);
    for (int i = 0; i < input.numSubgraph; ++i) {
        mParent[i] = i;
    }
    mOutput.numSubgraphUpdated = input.numSubgraph;
    mOutput.subgraphIdUpdated.resize(input.numSubgraph);
    for (int i = 0; i < input.numSubgraph; ++i) {
        mOutput.subgraphIdUpdated[i] = i;
    }
    estimateInput.execTime.resize(input.numSubgraph);
    estimateInput.isCube.resize(input.numSubgraph);
    estimateInput.outGraph = input.subGraphOutGraph;
    estimateInput.inGraph = input.subGraphInGraph;
    for (int i = 0; i < input.numSubgraph; ++i) {
        estimateInput.execTime[i] = input.subgraphAICLatency[i] + input.subgraphAIVLatency[i];
        estimateInput.isCube[i] = (input.subgraphAICLatency[i] > 0 ? true : false);
    }
    InitBoundaryTensorIndex();
}

void MixGraphMerger::InitBoundaryTensorIndex()
{
    mRootToBoundaryTensorIds.assign(mInput.numSubgraph, std::vector<int>());
    mTensorVisitStamp.assign(mInput.boundaryTensors.size(), 0);
    mVisitStamp = 0;
    if (mInput.subgraphToBoundaryTensorIds.empty()) {
        return;
    }
    mRootToBoundaryTensorIds = mInput.subgraphToBoundaryTensorIds;
}

int MixGraphMerger::FindParent(int x)
{
    if (mParent[x] != x) {
        mParent[x] = FindParent(mParent[x]);
    }
    return mParent[x];
}

void MixGraphMerger::UnionSets(int x, int y)
{
    int px = FindParent(x);
    int py = FindParent(y);
    if (px == py) {
        return;
    }
    if (mRank[px] < mRank[py]) {
        mParent[px] = py;
    } else if (mRank[px] > mRank[py]) {
        mParent[py] = px;
    } else {
        mParent[py] = px;
        mRank[px]++;
    }
}

std::vector<int> MixGraphMerger::GetActualGroup(const std::vector<int>& group)
{
    std::set<int> actualSet;
    for (int idx : group) {
        if (idx >= 0 && idx < mInput.numSubgraph) {
            actualSet.insert(FindParent(idx));
        }
    }
    return std::vector<int>(actualSet.begin(), actualSet.end());
}

void MixGraphMerger::BuildMergedGraph(std::vector<std::set<int>>& outGraph, std::vector<std::set<int>>& inGraph)
{
    int n = mInput.numSubgraph;
    outGraph.assign(n, std::set<int>());
    inGraph.assign(n, std::set<int>());
    for (int i = 0; i < n; ++i) {
        int pi = FindParent(i);
        for (int j : mInput.subGraphOutGraph[i]) {
            int pj = FindParent(j);
            if (pi != pj) {
                outGraph[pi].insert(pj);
                inGraph[pj].insert(pi);
            }
        }
    }
}

bool MixGraphMerger::HasCycle(const std::vector<std::set<int>>& outGraph, const std::vector<std::set<int>>& inGraph)
{
    int n = mInput.numSubgraph;
    std::vector<int> inDegree(n, 0);
    std::vector<bool> isRoot(n, false);
    int rootCount = 0;
    for (int i = 0; i < n; ++i) {
        if (FindParent(i) == i) {
            isRoot[i] = true;
            inDegree[i] = inGraph[i].size();
            rootCount++;
        }
    }
    std::queue<int> q;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (isRoot[i] && inDegree[i] == 0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        count++;
        for (int v : outGraph[u]) {
            if (isRoot[v]) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }
    }
    return count != rootCount;
}

bool MixGraphMerger::CanMergeWithoutCycle(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        return false;
    }
    std::vector<std::set<int>> inGraph;
    std::vector<std::set<int>> outGraph;
    BuildMergedGraph(outGraph, inGraph);
    int root = actualGroup[0];
    std::set<int> del(actualGroup.begin() + 1, actualGroup.end());
    for (int d : del) {
        for (int u : inGraph[d]) {
            outGraph[u].erase(d);
            outGraph[u].insert(root);
        }
        for (int v : outGraph[d]) {
            inGraph[v].erase(d);
            inGraph[v].insert(root);
        }
        inGraph[root].insert(inGraph[d].begin(), inGraph[d].end());
        outGraph[root].insert(outGraph[d].begin(), outGraph[d].end());
        inGraph[d].clear();
        outGraph[d].clear();
    }
    inGraph[root].erase(root);
    outGraph[root].erase(root);
    if (HasCycle(outGraph, inGraph)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: detect cycle.");
        return false;
    }
    return true;
}

bool MixGraphMerger::CheckLatencyConstraint(const std::vector<int>& actualGroup)
{
    int totalAIC = 0;
    int totalAIV = 0;
    for (int root : actualGroup) {
        for (int i = 0; i < mInput.numSubgraph; ++i) {
            if (FindParent(i) == root) {
                totalAIC += mInput.subgraphAICLatency[i];
                totalAIV += mInput.subgraphAIVLatency[i];
            }
        }
    }
    if (totalAIC == 0 || totalAIV == 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
            "Merge skipped: merged subgraph must be mixed (both AIC and AIV non-zero).");
        return false;
    }
    int totalLatency = totalAIC + totalAIV;
    if (totalLatency > mInput.maxLatency) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: total latency %d exceeds max latency %d.",
            totalLatency, mInput.maxLatency);
        return false;
    }
    double ratio = (double)totalAIV / (double)totalAIC;
    if (ratio < mInput.aivRatio.first || ratio > mInput.aivRatio.second) {
        APASS_LOG_DEBUG_F(Elements::Operation,
            "Merge skipped: AIV/AIC ratio %.2f out of range [%.2f, %.2f]",
            ratio, mInput.aivRatio.first, mInput.aivRatio.second);
        return false;
    }
    return true;
}

bool MixGraphMerger::CheckMergeBenefit(const std::vector<int>& actualGroup)
{
    std::unordered_set<int> mergedRoot(actualGroup.begin(), actualGroup.end());
    std::vector<std::set<int>> originalGroup;
    std::vector<std::set<int>> mergedGroup;
    std::unordered_map<int, std::set<int>> rootToNodes;
    for (size_t i = 0; i < mParent.size(); i++) {
        int root = FindParent(i);
        rootToNodes[root].insert(i);
    }
    mergedGroup.push_back({});
    for (auto& pr : rootToNodes) {
        originalGroup.push_back(pr.second);
        if (mergedRoot.count(pr.first) > 0) {
            mergedGroup[0].insert(pr.second.begin(), pr.second.end());
        } else {
            mergedGroup.push_back(pr.second);
        }
    }
    EstimateExecTime originalTimeEstimator;
    int originalTime = originalTimeEstimator.Estimate(estimateInput, originalGroup);
    EstimateExecTime mergedTimeEstimator;
    int mergedTime = mergedTimeEstimator.Estimate(estimateInput, mergedGroup);

    APASS_LOG_DEBUG_F(Elements::Operation,
        "Estimate exec time before merge: %d, after merge: %d.", originalTime, mergedTime);

    if (mergedTime > originalTime) {
        APASS_LOG_DEBUG_F(Elements::Operation,
            "Merge skipped: estimate merged exec time > original exec time.");
        return false;
    }
    return true;
}

bool MixGraphMerger::IsInvalidMergedInnerTensor(int tensorId, const std::unordered_set<int>& mergedRoots)
{
    const auto& tensorInfo = mInput.boundaryTensors[tensorId];
    bool hasProducerInMix = false;
    bool hasConsumerInMix = false;
    bool hasExternalEndpoint = false;
    for (int subgraphId : tensorInfo.producerSubgraphs) {
        int root = FindParent(subgraphId);
        if (mergedRoots.count(root) > 0) {
            hasProducerInMix = true;
        } else {
            hasExternalEndpoint = true;
        }
    }
    for (int subgraphId : tensorInfo.consumerSubgraphs) {
        int root = FindParent(subgraphId);
        if (mergedRoots.count(root) > 0) {
            hasConsumerInMix = true;
        } else {
            hasExternalEndpoint = true;
        }
    }
    return hasProducerInMix && hasConsumerInMix && hasExternalEndpoint;
}

bool MixGraphMerger::CheckNoExternalUseOfMergedInnerTensor(const std::vector<int>& actualGroup)
{
    std::unordered_set<int> mergedRoots(actualGroup.begin(), actualGroup.end());
    ++mVisitStamp;
    for (int root : actualGroup) {
        for (int tensorId : mRootToBoundaryTensorIds[root]) {
            if (mTensorVisitStamp[tensorId] == mVisitStamp) {
                continue;
            }
            mTensorVisitStamp[tensorId] = mVisitStamp;
            if (IsInvalidMergedInnerTensor(tensorId, mergedRoots)) {
                APASS_LOG_DEBUG_F(Elements::Operation,
                    "Merge skipped: tensor %d would become an inner tensor with external usage.",
                    mInput.boundaryTensors[tensorId].tensorMagic);
                return false;
            }
        }
    }
    return true;
}

bool MixGraphMerger::CanMergeWithConstraints(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: already merged.");
        return false;
    }
    if (!CanMergeWithoutCycle(actualGroup)) {
        return false;
    }
    if (!CheckNoExternalUseOfMergedInnerTensor(actualGroup)) {
        return false;
    }
    if (!CheckLatencyConstraint(actualGroup)) {
        return false;
    }
    if (!CheckMergeBenefit(actualGroup)) {
        return false;
    }
    return true;
}

void MixGraphMerger::UpdateBoundaryTensorIndex(const std::vector<int>& actualGroup)
{
    std::vector<int> mergedTensorIds;
    for (int root : actualGroup) {
        mergedTensorIds.insert(mergedTensorIds.end(), mRootToBoundaryTensorIds[root].begin(),
            mRootToBoundaryTensorIds[root].end());
        mRootToBoundaryTensorIds[root].clear();
    }
    int newRoot = FindParent(actualGroup[0]);
    mRootToBoundaryTensorIds[newRoot].swap(mergedTensorIds);
}

void MixGraphMerger::PerformMerge(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        return;
    }
    int root = actualGroup[0];
    for (size_t i = 1; i < actualGroup.size(); ++i) {
        UnionSets(root, actualGroup[i]);
    }
    UpdateBoundaryTensorIndex(actualGroup);
}

void MixGraphMerger::UpdateOutput()
{
    std::vector<int> mapping(mInput.numSubgraph, -1);
    int newId = 0;
    for (int i = 0; i < mInput.numSubgraph; ++i) {
        int root = FindParent(i);
        if (mapping[root] == -1) {
            mapping[root] = newId++;
        }
        mOutput.subgraphIdUpdated[i] = mapping[root];
    }
    mOutput.numSubgraphUpdated = newId;
}

static bool ValidateOutput(const MergeOutput& output, int numSubgraph)
{
    if (output.numSubgraphUpdated <= 0 || output.numSubgraphUpdated > numSubgraph) {
        return false;
    }
    if ((int)output.subgraphIdUpdated.size() != numSubgraph) {
        return false;
    }
    std::set<int> ids;
    for (int i = 0; i < numSubgraph; ++i) {
        int id = output.subgraphIdUpdated[i];
        if (id < 0 || id >= output.numSubgraphUpdated) {
            return false;
        }
        ids.insert(id);
    }
    for (int i = 0; i < output.numSubgraphUpdated; ++i) {
        if (ids.find(i) == ids.end()) {
            return false;
        }
    }
    return true;
}

MergeOutput MixGraphMerger::Merge(const MergeInput& input)
{
    if (!ValidateInput(input)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Invalid input parameters");
        return mOutput;
    }
    Initialize(input);
    const int mergeLoopNum = 5;
    for (int mergeLoopStep = 0; mergeLoopStep < mergeLoopNum; mergeLoopStep++) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Enter merge loop %d.", mergeLoopStep);
        bool hasUpdated = false;
        for (size_t i = 0; i < input.mergeGroup.size(); ++i) {
            const auto& group = input.mergeGroup[i];
            std::vector<int> actualGroup = GetActualGroup(group);
            if (actualGroup.size() <= 1) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Skip merge group %zu: already merged", i);
                continue;
            }
            if ((input.isEnforceMergeGroup[i] && CanMergeWithoutCycle(actualGroup)) ||
                (enableAutoMix && mergeLoopStep != 0 && input.isValidMergeGroup[i] &&
                    CanMergeWithConstraints(actualGroup))) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Merge group %zu succeeded", i);
                PerformMerge(actualGroup);
                hasUpdated = true;
            } else if (input.isEnforceMergeGroup[i] || mergeLoopStep != 0) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Merge group %zu skipped due to constraints.", i);
            }
        }
        if (mergeLoopStep > 0 && !hasUpdated) {
            break;
        }
    }
    UpdateOutput();
    if (!ValidateOutput(mOutput, input.numSubgraph)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Invalid output detected");
    }
    return mOutput;
}

void EstimateExecTime::InitMixData(MixScheduleContext& ctx, const std::vector<std::set<int>>& estimateCandidate)
{
    ctx.numMix = static_cast<int>(estimateCandidate.size());
    ctx.numSubgraph = static_cast<int>(ctx.subgraphToMix.size());
    for (int mixId = 0; mixId < ctx.numMix; ++mixId) {
        for (int idx : estimateCandidate[mixId]) {
            ctx.candidateSet.insert(idx);
            ctx.subgraphToMix[idx] = mixId;
        }
    }
}

void EstimateExecTime::BuildMixDeps(MixScheduleContext& ctx, const EstimateInput& input)
{
    for (int idx : ctx.candidateSet) {
        int mixId = ctx.subgraphToMix[idx];
        for (int pred : input.inGraph[idx]) {
            if (ctx.candidateSet.count(pred) > 0) {
                int predMixId = ctx.subgraphToMix[pred];
                if (predMixId != mixId) {
                    ctx.mixDeps[mixId].insert(predMixId);
                }
            }
        }
    }
}

void EstimateExecTime::InitMixTopology(MixScheduleContext& ctx)
{
    for (int mixId = 0; mixId < ctx.numMix; ++mixId) {
        ctx.mixInDegree[mixId] = static_cast<int>(ctx.mixDeps[mixId].size());
        if (ctx.mixInDegree[mixId] == 0) {
            ctx.mixReadyQueue.push(mixId);
        }
    }
}

int EstimateExecTime::CalcMixStartTime(int mixId, const MixScheduleContext& ctx, int scheduleTime)
{
    int startTime = 0;
    for (int predMixId : ctx.mixDeps[mixId]) {
        int predFinish = ctx.mixFinishTime[predMixId] + scheduleTime;
        startTime = std::max(startTime, predFinish);
    }
    return startTime;
}

void EstimateExecTime::InitSubgraphContext(
    SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx, const EstimateInput& input)
{
    subCtx.numSubgraph = ctx.numSubgraph;
    subCtx.finishTime.resize(ctx.numSubgraph, 0);
    subCtx.inDegree.resize(ctx.numSubgraph, 0);
    subCtx.coreState = {subCtx.mixStartTime, subCtx.mixStartTime, subCtx.mixStartTime};
    for (int idx : ctx.candidateSet) {
        if (ctx.subgraphToMix[idx] != subCtx.mixId) {
            continue;
        }
        int degree = 0;
        for (int pred : input.inGraph[idx]) {
            if (ctx.candidateSet.count(pred) > 0 && ctx.subgraphToMix[pred] == subCtx.mixId) {
                degree++;
            }
        }
        subCtx.inDegree[idx] = degree;
        if (degree == 0) {
            subCtx.readyQueue.push(idx);
        }
    }
}

void EstimateExecTime::ScheduleOneSubgraph(
    int current, SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx, const EstimateInput& input)
{
    int earliestStart = subCtx.mixStartTime;
    for (int pred : input.inGraph[current]) {
        if (ctx.candidateSet.count(pred) > 0 && ctx.subgraphToMix[pred] == subCtx.mixId) {
            earliestStart = std::max(earliestStart, subCtx.finishTime[pred]);
        }
    }
    bool isAic = input.isCube[current];
    int startTime = isAic ? std::max(earliestStart, subCtx.coreState.aic) :
                            std::max(earliestStart, std::min(subCtx.coreState.aiv0, subCtx.coreState.aiv1));
    subCtx.finishTime[current] = startTime + input.execTime[current];
    if (isAic) {
        subCtx.coreState.aic = subCtx.finishTime[current];
    } else if (subCtx.coreState.aiv0 <= subCtx.coreState.aiv1) {
        subCtx.coreState.aiv0 = subCtx.finishTime[current];
    } else {
        subCtx.coreState.aiv1 = subCtx.finishTime[current];
    }
}

void EstimateExecTime::ProcessSubgraphConsumers(
    int current, SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx, const EstimateInput& input)
{
    for (int consumer : input.outGraph[current]) {
        if (ctx.candidateSet.count(consumer) > 0 && ctx.subgraphToMix[consumer] == subCtx.mixId) {
            subCtx.inDegree[consumer]--;
            if (subCtx.inDegree[consumer] == 0) {
                subCtx.readyQueue.push(consumer);
            }
        }
    }
}

int EstimateExecTime::GetMixFinishTime(const SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx)
{
    int mixFinish = 0;
    for (int idx : ctx.candidateSet) {
        if (ctx.subgraphToMix[idx] == subCtx.mixId) {
            mixFinish = std::max(mixFinish, subCtx.finishTime[idx]);
        }
    }
    return mixFinish;
}

void EstimateExecTime::ProcessMixConsumers(int mixId, MixScheduleContext& ctx)
{
    for (int consumerMix = 0; consumerMix < ctx.numMix; ++consumerMix) {
        if (ctx.mixDeps[consumerMix].count(mixId) > 0) {
            ctx.mixInDegree[consumerMix]--;
            if (ctx.mixInDegree[consumerMix] == 0) {
                ctx.mixReadyQueue.push(consumerMix);
            }
        }
    }
}

int EstimateExecTime::Estimate(const EstimateInput& input, const std::vector<std::set<int>>& estimateCandidate)
{
    if (estimateCandidate.empty()) {
        return 0;
    }
    MixScheduleContext ctx;
    ctx.subgraphToMix.resize(input.outGraph.size(), -1);
    ctx.mixDeps.resize(estimateCandidate.size());
    ctx.mixStartTime.resize(estimateCandidate.size(), 0);
    ctx.mixFinishTime.resize(estimateCandidate.size(), 0);
    ctx.mixInDegree.resize(estimateCandidate.size(), 0);
    InitMixData(ctx, estimateCandidate);
    if (ctx.candidateSet.empty()) {
        return 0;
    }
    BuildMixDeps(ctx, input);
    InitMixTopology(ctx);
    while (!ctx.mixReadyQueue.empty()) {
        int mixId = ctx.mixReadyQueue.front();
        ctx.mixReadyQueue.pop();
        ctx.mixStartTime[mixId] = CalcMixStartTime(mixId, ctx, input.betweenSubgraphScheduleTime);
        SubgraphScheduleContext subCtx;
        subCtx.mixId = mixId;
        subCtx.mixStartTime = ctx.mixStartTime[mixId];
        InitSubgraphContext(subCtx, ctx, input);
        while (!subCtx.readyQueue.empty()) {
            int current = subCtx.readyQueue.front();
            subCtx.readyQueue.pop();
            ScheduleOneSubgraph(current, subCtx, ctx, input);
            ProcessSubgraphConsumers(current, subCtx, ctx, input);
        }
        ctx.mixFinishTime[mixId] = GetMixFinishTime(subCtx, ctx);
        ProcessMixConsumers(mixId, ctx);
    }
    int result = 0;
    for (int mixId = 0; mixId < ctx.numMix; ++mixId) {
        result = std::max(result, ctx.mixFinishTime[mixId]);
    }
    return result;
}

} // namespace npu::tile_fwk
