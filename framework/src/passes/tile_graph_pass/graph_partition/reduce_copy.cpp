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
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <sstream>

#define MODULE_NAME "ReduceCopy"

namespace npu::tile_fwk {

Status ReduceCopyMerge::RunOnFunction(Function &function)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip ReduceCopy Pass.");
        return SUCCESS;
    }
    MergeInput mergeInput;
    mergeInput.maxLatency = 1000;
    mergeInput.aivRatio = {0.5, 2.0};
    APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph Info before ReduceCopy Pass:");
    BuildGraph(function, mergeInput);
    MarkNoMergeSubgraph(function);
    BuildMergeGroup(function, mergeInput);
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(mergeInput);
    function.SetTotalSubGraphCount(output.numSubgraphUpdated);
    for (auto &op : function.Operations()) {
        int src = op.GetSubgraphID();
        if (src >= static_cast<int>(output.subgraphIdUpdated.size())) {
            APASS_LOG_ERROR_F(Elements::Operation, "Current op subgraphID not in ReduceCopy subgraph update record.");
            return FAILED;
        }
        op.UpdateSubgraphID(output.subgraphIdUpdated[src]);
    }
    MergeInput mergeInputTmp;
    APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph Info after ReduceCopy Pass:");
    BuildGraph(function, mergeInputTmp);
    return SUCCESS;
}

Status ReduceCopyMerge::MarkNoMergeSubgraph(Function &function)
{
    int subgraphNum = function.GetTotalSubGraphCount();
    std::vector<int> subgraphOpNum(subgraphNum, 0);
    std::vector<bool> subgraphHasReshape(subgraphNum, false);
    std::vector<bool> subgraphHasInnerDDR(subgraphNum, false);
    for (auto &op : function.Operations()) {
        int src = op.GetSubgraphID();
        subgraphOpNum[src] += 1;
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            subgraphHasReshape[src] = true;
        }
        for (auto &operand : op.GetOOperands()) {
            if (operand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            for (auto &consumer : operand->GetConsumers()) {
                if (consumer->GetSubgraphID() == src) {
                    subgraphHasInnerDDR[src] = true;
                    break;
                }
            }
        }
    }
    noMergeSubgraph.clear();
    for (int i = 0; i < subgraphNum; i++) {
        if (subgraphOpNum[i] == 1 && subgraphHasReshape[i] == true) {
            noMergeSubgraph.insert(i);
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d is not mergeable beacause it only has Reshape.", i);
        } else if (subgraphHasInnerDDR[i]) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d has inner DDR tensor.", i);
        }
    }
    return SUCCESS;
}

Status ReduceCopyMerge::BuildGraph(Function &function, MergeInput& mergeInput)
{
    int subgraphNum = function.GetTotalSubGraphCount();
    mergeInput.numSubgraph = subgraphNum;
    mergeInput.subgraphAICLatency.clear();
    mergeInput.subgraphAICLatency.resize(subgraphNum, 0);
    mergeInput.subgraphAIVLatency.clear();
    mergeInput.subgraphAIVLatency.resize(subgraphNum, 0);
    mergeInput.subGraphOutGraph.clear();
    mergeInput.subGraphOutGraph.resize(subgraphNum);
    for (auto &op : function.Operations()) {
        int src = op.GetSubgraphID();
        int opLatency = op.GetLatency();
        if (op.HasAttr(OpAttributeKey::isCube) && op.GetBoolAttribute(OpAttributeKey::isCube)) {
            mergeInput.subgraphAICLatency[src] += opLatency;
        } else {
            mergeInput.subgraphAIVLatency[src] += opLatency;
        }
        for (auto &consumer : op.ConsumerOps()) {
            int dst = consumer->GetSubgraphID();
            if (src != dst) {
                mergeInput.subGraphOutGraph[src].insert(dst);
            }
        }
    }
    for (int i = 0; i < subgraphNum; i++) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d : AIC Latency %d, AIV Latency %d.", i,
                          mergeInput.subgraphAICLatency[i], mergeInput.subgraphAIVLatency[i]);
    }
    return SUCCESS;
}

bool ReduceCopyMerge::IsEnforceMergeBoundary(LogicalTensorPtr &tensor)
{
    std::unordered_set<int> boundaryScopeIds;
    for (auto &op : tensor->GetProducers()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Boundary tensor %d has produce %d with scopeInfoCvFuseId %d.",
                          tensor->GetMagic(), op->GetOpMagic(), op->GetCvFuseId());
        if (op->GetCvFuseId() == -1) {
            return false;
        }
        boundaryScopeIds.insert(op->GetCvFuseId());
    }
    for (auto &op : tensor->GetConsumers()) {
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

static std::string IntVecToString(const std::vector<int> &vec)
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

static bool IsValidMergeGroup(const std::vector<int> &mergeGroup, const std::unordered_set<int> &noMergeSubgraph)
{
    for (int subidx : mergeGroup) {
        if (noMergeSubgraph.count(subidx) > 0) {
            return false;
        }
    }
    return true;
}

Status ReduceCopyMerge::BuildMergeGroup(Function &function, MergeInput& mergeInput)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Build merge group before mix subgraph merge start.");
    std::map<std::vector<int>, int> mergeGroupToPriority;
    std::set<std::vector<int>> enforceMergeGroup;
    for (const auto &item : function.GetTensorMap().inverseMap_) {
        LogicalTensorPtr tensor = item.second;
        if (item.second->GetProducers().size() == 0 || item.second->GetConsumers().size() == 0) {
            continue;
        }
        std::set<int> connectGraphs;
        for (auto &op : item.second->GetProducers()) {
            connectGraphs.insert(op->GetSubgraphID());
        }
        for (auto &op : item.second->GetConsumers()) {
            connectGraphs.insert(op->GetSubgraphID());
        }
        if (connectGraphs.size() <= 1) {
            continue;
        }
        std::vector<int> mergeGroup(connectGraphs.begin(), connectGraphs.end());
        APASS_LOG_DEBUG_F(Elements::Operation, "Found boundary tensor %d of subgraphs %s.",
            tensor->GetMagic(), IntVecToString(mergeGroup).c_str());
        mergeGroupToPriority[mergeGroup] += tensor->MemorySize();
        if (IsEnforceMergeBoundary(tensor)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "----boundary tensor %d is marked as enforced.", tensor->GetMagic());
            enforceMergeGroup.insert(mergeGroup);
        }
    }
    std::multimap<int, std::vector<int>> sortedMergeGroup;
    for (auto &pair : mergeGroupToPriority) {
        sortedMergeGroup.insert({pair.second, pair.first});
    }
    mergeInput.mergeGroup.clear();
    mergeInput.isEnforceMergeGroup.clear();
    int groupIdx = 0;
    for (auto it = sortedMergeGroup.rbegin(); it != sortedMergeGroup.rend(); it++) {
        if (!IsValidMergeGroup(it->second, noMergeSubgraph)) {
            continue;
        }
        mergeInput.mergeGroup.push_back(it->second);
        bool isEnforce = enforceMergeGroup.count(it->second) > 0 ? true : false;
        mergeInput.isEnforceMergeGroup.push_back(isEnforce);
        APASS_LOG_DEBUG_F(Elements::Operation, "merge group %d: %s, isEnforce: %s.",
            groupIdx, IntVecToString(it->second).c_str(), isEnforce ? "True" : "False");
        groupIdx++;
    }
    return SUCCESS;
}

Status ReduceCopyMerge::PostCheck(Function &function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for ReduceCopy.");
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip PostCheck for ReduceCopy Pass.");
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> Start PostCheck for ReduceCopy.");
    for (auto &op : function.Operations()) {
        if (op.GetInternalSubgraphID() < 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "Op %d does not belong to any internalSubgraph.", op.GetOpMagic());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> Finish PostCheck for ReduceCopy.");
    return SUCCESS;
}

static bool ValidateInput(const MergeInput& input)
{
    if (input.numSubgraph < 0 || input.maxLatency < 0) {
        return false;
    }
    int n = input.numSubgraph;
    if ((int)input.subgraphAICLatency.size() != n ||
        (int)input.subgraphAIVLatency.size() != n ||
        (int)input.subGraphOutGraph.size() != n) {
        return false;
    }
    if (input.mergeGroup.size() != input.isEnforceMergeGroup.size()) {
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
    if (px == py) return;
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
    if (actualGroup.size() <= 1) return false;
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
        APASS_LOG_INFO_F(Elements::Operation, "Merge failed: detect cycle.");
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
        APASS_LOG_INFO_F(Elements::Operation,
            "Merge failed: merged subgraph must be mixed (both AIC and AIV non-zero).");
        return false;
    }
    int totalLatency = totalAIC + totalAIV;
    if (totalLatency > mInput.maxLatency) {
        APASS_LOG_INFO_F(Elements::Operation, "Merge failed: total latency %d exceeds max latency %d.",
            totalLatency, mInput.maxLatency);
        return false;
    }
    double ratio = (double)totalAIV / (double)totalAIC;
    if (ratio < mInput.aivRatio.first || ratio > mInput.aivRatio.second) {
        APASS_LOG_INFO_F(Elements::Operation,
            "Merge failed: AIV/AIC ratio %.2f out of range [%.2f, %.2f]",
            ratio, mInput.aivRatio.first, mInput.aivRatio.second);
        return false;
    }
    return true;
}

bool MixGraphMerger::CanMergeWithConstraints(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        APASS_LOG_INFO_F(Elements::Operation, "Merge failed: already merged.");
        return false;
    }
    if (!CanMergeWithoutCycle(actualGroup)) {
        return false;
    }
    if (!CheckLatencyConstraint(actualGroup)) {
        return false;
    }
    return true;
}

void MixGraphMerger::PerformMerge(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) return;
    int root = actualGroup[0];
    for (size_t i = 1; i < actualGroup.size(); ++i) {
        UnionSets(root, actualGroup[i]);
    }
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
        APASS_LOG_INFO_F(Elements::Operation, "Invalid input parameters");
        return mOutput;
    }
    Initialize(input);
    const int mergeLoopNum = 5;
    for (int mergeLoopStep = 0; mergeLoopStep < mergeLoopNum; mergeLoopStep++) {
        APASS_LOG_INFO_F(Elements::Operation, "Enter merge loop %d.", mergeLoopStep);
        bool hasUpdated = false;
        for (size_t i = 0; i < input.mergeGroup.size(); ++i) {
            const auto& group = input.mergeGroup[i];
            std::vector<int> actualGroup = GetActualGroup(group);
            if (actualGroup.size() <= 1) {
                APASS_LOG_INFO_F(Elements::Operation, "Skip enforce merge group %zu: already merged", i);
                continue;
            }
            if ((input.isEnforceMergeGroup[i] && CanMergeWithoutCycle(actualGroup)) ||
                (mergeLoopStep != 0 && CanMergeWithConstraints(actualGroup))) {
                APASS_LOG_INFO_F(Elements::Operation, "Merge group %zu succeeded", i);
                PerformMerge(actualGroup);
                hasUpdated = true;
            } else if (input.isEnforceMergeGroup[i] || mergeLoopStep != 0) {
                APASS_LOG_INFO_F(Elements::Operation, "Merge group %zu skipped due to constraints.", i);
            }
        }
        if (mergeLoopStep > 0 && !hasUpdated) {
            break;
        }
    }
    UpdateOutput();
    if (!ValidateOutput(mOutput, input.numSubgraph)) {
        APASS_LOG_INFO_F(Elements::Operation, "Invalid output detected");
    }
    return mOutput;
}

}