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
#include "interface/utils/log.h"
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

#define MODULE_NAME "ReduceCopy"

namespace npu::tile_fwk {

Status ReduceCopyMerge::RunOnFunction(Function &function) {
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip ReduceCopy Pass.");
        return SUCCESS;
    }
    ReduceCopyRunner runner;
    const double lowerBound = 0.1;
    const double upperBound = 10.0;
    runner.mergeThresholds = {{lowerBound, upperBound}};
    runner.upperBound = function.paramConfigs_.sgPgUpperBound;
    if (runner.ReduceCopy(function) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status ReduceCopyMerge::PostCheck(Function &function) {
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

DSU::DSU(int n, const std::vector<int> &nodeWeights, std::vector<OpCoreType> &colorCoreType) {
    parent.resize(n);
    std::iota(parent.begin(), parent.end(), 0);
    AIVSupernodeWeights.resize(nodeWeights.size());
    AICSupernodeWeights.resize(nodeWeights.size());
    AIVSingleWeights.resize(nodeWeights.size());
    AICSingleWeights.resize(nodeWeights.size());
    coreType = colorCoreType;
    for (size_t i = 0; i < nodeWeights.size(); i++) {
        if (colorCoreType[i] == OpCoreType::AIC) {
            AIVSupernodeWeights[i] = 0;
            AICSupernodeWeights[i] = nodeWeights[i];
            AIVSingleWeights[i] = 0;
            AICSingleWeights[i] = nodeWeights[i];
        } else {
            AIVSupernodeWeights[i] = nodeWeights[i];
            AICSupernodeWeights[i] = 0;
            AIVSingleWeights[i] = nodeWeights[i];
            AICSingleWeights[i] = 0;
        }
    }
}

int DSU::Find(int i) {
    if (parent[i] == i) {
        return i;
    }
    return parent[i] = Find(parent[i]);
}

void DSU::Union(int i, int j) {
    int rootOfI = Find(i);
    int rootOfJ = Find(j);
    if (rootOfI != rootOfJ) {
        if (rootOfI < rootOfJ) {
            parent[rootOfJ] = rootOfI;
            AIVSupernodeWeights[rootOfI] += AIVSupernodeWeights[rootOfJ];
            AICSupernodeWeights[rootOfI] += AICSupernodeWeights[rootOfJ];
        } else {
            parent[rootOfI] = rootOfJ;
            AIVSupernodeWeights[rootOfJ] += AIVSupernodeWeights[rootOfI];
            AICSupernodeWeights[rootOfJ] += AICSupernodeWeights[rootOfI];
        }
    }
}

std::pair<int, int> DSU::GetWeight(int i) {
    int root = Find(i);
    return {AIVSupernodeWeights[root], AICSupernodeWeights[root]};
}

void DSU::ResetLink(int i) {
    parent[i] = i;
    AIVSupernodeWeights[i] = AIVSingleWeights[i];
    AICSupernodeWeights[i] = AICSingleWeights[i];
}

inline bool PathExistsDFS(int uDense, int targetDense, const std::vector<std::set<int>> &adj,
        std::vector<bool> &visited, int startNodeDense, int ignoredNeighbor) {
    visited[uDense] = true;
    for (int vDense : adj[uDense]) {
        if (uDense == startNodeDense && vDense == ignoredNeighbor) {
            continue;
        }
        if (vDense == targetDense) {
            return true;
        }
        if(!visited[vDense]) {
            if (PathExistsDFS(vDense, targetDense, adj, visited, startNodeDense, ignoredNeighbor)) {
                return true;
            }
        }
    }
    return false;
}

inline bool NoLoopDetected(int uDense, int vDense, const std::vector<std::set<int>> &adj) {
    int supernodeNum = adj.size();
    std::vector<bool> visited(supernodeNum, false);
    if (PathExistsDFS(uDense, vDense, adj, visited, uDense, vDense)) {
        return false;
    }
    return true;
}

inline bool IsMixGraph(int AIVLatency, int AICLatency) {
    return AIVLatency > 0 && AICLatency > 0;
}

inline bool IsValidMixGraph(int AIVLatency, int AICLatency, double aivFactorLowerbound, double aivFactorUpperbound) {
    if (AICLatency <=0) {
        return false;
    } else {
        double aivFactor = static_cast<double>(AIVLatency) / AICLatency;
        return aivFactor >= aivFactorLowerbound && aivFactor <= aivFactorUpperbound;
    }
}

inline bool IsPairMergeable(DSU &dsu, int uRoot, int vRoot, int upperBound, double thresLower, double thresUpper) {
    if (dsu.coreType[uRoot] == OpCoreType::AICPU || dsu.coreType[vRoot] == OpCoreType::AICPU) {
        return false;
    }
    std::pair<int, int> uWeight = dsu.GetWeight(uRoot);
    std::pair<int, int> vWeight = dsu.GetWeight(vRoot);
    int AIVbefore1 = uWeight.first;
    int AIVbefore2 = vWeight.first;
    int AICbefore1 = uWeight.second;
    int AICbefore2 = vWeight.second;
    int AIVafter = AIVbefore1 + AIVbefore2;
    int AICafter = AICbefore1 + AICbefore2;
    if (AICbefore1 == 0 && AICbefore2 == 0) {
        return false;
    }
    if (AICbefore1 > 0 && AICbefore2 > 0) {
        return false;
    }
    if (AIVafter > upperBound || AICafter > upperBound) {
        return false;
    }
    if ((IsValidMixGraph(AIVbefore1, AICbefore1, thresLower, thresUpper) || IsValidMixGraph(AIVbefore2, AICbefore2, thresLower, thresUpper)) &&
            !IsValidMixGraph(AIVafter, AICafter, thresLower, thresUpper)) {
        return false;            
    }
    return true;
}

inline void UpdateDSUForLowerBound(DSU &dsu, std::unordered_set<int> &updatedGraphId, const std::pair<double, double> &thres) {
    APASS_LOG_INFO_F(Elements::Operation, "Checking mix result: AIV threshold lowerbound=%f, upperbound=%f.", thres.first, thres.second);
    std::unordered_set<int> cancelMergeRootColor;
    std::unordered_set<int> visitedRootColor;
    for (int i : updatedGraphId) {
        auto rootColor = dsu.Find(i);
        if (visitedRootColor.count(rootColor) > 0) {
            continue;
        }
        visitedRootColor.insert(rootColor);
        std::pair<int, int> weight = dsu.GetWeight(rootColor);
        int AIVweight = weight.first;
        int AICweight = weight.second;
        if (!IsMixGraph(AIVweight, AICweight)) {
            continue;
        }
        if (!IsValidMixGraph(AIVweight, AICweight, thres.first, thres.second)) {
            APASS_LOG_INFO_F(Elements::Operation, "Found invalid mixGraph: AIC Latency=%d, AIV Latency=%d.", AICweight, AIVweight);
            cancelMergeRootColor.insert(rootColor);
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "Found valid mixGraph: AIC Latency=%d, AIV Latency=%d.", AICweight, AIVweight);
    }
    std::vector<int> cancelMergeColor;
    for (int i : updatedGraphId) {
        auto rootColor = dsu.Find(i);
        if (cancelMergeRootColor.count(rootColor) > 0) {
            cancelMergeColor.push_back(i);
        }
    }
    for (int cancelColor : cancelMergeColor) {
        dsu.ResetLink(cancelColor);
    }
}

inline bool isCrossTensor(LogicalTensorPtr tensor) {
    std::unordered_set<int> inOutSubgraph;
    for (auto &parentOpPtr : tensor->GetProducers()) {
        auto producerColor = parentOpPtr->GetSubgraphID();
        inOutSubgraph.insert(producerColor);
    }
    for (auto &childOpPtr : tensor->GetConsumers()) {
        auto consumerColor = childOpPtr->GetSubgraphID();
        inOutSubgraph.insert(consumerColor);
    }
    const int singleLinkNum = 2;
    if (inOutSubgraph.size() > singleLinkNum) {
        return true;
    }
    return false;
}

void ReduceCopyRunner::BuildGraphInner(const OperationsViewer &opOriList, int opIdx, int opColor) {
    for (auto tensor : opOriList[opIdx].GetIOperands()) {
        for (auto &parentOpPtr : tensor->GetProducers()) {
            auto producerColor = parentOpPtr->GetSubgraphID();
            if (producerColor == opColor || producerColor == -1) {
                continue;
            }
            originalEdges[std::make_pair(producerColor, opColor)].insert(tensor->magic);
            magic2Size[tensor->magic] = tensor->MemorySize();
            if (isCrossTensor(tensor)) {
                crossEdges.insert(std::make_pair(producerColor, opColor));
            }
        }
    }
}

void ReduceCopyRunner::BuildGraph(const OperationsViewer opOriList) {
    for (size_t i = 0; i < colorNode.size(); i++) {
        for (size_t opIdx : colorNode[i]) {
            auto opColor = opOriList[opIdx].GetSubgraphID();
            BuildGraphInner(opOriList, opIdx, opColor);
        }
    }
}

inline void GetCoreType(const OperationsViewer opOriList, std::vector<std::vector<size_t>> colorNode, std::map<std::pair<int,int>, std::set<int>>& originalEdges,
        std::vector<OpCoreType>& colorCoreType, std::vector<bool>& isReshape) {
    std::vector<std::vector<int>> colorInGraph(colorNode.size()), colorOutGraph(colorNode.size());
    for (const auto& edge : originalEdges) {
        int u = std::get<0>(edge.first);
        int v = std::get<1>(edge.first);
        if (u != v) {
            colorOutGraph[u].push_back(v);
            colorInGraph[v].push_back(u);
        }
    }
    for (size_t i = 0;  i< colorNode.size(); i++) {
        colorCoreType[i] = OpCoreType::AIV;
        for (int32_t opIdx : colorNode[i]) {
            auto coreType = OpcodeManager::Inst().GetCoreType(opOriList[opIdx].GetOpcode());
            if (coreType != OpCoreType::ANY) {
                colorCoreType[i] = coreType;
                break;
            }
        }
        isReshape[i] = (colorNode[i].size() == 1 && 
                        opOriList[colorNode[i][0]].GetOpcode() == Opcode::OP_RESHAPE);
    }
}

Status ReduceCopyRunner::RemarkInternalSubgraphID(Function &func) {
    auto opOriList = func.Operations();
    std::map<int, int> newColormap;
    newColormap[0] = 0;
    int currColor = 0;
    for (int i=0; i<color; i++) {
        auto newColor = dsu.Find(i);
        if (newColor == i) {
            newColormap[i] = currColor;
            currColor += 1;
        } else {
            newColormap[i] = newColormap[newColor];
        }
        for (size_t j : colorNode[i]) {
            opOriList[j].UpdateSubgraphID(newColormap[i]);
        }
    }
    func.SetTotalSubGraphCount(currColor);
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph num: %d -> %d.", color, currColor);
    return SUCCESS;
}

Status ReduceCopyRunner::Init(Function &func) {
    for (auto &op : func.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_F(Elements::Config, "Op %d does not belong to any subgraph before ReduceCopy.", op.GetOpMagic());
            return FAILED;
        }
    }
    int colorMax{0};
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        auto opColor = opOriList[i].GetSubgraphID();
        if (opColor > colorMax) {
            colorMax = opColor;
        }
    }
    color = colorMax + 1;
    colorNode.resize(color);
    std::vector<int> nodeWeights(color);
    for (size_t i = 0; i < opOriList.size(); i++) {
        auto opColor = opOriList[i].GetSubgraphID();
        colorNode[opColor].push_back(i);
        nodeWeights[opColor] += opOriList[i].GetLatency();
    }
    BuildGraph(opOriList);
    colorCoreType.resize(color);
    isReshape.resize(color);
    GetCoreType(opOriList, colorNode, originalEdges, colorCoreType, isReshape);

    dsu = DSU(color, nodeWeights, colorCoreType);
    return SUCCESS;
}

Status ReduceCopyRunner::MergePrepare(std::vector<std::tuple<int, int, size_t>> &candidates, 
        std::map<int, int> &rootToDense) {
    std::set<int> activeRootSet;
    for (int i=0; i< color; i++) {
        activeRootSet.insert(dsu.Find(i));
    }
    std::vector<int> activeRootVec(activeRootSet.begin(), activeRootSet.end());
    for (size_t i = 0; i < activeRootVec.size(); i++) {
        rootToDense[activeRootVec[i]] = i;
    }
    superNodeOutGraph.clear();
    superNodeOutGraph.resize(activeRootVec.size());
    superNodeInGraph.clear();
    superNodeInGraph.resize(activeRootVec.size());
    std::map<std::pair<int, int>, std::set<int>> superGraphEdges;
    for (const auto &edge : originalEdges) {
        int uRoot = dsu.Find(std::get<0>(edge.first));
        int vRoot = dsu.Find(std::get<1>(edge.first));
        if (uRoot != vRoot) {
            int uDense = rootToDense[uRoot];
            int vDense = rootToDense[vRoot];
            superNodeOutGraph[uDense].insert(vDense);
            superNodeInGraph[vDense].insert(uDense);
        }
        if (uRoot != vRoot && crossEdges.count(edge.first) == 0) {
            superGraphEdges[{uRoot, vRoot}].insert(edge.second.begin(), edge.second.end());
        }
    }
    for (const auto& edge : superGraphEdges) {
        int uRoot = std::get<0>(edge.first);
        int vRoot = std::get<1>(edge.first);
        size_t totalSize = 0;
        for (auto i : edge.second) {
            totalSize += magic2Size[i];
        }
        candidates.emplace_back(uRoot, vRoot, totalSize);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        const int sizeIdx = 2;
        return std::get<sizeIdx>(a) > std::get<sizeIdx>(b);
    });

    return SUCCESS;
}

Status ReduceCopyRunner::MergeLoop(std::vector<std::tuple<int, int, size_t>> &candidates, const std::pair<double, double> &thres,
    bool &mergedInLoop, std::map<int, int> &rootToDense) {
    for (const auto& candidate : candidates) {
        int uRoot = dsu.Find(std::get<0>(candidate));
        int vRoot = dsu.Find(std::get<1>(candidate));
        if (uRoot == vRoot) {
            continue;
        }
        std::set<OpCoreType> coreTypes{colorCoreType[uRoot], colorCoreType[vRoot]};
        if (isReshape[uRoot] || isReshape[vRoot]) {
            continue;
        }
        int uDense = rootToDense[uRoot];
        int vDense = rootToDense[vRoot];
        if (mergedGraphId.count(uDense) > 0 || mergedGraphId.count(vDense) > 0) {
            continue;
        }
        if (!IsPairMergeable(dsu, uRoot, vRoot, upperBound, thres.first, thres.second)) {
            continue;
        }
        if (!NoLoopDetected(uDense, vDense, superNodeOutGraph)) {
            continue;
        }
        dsu.Union(uRoot, vRoot);
        currMergedGraphId.insert(uRoot);
        currMergedGraphId.insert(vRoot);
        uDense = rootToDense[std::min(uRoot, vRoot)];
        vDense = rootToDense[std::max(uRoot, vRoot)];
        superNodeOutGraph[uDense].insert(superNodeOutGraph[vDense].begin(), superNodeOutGraph[vDense].end());
        superNodeOutGraph[uDense].erase(vDense);
        superNodeInGraph[uDense].insert(superNodeInGraph[vDense].begin(), superNodeInGraph[vDense].end());
        superNodeInGraph[uDense].erase(vDense);
        for (int i : superNodeOutGraph[uDense]) {
            superNodeInGraph[i].insert(uDense);
        }
        for (int i : superNodeInGraph[uDense]) {
            superNodeOutGraph[i].insert(uDense);
        }
        mergedInLoop = true;
    }
    return SUCCESS;
}

Status ReduceCopyRunner::ReduceCopy(Function &func) {
    APASS_LOG_INFO_F(Elements::Operation, "===> Start ReduceCopy.");
    if (Init(func) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot initialize ReduceCopy.");
        return FAILED;
    }
    for (size_t mergeThresId = 0; mergeThresId < mergeThresholds.size(); mergeThresId++) {
        bool mergedInLoop = true;
        int outerLoopTimes = 0;
        const int loopNumBound = 3;
        currMergedGraphId.clear();
        while (mergedInLoop && outerLoopTimes < loopNumBound) {
            mergedInLoop = false;
            superNodeInGraph.clear();
            superNodeOutGraph.clear();
            std::vector<std::tuple<int, int, size_t>> candidates;
            std::map<int, int> rootToDense;
            if (MergePrepare(candidates, rootToDense) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Prepare for mix graph failed.");
                return FAILED;
            }
            if (MergeLoop(candidates, mergeThresholds[mergeThresId], mergedInLoop, rootToDense) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Merge aic and aiv graph failed.");
                return FAILED;
            }
        }
        UpdateDSUForLowerBound(dsu, currMergedGraphId, mergeThresholds[mergeThresId]);
        for (int i = 0; i < color; i ++) {
            auto newColor = dsu.Find(i);
            if (newColor != i) {
                mergedGraphId.insert(newColor);
            }
        }
    }
    RemarkInternalSubgraphID(func);
    APASS_LOG_INFO_F(Elements::Operation, "===> Finish ReduceCopy.");
    return SUCCESS;
}

}