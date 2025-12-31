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
 * \file sort_ops.cpp
 * \brief
 */

#include "scheduler.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

Status DFSVisit(std::unordered_set<int> &visited, std::vector<int32_t> &tasks, std::vector<int> &visitEntrySeq,
        std::vector<std::set<int32_t>> &entryInGraph)
{
    while (tasks.size() > 0) {
        int currTask = tasks.back();
        if (visited.count(currTask) > 0) {
            tasks.pop_back();
            continue;
        }
        bool allVisited = true;
        for (auto prevIdx : entryInGraph[currTask]) {
            if (visited.count(prevIdx) == 0) {
                allVisited = false;
                tasks.push_back(prevIdx);
            }
        }
        if (allVisited) {
            visited.insert(currTask);
            visitEntrySeq.push_back(currTask);
            tasks.pop_back();
        }
    }
    return SUCCESS;
}

std::vector<int32_t> GetLayerTasks(std::map<int, std::set<int>> &depthToEntries,
                                   std::vector<std::set<int>> &entryOutGraph, int lastDepth, int currDepth)
{
    std::vector<int32_t> tasks;
    for (int dp = lastDepth; dp < currDepth; dp++) {
        for (int entryIdx : depthToEntries[dp]) {
            if (entryOutGraph[dp].size() == 0) {
                tasks.push_back(entryIdx);
            }
        }
    }
    for (int entryIdx : depthToEntries[currDepth]) {
        tasks.push_back(entryIdx);
    }
    return tasks;
}

Status EntryInOutGraph(std::vector<IssueEntryPtr> &issueEntries, std::vector<std::set<int>> &entryInGraph,
    std::vector<std::set<int>> &entryOutGraph, std::unordered_map<int, IssueEntryPtr> issueEntryMap) {
    entryInGraph.clear();
    entryOutGraph.clear();
    entryInGraph.resize(issueEntries.size());
    entryOutGraph.resize(issueEntries.size());
    std::unordered_map<IssueEntry*, int> entryPtr2Idx;
    for (int ptrIdx = 0; ptrIdx < static_cast<int>(issueEntries.size()); ptrIdx++) {
        entryPtr2Idx[issueEntries[ptrIdx].get()] = ptrIdx;
    }
    for (int ptrIdx = 0; ptrIdx < static_cast<int>(issueEntries.size()); ptrIdx++) {
        for (auto outPtrId : issueEntries[ptrIdx]->successors) {
            auto outPtr = issueEntryMap[outPtrId];
            entryOutGraph[ptrIdx].insert(entryPtr2Idx[outPtr.get()]);
            entryInGraph[entryPtr2Idx[outPtr.get()]].insert(ptrIdx);
        }
    }
    return SUCCESS;
}

Status EntryTopoSort(std::vector<std::set<int32_t>> &entryInGraph, std::vector<std::set<int32_t>> &entryOutGraph,
    std::vector<int32_t> &seqToColor, std::vector<int32_t> &colorToSeq)
{
    seqToColor.clear();
    colorToSeq.resize(entryInGraph.size());
    std::vector<int> inLinkNum(entryInGraph.size());
    std::deque<int> zeroInLinkColor;
    for (size_t i = 0; i < entryInGraph.size(); i++) {
        inLinkNum[i] = entryInGraph[i].size();
        if (inLinkNum[i] == 0) {
            zeroInLinkColor.push_back(i);
        }
    }
    std::vector<int32_t> visitOrder;
    while (zeroInLinkColor.size() > 0) {
        int currColor = zeroInLinkColor.front();
        zeroInLinkColor.pop_front();
        colorToSeq[currColor] = seqToColor.size();
        seqToColor.push_back(currColor);
        for (int consumerColor : entryOutGraph[currColor]) {
            inLinkNum[consumerColor] -= 1;
            if (inLinkNum[consumerColor] == 0) {
                zeroInLinkColor.push_back(consumerColor);
            }
        }
    }
    return SUCCESS;
}

Status OutputFixBasedDepth(std::vector<int32_t> &depth, std::vector<std::set<int32_t>> &entryInGraph,
        std::vector<std::set<int32_t>> &entryOutGraph, std::vector<int32_t> &seqToColor)
{
    for (int idx = static_cast<int>(entryInGraph.size())-1; idx >= 0; idx--) {
        int currEntryIdx = seqToColor[idx];
        if (entryOutGraph[currEntryIdx].size() == 0) {
            continue;
        }
        int minDepth = static_cast<int>(entryInGraph.size()) + 1;
        for (auto succIdx : entryOutGraph[currEntryIdx]) {
            minDepth = minDepth < depth[succIdx] ? minDepth : depth[succIdx];
        }
        depth[currEntryIdx] = minDepth - 1;
    }
    return SUCCESS;
}

Status InputFixBasedDepth(std::vector<int32_t> &depth, std::vector<std::set<int32_t>> &entryInGraph,
        std::vector<std::set<int32_t>> &entryOutGraph, std::vector<int32_t> &seqToColor)
{
    (void)entryOutGraph;
    for (int idx = 0; idx < static_cast<int>(entryInGraph.size()); idx++) {
        int currEntryIdx = seqToColor[idx];
        if (entryInGraph[currEntryIdx].size() == 0) {
            continue;
        }
        int maxDepth = -static_cast<int>(entryInGraph.size()) - 1;
        for (auto predIdx : entryInGraph[currEntryIdx]) {
            maxDepth = maxDepth > depth[predIdx] ? maxDepth : depth[predIdx];
        }
        depth[currEntryIdx] = maxDepth + 1;
    }
    return SUCCESS;
}

Status OoOScheduler::LayerBasedDFS(int layerDepth)
{
    std::vector<std::set<int>> entryInGraph;
    std::vector<std::set<int>> entryOutGraph;
    EntryInOutGraph(issueEntries, entryInGraph, entryOutGraph, issueEntryMap);
    std::vector<int32_t> seqToColor;
    std::vector<int32_t> colorToSeq;
    EntryTopoSort(entryInGraph, entryOutGraph, seqToColor, colorToSeq);
    std::vector<int32_t> depth(entryInGraph.size(), 0);
    OutputFixBasedDepth(depth, entryInGraph, entryOutGraph, seqToColor);
    InputFixBasedDepth(depth, entryInGraph, entryOutGraph, seqToColor);
    std::map<int, std::set<int>> depthToEntries;
    int lowerDepth = static_cast<int>(entryInGraph.size()) + 1;
    int upperDepth = -static_cast<int>(entryInGraph.size()) - 1;
    for (int idx = 0; idx < static_cast<int>(depth.size()); idx++) {
        lowerDepth = lowerDepth < depth[idx] ? lowerDepth : depth[idx];
        upperDepth = upperDepth > depth[idx] ? upperDepth : depth[idx];
        depthToEntries[depth[idx]].insert(idx);
    }
    std::vector<IssueEntryPtr> newIssueEntries;
    std::unordered_set<int> visited;
    std::vector<int> visitEntrySeq;
    int lastDepth = lowerDepth;
    int currDepth = lowerDepth + layerDepth - 1;
    currDepth = currDepth <= upperDepth ? currDepth : upperDepth;
    bool keepVisit = true;
    while (keepVisit) {
        std::vector<int32_t> tasks = GetLayerTasks(depthToEntries, entryOutGraph, lastDepth, currDepth);
        DFSVisit(visited, tasks, visitEntrySeq, entryInGraph);
        if (currDepth == upperDepth) {
            keepVisit = false;
        }
        lastDepth = currDepth;
        currDepth += layerDepth;
        currDepth = currDepth <= upperDepth ? currDepth : upperDepth;
    }
    for (auto idx : visitEntrySeq) {
        newIssueEntries.push_back(issueEntries[idx]);
    }
    issueEntries = newIssueEntries;
    return SUCCESS;
}

void OoOScheduler::UpdatePreNodeQueue(std::unordered_set<IssueEntryPtr> &curr,
    std::unordered_set<IssueEntryPtr> &preNodeTotal, std::map<IssueEntryPtr, bool>& visited) {
    std::unordered_set<IssueEntryPtr> next;
    for (auto& curIssue : curr) {
        for (auto& preIssueId : curIssue->predecessors) {
            auto preIssue = issueEntryMap[preIssueId];
            if (!visited[preIssue] && preNodeTotal.find(preIssue) == preNodeTotal.end()) {
                next.insert(preIssue);
            }
        }
    }
    for (auto& nextIssue : next) {
        preNodeTotal.insert(nextIssue);
    }
    curr.swap(next);
}

int OoOScheduler::GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited) {
    std::unordered_set<IssueEntryPtr> preNodeTotal;
    std::unordered_set<IssueEntryPtr> curr;
    for (auto& preIssueId : issue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            curr.insert(preIssue);
            preNodeTotal.insert(preIssue);
        }
    }
    while (!curr.empty()) {
        UpdatePreNodeQueue(curr, preNodeTotal, visited);
    }
    return preNodeTotal.size();
}

IssueEntryPtr OoOScheduler::FindNodeMinNumUnvisitedPreNode(
    std::map<IssueEntryPtr, bool> visited, std::vector<IssueEntryPtr> outNodeQueue) {
    IssueEntryPtr res = nullptr;
    int minUnvisitedNode = INT_MAX;
    for (auto& outNode : outNodeQueue) {
        if (visited[outNode]) {
            continue;
        }
        int curUnvisitedNode = GetNumUnvisitPreNode(outNode, visited);
        if (curUnvisitedNode < minUnvisitedNode) {
            res = outNode;
            minUnvisitedNode = curUnvisitedNode;
        }
    }
    return res;
}

int OoOScheduler::GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, IssueEntryPtr issue) {
    int prior = 10;
    if (preNodePriority.find(issue->tileOp.GetOpcode()) != preNodePriority.end()) {
        prior = preNodePriority[issue->tileOp.GetOpcode()];
    }
    return prior;
}

int OoOScheduler::GetDepth(IssueEntryPtr issue) {
    auto it = depthCache_.find(issue);
    if (it != depthCache_.end()) {
        return it->second;
    }

    int maxDepth = 0;
    for (const auto& preId : issue->predecessors) {
        auto preIssue = issueEntryMap[preId];
        maxDepth = std::max(maxDepth, GetDepth(preIssue));
    }

    int depth = maxDepth + 1;
    depthCache_[issue] = depth;
    return depth;
}

void OoOScheduler::QueueNotReadyPreNode(IssueEntryPtr curIssue, std::map<IssueEntryPtr, bool>& visited,
    std::unordered_map<Opcode, int> preNodePriority, std::deque<IssueEntryPtr> &queue) {
    std::vector<IssueEntryPtr> notReadyPreNode;
    for (auto& preIssueId : curIssue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            notReadyPreNode.push_back(preIssue);
        }
    }
    std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](IssueEntryPtr a, IssueEntryPtr b) {
        int priorA = GetNodePriority(preNodePriority, a);
        int priorB = GetNodePriority(preNodePriority, b);
        if (priorA != priorB) {
            return priorA < priorB;
        } else {
            int depA = GetDepth(a);
            int depB = GetDepth(b);
            if (depA == depB) {
                return a->execOrder < b->execOrder;
            }
            return depA < depB;
        }
    });
    for (auto& preIssue : notReadyPreNode) {
        queue.push_front(preIssue);
    }
}

void OoOScheduler::ForwardDfs(IssueEntryPtr curIssue, std::vector<IssueEntryPtr>& newIssueEntries,
    std::map<IssueEntryPtr, bool>& visited, std::unordered_map<Opcode, int> preNodePriority,
    std::deque<IssueEntryPtr> &queue) {
    bool ready = true;
    for (auto& preIssueId : curIssue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            ready = false;
            break;
        }
    }

    if (ready) {
        visited[curIssue] = true;
        queue.pop_front();
        newIssueEntries.push_back(curIssue);
    } else {
        QueueNotReadyPreNode(curIssue, visited, preNodePriority, queue);
    }
}

void OoOScheduler::DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
    std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority) {
    if (visited[issue]) {
        return;
    }

    std::deque<IssueEntryPtr> queue = {issue};
    while (!queue.empty()) {
        auto curIssue = queue.front();
        if (visited[curIssue]) {
            queue.pop_front();
            continue;
        }

        ForwardDfs(curIssue, newIssueEntries, visited, preNodePriority, queue);
    }
}

Status OoOScheduler::DFSFromOutNode(std::vector<IssueEntryPtr> outNodeQueue,
    std::unordered_map<Opcode, int> preNodePriority, std::map<IssueEntryPtr, bool> &visited) {
    std::vector<IssueEntryPtr> newIssueEntries;
    if (outNodeQueue.size() != 0) {
       DFSFromSingleNode(outNodeQueue[0], visited, newIssueEntries, preNodePriority);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Subgraph must have operation with outdegree 0.");
        return FAILED;
    }

    for (size_t i = 1; i < outNodeQueue.size(); i++) {
        while (!visited[outNodeQueue[i]]) {
            auto curNode = outNodeQueue[i];
            auto node = FindNodeMinNumUnvisitedPreNode(visited, outNodeQueue);
            if (node == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "FindNodeMinNumUnvisitedPreNode failed.");
                return FAILED;
            }
            DFSFromSingleNode(node, visited, newIssueEntries, preNodePriority);
        }
    }
    issueEntries = newIssueEntries;
    return SUCCESS;
}

Status OoOScheduler::PriorDFS(std::unordered_map<Opcode, int> preNodePriority) {
    std::map<IssueEntryPtr, bool> visited;
    std::vector<IssueEntryPtr> outNodeQueue;
    for (auto &issue : issueEntries) {
        visited[issue] = false;
        if (issue->successors.empty()) {
            outNodeQueue.push_back(issue);
        }
    }

    if (DFSFromOutNode(outNodeQueue, preNodePriority, visited) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "DFSFromOutNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::GetIssueIdx(IssueEntryPtr issue, size_t &index) {
    for (auto [idx, node] : issueEntryMap) {
        if (node == issue) {
            index = idx;
        }
    }
}

// rollBackIssue 和 backTraceIssue 是否存在前后序依赖
bool OoOScheduler::HasDependency(IssueEntryPtr rollBackIssue,  IssueEntryPtr backIssue) {
    size_t n = issueEntries.size();
    std::vector<bool> visited(n, false);
    size_t start;
    size_t target;
    GetIssueIdx(rollBackIssue, start);
    GetIssueIdx(backIssue, target);
    std::function<bool(size_t)> dfs = [&](size_t node) ->bool{
        if (node == target) return true;
        if (visited[node]) return false;

        visited[node] = true;
        for (auto succId : issueEntryMap[node]->successors) {
            if (dfs(succId)) {
                return true;
            }
        }
        return false;
    };
    return dfs(start);
}

// 在 curIssueEntries 中将 advanceIndexList 中的序列提前到 rollBackIndex 之前,更新 curIssueEntries
void OoOScheduler::ReplaceIndex(std::vector<IssueEntryPtr> &curIssueEntries,
    std::set<size_t> advanceIndexList, size_t rollBackIndex) {
    std::vector<IssueEntryPtr> moveIssueEntries;
    for (auto i : advanceIndexList) {
        APASS_LOG_DEBUG_F(Elements::Operation, "advance index: %d, issue: %s",
                i, curIssueEntries[i]->GetOpInfo().c_str());
        moveIssueEntries.push_back(curIssueEntries[i]);
    }
    for (auto it = advanceIndexList.rbegin(); it != advanceIndexList.rend(); ++it) {
        curIssueEntries.erase(curIssueEntries.begin() + (*it));
    }
    curIssueEntries.insert(curIssueEntries.begin() + rollBackIndex, moveIssueEntries.begin(), moveIssueEntries.end());
}

void OoOScheduler::GetPreNode(size_t i, std::vector<IssueEntryPtr> curIssueEntries, size_t rollBackIndex,
    size_t backTraceIndex, std::set<size_t> &dependencyIndexList) {
    dependencyIndexList.insert(i);
    APASS_LOG_DEBUG_F(Elements::Operation, "dependencyIndexList push index: %d, issue: %s",
        i, curIssueEntries[i]->GetOpInfo().c_str());
    for (auto preId : curIssueEntries[i]->predecessors) {
        auto issue = issueEntryMap[preId];
        auto it = std::find(curIssueEntries.begin() + rollBackIndex + 1, curIssueEntries.begin() + backTraceIndex, issue);
        if (it != curIssueEntries.begin() + backTraceIndex) {
            auto index = std::distance(curIssueEntries.begin(), it);
            GetPreNode(index, curIssueEntries, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
}

// 记录 curIssueEntries 中从 rollBackIndex 到 backTraceIndex 中所有和 rollBack 没有后继依赖的点
void OoOScheduler::GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
    std::vector<IssueEntryPtr> curIssueEntries, std::set<size_t> &advanceIndexList) {
    std::set<size_t> dependencyIndexList;
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (HasDependency(curIssueEntries[rollBackIndex], curIssueEntries[i])) {
            GetPreNode(i, curIssueEntries, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (dependencyIndexList.count(i) == 0) {
            advanceIndexList.insert(i);
            APASS_LOG_DEBUG_F(Elements::Operation, "advanceIndexList push index: %d, issue: %s",
                i, curIssueEntries[i]->GetOpInfo().c_str());
        }
    }
}

// curBackTrace 位置回退
Status OoOScheduler::RollBack(size_t &startIndex,
    std::vector<IssueEntryPtr> &curIssueEntries, std::map<MemoryType, int64_t> &curMemoryMap) {
    APASS_LOG_DEBUG_F(Elements::Operation, "=====> Start RollBack.");
    curIssueEntries = backTraceIssueEntries[backTraceIssue].second;
    MemoryType memType = recordIssueBuffer[backTraceIssue];
    size_t backTraceIndex = backTraceIssueEntries[backTraceIssue].first + 1;
    backTraceIssue = curIssueEntries[backTraceIndex];
    size_t rollBackIndex = backTraceIndex;
    APASS_LOG_DEBUG_F(Elements::Operation, "backTraceIssue: %s, backTraceIndex: %d, memType: %d",
        backTraceIssue->GetOpInfo().c_str(), backTraceIndex, memType);
    while (rollBackIndex < curIssueEntries.size() && rollBackIndex > 0) {
        rollBackIndex--;
        IssueEntryPtr rollBackIssue = curIssueEntries[rollBackIndex];
        if (recordIssueBuffer[rollBackIssue] != memType || !(rollBackIssue->isAlloc) || HasDependency(rollBackIssue, backTraceIssue)) {
            continue;
        }
        rollBackNodeIssue = rollBackIssue;
        APASS_LOG_DEBUG_F(Elements::Operation, "Select rollBackIssue: %s, rollBackIndex: %d",
            rollBackIssue->GetOpInfo().c_str(), rollBackIndex);
        recordBufferAllocate = backTraceBufferAllocate;
        recordIssueEntries = backTraceIssueEntries;
        recordBufRefCount = backTraceBufRefCount;
        std::set<size_t> advanceIndexList;
        GetListToAdvance(rollBackIndex, backTraceIndex, curIssueEntries, advanceIndexList);
        ReplaceIndex(curIssueEntries, advanceIndexList, rollBackIndex);
        startIndex = rollBackIndex;
        APASS_LOG_DEBUG_F(Elements::Operation, "RollBack==>change startIndex: %d", startIndex);
        if (rollBackIndex != 0) {
            curMemoryMap = recordBufferAllocate[curIssueEntries[rollBackIndex-1]];
            RecoverSymbol(startIndex - 1, curIssueEntries);
            return SUCCESS;
        }
        curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0}, {MemoryType::MEM_L0C, 0}};
        issueEntries = curIssueEntries;
        for (auto issue : curIssueEntries) {
            visitedIssue[issue] = false;
        }
        InitBufRefCount();
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "RollBack Failed");
    return FAILED;
}

// 在 curIssueEntries 中将 preIssue 中的序列提前到 startIndex 之后，更新 curIssueEntries
void OoOScheduler::ReorderIssue(std::vector<size_t> &preIdx, std::vector<IssueEntryPtr> &curIssueEntries,
    size_t startIndex) {
    // 对 perIssue 排序，再进行插入
    std::sort(preIdx.begin(), preIdx.end());
    std::vector<IssueEntryPtr> moveIssueEntries;
    APASS_LOG_DEBUG_F(Elements::Operation, "current index: %d, preIdx size: %d", startIndex, preIdx.size());
    for (auto i : preIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "preidx : %d, curIssue: %s", i, curIssueEntries[i]->GetOpInfo().c_str());
        moveIssueEntries.push_back(curIssueEntries[i]);
    }
    for (auto it = preIdx.rbegin(); it != preIdx.rend(); ++it) {
        curIssueEntries.erase(curIssueEntries.begin() + (*it));
    }
    curIssueEntries.insert(curIssueEntries.begin() + startIndex + 1, moveIssueEntries.begin(), moveIssueEntries.end());
}

void OoOScheduler::FindIndex(IssueEntryPtr issue, std::vector<IssueEntryPtr> curIssueEntries, size_t &index) {
    for (size_t i = 0; i < curIssueEntries.size(); i++) {
        if (curIssueEntries[i] == issue) {
            index = i;
            return;
        }
    }
}

// 在curIssueEntries中，向前遍历找到consumerIndex的前序未被访问的节点，并放入preIssue中
Status OoOScheduler::FindConsumerList(size_t consumerIndex, std::vector<size_t> &preIssue, std::vector<IssueEntryPtr> &curIssueEntries) {
    if (curIssueEntries[consumerIndex] == backTraceIssue) {
        APASS_LOG_WARN_F(Elements::Operation, "backTraceIssue is one of the predecessor node.");
        return FAILED;
    }
    if (curIssueEntries[consumerIndex] == rollBackNodeIssue) {
        APASS_LOG_WARN_F(Elements::Operation, "rollBackNodeIssue is one of the predecessor node.");
        return FAILED;
    }
    visitedIssue[curIssueEntries[consumerIndex]] = true;
    preIssue.push_back(consumerIndex);
    APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer idx: %d, issue: %s", consumerIndex, curIssueEntries[consumerIndex]->GetOpInfo().c_str());
    for (auto preId : curIssueEntries[consumerIndex]->predecessors) {
        auto issue =  issueEntryMap[preId];
        if (visitedIssue[issue] == false) {
            size_t index;
            FindIndex(issue, curIssueEntries, index);
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer preIdx: %d, issue: %s", index, issue->GetOpInfo().c_str());
            if (FindConsumerList(index, preIssue, curIssueEntries) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// 将 consumersGroup 和其前序依赖按原有顺序放入 preIssue
Status OoOScheduler::UpdateOOperandPreDependence(size_t startIndex, std::vector<IssueEntryPtr> &curIssueEntries,
    std::vector<IssueEntryPtr> consumersGroup) {
    // curIssueEntries 中向后找
    std::vector<size_t> preIssue;
    size_t index = startIndex;
    while (index < curIssueEntries.size()) {
        if (std::find(consumersGroup.begin(), consumersGroup.end(), curIssueEntries[index]) != consumersGroup.end()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer Idx: %d", index);
            if (FindConsumerList(index, preIssue, curIssueEntries) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
        index++;
    }
    ReorderIssue(preIssue, curIssueEntries, startIndex);
    return SUCCESS;
}

// 回溯后，将队列 startIndex 位置之后的 issue 的 visitedIssue 状态还原回 false
void OoOScheduler::RecoverSymbol(size_t startIndex, std::vector<IssueEntryPtr> curIssueEntries) {
    APASS_LOG_DEBUG_F(Elements::Operation, "RecoverSymbol  startIdx: %d, curIssue: %s", startIndex, curIssueEntries[startIndex]->GetOpInfo().c_str());
    bufRefCount = recordBufRefCount[curIssueEntries[startIndex]];
    for (size_t i = 0; i < curIssueEntries.size(); i++) {
        if (i > startIndex) {
            visitedIssue[curIssueEntries[i]] = false;
            continue;
        }
        visitedIssue[curIssueEntries[i]] = true;
    }
}

// 找未被执行的 consumer
void OoOScheduler::GetConsumerGroup(std::vector<IssueEntryPtr> consumers, std::vector<IssueEntryPtr> &consumersGroup) {
    for (auto issue : consumers) {
        if (visitedIssue[issue] == false) {
            consumersGroup.push_back(issue);
            APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer: %s", issue->GetOpInfo().c_str());
        }
    }
}

void OoOScheduler::GetStackTop(size_t &startIndex, std::vector<IssueEntryPtr> &curIssueEntries,
    std::map<MemoryType, int64_t> &curMemoryMap) {
    auto topNode = needFreeIssueStack.top();
    needFreeIssueStack.pop();
    curIssueEntries = recordIssueEntries[topNode.first].second;
    startIndex = recordIssueEntries[topNode.first].first;
    curMemoryMap = recordBufferAllocate[topNode.first];
}

Status OoOScheduler::BacktraceOnMemoryExceeded(size_t &startIndex,
    std::vector<IssueEntryPtr> &curIssueEntries, std::map<MemoryType, int64_t> &curMemoryMap) {
    MemoryType memType = curIssueEntries[startIndex]->tileOp.GetOutputOperand(0)->GetMemoryTypeOriginal();
    while (startIndex < curIssueEntries.size() && startIndex > 0) {
        startIndex--;
        IssueEntryPtr issue = curIssueEntries[startIndex];
        if (!needFreeIssueStack.empty() && needFreeIssueStack.top().first == curIssueEntries[startIndex]) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Having traversed %s, the stack needs to be popped", curIssueEntries[startIndex]->GetOpInfo().c_str());
            break;
        }
        if (recordIssueBuffer[issue] != memType || issue->isAlloc) {
            continue;
        }
        std::vector<IssueEntryPtr> consumers;
        APASS_LOG_DEBUG_F(Elements::Operation, "=====>start to find unvisited consumer, current index： %d", startIndex);
        for (auto succIdx : issue->successors) {
            consumers.push_back(issueEntryMap[succIdx]);
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer: %s", issueEntryMap[succIdx]->GetOpInfo().c_str());
        }
        std::vector<IssueEntryPtr> consumersGroup;
        GetConsumerGroup(consumers, consumersGroup);
        if (consumersGroup.empty()) {
            continue;
        }
        RecoverSymbol(startIndex, curIssueEntries);
        GetConsumerGroup(consumers, consumersGroup);
        APASS_LOG_DEBUG_F(Elements::Operation, "push %s to stack", issue->GetOpInfo().c_str());
        curMemoryMap = recordBufferAllocate[issue];
        needFreeIssueStack.push(make_pair(issue, recordIssueBuffer[issue]));
        if (UpdateOOperandPreDependence(startIndex, curIssueEntries, consumersGroup) != SUCCESS) {
            needFreeIssueStack.pop();
            APASS_LOG_DEBUG_F(Elements::Operation, "UpdateOOperandPreDependence failed.");
            continue;
        }
        startIndex++;
        APASS_LOG_DEBUG_F(Elements::Operation, "Backtrace==>change startIndex: %d", startIndex);
        return SUCCESS;
    }
    if (needFreeIssueStack.empty()) {
        APASS_LOG_WARN_F(Elements::Operation, "Stack is empty. Start to rollBack");
        return FAILED;
    }
    GetStackTop(startIndex, curIssueEntries, curMemoryMap);
    RecoverSymbol(startIndex, curIssueEntries);
    APASS_LOG_DEBUG_F(Elements::Operation, "pop %s from stack", curIssueEntries[startIndex]->GetOpInfo().c_str());
    if (BacktraceOnMemoryExceeded(startIndex, curIssueEntries, curMemoryMap) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "BacktraceOnMemoryExceeded Failed");
        return FAILED;
    }
    return SUCCESS;
}

// 计算 tensor 对应的 memType （只对 L0C L0A L0B 进行内存处理） 是否已满
bool OoOScheduler::IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size) {
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return false;
    }
    if (curMemoryMap[memType] + size > localMemorySize[memType]) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The %d-memType memory is full, current memory: %d, memory to add: %d",
            memType, curMemoryMap[memType], size);
        return true;
    }
    return false;
}

// 修改内存
Status OoOScheduler::ModifyBuffer(std::map<MemoryType, int64_t> &curMemoryMap, MemoryType memType, int64_t size, bool isAdd) {
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return SUCCESS;
    }
    if (isAdd) {
        if (curMemoryMap[memType] + size > localMemorySize[memType]) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to increase memory");
            return FAILED;
        }
        curMemoryMap[memType] = curMemoryMap[memType] + size;
        APASS_LOG_DEBUG_F(Elements::Operation, "Increase %d-memType memory, size: %d, total memory %d", memType, size, curMemoryMap[memType]);
        return SUCCESS;
    }
    if (curMemoryMap[memType] - size < 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to reduce memory");
        return FAILED;
    }
    curMemoryMap[memType] = curMemoryMap[memType] - size;
    APASS_LOG_DEBUG_F(Elements::Operation, "Reduce %d-memType memory, size: %d, total memory %d", memType, size, curMemoryMap[memType]);
    return SUCCESS;
}

// 释放内存
Status OoOScheduler::RetireIssueBuffer(std::map<MemoryType, int64_t> &curMemoryMap, IssueEntryPtr issue) {
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount[memId] == 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Start to free memory:");
            if (ModifyBuffer(curMemoryMap, localBufferMap[memId]->memType, localBufferMap[memId]->size, false) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::issueMemoryUpdate(IssueEntryPtr issue, size_t startIndex, std::vector<IssueEntryPtr> curIssueEntries,
    std::map<MemoryType, int64_t> curMemoryMap) {
    recordIssueEntries[issue] = make_pair(startIndex, curIssueEntries);
    recordBufferAllocate[issue] = curMemoryMap;
    recordIssueBuffer[issue] = issue->tileOp.GetOutputOperand(0)->GetMemoryTypeOriginal();
    recordBufRefCount[issue] = bufRefCount;
}

Status OoOScheduler::AllocExecute(IssueEntryPtr issue, std::vector<IssueEntryPtr> &curIssueEntries,
    std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex, bool &isContinue) {
    APASS_LOG_DEBUG_F(Elements::Operation, "alloc issue: %s", issue->GetOpInfo().c_str());
    auto allocBuffer = localBufferMap[issue->reqMemIds[0]];
    if (IsBufferFull(curMemoryMap, allocBuffer->memType, allocBuffer->size)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The memory of %s needs to be released", std::to_string(allocBuffer->memType).c_str());
        backTraceIssue = curIssueEntries[startIndex];
        backTraceBufferAllocate = recordBufferAllocate;
        backTraceIssueEntries = recordIssueEntries;
        backTraceBufRefCount = recordBufRefCount;
        APASS_LOG_DEBUG_F(Elements::Operation, "backTraceIssue: %s, backTraceIndex: %d, memType: %d",
            backTraceIssue->GetOpInfo().c_str(), backTraceIssueEntries[backTraceIssue].first, recordIssueBuffer[backTraceIssue]);
        APASS_LOG_DEBUG_F(Elements::Operation, "=====> Need backtrace.");
        if (BacktraceOnMemoryExceeded(startIndex, curIssueEntries, curMemoryMap) != SUCCESS) {
            if (RollBack(startIndex, curIssueEntries, curMemoryMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocExecute failed.");
                return FAILED;
            }
            isContinue = true;
            return SUCCESS;
        }
        isContinue = true;
        return SUCCESS;
    }
    return SUCCESS;
}

Status OoOScheduler::IssueEntriesExecute(std::vector<IssueEntryPtr> &curIssueEntries,
    std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex) {
    APASS_LOG_DEBUG_F(Elements::Operation, "===>Start issueEntriesExecute, startIndex: %d", startIndex);
    if (curIssueEntries.empty()) {
        curIssueEntries = issueEntries;
    }
    while (startIndex < curIssueEntries.size()) {
        auto issue = curIssueEntries[startIndex];
        issueMemoryUpdate(issue, startIndex, curIssueEntries, curMemoryMap);
        APASS_LOG_DEBUG_F(Elements::Operation, "execute issue: %s, index: %d", issue->GetOpInfo().c_str(), startIndex);
        if (issue->isAlloc) {
            bool isContinue = false;
            if (AllocExecute(issue, curIssueEntries, curMemoryMap,  startIndex, isContinue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocExecute failed.");
                return FAILED;
            }
            if (isContinue) {
                return SUCCESS;
            }
            auto allocBuffer = localBufferMap[issue->reqMemIds[0]];
            APASS_LOG_DEBUG_F(Elements::Operation, "Start to increase memory:");
            if (ModifyBuffer(curMemoryMap, allocBuffer->memType, allocBuffer->size, true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Allocate tensor[%u] failed.", allocBuffer->id);
                return FAILED;
            }
        }
        visitedIssue[issue] = true;
        issueMemoryUpdate(issue, startIndex, curIssueEntries, curMemoryMap);
        if (RetireIssueBuffer(curMemoryMap, issue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(issue->tileOp).c_str());
            return FAILED;
        }
        issueMemoryUpdate(issue, startIndex, curIssueEntries, curMemoryMap);
        startIndex += 1;
    }
    issueFinish = true;
    return SUCCESS;
}

Status OoOScheduler::ExecuteIssue() {
    std::vector<IssueEntryPtr> curIssueEntries;
    std::map<MemoryType, int64_t> curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}};
    size_t startIndex{0};
    for (auto &issue : issueEntries) {
        visitedIssue[issue] = false;
    }
    while(!issueFinish) {
        if (IssueEntriesExecute(curIssueEntries, curMemoryMap, startIndex) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueEntriesExecute failed.");
            return FAILED;
        }
    }
    issueEntries = curIssueEntries;
    // 初始化修改了的 refcount
    InitBufRefCount();
    if (InitDependencies() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SortOps() {
    std::string sortMethodStr;
    std::string funcName = function_.GetMagicName();

    sortMethodStr = function_.paramConfigs_.OoOPreScheduleMethod;
    if (sortMethodStr == "PriorDFS") {
        std::unordered_map<Opcode, int> preNodePriority = {
            // ALLOC 节点优先级最高，因为一个节点的前序ALLOC节点要在最靠近该节点的地方访问。
            {Opcode::OP_UB_ALLOC, 0}, {Opcode::OP_L1_ALLOC, 0}, {Opcode::OP_L0A_ALLOC, 0}, {Opcode::OP_L0B_ALLOC, 0},
            {Opcode::OP_L0C_ALLOC, 0}, {Opcode::OP_BT_ALLOC, 0}, {Opcode::OP_FIX_ALLOC, 0},
            // 其次是L0级数据搬运Op。
            {Opcode::OP_L1_TO_L0A, 1}, {Opcode::OP_L1_TO_L0B, 1}, {Opcode::OP_L1_TO_L0_AT, 1},
            {Opcode::OP_L1_TO_L0_BT, 1}, {Opcode::OP_L1_TO_FIX, 1}, {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
            {Opcode::OP_L1_TO_FIX_RELU_PRE, 1}, {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
            {Opcode::OP_L1_TO_FIX_QUANT_POST, 1}, {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
            {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1}, {Opcode::OP_L1_TO_BT, 1},
            // 再其次是L1级数据搬运Op。
            {Opcode::OP_COPY_IN, 2}, {Opcode::OP_UB_COPY_IN, 2}, {Opcode::OP_L1_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2}, {Opcode::OP_L1_COPY_UB, 2},
            {Opcode::OP_L0C_COPY_UB, 2}, {Opcode::OP_UB_COPY_L1, 2},
            // 最后访问其它计算节点（其它节点默认的优先级为10）。
        };
        if (PriorDFS(preNodePriority) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PriorDFS failed.");
            return FAILED;
        }
        if (ExecuteIssue() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteIssueEntries failed.");
            return FAILED;
        }
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "PreSchedule method not recognized.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk