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
 * \file core_assign.cpp
 * \brief
 */

#include "core_assign.h"
#include "passes/pass_log/pass_log.h"

#ifndef MODULE_NAME
#define MODULE_NAME "CoreAssign"
#endif

namespace npu::tile_fwk {

constexpr int64_t NEGATIVE_ONE = -1;

inline std::string ScheduleCoreTypeToString(ScheduleCoreType coreType) {
    if (coreType == ScheduleCoreType::AIC) {
        return "AIC";
    }
    if (coreType == ScheduleCoreType::AIV) {
        return "AIV";
    }
    return "UNKNOWN";
}

inline std::string TargetCoreTypeToString(TargetCoreType coreType) {
    std::unordered_map<TargetCoreType, std::string> targetToString{
        {TargetCoreType::AIC, "AIC"},
        {TargetCoreType::AIV0, "AIV0"}, 
        {TargetCoreType::AIV1, "AIV1"}, 
        {TargetCoreType::UNKNOWN, "UNKNOWN"}};
    if (targetToString.count(coreType) > 0) {
        return targetToString[coreType];
    }
    return "UNKNOWN";
}

// 判断候选规划是否好于当前规划，是则将候选规划设为当前规划
void TaskGraph::ApplyCandidate() {
    int prevTime = makespan;
    int currTime = -1;
    for (auto &task : tasks) {
        currTime = currTime > task.endTimeCandidate ? currTime : task.endTimeCandidate;
    }
    if (prevTime >= 0 && prevTime <= currTime) {
        return;
    }
    for (auto &task : tasks) {
        task.startTime = task.startTimeCandidate;
        task.endTime = task.endTimeCandidate;
        task.targetCoreType = task.targetCoreTypeCandidate;
    }
    makespan = currTime;
    APASS_LOG_INFO_F(Elements::Operation, "Found better schedule, update makespan to %d.", makespan);
}

int TaskGraph::AddTask(const std::string &name, ScheduleCoreType coreType, int latency) {
    int newTaskIdx = static_cast<int>(tasks.size());
    tasks.emplace_back(name, newTaskIdx, coreType, latency);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create new taskNode with idx=%d and coreType=%s.",
                      newTaskIdx, ScheduleCoreTypeToString(coreType).c_str());
    return newTaskIdx;
}

void TaskGraph::AddDependency(int src, int dst) {
    if (src == dst) {
        return;
    }
    tasks[src].outTasks.push_back(dst);
    tasks[dst].inTasks.push_back(src);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create dependency from taskNode %d to taskNode %d.", src, dst);
}

inline int UDSFind(std::vector<int> &parent, int i) {
    if (parent[i] == i) {
        return i;
    }
    return parent[i] = UDSFind(parent, parent[i]);
}

inline void UDSUnion(std::vector<int> &parent, int i, int j) {
    int rootOfI = UDSFind(parent, i);
    int rootOfJ = UDSFind(parent, j);
    if (rootOfI != rootOfJ) {
        if (rootOfI < rootOfJ) {
            parent[rootOfJ] = rootOfI;
        } else {
            parent[rootOfI] = rootOfJ;
        }
    }
}

void TaskGraph::ClearSchedule() {
    makespan = -1;
    for (auto &task : tasks) {
        task.targetCoreType  = TargetCoreType::UNKNOWN;
        task.startTime = 0;
        task.endTime = 0;
        task.targetCoreTypeCandidate  = TargetCoreType::UNKNOWN;
        task.startTimeCandidate = 0;
        task.endTimeCandidate = 0;
    }
}

// 寻找时间槽不重叠情况下的最早执行时间
void CoreScheduler::FindEarliestSlot(std::vector<std::pair<int, int>> &timeSlot, int earliestStart, int latency, int &currentIdx, std::pair<int, int> &currentInterval) {
    int currentEarliestStart = INT32_MAX;
    currentIdx = -1;
    currentInterval = std::make_pair(-1, -1);
    APASS_LOG_DEBUG_F(Elements::Operation, "Try to find earliest slot with earliestStart=%d and latency=%d.", earliestStart, latency);
    for (int i = 0; i < static_cast<int>(timeSlot.size()); i++) {
        int validStart = std::max(timeSlot[i].first, earliestStart);
        if (timeSlot[i].second - validStart < latency) {
            continue;
        }
        if (validStart < currentEarliestStart) {
            currentEarliestStart = validStart;
            currentInterval = std::make_pair(validStart, validStart + latency);
            currentIdx = i;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "The earliest slot is from %d to %d.", currentInterval.first, currentInterval.second);
}

// 更新空闲时间槽
void CoreScheduler::UpdateInterval(std::vector<std::pair<int,int>> &timeSlot, int &insertIdx, std::pair<int,int> &insertInterval) {
    auto origInterval = timeSlot[insertIdx];
    APASS_LOG_DEBUG_F(Elements::Operation, "The original slot [%d, %d] is removed.", origInterval.first, origInterval.second);
    timeSlot.erase(timeSlot.begin() + insertIdx);
    if (origInterval.first < insertInterval.first) {
        timeSlot.push_back(std::make_pair(origInterval.first, insertInterval.first));
        APASS_LOG_DEBUG_F(Elements::Operation, "New slot [%d, %d] is added.", origInterval.first, insertInterval.first);
    }
    if (origInterval.second > insertInterval.second) {
        timeSlot.push_back(std::make_pair(insertInterval.second, origInterval.second));
        APASS_LOG_DEBUG_F(Elements::Operation, "New slot [%d, %d] is added.", insertInterval.second, origInterval.second);
    }
}

// 使用DFS得到taskNode的一个拓扑排序
std::vector<int> CoreScheduler::GetDFSTopoSeq(TaskGraph &taskGraph) {
    std::vector<bool> finishedTasks(taskGraph.tasks.size(), false);
    std::vector<int> taskStack;
    std::vector<int> topoSeq;
    for (auto &task : taskGraph.tasks) {
        if (task.outTasks.size() == 0) {
            taskStack.push_back(task.idx);
        }
    }
    std::vector<int> notReadyPrevTaskIds;
    while (taskStack.size() > 0) {
        int taskId = taskStack.back();
        taskStack.pop_back();
        if (finishedTasks[taskId]) {
            continue;
        }
        notReadyPrevTaskIds.clear();
        for (int prevTaskId : taskGraph.tasks[taskId].inTasks) {
            if (!finishedTasks[prevTaskId]) {
                notReadyPrevTaskIds.push_back(prevTaskId);
            }
        }
        if (notReadyPrevTaskIds.size() > 0) {
            taskStack.push_back(taskId);
            taskStack.insert(taskStack.end(), notReadyPrevTaskIds.begin(), notReadyPrevTaskIds.end());
            continue;
        }
        topoSeq.push_back(taskId);
        finishedTasks[taskId] = true;
    }
    return topoSeq;
}

// 基于最早完成时间和空闲时间槽的任务排布
void CoreScheduler::EFTWithInsertSchedule(TaskGraph &taskGraph, std::vector<int> &topoSeq) {
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>> availTime;
    availTime[TargetCoreType::AIC] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV0] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV1] = {{0, INT32_MAX}};
    int currentIdx = -1;
    std::pair<int, int> currentInterval{-1, -1};
    for (int taskId : topoSeq) {
        int evalDepTimeStart = 0;
        for (int prevTaskId : taskGraph.tasks[taskId].inTasks) {
            evalDepTimeStart = std::max(evalDepTimeStart, taskGraph.tasks[prevTaskId].endTimeCandidate);
        }
        TargetCoreType evalCore = TargetCoreType::UNKNOWN;
        if (taskGraph.tasks[taskId].coreType == ScheduleCoreType::AIC) {
            evalCore = TargetCoreType::AIC;
            FindEarliestSlot(availTime[evalCore], evalDepTimeStart, taskGraph.tasks[taskId].latency, currentIdx, currentInterval);
        } else {
            int currentIdxAIV0 = -1;
            std::pair<int, int> currentIntervalAIV0{-1, -1};
            int currentIdxAIV1 = -1;
            std::pair<int, int> currentIntervalAIV1{-1, -1};
            FindEarliestSlot(availTime[TargetCoreType::AIV0], evalDepTimeStart, taskGraph.tasks[taskId].latency, currentIdxAIV0, currentIntervalAIV0);
            FindEarliestSlot(availTime[TargetCoreType::AIV1], evalDepTimeStart, taskGraph.tasks[taskId].latency, currentIdxAIV1, currentIntervalAIV1);
            if (currentIntervalAIV0.first <= currentIntervalAIV1.first) {
                evalCore = TargetCoreType::AIV0;
                currentIdx = currentIdxAIV0;
                currentInterval = currentIntervalAIV0;
            } else {
                evalCore = TargetCoreType::AIV1;
                currentIdx = currentIdxAIV1;
                currentInterval = currentIntervalAIV1;
            }
        }
        taskGraph.tasks[taskId].targetCoreTypeCandidate = evalCore;
        taskGraph.tasks[taskId].startTimeCandidate = currentInterval.first;
        taskGraph.tasks[taskId].endTimeCandidate = currentInterval.second;
        UpdateInterval(availTime[evalCore], currentIdx, currentInterval);
    }
    taskGraph.ApplyCandidate();
    APASS_LOG_INFO_F(Elements::Operation, "EFTWithInsertSchedule get final makespan %d.", taskGraph.makespan);
}

// 基于最早完成时间的任务排布
void CoreScheduler::EFTSchedule(TaskGraph &taskGraph, std::vector<int> &topoSeq) {
    std::unordered_map<TargetCoreType, int> currentTime{{TargetCoreType::AIC, 0},
        {TargetCoreType::AIV0, 0}, {TargetCoreType::AIV1, 0}};
    for (int taskId : topoSeq) {
        int evalDepTimeStart = 0;
        for (int prevTaskId : taskGraph.tasks[taskId].inTasks) {
            evalDepTimeStart = std::max(evalDepTimeStart, taskGraph.tasks[prevTaskId].endTimeCandidate);
        }
        TargetCoreType evalCore = TargetCoreType::UNKNOWN;
        if (taskGraph.tasks[taskId].coreType == ScheduleCoreType::AIC) {
            evalCore = TargetCoreType::AIC;
        } else {
            evalCore = currentTime[TargetCoreType::AIV0] <= currentTime[TargetCoreType::AIV1] ? TargetCoreType::AIV0 : TargetCoreType::AIV1;
        }
        taskGraph.tasks[taskId].targetCoreTypeCandidate = evalCore;
        taskGraph.tasks[taskId].startTimeCandidate = std::max(evalDepTimeStart, currentTime[evalCore]);
        taskGraph.tasks[taskId].endTimeCandidate = taskGraph.tasks[taskId].startTimeCandidate + taskGraph.tasks[taskId].latency;
        currentTime[evalCore] = taskGraph.tasks[taskId].endTimeCandidate;
    }
    taskGraph.ApplyCandidate();
    APASS_LOG_INFO_F(Elements::Operation, "EFTSchedule get final makespan %d.", taskGraph.makespan);
}

// 对所有的拓扑序执行基于最早完成时间的任务排布
void CoreScheduler::BruteForceScheduleRecursiveStep(std::vector<bool> &visited, int recursiveLevel, TaskGraph &taskGraph, std::vector<int> &topoList) {
    if (recursiveLevel >= static_cast<int>(taskGraph.tasks.size())) {
        EFTSchedule(taskGraph, topoList);
    }
    for (auto &task : taskGraph.tasks) {
        if (visited[task.idx]) {
            continue;
        }
        bool canDeploy = true;
        for (int prevTaskIdx : task.inTasks) {
            if (!visited[prevTaskIdx]) {
                canDeploy = false;
                break;
            }
        }
        if (!canDeploy) {
            continue;
        }
        visited[task.idx] = true;
        topoList.push_back(task.idx);
        BruteForceScheduleRecursiveStep(visited, recursiveLevel + 1, taskGraph, topoList);
        topoList.pop_back();
        visited[task.idx] = false;
    }
}

// 根据节点数量，判断是否遍历所有拓扑序进行任务排布
void CoreScheduler::Schedule(TaskGraph &taskGraph, int bruteForceThreshold) {
    taskGraph.ClearSchedule();
    APASS_LOG_INFO_F(Elements::Operation, "Start schedule with brute force threshold %d.", bruteForceThreshold);
    if (static_cast<int>(taskGraph.tasks.size()) > bruteForceThreshold) {
        std::vector<int> topoSeq = GetDFSTopoSeq(taskGraph);
        EFTWithInsertSchedule(taskGraph, topoSeq);
    } else {
        std::vector<bool> visited(taskGraph.tasks.size(), false);
        std::vector<int> topoList;
        BruteForceScheduleRecursiveStep(visited, 0, taskGraph, topoList);
    }
}

// Alloc op需要与其同级的op处于同一个子图中, Convert的alloc应跟随其后op
void TaskSpliter::BuildSameLayerConnectionWithBack() {
    for (size_t i = 0; i < opList_.size(); i++) {
        if (ALLOC_OPCODE.count(opList_[i]->GetOpcode()) == 0) {
            continue;
        }
    ScheduleCoreType srcCoreType = opCoreTypes_[i];
    APASS_LOG_DEBUG_F(Elements::Operation, "Found alloc op %s[%d].", opList_[i]->GetOpcodeStr().c_str(), opList_[i]->GetOpMagic());
        for (auto & oop : opList_[i]->GetOOperands()) {
            for (auto &sameLayerOpPtr : oop->GetProducers()) {
                int dstOpMagic = sameLayerOpPtr->GetOpMagic();
                if (opMagicToIdx_.count(dstOpMagic) == 0) {
                    continue;
                }
                if (opCoreTypes_[opMagicToIdx_[dstOpMagic]] != srcCoreType) {
                    continue;
                }
            APASS_LOG_DEBUG_F(Elements::Operation, "-- add %s[%d] to same layer connection because of the alloc op.",
                sameLayerOpPtr->GetOpcodeStr().c_str(), sameLayerOpPtr->GetOpMagic());
            sameLayerConnection_.push_back({i, opMagicToIdx_[sameLayerOpPtr->GetOpMagic()]});
            }
        }
    }
}

// Alloc op需要与其同级的op处于同一个子图中, Convert的alloc应跟随其前op
void TaskSpliter::BuildSameLayerConnectionWithFront() {
    for (size_t i = 0; i < opList_.size(); i++) {
        if (ALLOC_OPCODE.count(opList_[i]->GetOpcode()) == 0) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Found alloc op %s[%d].", opList_[i]->GetOpcodeStr().c_str(), opList_[i]->GetOpMagic());
        for (auto & oop : opList_[i]->GetOOperands()) {
            for (auto &sameLayerOpPtr : oop->GetProducers()) {
                int dstOpMagic = sameLayerOpPtr->GetOpMagic();
                if (opMagicToIdx_.count(dstOpMagic) == 0) {
                    continue;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "-- add %s[%d] to same layer connection because of the alloc op.",
                    sameLayerOpPtr->GetOpcodeStr().c_str(), sameLayerOpPtr->GetOpMagic());
                sameLayerConnection_.push_back({i, opMagicToIdx_[sameLayerOpPtr->GetOpMagic()]});
                opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[dstOpMagic]];
            }
        }
    }
}

// 构建op的coreType和连接图
void TaskSpliter::BuildOpGraph() {
    int opNum = static_cast<int>(opList_.size());
    opCoreTypes_.resize(opNum);
    opMagicToIdx_.clear();
    for (int i = 0; i < opNum; i++) {
        opMagicToIdx_[opList_[i]->GetOpMagic()] = i;
        opCoreTypes_[i] = OpcodeManager::Inst().GetCoreType(opList_[i]->GetOpcode()) == OpCoreType::AIC ?
                         ScheduleCoreType::AIC : ScheduleCoreType::AIV;
    }
    for (int i = 0; i < opNum; i++) {
        if (opList_[i]->GetOpcode() == Opcode::OP_COPY_IN) {
            auto nextOp = *opList_[i]->ConsumerOps().begin();
            opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[nextOp->GetOpMagic()]];
        }
        else if (opList_[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            auto prevOp = *opList_[i]->ProducerOps().begin();
            opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[prevOp->GetOpMagic()]];
        }
        if (opList_[i]->HasAttribute(OpAttributeKey::isCube)) {
            bool isCube = opList_[i]->GetBoolAttribute(OpAttributeKey::isCube);
            opCoreTypes_[i] = isCube ? ScheduleCoreType::AIC : ScheduleCoreType::AIV;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Mark %s[%d] as %s core type.", opList_[i]->GetOpcodeStr().c_str(), opList_[i]->GetOpMagic(),
            opCoreTypes_[i] == ScheduleCoreType::AIC ? "AIC" : "AIV");
    }
    APASS_LOG_INFO_F(Elements::Operation, "Mark core type finished.");
    opInGraph_.resize(opNum);
    opOutGraph_.resize(opNum);
    for (int i = 0; i < opNum; i++) {
        for (auto consumerOp : opList_[i]->ConsumerOps()) {
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

// mix子图切分主函数
void TaskSpliter::SplitGraph(const std::vector<Operation *> &opList) {
    APASS_LOG_INFO_F(Elements::Operation, "Start to split mix graph with op num %d.", opList.size());
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
    APASS_LOG_INFO_F(Elements::Operation, "Build the task graph finished.");
}

// 将强连通分量展开，避免成环
inline int FlattenSCC(std::vector<ScheduleCoreType> &clusterCoreTypes, std::vector<std::vector<int>> &sccResult,
        std::unordered_map<int, int> &oldClusterIdToSCCId, std::unordered_map<int, std::vector<int>> &sccIdToNewClusters,
        std::unordered_map<int, int> &oldClusterToNewCluster) {
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

// 将强连通分量展开，并构建新的连接图
void TaskSpliter::CombineSCC(std::vector<int> &clusterIds, std::vector<ScheduleCoreType> &clusterCoreTypes,
        std::vector<std::set<int>> &inGraph, std::vector<std::set<int>> &outGraph, std::vector<std::vector<int>> &sccResult) {
    std::unordered_map<int, int> oldClusterIdToSCCId;
    std::unordered_map<int, std::vector<int>> sccIdToNewClusters;
    std::unordered_map<int, int> oldClusterToNewCluster;
    int newClusterNum = FlattenSCC(clusterCoreTypes, sccResult, oldClusterIdToSCCId, sccIdToNewClusters, oldClusterToNewCluster);
    APASS_LOG_INFO_F(Elements::Operation, "Cluster num after flatten strongly connected components is %d.", newClusterNum);
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
void StrongConnectionComponentFinder::Find(std::vector<std::set<int>> &inGraph,
        std::vector<std::set<int>> &outGraph, std::vector<std::vector<int>> &sccResult) {
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
        if (inGraph[i].size() == 0) {
            TarJanAlg(i, outGraph, sccResult);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "TarJan Algorithm finished.");
}

// 递归使用TarJan算法获得强连通分量
void StrongConnectionComponentFinder::TarJanAlg(int idx, std::vector<std::set<int>> &outGraph, std::vector<std::vector<int>> &sccResult) {
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
void TaskSpliter::BuildInOutGraph(std::vector<std::set<int>> &inGraph, std::vector<std::set<int>> &outGraph,
        std::vector<int> &clusterIds, int clusterNum) {
    inGraph.clear();
    inGraph.resize(clusterNum);
    outGraph.clear();
    outGraph.resize(clusterNum);
    int opNum = static_cast<int>(opList_.size());
    for (int i = 0; i < opNum; i++) {
        int currTaskIdx = clusterIds[i];
        for (auto consumerOp : opList_[i]->ConsumerOps()) {
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
TaskGraph TaskSpliter::BuildTaskGraph() {
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
inline bool IsFromAICToAIV(Operation* op) {
    const std::unordered_set<MemoryType> AICmem{MemoryType::MEM_L0C, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B};
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
    APASS_LOG_DEBUG_F(Elements::Operation, "op %s[%d] is from AIC to AIV.", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return true;
}

// 判断op的ioperand为AIV类型且ooperand为AIC类型
inline bool IsFromAIVToAIC(Operation* op) {
    const std::unordered_set<MemoryType> AICmem{MemoryType::MEM_L0C, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B};
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
    APASS_LOG_DEBUG_F(Elements::Operation, "op %s[%d] is from AIV to AIC.", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return true;
}

// 根据op的CoreType构建连通集
int TaskSpliter::BuildCluster(std::vector<int> &clusterIds, std::vector<ScheduleCoreType> &clusterCoreTypes) {
    DSUWithOrder dsu(opList_.size());
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        for (int nextOpIdx : opOutGraph_[idx]) {
            if (opCoreTypes_[idx] == opCoreTypes_[nextOpIdx]) {
                dsu.Union(idx, nextOpIdx);
            }
        }
    }
    for (auto pr : sameLayerConnection_) {
        dsu.Union(pr.first, pr.second);
    }
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        if (IsFromAICToAIV(opList_[idx]) || IsFromAIVToAIC(opList_[idx])) {
            for (int nextOpIdx : opOutGraph_[idx]) {
                dsu.Union(nextOpIdx, *opOutGraph_[idx].begin());
            }
        }
    }
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
inline bool NoDepDeteched(const std::vector<int> &currOldTasks, int currTaskId, DAGReachableJudger &judger) {
    for (int oldTaskId : currOldTasks) {
        if (judger.IsReachable(oldTaskId, currTaskId)) {
            return false;
        }
    }
    return true;
}

// 根据taskNode在模拟泳道图中的位置和可达性，返回合并后的taskNode列表
std::vector<std::vector<int>> TaskSpliter::FindMergeableTaskNodes() {
    std::vector<std::vector<int>> newTaskToOldTasks;
    std::unordered_map<TargetCoreType, std::vector<int>> targetTypeToTasks{{TargetCoreType::AIC, {}},
        {TargetCoreType::AIV0, {}}, {TargetCoreType::AIV1, {}}};
    DAGReachableJudger reachableJudger;
    reachableJudger.Build(inGraph_, outGraph_);
    for (int i = 0; i < static_cast<int>(taskGraph_.tasks.size()); i++) {
        targetTypeToTasks[taskGraph_.tasks[i].targetCoreType].push_back(i);
    }
    for (auto &tasksPair : targetTypeToTasks) {
        std::sort(tasksPair.second.begin(), tasksPair.second.end(), [this](int i, int j) {
            return taskGraph_.tasks[i].startTime < taskGraph_.tasks[j].startTime;
        });
        std::vector<int> oldTasks;
        for (int currTaskId : tasksPair.second) {
            if (oldTasks.empty() || NoDepDeteched(oldTasks, currTaskId, reachableJudger) ) {
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
void TaskSpliter::MergeTask() {
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
            s.tasks[newTaskId].opList_.insert(s.tasks[newTaskId].opList_.end(),
                taskGraph_.tasks[oldTaskId].opList_.begin(), taskGraph_.tasks[oldTaskId].opList_.end());
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
void TaskSpliter::MergeTaskByTargetCoreType() {
    std::unordered_map<TargetCoreType, std::vector<int>> targetTypeToTasks{{TargetCoreType::AIC, {}},
        {TargetCoreType::AIV0, {}}, {TargetCoreType::AIV1, {}}};
    std::unordered_map<TargetCoreType, ScheduleCoreType> targetTypeToScheduleType{{TargetCoreType::AIC, ScheduleCoreType::AIC},
        {TargetCoreType::AIV0, ScheduleCoreType::AIV}, {TargetCoreType::AIV1, ScheduleCoreType::AIV}};
    for (int i = 0; i < static_cast<int>(taskGraph_.tasks.size()); i++) {
        targetTypeToTasks[taskGraph_.tasks[i].targetCoreType].push_back(i);
    }
    TaskGraph s;
    int newTaskIdx = 0;
    for (auto &tasksPair : targetTypeToTasks) {
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
            s.tasks[newTaskId].opList_.insert(s.tasks[newTaskId].opList_.end(),
                taskGraph_.tasks[oldTaskId].opList_.begin(), taskGraph_.tasks[oldTaskId].opList_.end());
        }
    }
    taskGraph_ = s;
}

// 根据划分结果标记op的AIVCore与internalSubgraphID
void TaskSpliter::MarkInternalSubgraphID() {
    std::unordered_map<TargetCoreType, AIVCore> targetMap{{TargetCoreType::AIC, AIVCore::UNSPECIFIED},
        {TargetCoreType::UNKNOWN, AIVCore::UNSPECIFIED}, {TargetCoreType::AIV0, AIVCore::AIV0}, {TargetCoreType::AIV1, AIVCore::AIV1}};
    std::unordered_map<TargetCoreType, int> subGraphIdMap{{TargetCoreType::AIC, NEGATIVE_ONE},
        {TargetCoreType::AIV0, NEGATIVE_ONE}, {TargetCoreType::AIV1, NEGATIVE_ONE}, {TargetCoreType::UNKNOWN, NEGATIVE_ONE}};
    int id = 0;
    for (auto &task : taskGraph_.tasks) {
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
    for (auto &task : taskGraph_.tasks) {
        auto subGraphId = subGraphIdMap[task.targetCoreType];
        for (auto opPtr : task.opList_) {
            opPtr->UpdateInternalSubgraphID(subGraphId);
        }
    }
}

// 将多个taskNode的opList在保持内部顺序的前提下合并成符合拓扑序的一个opList
std::vector<Operation*> TaskSpliter::GetMergedOperations() {
    std::priority_queue<std::pair<int, Operation*>, std::vector<std::pair<int, Operation*>>,
                        std::greater<std::pair<int, Operation*>>> pQueue;
    std::unordered_map<Operation*, int> opPriority;
    std::unordered_map<Operation*, int> inLinkNum;
    std::vector<Operation*> topoSeq;
    for (auto &task : taskGraph_.tasks) {
        for (size_t opIdx = 0; opIdx < task.opList_.size(); opIdx++) {
            Operation *opPtr = task.opList_[opIdx];
            opPriority[opPtr] = opIdx;
            inLinkNum[opPtr] = opPtr->ProducerOps().size();
            if (inLinkNum[opPtr] == 0) {
                pQueue.push({opIdx, opPtr});
            }
        }
    }
    while (pQueue.size() > 0) {
        auto ele = pQueue.top();
        pQueue.pop();
        topoSeq.push_back(ele.second);
        for (auto &nextOpPtr : ele.second->ConsumerOps()) {
            inLinkNum[nextOpPtr]--;
            if (inLinkNum[nextOpPtr] == 0) {
                pQueue.push({opPriority[nextOpPtr], nextOpPtr});
            }
        }
    }
    return topoSeq;
}


DSUWithOrder::DSUWithOrder(int num) {
    parent.resize(num);
    for (int i = 0; i < num; i++) {
        parent[i] = i;
    }
}

int DSUWithOrder::Find(int i) {
    if (parent[i] == i) {
        return i;
    }
    parent[i] = Find(parent[i]);
    return parent[i];
}

void DSUWithOrder::Union(int i, int j) {
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
void DAGReachableJudger::Build(const std::vector<std::set<int>> &inGraph, const std::vector<std::set<int>> &outGraph) {
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
void DAGReachableJudger::SetReachable(const int src, const int dst) {
    const int bitPerBlock = 32;
    size_t index = dst / bitPerBlock;
    size_t offset = dst % bitPerBlock;
    if (reachableSet[src].size() < index + 1) {
        reachableSet[src].resize(index + 1, 0);
    }
    reachableSet[src][index] |= (1U << offset);
}

// 设定从src可以到达dst可达的所有节点
void DAGReachableJudger::MergeReachable(int src, int dst) {
    if (reachableSet[src].size() < reachableSet[dst].size()) {
        reachableSet[src].resize(reachableSet[dst].size(), 0);
    }
    for (size_t i = 0; i < reachableSet[dst].size(); i++) {
        reachableSet[src][i] |= reachableSet[dst][i];
    }
}

// 判断有向无环图中是否存在从src到dst的路径
bool DAGReachableJudger::IsReachable(int src, int dst) {
    const int bitPerBlock = 32;
    size_t index = dst / bitPerBlock;
    size_t offset = dst % bitPerBlock;
    return (reachableSet[src][index] & (1U << offset)) != 0;
}

} // namespace npu::tile_fwk
