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

inline std::string ScheduleCoreTypeToString(ScheduleCoreType coreType)
{
    if (coreType == ScheduleCoreType::AIC) {
        return "AIC";
    }
    if (coreType == ScheduleCoreType::AIV) {
        return "AIV";
    }
    return "UNKNOWN";
}

inline std::string TargetCoreTypeToString(TargetCoreType coreType)
{
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
void TaskGraph::ApplyCandidate()
{
    int prevTime = makespan;
    int currTime = -1;
    for (auto& task : tasks) {
        currTime = currTime > task.endTimeCandidate ? currTime : task.endTimeCandidate;
    }
    if (prevTime >= 0 && prevTime <= currTime) {
        return;
    }
    for (auto& task : tasks) {
        task.startTime = task.startTimeCandidate;
        task.endTime = task.endTimeCandidate;
        task.targetCoreType = task.targetCoreTypeCandidate;
    }
    makespan = currTime;
    APASS_LOG_INFO_F(Elements::Operation, "Found better schedule, update makespan to %d.", makespan);
}

// 无条件把 Candidate 字段拷贝为 final；makespan 仅作为字段完整性保留，不参与任何决策
void TaskGraph::ApplyCandidateUnconditional()
{
    int currTime = -1;
    for (auto& task : tasks) {
        task.startTime = task.startTimeCandidate;
        task.endTime = task.endTimeCandidate;
        task.targetCoreType = task.targetCoreTypeCandidate;
        currTime = currTime > task.endTime ? currTime : task.endTime;
    }
    makespan = currTime;
}

int TaskGraph::AddTask(const std::string& name, ScheduleCoreType coreType, int latency)
{
    int newTaskIdx = static_cast<int>(tasks.size());
    tasks.emplace_back(name, newTaskIdx, coreType, latency);
    APASS_LOG_DEBUG_F(
        Elements::Operation, "Create new taskNode with idx=%d and coreType=%s.", newTaskIdx,
        ScheduleCoreTypeToString(coreType).c_str());
    return newTaskIdx;
}

void TaskGraph::AddDependency(int src, int dst)
{
    if (src == dst) {
        return;
    }
    tasks[src].outTasks.push_back(dst);
    tasks[dst].inTasks.push_back(src);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create dependency from taskNode %d to taskNode %d.", src, dst);
}

inline int UDSFind(std::vector<int>& parent, int i)
{
    if (parent[i] == i) {
        return i;
    }
    return parent[i] = UDSFind(parent, parent[i]);
}

inline void UDSUnion(std::vector<int>& parent, int i, int j)
{
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

void TaskGraph::ClearSchedule()
{
    makespan = -1;
    for (auto& task : tasks) {
        task.targetCoreType = TargetCoreType::UNKNOWN;
        task.startTime = 0;
        task.endTime = 0;
        task.targetCoreTypeCandidate = TargetCoreType::UNKNOWN;
        task.startTimeCandidate = 0;
        task.endTimeCandidate = 0;
    }
}

// 寻找时间槽不重叠情况下的最早执行时间
void CoreScheduler::FindEarliestSlot(
    std::vector<std::pair<int, int>>& timeSlot, int earliestStart, int latency, int& currentIdx,
    std::pair<int, int>& currentInterval)
{
    int currentEarliestStart = INT32_MAX;
    currentIdx = -1;
    currentInterval = std::make_pair(-1, -1);
    APASS_LOG_DEBUG_F(
        Elements::Operation, "Try to find earliest slot with earliestStart=%d and latency=%d.", earliestStart, latency);
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
    APASS_LOG_DEBUG_F(
        Elements::Operation, "The earliest slot is from %d to %d.", currentInterval.first, currentInterval.second);
}

// 更新空闲时间槽
void CoreScheduler::UpdateInterval(
    std::vector<std::pair<int, int>>& timeSlot, int& insertIdx, std::pair<int, int>& insertInterval)
{
    auto origInterval = timeSlot[insertIdx];
    APASS_LOG_DEBUG_F(
        Elements::Operation, "The original slot [%d, %d] is removed.", origInterval.first, origInterval.second);
    timeSlot.erase(timeSlot.begin() + insertIdx);
    if (origInterval.first < insertInterval.first) {
        timeSlot.push_back(std::make_pair(origInterval.first, insertInterval.first));
        APASS_LOG_DEBUG_F(Elements::Operation, "New slot [%d, %d] is added.", origInterval.first, insertInterval.first);
    }
    if (origInterval.second > insertInterval.second) {
        timeSlot.push_back(std::make_pair(insertInterval.second, origInterval.second));
        APASS_LOG_DEBUG_F(
            Elements::Operation, "New slot [%d, %d] is added.", insertInterval.second, origInterval.second);
    }
}

// 使用DFS得到taskNode的一个拓扑排序
std::vector<int> CoreScheduler::GetDFSTopoSeq(TaskGraph& taskGraph)
{
    std::vector<bool> finishedTasks(taskGraph.tasks.size(), false);
    std::vector<int> taskStack;
    std::vector<int> topoSeq;
    for (auto& task : taskGraph.tasks) {
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
void CoreScheduler::EFTWithInsertSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
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
            FindEarliestSlot(
                availTime[evalCore], evalDepTimeStart, taskGraph.tasks[taskId].latency, currentIdx, currentInterval);
        } else {
            int currentIdxAIV0 = -1;
            std::pair<int, int> currentIntervalAIV0{-1, -1};
            int currentIdxAIV1 = -1;
            std::pair<int, int> currentIntervalAIV1{-1, -1};
            FindEarliestSlot(
                availTime[TargetCoreType::AIV0], evalDepTimeStart, taskGraph.tasks[taskId].latency, currentIdxAIV0,
                currentIntervalAIV0);
            FindEarliestSlot(
                availTime[TargetCoreType::AIV1], evalDepTimeStart, taskGraph.tasks[taskId].latency, currentIdxAIV1,
                currentIntervalAIV1);
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
void CoreScheduler::EFTSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    std::unordered_map<TargetCoreType, int> currentTime{
        {TargetCoreType::AIC, 0}, {TargetCoreType::AIV0, 0}, {TargetCoreType::AIV1, 0}};
    for (int taskId : topoSeq) {
        int evalDepTimeStart = 0;
        for (int prevTaskId : taskGraph.tasks[taskId].inTasks) {
            evalDepTimeStart = std::max(evalDepTimeStart, taskGraph.tasks[prevTaskId].endTimeCandidate);
        }
        TargetCoreType evalCore = TargetCoreType::UNKNOWN;
        if (taskGraph.tasks[taskId].coreType == ScheduleCoreType::AIC) {
            evalCore = TargetCoreType::AIC;
        } else {
            evalCore = currentTime[TargetCoreType::AIV0] <= currentTime[TargetCoreType::AIV1] ? TargetCoreType::AIV0 :
                                                                                                TargetCoreType::AIV1;
        }
        taskGraph.tasks[taskId].targetCoreTypeCandidate = evalCore;
        taskGraph.tasks[taskId].startTimeCandidate = std::max(evalDepTimeStart, currentTime[evalCore]);
        taskGraph.tasks[taskId].endTimeCandidate =
            taskGraph.tasks[taskId].startTimeCandidate + taskGraph.tasks[taskId].latency;
        currentTime[evalCore] = taskGraph.tasks[taskId].endTimeCandidate;
    }
    taskGraph.ApplyCandidate();
    APASS_LOG_INFO_F(Elements::Operation, "EFTSchedule get final makespan %d.", taskGraph.makespan);
}

// 对所有的拓扑序执行基于最早完成时间的任务排布
void CoreScheduler::BruteForceScheduleRecursiveStep(
    std::vector<bool>& visited, int recursiveLevel, TaskGraph& taskGraph, std::vector<int>& topoList)
{
    if (recursiveLevel >= static_cast<int>(taskGraph.tasks.size())) {
        EFTSchedule(taskGraph, topoList);
    }
    for (auto& task : taskGraph.tasks) {
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

void CoreScheduler::SelectAIVCore(
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
    int evalDepTimeStart, int maxCrossDepEnd, int latency,
    TargetCoreType& evalCore, int& currentIdx, std::pair<int, int>& currentInterval)
{
    int idxAIV0 = -1;
    std::pair<int, int> intervalAIV0{-1, -1};
    FindEarliestSlot(availTime[TargetCoreType::AIV0], evalDepTimeStart, latency, idxAIV0, intervalAIV0);
    int idxAIV1 = -1;
    std::pair<int, int> intervalAIV1{-1, -1};
    FindEarliestSlot(availTime[TargetCoreType::AIV1], evalDepTimeStart, latency, idxAIV1, intervalAIV1);

    auto calcGap = [&](const std::pair<int, int>& iv) -> int64_t {
        if (iv.first < 0)
            return INT64_MAX;
        if (maxCrossDepEnd == INT32_MIN)
            return 0;
        return std::max<int64_t>(0, static_cast<int64_t>(iv.first) - maxCrossDepEnd);
    };
    int64_t gap0 = calcGap(intervalAIV0);
    int64_t gap1 = calcGap(intervalAIV1);

    auto getLastFinishBefore = [&](TargetCoreType core, std::pair<int, int> intervalAIV) -> int {
        for (auto& slot : availTime[core]) {
            if (slot.second >= intervalAIV.first) {
                return slot.first;
            }
        }
        return 0;
    };

    bool chooseAIV0 = false;
    if (gap0 < gap1) {
        chooseAIV0 = true;
    } else if (gap0 == gap1) {
        if (intervalAIV0.first < intervalAIV1.first) {
            chooseAIV0 = true;
        } else if (intervalAIV0.first == intervalAIV1.first) {
            int lastFinish0 = getLastFinishBefore(TargetCoreType::AIV0, intervalAIV0);
            int lastFinish1 = getLastFinishBefore(TargetCoreType::AIV1, intervalAIV1);
            chooseAIV0 = (lastFinish0 <= lastFinish1);
        }
    }
    if (chooseAIV0) {
        evalCore = TargetCoreType::AIV0;
        currentIdx = idxAIV0;
        currentInterval = intervalAIV0;
    } else {
        evalCore = TargetCoreType::AIV1;
        currentIdx = idxAIV1;
        currentInterval = intervalAIV1;
    }
}

// --------- GapMin: 启发式最小化跨核相邻依赖边的等待间隔 ---------
// 紧耦合调度单个任务
void CoreScheduler::ScheduleOneTask(
    TaskGraph& taskGraph, int taskId, std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
    std::function<bool(TargetCoreType)> isAicCore)
{
    auto& task = taskGraph.tasks[taskId];
    int evalDepTimeStart = 0;
    int maxCrossDepEnd = INT32_MIN;
    bool taskIsAic = (task.coreType == ScheduleCoreType::AIC);
    for (int prevTaskId : task.inTasks) {
        auto& prev = taskGraph.tasks[prevTaskId];
        evalDepTimeStart = std::max(evalDepTimeStart, prev.endTimeCandidate);
        bool prevIsAic = isAicCore(prev.targetCoreTypeCandidate);
        if (taskIsAic != prevIsAic) {
            maxCrossDepEnd = std::max(maxCrossDepEnd, prev.endTimeCandidate);
        }
    }

    TargetCoreType evalCore = TargetCoreType::UNKNOWN;
    int currentIdx = -1;
    std::pair<int, int> currentInterval{-1, -1};

    if (taskIsAic) {
        evalCore = TargetCoreType::AIC;
        FindEarliestSlot(availTime[evalCore], evalDepTimeStart, task.latency, currentIdx, currentInterval);
    } else {
        SelectAIVCore(availTime, evalDepTimeStart, maxCrossDepEnd, task.latency, evalCore, currentIdx, currentInterval);
    }
    task.targetCoreTypeCandidate = evalCore;
    task.startTimeCandidate = currentInterval.first;
    task.endTimeCandidate = currentInterval.second;
    UpdateInterval(availTime[evalCore], currentIdx, currentInterval);
}

// 尝试调度跨核后继
void CoreScheduler::TryScheduleCrossCoreSuccessors(
    TaskGraph& taskGraph, int taskId, std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>>& availTime,
    std::set<int>& scheduledTasks, std::function<bool(TargetCoreType)> isAicCore)
{
    auto& task = taskGraph.tasks[taskId];

    for (int succId : task.outTasks) {
        if (scheduledTasks.count(succId) > 0) {
            continue;
        }

        auto& succ = taskGraph.tasks[succId];
        bool isCrossCore = (task.coreType != succ.coreType);
        if (!isCrossCore) {
            continue;
        }

        bool allPredScheduled = true;
        for (int predId : succ.inTasks) {
            if (scheduledTasks.count(predId) == 0) {
                allPredScheduled = false;
                break;
            }
        }

        if (allPredScheduled) {
            ScheduleOneTask(taskGraph, succId, availTime, isAicCore);
            scheduledTasks.insert(succId);
            TryScheduleCrossCoreSuccessors(taskGraph, succId, availTime, scheduledTasks, isAicCore);
        }
    }
}

// 轮 1：gap-aware 前向排布，紧耦合调度：C完成后立即调度跨核V后继
void CoreScheduler::GapMinForwardPass(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    std::unordered_map<TargetCoreType, std::vector<std::pair<int, int>>> availTime;
    availTime[TargetCoreType::AIC] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV0] = {{0, INT32_MAX}};
    availTime[TargetCoreType::AIV1] = {{0, INT32_MAX}};
    auto isAicCore = [](TargetCoreType c) { return c == TargetCoreType::AIC; };

    std::set<int> scheduledTasks;

    for (int taskId : topoSeq) {
        if (scheduledTasks.count(taskId) > 0) {
            continue;
        }

        ScheduleOneTask(taskGraph, taskId, availTime, isAicCore);
        scheduledTasks.insert(taskId);

        TryScheduleCrossCoreSuccessors(taskGraph, taskId, availTime, scheduledTasks, isAicCore);
    }
}

// 轮 2.：按拓扑序反向 ALAP 位移 —— 把每个任务右移到 "同核右邻 start" 与 "所有后继最早 start" 的下界，收敛 outbound gap
void CoreScheduler::GapMinBackwardShift(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    // 1. 按核分组并按 startTimeCandidate 排序，建立同核右邻映射
    std::unordered_map<TargetCoreType, std::vector<int>> coreGroups;
    for (auto& t : taskGraph.tasks) {
        coreGroups[t.targetCoreTypeCandidate].push_back(t.idx);
    }
    std::unordered_map<int, int> sameCoreNext; // taskId -> next taskId on same core (-1 if none)
    for (auto& pr : coreGroups) {
        auto& ids = pr.second;
        std::sort(ids.begin(), ids.end(), [&](int i, int j) {
            return taskGraph.tasks[i].startTimeCandidate < taskGraph.tasks[j].startTimeCandidate;
        });
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            sameCoreNext[ids[i]] = ids[i + 1];
        }
        if (!ids.empty()) {
            sameCoreNext[ids.back()] = -1;
        }
    }

    // 2. 按 topo 逆序 shift
    for (auto it = topoSeq.rbegin(); it != topoSeq.rend(); ++it) {
        int taskId = *it;
        auto& t = taskGraph.tasks[taskId];
        if (t.outTasks.empty()) {
            continue;
        }
        int minSuccStart = INT32_MAX;
        for (int s : t.outTasks) {
            minSuccStart = std::min(minSuccStart, taskGraph.tasks[s].startTimeCandidate);
        }
        int sameCoreCap = INT32_MAX;
        auto sit = sameCoreNext.find(taskId);
        if (sit != sameCoreNext.end() && sit->second >= 0) {
            sameCoreCap = taskGraph.tasks[sit->second].startTimeCandidate;
        }
        int newEnd = std::min(minSuccStart, sameCoreCap);
        if (newEnd <= t.endTimeCandidate) {
            continue;
        }
        int delta = newEnd - t.endTimeCandidate;
        t.startTimeCandidate += delta;
        t.endTimeCandidate = newEnd;
    }
}

// 统计所有 AIC<->AIV 双向跨核边的等待间隔之和
int64_t CoreScheduler::SumCrossCoreGap(const TaskGraph& g) const
{
    auto isAic = [](TargetCoreType c) { return c == TargetCoreType::AIC; };
    int64_t sum = 0;
    for (auto& u : g.tasks) {
        for (int vId : u.outTasks) {
            auto& v = g.tasks[vId];
            bool cross = isAic(u.targetCoreType) != isAic(v.targetCoreType);
            if (!cross) {
                continue;
            }
            sum += std::max(0, v.startTime - u.endTime);
        }
    }
    return sum;
}

// 顶层：两轮 + 无条件应用 + 统计日志
void CoreScheduler::GapMinSchedule(TaskGraph& taskGraph, std::vector<int>& topoSeq)
{
    GapMinForwardPass(taskGraph, topoSeq);
    GapMinBackwardShift(taskGraph, topoSeq);
    taskGraph.ApplyCandidateUnconditional();
    APASS_LOG_INFO_F(
        Elements::Operation, "GapMinSchedule total cross-core gap=%lld, endTime(max)=%d.",
        static_cast<long long>(SumCrossCoreGap(taskGraph)), taskGraph.makespan);
}

// 根据节点数量，判断是否遍历所有拓扑序进行任务排布
void CoreScheduler::Schedule(TaskGraph& taskGraph, int bruteForceThreshold)
{
    taskGraph.ClearSchedule();
    APASS_LOG_INFO_F(Elements::Operation, "Start schedule with brute force threshold %d.", bruteForceThreshold);
    if (static_cast<int>(taskGraph.tasks.size()) > bruteForceThreshold) {
        std::vector<int> topoSeq = GetDFSTopoSeq(taskGraph);
        GapMinSchedule(taskGraph, topoSeq);
    } else {
        std::vector<bool> visited(taskGraph.tasks.size(), false);
        std::vector<int> topoList;
        BruteForceScheduleRecursiveStep(visited, 0, taskGraph, topoList);
    }
}

// Alloc op需要与其同级的op处于同一个子图中, Convert的alloc应跟随其后op
void TaskSpliter::BuildSameLayerConnectionWithBack()
{
    for (size_t i = 0; i < opList_.size(); i++) {
        if (ALLOC_OPCODE.count(opList_[i]->GetOpcode()) == 0) {
            continue;
        }
        ScheduleCoreType srcCoreType = opCoreTypes_[i];
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Found alloc op %s[%d].", opList_[i]->GetOpcodeStr().c_str(),
            opList_[i]->GetOpMagic());
        for (auto& oop : opList_[i]->GetOOperands()) {
            for (auto& sameLayerOpPtr : oop->GetProducers()) {
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
void TaskSpliter::BuildSameLayerConnectionWithFront()
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
void TaskSpliter::BuildOpGraph()
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
            auto nextOp = *opList_[i]->ConsumerOps().begin();
            opCoreTypes_[i] = opCoreTypes_[opMagicToIdx_[nextOp->GetOpMagic()]];
        } else if (opList_[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            auto prevOp = *opList_[i]->ProducerOps().begin();
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

// 记录成环的 Cluster 对
void TaskSpliter::RecordCycledClusters(
    const std::vector<ScheduleCoreType> &clusterCoreTypes,
    const std::vector<std::vector<int>> &sccResult)
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
        std::string sccInfo = "SCC " + std::to_string(sccId) + " has cycle with "
            + std::to_string(sccResult[sccId].size()) + " clusters: [";
        for (size_t j = 0; j < sccResult[sccId].size(); j++) {
            int cid = sccResult[sccId][j];
            sccInfo += "(cluster=" + std::to_string(cid)
                + ", core=" + ScheduleCoreTypeToString(clusterCoreTypes[cid]) + ")";
            if (j + 1 < sccResult[sccId].size()) {
                sccInfo += ", ";
            }
        }
        sccInfo += "]";
        APASS_LOG_WARN_F(Elements::Operation, "%s", sccInfo.c_str());

        // 如果同时包含 AIC 和 AIV，FlattenSCC 后会拆成两个新 Cluster,CombineSCC 会消除 SCC 内部边,故记录下这对关系
        if (hasAIC && hasAIV) {
            APASS_LOG_WARN_F(Elements::Operation,
                "SCC %d contains both AIC and AIV clusters, "
                "potential cycle after FlattenSCC between the resulting AIC-group and AIV-group taskNodes.", sccId);
            // 标记：这个 SCC 包含的所有 cluster ID，后续在 CombineSCC 映射后
            // 会被映射到新的 taskNode ID。我们在 SplitGraph 结束后再进行映射。
            cycledSCCClusters_.push_back(sccResult[sccId]);
        }
    }

    if (cycledSCCClusters_.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "No AIC-AIV mixed cycles detected in SCC results.");
    } else {
        APASS_LOG_WARN_F(Elements::Operation,
            "Detected %zu SCC(s) with AIC-AIV mixed cycles, will record for post-schedule reorder.",
            cycledSCCClusters_.size());
    }
}
// mix子图切分主函数
void TaskSpliter::SplitGraph(const std::vector<Operation*>& opList)
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
void TaskSpliter::RecordIDMap(std::unordered_map<int, int>& oldClusterToNewCluster,
    std::vector<ScheduleCoreType>& clusterCoreTypes)
{
    for (auto &oldClusters : cycledSCCClusters_) {
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
                APASS_LOG_INFO_F(Elements::Operation,
                    "Recorded cycled taskNode pair: taskNode %d (AIC) <-> taskNode %d (AIV).", aicId, aivId);
            }
        }
    }
}

// 将强连通分量展开，并构建新的连接图
void TaskSpliter::CombineSCC(
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
void TaskSpliter::BuildInOutGraph(
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
TaskGraph TaskSpliter::BuildTaskGraph()
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
void TaskSpliter::ReverseDFSFindByOutputMemType(int opIdx, MemoryType targetMemType, std::vector<int>& result, std::vector<bool>& visited)
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

// Union 同核操作
void TaskSpliter::UnionSameCoreOps(DSUWithOrder& dsu)
{
    for (size_t idx = 0; idx < opOutGraph_.size(); idx++) {
        // 判断后接 tensor 为 L1 且存在多个消费者时，不进行 union
        bool skip = false;
        if (opList_[idx]->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
            opOutGraph_[idx].size() > 1) {
            skip = true;
            APASS_LOG_DEBUG_F(Elements::Operation, "Skip union op: %s[%d]",
                opList_[idx]->GetOpcodeStr().c_str(), opList_[idx]->GetOpMagic());
        }
        for (int nextOpIdx : opOutGraph_[idx]) {
            if (opCoreTypes_[idx] == opCoreTypes_[nextOpIdx] && !skip &&
                opList_[nextOpIdx]->GetOpcodeStr().find("L1_TO_L0") == std::string::npos) {
                dsu.Union(idx, nextOpIdx);
            }
        }
    }
}

// Union 同层连接
void TaskSpliter::UnionSameLayerConnections(DSUWithOrder& dsu)
{
    for (auto pr : sameLayerConnection_) {
        if (opCoreTypes_[pr.first] == opCoreTypes_[pr.second]) {
            dsu.Union(pr.first, pr.second);
        }
    }
}

// Union AIC->AIV 跨核操作
void TaskSpliter::UnionCrossCoreAICToAIV(DSUWithOrder& dsu)
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
void TaskSpliter::UnionL0CToL1CopyIn(DSUWithOrder& dsu)
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
            for (auto& consumer : opList_[l1CopyInOpIdx]->ConsumerOps()) {
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

// 根据op的CoreType构建连通集
int TaskSpliter::BuildCluster(std::vector<int>& clusterIds, std::vector<ScheduleCoreType>& clusterCoreTypes)
{
    DSUWithOrder dsu(opList_.size());
    UnionSameCoreOps(dsu);
    UnionSameLayerConnections(dsu);
    UnionCrossCoreAICToAIV(dsu);
    UnionL0CToL1CopyIn(dsu);
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
inline bool NoDepDeteched(const std::vector<int>& currOldTasks, int currTaskId, DAGReachableJudger& judger)
{
    for (int oldTaskId : currOldTasks) {
        if (judger.IsReachable(oldTaskId, currTaskId)) {
            return false;
        }
    }
    return true;
}

// 根据taskNode在模拟泳道图中的位置和可达性，返回合并后的taskNode列表
std::vector<std::vector<int>> TaskSpliter::FindMergeableTaskNodes()
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
            if (oldTasks.empty() || NoDepDeteched(oldTasks, currTaskId, reachableJudger)) {
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
void TaskSpliter::MergeTask()
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
void TaskSpliter::MergeTaskByTargetCoreType()
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

// 根据划分结果标记op的AIVCore与internalSubgraphID
void TaskSpliter::MarkInternalSubgraphID()
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
std::vector<Operation*> TaskSpliter::GetMergedOperations()
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
        for (auto& nextOpPtr : ele.second->ConsumerOps()) {
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
