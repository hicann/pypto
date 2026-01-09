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
 * \file core_assign.h
 * \brief
 */

#ifndef PASS_CORE_ASSIGN_H
#define PASS_CORE_ASSIGN_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <queue>
#include <functional>

#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

enum class ScheduleCoreType {
    AIC = 0,
    AIV = 1
};

enum class TargetCoreType {
    AIC = 0,
    AIV0 = 1,
    AIV1 = 2,
    UNKNOWN = 3
};

// 切分后的AIC或AIV子图
class TaskNode {
public:
    TaskNode(const std::string &taskName, int index, ScheduleCoreType taskCoreType, int taskLatency) :
        name(taskName), idx(index), coreType(taskCoreType), latency(taskLatency) {}
    std::string name;
    int idx;
    ScheduleCoreType coreType;
    int latency;
    std::vector<int> inTasks;
    std::vector<int> outTasks;
    TargetCoreType targetCoreType{TargetCoreType::UNKNOWN};
    int startTime{0};
    int endTime{0};
    TargetCoreType targetCoreTypeCandidate{TargetCoreType::UNKNOWN};
    int startTimeCandidate{0};
    int endTimeCandidate{0};
    std::vector<Operation *> opList_;
};

// 完整的mix子图
class TaskGraph {
public:
    void ApplyCandidate();
    int AddTask(const std::string &name, ScheduleCoreType coreType, int latency);
    void AddDependency(int src, int dst);
    void ClearSchedule();
    std::vector<TaskNode> tasks;
    int makespan{-1};
};

// 用于将切分后的AIC和AIV子图调度到AIC,AIV0和AIV1核心上
class CoreScheduler {
public:
    void FindEarliestSlot(std::vector<std::pair<int,int>> &timeSlot, int earliestStart, int latency, int &currentIdx, std::pair<int,int> &currentInterval);
    void UpdateInterval(std::vector<std::pair<int,int>> &timeSlot, int &insertIdx, std::pair<int,int> &insertInterval);
    std::vector<int> GetDFSTopoSeq(TaskGraph &taskGraph);
    void EFTWithInsertSchedule(TaskGraph &taskGraph, std::vector<int> &topoSeq);
    void EFTSchedule(TaskGraph &taskGraph, std::vector<int> &topoSeq);
    void BruteForceScheduleRecursiveStep(std::vector<bool> &visited, int recursiveLevel, TaskGraph &taskGraph, std::vector<int> &topoList);
    void Schedule(TaskGraph &taskGraph, int bruteForceThreshold);
};

// 并查集
class DSUWithOrder {
public:
    DSUWithOrder(int num);
    int Find(int i);
    void Union(int i, int j);
    std::vector<int> parent;
};

// 用于进行子图切分，任务排布和internalSubgraphId写回
class TaskSpliter {
public:
    void SplitGraph(const std::vector<Operation *> &opList); // 此处opList必须符合拓扑序
    void BuildOpGraph();
    void BuildInOutGraph(std::vector<std::set<int>> &inGraph, std::vector<std::set<int>> &outGraph,
        std::vector<int> &clusterIds, int clusterNum);
    TaskGraph BuildTaskGraph();
    void BuildSameLayerConnectionWithBack();
    void BuildSameLayerConnectionWithFront();
    int BuildCluster(std::vector<int> &clusterIds, std::vector<ScheduleCoreType> &clusterCoreTypes);
    std::vector<std::vector<int>> FindMergeableTaskNodes();
    void MergeTask();
    void MergeTaskByTargetCoreType();
    void MarkInternalSubgraphID();
    void CombineSCC(std::vector<int> &clusterIds, std::vector<ScheduleCoreType> &clusterCoreTypes,
        std::vector<std::set<int>> &inGraph, std::vector<std::set<int>> &outGraph, std::vector<std::vector<int>> &sccResult);
    TaskGraph &GetTaskGraph() {return taskGraph_;}
    std::vector<Operation*> GetMergedOperations();
    std::vector<Operation *> opList_;
    std::vector<ScheduleCoreType> opCoreTypes_;
    std::vector<std::set<int>> opInGraph_;
    std::vector<std::set<int>> opOutGraph_;
    std::vector<std::pair<int, int>> sameLayerConnection_;
    std::vector<std::vector<int>> taskIdToOps_;
    std::vector<int> opIdxToTaskId_;
    std::vector<ScheduleCoreType> taskCoreTypes_;
    std::vector<std::set<int>> inGraph_;
    std::vector<std::set<int>> outGraph_;
    std::unordered_map<int, int> opMagicToIdx_;
    TaskGraph taskGraph_;
};

// 使用TarJan算法寻找强连通分量
class StrongConnectionComponentFinder {
public:
    void Find(std::vector<std::set<int>> &inGraph,
        std::vector<std::set<int>> &outGraph, std::vector<std::vector<int>> &sccResult);
    void TarJanAlg(int idx, std::vector<std::set<int>> &outGraph, std::vector<std::vector<int>> &sccResult);
    std::vector<std::vector<int>> strongConnectionComponent_;
    int index_;
    std::vector<int> dfn_;
    std::vector<int> low_;
    std::vector<int> stack_;
    std::vector<bool> instack_;
    std::unordered_set<int> visited_;
};

// 使用传递闭包判断有向无环图中节点可达性
class DAGReachableJudger {
public:
    void Build(const std::vector<std::set<int>> &inGraph, const std::vector<std::set<int>> &outGraph);
    void SetReachable(const int src, const int dst);
    void MergeReachable(int src, int dst);
    bool IsReachable(int src, int dst);
    std::vector<std::vector<uint32_t>> reachableSet;
};
} // namespace npu::tile_fwk
#endif // PASS_CORE_ASSIGN_H