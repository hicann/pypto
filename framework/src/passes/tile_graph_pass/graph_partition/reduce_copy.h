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
 * \file reduce_copy.h
 * \brief
 */

#ifndef PASS_REDUCE_COPY_H_
#define PASS_REDUCE_COPY_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/logical_tensor.h"
#include <vector>
#include <set>
#include <unordered_set>
#include <queue>
#include <utility>

namespace npu::tile_fwk {
struct EstimateInput {
    std::vector<int> execTime;
    std::vector<bool> isCube;
    std::vector<std::set<int>> outGraph;
    std::vector<std::set<int>> inGraph;
    int betweenSubgraphScheduleTime{1500};  // estimated task issuance time
};

struct EstimateCoreState {
    int aic;
    int aiv0;
    int aiv1;
};

struct MixScheduleContext {
    std::vector<int> subgraphToMix;
    std::unordered_set<int> candidateSet;
    std::vector<std::set<int>> mixDeps;
    std::vector<int> mixStartTime;
    std::vector<int> mixFinishTime;
    std::vector<int> mixInDegree;
    std::queue<int> mixReadyQueue;
    int numMix;
    int numSubgraph;
};

struct SubgraphScheduleContext {
    std::vector<int> finishTime;
    std::vector<int> inDegree;
    std::queue<int> readyQueue;
    EstimateCoreState coreState;
    int mixStartTime;
    int mixId;
    int numSubgraph;
};

class EstimateExecTime {
public:
    int Estimate(const EstimateInput& input, const std::vector<std::set<int>>& estimateCandidate);
private:
    void InitMixData(MixScheduleContext& ctx, const std::vector<std::set<int>>& estimateCandidate);
    void BuildMixDeps(MixScheduleContext& ctx, const EstimateInput& input);
    void InitMixTopology(MixScheduleContext& ctx);
    int CalcMixStartTime(int mixId, const MixScheduleContext& ctx, int scheduleTime);
    void InitSubgraphContext(SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx,
        const EstimateInput& input);
    void ScheduleOneSubgraph(int current, SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx,
        const EstimateInput& input);
    void ProcessSubgraphConsumers(int current, SubgraphScheduleContext& subCtx,
        const MixScheduleContext& ctx, const EstimateInput& input);
    int GetMixFinishTime(const SubgraphScheduleContext& subCtx, const MixScheduleContext& ctx);
    void ProcessMixConsumers(int mixId, MixScheduleContext& ctx);
};

struct MergeInput {
    int numSubgraph;
    int maxLatency;
    std::pair<double, double> aivRatio;
    std::vector<int> subgraphAICLatency;
    std::vector<int> subgraphAIVLatency;
    std::vector<std::set<int>> subGraphInGraph;
    std::vector<std::set<int>> subGraphOutGraph;
    std::vector<std::vector<int>> mergeGroup;
    std::vector<bool> isEnforceMergeGroup;
    std::vector<bool> isValidMergeGroup;
};

struct MergeOutput {
    int numSubgraphUpdated;
    std::vector<int> subgraphIdUpdated;
};

class MixGraphMerger {
public:
    MixGraphMerger() = default;
    ~MixGraphMerger() = default;

    MergeOutput Merge(const MergeInput& input);

private:
    MergeInput mInput;
    MergeOutput mOutput;
    EstimateInput estimateInput;
    std::vector<int> mParent;
    std::vector<int> mRank;

    void Initialize(const MergeInput& input);
    int FindParent(int x);
    void UnionSets(int x, int y);
    bool CanMergeWithoutCycle(const std::vector<int>& actualGroup);
    bool CanMergeWithConstraints(const std::vector<int>& actualGroup);
    void PerformMerge(const std::vector<int>& actualGroup);
    void UpdateOutput();
    bool CheckLatencyConstraint(const std::vector<int>& actualGroup);
    bool CheckMergeBenefit(const std::vector<int>& actualGroup);
    std::vector<int> GetActualGroup(const std::vector<int>& group);
    void BuildMergedGraph(std::vector<std::set<int>>& outGraph,
                          std::vector<std::set<int>>& inGraph);
    bool HasCycle(const std::vector<std::set<int>>& outGraph,
                  const std::vector<std::set<int>>& inGraph);
};

class ReduceCopyMerge : public Pass {
public:
    ReduceCopyMerge() : Pass("ReduceCopyMerge")
    {
        SetSupportedArches({NPUArch::DAV_3510});
    }
    ~ReduceCopyMerge() override = default;
private:
    Status BuildGraph(Function &function, MergeInput& mergeInput);
    Status BuildMergeGroup(Function &function, MergeInput& mergeInput);
    std::unordered_set<int> FindForkSubgraph(Function &function);
    Status MarkNoMergeSubgraph(Function &function);
    void UpdateConnectRecord(Function &function);
    bool IsEnforceMergeBoundary(LogicalTensorPtr &tensor);
    Status RunOnFunction(Function &function) override;
    Status PostCheck(Function &function) override;
    std::unordered_set<int> noMergeSubgraph;
    std::map<std::vector<int>, int> mergeGroupToPriority;
    std::set<std::vector<int>> enforceMergeGroup;
    std::vector<int> subgraphInputSize;
    std::vector<int> subgraphOutputSize;
};

}
#endif
