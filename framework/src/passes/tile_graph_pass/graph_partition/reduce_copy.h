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
#include <utility>

namespace npu::tile_fwk {
struct MergeInput {
    int numSubgraph;
    int maxLatency;
    std::pair<double, double> aivRatio;
    std::vector<int> subgraphAICLatency;
    std::vector<int> subgraphAIVLatency;
    std::vector<std::set<int>> subGraphOutGraph;
    std::vector<std::vector<int>> mergeGroup;
    std::vector<bool> isEnforceMergeGroup;
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
    Status MarkNoMergeSubgraph(Function &function);
    bool IsEnforceMergeBoundary(LogicalTensorPtr &tensor);
    Status RunOnFunction(Function &function) override;
    Status PostCheck(Function &function) override;
    std::unordered_set<int> noMergeSubgraph;
};

}
#endif
