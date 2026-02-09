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
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class DSU {
public:
    DSU() = default;
    DSU(int n, const std::vector<int>& nodeWeights, std::vector<OpCoreType> &colorCoreType);
    int Find(int i);
    void Union(int i, int j);
    std::pair<int, int> GetWeight(int i);
    void ResetLink(int i);

    std::vector<int> parent;
    std::vector<int> AIVSupernodeWeights;
    std::vector<int> AICSupernodeWeights;
    std::vector<int> AIVSingleWeights;
    std::vector<int> AICSingleWeights;
    std::vector<OpCoreType> coreType;
};

class ReduceCopyRunner {
public:
    Status ReduceCopy(Function &func);
    Status Init(Function &func); 
    Status MergePrepare(std::vector<std::tuple<int, int, size_t>> &candidates, std::map<int, int> &rootToDense);
    Status MergeLoop(std::vector<std::tuple<int, int, size_t>> &candidates, const std::pair<double, double> &thres,
    bool &mergedInLoop, std::map<int, int> &rootToDense);
    Status RemarkInternalSubgraphID(Function &func);
    void BuildGraph(const OperationsViewer opOriList);
    void BuildGraphInner(const OperationsViewer &opOriList, int opIdx, int opColor);
    std::map<int, size_t> magic2Size;
    std::map<std::pair<int, int>, std::set<int>> originalEdges;
    std::set<std::pair<int, int>> crossEdges;
    std::vector<std::set<int>> superNodeInGraph;
    std::vector<std::set<int>> superNodeOutGraph;
    std::vector<bool> isReshape;
    std::vector<OpCoreType> colorCoreType;
    std::vector<std::vector<size_t>> colorNode;
    std::vector<std::pair<double, double>> mergeThresholds;
    std::unordered_set<int> mergedGraphId;
    std::unordered_set<int> currMergedGraphId;
    DSU dsu;
    int upperBound{10000};
    int color;
};

class ReduceCopyMerge : public Pass {
public:
    ReduceCopyMerge() : Pass("ReduceCopyMerge") {
        SetSupportedArches({NPUArch::DAV_UNKNOWN});
    }
    ~ReduceCopyMerge() override = default;
private:
    Status RunOnFunction(Function &function) override;
    Status PostCheck(Function &function) override;
};

}
#endif