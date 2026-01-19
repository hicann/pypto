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
 * \file l1_copy_reuse.h
 * \brief
 */

#ifndef PASS_L1_COPY_REUSE_H_
#define PASS_L1_COPY_REUSE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/log.h"
#include "passes/statistics/tensor_and_tile_graph_statistic.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "L1CopyInReuseMerge"

namespace npu::tile_fwk {
class L1CopyInReuseRunner {
  public:
    explicit L1CopyInReuseRunner(const std::vector<std::vector<int>> &inGraph1) : inGraph(inGraph1) {}
    ~L1CopyInReuseRunner() {}
    Status Run(Function &func, int color, std::vector<std::vector<int>> &colorNode);
  private:
    void GetOpHash(std::vector<uint64_t> &hashList, const std::string op, int idx);
    void GetColorHash(const OperationsViewer &opOriList, std::vector<uint64_t> &hashColor);
    int GetMaxInColor(const std::vector<int> &nodes, const OperationsViewer &opOriList, int curColor);
    Status MergeDupL1CopyIn(Function &func, std::vector<std::vector<int>> &colorNode, int color);
    void MergeProcessIdUpdate(Function &func, std::vector<std::vector<int>> &colorNode, int color);
    std::vector<int> GetOpInputFeature(const OperationsViewer &opOriList,
                                      const int opIdx, const int ioperandIdx);
    void RemoveUselessViews(Function &func) const;
    Status GetDuplicateOps(std::vector<Operation *> &opOriList, const std::vector<int> &opIdx);
    void TackleOp(int i, Operation *op, std::vector<std::vector<int>> &replacedInputs,
                        std::vector<std::vector<int>> &replacedOutputs);
    Status Phase1(Function &func, int color, std::vector<std::vector<int>> &colorNode,
                  std::vector<int> &colorCopyIn, std::vector<uint64_t> &hashColor);
    void GetL1ReuseOpOrder(std::vector<std::pair<int, int>> &opOrder,
                    std::map<uint64_t, int> &mgRem, std::vector<int> &numLRList, std::vector<uint64_t> &hashColor, int color);
    bool GetMergedL1(int maxInColor, std::vector<int> &mergedNum, int maxMergeNum, int &tmpColor, int i,
                    std::map<std::vector<uint64_t>, int> &l1InputList, std::vector<uint64_t> &vec, std::vector<int> &colorCopyIn,
                    std::map<uint64_t, int> &mgRem, int idx);
    Status L1MergeProcess(OperationsViewer &opOriList, std::vector<std::vector<int>> &colorNode,
                          std::vector<uint64_t> &hashColor, std::vector<int> &colorCopyIn,
                          std::map<std::vector<uint64_t>, int> &l1InputList, int &tmpColor,
                          std::vector<int> &mergedNum, int &i);
    void CubeMergeProcess(std::vector<std::vector<int>> &colorNode, OperationsViewer &opOriList,
                          std::vector<int> &hashMergeNum, std::vector<int> &colorCopyIn);
    Status SetNumLR(std::vector<int> &numLRList);
    Status SetNumDB(std::vector<int> &numDBList);
    const std::vector<std::vector<int>> &inGraph;
    std::unordered_map<int, int> replacedCopyMap_;
    std::unordered_map<int, int> tensormagic2Op_;
    std::unordered_map<uint64_t, std::vector<int>> hashMap;
    std::unordered_map<uint64_t, int> hashOrder;
    std::map<int64_t, int64_t> numLRMap;
    std::map<int64_t, int64_t> numDBMap;
    int mgCopyInUpperBound;
    int L1ReuseMode;
    int cubeNBufferMode;
};

class L1CopyInReuseMerge : public Pass {
public:
    L1CopyInReuseMerge() : Pass("L1CopyInReuseMerge") {}
    ~L1CopyInReuseMerge() override = default;

private:
    Status InitColorNode(Function &func, std::vector<std::vector<int>> &colorNode) const;
    Status CheckOpListValid(Function &func) const;
    Status L1CopyInReuse(Function &func) const;
    Status RunOnFunction(Function &function) override {
        APASS_LOG_INFO_F(Elements::Operation, "===> Start L1CopyInReuseMerge.");
        if (L1CopyInReuse(function) == FAILED) {
          return FAILED;
        }
        DeadOperationEliminator eliminator;
        eliminator.EliminateDeadOperationBackward(function);
        APASS_LOG_INFO_F(Elements::Operation, "===> Finish L1CopyInReuseMerge.");
        return SUCCESS;
    }
    void DoHealthCheckAfter(Function &function, const std::string &folderPath) override;
};
} // namespace npu::tile_fwk
#endif // PASS_L1_COPY_REUSE_H_