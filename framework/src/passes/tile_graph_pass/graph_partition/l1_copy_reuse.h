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

#include <functional>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

#include "passes/statistics/tensor_and_tile_graph_statistic.h"
#include "passes/pass_log/pass_log.h"
#include "passes/tile_graph_pass/graph_partition/hash_order_utils.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "L1CopyInReuseMerge"

namespace npu::tile_fwk {
class L1CopyInReuseRunner {
public:
    explicit L1CopyInReuseRunner(const std::vector<std::vector<int>>& inGraph1) : inGraph_(inGraph1) {}
    ~L1CopyInReuseRunner() {}
    Status Run(Function& func, int color, std::vector<std::vector<int>>& colorNode);
    static bool CanReuse(const Operation& op);
    // Returns true if op's L1 output is consumed by a copy into the left(L0A)/right(L0B)
    // matrix of a downstream cube op. Public+static so unit tests can assert side detection.
    static bool IsLeftMatrixCopy(const Operation& op);
    static bool IsRightMatrixCopy(const Operation& op);
    static int GetModeBySetting(const std::map<int64_t, int64_t>& setting,
                                const std::map<std::string, int64_t>& settingByFunc);
    static std::vector<int> GetCopyIn(const OperationsViewer& opOriList, int color,
                                      std::vector<std::vector<int>>& colorNode);
    static void GetOriList(Function& func, std::vector<Operation*>& oriList);

private:
    void GetOpHash(std::vector<uint64_t>& hashList, const std::string op, int idx);
    void GetColorHash(const Function& func, const OperationsViewer& opOriList, std::vector<uint64_t>& hashColor,
                      const std::vector<std::vector<int>>& colorNode);
    int GetMaxInColor(const std::vector<int>& nodes, const OperationsViewer& opOriList, int curColor);
    Status MergeDupL1CopyIn(Function& func, std::vector<std::vector<int>>& colorNode, int color);
    void MergeProcessIdUpdate(Function& func, std::vector<std::vector<int>>& colorNode, int color);
    std::vector<int> GetOpInputFeature(const OperationsViewer& opOriList, const int opIdx, const int ioperandIdx);
    void RemoveUselessViews(Function& func) const;
    Status GetDuplicateOps(std::vector<Operation*>& opOriList, const std::vector<int>& opIdx);
    void TackleOp(int i, Operation* op, std::vector<std::vector<int>>& replacedInputs,
                  std::vector<std::vector<int>>& replacedOutputs);
    Status Phase1(Function& func, int color, std::vector<std::vector<int>>& colorNode, std::vector<int>& colorCopyIn,
                  std::vector<uint64_t>& hashColor);
    void GetL1ReuseOpOrder(std::vector<std::pair<int, int>>& opOrder, std::map<uint64_t, int>& mgRem,
                           std::vector<int>& numLRList, std::vector<uint64_t>& hashColor, int color);
    bool GetMergedL1(int maxInColor, std::vector<int>& mergedNum, int maxMergeNum, int& tmpColor, int i,
                     std::map<std::vector<uint64_t>, int>& l1InputList, std::vector<uint64_t>& vec,
                     std::vector<int>& colorCopyIn, std::map<uint64_t, int>& mgRem, uint64_t idx);
    // Build the per-subgraph matrix-side list (0=auto, 1=left, 2=right) from the
    // global / by-func / by-label side settings, mirroring numLRList expansion.
    Status BuildMatrixSideList(const Function& func, const OperationsViewer& opOriList, std::vector<int>& numLRSideList,
                               const std::vector<uint64_t>& hashColor, int color);
    // Resolve the per-subgraph merge-count list (numLRList) from the global/func/label settings.
    Status BuildMergeCountList(const Function& func, const OperationsViewer& opOriList, std::vector<int>& numLRList,
                               const std::vector<uint64_t>& hashColor, int color);
    // DEBUG-log one side-tagged subgraph's per-step outcome (merged on the requested side, or
    // no same-side partner yet / merge anchor). Aggregate numbers are summarised after the loop.
    void RecordSideMergeOutcome(int subgraphIdx, int side, int tmpColor);
    // Search colorNode[subgraphIdx] for a merge candidate, optionally restricted by
    // `filter` (e.g. left/right matrix). Sets tmpColor when a candidate is found.
    Status FindMergeCandidate(const OperationsViewer& opOriList, int subgraphIdx, int maxInColor,
                              std::vector<int>& mergedNum, std::vector<int>& numLRList, int& tmpColor,
                              std::vector<std::vector<int>>& colorNode,
                              std::map<std::vector<uint64_t>, int>& l1InputList, std::vector<int>& colorCopyIn,
                              std::map<uint64_t, int>& mgRem, std::vector<uint64_t>& hashColor,
                              const std::function<bool(const Operation&)>& filter);
    Status L1MergeProcess(OperationsViewer& opOriList, std::vector<std::vector<int>>& colorNode,
                          std::vector<uint64_t>& hashColor, std::vector<int>& colorCopyIn,
                          std::map<std::vector<uint64_t>, int>& l1InputList, int& tmpColor, std::vector<int>& mergedNum,
                          int& i, int side);
    void CubeMergeProcess(std::vector<std::vector<int>>& colorNode, OperationsViewer& opOriList,
                          std::map<int, int>& hashMergeNumMap, std::vector<int>& colorCopyIn);
    Status SetNumFromConfig(const Function& func, const std::map<int64_t, int64_t>& configMap,
                            const std::map<std::string, int64_t>& configMapByFunc, std::map<int, int>& resultMap,
                            const std::string& configName);
    Status ApplyByFuncConfig(int currentFuncMagic, const std::map<std::string, int64_t>& configMapByFunc,
                             std::map<int, int>& resultMap, const std::string& configName);
    Status ApplyGlobalConfig(const std::map<int64_t, int64_t>& configMap, std::map<int, int>& resultMap,
                             const std::string& configName);
    void HashUpdate(Function& func, int color, const std::vector<uint64_t>& hashColor, OperationsViewer& opOriList,
                    std::vector<std::vector<int>>& colorNode);
    Status ApplySemanticLabelSettingsL1Reuse(const OperationsViewer& opOriList,
                                             const std::map<std::string, int64_t>& labelMap,
                                             std::vector<int>& numLRList, const std::vector<uint64_t>& hashColor,
                                             int color);
    Status ApplySemanticLabelSettingsCubeNBuffer(const OperationsViewer& opOriList, std::map<int, int>& hashMergeNumMap,
                                                 const std::vector<uint64_t>& hashColor, int color);
    const std::vector<std::vector<int>>& inGraph_;
    std::unordered_map<int, int> replacedCopyMap_;
    std::unordered_map<int, int> tensormagic2Op_;
    std::unordered_map<uint64_t, std::vector<int>> hashMap_;
    std::map<uint64_t, int> hashOrder_; // Deterministic ordered map for compilation repeatability
    std::map<int64_t, int64_t> numLRMap_;
    std::map<int64_t, int64_t> numDBMap_;
    std::map<std::string, int64_t> numLRMapByFunc_;
    std::map<std::string, int64_t> numDBMapByFunc_;
    std::map<std::string, int64_t> numLRMapByLabel_;
    std::map<std::string, int64_t> numDBMapByLabel_;
    // L1 reuse matrix-side preference (1=left/L0A, 2=right/L0B), shares keys with numLR*.
    std::map<int64_t, int64_t> numLRSideMap_;
    std::map<std::string, int64_t> numLRSideMapByFunc_;
    std::map<std::string, int64_t> numLRSideMapByLabel_;
    int mgCopyInUpperBound_;
    int L1ReuseMode_;
    int cubeNBufferMode_;
    std::set<int> mulaccGraph_;
};

class L1CopyInReuseMerge : public Pass {
public:
    L1CopyInReuseMerge() : Pass("L1CopyInReuseMerge") {}
    ~L1CopyInReuseMerge() override = default;

private:
    Status InitColorNode(Function& func, std::vector<std::vector<int>>& colorNode) const;
    Status CheckOpListValid(Function& func) const;
    Status L1CopyInReuse(Function& func) const;
    Status RunOnFunction(Function& function) override
    {
        APASS_LOG_INFO_F(Elements::Operation, "===> Start L1CopyInReuseMerge.");
        if (L1CopyInReuse(function) == FAILED) {
            return FAILED;
        }
        DeadOperationEliminator eliminator;
        eliminator.EliminateDeadOperationBackward(function);
        APASS_LOG_INFO_F(Elements::Operation, "===> Finish L1CopyInReuseMerge.");
        return SUCCESS;
    }
    void DoHealthCheckAfter(Function& function, const std::string& folderPath) override;
};
} // namespace npu::tile_fwk
#endif // PASS_L1_COPY_REUSE_H_
