/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PASS_VF_AFFINITY_SORT_H
#define PASS_VF_AFFINITY_SORT_H

#include "optimize_sort.h"

#include <unordered_map>
#include <unordered_set>

namespace npu::tile_fwk {

class VfAffinitySort : public OptimizeSort {
public:
    using OptimizeSort::OptimizeSort;

private:
    Status DoSortOps() override;
    bool IsVfOp(Operation* op) const;
    bool IsFrontNonVfOp(Operation* op);
    bool IsBackNonVfOp(Operation* op);
    bool HasBackNonVfConsumer(Operation* allocOp);
    int GetEntryCount(Operation* op);
    int GetNodePriority(Operation* op) const;
    int64_t GetOutSize(Operation* op) const;
    int64_t GetBranchNetRelease(Operation* op);
    int GetFrontScope(Operation* op);
    int GetBackScope(Operation* op);
    Status SortMultiScope(std::vector<Operation*>& result);
    Status SortOneSuperNode(const std::vector<Operation*>& ops, std::vector<Operation*>& out);
    void ReverseDfs(Operation* op, std::unordered_set<Operation*>& visited, std::vector<Operation*>& result);
    Status PostProcess(std::vector<Operation*>& result);

    std::unordered_set<Operation*> opSet_;
    std::unordered_set<Operation*> vfOpSet_;
    std::unordered_map<Operation*, int> entryCountCache_;
    std::unordered_map<Operation*, int64_t> branchReleaseCache_;
    std::unordered_map<Operation*, int> frontScopeCache_;
    std::unordered_map<Operation*, int> backScopeCache_;
    std::unordered_map<Operation*, bool> frontCache_;
    std::unordered_map<Operation*, bool> backCache_;
};

} // namespace npu::tile_fwk
#endif // PASS_VF_AFFINITY_SORT_H
