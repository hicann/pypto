/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vf_affinity_sort.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>

namespace npu::tile_fwk {

namespace {
const std::unordered_map<Opcode, int>& GetPreNodePriority()
{
    static const std::unordered_map<Opcode, int> preNodePriority = {
        {Opcode::OP_UB_ALLOC, 0},
        {Opcode::OP_L1_ALLOC, 0},
        {Opcode::OP_L0A_ALLOC, 0},
        {Opcode::OP_L0B_ALLOC, 0},
        {Opcode::OP_L0C_ALLOC, 0},
        {Opcode::OP_BT_ALLOC, 0},
        {Opcode::OP_FIX_ALLOC, 0},
        {Opcode::OP_L1_TO_L0A, 1},
        {Opcode::OP_L1_TO_L0B, 1},
        {Opcode::OP_L1_TO_L0_AT, 1},
        {Opcode::OP_L1_TO_L0_BT, 1},
        {Opcode::OP_L1_TO_FIX, 1},
        {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
        {Opcode::OP_L1_TO_FIX_RELU_PRE, 1},
        {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
        {Opcode::OP_L1_TO_FIX_QUANT_POST, 1},
        {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
        {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1},
        {Opcode::OP_L1_TO_BT, 1},
        {Opcode::OP_COPY_IN, 2},
        {Opcode::OP_UB_COPY_IN, 2},
        {Opcode::OP_L1_COPY_IN, 2},
        {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
        {Opcode::OP_L1_COPY_UB, 2},
        {Opcode::OP_L0C_COPY_UB, 2},
        {Opcode::OP_UB_COPY_L1, 2},
    };
    return preNodePriority;
}
} // namespace

bool VfAffinitySort::IsVfOp(Operation* op) const { return op->GetAtomicScopeId() >= VF_CLUSTER_ID_START; }

bool VfAffinitySort::IsFrontNonVfOp(Operation* op)
{
    if (IsVfOp(op)) {
        return false;
    }
    auto it = frontCache_.find(op);
    if (it != frontCache_.end()) {
        return it->second;
    }
    frontCache_[op] = false;
    for (auto* succ : state_.depManager.GetSuccessors(op)) {
        if (opSet_.count(succ) == 0) {
            continue;
        }
        if (IsVfOp(succ) || IsFrontNonVfOp(succ)) {
            frontCache_[op] = true;
            return true;
        }
    }
    return false;
}

bool VfAffinitySort::IsBackNonVfOp(Operation* op)
{
    if (IsVfOp(op)) {
        return false;
    }
    auto it = backCache_.find(op);
    if (it != backCache_.end()) {
        return it->second;
    }
    backCache_[op] = false;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) == 0) {
            continue;
        }
        if (IsVfOp(pred) || IsBackNonVfOp(pred)) {
            backCache_[op] = true;
            return true;
        }
    }
    return false;
}

bool VfAffinitySort::HasBackNonVfConsumer(Operation* allocOp)
{
    for (auto* succ : state_.depManager.GetSuccessors(allocOp)) {
        if (!IsVfOp(succ) && !IsFrontNonVfOp(succ) && IsBackNonVfOp(succ)) {
            return true;
        }
    }
    return false;
}

int VfAffinitySort::GetEntryCount(Operation* op)
{
    auto it = entryCountCache_.find(op);
    if (it != entryCountCache_.end()) {
        return it->second;
    }
    entryCountCache_[op] = 1;
    int count = 0;
    bool hasPred = false;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) == 0) {
            continue;
        }
        hasPred = true;
        count += GetEntryCount(pred);
    }
    if (!hasPred) {
        count = 1;
    }
    entryCountCache_[op] = count;
    return count;
}

int VfAffinitySort::GetNodePriority(Operation* op) const
{
    const auto& preNodePriority = GetPreNodePriority();
    int prior = 10;
    auto it = preNodePriority.find(op->GetOpcode());
    if (it != preNodePriority.end()) {
        prior = it->second;
    }
    return prior;
}

int64_t VfAffinitySort::GetOutSize(Operation* op) const
{
    int64_t outSize = 0;
    for (auto& oOperand : op->GetOOperands()) {
        if (oOperand != nullptr && oOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR &&
            oOperand->tensor != nullptr) {
            outSize += oOperand->tensor->GetRawDataSize();
        }
    }
    return outSize;
}

int64_t VfAffinitySort::GetBranchNetRelease(Operation* op)
{
    auto it = branchReleaseCache_.find(op);
    if (it != branchReleaseCache_.end()) {
        return it->second;
    }
    // 断环占位: 依赖图若带环, 递归回到未完成的 op 时直接取 0 贡献; 环由上层 Kahn/size 检查报出。
    branchReleaseCache_[op] = 0;
    int64_t total = 0;
    if (!IsVfOp(op)) {
        total = GetOutSize(op);
    } else {
        for (auto* pred : state_.depManager.GetPredecessors(op)) {
            if (opSet_.count(pred) > 0) {
                total += GetBranchNetRelease(pred);
            }
        }
    }
    branchReleaseCache_[op] = total;
    return total;
}

int VfAffinitySort::GetFrontScope(Operation* op)
{
    auto it = frontScopeCache_.find(op);
    if (it != frontScopeCache_.end()) {
        return it->second;
    }
    // 断环占位: 递归回到未完成的 op 时取 0(无 vf scope); 环由上层 Kahn/size 检查报出。
    frontScopeCache_[op] = 0;
    int scopeId = 0;
    for (auto* succ : state_.depManager.GetSuccessors(op)) {
        if (opSet_.count(succ) == 0) {
            continue;
        }
        if (IsVfOp(succ)) {
            scopeId = succ->GetAtomicScopeId();
        } else {
            scopeId = GetFrontScope(succ);
        }
        if (scopeId >= VF_CLUSTER_ID_START) {
            break;
        }
    }
    frontScopeCache_[op] = scopeId;
    return scopeId;
}

int VfAffinitySort::GetBackScope(Operation* op)
{
    auto it = backScopeCache_.find(op);
    if (it != backScopeCache_.end()) {
        return it->second;
    }
    // 断环占位: 递归回到未完成的 op 时取 0(无 vf scope); 环由上层 Kahn/size 检查报出。
    backScopeCache_[op] = 0;
    int scopeId = 0;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) == 0) {
            continue;
        }
        if (IsVfOp(pred)) {
            scopeId = pred->GetAtomicScopeId();
        } else {
            scopeId = GetBackScope(pred);
        }
        if (scopeId >= VF_CLUSTER_ID_START) {
            break;
        }
    }
    backScopeCache_[op] = scopeId;
    return scopeId;
}

void VfAffinitySort::ReverseDfs(Operation* op, std::unordered_set<Operation*>& visited, std::vector<Operation*>& result)
{
    if (visited.count(op) > 0) {
        return;
    }
    visited.insert(op);

    std::vector<Operation*> preds;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) > 0) {
            preds.push_back(pred);
        }
    }
    std::sort(preds.begin(), preds.end(), [this](Operation* a, Operation* b) {
        int priorA = GetNodePriority(a);
        int priorB = GetNodePriority(b);
        if (priorA != priorB) {
            return priorA < priorB;
        }
        int entryA = GetEntryCount(a);
        int entryB = GetEntryCount(b);
        if (entryA != entryB) {
            return entryA > entryB;
        }
        return GetBranchNetRelease(a) > GetBranchNetRelease(b);
    });

    for (auto* pred : preds) {
        ReverseDfs(pred, visited, result);
    }

    result.push_back(op);
}

Status VfAffinitySort::PostProcess(std::vector<Operation*>& result)
{
    std::vector<Operation*> front;
    std::vector<Operation*> mid;
    std::vector<Operation*> back;
    for (auto* op : result) {
        if (IsVfOp(op)) {
            mid.push_back(op);
        } else if (IsAllocOpCode(op->GetOpcode())) {
            if (HasBackNonVfConsumer(op)) {
                back.push_back(op);
            } else {
                front.push_back(op);
            }
        } else if (IsFrontNonVfOp(op)) {
            front.push_back(op);
        } else if (IsBackNonVfOp(op)) {
            back.push_back(op);
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: non-vf op %s is neither front nor back.",
                              state_.GetOpInfo(op).c_str());
            return FAILED;
        }
    }

    result.clear();
    result.insert(result.end(), front.begin(), front.end());
    result.insert(result.end(), mid.begin(), mid.end());
    result.insert(result.end(), back.begin(), back.end());
    return SUCCESS;
}

Status VfAffinitySort::SortOneSuperNode(const std::vector<Operation*>& ops, std::vector<Operation*>& out)
{
    std::unordered_set<Operation*> subset(ops.begin(), ops.end());
    std::unordered_set<Operation*> savedOpSet = std::move(opSet_);
    opSet_ = subset;
    frontCache_.clear();
    backCache_.clear();
    entryCountCache_.clear();
    branchReleaseCache_.clear();

    std::vector<Operation*> exits;
    for (auto* op : ops) {
        bool hasSucc = false;
        for (auto* succ : state_.depManager.GetSuccessors(op)) {
            if (subset.count(succ) > 0) {
                hasSucc = true;
                break;
            }
        }
        if (!hasSucc) {
            exits.push_back(op);
        }
    }

    std::unordered_set<Operation*> visited;
    std::vector<Operation*> subResult;
    subResult.reserve(ops.size());
    for (auto* op : exits) {
        ReverseDfs(op, visited, subResult);
    }

    Status st = SUCCESS;
    if (subResult.size() != ops.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: super-node ordered %zu vs %zu ops (cycle?).",
                          subResult.size(), ops.size());
        st = FAILED;
    } else if (PostProcess(subResult) != SUCCESS) {
        st = FAILED;
    } else {
        out.insert(out.end(), subResult.begin(), subResult.end());
    }

    opSet_ = std::move(savedOpSet);
    return st;
}

Status VfAffinitySort::SortMultiScope(std::vector<Operation*>& result)
{
    // 1. 每个 op 归属一个超节点：scope op → atomicScopeId；非 vf op → 就近 scope / 单例
    std::unordered_map<Operation*, int> opToSuperNode;
    std::map<int, std::vector<Operation*>> superNodeToOps;
    for (auto* op : operations) {
        if (IsVfOp(op)) {
            opToSuperNode[op] = op->GetAtomicScopeId();
        }
    }
    int nextSingletonId = -1;
    for (auto* op : operations) {
        if (IsVfOp(op)) {
            continue;
        }
        int scopeId = GetBackScope(op);
        if (scopeId < VF_CLUSTER_ID_START) {
            scopeId = GetFrontScope(op);
        }
        opToSuperNode[op] = scopeId >= VF_CLUSTER_ID_START ? scopeId : nextSingletonId--;
    }
    for (auto& [op, sn] : opToSuperNode) {
        superNodeToOps[sn].push_back(op);
    }

    // 2. 构建超节点粗粒度图
    std::map<int, int> snToIdx;
    std::vector<int> snIds;
    for (auto& [sn, ops] : superNodeToOps) {
        (void)ops;
        snToIdx[sn] = static_cast<int>(snIds.size());
        snIds.push_back(sn);
    }
    int n = static_cast<int>(snIds.size());
    std::vector<std::set<int>> inGraph(n);
    std::vector<std::set<int>> outGraph(n);
    for (auto* op : operations) {
        int srcIdx = snToIdx[opToSuperNode[op]];
        for (auto* succ : state_.depManager.GetSuccessors(op)) {
            if (opSet_.count(succ) == 0) {
                continue;
            }
            int dstIdx = snToIdx[opToSuperNode[succ]];
            if (srcIdx != dstIdx) {
                outGraph[srcIdx].insert(dstIdx);
                inGraph[dstIdx].insert(srcIdx);
            }
        }
    }

    // 3. Kahn 拓扑排序（纯拓扑，无贪心）
    std::vector<int> indeg(n, 0);
    for (int i = 0; i < n; i++) {
        indeg[i] = static_cast<int>(inGraph[i].size());
    }
    std::queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indeg[i] == 0) {
            q.push(i);
        }
    }
    std::vector<int> topo;
    topo.reserve(n);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        topo.push_back(u);
        for (int v : outGraph[u]) {
            if (--indeg[v] == 0) {
                q.push(v);
            }
        }
    }
    if (static_cast<int>(topo.size()) != n) {
        APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: super-node graph has cycle (%d vs %d).",
                          static_cast<int>(topo.size()), n);
        return FAILED;
    }

    // 4. 逐超节点排序并拼接
    for (int idx : topo) {
        int sn = snIds[idx];
        auto& ops = superNodeToOps[sn];
        if (sn < 0 && ops.size() == 1 && !IsVfOp(ops[0])) {
            result.push_back(ops[0]);
            continue;
        }
        if (SortOneSuperNode(ops, result) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

Status VfAffinitySort::DoSortOps()
{
    opSet_.clear();
    vfOpSet_.clear();
    entryCountCache_.clear();
    branchReleaseCache_.clear();
    frontCache_.clear();
    backCache_.clear();
    frontScopeCache_.clear();
    backScopeCache_.clear();
    for (auto* op : operations) {
        opSet_.insert(op);
        if (IsVfOp(op)) {
            vfOpSet_.insert(op);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "VfAffinitySort: %zu ops, %zu vf ops.", operations.size(), vfOpSet_.size());

    std::set<int> scopeIds;
    for (auto* op : operations) {
        if (IsVfOp(op)) {
            scopeIds.insert(op->GetAtomicScopeId());
        }
    }

    std::vector<Operation*> result;
    result.reserve(operations.size());
    if (scopeIds.size() <= 1) {
        std::vector<Operation*> exits;
        for (auto* op : operations) {
            bool hasSucc = false;
            for (auto* succ : state_.depManager.GetSuccessors(op)) {
                if (opSet_.count(succ) > 0) {
                    hasSucc = true;
                    break;
                }
            }
            if (!hasSucc) {
                exits.push_back(op);
            }
        }
        std::unordered_set<Operation*> visited;
        for (auto* op : exits) {
            ReverseDfs(op, visited, result);
        }
        if (result.size() != operations.size()) {
            APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: ordered %zu vs %zu ops (cycle?).", result.size(),
                              operations.size());
            return FAILED;
        }
        if (PostProcess(result) != SUCCESS) {
            return FAILED;
        }
    } else {
        if (SortMultiScope(result) != SUCCESS) {
            return FAILED;
        }
    }

    operations = std::move(result);
    APASS_LOG_INFO_F(Elements::Operation, "VfAffinitySort: sorted %zu ops.", operations.size());

    if (ExecuteOp() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ExecuteOp failed.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
