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
 * \file iso_matcher.cpp
 * \brief Implementation of IsoMatchChains and FindTaskEntryOps.
 */

#include "passes/block_graph_pass/schedule_ooo/common/iso_matcher.h"

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/operation/operation.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "IsoMatcher"

namespace npu::tile_fwk {

namespace {

// 签名由 opcode、输出数量、每个输出 tensor 的 MemoryType 和 shape 组成。
// alloc 节点额外追加 (基于唯一输出 tensor 的 GetProducers/GetConsumers):
//   || prod:<count> | <prodOpcode>:<outIdx> | ...   (生产者: opcode + tensor 是生产者第几输出)
//   || cons:<count> | <consOpcode>:<inIdx> | ...     (消费者: opcode + tensor 是消费者第几输入)
// 非 alloc 节点额外追加:
//   || pred:<count> | <predOpcode> | ...   (前驱 opcode 多集)
//   || succ:<count> | <succOpcode> | ...   (后继 opcode 多集)
void AppendOutputTensorSig(const Operation* op, std::ostringstream& oss)
{
    for (const auto& out : op->GetOOperands()) {
        oss << '|';
        if (out == nullptr) {
            oss << "null";
            continue;
        }
        oss << static_cast<int>(out->GetMemoryTypeOriginal()) << ':';
        for (auto dim : out->GetShape()) {
            oss << dim << ',';
        }
    }
}

void AppendSortedOpcodes(const std::string& tag, const std::vector<int>& opcodes, std::ostringstream& oss)
{
    auto sortedOpcodes = opcodes;
    std::sort(sortedOpcodes.begin(), sortedOpcodes.end());
    oss << tag << sortedOpcodes.size();
    for (int code : sortedOpcodes) {
        oss << '|' << code;
    }
}

void AppendNonAllocNeighborSig(const Operation* op, std::ostringstream& oss)
{
    std::vector<int> predOpcodes;
    for (Operation* pred : op->ProducerOps()) {
        if (pred == nullptr)
            continue;
        predOpcodes.push_back(static_cast<int>(pred->GetOpcode()));
    }
    AppendSortedOpcodes("||pred:", predOpcodes, oss);

    std::vector<int> succOpcodes;
    for (Operation* succ : op->ConsumerOps()) {
        if (succ == nullptr)
            continue;
        succOpcodes.push_back(static_cast<int>(succ->GetOpcode()));
    }
    AppendSortedOpcodes("||succ:", succOpcodes, oss);
}

std::vector<std::string> BuildProducerEntries(const LogicalTensorPtr& tensor)
{
    std::vector<std::string> entries;
    for (Operation* prod : tensor->GetProducers()) {
        if (prod == nullptr)
            continue;
        const auto& prodOutputs = prod->GetOOperands();
        for (size_t j = 0; j < prodOutputs.size(); ++j) {
            if (prodOutputs[j] == tensor) {
                entries.push_back(std::to_string(static_cast<int>(prod->GetOpcode())) + ":" + std::to_string(j));
                break;
            }
        }
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

std::vector<std::string> BuildConsumerEntries(const LogicalTensorPtr& tensor)
{
    std::vector<std::string> entries;
    for (Operation* cons : tensor->GetConsumers()) {
        if (cons == nullptr)
            continue;
        const auto& consInputs = cons->GetIOperands();
        for (size_t i = 0; i < consInputs.size(); ++i) {
            if (consInputs[i] == tensor) {
                entries.push_back(std::to_string(static_cast<int>(cons->GetOpcode())) + ":" + std::to_string(i));
                break;
            }
        }
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

void AppendEntries(const std::string& tag, const std::vector<std::string>& entries, std::ostringstream& oss)
{
    oss << tag << entries.size();
    for (const auto& entry : entries) {
        oss << '|' << entry;
    }
}

void AppendAllocTensorRelationSig(const Operation* op, std::ostringstream& oss)
{
    const auto& outputs = op->GetOOperands();
    if (outputs.empty() || outputs[0] == nullptr) {
        return;
    }
    const auto& tensor = outputs[0];
    AppendEntries("||prod:", BuildProducerEntries(tensor), oss);
    AppendEntries("||cons:", BuildConsumerEntries(tensor), oss);
}

std::string SigOf(const Operation* op)
{
    if (op == nullptr)
        return std::string();
    std::ostringstream oss;
    oss << static_cast<int>(op->GetOpcode()) << '|' << op->GetOOperands().size();
    AppendOutputTensorSig(op, oss);
    if (IsAllocOpCode(op->GetOpcode())) {
        AppendAllocTensorRelationSig(op, oss);
    } else {
        AppendNonAllocNeighborSig(op, oss);
    }
    return oss.str();
}

// Append direct data-dependency successors (ConsumerOps) to `out`, skipping
// visited, alloc, and any successor outside the task boundary (taskOps).
void CollectSuccessors(Operation* op, std::unordered_set<int>& visited, const std::unordered_set<Operation*>& taskOps,
                       std::vector<Operation*>& out)
{
    if (op == nullptr)
        return;
    for (Operation* succ : op->ConsumerOps()) {
        if (succ == nullptr)
            continue;
        if (taskOps.find(succ) == taskOps.end())
            continue; // stop at task boundary
        int magic = succ->GetOpMagic();
        if (visited.count(magic) > 0)
            continue;
        visited.insert(magic);
        out.push_back(succ);
    }
}

// 对 task 内 op 做拓扑排序后反转,得到反向拓扑序(出口在前、入口在后)。
// ProducerOps/ConsumerOps 均限制在 taskOps 边界内。
// 保证计算某 op 的下游子图 hash 时,它的全部后继已经算完。
std::vector<Operation*> ReverseTopoSort(const std::vector<Operation*>& taskOps,
                                        const std::unordered_set<Operation*>& taskSet)
{
    std::unordered_map<Operation*, int> inDegree;
    inDegree.reserve(taskOps.size());
    for (auto* op : taskOps) {
        if (op == nullptr)
            continue;
        int deg = 0;
        for (Operation* pred : op->ProducerOps()) {
            if (pred != nullptr && taskSet.count(pred) > 0)
                deg++;
        }
        inDegree[op] = deg;
    }
    std::deque<Operation*> q;
    for (auto* op : taskOps) {
        if (op != nullptr && inDegree[op] == 0)
            q.push_back(op);
    }
    std::vector<Operation*> sorted;
    sorted.reserve(inDegree.size());
    while (!q.empty()) {
        Operation* cur = q.front();
        q.pop_front();
        sorted.push_back(cur);
        for (Operation* succ : cur->ConsumerOps()) {
            if (succ == nullptr || taskSet.count(succ) == 0)
                continue;
            auto it = inDegree.find(succ);
            if (it != inDegree.end() && --(it->second) == 0)
                q.push_back(succ);
        }
    }
    std::reverse(sorted.begin(), sorted.end());
    return sorted;
}

// Merkle 风格的下游子图 hash:反向拓扑序自底向上折叠,每个 op 只存一个 size_t。
// h = hash(SigOf(op)) 起手,把 task 内后继的子图 hash 按【hash 值】排序后逐个折入。
// 按 hash 值(而非 opmagic)排序,保证核无关:两个同构下游子图必得相同 hash,
// 不受 ConsumerOps 枚举顺序、也不受 AIV0/AIV1 跨核创建序影响。
void ComputeSubgraphHashes(const std::vector<Operation*>& reverseTopoOps, const std::unordered_set<Operation*>& taskSet,
                           std::unordered_map<Operation*, size_t>& cache)
{
    cache.reserve(reverseTopoOps.size());
    for (Operation* op : reverseTopoOps) {
        if (op == nullptr)
            continue;
        std::vector<size_t> childHashes;
        for (Operation* succ : op->ConsumerOps()) {
            if (succ == nullptr || taskSet.count(succ) == 0)
                continue;
            auto it = cache.find(succ);
            if (it != cache.end()) // 反向拓扑序保证后继已算完
                childHashes.push_back(it->second);
        }
        std::sort(childHashes.begin(), childHashes.end());
        size_t h = std::hash<std::string>{}(SigOf(op));
        for (size_t ch : childHashes)
            HashCombine(h, ch);
        cache[op] = h;
    }
}

// 多候选消歧:单候选直接返回;多候选时按 opmagic 升序取第一个下游子图 hash 与 a 相等的。
// 一对一场景下双射唯一,第一个匹配即唯一解,无需回溯。无匹配返回 nullptr(交由上层等值校验兜底)。
Operation* PickCandidateBySubgraphHash(Operation* a, const std::vector<Operation*>& cands,
                                       const std::unordered_map<Operation*, size_t>& subHashA,
                                       const std::unordered_map<Operation*, size_t>& subHashB)
{
    if (cands.size() == 1)
        return cands[0];
    auto itA = subHashA.find(a);
    if (itA == subHashA.end())
        return nullptr;
    const size_t aHash = itA->second;
    for (Operation* cand : cands) {
        auto itB = subHashB.find(cand);
        if (itB != subHashB.end() && itB->second == aHash)
            return cand;
    }
    return nullptr;
}

bool CheckRootSignatures(const std::vector<Operation*>& rootsA, const std::vector<Operation*>& rootsB)
{
    std::vector<std::string> sigA;
    std::vector<std::string> sigB;
    for (auto* op : rootsA)
        sigA.push_back(SigOf(op));
    for (auto* op : rootsB)
        sigB.push_back(SigOf(op));
    std::sort(sigA.begin(), sigA.end());
    std::sort(sigB.begin(), sigB.end());
    if (sigA.size() != sigB.size())
        return false;
    for (size_t i = 0; i < sigA.size(); i++) {
        if (sigA[i] != sigB[i])
            return false;
    }
    return true;
}

using DepthSigGroups = std::map<int, std::unordered_map<std::string, std::vector<Operation*>>>;

DepthSigGroups BuildDepthSigGroups(const std::vector<Operation*>& roots, const std::unordered_set<Operation*>& taskOps)
{
    std::unordered_set<int> visited;
    for (auto* root : roots)
        visited.insert(root->GetOpMagic());

    DepthSigGroups depthSigGroups;
    std::vector<Operation*> cur = roots;
    int depth = 0;
    while (!cur.empty()) {
        std::vector<Operation*> next;
        for (Operation* op : cur) {
            depthSigGroups[depth][SigOf(op)].push_back(op);
            CollectSuccessors(op, visited, taskOps, next);
        }
        cur = std::move(next);
        ++depth;
    }
    return depthSigGroups;
}

void MatchIsoDepth(int depth, const std::vector<Operation*>& curA, const DepthSigGroups& depthSigGroupsB,
                   const std::unordered_set<Operation*>& taskOpsA, std::unordered_set<int>& visitedA,
                   std::unordered_set<int>& pairedB, std::vector<Operation*>& nextA, IsoMatchResult& res,
                   const std::unordered_map<Operation*, size_t>& subHashA,
                   const std::unordered_map<Operation*, size_t>& subHashB)
{
    auto byMagic = [](Operation* a, Operation* b) { return a->GetOpMagic() < b->GetOpMagic(); };
    auto layerIt = depthSigGroupsB.find(depth);
    if (layerIt == depthSigGroupsB.end())
        return;
    auto& groupB = layerIt->second;
    for (Operation* a : curA) {
        auto it = groupB.find(SigOf(a));
        if (it == groupB.end())
            continue;
        std::vector<Operation*> cands;
        for (Operation* cand : it->second) {
            if (pairedB.count(cand->GetOpMagic()) == 0)
                cands.push_back(cand);
        }
        if (cands.empty())
            continue;
        std::sort(cands.begin(), cands.end(), byMagic);

        // 多候选时靠下游子图 hash 消歧,而非 opmagic 序(magic 跨核不保证同序)。
        Operation* chosen = PickCandidateBySubgraphHash(a, cands, subHashA, subHashB);
        if (cands.size() > 1) {
            res.truncatedCount++; // 降级为纯诊断计数,不再阻止下探
            APASS_LOG_INFO_F(
                Elements::Operation, "IsoMatch multi-cand A depth=%d opmagic=%d opcode=%d cands=%zu matched=%d", depth,
                a->GetOpMagic(), static_cast<int>(a->GetOpcode()), cands.size(), static_cast<int>(chosen != nullptr));
        }
        if (chosen == nullptr) {
            // 无子图 hash 匹配候选:不配对且不下探。a 的子链缺失会让 pairs.size()
            // 少于 nonAllocA,由调用方等值校验安全回退阻断。
            continue;
        }

        pairedB.insert(chosen->GetOpMagic());
        if (a->GetOpcode() == Opcode::OP_UB_ALLOC) {
            res.allocPairs.push_back({a, chosen, depth});
        } else if (!IsAllocOpCode(a->GetOpcode())) {
            res.pairs.push_back({a, chosen, depth});
            if (depth > res.maxMatchedDepth)
                res.maxMatchedDepth = depth;
        }
        CollectSuccessors(a, visitedA, taskOpsA, nextA); // 多候选也无条件下探
    }
}

} // namespace

std::vector<Operation*> FindTaskEntryOps(const std::vector<Operation*>& taskOps,
                                         const std::unordered_set<Operation*>& taskSet)
{
    std::vector<Operation*> entries;
    for (Operation* op : taskOps) {
        if (op == nullptr)
            continue;
        bool hasInTaskProducer = false;
        for (Operation* prod : op->ProducerOps()) {
            if (prod != nullptr && taskSet.count(prod) > 0) {
                hasInTaskProducer = true;
                break;
            }
        }
        if (!hasInTaskProducer) {
            entries.push_back(op);
        }
    }
    return entries;
}

IsoMatchResult IsoMatchChains(const std::vector<Operation*>& rootsA, const std::vector<Operation*>& rootsB,
                              const std::unordered_set<Operation*>& taskOpsA,
                              const std::unordered_set<Operation*>& taskOpsB)
{
    IsoMatchResult res;
    if (rootsA.empty() || rootsB.empty())
        return res;

    auto byMagic = [](Operation* a, Operation* b) { return a->GetOpMagic() < b->GetOpMagic(); };

    if (!CheckRootSignatures(rootsA, rootsB))
        return res;

    res.rootIsomorphic = true;

    // 预计算两侧 task 内每个 op 的下游子图 hash,供多候选消歧使用(核无关)。
    std::unordered_map<Operation*, size_t> subHashA;
    std::unordered_map<Operation*, size_t> subHashB;
    {
        std::vector<Operation*> taskListA(taskOpsA.begin(), taskOpsA.end());
        std::vector<Operation*> taskListB(taskOpsB.begin(), taskOpsB.end());
        auto rtopoA = ReverseTopoSort(taskListA, taskOpsA);
        auto rtopoB = ReverseTopoSort(taskListB, taskOpsB);
        ComputeSubgraphHashes(rtopoA, taskOpsA, subHashA);
        ComputeSubgraphHashes(rtopoB, taskOpsB, subHashB);
    }

    // BFS 前先标记全部入口点，避免被后继重复加入。
    std::unordered_set<int> visitedA;
    for (auto* a : rootsA)
        visitedA.insert(a->GetOpMagic());

    auto depthSigGroupsB = BuildDepthSigGroups(rootsB, taskOpsB);

    // === 阶段 2: A 侧 BFS, depth=d 的 A 节点只在 B 侧 depth=d 的节点中找同构候选 ===
    std::unordered_set<int> pairedB; // 全局已配对 B 节点 opmagic
    std::vector<Operation*> curA = rootsA;
    int depth = 0;
    while (!curA.empty()) {
        std::sort(curA.begin(), curA.end(), byMagic);
        std::vector<Operation*> nextA;

        MatchIsoDepth(depth, curA, depthSigGroupsB, taskOpsA, visitedA, pairedB, nextA, res, subHashA, subHashB);
        curA = std::move(nextA);
        ++depth;
    }
    return res;
}

} // namespace npu::tile_fwk
