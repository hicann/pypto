/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_aware_topo_sort.h"
#include "passes/block_graph_pass/schedule_ooo/common/schedule_base.h"
#include "passes/pass_log/pass_log.h"
#include "interface/function/function.h"
#include <cmath>
#include <queue>
#include <algorithm>

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "MemoryAwareTopoSort"

namespace npu::tile_fwk {

// 内存状态阈值与调度权重常量
constexpr double kLargeConstraintBaseWeight = 2.0;
constexpr double kMediumConstraintBaseWeight = 1.0;
constexpr double kSmallConstraintBaseWeight = 0.5;         
constexpr double kCriticalUsageRatio = 0.9;
constexpr double kTightUsageRatio = 0.7;
constexpr double kNormalUsageRatio = 0.3;
constexpr double kCriticalAdjustFactor = 1.5;       // Critical 状态调整因子
constexpr double kTightAdjustFactor = 1.2;          // Tight 状态调整因子
constexpr double kNormalAdjustFactor = 1.0;        // Normal/默认状态调整因子
constexpr double kAbundantAdjustFactor = 0.5;       // Abundant 状态调整因子

static ConstraintType GetConstraintType(MemoryType mem_type)
{
    if (mem_type == MemoryType::MEM_BT) {
        return ConstraintType::HardConstraint;
    }

    if (mem_type == MemoryType::MEM_L0A ||
        mem_type == MemoryType::MEM_L0B ||
        mem_type == MemoryType::MEM_L0C ||
        mem_type == MemoryType::MEM_L1 ||
        mem_type == MemoryType::MEM_UB ||
        mem_type == MemoryType::MEM_L0AMX ||
        mem_type == MemoryType::MEM_L0BMX ||
        mem_type == MemoryType::MEM_FIX ||
        mem_type == MemoryType::MEM_FIX_QUANT_PRE) {
        return ConstraintType::SoftConstraint;
    }

    return ConstraintType::Optional;
}

static bool SupportsSpill(MemoryType mem_type)
{
    return (mem_type == MemoryType::MEM_UB || mem_type == MemoryType::MEM_L1);
}

static double GetBaseWeight(ConstraintType constraint_type)
{
    switch (constraint_type) {
        case ConstraintType::HardConstraint:
            return kLargeConstraintBaseWeight;
        case ConstraintType::SoftConstraint:
            return kMediumConstraintBaseWeight;
        case ConstraintType::Optional:
            return kSmallConstraintBaseWeight;
        default:
            return kMediumConstraintBaseWeight;
    }
}

MemoryState GetMemoryState(double usage_ratio)
{
    if (usage_ratio >= kCriticalUsageRatio) {
        return MemoryState::Critical;
    } else if (usage_ratio >= kTightUsageRatio) {
        return MemoryState::Tight;
    } else if (usage_ratio >= kNormalUsageRatio) {
        return MemoryState::Normal;
    } else {
        return MemoryState::Abundant;
    }
}

double CalcStateAdjustFactor(MemoryType mem_type, const SchedulingContext& context)
{
    auto it = context.memory_pools.find(mem_type);
    if (it == context.memory_pools.end()) {
        return kNormalAdjustFactor;
    }

    const MemoryPoolContext& pool_ctx = it->second;

    if (pool_ctx.limit == 0) {
        return kNormalAdjustFactor;
    }

    double usage_ratio = static_cast<double>(pool_ctx.usage) / static_cast<double>(pool_ctx.limit);
    MemoryState state = GetMemoryState(usage_ratio);

    switch (state) {
        case MemoryState::Critical:
            return kCriticalAdjustFactor;
        case MemoryState::Tight:
            return kTightAdjustFactor;
        case MemoryState::Normal:
            return kNormalAdjustFactor;
        case MemoryState::Abundant:
            return kAbundantAdjustFactor;
        default:
            return kNormalAdjustFactor;
    }
}

double CalcDynamicTypeWeight(MemoryType mem_type, const SchedulingContext& context)
{
    ConstraintType constraint_type = GetConstraintType(mem_type);
    double base_weight = GetBaseWeight(constraint_type);
    double state_factor = CalcStateAdjustFactor(mem_type, context);
    bool can_spill = SupportsSpill(mem_type);
    double spill_factor = can_spill ? 1.0 : 1.3;

    return base_weight * state_factor * spill_factor;
}

double CalcReleaseContribution(int memId, Operation* node,
    const SchedulingContext& context, const ScoringParams& params)
{
    if (node == nullptr) {
        return 0.0;
    }

    LogicalTensorPtr input_tensor = nullptr;
    for (auto tensor : node->GetIOperands()) {
        if (tensor->memoryrange.memId == memId) {
            input_tensor = tensor;
            break;
        }
    }

    if (input_tensor == nullptr) {
        return 0.0;
    }

    uint64_t size = input_tensor->tensor->GetRawDataSize();
    if (size == 0) {
        return 0.0;
    }

    MemoryType mem_type = input_tensor->GetMemoryTypeOriginal();
    double type_weight = CalcDynamicTypeWeight(mem_type, context);

    int total_consumers = static_cast<int>(input_tensor->GetConsumers().size());
    if (total_consumers == 0) {
        return 0.0;
    }

    auto exec_it = context.executed_consumers.find(memId);
    int executed_consumers = (exec_it != context.executed_consumers.end()) ? exec_it->second : 0;

    int remaining_consumers = total_consumers - (executed_consumers + 1);

    double marginal_factor = 0.0;
    if (remaining_consumers == 0) {
        marginal_factor = 1.0;
    } else {
        double base_factor = 1.0 / (remaining_consumers + 1);
        double progress_ratio = static_cast<double>(executed_consumers) / static_cast<double>(total_consumers);
        double progress_bonus = 1.0 + params.alpha * progress_ratio * progress_ratio;
        marginal_factor = base_factor * progress_bonus;
    }

    double release_contribution = static_cast<double>(size) * type_weight * marginal_factor;

    return release_contribution;
}

double CalcAllocationPressure(LogicalTensorPtr output, const SchedulingContext& context, const ScoringParams& params)
{
    if (output == nullptr) {
        return 0.0;
    }

    uint64_t size = output->tensor->GetRawDataSize();
    if (size == 0) {
        return 0.0;
    }

    MemoryType mem_type = output->GetMemoryTypeOriginal();
    double type_weight = CalcDynamicTypeWeight(mem_type, context);
    int consumer_count = static_cast<int>(output->GetConsumers().size());

    double consumer_pressure_factor = 1.0;
    if (consumer_count > 0 && context.max_consumer_count > 0) {
        consumer_pressure_factor = 1.0 + params.beta * (consumer_count - 1) / context.max_consumer_count;
    }

    double allocation_pressure = static_cast<double>(size) * type_weight * consumer_pressure_factor;

    return allocation_pressure;
}

double CalcNodeScore(Operation* node, const SchedulingContext& context, const ScoringParams& params, int node_id)
{
    if (node == nullptr) {
        return 0.0;
    }

    double release_score = 0.0;
    for (auto input_tensor : node->GetIOperands()) {
        int memId = input_tensor->memoryrange.memId;
        double contribution = CalcReleaseContribution(memId, node, context, params);
        MemoryType mem_type = input_tensor->GetMemoryTypeOriginal();
        double state_factor = CalcStateAdjustFactor(mem_type, context);
        release_score += contribution * state_factor;
    }

    double allocation_pressure = 0.0;
    for (auto output_tensor : node->GetOOperands()) {
        double pressure = CalcAllocationPressure(output_tensor, context, params);
        MemoryType mem_type = output_tensor->GetMemoryTypeOriginal();
        double state_factor = CalcStateAdjustFactor(mem_type, context);
        allocation_pressure += pressure * state_factor;
    }

    constexpr double epsilon = 1e-9;
    double score = release_score - allocation_pressure + epsilon * static_cast<double>(node_id);

    return score;
}

static int CalcTopoPriority(Operation* node, const std::unordered_map<Operation*, int>& topo_depth)
{
    if (node == nullptr) {
        return 0;
    }
    auto it = topo_depth.find(node);
    if (it != topo_depth.end()) {
        return it->second;
    }
    return 0;
}

static bool HasMemBtOutputOp(Operation* op)
{
    if (op == nullptr) {
        return false;
    }
    for (auto tensor : op->GetOOperands()) {
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_BT) {
            return true;
        }
    }
    return false;
}

static bool CanReleaseMemBt(Operation* op)
{
    if (op == nullptr) {
        return false;
    }
    for (auto tensor : op->GetIOperands()) {
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_BT) {
            return true;
        }
    }
    return false;
}

static void UpdateMemoryPoolState(SchedulingContext& context, Operation* op)
{
    for (auto input_tensor : op->GetIOperands()) {
        int memId = input_tensor->memoryrange.memId;
        MemoryType mem_type = input_tensor->GetMemoryTypeOriginal();
        if (mem_type >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }

        context.executed_consumers[memId]++;

        int total_consumers = static_cast<int>(input_tensor->GetConsumers().size());
        if (context.executed_consumers[memId] >= total_consumers) {
            auto pool_it = context.memory_pools.find(mem_type);
            if (pool_it != context.memory_pools.end()) {
                uint64_t size = input_tensor->tensor->GetRawDataSize();
                if (pool_it->second.usage >= size) {
                    pool_it->second.usage -= size;
                }
                if (pool_it->second.current_slot_count > 0) {
                    pool_it->second.current_slot_count--;
                }
            }
        }
    }

    for (auto output_tensor : op->GetOOperands()) {
        MemoryType mem_type = output_tensor->GetMemoryTypeOriginal();
        if (mem_type >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }

        uint64_t size = output_tensor->tensor->GetRawDataSize();
        if (size == 0) {
            continue;
        }

        auto pool_it = context.memory_pools.find(mem_type);
        if (pool_it != context.memory_pools.end()) {
            pool_it->second.usage += size;
            pool_it->second.current_slot_count++;
        }
    }
}

static void InitSchedulingState(
    const std::vector<Operation*>& operations,
    DependencyManager& depManager,
    std::unordered_map<Operation*, int>& in_degree,
    std::unordered_map<Operation*, int>& topo_depth,
    std::unordered_map<Operation*, int>& op_index,
    std::vector<Operation*>& ready_queue)
{
    for (auto op : operations) {
        in_degree[op] = 0;
    }

    for (auto op : operations) {
        auto& successors = depManager.GetSuccessors(op);
        for (auto succ : successors) {
            if (in_degree.find(succ) != in_degree.end()) {
                in_degree[succ]++;
            }
        }
    }
    
    // 计算拓扑深度（依赖深度）
    std::queue<Operation*> depth_queue;
    for (auto op : operations) {
        if (in_degree[op] == 0) {
            topo_depth[op] = 0;
            depth_queue.push(op);
        }
    }
    
    std::unordered_map<Operation*, int> temp_in_degree = in_degree;
    while (!depth_queue.empty()) {
        Operation* current = depth_queue.front();
        depth_queue.pop();
        
        auto& successors = depManager.GetSuccessors(current);
        for (auto succ : successors) {
            if (temp_in_degree.find(succ) != temp_in_degree.end()) {
                temp_in_degree[succ]--;
                if (temp_in_degree[succ] == 0) {
                    topo_depth[succ] = topo_depth[current] + 1;
                    depth_queue.push(succ);
                }
            }
        }
    }

    for (size_t i = 0; i < operations.size(); i++) {
        op_index[operations[i]] = static_cast<int>(i);
    }
    
    for (auto op : operations) {
        if (in_degree[op] == 0) {
            ready_queue.push_back(op);
        }
    }
    
    // 确定性：按node_id排序ready_queue
    std::sort(ready_queue.begin(), ready_queue.end(), [&op_index](Operation* a, Operation* b) {
        int id_a = op_index.find(a) != op_index.end() ? op_index[a] : INT_MAX;
        int id_b = op_index.find(b) != op_index.end() ? op_index[b] : INT_MAX;
        return id_a < id_b;
    });
}

static bool SelectAllocFromReadyQueue(
    const std::vector<Operation*>& ready_queue,
    const std::unordered_map<Operation*, int>& op_index,
    bool mem_bt_full,
    int& best_idx,
    int& best_node_id)
{
    best_idx = -1;
    best_node_id = INT_MAX;

    // ALLOC nodes must always be scheduled before any non-ALLOC node.
    // They have in_degree=0 and produce tensor memory; consumers depend on them.
    // Selecting ALLOCs first guarantees alloc→copyin→compute ordering without
    // relying on error-prone post-processing like EnsureAllocInterleaving.
    for (size_t i = 0; i < ready_queue.size(); i++) {
        Operation* candidate = ready_queue[i];
        if (candidate->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            if (HasMemBtOutputOp(candidate) && mem_bt_full && !CanReleaseMemBt(candidate)) {
                continue;
            }
            auto idx_it = op_index.find(candidate);
            int node_id = (idx_it != op_index.end()) ? idx_it->second : INT_MAX;
            if (node_id < best_node_id) {
                best_idx = static_cast<int>(i);
                best_node_id = node_id;
            }
        }
    }

    return (best_idx != -1);
}

static void SelectScoredFromReadyQueue(
    const std::vector<Operation*>& ready_queue,
    const std::unordered_map<Operation*, int>& op_index,
    const std::unordered_map<Operation*, int>& topo_depth,
    SchedulingContext& context,
    const ScoringParams& params,
    bool mem_bt_full,
    double& best_score,
    int& best_idx,
    int& best_node_id)
{
    best_score = -1e300;

    auto evaluate = [&](Operation* candidate, size_t i) {
        int node_id = 0;
        auto idx_it = op_index.find(candidate);
        if (idx_it != op_index.end()) { node_id = idx_it->second; }
        int topo_priority = CalcTopoPriority(candidate, topo_depth);
        double base_score = CalcNodeScore(candidate, context, params, node_id);
        double score = static_cast<double>(topo_priority) * 1000.0 + base_score;
        bool score_equal = std::abs(score - best_score) < 1e-10;
        if (score > best_score || (score_equal && node_id < best_node_id)) {
            best_score = score;
            best_idx = static_cast<int>(i);
            best_node_id = node_id;
        }
    };

    for (size_t i = 0; i < ready_queue.size(); i++) {
        Operation* candidate = ready_queue[i];

        if (HasMemBtOutputOp(candidate) && mem_bt_full) {
            if (CanReleaseMemBt(candidate)) { evaluate(candidate, i); }
            continue;
        }

        evaluate(candidate, i);
    }
}

static bool HandleNoSelectableCandidate(
    const std::vector<Operation*>& ready_queue,
    bool mem_bt_full,
    const SchedulingContext& context,
    int executed_count,
    size_t total_ops)
{
    bool all_blocked = true;
    bool has_releaser = false;
    
    for (auto op : ready_queue) {
        if (HasMemBtOutputOp(op) && mem_bt_full) {
            if (CanReleaseMemBt(op)) {
                has_releaser = true;
            }
            continue;
        }
        all_blocked = false;
        break;
    }

    if (all_blocked && !has_releaser && executed_count < static_cast<int>(total_ops)) {
        auto pool_it = context.memory_pools.find(MemoryType::MEM_BT);
        APASS_LOG_ERROR_F(Elements::Operation,
            "MemoryAwareTopologicalSort: MEM_BT slot exhausted. "
            "Executed %d / %zu ops, ready_queue=%zu, mem_bt_full=%d, "
            "mem_bt_slots=%lu/%lu.",
            executed_count, total_ops, ready_queue.size(),
            mem_bt_full ? 1 : 0,
            pool_it != context.memory_pools.end() ? pool_it->second.current_slot_count : 0,
            pool_it != context.memory_pools.end() ? pool_it->second.max_slot_count : 0);
        return true;
    }

    if (executed_count < static_cast<int>(total_ops)) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "MemoryAwareTopologicalSort: no selectable candidate. "
            "Executed %d / %zu ops, ready_queue=%zu, mem_bt_full=%d.",
            executed_count, total_ops, ready_queue.size(),
            mem_bt_full ? 1 : 0);
        return true;
    }
    return false;
}

static void CommitCandidate(
    int best_idx,
    std::vector<Operation*>& ready_queue,
    std::vector<Operation*>& result,
    int& executed_count,
    SchedulingContext& context,
    DependencyManager& depManager,
    std::unordered_map<Operation*, int>& in_degree,
    const std::unordered_map<Operation*, int>& op_index)
{
    Operation* selected = ready_queue[best_idx];
    ready_queue.erase(ready_queue.begin() + best_idx);

    result.push_back(selected);
    executed_count++;

    UpdateMemoryPoolState(context, selected);

    auto& successors = depManager.GetSuccessors(selected);
    std::vector<Operation*> new_ready_nodes;
    for (auto succ : successors) {
        if (in_degree.find(succ) != in_degree.end()) {
            in_degree[succ]--;
            if (in_degree[succ] == 0) {
                new_ready_nodes.push_back(succ);
            }
        }
    }
    
    // 确定性：按node_id排序新加入的节点，然后插入到ready_queue末尾
    std::sort(new_ready_nodes.begin(), new_ready_nodes.end(), [&op_index](Operation* a, Operation* b) {
        auto it_a = op_index.find(a);
        int id_a = (it_a != op_index.end()) ? it_a->second : INT_MAX;
        auto it_b = op_index.find(b);
        int id_b = (it_b != op_index.end()) ? it_b->second : INT_MAX;
        return id_a < id_b;
    });
    
    for (auto node : new_ready_nodes) {
        ready_queue.push_back(node);
    }
}

std::vector<Operation*> MemoryAwareTopologicalSort(
    const std::vector<Operation*>& operations,
    DependencyManager& depManager,
    SchedulingContext& context,
    const ScoringParams& params)
{
    std::vector<Operation*> result;

    if (operations.empty()) {
        return result;
    }

    if (operations.size() == 1) {
        result.push_back(operations[0]);
        return result;
    }

    std::unordered_map<Operation*, int> in_degree, topo_depth, op_index;
    std::vector<Operation*> ready_queue;
    InitSchedulingState(operations, depManager, in_degree, topo_depth, op_index, ready_queue);

    int executed_count = 0;

    APASS_LOG_EVENT_F(Elements::Operation,
        "MemoryAwareTopologicalSort: ENTER, ops=%zu, ready=%zu",
        operations.size(), ready_queue.size());

    while (!ready_queue.empty()) {
        double best_score;
        int best_idx;
        int best_node_id;

        auto pool_it = context.memory_pools.find(MemoryType::MEM_BT);
        bool mem_bt_full = (pool_it != context.memory_pools.end()) &&
            (pool_it->second.current_slot_count >= pool_it->second.max_slot_count);
        
        APASS_LOG_DEBUG_F(Elements::Operation,
            "Iteration %d: ready_queue=%zu, mem_bt_full=%d, mem_bt_slots=%lu/%lu",
            executed_count, ready_queue.size(), mem_bt_full ? 1 : 0,
            pool_it != context.memory_pools.end() ? pool_it->second.current_slot_count : 0,
            pool_it != context.memory_pools.end() ? pool_it->second.max_slot_count : 0);
        
        bool found = SelectAllocFromReadyQueue(ready_queue, op_index, mem_bt_full, best_idx, best_node_id);
        if (!found) {
            SelectScoredFromReadyQueue(ready_queue, op_index, topo_depth, context, params,
                mem_bt_full, best_score, best_idx, best_node_id);
        }

        if (best_idx == -1) {
            if (HandleNoSelectableCandidate(ready_queue, mem_bt_full, context, executed_count, operations.size())) {
                return std::vector<Operation*>();
            }
            break;
        }

        CommitCandidate(best_idx, ready_queue, result, executed_count, context, depManager, in_degree, op_index);
    }

    if (executed_count != static_cast<int>(operations.size())) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "MemoryAwareTopologicalSort: incomplete sort. Executed %d / %zu.",
            executed_count, operations.size());
        return std::vector<Operation*>();
    }

    return result;
}

Status MemoryAwareTopoSort::InitContext()
{
    context_.memory_pools.clear();
    context_.executed_consumers.clear();
    context_.max_consumer_count = 0;

    // Initialize DependencyManager from operations
    Status init_res = depManager_.InitDependencies(operations, false);
    if (init_res != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "MemoryAwareTopoSort::InitContext: DependencyManager.InitDependencies failed.");
        return FAILED;
    }

    for (auto& mem_pair : localMemSize) {
        MemoryType mem_type = mem_pair.first;
        int64_t limit = mem_pair.second;

        MemoryPoolContext pool_ctx;
        pool_ctx.limit = static_cast<uint64_t>(limit);
        pool_ctx.usage = 0;
        pool_ctx.current_slot_count = 0;
        pool_ctx.type = GetConstraintType(mem_type);
        pool_ctx.can_spill = SupportsSpill(mem_type);

        pool_ctx.max_slot_count = UINT64_MAX;

        context_.memory_pools[mem_type] = pool_ctx;
    }

    for (auto op : operations) {
        for (auto tensor : op->GetIOperands()) {
            int memId = tensor->memoryrange.memId;
            if (context_.executed_consumers.find(memId) == context_.executed_consumers.end()) {
                context_.executed_consumers[memId] = 0;
            }
        }
        for (auto tensor : op->GetOOperands()) {
            int memId = tensor->memoryrange.memId;
            if (context_.executed_consumers.find(memId) == context_.executed_consumers.end()) {
                context_.executed_consumers[memId] = 0;
            }

            int consumer_count = static_cast<int>(tensor->GetConsumers().size());
            if (consumer_count > context_.max_consumer_count) {
                context_.max_consumer_count = consumer_count;
            }
        }
    }

    return SUCCESS;
}

void MemoryAwareTopoSort::UpdateMemoryState(Operation* op)
{
    UpdateMemoryPoolState(context_, op);
}

bool MemoryAwareTopoSort::HasMemBtOutput(Operation* op) const
{
    return npu::tile_fwk::HasMemBtOutputOp(op);
}

Status MemoryAwareTopoSort::SortOps()
{
    if (operations.empty()) {
        return SUCCESS;
    }

    if (InitContext() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "MemoryAwareTopoSort::InitContext failed.");
        return FAILED;
    }

    std::vector<Operation*> sorted = MemoryAwareTopologicalSort(
        operations, depManager_, context_, params_);
    if (sorted.empty() && !operations.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "MemoryAwareTopoSort::SortOps failed.");
        return FAILED;
    }

    operations = sorted;

    return SUCCESS;
}

} // namespace npu::tile_fwk
