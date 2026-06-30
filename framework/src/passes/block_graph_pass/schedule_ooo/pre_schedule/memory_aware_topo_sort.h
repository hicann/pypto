/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_TILE_FWK_MEMORY_AWARE_TOPO_SORT_H
#define NPU_TILE_FWK_MEMORY_AWARE_TOPO_SORT_H

#include "passes/block_graph_pass/schedule_ooo/common/schedule_base.h"
#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace npu::tile_fwk {

enum class MemoryState {
    Critical,
    Tight,
    Normal,
    Abundant
};

enum class ConstraintType {
    HardConstraint,
    SoftConstraint,
    Optional
};

struct MemoryPoolContext {
    uint64_t usage;
    uint64_t limit;
    ConstraintType type;
    bool can_spill;
    uint64_t max_slot_count;
    uint64_t current_slot_count;

    MemoryPoolContext()
        : usage(0),
          limit(0),
          type(ConstraintType::Optional),
          can_spill(false),
          max_slot_count(UINT64_MAX),
          current_slot_count(0) {}
};

struct SchedulingContext {
    std::unordered_map<MemoryType, MemoryPoolContext> memory_pools;
    std::unordered_map<int, int> executed_consumers;
    int max_consumer_count;

    SchedulingContext()
        : max_consumer_count(0) {}
};

struct ScoringParams {
    // 默认参数 (命名规范: 开头小写, 每个单词首字母大写, 以_结尾)
    static constexpr double alpha_ = 0.7;                // 评分权重 alpha
    static constexpr double beta_ = 0.4;                 // 评分权重 beta
    static constexpr double criticalThreshold_ = 0.9;    // Critical 状态占用阈值
    static constexpr double tightThreshold_ = 0.7;       // Tight 状态占用阈值
    static constexpr double abundantThreshold_ = 0.3;    // Abundant 状态占用阈值
    static constexpr double criticalFactor_ = 1.5;       // Critical 状态因子
    static constexpr double tightFactor_ = 1.2;          // Tight 状态因子
    static constexpr double normalFactor_ = 1.0;         // Normal 状态因子
    static constexpr double abundantFactor_ = 0.5;       // Abundant 状态因子
    static constexpr double spillFactor_ = 1.3;          // Spill 状态因子

    double alpha;
    double beta;
    double critical_threshold;
    double tight_threshold;
    double abundant_threshold;
    double critical_factor;
    double tight_factor;
    double normal_factor;
    double abundant_factor;
    double spill_factor;

    ScoringParams()
        : alpha(alpha_),
          beta(beta_),
          critical_threshold(criticalThreshold_),
          tight_threshold(tightThreshold_),
          abundant_threshold(abundantThreshold_),
          critical_factor(criticalFactor_),
          tight_factor(tightFactor_),
          normal_factor(normalFactor_),
          abundant_factor(abundantFactor_),
          spill_factor(spillFactor_) {}
};

MemoryState GetMemoryState(double usage_ratio);

double CalcStateAdjustFactor(MemoryType mem_type, const SchedulingContext& context);

double CalcDynamicTypeWeight(MemoryType mem_type, const SchedulingContext& context);

double CalcReleaseContribution(int memId, Operation* node,
    const SchedulingContext& context, const ScoringParams& params);

double CalcAllocationPressure(LogicalTensorPtr output, const SchedulingContext& context, const ScoringParams& params);

double CalcNodeScore(Operation* node, const SchedulingContext& context, const ScoringParams& params, int node_id);

std::vector<Operation*> MemoryAwareTopologicalSort(
    const std::vector<Operation*>& operations,
    DependencyManager& depManager,
    SchedulingContext& context,
    const ScoringParams& params);

class MemoryAwareTopoSort : public ScheduleBase {
public:
    MemoryAwareTopoSort(std::vector<Operation*> opList, Function& function)
        : ScheduleBase(), function_(&function)
    {
        operations = opList;
    }

    ~MemoryAwareTopoSort() = default;

    Status SortOps();
    Status InitContext();
    void UpdateMemoryState(Operation* op);
    bool HasMemBtOutput(Operation* op) const;

private:
     Function* function_;         // Function context
     SchedulingContext context_;  // Scheduling context
     ScoringParams params_;       // Scoring parameters
     DependencyManager depManager_; // Dependency manager for topological sorting
};

} // namespace npu::tile_fwk

#endif // NPU_TILE_FWK_MEMORY_AWARE_TOPO_SORT_H
