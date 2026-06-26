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
 * \file schedule_ooo.h
 * \brief
 */

#ifndef PASS_SCHEDULE_OOO_H
#define PASS_SCHEDULE_OOO_H

#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/ooo_scheduler.h"
#include "passes/statistics/ooo_schedule_statistic.h"
#include "passes/statistics/memory_tracer.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/schedule_ooo/optimize_sort.h"
#include "passes/block_graph_pass/schedule_ooo/latency_estimator.h"
#include "passes/block_graph_pass/schedule_ooo/core_assign.h"
#include "passes/block_graph_pass/schedule_ooo/task_splitter.h"

namespace npu::tile_fwk {
struct ScheduleUnit {
    int earliestStartTime;
    std::vector<Operation*> mergedOps;
};

const std::unordered_map<TargetCoreType, CoreLocationType> targetCoreTypeMap{
    {TargetCoreType::AIC, CoreLocationType::AIC},
    {TargetCoreType::AIV0, CoreLocationType::AIV0},
    {TargetCoreType::AIV1, CoreLocationType::AIV1}};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_TWO_AIV = {
    CoreLocationType::AIC, CoreLocationType::AIV0, CoreLocationType::AIV1};

class OoOSchedule : public Pass {
public:
    OoOSchedule() : Pass("OoOSchedule") {}
    ~OoOSchedule() override {}

private:
    Status RunOnFunction(Function& function) override;
    bool IsAicpuProgram(std::vector<Operation*> opList);
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    void DoHealthCheckAfter(Function& function, const std::string& folderPath) override;
    void SortTaskList(std::vector<Operation*>& operations, std::vector<Operation*>& taskList);
    Status SortAndLatencyEstimate(std::vector<Operation*>& opList, std::vector<Operation*>& taskOpList, int& latency);
    void CollectStatistic(
        OoOScheduleStatistic& oooHealthCheck, Function& function, std::pair<uint64_t, Function*>& program);
    Status CollectLastUseInfo(Function& function);
    void SetLastUseAttributes();
    Status NonMixSchedule(
        std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
        int64_t& maxWorkeSpaceSize);
    Status MixSchedule(
        std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
        int64_t& maxWorkeSpaceSize);
    Status EstimateTaskLatencyAndSchedule(TaskSplitter& splitter, std::vector<Operation*>& opList,
        const std::string& schedMode = "");
    Status BuildMixedScheduleOps(
        TaskSplitter& splitter, std::vector<Operation*>& opList,
        std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    Status BuildMemIdToAllocIdx(
        const std::vector<Operation*>& opList, std::unordered_map<uint64_t, size_t>& memIdToAllocIdx);
    bool MoveAllocBeforeOp(
        std::vector<Operation*>& opList, size_t allocIdx, int targetIdx,
        std::unordered_map<uint64_t, size_t>& memIdToAllocIdx, uint64_t memId);
    Status ModifyAllocOrder(std::vector<Operation*>& opList);
    Status UpdateOpCoreMap(const TaskNode& taskNode, std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    std::vector<ScheduleUnit> BuildScheduleUnits(
        const std::vector<TaskNode>& taskNodeList, const std::vector<std::pair<int, int>>& cyclePairs,
        std::vector<Operation*>& opList);
    std::vector<Function*> oriFunctions;
    std::map<uint64_t, OoOScheduleStatistic> statisticMap_;
    // Per-program tracers, populated on SUCCESS only; failure path flushes inline.
    std::map<uint64_t, MemoryTracer> tracerMap_;
    void CollectMemoryTrace(MemoryTracer& tracer, Function& function, std::pair<uint64_t, Function*>& program);
    void FlushMemoryTraceOnFailure(MemoryTracer& tracer, Function& function, std::pair<uint64_t, Function*>& program);
    std::unordered_map<LogicalTensorPtr, Operation*> lastUseMap_;
    OoOScheduleChecker checker;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_OOO_H
