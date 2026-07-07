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
 * \file ooo_scheduler.h
 * \brief
 */

#ifndef PASS_SCHEDULER_H
#define PASS_SCHEDULER_H

#include <climits>
#include "interface/tensor/irbuilder.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_check/schedule_ooo_checker.h"
#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include "passes/statistics/schedule_observer.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/spill_engine.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/dualdst_engine.h"

namespace npu::tile_fwk {

class OoOScheduler {
public:
    ScheduleState state_;

    template <typename Scheduler>
    friend Status RunSchedulerMainLoop(Scheduler& self);
public:
    Status Schedule(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function& function)
        : function_(function),
          spillEngine_(state_, function_),
          dualDstEngine_(state_, function_) {}

    void AddObserver(ScheduleObserver* observer) { state_.observers_.push_back(observer); }
    std::vector<Operation*> GetNewOperations();

    // === DualDst 开关 ===
    // OoOSchedule 在调用 Schedule() 之前设置(默认 false 保持原行为)。
    void SetEnableDualDst(bool v) { dualDstEngine_.SetEnableDualDst(v); }

private:

    Function& function_;
    SpillEngine spillEngine_;
    DualDstEngine dualDstEngine_;

    std::unordered_map<CoreLocationType, std::map<PipeType, OpQueue>> issueQueues;

    uint64_t spillIssueCnt{0};
    std::vector<ScheduleObserver*>& observers_ = state_.observers_;

    IRBuilder irBuilder_;
    std::unordered_map<int, DDRBufferKind> ddrKindMap_;

    void NotifyOpLaunch(Operation* op, int cycleEnd);
    void NotifyOpRetire(Operation* op, const std::vector<int>& freedMemIds);
    void NotifyAllocExec(Operation* op, int memId);
    void NotifyBufferRearrange(Operation* triggerOp, MemoryType memType,
        std::vector<BufferRearrangeEvent::Change> changes);
    void NotifyAllocFail(Operation* triggerOp, MemoryType memType, uint64_t requestSize);
    void NotifyScheduleEnd(bool success);
    void NotifyInitDDRBuffers();
    void NotifyMainLoopBegin();
    void NotifyMainLoopEnd();
    void EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind);

    static CoreLocation ToCoreLocation(CoreLocationType c);

    std::vector<DDRRef> BuildDDRRefs(Operation* op) const;

    // scheduler
    Status Init(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);

    Status InitOpEntry(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    Status InitOpCoreType(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    void InitOpViewOps(Operation* op);

    void InitCoreConfig(const std::vector<Operation *> &opList);
    void InitTensorCoreMap();
    void InitIssueQueuesAndBufferManager();

    void AllocWorkspaceGM(const std::vector<Operation *> &opList);
    Status SeqSchedule();
    Status ExecuteAllocIssue(Operation* op, size_t &pcIdx);
    Status RetireIssue(Operation* op);
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireCoreIssue(CoreLocationType targetCore, uint64_t& commitCnt, int& nextCycle);
    Status PreMainLoop();
    Status PostMainLoop();
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op, std::vector<int>& freedMemIds);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType,
        OpQueue &pipe);
    Status TryDualDstAllocOnce(Operation* op, uint64_t& commitCnt, bool& allocated);
    Status TryRegularAllocOnce(Operation* op, MemoryType memType, CoreLocationType coreLocation,
                               const std::vector<int>& reqMemIds,
                               uint64_t& commitCnt, bool& allocated);
    void HandleViewOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle);
    Status AllocTensorMemRange(Operation* op);
    Status AllocViewTensorMemRange(Operation &operation);
    Status CheckAndUpdateLifecycle();
    void UpdateIssueExecOrder();
    void PrintOpList(std::vector<Operation *> opList);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType);
    Status GenBufferSpill(Operation* allocOp, SpillContext& ctx);
    std::vector<int> SelectSpillBuffers(Operation* allocOp);
    Status ApplySpillContext(SpillContext& ctx, Operation* allocOp);
    Status PrintSpillFailedInfo(Operation* allocOp);
    std::vector<std::vector<int>> GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill);
    std::vector<std::vector<int>> GetDualSpillGroup(BufferPool& poolA, BufferPool& poolB, size_t sizeNeedSpill);
    Status GetGroupNextUseTime(std::vector<int> group, Operation* allocOp,
        std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache);

    Status SpillOnBlock();
    Status SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> coreLocation);
    Status FindFirstOrder(std::pair<CoreLocationType, MemoryType> &orderFirstPair);
    Status FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType &spillMemType);

    std::vector<Operation*>& GetViewOps(Operation* op) { return state_.schedInfoMap[op].viewOps; }
    void SetIsRetired(Operation* op, bool isRetired) { state_.schedInfoMap[op].isRetired = isRetired; }
    void SetCoreLocation(Operation* op, CoreLocationType loc) { state_.schedInfoMap[op].coreLocation = loc; }

    void UpdateL0MXMap(const std::vector<Operation*> &opList);
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H
