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
#include "passes/block_graph_pass/schedule_ooo/common/schedule_base.h"
#include "passes/statistics/schedule_observer.h"
#include "passes/block_graph_pass/schedule_ooo/common/schedule_main_loop_base.h"
#include "passes/block_graph_pass/schedule_ooo/common/schedule_notifier.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/spill_engine.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/dualdst_engine.h"

namespace npu::tile_fwk {

// CoreLocationType, COPY_IN_OPS, CORE_INIT_CONFIGS_HARDWARE_ONE/TWO,
// OpSchedInfo, SpillContext, SingleSpillCreatedOps, DualDstPair, DualDstAllocCtx
// are now defined in schedule_state.h (included via schedule_base.h).

class OoOScheduler : public ScheduleBase, public ScheduleMainLoopBase {
public:
    Status Schedule(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function& function)
        : function_(function), notifier_(),
          spillEngine_(state_, notifier_, function_),
          dualDstEngine_(state_, notifier_, function_) {}

    // Non-owning observer. Caller must ensure the observer outlives the whole Schedule() call.
    void AddObserver(ScheduleObserver* observer) { notifier_.AddObserver(observer); }
    // Defensive dedupe: dualdst 跨核 wake / spill 重入路径下,
    // 上层 Function::ScheduleBy -> RefreshOpPosition 不允许重复 op 出现。
    // 返回保留首次出现顺序的去重副本(内部 newOperations_ 不变,便于诊断)。
    std::vector<Operation*> GetNewOperations();

    // === DualDst 开关 ===
    // OoOSchedule 在调用 Schedule() 之前设置(默认 false 保持原行为)。
    void SetEnableDualDst(bool v) { dualDstEngine_.SetEnableDualDst(v); }

    // ScheduleMainLoopBase virtual getters — delegate to ScheduleState
    int& GetClock() override { return state_.clock; }
    uint64_t& GetNumTotalIssues() override { return state_.numTotalIssues; }

private:
    // orderedOps, clock, numTotalIssues, allocIssueQueue are now in ScheduleState
    // (via ScheduleBase proxy). InsertOrdered is also in ScheduleState.

    Function& function_;
    ScheduleNotifier notifier_;
    SpillEngine spillEngine_;
    DualDstEngine dualDstEngine_;

    // Fields below are now in ScheduleState (via ScheduleBase proxy).
    // schedInfoMap_, tensorOccupyMap, tensorAllocMap, bufferManagerMap,
    // CORE_INIT_CONFIGS, dualDstMemIdCoreOverride_, newOperations_,
    // workspaceOffset, pipeEndTime, workspaceMemId
    // orderedOps, clock, numTotalIssues, allocIssueQueue
    // are inherited from ScheduleBase as proxy references to state_.

    std::unordered_map<CoreLocationType, std::map<PipeType, OpQueue>> issueQueues;

    uint64_t spillIssueCnt{0};
    std::vector<ScheduleObserver*>& observers_ = notifier_.observers();

    IRBuilder irBuilder_;
    std::unordered_map<int, DDRBufferKind> ddrKindMap_;

    // Notification helpers — bodies live in ooo_scheduler_notify.cpp.
    void NotifyOpLaunch(Operation* op, int cycleEnd);
    void NotifyOpRetire(Operation* op, const std::vector<int>& freedMemIds);
    void NotifyAllocExec(Operation* op, int memId);
    void NotifySpill(LogicalTensorPtr spillTensor, int spillMemId, Operation* spillAllocOp,
        const SingleSpillCreatedOps& created);
    void NotifyBufferRearrange(Operation* triggerOp, MemoryType memType,
        std::vector<BufferRearrangeEvent::Change> changes);
    void NotifyAllocFail(Operation* triggerOp, MemoryType memType, uint64_t requestSize);
    void NotifyScheduleEnd(bool success);
    void NotifyInitDDRBuffers();
    void NotifyMainLoopBegin();
    void NotifyMainLoopEnd();
    // Emit a single INIT_DDR_BUFFER event for `t` if its memId hasn't been registered yet.
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
     // ScheduleMainLoopBase 钩子实现
    Status PreMainLoop() override;
    Status PostMainLoop() override;
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle) override;
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op, std::vector<int>& freedMemIds);
    Status BufferAllocStage(uint64_t& commitCnt) override;
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType,
        OpQueue &pipe);
    // ExecuteAllocIssue 的两个分支被拆出,统一以 (allocated 是否成功) 与调用方约定:
    //   返回 SUCCESS && allocated=true  : 已分配,调用方应 PopFront 并继续;
    //   返回 SUCCESS && allocated=false : 暂时无法分配 (Full),调用方应 break;
    //   返回 FAILED                       : 真错误。
    Status TryDualDstAllocOnce(Operation* op, uint64_t& commitCnt, bool& allocated);
    Status TryRegularAllocOnce(Operation* op, MemoryType memType, CoreLocationType coreLocation,
                               const std::vector<int>& reqMemIds,
                               uint64_t& commitCnt, bool& allocated);
    // 新增：基于Operation*的版本
    void HandleViewOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle) override;
    Status AllocTensorMemRange(Operation* op);
    Status AllocViewTensorMemRange(Operation &operation);
    Status CheckAndUpdateLifecycle();
    void UpdateIssueExecOrder();
    void PrintOpList(std::vector<Operation *> opList);
    // PrintSpillFailedInfo moved back to OoOScheduler (spill_buffer.cpp).
    // gen spill — orchestration layer (spill_buffer.cpp)
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

    Status SpillOnBlock() override;
    Status SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> coreLocation);
    Status FindFirstOrder(std::pair<CoreLocationType, MemoryType> &orderFirstPair);
    Status FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType &spillMemType);

    // InsertOrdered is now in ScheduleState (via ScheduleBase proxy).

    // === DualDst spill ===
    // SelectSpillBuffers, GenBufferSpill, HasEnoughBuffer, RearrangeBuffer,
    // ApplySpillContext, PrintSpillFailedInfo, GetSpillGroup, GetDualSpillGroup,
    // GetGroupNextUseTime are now in OoOScheduler (spill_buffer.cpp).
    // SpillBuffer (pure execution) remains in SpillEngine; called via
    // spillEngine_.SpillBuffer(). SpillEngine query helpers
    // (GetSpillOp, GetBufNextUseTime, IsBelongSpillBlackList, EraseSchedulerSideMaps) are public
    // so OoOScheduler orchestration can call them.

    std::vector<Operation*>& GetViewOps(Operation* op) { return schedInfoMap_[op].viewOps; }
    void SetIsRetired(Operation* op, bool isRetired) { schedInfoMap_[op].isRetired = isRetired; }
    void SetCoreLocation(Operation* op, CoreLocationType loc) { schedInfoMap_[op].coreLocation = loc; }

    void UpdateL0MXMap(const std::vector<Operation*> &opList);
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H
