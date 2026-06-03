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
#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/dep_manager.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_base.h"
#include "passes/statistics/schedule_observer.h"
#include "schedule_main_loop_base.h"

namespace npu::tile_fwk {

inline int BytesPerElement(DataType dataType) { return BytesOf(dataType); }

// AIV0/AIV1 split: each has its own BufferPool. Observer collapses via ToCoreLocation.
enum class CoreLocationType { AIC = 0, AIV0 = 1, AIV1 = 2, UNKNOWN = 3 };

const std::unordered_set<Opcode> USE_LESS_OPS = {
    Opcode::OP_NOP,         Opcode::OP_RESHAPE,   Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_SHMEM_WAIT_UNTIL,
    Opcode::OP_BIND_TENSOR, Opcode::OP_VIEW_TYPE, Opcode::OP_HUB};

const std::unordered_set<Opcode> COPY_IN_OPS = {
    Opcode::OP_COPY_IN,        Opcode::OP_UB_COPY_IN, Opcode::OP_L1_COPY_IN,  Opcode::OP_L1_COPY_IN_FRACTAL_Z,
    Opcode::OP_L1_COPY_IN_DMA, Opcode::OP_L1_COPY_UB, Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_ONE = {
    CoreLocationType::AIC, CoreLocationType::AIV0};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_TWO = {
    CoreLocationType::AIC, CoreLocationType::AIV0, CoreLocationType::AIV1};

struct IssueQueue {
    bool busy{false};
    Operation* curIssue = nullptr;
    int curOpRetireCycle{-1};
    std::vector<Operation*> queue;
    // 比较函数指针，由OoOScheduler设置
    std::function<bool(Operation*, Operation*)> compareFunc;

    IssueQueue() {}
    ~IssueQueue() {}

    void SetCompareFunc(std::function<bool(Operation*, Operation*)> func)
    {
        compareFunc = func;
    }

    void Insert(Operation* op)
    {
        queue.push_back(op);
        if (compareFunc) {
            std::push_heap(queue.begin(), queue.end(), compareFunc);
        }
    }

    void ForceInsertAfterFirst(Operation* op) {
        // 插入到第一个元素之后
        auto insertPos = queue.begin() + 1;
        queue.insert(insertPos, op);
    }

    bool Empty() { return queue.size() == 0; }

    Operation* Front()
    {
        return queue[0];
    }

    Operation* PopFront()
    {
        if (compareFunc) {
            std::pop_heap(queue.begin(), queue.end(), compareFunc);
        }
        Operation* op = queue.back();
        queue.pop_back();
        return op;
    }

    void DeleteOp(Operation* op) {
        auto it = std::find(queue.begin(), queue.end(), op);
        if (it != queue.end()) {
            queue.erase(it);
            if (compareFunc) {
                std::make_heap(queue.begin(), queue.end(), compareFunc);
            }
        }
    }
};

struct SpillContext {
    std::vector<Operation*> newCopyoutOps;    // spill产生的copyout ops
    std::vector<Operation*> newAllocOps; // 需要加入allocIssueQueue的alloc ops
    std::vector<int> spillMemIds;             // 需要从tensorOccupyMap清理的memIds
    int newNotRetiredCopyOutSize{0};      // spill 新插的未执行的 copy_out 数量
    int deleteRetiredOpSize{0};       // 删除 op 中 已执行的 op 数量
    int deleteNotRetiredOpSize{0};      // 删除 op 中 未被执行的 op 数量
    std::vector<std::tuple<Operation*, MemoryType, CoreLocationType>> deleteAllocOps;
};

struct SingleSpillCreatedOps {
    Operation* copyoutOp{nullptr};
    Operation* allocOp{nullptr};
    Operation* copyinOp{nullptr};
    LogicalTensorPtr gmTensor{nullptr};

    void Record(Operation* copyout = nullptr,
                Operation* alloc = nullptr,
                Operation* copyin = nullptr,
                LogicalTensorPtr gm = nullptr) {
        if (copyout) copyoutOp = copyout;
        if (alloc)   allocOp   = alloc;
        if (copyin)  copyinOp  = copyin;
        if (gm)      gmTensor  = gm;
    }
};

class OoOScheduler : public ScheduleBase, public ScheduleMainLoopBase {
public:
    Status Schedule(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function& function) : function_(function) {}

    // Non-owning observer. Caller must ensure the observer outlives the whole Schedule() call.
    void AddObserver(ScheduleObserver* observer) { observers_.push_back(observer); }
    std::vector<Operation*> GetNewOperations() { return newOperations_; }
    int64_t workspaceOffset{0};
    std::unordered_map<PipeType, int> pipeEndTime;

private:
    // 存储排好顺序的Operation指针
    std::vector<Operation*> orderedOps;
    // Operation属性管理
    std::unordered_map<Operation*, int> opExecOrderMap;
    std::unordered_map<Operation*, PipeType> opPipeTypeMap;
    std::unordered_map<Operation*, bool> opIsAllocMap;
    std::unordered_map<Operation*, bool> opIsRetiredMap;
    std::unordered_map<Operation*, std::vector<Operation*>> opViewOpsMap;
    std::unordered_map<Operation*, CoreLocationType> opCoreLocationMap;

    std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS;

    // 分核数据结构
    std::unordered_map<CoreLocationType, std::map<npu::tile_fwk::MemoryType, BufferPool>> bufferManagerMap;

    std::unordered_map<int, Operation*> tensorOccupyMap;
    // tensor和其初始化时对应的alloc的core类型 memId-core类型
    std::unordered_map<int, Operation*> tensorAllocMap;

    std::unordered_map<CoreLocationType, std::map<MemoryType, IssueQueue>> allocIssueQueue;

    std::unordered_map<CoreLocationType, std::map<PipeType, IssueQueue>> issueQueues;

    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> l02L0MXMap_;

    Function& function_;
    int issueId{0};
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    std::vector<Operation*> newOperations_;
    std::vector<Operation*> operations_;
    std::vector<ScheduleObserver*> observers_;
    IRBuilder irBuilder_;
    // memId → DDRBufferKind. Doubles as the seen-set for EmitInitDDRBuffer.
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
    std::string GetOpInfo(Operation* op) const;
    int GetOOperandIdx(Operation* op, int curMemId);

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
        IssueQueue &pipe);
    void HandleViewOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle) override;
    Status AllocTensorMemRange(Operation* op);
    Status AllocViewTensorMemRange(Operation &operation);
    Status CheckAndUpdateLifecycle();
    void ApplySpillContext(SpillContext& ctx, Operation* allocOp);

    void UpdateIssueExecOrder();
    void PrintOpList(std::vector<Operation *> opList);
    Status PrintSpillFailedInfo(Operation* allocOp);

    // gen spill
    Status GenBufferSpill(Operation* allocOp, SpillContext &ctx);
    std::vector<int> SelectSpillBuffers(Operation* allocOp);
    Status GetGroupNextUseTime(std::vector<int> group, Operation* allocOp, std::vector<int> &groupNextUseTime,
        std::unordered_map<int, size_t> &nextUseTimeCache);
    bool IsBelongSpillBlackList(Operation* spillOp, Operation* op);
    void FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags);
    bool CheckMachineAndL1(Operation* spillOp, Operation* allocOp);

    Status SpillBuffer(int memId, Operation* spillAllocOp, SpillContext &ctx);
    Status HandleSpillMode(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillBufferFromDDR(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillGeneralBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillL1BufferFor3510(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillReshapeFromDDRFor3510(int spillMemId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillReshapeL1BufferFor3510(int memId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillL0CBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillMultiProducerBufferFor3510(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status CopyoutParticalBuffer(LogicalTensorPtr spillTensor, LogicalTensorPtr gmTensor, SpillContext &ctx);
    Status CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleOOperand, Operation* copyoutOp, Operation* spillAllocOp);
    LogicalTensorPtr CreateLocalTensor(LogicalTensorPtr spillTensor);
    LogicalTensorPtr CreateGMTensor(LogicalTensorPtr spillTensor, LogicalTensorPtr actualSpillTensor, int spillMemId);
    LogicalTensorPtr CreateParticalTensor(LogicalTensorPtr iOperand, LogicalTensorPtr oriOperand, LogicalTensorPtr spillTensor, std::vector<int64_t> toOffset);
    Operation* CreateAllocOp(LogicalTensorPtr oOperand);
    Operation* CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<OpImmediate> offset, bool isND2NZ = false);
    Operation* CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<OpImmediate> offset);
    Operation* CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<int64_t> toOffset, std::vector<SymbolicScalar> toDynOffset, std::vector<SymbolicScalar> fromDynValidShape);

    const std::vector<int64_t>& GetLargerShape(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);

    bool IsSmallShapeSpill(Operation* op);
    bool HasUnexecutedProducer(LogicalTensorPtr spillTensor);
    void UpdateSuccessorDependencies(Operation* succOp, Operation* spillOp,
        Operation* reloadCopyin, int spillMemId, int reloadMemId);
    void UpdatePredecessorAllocDependencies(Operation* succOp, Operation* reloadAlloc, int spillMemId);
    Status UpdateSmallShapeDependAndBuf(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap,
        int spillMemId, Operation* spillOp);

    // RemoveSmallShapeSpillResources 辅助函数
    void CollectUBSceneOpsAndTensors(Operation* producerOp, std::vector<Operation*>& opsToDelete,
        std::vector<LogicalTensorPtr>& tensorsToDelete);
    void CollectProducerChainForDeletion(
        LogicalTensorPtr spillTensor, std::vector<Operation*>& opsToDelete,
        std::vector<LogicalTensorPtr>& tensorsToDelete);
    size_t CleanupCollectedOperations(const std::vector<Operation*>& opsToDelete);
    void CleanupCollectedTensors(
        const std::vector<LogicalTensorPtr>& tensorsToDelete);
    void EraseOrphanedTensors(
        const std::vector<LogicalTensorPtr>& tensorsToDelete, const std::vector<Operation*>& opsToDelete);
    Status RemoveSmallShapeSpillResources(int spillMemId, LogicalTensorPtr spillTensor, SpillContext &ctx);

    Operation* GetSpillOp(int memId);
    LogicalTensorPtr GetSpillTensor(Operation* spillOp, int spillMemId);
    void CollectL0CConsumers(LogicalTensorPtr spillTensor, std::vector<Operation*> &consumers);
    Status GetActualSpill(Operation* op, Operation* &actualOp, LogicalTensorPtr &actualTensor);
    void EraseSchedulerSideMaps(Operation* op);
    Status UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    Status SpillOnBlock() override;
    Status SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> coreLocation);
    Status FindFirstOrder(std::pair<CoreLocationType, MemoryType> &orderFirstPair);
    Status FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType &spillMemType);
    Operation* SkipViewChain(Operation* start, bool followProducers);

    // 新增：插入Operation到orderedOps
    void InsertOrdered(Operation* insertOp);
    // 新增：Tensor输入更新辅助函数
    void UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr tensor);
    void UpdateTensorInputForView(Operation& op, Operation* spillSrcOp, LogicalTensorPtr tensor);

    void ReplaceViewOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId);
    void ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId);
    Status UpdateRemainMemid(int oldMemId, int newMemId);
    void UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp);
    int GetBufLastUseTime(Operation* op, int curMemId);
    int GetBufNextUseTime(int curMemId);
    int64_t CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset, DataType dataType);
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType);
    Status UpdateCopyoutScheduleInfo(Operation* op, LogicalTensorPtr spillTensor, int spillMemId, Operation* spillAllocOp, bool isRetired = true);
    void UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp);
    Status InsertOps(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, Operation* spillAllocOp, int memId);
    Status UpdateScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
        Operation* spillAllocOp, LogicalTensorPtr localTensor, Operation* spillOp);
    Status UpdateNeedDeleteScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
        Operation* spillAllocOp, LogicalTensorPtr spillTensor, Operation* spillOp, SpillContext &ctx);
    bool IsMultiProducerTensor(LogicalTensorPtr tensor);
    Status GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t> &toOffset,
 	    std::vector<SymbolicScalar> &toDynOffset, std::vector<SymbolicScalar> &fromDynValidShape) const;

    // ============ 辅助函数：获取Operation属性 ============
    int GetExecOrder(Operation* op) const { return opExecOrderMap.at(op); }
    PipeType GetPipeType(Operation* op) const { return opPipeTypeMap.at(op); }
    bool IsAlloc(Operation* op) const { return opIsAllocMap.at(op); }
    bool IsRetired(Operation* op) const { return opIsRetiredMap.at(op); }
    CoreLocationType& GetCoreLocation(Operation* op) { return opCoreLocationMap[op]; }
    const CoreLocationType& GetCoreLocation(Operation* op) const { return opCoreLocationMap.at(op); }
    std::vector<Operation*>& GetViewOps(Operation* op) { return opViewOpsMap[op]; }

    // 辅助函数：设置Operation属性
    void SetExecOrder(Operation* op, int order) { opExecOrderMap[op] = order; }
    void SetPipeType(Operation* op, PipeType type) { opPipeTypeMap[op] = type; }
    void SetIsAlloc(Operation* op, bool isAlloc) { opIsAllocMap[op] = isAlloc; }
    void SetIsRetired(Operation* op, bool isRetired) { opIsRetiredMap[op] = isRetired; }
    void SetCoreLocation(Operation* op, CoreLocationType loc) { opCoreLocationMap[op] = loc; }

    // 辅助函数：检查Operation是否在调度中
    bool IsOpInSchedule(Operation* op) const { return opExecOrderMap.find(op) != opExecOrderMap.end(); }

    void UpdateL0MXMap(const std::vector<Operation*> &opList);
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H
