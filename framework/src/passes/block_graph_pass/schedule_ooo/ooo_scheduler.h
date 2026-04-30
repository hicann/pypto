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
};

struct SpillInfo {
    int spillMemId_;
    Operation* spillOp_ = nullptr;
    LogicalTensorPtr spillTensor_;
    LogicalTensorPtr ddrTensor_;
    // A5 中 L1-spill 且 前序 op 不为 COPY_IN 时 为 true
    bool isSpecialL1_{false};
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
    // ============ Operation属性管理数据结构 ============
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

    std::unordered_map<MemoryType, std::map<int, Operation*>> tensorOccupyMap;
    // tensor和其初始化时对应的alloc的core类型 memId-core类型
    std::unordered_map<int, CoreLocationType> tensorAllocCoreMap;

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

    // Notification helpers — event construction lives in ooo_scheduler_notify.cpp
    // to keep scheduler main flows focused on scheduling logic.
    void NotifyPipeIssued(PipeType pipeType, int latency);
    void NotifyBufferAllocated(MemoryType memType, int memId);
    void NotifyBufferFreed(MemoryType memType, int memId);
    void NotifySpill(const SpillInfo& info, LocalBufferPtr allocBuffer);
    void NotifyScheduleEnd(bool success);

    // scheduler
    Status Init(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);

    // ============ 新增：基于Operation的初始化函数 ============
    Status InitOpEntry(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    Status InitOpCoreType(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    void InitOpViewOps(Operation* op);
    std::string GetOpInfo(Operation* op) const;
    int GetOOperandIdx(Operation* op, int curMemId);

    void InitCoreConfig(const std::vector<Operation *> &opList);
    void InitTensorCoreMap();
    void InitIssueQueuesAndBufferManager();

    Status GenSpillSchedule();
    Status ExecuteAllocIssue(Operation* op, size_t &pcIdx);
    Status RetireIssue(Operation* op);
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireCoreIssue(CoreLocationType targetCore, uint64_t& commitCnt, int& nextCycle);
     // ScheduleMainLoopBase 钩子实现
    Status PreMainLoop() override;
    Status PostMainLoop() override;
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle) override;
    // 新增：基于Operation*的版本
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op);
    Status BufferAllocStage(uint64_t& commitCnt) override;
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType,
        IssueQueue &pipe);
    // 新增：基于Operation*的版本
    void HandleViewOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle) override;
    // 新增：基于Operation*的版本
    Status AllocTensorMemRange(Operation* op);
    Status AllocViewTensorMemRange(Operation &operation);
    Status CheckAndUpdateLifecycle();

    void UpdateIssueExecOrder();
    void PrintOpList(std::vector<Operation *> opList);
    Status PrintSpillFailedInfo(Operation* allocOp, bool isGenSpill);

    // gen spill
    Status GenSpillOp(size_t &pcIdx);
    Status GenBufferSpill(Operation* allocOp);
    Status SelectSpillBuffers(LocalBufferPtr allocBuffer, Operation* allocOp,
        std::vector<int> &spillGroup, bool isGenSpill);
    Status GetGroupNextUseOrder(std::vector<int> group, Operation* allocOp,
        std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill);
    Operation* GetSpillIssue(Operation* allocOp, int memId, bool isGenSpill);
    bool CheckMachineAndL1(Operation* spillOp, Operation* allocOp);
    bool IsBelongSpillBlackList(Operation* spillOp, Operation* op);
    void FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags);
    Status SpillAllBuffer(Operation* allocOp, size_t &pcIdx, bool isGenSpill, LocalBufferPtr allocBuffer);
    bool IsPartialWrite(const Operation &op) const;
    Status SpillMultiBuffer(Operation* allocOp, std::vector<int> spillGroup, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status GetSpillInfo(Operation* allocOp, int spillMemId, bool isGenSpill, SpillInfo &spillInfo);
    Status GetSpillTensor(Operation* spillOp, int spillMemId, LogicalTensorPtr &spillTensor);
    Status SpillBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status SpillOutBuffer(SpillInfo &spillInfo, Operation* op, size_t &pcIdx, bool isGenSpill);
    Status CreateSpecialL1Copyout(SpillInfo &spillInfo, Operation* &spillCopyoutOp, int &bufLastUseOrder,
        bool &isFinish, bool isGenSpill, size_t &pcIdx);
    Status CreateSpillCopyout(Operation* spillOp, LogicalTensorPtr spillTensor, int spillMemId,
        Operation* &spillCopyoutOp, const SpillInfo &spillInfo);
    Status HandleReshapeSpillPath(SpillInfo &spillInfo, Operation* &actualSpillOp,
        LogicalTensorPtr &actualSpillTensor, bool &isFinish, bool isGenSpill, size_t &pcIdx);
    Status CreateSpillCopyoutForSmallShape(SpillInfo &spillInfo, LogicalTensorPtr l1Tensor, bool &isFinish,
        bool isGenSpill, size_t &pcIdx);
    Status TryCreateSpillCopyoutForSmallShape(SpillInfo &spillInfo, Operation* candidateOp, bool &isFinish,
        bool isGenSpill, size_t &pcIdx);
    Status BuildSmallShapeDdrTensor(SpillInfo &spillInfo, LogicalTensorPtr l1Tensor,
        LogicalTensorPtr &ddrTensor);
    Status ResolveSmallShapeActualSpill(Operation* producerOp,
        Operation* &actualOp, LogicalTensorPtr &actualTensor);
    Status CreateSmallShapeCopyout(SpillInfo &spillInfo, Operation* producerOp,
        LogicalTensorPtr ddrTensor, bool isGenSpill, size_t &pcIdx);
    Status ConfigSmallShapeCopyoutAttrs(Operation &copyOutOp, Operation* producerOp,
        LogicalTensorPtr actualTensor);
    Status SpillInBuffer(SpillInfo &spillInfo, Operation* allocOp, MemoryType bufferType, bool isGenSpill);
    Status SpillInReshapeBuffer(SpillInfo &spillInfo, Operation* allocOp, bool isGenSpill);
    Status SpillReshapeParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, LogicalTensorPtr reshapeTensor,
        bool isGenSpill);
    LogicalTensorPtr CreateReshapeL1Tensor(LogicalTensorPtr iOperand, LogicalTensorPtr reshapeTensor);
    Status UpdateReshapeDependAndBuf(Operation* allocOp, SpillInfo &spillInfo, LogicalTensorPtr reshapeTensor);
    Status CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor, LogicalTensorPtr spillTensor,
        Operation* spillOp, std::pair<Operation*, Operation*> &reloadOps, bool isSpecialL1);
    Status UpdateReloadIssueInfo(Operation* reloadAlloc, Operation* reloadCopyin,
        Operation* spillOp, int spillMemId, Operation* allocOp);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    // spill assemble
    Status SpillAssembleBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status SpillParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, Operation* producerOp,
        LogicalTensorPtr assembleTensor, bool &isFirst, bool isGenSpill);
    Status FindAssembleWithSpillTensor(SpillInfo &spillInfo, std::vector<Operation*> &assembleOps);
    bool IsSupportedPartialWriteProducer(const Operation &op) const;
    Status GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t> &toOffset,
        std::vector<SymbolicScalar> &toDynOffset, std::vector<SymbolicScalar> &fromDynValidShape) const;
    Operation* FindAllocForAssembleProducers(const std::vector<Operation*> &assembleOps) const;
    bool HasNZHorizontalSlice(const std::vector<Operation*> &assembleOps) const;
    Status RejectIfNZHorizontalSlice(SpillInfo &spillInfo, std::vector<Operation*> &assembleOps);
    Status ReplayPartialWriteProducers(SpillInfo &spillInfo, Operation* allocOp,
        LogicalTensorPtr assembleTensor, const std::vector<Operation*> &assembleOps, bool isGenSpill);
    Status SpillOnBlock() override;
    Status SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> orderFirstPair);
    Status FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType &spillMemType);
    Status FindFirstOrder(std::pair<CoreLocationType, MemoryType> &orderFirstPair);
    Operation* SkipViewChain(Operation* start, bool followProducers);

    // 新增：插入Operation到orderedOps
    void InsertOrdered(Operation* insertOp);
    // 新增：Tensor输入更新辅助函数
    void UpdateTensorInputFor(Operation* targetOp, Operation* spillSrcOp, LogicalTensorPtr tensor);
    void UpdateTensorInputForOperand(Operation* targetOp, size_t index, Operation* spillSrcOp, LogicalTensorPtr tensor);
    void UpdateTensorInputForView(Operation& op, Operation* spillSrcOp, LogicalTensorPtr tensor);

    Status UpdateReloadIssueDepend(Operation* reloadCopyin, Operation* spillOp, int spillMemId);
    Status UpdateRemainOpBufId(int oldMemId, int newMemId);
    void ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId);
    void ReplaceViewOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId);
    void UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp);
    void GetActualSpillInfo(Operation* spillOp, std::pair<LogicalTensorPtr, Operation*>& actualInfo);
    void UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset,
        Operation* spillOp, bool isSpecialL1);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    int GetBufNextUseOrder(Operation* op, int curMemId);
    int GetBufLastUseOrder(Operation* op, int curMemId);
    Operation* GetBufLastWriteOp(Operation* op, int curMemId);
    bool CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType);
    int GetMemidAllocPriority(int memId);
    Operation* UpdateIssueAttr(Operation &newOp, std::vector<int> memIds, Operation* allocOp,
        int &bufNextUseOrder, bool isGenSpill);
    Status UpdateAssembleBuffer(SpillInfo &spillInfo, LocalBufferPtr allocBuffer, LogicalTensorPtr assembleTensor);
    LogicalTensorPtr CreateAssemblePartTensor(
        LogicalTensorPtr iOperand, LogicalTensorPtr assembleTensor, const std::vector<int64_t> &toOffset);
    Status UpdateCopyOutMode(Operation& copyOutOp);
    Status UpdateCopyInMode(Operation& copyInOp);
    void AllocWorkspaceGM(const std::vector<Operation *> &opList);

    // buffer rearrange
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType, CoreLocationType coreLocation, bool isGenSpill);
    Status RearrangeBuffers(Operation* op, bool isGenSpillStage, bool &rearrangeUBBF16);
    Status GenRearrangeCopyOp(Operation* op, MemoryType memType, int memId, int &newMemId, bool &rearrangeUBBF16);
    Status UpdateMemId(int oldMemId, int newMemId);
    void UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp);
    void ProcessMoveIssue(Operation* moveOp, Operation* allocOp, MemoryType memType, int oldMemId, int newMemId);
    Status UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool &bufferManager);
    Status FindMoveFromTensor(Operation &occupyOp, int oldMemId, MemoryType memType,
        bool &rearrangeUBBF16, LogicalTensorPtr &moveFromTensor);
    Status GetMoveOpInTensor(Opcode moveOpcode, Operation &occupyOp, LogicalTensorPtr &inTensor,
        LogicalTensorPtr &moveFromTensor);

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
