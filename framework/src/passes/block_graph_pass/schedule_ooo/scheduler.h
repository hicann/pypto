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
 * \file scheduler.h
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
#include "passes/statistics/ooo_schedule_statistic.h"

namespace npu::tile_fwk {

inline int BytesPerElement(DataType dataType) { return BytesOf(dataType); }

inline uint64_t CeilAlign(uint64_t a, int b) { return ((a + b - 1) / b) * b; }

using LocalBufferPtr = std::shared_ptr<LocalBuffer>;

const std::unordered_set<Opcode> USE_LESS_OPS = {
    Opcode::OP_NOP,         Opcode::OP_RESHAPE,   Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_SHMEM_WAIT_UNTIL,
    Opcode::OP_BIND_TENSOR, Opcode::OP_VIEW_TYPE, Opcode::OP_HUB};

const std::unordered_set<Opcode> COPY_IN_OPS = {
    Opcode::OP_COPY_IN,        Opcode::OP_UB_COPY_IN, Opcode::OP_L1_COPY_IN,  Opcode::OP_L1_COPY_IN_FRACTAL_Z,
    Opcode::OP_L1_COPY_IN_DMA, Opcode::OP_L1_COPY_UB, Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND};

const std::unordered_map<OpCoreType, std::vector<int>> CORE_INIT_CONFIGS_HARDWARE_TWO = {
    {OpCoreType::AIV, {0, 1}}, {OpCoreType::AIC, {0}}};

const std::unordered_map<OpCoreType, std::vector<int>> CORE_INIT_CONFIGS_HARDWARE_ONE = {
    {OpCoreType::AIV, {0}}, {OpCoreType::AIC, {0}}};

const std::unordered_map<OpCoreType, std::pair<OpCoreType, int>> opCoreTypeMap{
    {OpCoreType::AIV, std::make_pair(OpCoreType::AIV, 0)}, {OpCoreType::AIC, std::make_pair(OpCoreType::AIC, 0)}};

struct IssueQueue {
    bool busy{false};
    Operation* curIssue = nullptr;
    int curOpRetireCycle{-1};
    std::vector<Operation*> queue;
    // 比较函数指针，由OoOScheduler设置
    std::function<bool(Operation*, Operation*)> compareFunc;

    IssueQueue() {}
    ~IssueQueue() {}

    void SetCompareFunc(std::function<bool(Operation*, Operation*)> func) {
        compareFunc = func;
    }

    void Insert(Operation* op) {
        queue.push_back(op);
        if (compareFunc) {
            std::push_heap(queue.begin(), queue.end(), compareFunc);
        }
    }

    bool Empty() { return queue.size() == 0; }

    Operation* Front() {
        return queue[0];
    }

    Operation* PopFront() {
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

class OoOScheduler {
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
    std::unordered_map<Operation*, std::pair<OpCoreType, int>> opCoreLocationMap;
    std::unordered_map<Operation*, std::vector<int>> opReqMemIdsMap;

    DependencyManager depManager_;

    std::unordered_map<OpCoreType, std::vector<int>> CORE_INIT_CONFIGS;

    std::unordered_map<int, LocalBufferPtr> localBufferMap;
    // 分核数据结构
    std::unordered_map<OpCoreType, std::map<int, std::map<npu::tile_fwk::MemoryType, BufferPool>>> bufferManagerMap;

    std::unordered_map<int, int> bufRefCount_;
    std::unordered_map<MemoryType, std::map<int, Operation*>> tensorOccupyMap;
    // tensor和其初始化时对应的alloc的core类型 memId-core类型
    std::unordered_map<int, std::pair<OpCoreType, int>> tensorAllocCoreMap;

    std::unordered_map<OpCoreType, std::map<int, std::map<MemoryType, IssueQueue>>> allocIssueQueue;

    std::unordered_map<OpCoreType, std::map<int, std::map<PipeType, IssueQueue>>> issueQueues;

    std::unordered_map<MemoryType, int64_t> localMemorySize;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> l02L0MXMap_;

    Function& function_;
    int issueId{0};
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    uint64_t numTotalIssues{0};
    std::vector<Operation*> newOperations_;
    std::vector<Operation*> operations_;

    // scheduler
    Status Init(
        const std::vector<Operation*>& operations,
        const std::unordered_map<Operation*, std::pair<OpCoreType, int>>& opCoreMap =
            std::unordered_map<Operation*, std::pair<OpCoreType, int>>(),
        const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);

    // ============ 新增：基于Operation的初始化函数 ============
    Status InitOpEntry(Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap);
    Status InitOpCoreType(Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap);
    void InitOpViewOps(Operation* op);
    std::string GetOpInfo(Operation* op) const;
    int GetOOperandIdx(Operation* op, int curMemId);

    void InitCoreConfig(const std::vector<Operation *> &operations);
    Status CheckOpBufferSize(Operation *op);
    std::string DumpOpInfo(Operation &op);
    Status CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap);
    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    Status InitBufRefCount();
    void UpdateBufRefCount(Operation* op, LogicalTensorPtr tensor);
    Status CheckAllocIssue();
    void InitTensorCoreMap();
    void UpdateAllocMap(Operation* op, std::map<int, Operation*> &tensorAllocMap);
    void InitIssueQueuesAndBufferManager();

    Status GenSpillSchedule();
    Status ExecuteAllocIssue(Operation* op, size_t &pcIdx);
    Status RetireIssue(Operation* op);
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireCoreIssue(OpCoreType coreType, int idx, uint64_t& commitCnt, int& nextCycle);
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    // 新增：基于Operation*的版本
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType,
        IssueQueue &pipe);
    // 新增：基于Operation*的版本
    void HandleViewOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle);
    // 新增：基于Operation*的版本
    Status AllocTensorMemRange(Operation* op);
    Status AllocViewTensorMemRange(Operation &operation);
    Status CheckAndUpdateLifecycle();

    void UpdateIssueExecOrder();
    size_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype);
    Status DelBufRefCount(const int memId);
    void UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree);
    void PrintOpList(std::vector<Operation *> operations);
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
    bool CheckParallelL0C2L1(Operation* spillOp);
    bool IsBelongSpillBlackList(Operation* spillOp, Operation* op);
    void FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags);
    Status SpillAllBuffer(Operation* allocOp, size_t &pcIdx, bool isGenSpill, LocalBufferPtr allocBuffer);
    Status SpillMultiBuffer(Operation* allocOp, std::vector<int> spillGroup, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status GetSpillInfo(Operation* allocOp, int spillMemId, bool isGenSpill, SpillInfo &spillInfo);
    Status GetSpillTensor(Operation* spillOp, int spillMemId, LogicalTensorPtr &spillTensor);
    Status SpillBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status SpillOutBuffer(SpillInfo &spillInfo, Operation* op, size_t &pcIdx, bool isGenSpill);
    Status CreateSpecialL1Copyout(SpillInfo &spillInfo, Operation* &spillCopyoutOp, int &bufLastUseOrder, bool &isFinish);
    Status CreateSpillCopyout(Operation* spillOp, LogicalTensorPtr spillTensor, int spillMemId,
        Operation* &spillCopyoutOp, const SpillInfo &spillInfo);
    Status SpillInBuffer(SpillInfo &spillInfo, Operation* allocOp, MemoryType bufferType, bool isGenSpill);
    Status SpillInReshapeBuffer(SpillInfo &spillInfo, Operation* allocOp, bool isGenSpill);
    Status SpillReshapeParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, LogicalTensorPtr reshapeTensor, bool isGenSpill);
    LogicalTensorPtr CreateReshapeL1Tensor(LogicalTensorPtr iOperand, LogicalTensorPtr reshapeTensor);
    Status UpdateReshapeDependAndBuf(Operation* allocOp, SpillInfo &spillInfo, LogicalTensorPtr reshapeTensor);
    Status CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor, LogicalTensorPtr spillTensor,
        Operation* spillOp, std::pair<Operation*, Operation*> &reloadOps);
    Status UpdateReloadIssueInfo(Operation* reloadAlloc, Operation* reloadCopyin,
        Operation* spillOp, int spillMemId, Operation* allocOp);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    Status SpillAssembleBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status SpillParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, Operation* assembleOp,
        LogicalTensorPtr assembleTensor, bool &isFirst, bool isGenSpill);
    Status FindAssembleWithSpillTensor(SpillInfo &spillInfo, std::vector<Operation*> &assembleOps);
    Status SpillOnBlock();
    Status SpillOnCoreBlock(OpCoreType coreType, int idx, bool &didSpill);
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
    void UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp);
    void UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset,
        Operation* spillOp, int64_t workspaceBaseOffset);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    int GetBufNextUseOrder(Operation* op, int curMemId);
    int GetBufLastUseOrder(Operation* op, int curMemId);
    Operation* GetBufLastWriteOp(Operation* op, int curMemId);
    OoOSchedulerCheck::SpillInfo RecordSpillInfo(MemoryType bufferType, int memId, LocalBufferPtr allocIssue,
        LogicalTensorPtr spillOutTensor, bool needCopyOut);
    bool CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType);
    int GetMemidAllocPriority(int memId);
    Operation* UpdateIssueAttr(Operation &newOp, std::vector<int> memIds, Operation* allocOp, int &bufNextUseOrder, bool isGenSpill);
    Status UpdateAssembleBuffer(SpillInfo &spillInfo, LocalBufferPtr allocBuffer, LogicalTensorPtr assembleTensor);
    LogicalTensorPtr CreateAssemblePartTensor(LogicalTensorPtr iOperand, LogicalTensorPtr assembleTensor,
        SpillInfo &spillInfo, std::shared_ptr<AssembleOpAttribute> assembleAttr);
    int64_t CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset);
    void GetWorkspaceBaseOffset(LogicalTensorPtr ddrTensor, int64_t& base);
    Status UpdateCopyOutMode(Operation& copyOutOp);
    Status UpdateCopyInMode(Operation& copyInOp);

    // buffer rearrange
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType, std::pair<OpCoreType, int> corePair, bool isGenSpill);
    Status RearrangeBuffers(Operation* op, bool isGenSpillStage, bool &rearrangeUBBF16);
    Status GenRearrangeCopyOp(Operation* op, MemoryType memType, int memId, int &newMemId, bool &rearrangeUBBF16);
    Status UpdateMemId(int oldMemId, int newMemId);
    void UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp);
    void ProcessMoveIssue(Operation* moveOp, Operation* allocOp, MemoryType memType, int oldMemId, int newMemId);
    Status UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool &bufferManager);
    Status FindMoveFromTensor(Operation &occupyOp, int oldMemId, MemoryType memType, bool &rearrangeUBBF16, LogicalTensorPtr &moveFromTensor);
    Status GetMoveOpInTensor(Opcode moveOpcode, Operation &occupyOp, LogicalTensorPtr &inTensor, LogicalTensorPtr &moveFromTensor);

    // ============ 辅助函数：获取Operation属性 ============
    int GetExecOrder(Operation* op) const { return opExecOrderMap.at(op); }
    PipeType GetPipeType(Operation* op) const { return opPipeTypeMap.at(op); }
    bool IsAlloc(Operation* op) const { return opIsAllocMap.at(op); }
    bool IsRetired(Operation* op) const { return opIsRetiredMap.at(op); }
    std::pair<OpCoreType, int>& GetCoreLocation(Operation* op) { return opCoreLocationMap[op]; }
    const std::pair<OpCoreType, int>& GetCoreLocation(Operation* op) const { return opCoreLocationMap.at(op); }
    std::vector<Operation*>& GetViewOps(Operation* op) { return opViewOpsMap[op]; }
    std::unordered_set<Operation *> &GetPredecessors(Operation *op) { return depManager_.GetPredecessors(op); }
    std::unordered_set<Operation *> &GetSuccessors(Operation *op) { return depManager_.GetSuccessors(op); }
    std::vector<int>& GetReqMemIds(Operation* op) { return opReqMemIdsMap[op]; }

    // 辅助函数：设置Operation属性
    void SetExecOrder(Operation* op, int order) { opExecOrderMap[op] = order; }
    void SetPipeType(Operation* op, PipeType type) { opPipeTypeMap[op] = type; }
    void SetIsAlloc(Operation* op, bool isAlloc) { opIsAllocMap[op] = isAlloc; }
    void SetIsRetired(Operation* op, bool isRetired) { opIsRetiredMap[op] = isRetired; }
    void SetCoreLocation(Operation* op, const std::pair<OpCoreType, int>& loc) { opCoreLocationMap[op] = loc; }
    void SetReqMemIds(Operation* op, const std::vector<int>& memIds) { opReqMemIdsMap[op] = memIds; }

    // 辅助函数：检查Operation是否在调度中
    bool IsOpInSchedule(Operation* op) const { return opExecOrderMap.find(op) != opExecOrderMap.end(); }

public:
    Status Schedule(
        const std::vector<Operation*>& operations,
        const std::unordered_map<Operation*, std::pair<OpCoreType, int>>& opCoreMap =
            std::unordered_map<Operation*, std::pair<OpCoreType, int>>(),
        const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function& function) : function_(function) {}

    std::vector<Operation*> GetNewOperations() { return newOperations_; }
    int64_t workspaceOffset{0};
    int clock{0};
    OoOSchedulerCheck oooCheck;
    std::unordered_map<PipeType, int> pipeEndTime;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H
