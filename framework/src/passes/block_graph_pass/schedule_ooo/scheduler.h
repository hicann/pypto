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
#include "tilefwk/platform.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_check/schedule_ooo_checker.h"
#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/statistics/ooo_schedule_statistic.h"

namespace npu::tile_fwk {

inline int BytesPerElement(DataType dataType) {
    return BytesOf(dataType);
}

inline uint64_t CeilAlign(uint64_t a, int b) {
    return ((a + b - 1) / b) * b;
}

using LocalBufferPtr = std::shared_ptr<LocalBuffer>;

const std::unordered_set<Opcode> USE_LESS_OPS = {
    Opcode::OP_NOP,
    Opcode::OP_RESHAPE, 
    Opcode::OP_VIEW, 
    Opcode::OP_ASSEMBLE, 
    Opcode::OP_COMM_WAIT_FLAG, 
    Opcode::OP_SHMEM_WAIT_UNTIL,
    Opcode::OP_BIND_TENSOR,
    Opcode::OP_VIEW_TYPE,
    Opcode::OP_HUB
};

const std::unordered_set<Opcode> COPY_IN_OPS = {
    Opcode::OP_COPY_IN,
    Opcode::OP_UB_COPY_IN,
    Opcode::OP_L1_COPY_IN,
    Opcode::OP_L1_COPY_IN_FRACTAL_Z,
    Opcode::OP_L1_COPY_IN_DMA,
    Opcode::OP_L1_COPY_UB,
    Opcode::OP_L0C_COPY_UB,
    Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND
};

struct IssueEntry {
    Operation &tileOp;
    int id{-1};
    int execOrder{-1};
    PipeType type{PipeType::PIPE_ALL};
    bool isAlloc{false};
    bool isRetired{false};
    std::vector<Operation*> viewOps;

    // 当前op的前序op
    std::unordered_set<int> predecessors;

    // 当前op的后序op
    std::unordered_set<int> successors;

    // op计算所需的memId
    std::vector<int> reqMemIds;

    IssueEntry(Operation &op, uint64_t issueId);
    void Clear();
    int GetOOperandIdx(int curMemId);
    void UpdateTensorInput(std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const;
    void UpdateTensorInputForOperand(size_t index, std::shared_ptr<IssueEntry> &spillSrcIssue,
        LogicalTensorPtr tensor) const;
    void UpdateTensorInputForView(Operation *op, std::shared_ptr<IssueEntry> &spillSrcIssue,
        LogicalTensorPtr tensor) const;
    std::string GetOpInfo();
};

using IssueEntryPtr = std::shared_ptr<IssueEntry>;

struct IssueQueue {
    bool busy{false};
    IssueEntryPtr curIssue = nullptr;
    int curOpRetireCycle{-1};
    std::vector<IssueEntryPtr> queue;

    IssueQueue() {}
    ~IssueQueue() {}

    void Insert(IssueEntryPtr op) {
        queue.push_back(op);
        std::push_heap(queue.begin(), queue.end(),
            [](IssueEntryPtr &a, IssueEntryPtr &b) { return a->execOrder > b->execOrder; });
    }

    bool Empty() {
        return queue.size() == 0;
    }

    IssueEntryPtr Front() {
        return queue[0];
    }

    IssueEntryPtr PopFront() {
        std::pop_heap(queue.begin(), queue.end(), [](IssueEntryPtr& a, IssueEntryPtr& b){
            return a->execOrder > b->execOrder;
        });
        IssueEntryPtr op = queue.back();
        queue.pop_back();
        return op;
    }
};

struct SpillInfo {
    int spillMemId_;
    IssueEntryPtr spillIssue_;
    LogicalTensorPtr spillTensor_;
    LogicalTensorPtr ddrTensor_;
};

class OoOScheduler {
private:
    std::vector<IssueEntryPtr> issueEntries;
    std::unordered_map<int, IssueEntryPtr> issueEntryMap;

    std::unordered_map<int, LocalBufferPtr> localBufferMap;
    std::unordered_map<npu::tile_fwk::MemoryType, BufferPool> bufferManagerMap;
    std::unordered_map<int, int> bufRefCount;
    std::unordered_map<MemoryType, std::map<int, IssueEntryPtr>> tensorOccupyMap;

    std::map<MemoryType, IssueQueue> allocIssueQueue;
    std::map<PipeType, IssueQueue> issueQueues;
    std::unordered_map<MemoryType, int64_t> localMemorySize;

    Function &function_;
    int issueId{0};
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    uint64_t numTotalIssues{0};
    std::vector<Operation *> newOperations_;

    bool issueFinish{false};
    std::map<IssueEntryPtr, std::map<MemoryType, int64_t>> recordBufferAllocate;
    std::map<IssueEntryPtr, std::pair<size_t, std::vector<IssueEntryPtr>>> recordIssueEntries;
    std::map<IssueEntryPtr, MemoryType> recordIssueBuffer;
    std::stack<std::pair<IssueEntryPtr, MemoryType>> needFreeIssueStack;
    std::map<IssueEntryPtr, bool> visitedIssue;
    std::map<IssueEntryPtr, std::unordered_map<int, int>> recordBufRefCount;
    // 回溯点位置,当前执行issue的全部信息,用于后期回退
    IssueEntryPtr backTraceIssue{nullptr};
    std::map<IssueEntryPtr, std::map<MemoryType, int64_t>> backTraceBufferAllocate;
    std::map<IssueEntryPtr, std::pair<size_t, std::vector<IssueEntryPtr>>> backTraceIssueEntries;
    std::map<IssueEntryPtr, std::unordered_map<int, int>> backTraceBufRefCount;
    std::unordered_map<IssueEntryPtr, int> depthCache_;
    // 回退点,防止死循环
    IssueEntryPtr rollBackNodeIssue{nullptr};
    int GetMaxDepthSimple(IssueEntryPtr issue);
    // scheduler
    Status Init(const std::vector<Operation *> &operations);
    void InitMemorySize();
    Status CheckOpBufferSize(Operation *op);
    std::string dumpOpInfo(Operation &op);
    void CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap);
    Status InitDependencies();
    void AddDependency(IssueEntryPtr preIssue, IssueEntryPtr postIssue, bool isAlloc);
    Status InitAllocDependencies(IssueEntryPtr issue, std::map<int, IssueEntryPtr> tensor2AllocMap);
    void InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    void InitLocalBufferForAxisCombine(LogicalTensorPtr oOperand, int memId);
    void InitBufRefCount();
    void UpdateBufRefCount(IssueEntryPtr issue, LogicalTensorPtr tensor);
    Status CheckAllocIssue();
    void UpdateAllocMap(IssueEntryPtr issue, std::map<int, IssueEntryPtr> &tensorAllocMap);
    void InitIssueQueuesAndBufferManager();

    Status GenSpillSchedule();
    Status ExecuteAllocIssue(IssueEntryPtr issue, size_t &pcIdx);
    Status RetireIssue(IssueEntryPtr issue);
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt);
    Status FreeBuffer(IssueEntryPtr issue);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType, 
        IssueQueue &pipe);
    Status LaunchIssueStage(int& nextCycle);
    Status AllocTensorMemRange(IssueEntryPtr issue);
    Status AllocViewTensorMemRange(Operation &operation);
    Status SpillOnBlock();
    Status CheckAndUpdateLifecycle();
    
    void InsertIssueEntries(IssueEntryPtr insertIssue);
    void UpdateIssueExecOrder();
    size_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype);
    Status DelBufRefCount(const int memId);
    void UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree);
    void PrintOpList(std::vector<Operation *> operations);
    void PrintDependencies();
    void PrintSpillFailedInfo(IssueEntryPtr allocIssue, MemoryType bufferType);
    Status PrintSpillFailedInfo(IssueEntryPtr allocIssue);

    // sort ops
    Status SortOps();
    Status PriorDFS(std::unordered_map<Opcode, int> preNodePriority);
    int GetDepth(IssueEntryPtr issue);
    Status DFSFromOutNode(std::vector<IssueEntryPtr> outNodeQueue, 
    std::unordered_map<Opcode, int> preNodePriority, std::map<IssueEntryPtr, bool> &visited);
    void DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
        std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority);
    void ForwardDfs(IssueEntryPtr curIssue, std::vector<IssueEntryPtr>& newIssueEntries,
        std::map<IssueEntryPtr, bool>& visited, std::unordered_map<Opcode, int> preNodePriority,
        std::deque<IssueEntryPtr> &queue);
    void QueueNotReadyPreNode(IssueEntryPtr curIssue, std::map<IssueEntryPtr, bool>& visited,
    std::unordered_map<Opcode, int> preNodePriority, std::deque<IssueEntryPtr> &queue);
    int GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, IssueEntryPtr issue);
    IssueEntryPtr FindNodeMinNumUnvisitedPreNode(
        std::map<IssueEntryPtr, bool> visited, std::vector<IssueEntryPtr> outNodeQueue);
    int GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited);
    void UpdatePreNodeQueue(std::unordered_set<IssueEntryPtr> &curr, 
        std::unordered_set<IssueEntryPtr> &preNodeTotal, std::map<IssueEntryPtr, bool>& visited);

    Status RollBack(size_t &startIndex, std::vector<IssueEntryPtr> &curIssueEntries,
        std::map<MemoryType, int64_t> &curMemoryMap);
    void GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
        std::vector<IssueEntryPtr> curIssueEntries, std::set<size_t> &AdvanceIndexList);
    void ReplaceIndex(std::vector<IssueEntryPtr> &curIssueEntries,
        std::set<size_t> AdvanceIndexList, size_t rollBackIndex);
    bool HasDependency(IssueEntryPtr rollBackIssue, IssueEntryPtr backIssue);
    void GetIssueIdx(IssueEntryPtr issue, size_t &index);
    void GetPreNode(size_t i, std::vector<IssueEntryPtr> curIssueEntries, size_t rollBackIndex,
    size_t backTraceIndex, std::set<size_t> &dependencyIndexList);
    
    Status LayerBasedDFS(int layerDepth);

    void ReorderIssue(std::vector<size_t> &preIdx, std::vector<IssueEntryPtr> &curIssueEntries, size_t startIndex);
    void FindIndex(IssueEntryPtr issue, std::vector<IssueEntryPtr> curIssueEntries, size_t &index);
    Status FindConsumerList(size_t consumerIndex, std::vector<size_t> &preIssue, std::vector<IssueEntryPtr> &curIssueEntries);
    Status UpdateOOperandPreDependence(size_t startIndex, std::vector<IssueEntryPtr> &curIssueEntries,
        std::vector<IssueEntryPtr> consumersGroup);
    void RecoverSymbol(size_t startIndex, std::vector<IssueEntryPtr> curIssueEntries);
    void GetConsumerGroup(std::vector<IssueEntryPtr> consumers, std::vector<IssueEntryPtr> &consumersGroup);
    void GetStackTop(size_t &startIndex, std::vector<IssueEntryPtr> &curIssueEntries, std::map<MemoryType, int64_t> &curMemoryMap);
    Status BacktraceOnMemoryExceeded(size_t &startIndex,
        std::vector<IssueEntryPtr> &curIssueEntries, std::map<MemoryType, int64_t> &curMemoryMap);
    bool IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size);
    Status ModifyBuffer(std::map<MemoryType, int64_t> &curMemoryMap, MemoryType memType, int64_t size, bool isAdd);
    Status RetireIssueBuffer(std::map<MemoryType, int64_t> &curMemoryMap, IssueEntryPtr issue);
    void issueMemoryUpdate(IssueEntryPtr issue, size_t startIndex, std::vector<IssueEntryPtr> curIssueEntries,
        std::map<MemoryType, int64_t> curMemoryMap);
    Status AllocExecute(IssueEntryPtr issue, std::vector<IssueEntryPtr> &curIssueEntries,
        std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex, bool &isContinue);
    Status IssueEntriesExecute(std::vector<IssueEntryPtr> &curIssueEntries,
        std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex);
    Status ExecuteIssue();

    // gen spill
    Status GenSpillOp(LocalBufferPtr allocBuffer, size_t &pcIdx);
    Status GenBufferSpill(IssueEntryPtr allocIssue);
    Status SelectSpillBuffers(LocalBufferPtr allocBuffer, IssueEntryPtr issue, 
        std::vector<int> &spillGroup, bool isGenSpill);
    Status GetGroupNextUseOrder(std::vector<int> group, IssueEntryPtr allocIssue, 
        std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill);
    IssueEntryPtr GetSpillIssue(IssueEntryPtr allocIssue, int memId, bool isGenSpill);
    bool IsBelongSpillBlackList(IssueEntryPtr spillIssue, IssueEntryPtr issue);
    void FindFilterLtags(IssueEntryPtr allocIssue, std::set<IssueEntryPtr> &filterLtags);
    Status SpillMultiBuffer(IssueEntryPtr allocIssue, std::vector<int> spillGroup, size_t &pcIdx, 
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status GetSpillInfo(IssueEntryPtr allocIssue, int spillMemId, bool isGenSpill, SpillInfo &spillInfo);
    Status GetSpillTensor(IssueEntryPtr spillIssue, int spillMemId, LogicalTensorPtr &spillTensor);
    Status SpillBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, size_t &pcIdx, 
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status SpillOutBuffer(SpillInfo &spillInfo, IssueEntryPtr issue, size_t &pcIdx, bool isGenSpill);
    Status CreateSpillCopyout(IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor, int spillMemId,
        IssueEntryPtr &spillCopyout);    
    Status SpillInBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, MemoryType bufferType, bool isGenSpill);
    Status CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor, LogicalTensorPtr spillTensor,
        IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues);
    Status UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue,
        int spillMemId, IssueEntryPtr allocIssue);
    Status UpdateReloadIssueDepend(IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue, int spillMemId);
    Status UpdateRemainOpBufId(int oldMemId, int newMemId);
    void ReplaceTensorMemId(IssueEntryPtr &issue, int oldMemId, int newMemId);
    void UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset,
        IssueEntryPtr spillIssue);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    int GetBufNextUseOrder(IssueEntryPtr issue, int curMemId);
    int GetBufLastUseOrder(IssueEntryPtr issue, int curMemId);
    IssueEntryPtr GetBufLastWriteIssue(IssueEntryPtr issue, int curMemId);
    OoOSchedulerCheck::SpillInfo RecordSpillInfo(MemoryType bufferType, int memId, LocalBufferPtr allocIssue, 
        LogicalTensorPtr spillOutTensor, bool needCopyOut);
    bool HasEnoughBuffer(IssueEntryPtr allocIssue, MemoryType memType);
    bool CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType);
    int GetMemidAllocPriority(int memId);
    Status SpillAssembleBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status SpillParticalBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, IssueEntryPtr assemble, 
        LogicalTensorPtr assembleTensor, bool &isFirst);
    Status FindAssembleWithSpillTensor(SpillInfo &spillInfo, std::vector<IssueEntryPtr> &assembleList);
    Status UpdateAssembleBuffer(SpillInfo &spillInfo, LocalBufferPtr allocBuffer, LogicalTensorPtr assembleTensor);
    LogicalTensorPtr CreateAssemblePartTensor(LogicalTensorPtr iOperand, LogicalTensorPtr assembleTensor,
        SpillInfo &spillInfo, std::shared_ptr<AssembleOpAttribute> assembleAttr);
    int64_t CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset);

    // buffer rearrange
    Status RearrangeBuffers(IssueEntryPtr issue, bool isGenSpillStage, bool &rearrangeUBBF16);
    Status GenRearrangeCopyOp(MemoryType memType, int memId, int &newMemId, bool &rearrangeUBBF16);
    Status UpdateMemId(int oldMemId, int newMemId);
    void UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp);
    IssueEntryPtr ProcessMoveOp(Operation &moveOp,  Operation &occupyOp, int oldMemId, int newMemId);
    Status UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool &bufferManager);
    Status FindMoveFromTensor(Operation &occupyOp, int oldMemId, MemoryType memType, bool &rearrangeUBBF16, LogicalTensorPtr &moveFromTensor);
    Status GetMoveOpInTensor(Opcode moveOpcode, Operation &occupyOp, LogicalTensorPtr &inTensor, LogicalTensorPtr &moveFromTensor);

public:
    Status Schedule(const std::vector<Operation *> &operations);
    OoOScheduler(Function &function, bool combineAxis=false) : function_(function), isCombineAxis_(combineAxis) {}

    std::vector<Operation *> GetNewOperations() { return newOperations_; }
    int workspaceOffset{0};
    int clock{0};
    OoOSchedulerCheck oooCheck;
    bool isCombineAxis_{false};
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H