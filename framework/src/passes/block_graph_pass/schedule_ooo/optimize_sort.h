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
 * \file optimize_sort.h
 * \brief
 */

#ifndef PASS_OPTIMIZE_SORT_H
#define PASS_OPTIMIZE_SORT_H

#include "schedule_base.h"
#include <vector>

namespace npu::tile_fwk {
class OptimizeSort : public ScheduleBase {
public:
    OptimizeSort(std::vector<Operation*> opList, Function &function) :
        ScheduleBase(), operations(opList), function_(function) {}

    std::vector<Operation*> operations;
    Function &function_;

    bool opFinish_{false};
    std::map<Operation*, std::map<MemoryType, int64_t>> recordBufferAllocate_;
    std::map<Operation*, std::pair<size_t, std::shared_ptr<std::vector<Operation*>>>> recordOpList_;
    std::map<Operation*, MemoryType> recordOpBuffer_;
    std::stack<std::pair<Operation*, MemoryType>> needFreeOpStack_;
    std::map<Operation*, bool> visitedOp_;
    std::map<Operation*, std::unordered_map<int, int>> recordBufRefCount_;

    // 回溯点位置,当前执行op的全部信息,用于后期回退
    Operation* backTraceOp_{nullptr};
    std::map<Operation*, std::map<MemoryType, int64_t>> backTraceBufferAllocate_;
    std::map<Operation*, std::pair<size_t, std::shared_ptr<std::vector<Operation*>>>> backTraceOpList_;
    std::map<Operation*, std::unordered_map<int, int>> backTraceBufRefCount_;
    // 回退点,防止死循环
    Operation* rollBackNodeOp_{nullptr};
    std::unordered_map<Operation*, int> depthCache_;

    void opListInit();
    Status SortOps();
    Status PriorDFS(std::unordered_map<Opcode, int> preNodePriority);
    Status DFSFromOutNode(std::vector<Operation*> outNodeQueue, std::unordered_map<Opcode, int> preNodePriority,
        std::map<Operation*, bool> &visited);
    void DFSFromSingleNode(Operation* op, std::map<Operation*, bool>& visited,
        std::vector<Operation*>& newOpList, std::unordered_map<Opcode, int> preNodePriority);
    void ForwardDfs(Operation* curOp, std::vector<Operation*>& newOpList, std::map<Operation*, bool>& visited,
        std::unordered_map<Opcode, int> preNodePriority, std::deque<Operation*> &queue);
    void QueueNotReadyPreNode(Operation* curOp, std::map<Operation*, bool>& visited,
        std::unordered_map<Opcode, int> preNodePriority, std::deque<Operation*> &queue);
    int GetMaxDepthSimple(Operation* op);
    int GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, Operation* op);
    Operation* FindNodeMinNumUnvisitedPreNode(
        std::map<Operation*, bool> visited, std::vector<Operation*> outNodeQueue);
    int GetNumUnvisitPreNode(Operation* op, std::map<Operation*, bool>& visited);
    void UpdatePreNodeQueue(std::unordered_set<Operation*> &curr, std::unordered_set<Operation*> &preNodeTotal,
        std::map<Operation*, bool>& visited);

    std::shared_ptr<std::vector<Operation*>> ReorderOp(std::vector<size_t> &preIdx, std::shared_ptr<std::vector<Operation*>> curOpList, size_t startIndex);
    void FindIndex(Operation* op, std::shared_ptr<std::vector<Operation*>> curOpList, size_t &index);
    Status FindConsumerList(size_t consumerIndex, std::vector<size_t> &preOpList, std::shared_ptr<std::vector<Operation*>> curOpList);
    Status UpdateOOperandPreDependence(size_t startIndex, std::shared_ptr<std::vector<Operation*>> &curOpList,
        std::vector<Operation*> consumersGroup);
    void RecoverSymbol(size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList);
    void GetConsumerGroup(std::unordered_set<Operation*> &consumers, std::vector<Operation*> &consumersGroup);
    void GetStackTop(size_t &startIndex, std::shared_ptr<std::vector<Operation*>> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap);
    Status BacktraceOnMemoryExceeded(size_t &startIndex, std::shared_ptr<std::vector<Operation*>> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap);
    bool IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size);
    Status ModifyBuffer(std::map<MemoryType, int64_t> &curMemoryMap, MemoryType memType, int64_t size, bool isAdd);
    Status RetireOpBuffer(std::map<MemoryType, int64_t> &curMemoryMap, Operation* op);
    void OpMemoryUpdate(Operation* op, size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList,
        std::map<MemoryType, int64_t> curMemoryMap);
    Status AllocExecute(Operation* op, std::shared_ptr<std::vector<Operation*>> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex, bool &isContinue);
    Status OpListExecute(std::shared_ptr<std::vector<Operation*>> &curOpList, std::map<MemoryType, int64_t> &curMemoryMap,
        size_t &startIndex);
    Status ExecuteOp();
    void AllocAhead();

    std::shared_ptr<std::vector<Operation*>> ReplaceIndex(std::shared_ptr<std::vector<Operation*>> curOpList, std::set<size_t> &advanceIndexList, size_t rollBackIndex);
    bool HasDependency(Operation* rollBackOp, Operation* backOp);
    void GetPreNode(size_t i, std::shared_ptr<std::vector<Operation*>> curOpList, size_t rollBackIndex,
        size_t backTraceIndex, std::set<size_t> &dependencyIndexList);
    void GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
        std::shared_ptr<std::vector<Operation*>> curOpList, std::set<size_t> &advanceIndexList);
    Status RollBack(size_t &startIndex, std::shared_ptr<std::vector<Operation*>> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap);


};
} // namespace npu::tile_fwk
#endif // PASS_OPTIMIZE_SORT_H