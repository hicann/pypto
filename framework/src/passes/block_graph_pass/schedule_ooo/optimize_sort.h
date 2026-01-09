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

    bool opFinish{false};
    std::map<Operation*, std::map<MemoryType, int64_t>> recordBufferAllocate;
    std::map<Operation*, std::pair<size_t, std::vector<Operation*>>> recordOpList;
    std::map<Operation*, MemoryType> recordOpBuffer;
    std::stack<std::pair<Operation*, MemoryType>> needFreeOpStack;
    std::map<Operation*, bool> visitedOp;
    std::map<Operation*, std::unordered_map<int, int>> recordBufRefCount;

    // 回溯点位置,当前执行op的全部信息,用于后期回退
    Operation* backTraceOp{nullptr};
    std::map<Operation*, std::map<MemoryType, int64_t>> backTraceBufferAllocate;
    std::map<Operation*, std::pair<size_t, std::vector<Operation*>>> backTraceOpList;
    std::map<Operation*, std::unordered_map<int, int>> backTraceBufRefCount;
    // 回退点,防止死循环
    Operation* rollBackNodeOp{nullptr};

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

    void ReorderOp(std::vector<size_t> &preIdx, std::vector<Operation*> &curOpList, size_t startIndex);
    void FindIndex(Operation* op, std::vector<Operation*> curOpList, size_t &index);
    Status FindConsumerList(size_t consumerIndex, std::vector<size_t> &preOpList, std::vector<Operation*> &curOpList);
    Status UpdateOOperandPreDependence(size_t startIndex, std::vector<Operation*> &curOpList,
        std::vector<Operation*> consumersGroup);
    void RecoverSymbol(size_t startIndex, std::vector<Operation*> curOpList);
    void GetConsumerGroup(std::set<Operation*> consumers, std::vector<Operation*> &consumersGroup);
    void GetStackTop(size_t &startIndex, std::vector<Operation*> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap);
    Status BacktraceOnMemoryExceeded(size_t &startIndex, std::vector<Operation*> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap);
    bool IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size);
    Status ModifyBuffer(std::map<MemoryType, int64_t> &curMemoryMap, MemoryType memType, int64_t size, bool isAdd);
    Status RetireOpBuffer(std::map<MemoryType, int64_t> &curMemoryMap, Operation* op);
    void OpMemoryUpdate(Operation* op, size_t startIndex, std::vector<Operation*> curOpList,
        std::map<MemoryType, int64_t> curMemoryMap);
    Status AllocExecute(Operation* op, std::vector<Operation*> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex, bool &isContinue);
    Status OpListExecute(std::vector<Operation*> &curOpList, std::map<MemoryType, int64_t> &curMemoryMap,
        size_t &startIndex);
    Status ExecuteOp();

    void ReplaceIndex(std::vector<Operation*> &curOpList, std::set<size_t> advanceIndexList, size_t rollBackIndex);
    bool HasDependency(Operation* rollBackOp, Operation* backOp);
    void GetPreNode(size_t i, std::vector<Operation*> curOpList, size_t rollBackIndex,
        size_t backTraceIndex, std::set<size_t> &dependencyIndexList);
    void GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
        std::vector<Operation*> curOpList, std::set<size_t> &advanceIndexList);
    Status RollBack(size_t &startIndex, std::vector<Operation*> &curOpList,
        std::map<MemoryType, int64_t> &curMemoryMap);


};
} // namespace npu::tile_fwk
#endif // PASS_OPTIMIZE_SORT_H