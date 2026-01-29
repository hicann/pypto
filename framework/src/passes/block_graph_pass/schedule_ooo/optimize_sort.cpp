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
 * \file optimize_sort.cpp
 * \brief
 */

#include "optimize_sort.h"
#include "passes/pass_log/pass_log.h"


namespace npu::tile_fwk {
void OptimizeSort::UpdatePreNodeQueue(std::unordered_set<Operation*> &curr,
    std::unordered_set<Operation*> &preNodeTotal, std::map<Operation*, bool>& visited) {
    std::unordered_set<Operation*> next;
    for (auto& curOp : curr) {
        for (auto& preOp : inGraph[curOp]) {
            if (!visited[preOp] && preNodeTotal.find(preOp) == preNodeTotal.end()) {
                next.insert(preOp);
            }
        }
    }
    for (auto& nextOp : next) {
        preNodeTotal.insert(nextOp);
    }
    curr.swap(next);
}

int OptimizeSort::GetNumUnvisitPreNode(Operation* op, std::map<Operation*, bool>& visited) {
    std::unordered_set<Operation*> preNodeTotal;
    std::unordered_set<Operation*> curr;
    for (auto& preOp : inGraph[op]) {
        if (!visited[preOp]) {
            curr.insert(preOp);
            preNodeTotal.insert(preOp);
        }
    }
    while (!curr.empty()) {
        UpdatePreNodeQueue(curr, preNodeTotal, visited);
    }
    return preNodeTotal.size();
}

Operation* OptimizeSort::FindNodeMinNumUnvisitedPreNode(
    std::map<Operation*, bool> visited, std::vector<Operation*> outNodeQueue) {
    Operation* res = nullptr;
    int minUnvisitedNode = INT_MAX;
    for (auto& outNode : outNodeQueue) {
        if (visited[outNode]) {
            continue;
        }
        int curUnvisitedNode = GetNumUnvisitPreNode(outNode, visited);
        if (curUnvisitedNode < minUnvisitedNode) {
            res = outNode;
            minUnvisitedNode = curUnvisitedNode;
        }
    }
    return res;
}

int OptimizeSort::GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, Operation* op) {
    int prior = 10;
    if (preNodePriority.find(op->GetOpcode()) != preNodePriority.end()) {
        prior = preNodePriority[op->GetOpcode()];
    }
    return prior;
}

int OptimizeSort::GetMaxDepthSimple(Operation* op) {
    auto it = depthCache_.find(op);
    if (it != depthCache_.end()) {
        return it->second;
    }

    int maxDepth = 0;
    for (const auto& pre : inGraph[op]) {
        maxDepth = std::max(maxDepth, GetMaxDepthSimple(pre));
    }

    int depth = maxDepth + 1;
    depthCache_[op] = depth;
    return depth;
}

void OptimizeSort::QueueNotReadyPreNode(Operation* curOp, std::map<Operation*, bool>& visited,
    std::unordered_map<Opcode, int> preNodePriority, std::deque<Operation*> &queue) {
    std::vector<Operation*> notReadyPreNode;
    for (auto& preOp : inGraph[curOp]) {
        if (!visited[preOp]) {
            notReadyPreNode.push_back(preOp);
        }
    }
    std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](Operation* a, Operation* b) {
        int priorA = GetNodePriority(preNodePriority, a);
        int priorB = GetNodePriority(preNodePriority, b);
        if (priorA != priorB) {
            return priorA < priorB;
        } else {
            int depA = GetMaxDepthSimple(a);
            int depB = GetMaxDepthSimple(b);
            if (depA == depB) {
                int aIdx = std::find(operations.begin(), operations.end(), a) - operations.begin();
                int bIdx = std::find(operations.begin(), operations.end(), b) - operations.begin();
                return aIdx < bIdx;
            }
            return depA < depB;
        }
    });
    for (auto& preOp : notReadyPreNode) {
        queue.push_front(preOp);
    }
}

void OptimizeSort::ForwardDfs(Operation* curOp, std::vector<Operation*>& newOpList,
    std::map<Operation*, bool>& visited, std::unordered_map<Opcode, int> preNodePriority,
    std::deque<Operation*> &queue) {
    bool ready = true;
    for (auto& preOp : inGraph[curOp]) {
        if (!visited[preOp]) {
            ready = false;
            break;
        }
    }

    if (ready) {
        visited[curOp] = true;
        queue.pop_front();
        newOpList.push_back(curOp);
    } else {
        QueueNotReadyPreNode(curOp, visited, preNodePriority, queue);
    }
}

void OptimizeSort::DFSFromSingleNode(Operation* op, std::map<Operation*, bool>& visited,
    std::vector<Operation*>& newOpList, std::unordered_map<Opcode, int> preNodePriority) {
    if (visited[op]) {
        return;
    }

    std::deque<Operation*> queue = {op};
    while (!queue.empty()) {
        auto curOp = queue.front();
        if (visited[curOp]) {
            queue.pop_front();
            continue;
        }

        ForwardDfs(curOp, newOpList, visited, preNodePriority, queue);
    }
}

Status OptimizeSort::DFSFromOutNode(std::vector<Operation*> outNodeQueue,
    std::unordered_map<Opcode, int> preNodePriority, std::map<Operation*, bool> &visited) {
    std::vector<Operation*> newOpList;
    if (outNodeQueue.size() != 0) {
       DFSFromSingleNode(outNodeQueue[0], visited, newOpList, preNodePriority);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Subgraph must have operation with outdegree 0.");
        return FAILED;
    }

    for (size_t i = 1; i < outNodeQueue.size(); i++) {
        while (!visited[outNodeQueue[i]]) {
            auto node = FindNodeMinNumUnvisitedPreNode(visited, outNodeQueue);
            if (node == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "FindNodeMinNumUnvisitedPreNode failed.");
                return FAILED;
            }
            DFSFromSingleNode(node, visited, newOpList, preNodePriority);
        }
    }
    operations = newOpList;
    return SUCCESS;
}

Status OptimizeSort::PriorDFS(std::unordered_map<Opcode, int> preNodePriority) {
    std::map<Operation*, bool> visited;
    std::vector<Operation*> outNodeQueue;
    for (size_t i = 0; i < operations.size(); i++) {
        visited[operations[i]] = false;
        if (outGraph[operations[i]].empty()) {
            outNodeQueue.emplace_back(operations[i]);
        }
    }

    if (DFSFromOutNode(outNodeQueue, preNodePriority, visited) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "DFSFromOutNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

// rollBackOp 和 backOp 是否存在前后序依赖
bool OptimizeSort::HasDependency(Operation* rollBackOp, Operation* backOp) {
    std::map<Operation*, bool> visited;
    for (auto op : operations) {
        visited[op] = false;
    }
    std::function<bool(Operation*)> dfs = [&](Operation* op) ->bool{
        if (op == backOp) return true;
        if (visited[op]) return false;

        visited[op] = true;
        for (auto succOp : outGraph[op]) {
            if (dfs(succOp)) {
                return true;
            }
        }
        return false;
    };
    return dfs(rollBackOp);
}

// 在 curOpList 中将 advanceIndexList 中的序列提前到 rollBackIndex 之前,更新 curOpList
std::shared_ptr<std::vector<Operation*>> OptimizeSort::ReplaceIndex(std::shared_ptr<std::vector<Operation*>> curOpList,
    std::set<size_t> &advanceIndexList, size_t rollBackIndex) {
    std::vector<Operation*> moveOpList;
    for (auto i : advanceIndexList) {
        APASS_LOG_DEBUG_F(Elements::Operation, "advance index: %d, op: %s",
                i, GetOpInfo((*curOpList)[i]).c_str());
        moveOpList.push_back((*curOpList)[i]);
    }
    auto copyCurOpList = std::make_shared<std::vector<Operation* >>(*curOpList);
    for (auto it = advanceIndexList.rbegin(); it != advanceIndexList.rend(); ++it) {
        copyCurOpList->erase(copyCurOpList->begin() + (*it));
    }
    copyCurOpList->insert(copyCurOpList->begin() + rollBackIndex, moveOpList.begin(), moveOpList.end());
    return copyCurOpList;
}

void OptimizeSort::GetPreNode(size_t i, std::shared_ptr<std::vector<Operation*>> curOpList, size_t rollBackIndex,
    size_t backTraceIndex, std::set<size_t> &dependencyIndexList) {
    dependencyIndexList.insert(i);
    APASS_LOG_DEBUG_F(Elements::Operation, "dependencyIndexList push index: %d, Op: %s",
        i, GetOpInfo((*curOpList)[i]).c_str());
    for (auto preOp : inGraph[(*curOpList)[i]]) {
        auto it = std::find(curOpList->begin() + rollBackIndex + 1, curOpList->begin() + backTraceIndex, preOp);
        if (it != curOpList->begin() + backTraceIndex) {
            auto index = std::distance(curOpList->begin(), it);
            GetPreNode(index, curOpList, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
}

// 记录 curOpList 中从 rollBackIndex 到 backTraceIndex 中所有和 rollBack 没有后继依赖的点
void OptimizeSort::GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
    std::shared_ptr<std::vector<Operation*>> curOpList, std::set<size_t> &advanceIndexList) {
    std::set<size_t> dependencyIndexList;
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (HasDependency((*curOpList)[rollBackIndex], (*curOpList)[i])) {
            GetPreNode(i, curOpList, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (dependencyIndexList.count(i) == 0) {
            advanceIndexList.insert(i);
            APASS_LOG_DEBUG_F(Elements::Operation, "advanceIndexList push index: %d, op: %s",
                i, GetOpInfo((*curOpList)[i]).c_str());
        }
    }
}

// rollBackIndex 位置回退
Status OptimizeSort::RollBack(size_t &startIndex,
    std::shared_ptr<std::vector<Operation*>> &curOpList, std::map<MemoryType, int64_t> &curMemoryMap) {
    APASS_LOG_DEBUG_F(Elements::Operation, "=====> Start RollBack.");
    curOpList = backTraceOpList_[backTraceOp_].second;
    MemoryType memType = recordOpBuffer_[backTraceOp_];
    size_t backTraceIndex = backTraceOpList_[backTraceOp_].first + 1;
    backTraceOp_ = (*curOpList)[backTraceIndex];
    size_t rollBackIndex = backTraceIndex;
    APASS_LOG_DEBUG_F(Elements::Operation, "backTraceOp_: %s, backTraceIndex: %d, memType: %d",
        GetOpInfo(backTraceOp_).c_str(), backTraceIndex, memType);
    std::set<size_t> advanceIndexList;
    while (rollBackIndex < curOpList->size() && rollBackIndex > 0) {
        rollBackIndex--;
        Operation* rollBackOp = (*curOpList)[rollBackIndex];
        if (recordOpBuffer_[rollBackOp] != memType || !(IsOpAlloc(rollBackOp)) || HasDependency(rollBackOp, backTraceOp_)) {
            continue;
        }
        rollBackNodeOp_ = rollBackOp;
        APASS_LOG_DEBUG_F(Elements::Operation, "Select rollBackOp: %s, rollBackIndex: %d",
            GetOpInfo(rollBackOp).c_str(), rollBackIndex);
        recordBufferAllocate_ = backTraceBufferAllocate_;
        recordOpList_ = backTraceOpList_;
        recordBufRefCount_ = backTraceBufRefCount_;
        advanceIndexList.clear();
        GetListToAdvance(rollBackIndex, backTraceIndex, curOpList, advanceIndexList);
        curOpList = ReplaceIndex(curOpList, advanceIndexList, rollBackIndex);
        startIndex = rollBackIndex;
        APASS_LOG_DEBUG_F(Elements::Operation, "RollBack==>change startIndex: %d", startIndex);
        if (rollBackIndex != 0) {
            curMemoryMap = recordBufferAllocate_[(*curOpList)[rollBackIndex-1]];
            RecoverSymbol(startIndex - 1, curOpList);
            return SUCCESS;
        }
        curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0}, {MemoryType::MEM_L0C, 0}};
        operations = *curOpList;
        for (auto op : (*curOpList)) {
            visitedOp_[op] = false;
        }
        InitBufRefCount();
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "RollBack Failed");
    return FAILED;
}

// 在 curOpList 中将 preOpList 中的序列提前到 startIndex 之后，更新 curOpList
std::shared_ptr<std::vector<Operation*>> OptimizeSort::ReorderOp(std::vector<size_t> &preIdx, std::shared_ptr<std::vector<Operation*>> curOpList,
    size_t startIndex) {
    // 对 perOpList 排序，再进行插入
    std::sort(preIdx.begin(), preIdx.end());
    std::vector<Operation*> moveOpList;
    APASS_LOG_DEBUG_F(Elements::Operation, "current index: %d, preIdx size: %d", startIndex, preIdx.size());
    for (auto i : preIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "preidx : %d, curOp: %s", i, GetOpInfo((*curOpList)[i]).c_str());
        moveOpList.push_back((*curOpList)[i]);
    }
    auto copyCurOpList = std::make_shared<std::vector<Operation*>>(*curOpList);
    for (auto it = preIdx.rbegin(); it != preIdx.rend(); ++it) {
        copyCurOpList->erase(copyCurOpList->begin() + (*it));
    }
    copyCurOpList->insert(copyCurOpList->begin() + startIndex + 1, moveOpList.begin(), moveOpList.end());
    return copyCurOpList;
}

void OptimizeSort::FindIndex(Operation* op, std::shared_ptr<std::vector<Operation*>> curOpList, size_t &index) {
    for (size_t i = 0; i < curOpList->size(); i++) {
        if ((*curOpList)[i] == op) {
            index = i;
            return;
        }
    }
}

// 在curOpList中，向前遍历找到consumerIndex的前序未被访问的节点，并放入preOpList中
Status OptimizeSort::FindConsumerList(size_t consumerIndex, std::vector<size_t> &preOpList, std::shared_ptr<std::vector<Operation*>> curOpList) {
    if ((*curOpList)[consumerIndex] == backTraceOp_) {
        APASS_LOG_WARN_F(Elements::Operation, "backTraceOp_ is one of the predecessor node.");
        return FAILED;
    }
    if ((*curOpList)[consumerIndex] == rollBackNodeOp_) {
        APASS_LOG_WARN_F(Elements::Operation, "rollBackNodeOp_ is one of the predecessor node.");
        return FAILED;
    }
    visitedOp_[(*curOpList)[consumerIndex]] = true;
    preOpList.push_back(consumerIndex);
    APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer idx: %d, op: %s", consumerIndex, GetOpInfo((*curOpList)[consumerIndex]).c_str());
    for (auto op : inGraph[(*curOpList)[consumerIndex]]) {
        if (visitedOp_[op] == false) {
            size_t index;
            FindIndex(op, curOpList, index);
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer preIdx: %d, op: %s", index, GetOpInfo(op).c_str());
            if (FindConsumerList(index, preOpList, curOpList) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// 将 consumersGroup 和其前序依赖按原有顺序放入 preOpList
Status OptimizeSort::UpdateOOperandPreDependence(size_t startIndex, std::shared_ptr<std::vector<Operation*>> &curOpList,
    std::vector<Operation*> consumersGroup) {
    // curOpList 中向后找
    std::vector<size_t> preOpList;
    size_t index = startIndex;
    while (index < curOpList->size()) {
        if (std::find(consumersGroup.begin(), consumersGroup.end(), (*curOpList)[index]) != consumersGroup.end()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer Idx: %d", index);
            if (FindConsumerList(index, preOpList, curOpList) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
        index++;
    }
    curOpList = ReorderOp(preOpList, curOpList, startIndex);
    return SUCCESS;
}

// 回溯后，将队列后面 op 的 visitedOp_ 状态还原回 false，并对应修改 refcount
void OptimizeSort::RecoverSymbol(size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList) {
    APASS_LOG_DEBUG_F(Elements::Operation, "RecoverSymbol  startIdx: %d, curOp: %s", startIndex, GetOpInfo((*curOpList)[startIndex]).c_str());
    bufRefCount = recordBufRefCount_[(*curOpList)[startIndex]];
    for (size_t i = 0; i < curOpList->size(); i++) {
        if (i > startIndex) {
            visitedOp_[(*curOpList)[i]] = false;
            continue;
        }
        visitedOp_[(*curOpList)[i]] = true;
    }
}

// 找未被执行的 consumer
void OptimizeSort::GetConsumerGroup(std::unordered_set<Operation*> &consumers, std::vector<Operation*> &consumersGroup) {
    for (auto op : consumers) {
        APASS_LOG_DEBUG_F(Elements::Operation, "consumer: %s", GetOpInfo(op).c_str());
        if (!visitedOp_[op]) {
            consumersGroup.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer: %s", GetOpInfo(op).c_str());
        }
    }
}

void OptimizeSort::GetStackTop(size_t &startIndex, std::shared_ptr<std::vector<Operation*>> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap) {
    auto topNode = needFreeOpStack_.top();
    needFreeOpStack_.pop();
    curOpList = recordOpList_[topNode.first].second;
    startIndex = recordOpList_[topNode.first].first;
    curMemoryMap = recordBufferAllocate_[topNode.first];
}

Status OptimizeSort::BacktraceOnMemoryExceeded(size_t &startIndex,
    std::shared_ptr<std::vector<Operation*>> &curOpList, std::map<MemoryType, int64_t> &curMemoryMap) {
    APASS_LOG_DEBUG_F(Elements::Tensor, "=====> Start Backtrace.");
    MemoryType memType = (*curOpList)[startIndex]->GetOutputOperand(0)->GetMemoryTypeOriginal();
    std::vector<Operation*> consumersGroup;
    while (startIndex < curOpList->size() && startIndex > 0) {
        startIndex--;
        auto op = (*curOpList)[startIndex];
        if (!needFreeOpStack_.empty() && needFreeOpStack_.top().first == (*curOpList)[startIndex]) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Having traversed %s, the stack needs to be popped", GetOpInfo((*curOpList)[startIndex]).c_str());
            break;
        }
        if (recordOpBuffer_[op] != memType || IsOpAlloc(op)) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "===>start to find unvisited consumer");
        APASS_LOG_DEBUG_F(Elements::Operation, "current index： %d", startIndex);
        consumersGroup.clear();
        GetConsumerGroup(outGraph[op], consumersGroup);
        if (consumersGroup.empty()) {
            continue;
        }
        RecoverSymbol(startIndex, curOpList);
        GetConsumerGroup(outGraph[op], consumersGroup);
        APASS_LOG_DEBUG_F(Elements::Operation, "push %s to stack", GetOpInfo(op).c_str());
        curMemoryMap = recordBufferAllocate_[op];
        needFreeOpStack_.push(make_pair(op, recordOpBuffer_[op]));
        if (UpdateOOperandPreDependence(startIndex, curOpList, consumersGroup) != SUCCESS) {
            needFreeOpStack_.pop();
            APASS_LOG_DEBUG_F(Elements::Operation, "UpdateOOperandPreDependence failed.");
            continue;
        }
        startIndex++;
        APASS_LOG_DEBUG_F(Elements::Operation, "Backtrace==>change startIndex: %d", startIndex);
        return SUCCESS;
    }
    if (needFreeOpStack_.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Stack is empty. Start to rollback.");
        return FAILED;
    }
    GetStackTop(startIndex, curOpList, curMemoryMap);
    RecoverSymbol(startIndex, curOpList);
    APASS_LOG_DEBUG_F(Elements::Operation, "pop %s from stack", GetOpInfo((*curOpList)[startIndex]).c_str());
    if (BacktraceOnMemoryExceeded(startIndex, curOpList, curMemoryMap) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "BacktraceOnMemoryExceeded Failed");
        return FAILED;
    }
    return SUCCESS;
}

// 计算 tensor 对应的 memType （只对 L0C L0A L0B 进行内存处理） 是否已满
bool OptimizeSort::IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size) {
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return false;
    }
    if (curMemoryMap[memType] + size > localMemSize[memType]) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The %d-memType memory is full, current memory: %d, memory to add: %d",
            memType, curMemoryMap[memType], size);
        return true;
    }
    return false;
}

// 修改内存
Status OptimizeSort::ModifyBuffer(std::map<MemoryType, int64_t> &curMemoryMap, MemoryType memType, int64_t size, bool isAdd) {
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return SUCCESS;
    }
    if (isAdd) {
        if (curMemoryMap[memType] + size > localMemSize[memType]) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to increase memory");
            return FAILED;
        }
        curMemoryMap[memType] = curMemoryMap[memType] + size;
        APASS_LOG_DEBUG_F(Elements::Operation, "Increase %d-memType memory, size: %d, total memory %d", memType, size, curMemoryMap[memType]);
        return SUCCESS;
    }
    if (curMemoryMap[memType] - size < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to reduce memory");
        return FAILED;
    }
    curMemoryMap[memType] = curMemoryMap[memType] - size;
    APASS_LOG_DEBUG_F(Elements::Operation, "Reduce %d-memType memory, size: %d, total memory %d", memType, size, curMemoryMap[memType]);
    return SUCCESS;
}

// 释放内存 notTaskOp需要减去bufRefCount
Status OptimizeSort::RetireOpBuffer(std::map<MemoryType, int64_t> &curMemoryMap, Operation* op) {
    for (auto tensor : GetInOutOperand(op)) {
        auto memId = tensor->memoryrange.memId;
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount[memId] == 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Start to free memory:");
            if (ModifyBuffer(curMemoryMap, tensor->GetMemoryTypeOriginal(), ShapeCeilAlign(tensor->tensor->rawshape, tensor->Datatype()), false) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OptimizeSort::OpMemoryUpdate(Operation* op, size_t startIndex, std::shared_ptr<std::vector<Operation*>> curOpList,
    std::map<MemoryType, int64_t> curMemoryMap) {
    recordOpList_[op] = make_pair(startIndex, curOpList);
    recordBufferAllocate_[op] = curMemoryMap;
    recordOpBuffer_[op] = op->GetOutputOperand(0)->GetMemoryTypeOriginal();
    recordBufRefCount_[op] = bufRefCount;
}

Status OptimizeSort::AllocExecute(Operation* op, std::shared_ptr<std::vector<Operation*>> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex, bool &isContinue) {
    APASS_LOG_DEBUG_F(Elements::Operation, "alloc op: %s", GetOpInfo(op).c_str());
    auto tensor = op->GetOutputOperand(0);
    if (IsBufferFull(curMemoryMap, tensor->GetMemoryTypeOriginal(), ShapeCeilAlign(tensor->GetShape(), tensor->Datatype()))) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The memory of %s needs to be released", std::to_string(tensor->GetMemoryTypeOriginal()).c_str());
        backTraceOp_ = (*curOpList)[startIndex];
        backTraceBufferAllocate_ = recordBufferAllocate_;
        backTraceOpList_ = recordOpList_;
        backTraceBufRefCount_ = recordBufRefCount_;
        APASS_LOG_DEBUG_F(Elements::Operation, "backTraceOp_: %s, backTraceIndex: %d, memType: %d",
            GetOpInfo(backTraceOp_).c_str(), backTraceOpList_[backTraceOp_].first, recordOpBuffer_[backTraceOp_]);
        APASS_LOG_DEBUG_F(Elements::Operation, "=====> Need backtrace.");
        if (BacktraceOnMemoryExceeded(startIndex, curOpList, curMemoryMap) != SUCCESS) {
            if (RollBack(startIndex, curOpList, curMemoryMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocExecute failed.");
                return FAILED;
            }
            isContinue = true;
            return SUCCESS;
        }
        isContinue = true;
        return SUCCESS;
    }
    return SUCCESS;
}

Status OptimizeSort::OpListExecute(std::shared_ptr<std::vector<Operation*>> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex) {
    APASS_LOG_DEBUG_F(Elements::Operation, "===>Start opListExecute, startIndex: %d", startIndex);
    if (curOpList->empty()) {
        curOpList = std::make_shared<std::vector<Operation*>>(operations);
    }
    while (startIndex < curOpList->size()) {
        auto op = (*curOpList)[startIndex];
        OpMemoryUpdate(op, startIndex, curOpList, curMemoryMap);
        APASS_LOG_DEBUG_F(Elements::Operation, "execute op: %s, index: %d", GetOpInfo(op).c_str(), startIndex);
        if (IsOpAlloc(op)) {
            bool isContinue = false;
            if (AllocExecute(op, curOpList, curMemoryMap,  startIndex, isContinue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "AllocExecute failed.");
                return FAILED;
            }
            if (isContinue) {
                return SUCCESS;
            }
            auto tensor = op->GetOutputOperand(0);
            if (ModifyBuffer(curMemoryMap, tensor->GetMemoryTypeOriginal(), ShapeCeilAlign(tensor->tensor->rawshape, tensor->Datatype()), true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%u] failed.", tensor->GetMagic());
                return FAILED;
            }
            OpMemoryUpdate(op, startIndex, curOpList, curMemoryMap);
        }
        visitedOp_[op] = true;
        if (RetireOpBuffer(curMemoryMap, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireOp failed! %s", GetOpInfo(op).c_str());
            return FAILED;
        }
        OpMemoryUpdate(op, startIndex, curOpList, curMemoryMap);
        startIndex += 1;
    }
    opFinish_ = true;
    return SUCCESS;
}

Status OptimizeSort::ExecuteOp() {
    auto curOpList = std::make_shared<std::vector<Operation*>>();
    std::map<MemoryType, int64_t> curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}};
    size_t startIndex{0};
    for (auto &op : operations) {
        visitedOp_[op] = false;
    }
    while(!opFinish_) {
        if (OpListExecute(curOpList, curMemoryMap, startIndex) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "OpListExecute failed.");
            return FAILED;
        }
    }
    operations = *curOpList;
    return SUCCESS;
}

void OptimizeSort::AllocAhead() {
    std::vector<Operation*> allocOps;
    std::vector<Operation*> normalOps;
    for (auto& op : operations) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocOps.emplace_back(op);
            continue;
        }
        normalOps.emplace_back(op);
    }
    std::vector<Operation *> newOperations;
    std::reverse(allocOps.begin(), allocOps.end());
    newOperations.swap(allocOps);
    newOperations.insert(newOperations.end(), normalOps.begin(), normalOps.end());
    operations = newOperations;
}

Status OptimizeSort::SortOps() {
    LOG_SCOPE_BEGIN(tSortOps, Elements::Function, "SortOps");
    Init(operations);
    if (CheckAllocOp(operations) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocOp failed!");
        return FAILED;
    }
    if (operations.empty()) {
        return SUCCESS;
    }
    AllocAhead();
    std::string sortMethodStr;
    std::string funcName = function_.GetMagicName();

    sortMethodStr = function_.paramConfigs_.OoOPreScheduleMethod;
    if (sortMethodStr == "PriorDFS") {
        std::unordered_map<Opcode, int> preNodePriority = {
            // ALLOC 节点优先级最高，因为一个节点的前序ALLOC节点要在最靠近该节点的地方访问。
            {Opcode::OP_UB_ALLOC, 0}, {Opcode::OP_L1_ALLOC, 0}, {Opcode::OP_L0A_ALLOC, 0}, {Opcode::OP_L0B_ALLOC, 0},
            {Opcode::OP_L0C_ALLOC, 0}, {Opcode::OP_BT_ALLOC, 0}, {Opcode::OP_FIX_ALLOC, 0},
            // 其次是L0级数据搬运Op。
            {Opcode::OP_L1_TO_L0A, 1}, {Opcode::OP_L1_TO_L0B, 1}, {Opcode::OP_L1_TO_L0_AT, 1},
            {Opcode::OP_L1_TO_L0_BT, 1}, {Opcode::OP_L1_TO_FIX, 1}, {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
            {Opcode::OP_L1_TO_FIX_RELU_PRE, 1}, {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
            {Opcode::OP_L1_TO_FIX_QUANT_POST, 1}, {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
            {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1}, {Opcode::OP_L1_TO_BT, 1},
            // 再其次是L1级数据搬运Op。
            {Opcode::OP_COPY_IN, 2}, {Opcode::OP_UB_COPY_IN, 2}, {Opcode::OP_L1_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2}, {Opcode::OP_L1_COPY_UB, 2},
            {Opcode::OP_L0C_COPY_UB, 2}, {Opcode::OP_UB_COPY_L1, 2},
            // 最后访问其它计算节点（其它节点默认的优先级为10）。
        };
        if (PriorDFS(preNodePriority) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PriorDFS failed.");
            return FAILED;
        }
        if (ExecuteOp() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteOp failed.");
            return FAILED;
        }
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "PreSchedule method not recognized.");
        return FAILED;
    }
    LOG_SCOPE_END(tSortOps);
    return SUCCESS;
}

} // namespace npu::tile_fwk