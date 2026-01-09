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
 * \file core_assign.cpp
 * \brief
 */


#include "passes/block_graph_pass/schedule_ooo/estimate_latency.h"


namespace npu::tile_fwk {

void LatencyEstimator::LaunchReadyIssue() {
    for (auto &op : taskList) {
        if (USE_LESS_OPS2.find(op->GetOpcode()) != USE_LESS_OPS2.end() && inGraph[op].empty()) {
            auto type = RescheduleUtils::GetOpPipeType(op);
            opQueues[type].Insert(op);
        }
        if (IsOpAlloc(op)) {
            auto tensor = op->GetOOperands()[0];
            auto memId = tensor->memoryrange.memId;
            allocIssueQueue[localBufferMap[memId]->memType].Insert(op);
        }
    }
}

Status LatencyEstimator::FreeBuffer(Operation* op) {
    for (auto tensor : GetInOutOperand(op)) {
        auto memId = tensor->memoryrange.memId;
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount[memId] == 0) {
            auto freeMemSize = localBufferMap[memId]->size;
            localMemoryCurrentSize[localBufferMap[memId]->memType] += freeMemSize;
            if (localMemoryCurrentSize[localBufferMap[memId]->memType] > localMemSize[localBufferMap[memId]->memType]
                || localMemoryCurrentSize[localBufferMap[memId]->memType] < 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Tensor, "Free tensor [%d] success.", memId);
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt) {
    commitCnt++;
    opRetiredInfo[op] = true;
    if (FreeBuffer(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBuffer failed. %s", GetOpInfo(op).c_str());
        return FAILED;
    }

    for (auto succ : outGraph[op]) {
        if (opRetiredInfo[succ]) {
            continue;
        }
        bool ready = true;
        for (auto pred : inGraph[succ]) {
            if (!opRetiredInfo[pred]) {
                ready = false;
                break;
            }
        }
        if (ready) {
            opQueues[RescheduleUtils::GetOpPipeType(succ)].Insert(succ);
            APASS_LOG_DEBUG_F(Elements::Operation, "Wakeup: %s", GetOpInfo(succ).c_str());
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::RetireIssueStage(uint64_t& commitCnt, int& nextCycle) {
    for (auto& [pipeType, pipe] : opQueues) {
        (void)pipeType;
        if (!pipe.busy) {
            continue;
        }
        if (pipe.curOpRetireCycle <= clock) {   // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            Operation* op = pipe.curOp;
            pipe.busy = false;
            pipe.curOp = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", GetOpInfo(op).c_str());
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed! %s", GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%ld]: %s", pipe.curOpRetireCycle, GetOpInfo(pipe.curOp).c_str());
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType, OpQueue &pipe) {
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        Operation* op = pipe.Front();
        auto needMemSize = GetInOutOperand(op)[0]->MemorySize();
        if (localMemoryCurrentSize[memType] >= static_cast<long int>(needMemSize)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", GetOpInfo(op).c_str());
            localMemoryCurrentSize[memType] -= needMemSize;
            if (localMemoryCurrentSize[memType] > localMemSize[memType] ||
                localMemoryCurrentSize[memType] < 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", GetInOutOperand(op)[0]);
                return FAILED;
            }
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed. %s", GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            canAlloc = false;
            APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot alloc Tensor[%d] ", GetInOutOperand(op)[0]);
            break;
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::BufferAllocStage(uint64_t &commitCnt) {
    for (auto& [memoryType, pipe] : allocIssueQueue) {
        if (pipe.Empty()) {
            continue;
        }
        // 不断按顺序执行alloc指令，直到buffer被占满为止。
        if (ExecuteAllocIssue(commitCnt, memoryType, pipe) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::LaunchIssueStage(int& nextCycle) {
    // issue from all pipes
    for (auto &[pipeType, pipe] : opQueues) {
        if (pipe.Empty() || pipe.busy) {
            continue;
        }
        Operation* op = pipe.PopFront();
        pipe.busy = true;
        pipe.curOp = op;
        pipe.curOpRetireCycle = clock + op->GetLatency();
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }

        APASS_LOG_DEBUG_F(Elements::Operation, "opQueues Insert: %s.", GetOpInfo(op).c_str());
    }
    return SUCCESS;
}

Status LatencyEstimator::SpillOnBlock() {
    MemoryType spillMemType;
    if (!allocIssueQueue[MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!allocIssueQueue[MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
        return FAILED;
    }

    Operation* op = allocIssueQueue[spillMemType].Front();
    size_t needMemSize = GetInOutOperand(op)[0]->MemorySize();
    localMemoryCurrentSize[spillMemType] += static_cast<long int>(needMemSize);
    if (localMemoryCurrentSize[spillMemType] < 0 || localMemoryCurrentSize[spillMemType] > localMemSize[spillMemType]){
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[%d] is valid. Please check", spillMemType);
        return FAILED;
    }
    return SUCCESS;
}

void LatencyEstimator::initLatencyEstimatorOpQueues(){
    for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
        opQueues[static_cast<PipeType>(i)] = OpQueue();
    }
}

void LatencyEstimator::InitMemWithoutAlloc() {
    std::set<int> needAllocMem;
    for (const auto &op : taskList) {
        for (auto &iOperand : op->GetIOperands()) {
            bool needAlloc = true;
            if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR){
                needAlloc = false;
                continue;
            }
            for (auto pre : iOperand->GetProducers()) {
                if (std::find(taskList.begin(), taskList.end(), pre) != taskList.end()) {
                    needAlloc = false;
                    break;
                }
            }
            if (needAlloc){
                auto memId = iOperand->memoryrange.memId;
                needAllocMem.insert(memId);
                APASS_LOG_DEBUG_F(Elements::Tensor, "Buffer[%d] is ALLOC, it has no producers", memId);
            }
        }
    }
    for (auto memId : needAllocMem){
        auto freeMemSize = localBufferMap[memId]->size;
        localMemoryCurrentSize[localBufferMap[memId]->memType] -= freeMemSize;
    }
}

Status LatencyEstimator::LatencyEstimatorMainLoop() {
    LaunchReadyIssue();
    initLatencyEstimatorOpQueues();
    InitMemWithoutAlloc();
    numTotalIssues = taskList.size();
    uint64_t commitCount = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycleTime = -1;
        // Retire Stage : 检查现有pipe中的op是否执行完。如果op执行完，则将op标记为retired状态，将可以被释放的buffer释放掉，并唤醒后续已经就绪的op。
        // 完毕后更新整个pipe的状态。
        if (RetireIssueStage(commitCount, nextCycleTime) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed.");
            return FAILED;
        }
        // Buffer Allocation Stage : 分配buffer。对于所有类型的buffer，按顺序执行alloc指令，并激活后续已经就绪的op。不断执行alloc直到buffer被占满为止。
        if (BufferAllocStage(commitCount) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "BufferAllocStage failed.");
            return FAILED;
        }
        // Launch Stage ：检查idle的pipe中是否有已经就绪的指令。如果有，则执行该指令，并更新pipe的状态为busy。
        if (LaunchIssueStage(nextCycleTime) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "LaunchIssueStage failed.");
            return FAILED;
        }
        if (numTotalIssues == commitCount && nextCycleTime == -1) {
            isAllRetired = true;
            break;
        }
        // 如果nextCycleTime为-1，说明每个pipe都处于idle的状态，判断出现阻塞。需要spill调整内存
        if (nextCycleTime == -1) {
            if (SpillOnBlock() != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed.");
                return FAILED;
            }
        } else {
            clock = nextCycleTime;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "\n Estimate Latency: %d", clock);
    return SUCCESS;
}
} // namespace npu::tile_fwk
