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
 * \file ooo_scheduler.cpp
 * \brief
 */

#include "ooo_scheduler.h"
#include "buffer_rearrange.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

inline std::string coreTypeToString(CoreLocationType coreLocation)
{
    switch (coreLocation) {
        case CoreLocationType::AIV0:
            return "AIV0";
        case CoreLocationType::AIV1:
            return "AIV1";
        case CoreLocationType::AIC:
            return "AIC";
        default:
            return "MEM_UNKNOWN";
    }
}

inline bool IsMixGraph(const std::vector<Operation*>& operations)
{
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : operations) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
            hasAIV = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

Operation* OoOScheduler::SkipViewChain(Operation* start, bool followProducers)
{
    if (start == nullptr) return nullptr;
    Operation* op = start;
    Operation* lastView = nullptr;
    while (op != nullptr && IsViewOp(*op)) {
        lastView = op;
        if (followProducers) {
            const auto& nextOps = op->GetInputOperand(0)->GetProducers();
            if (nextOps.size() != 1) break;
            op = *nextOps.begin();
        } else {
            const auto& nextOps = op->GetOutputOperand(0)->GetConsumers();
            if (nextOps.size() != 1) break;
            op = *nextOps.begin();
        }
    }
    return lastView;
}

int OoOScheduler::GetOOperandIdx(Operation* op, int curMemId)
{
    for (size_t i = 0; i < op->GetOOperands().size(); i++) {
        if (op->GetOOperands()[i]->memoryrange.memId == curMemId) {
            return i;
        }
    }
    return -1;
}

Status OoOScheduler::PrintSpillFailedInfo(Operation* allocOp, bool isGenSpill)
{
    auto memType = localBufferMap_[GetOpMemIds(allocOp)[0]]->memType;
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s",
        MemoryTypeToString(memType).c_str(), GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
    APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", GetOpInfo(allocOp).c_str(),
        localBufferMap_[GetOpMemIds(allocOp)[0]]->size, GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
    auto coreLocation = opCoreLocationMap[allocOp];
    if (isGenSpill) {
        auto bufferSlices = bufferManagerMap[coreLocation][memType].GetBufferSlices();
        for (auto memId : bufferSlices) {
            auto occupyOp = GetBufLastWriteOp(allocOp, memId);
            if (occupyOp == nullptr) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write time.", memId);
                return FAILED;
            }
            APASS_LOG_ERROR_F(
                Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer:%s. %s", memId,
                localBufferMap_[memId]->size, localBufferMap_[memId]->start, localBufferMap_[memId]->end,
                GetOpInfo(occupyOp).c_str(), GetFormatBacktrace(*occupyOp).c_str());
        }
    } else {
        if (tensorOccupyMap.find(memType) != tensorOccupyMap.end()) {
            for (auto& occupy : tensorOccupyMap[memType]) {
                int memId = occupy.first;
                auto occupyOp = occupy.second;
                if (occupyOp == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write time.", memId);
                    return FAILED;
                }
                APASS_LOG_ERROR_F(
                    Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer:%s. %s", memId,
                    localBufferMap_[memId]->size, localBufferMap_[memId]->start, localBufferMap_[memId]->end,
                    GetOpInfo(occupyOp).c_str(), GetFormatBacktrace(*occupyOp).c_str());
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree)
{
    if (isFree) {
        int freeBufferSize = localBufferMap_[memId]->size;
        oooCheck.bufferTotalUsage[bufferType] +=
            oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] -= freeBufferSize;
        oooCheck.lastClock[bufferType] = clock;
    } else {
        oooCheck.bufferTotalUsage[bufferType] +=
            oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] += localBufferMap_[memId]->size;
        oooCheck.lastClock[bufferType] = clock;
        oooCheck.bufferMaxUsage[bufferType] =
            std::max(oooCheck.bufferMaxUsage[bufferType], oooCheck.bufferLastUsage[bufferType]);
    }
}

void OoOScheduler::PrintOpList(std::vector<Operation*> opList)
{
    APASS_LOG_INFO_F(Elements::Operation, "==================== OP_LIST =====================");
    bool needMark = false;
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        needMark = true;
    }
    for (auto& op : opList) {
        if (needMark) {
            bool isCubeComponent =
                op->HasAttribute(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube);
            if (!isCubeComponent) {
                op->SetAIVCore(AIVCore::AIV0);
            }
        }
        if (!op->oOperand.empty()) {
            APASS_LOG_INFO_F(
                Elements::Operation, "%s[%d], range[%zu, %zu]", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
                op->oOperand[0]->memoryrange.start, op->oOperand[0]->memoryrange.end);
        } else {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
}

void OoOScheduler::UpdateIssueExecOrder()
{
    for (size_t idx = 0; idx < orderedOps.size(); idx++) {
        opExecOrderMap[orderedOps[idx]] = idx;
    }
}

Status OoOScheduler::CheckAndUpdateLifecycle()
{
    for (const auto &op : orderedOps) {
        if (!opIsRetiredMap[op]) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", GetOpInfo(op).c_str(),
                GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (opIsAllocMap[op]) {
            op->GetOutputOperand(0)->memoryrange.lifeStart =
                localBufferMap_[GetOpMemIds(op)[0]]->startCycle;
            op->GetOutputOperand(0)->memoryrange.lifeEnd =
                localBufferMap_[GetOpMemIds(op)[0]]->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnCoreBlock(CoreLocationType coreLocation, bool& didSpill)
{
    bool anyNotEmpty = false;
    for (auto& kv : allocIssueQueue[coreLocation]) {
        if (!kv.second.Empty()) {
            anyNotEmpty = true;
            break;
        }
    }
    if (!anyNotEmpty) {
        return FAILED;
    }

    MemoryType spillMemType;
    if (!allocIssueQueue[coreLocation][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!allocIssueQueue[coreLocation][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        for (auto& memType : allocIssueQueue[coreLocation]) {
            if (memType.second.Empty()) {
                continue;
            }
            PrintSpillFailedInfo(memType.second.Front(), false);
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Buffer[L0A/B/C] is Full. Possible causes: incorrect memory reuse, memory fragmentation. "
            "Please check tile shape and OOO spill failed info.");
        return FAILED;
    }
    if (GenBufferSpill(allocIssueQueue[coreLocation][spillMemType].Front()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    didSpill = true;
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock()
{
    bool didSpill = false;
    for (auto coreLocation : CORE_INIT_CONFIGS) {
        if (SpillOnCoreBlock(coreLocation, didSpill) != SUCCESS) {
            APASS_LOG_WARN_F(
                Elements::Operation, "SpillOnBlock failed/skipped at coreType: %s",
                coreTypeToString(coreLocation).c_str());
        }
    }
    if (!didSpill) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at all coreType.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AllocViewTensorMemRange(Operation& operation)
{
    auto outTensor = operation.GetOOperands()[0];
    int memId = outTensor->memoryrange.memId;
    if (localBufferMap_.find(memId) == localBufferMap_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap_.", memId);
        return FAILED;
    }
    outTensor->memoryrange = TileRange(localBufferMap_[memId]->start, localBufferMap_[memId]->end, memId);
    return SUCCESS;
}

Status OoOScheduler::AllocTensorMemRange(Operation* op)
{
    auto& viewOps = opViewOpsMap[op];
    for (auto& viewOp : viewOps) {
        if (!IsViewOp(*viewOp)) {
            APASS_LOG_ERROR_F(Elements::Operation, "op[%d] is not OP_VIEW.", viewOp->GetOpMagic());
            return FAILED;
        }
        if (AllocViewTensorMemRange(*viewOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AllocViewTensorMemRange failed.");
            return FAILED;
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        MemoryType memType = outTensor->GetMemoryTypeOriginal();
        if (memType >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (tensorOccupyMap.find(memType) != tensorOccupyMap.end()) {
            if (tensorOccupyMap[memType].find(memId) == tensorOccupyMap[memType].end()) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in tensorOccupyMap.", memId);
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "%s cannot find in tensorOccupyMap. %s",
                MemoryTypeToString(memType).c_str(), GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (localBufferMap_.find(memId) == localBufferMap_.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap_.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%d] %s --> %s.",
            memId, GetOpInfo(tensorOccupyMap[memType][memId]).c_str(), GetOpInfo(op).c_str());
        tensorOccupyMap[memType][memId] = op;
        outTensor->memoryrange =
            TileRange(localBufferMap_[memId]->start, localBufferMap_[memId]->end, memId);
    }
    return SUCCESS;
}

void OoOScheduler::HandleViewOp(Operation* op)
{
    auto& viewOps = opViewOpsMap[op];
    for (auto& viewOp : viewOps) {
        if (std::find(newOperations_.begin(), newOperations_.end(), viewOp) != newOperations_.end()) {
            continue;
        }
        newOperations_.emplace_back(viewOp);
    }
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle)
{
    // issue from all pipes
    for (auto coreLocation : CORE_INIT_CONFIGS) {
        for (auto& [pipeType, pipe] : issueQueues[coreLocation]) {
            if (pipe.Empty() || pipe.busy) {
                continue;
            }
            Operation* op = pipe.PopFront();
            // 标注op的生命周期
            op->cycleStart = clock;
            op->cycleEnd = clock + op->GetLatency();
            pipe.busy = true;
            pipe.curIssue = op;
            pipe.curOpRetireCycle = clock + op->GetLatency();
            oooCheck.pipeUsageCount[pipeType] += op->GetLatency();
            HandleViewOp(op);
            newOperations_.emplace_back(op);
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
            if (AllocTensorMemRange(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRangeOp failed at coreType: %s. %s",
                    coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", GetOpInfo(op).c_str());
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, IssueQueue& pipe)
{
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        Operation* op = pipe.Front();
        auto& coreLocation = opCoreLocationMap[op];
        auto& reqMemIds = GetOpMemIds(op);
        if (!bufferManagerMap[coreLocation][memType].IsFull(localBufferMap_[reqMemIds[0]])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", GetOpInfo(op).c_str());
            if (bufferManagerMap[coreLocation][memType].Allocate(localBufferMap_[reqMemIds[0]]) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", reqMemIds[0]);
                return FAILED;
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(memType, reqMemIds[0], false);
            }
            tensorOccupyMap[memType][reqMemIds[0]] = op;
            localBufferMap_[reqMemIds[0]]->startCycle = clock;
            if (op->GetOutputOperand(0) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0]. %s",
                    op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            newOperations_.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", GetOpInfo(op).c_str());
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed. %s",
                    GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        } else {
            canAlloc = false;
            break;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t& commitCnt)
{
    for (auto coreLocation : CORE_INIT_CONFIGS) {
        for (auto& [memType, pipe] : allocIssueQueue[coreLocation]) {
            if (pipe.Empty()) {
                continue;
            }
            // 不断按顺序执行alloc指令，直到buffer被占满为止。
            if (ExecuteAllocIssue(commitCnt, memType, pipe) != SUCCESS) {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "ExecuteAllocIssue failed at coreType: %s.",
                    coreTypeToString(coreLocation).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// ============ 新增：基于Operation*的版本 ============
Status OoOScheduler::FreeBuffer(Operation* op)
{
    auto& reqMemIds = GetOpMemIds(op);
    for (auto memId : reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            auto coreLocation = tensorAllocCoreMap[memId];
            auto memType = localBufferMap_[memId]->memType;
            if (bufferManagerMap[coreLocation][memType].Free(localBufferMap_[memId]->id)
                != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(memType, memId, true);
            }
            localBufferMap_[memId]->retireCycle = clock;
            if (tensorOccupyMap[memType].erase(localBufferMap_[memId]->id) == 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt)
{
    commitCnt++;
    opIsRetiredMap[op] = true;
    if (FreeBuffer(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBufferOp failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    auto& successors = depManager_.GetSuccessors(op);
    auto& coreLocation = opCoreLocationMap[op];
    for (auto succOp : successors) {
        if (opIsRetiredMap[succOp]) {
            continue;
        }
        bool ready = true;
        auto &preds = depManager_.GetPredecessors(succOp);
        for (auto predOp : preds) {
            if (!opIsRetiredMap[predOp]) {
                ready = false;
                break;
            }
        }
        if (ready) {
            issueQueues[coreLocation][opPipeTypeMap[succOp]].Insert(succOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d",
                GetOpInfo(succOp).c_str(), opExecOrderMap[succOp]);
        }
    }
    return SUCCESS;
}
// ============ 新增函数结束 ============

Status OoOScheduler::RetireCoreIssue(CoreLocationType coreLocation, uint64_t& commitCnt, int& nextCycle)
{
    for (auto& [pipeType, pipe] : issueQueues[coreLocation]) {
        if (!pipe.busy) {
            continue;
        }
        if (!pipeEndTime.count(pipeType)) {
            pipeEndTime.emplace(pipeType, pipe.curOpRetireCycle);
        } else {
            auto curEndTime = pipeEndTime[pipeType];
            pipeEndTime[pipeType] = std::max(curEndTime, pipe.curOpRetireCycle);
        }
        if (pipe.curOpRetireCycle <= clock) {   // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            Operation* op = pipe.curIssue;
            pipe.busy = false;
            pipe.curIssue = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", GetOpInfo(op).c_str());
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed at coreType: %s! %s",
                    coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", pipe.curOpRetireCycle,
            GetOpInfo(pipe.curIssue).c_str());
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle)
{
    for (auto coreLocation : CORE_INIT_CONFIGS) {
        if (RetireCoreIssue(coreLocation, commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue()
{
    // 初始化 Queue
    for (auto &op : orderedOps) {
        auto& coreLocation = opCoreLocationMap[op];
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && depManager_.GetPredecessors(op).empty()) {
            issueQueues[coreLocation][opPipeTypeMap[op]].Insert(op);
        }
        if (opIsAllocMap[op]) {
            auto& reqMemIds = GetOpMemIds(op);
            if (!reqMemIds.empty()) {
                auto memType = localBufferMap_[reqMemIds[0]]->memType;
                allocIssueQueue[coreLocation][memType].Insert(op);
            }
        }
    }
}

Status OoOScheduler::ScheduleMainLoop()
{
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    if (RunMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RunMainLoop failed.");
        return FAILED;
    }
    LOG_SCOPE_END(tScheduleMainLoop);
    return SUCCESS;
}

Status OoOScheduler::PreMainLoop()
{
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    numTotalIssues = orderedOps.size();
    return SUCCESS;
}

Status OoOScheduler::PostMainLoop()
{
    return SUCCESS;
}

Status OoOScheduler::RetireIssue(Operation* op)
{
    opIsRetiredMap[op] = true;
    for (auto memId : GetOpMemIds(op)) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            // 加载时的核信息
            auto coreLocation = tensorAllocCoreMap[memId];
            auto memType = localBufferMap_[memId]->memType;
            if (bufferManagerMap[coreLocation][memType].Free(localBufferMap_[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(Operation* op, size_t &pcIdx)
{
    if (localBufferMap_.find(GetOpMemIds(op)[0]) == localBufferMap_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap_!", GetOpMemIds(op)[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = localBufferMap_[GetOpMemIds(op)[0]];
    auto coreLocation = opCoreLocationMap[op];
    if (bufferManagerMap[coreLocation][allocBuffer->memType].IsFull(allocBuffer)) {
        if (GenSpillOp(pcIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenSpillOp failed at ExecuteAllocIssue. %s",
                GetFormatBacktrace(*orderedOps[pcIdx]).c_str());
            return FAILED;
        }
    }
    if (bufferManagerMap[coreLocation][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", allocBuffer->id);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule()
{
    UpdateIssueExecOrder();
    size_t pcIdx = 0;
    LOG_SCOPE_BEGIN(tGenSpillSchedule, Elements::Function, "GenSpillSchedule");
    while (pcIdx < orderedOps.size()) {
        auto op = orderedOps[pcIdx];
        APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", GetOpInfo(op).c_str());
        if (opIsAllocMap[op]) {
            if (ExecuteAllocIssue(op, pcIdx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed! %s", GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        }
        if (RetireIssue(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        pcIdx += 1;
    }
    for (auto bufRef : bufRefCount_) {
        if (bufRef.second != 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first);
            return FAILED;
        }
    }
    LOG_SCOPE_END(tGenSpillSchedule);
    if (InitBufRefCount(orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    for (const auto &op : orderedOps) {
        opIsRetiredMap[op] = false;
    }
    // 更新依赖关系
    if (depManager_.InitDependencies(orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    depManager_.PrintDependencies(orderedOps);
    return SUCCESS;
}

void OoOScheduler::InitIssueQueuesAndBufferManager()
{
    // 设置比较函数用于IssueQueue排序
    auto compareFunc = [this](Operation* a, Operation* b) {
        return opExecOrderMap[a] > opExecOrderMap[b];
    };

    // 初始化
    for (auto coreLocation : CORE_INIT_CONFIGS) {
        for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
            IssueQueue queue;
            queue.SetCompareFunc(compareFunc);
            issueQueues[coreLocation][static_cast<PipeType>(i)] = queue;
        }
    }

    bufferManagerMap.clear();
    for (auto coreLocation : CORE_INIT_CONFIGS) {
        if (coreLocation == CoreLocationType::AIV0 || coreLocation == CoreLocationType::AIV1) {
            IssueQueue queue;
            queue.SetCompareFunc(compareFunc);
            allocIssueQueue[coreLocation][MemoryType::MEM_UB] = queue;
            bufferManagerMap[coreLocation].insert({MemoryType::MEM_UB,
                    BufferPool(MemoryType::MEM_UB, localMemSize[MemoryType::MEM_UB])});
            continue;
        }
        for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
            IssueQueue queue;
            queue.SetCompareFunc(compareFunc);
            allocIssueQueue[coreLocation][static_cast<MemoryType>(i)] = queue;
            if (localMemSize.find(static_cast<MemoryType>(i)) != localMemSize.end()) {
                bufferManagerMap[coreLocation].insert(
                    {static_cast<MemoryType>(i),
                        BufferPool(static_cast<MemoryType>(i), localMemSize[static_cast<MemoryType>(i)])});
            }
        }
    }
}

void OoOScheduler::InitTensorCoreMap()
{
    // 不存在 no producer情况
    for (auto op : orderedOps) {
        if (opIsAllocMap[op]) {
            auto memId = op->GetOutputOperand(0)->memoryrange.memId;
            tensorAllocCoreMap[memId] = opCoreLocationMap[op];
        }
    }
}

void OoOScheduler::InitCoreConfig(const std::vector<Operation *> &opList)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        CORE_INIT_CONFIGS = CORE_INIT_CONFIGS_HARDWARE_ONE;
    } else {
        CORE_INIT_CONFIGS = CORE_INIT_CONFIGS_HARDWARE_TWO;
    }
}

std::string OoOScheduler::GetOpInfo(Operation* op) const
{
    if (op == nullptr) return "nullptr";
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

void OoOScheduler::InitOpViewOps(Operation* op)
{
    if (op == nullptr) return;
    std::vector<Operation*> viewOps;
    for (auto iOperand : op->GetIOperands()) {
        for (auto pre : iOperand->GetProducers()) {
            while (IsViewOp(*pre) && pre->GetOutputOperand(0)->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                viewOps.push_back(pre);
                pre = *(pre->GetInputOperand(0)->GetProducers().begin());
            }
        }
    }
    opViewOpsMap[op] = viewOps;
}

Status OoOScheduler::InitOpCoreType(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    if (op == nullptr) return FAILED;

    CoreLocationType coreLocation;
    if (!opCoreMap.empty()) {
        coreLocation = opCoreMap.at(op);
    } else if (op->GetCoreType() == CoreType::AIC) {
        coreLocation = CoreLocationType::AIC;
    } else if (op->GetCoreType() == CoreType::AIV) {
        coreLocation = CoreLocationType::AIV0;
    } else {
        // 对 ANY 类型进行处理
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            coreLocation = CoreLocationType::AIV0;
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
            coreLocation = CoreLocationType::AIC;
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            if (op->GetIOperands().size() == 0 ||
                op->GetInputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                coreLocation = CoreLocationType::AIC;
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                coreLocation = CoreLocationType::AIV0;
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
                coreLocation = CoreLocationType::AIC;
            } else {
                APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. IOperand memoryType is %s",
                    GetOpInfo(op).c_str(), MemoryTypeToString(op->GetInputOperand(0)->GetMemoryTypeOriginal()).c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. OOperand memoryType is %s",
                GetOpInfo(op).c_str(), MemoryTypeToString(op->GetOutputOperand(0)->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    opCoreLocationMap[op] = coreLocation;
    return SUCCESS;
}

Status OoOScheduler::InitOpEntry(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    if (op == nullptr) return FAILED;

    if (IsViewOp(*op)) {
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            newOperations_.push_back(op);
        }
        return SUCCESS;
    }
    if (CheckOpBufferSize(op) != SUCCESS) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "%s[%d] CheckOpBufferSize failed! %s", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
            GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 初始化Operation属性到map
    int order = static_cast<int>(orderedOps.size());
    orderedOps.push_back(op);
    opExecOrderMap[op] = order;
    opPipeTypeMap[op] = RescheduleUtils::GetOpPipeType(op);
    opIsAllocMap[op] = (op->GetOpcodeStr().find("ALLOC") != std::string::npos);
    opIsRetiredMap[op] = false;
    SetOpMemIds(op, {});

    // 初始化viewOps
    InitOpViewOps(op);

    // 初始化核属性
    if (InitOpCoreType(op, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation %s init coreType failed!", GetOpInfo(op).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "issue: %s, coreType: %s",
        GetOpInfo(op).c_str(), coreTypeToString(opCoreLocationMap[op]).c_str());
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation*>& opList, const std::unordered_map<Operation*,
    CoreLocationType>& opCoreMap, const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    orderedOps.clear();
    opExecOrderMap.clear();
    opPipeTypeMap.clear();
    opIsAllocMap.clear();
    opIsRetiredMap.clear();
    ClearAllOpMemIds();
    opViewOpsMap.clear();
    opCoreLocationMap.clear();
    localBufferMap_.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    localMemSize = CommonUtils::GetLocalMemorySize();
    if (fixCoreConfig.empty()) {
        InitCoreConfig(opList);
    } else {
        CORE_INIT_CONFIGS = fixCoreConfig;
    }
    // 校验并初始化Operation
    for (const auto &op : opList) {
        if (InitOpEntry(op, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init issue failed!",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    numTotalIssues = orderedOps.size();

    if (InitBufRefCount(orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 初始化依赖关系
    if (depManager_.InitDependencies(orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    depManager_.PrintDependencies(orderedOps);
    if (CheckAllocOp(orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocOp failed!");
        return FAILED;
    }
    InitTensorCoreMap();
    // 初始化内存管理器
    InitIssueQueuesAndBufferManager();
    LOG_SCOPE_END(tInit);
    return SUCCESS;
}

void OoOScheduler::UpdateL0MXMap(const std::vector<Operation*> &opList)
{
    for (size_t i = 0; i < opList.size(); i++) {
        if (opList[i]->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
            auto l0MxOut = opList[i]->GetOOperands()[0];
            auto consOp = *l0MxOut->GetConsumers().begin();
            LogicalTensorPtr l0ATensor;
            LogicalTensorPtr l0BTensor;
            LogicalTensorPtr l0AMXTensor;
            LogicalTensorPtr l0BMXTensor;
            for (auto& l0Tensor : consOp->GetIOperands()) {
                if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
                    l0ATensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
                    l0BTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0AMX) {
                    l0AMXTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0BMX) {
                    l0BMXTensor = l0Tensor;
                }
            }
            l02L0MXMap_[l0ATensor] = l0AMXTensor;
            l02L0MXMap_[l0BTensor] = l0BMXTensor;
        }
    }
}

Status OoOScheduler::Schedule(
    const std::vector<Operation*>& opList,
    const std::unordered_map<Operation*, CoreLocationType>& opCoreMap,
    const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    if (opList.empty()) {
        return SUCCESS;
    }
    PrintOpList(opList);
    if (Init(opList, opCoreMap, fixCoreConfig) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed!");
        return FAILED;
    }
    // 生成spill指令
    if (GenSpillSchedule() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenSpillSchedule failed!");
        return FAILED;
    }
    // 模拟调度
    if (ScheduleMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleMainLoop failed!");
        return FAILED;
    }
    if (CheckAndUpdateLifecycle() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAndUpdateLifecycle failed!");
        return FAILED;
    }
    UpdateL0MXMap(opList);
    for (auto& entry : l02L0MXMap_) {
        auto l0Tensor = entry.first;
        auto l0MXTensor = entry.second;
        int l0MemID = l0Tensor->memoryrange.memId;
        int l0MemMXID = l0MXTensor->memoryrange.memId;
        l0MXTensor->memoryrange =
            TileRange(localBufferMap_[l0MemID]->start >> 4, localBufferMap_[l0MemID]->end >> 4, l0MemMXID);
    }
    PrintOpList(newOperations_);
    function_.SetStackWorkespaceSize(workspaceOffset);
    function_.pipeEndTime = pipeEndTime;
    return SUCCESS;
}

} // namespace npu::tile_fwk
