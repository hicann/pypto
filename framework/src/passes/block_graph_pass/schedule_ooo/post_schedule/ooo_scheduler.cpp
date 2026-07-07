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

void OoOScheduler::PrintOpList(std::vector<Operation*> opList)
{
    APASS_LOG_INFO_F(Elements::Operation, "====================OoO OP_LIST =====================");
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
    for (size_t idx = 0; idx < state_.orderedOps.size(); idx++) {
        state_.schedInfoMap[state_.orderedOps[idx]].execOrder = idx;
    }
}

Status OoOScheduler::CheckAndUpdateLifecycle()
{
    for (const auto &op : state_.orderedOps) {
        if (!state_.schedInfoMap[op].isRetired) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", state_.GetOpInfo(op).c_str(),
                GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (state_.schedInfoMap[op].isAlloc) {
            op->GetOutputOperand(0)->memoryrange.lifeStart =
                state_.localBufferMap[state_.GetOpMemIds(op)[0]]->startCycle;
            op->GetOutputOperand(0)->memoryrange.lifeEnd =
                state_.localBufferMap[state_.GetOpMemIds(op)[0]]->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> orderFirstPair)
{
    SpillContext ctx;
    if (GenBufferSpill(state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front(), ctx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    if (ApplySpillContext(ctx, state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ApplySpillContext failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType &spillMemType)
{
    bool anyNotEmpty = false;
    for (auto& kv : state_.allocIssueQueue[coreLocation]) {
        if (!kv.second.Empty()) {
            anyNotEmpty = true;
            break;
        }
    }
    if (!anyNotEmpty) {
        return FAILED;
    }
    if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else if (coreLocation == CoreLocationType::AIC &&
               !state_.allocIssueQueue[coreLocation][MemoryType::MEM_L0C].Empty()) {
        spillMemType = MemoryType::MEM_L0C;
    } else {
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindFirstOrder(std::pair<CoreLocationType, MemoryType> &orderFirstPair)
{
    std::unordered_map<CoreLocationType, MemoryType> spillMemTypeMap;
    std::vector<CoreLocationType> coreVec;
    for (auto& coreLocation : state_.coreInitConfigs) {
        MemoryType spillMemType;
        if (FindCoreLocationMemoryType(coreLocation, spillMemType) != SUCCESS) {
            APASS_LOG_INFO_F(Elements::Operation, "FindCoreLocationMemoryType %s failed.", coreTypeToString(coreLocation).c_str());
            continue;
        }
        spillMemTypeMap[coreLocation] = spillMemType;
        coreVec.push_back(coreLocation);
    }
    if (spillMemTypeMap.empty() || coreVec.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "All coreLocation have no spillMemType.");
        return FAILED;
    }
    std::sort(coreVec.begin(), coreVec.end(), [&spillMemTypeMap, this](const CoreLocationType& a, const CoreLocationType& b) {
        return state_.schedInfoMap[state_.allocIssueQueue[a][spillMemTypeMap[a]].Front()].execOrder < state_.schedInfoMap[state_.allocIssueQueue[b][spillMemTypeMap[b]].Front()].execOrder;
    });
    orderFirstPair = std::make_pair(coreVec[0], spillMemTypeMap[coreVec[0]]);
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock()
{
    std::pair<CoreLocationType, MemoryType> orderFirstPair;
    if (FindFirstOrder(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FindFirstOrder failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Start to spillOnBlock at %s", coreTypeToString(orderFirstPair.first).c_str());
    if (SpillOnCoreBlock(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnCoreBlock failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AllocViewTensorMemRange(Operation& operation)
{
    auto outTensor = operation.GetOOperands()[0];
    int memId = outTensor->memoryrange.memId;
    if (state_.localBufferMap.find(memId) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in state_.localBufferMap.", memId);
        return FAILED;
    }
    outTensor->memoryrange = TileRange(state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end, memId);
    return SUCCESS;
}

Status OoOScheduler::AllocTensorMemRange(Operation* op)
{
    auto& viewOps = state_.schedInfoMap[op].viewOps;
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
        if (state_.tensorOccupyMap.find(memId) == state_.tensorOccupyMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in tensorOccupyMap.", memId);
            return FAILED;
        }
        if (state_.localBufferMap.find(memId) == state_.localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%d] %s --> %s.",
            memId, state_.GetOpInfo(state_.tensorOccupyMap[memId]).c_str(), state_.GetOpInfo(op).c_str());
        state_.tensorOccupyMap[memId] = op;
        outTensor->memoryrange =
            TileRange(state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end, memId);
    }
    return SUCCESS;
}

void OoOScheduler::HandleViewOp(Operation* op)
{
    auto& viewOps = state_.schedInfoMap[op].viewOps;
    for (auto& viewOp : viewOps) {
        if (std::find(state_.newOperations.begin(), state_.newOperations.end(), viewOp) != state_.newOperations.end()) {
            continue;
        }
        state_.newOperations.emplace_back(viewOp);
    }
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle)
{
    for (auto coreLocation : state_.coreInitConfigs) {
        for (auto& pipeEntry : issueQueues[coreLocation]) {
            auto& pipe = pipeEntry.second;
            if (pipe.Empty() || pipe.busy) {
                continue;
            }
            Operation* op = pipe.PopFront();
            // 标注op的生命周期
            op->cycleStart = state_.clock;
            op->cycleEnd = state_.clock + op->GetLatency();
            pipe.busy = true;
            pipe.curOp = op;
            pipe.curOpRetireCycle = state_.clock + op->GetLatency();
            NotifyOpLaunch(op, op->cycleEnd);
            HandleViewOp(op);
            state_.newOperations.emplace_back(op);
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
            if (AllocTensorMemRange(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRangeOp failed at coreType: %s. %s",
                    coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", state_.GetOpInfo(op).c_str());
        }
    }
    return SUCCESS;
}

Status OoOScheduler::TryDualDstAllocOnce(Operation* op, uint64_t& commitCnt, bool& allocated)
{
    if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS) return FAILED;
    if (!allocated) return SUCCESS;     // 当前 Full,由调用方 break 触发 SpillOnBlock
    state_.newOperations.push_back(op);
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert(dualdst): %s.", state_.GetOpInfo(op).c_str());
    if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "RetireOpAndAwakeSucc failed for dualdst alloc. %s",
            GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::TryRegularAllocOnce(Operation* op, MemoryType memType,
                                         CoreLocationType coreLocation,
                                         const std::vector<int>& reqMemIds,
                                         uint64_t& commitCnt, bool& allocated)
{
    auto& pool = state_.bufferManagerMap[coreLocation][memType];
    auto buf = state_.localBufferMap[reqMemIds[0]];
    if (pool.IsFull(buf, true)) { allocated = false; return SUCCESS; }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", state_.GetOpInfo(op).c_str());
    if (pool.Allocate(buf) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", reqMemIds[0]);
        return FAILED;
    }
    NotifyAllocExec(op, reqMemIds[0]);
    state_.tensorOccupyMap[reqMemIds[0]] = op;
    buf->startCycle = state_.clock;
    if (op->GetOutputOperand(0) == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0]. %s",
            op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    state_.newOperations.push_back(op);
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", state_.GetOpInfo(op).c_str());
    if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed. %s",
            GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    allocated = true;
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, OpQueue& pipe)
{
    while (!pipe.Empty()) {
        Operation* op = pipe.Front();
        auto& coreLocation = state_.schedInfoMap[op].coreLocation;
        auto& reqMemIds = state_.GetOpMemIds(op);
        bool allocated = false;
        Status st = (memType == MemoryType::MEM_UB && dualDstEngine_.IsDualDstAlloc(op))
                  ? TryDualDstAllocOnce(op, commitCnt, allocated)
                  : TryRegularAllocOnce(op, memType, coreLocation, reqMemIds, commitCnt, allocated);
        if (st != SUCCESS) return FAILED;
        if (!allocated) break;     // Full;退出循环交给 SpillOnBlock
        pipe.PopFront();
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t& commitCnt)
{
    for (auto coreLocation : state_.coreInitConfigs) {
        for (auto& [memType, pipe] : state_.allocIssueQueue[coreLocation]) {
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

Status OoOScheduler::FreeBuffer(Operation* op, std::vector<int>& freedMemIds)
{
    auto& reqMemIds = state_.GetOpMemIds(op);
    for (auto memId : reqMemIds) {
        if (state_.DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (state_.bufRefCount[memId] == 0) {
            CoreLocationType coreLocation = dualDstEngine_.ResolveCoreForFree(memId);
            auto memType = state_.localBufferMap[memId]->memType;
            if (state_.bufferManagerMap[coreLocation][memType].Free(state_.localBufferMap[memId]->id)
                != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            freedMemIds.push_back(memId);
            if (state_.tensorOccupyMap.erase(memId) == 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", memId);
                return FAILED;
            }
            state_.dualDstMemIdCoreOverride.erase(memId);
        }
    }
    return SUCCESS;
}

std::vector<Operation*> OoOScheduler::GetNewOperations()
{
    std::vector<Operation*> uniq;
    uniq.reserve(state_.newOperations.size());
    std::unordered_set<Operation*> seen;
    for (auto* op : state_.newOperations) {
        if (op == nullptr) continue;
        if (seen.insert(op).second) {
            uniq.push_back(op);
        } else {
            // 命中即代表上游某条 push 路径漏了去重 (期望随后续根因修复降为 0)。
            APASS_LOG_WARN_F(Elements::Operation,
                "GetNewOperations dedupe drop duplicate op[%d] %s",
                op->GetOpMagic(), op->GetOpcodeStr().c_str());
        }
    }
    return uniq;
}

Status OoOScheduler::RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt)
{
    commitCnt++;
    state_.schedInfoMap[op].isRetired = true;
    std::vector<int> freedMemIds;
    if (FreeBuffer(op, freedMemIds) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBufferOp failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    for (auto memId : freedMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    NotifyOpRetire(op, freedMemIds);

    auto& successors = state_.depManager.GetSuccessors(op);
    for (auto succOp : successors) {
        if (state_.schedInfoMap[succOp].isRetired) {
            continue;
        }
        bool ready = true;
        auto &preds = state_.depManager.GetPredecessors(succOp);
        for (auto predOp : preds) {
            if (!state_.schedInfoMap[predOp].isRetired) {
                ready = false;
                break;
            }
        }
        if (ready) {
            issueQueues[state_.schedInfoMap[succOp].coreLocation][state_.schedInfoMap[succOp].pipeType].Insert(succOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d",
                state_.GetOpInfo(succOp).c_str(), state_.schedInfoMap[succOp].execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireCoreIssue(CoreLocationType coreLocation, uint64_t& commitCnt, int& nextCycle)
{
    for (auto& [pipeType, pipe] : issueQueues[coreLocation]) {
        if (!pipe.busy) {
            continue;
        }
        if (!state_.pipeEndTime.count(pipeType)) {
            state_.pipeEndTime.emplace(pipeType, pipe.curOpRetireCycle);
        } else {
            auto curEndTime = state_.pipeEndTime[pipeType];
            state_.pipeEndTime[pipeType] = std::max(curEndTime, pipe.curOpRetireCycle);
        }
        if (pipe.curOpRetireCycle <= state_.clock) {   // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            Operation* op = pipe.curOp;
            pipe.busy = false;
            pipe.curOp = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", state_.GetOpInfo(op).c_str());
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed at coreType: %s! %s",
                    coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", pipe.curOpRetireCycle,
            state_.GetOpInfo(pipe.curOp).c_str());
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle)
{
    for (auto coreLocation : state_.coreInitConfigs) {
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
    for (auto &op : state_.orderedOps) {
        auto& coreLocation = state_.schedInfoMap[op].coreLocation;
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && state_.depManager.GetPredecessors(op).empty()) {
            issueQueues[coreLocation][state_.schedInfoMap[op].pipeType].Insert(op);
        }
        if (state_.schedInfoMap[op].isAlloc) {
            auto& reqMemIds = state_.GetOpMemIds(op);
            if (!reqMemIds.empty()) {
                auto memType = state_.localBufferMap[reqMemIds[0]]->memType;
                state_.allocIssueQueue[coreLocation][memType].Insert(op);
            }
        }
    }
}

Status OoOScheduler::ScheduleMainLoop()
{
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    auto ret = RunSchedulerMainLoop(*this);
    LOG_SCOPE_END(tScheduleMainLoop);
    return ret;
}

Status OoOScheduler::PreMainLoop()
{
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    state_.numTotalIssues = state_.orderedOps.size();
    NotifyMainLoopBegin();
    return SUCCESS;
}

Status OoOScheduler::PostMainLoop()
{
    NotifyMainLoopEnd();
    return SUCCESS;
}

Status OoOScheduler::RetireIssue(Operation* op)
{
    state_.schedInfoMap[op].isRetired = true;
    std::vector<int> freedMemIds;
    return FreeBuffer(op, freedMemIds);
}

Status OoOScheduler::ExecuteAllocIssue(Operation* op, size_t &pcIdx)
{
    if (state_.localBufferMap.find(state_.GetOpMemIds(op)[0]) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", state_.GetOpMemIds(op)[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.GetOpMemIds(op)[0]];
    auto coreLocation = state_.schedInfoMap[op].coreLocation;
    const bool isDualDst = (allocBuffer->memType == MemoryType::MEM_UB) && dualDstEngine_.IsDualDstAlloc(op);

    bool needSpill = false;
    if (isDualDst) {
        bool allocated = false;
        if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS) return FAILED;
        if (allocated) return SUCCESS;
        needSpill = true;
    } else {
        needSpill = state_.bufferManagerMap[coreLocation][allocBuffer->memType].IsFull(allocBuffer, false);
    }

    // === Spill (单池 / dualdst 共用 GenBufferSpill, 决策分叉在 SelectSpillBuffers 内) ===
    if (needSpill) {
        SpillContext ctx;
        if (GenBufferSpill(op, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenBufferSpill failed at ExecuteAllocIssue. %s",
                GetFormatBacktrace(*state_.orderedOps[pcIdx]).c_str());
            return FAILED;
        }
        pcIdx += ctx.newCopyoutOps.size() - ctx.deleteRetiredOpSize;
    }

    // === 最终 alloc (dualdst 走双池同址 retry, 单池走 BufferPool::Allocate) ===
    if (isDualDst) {
        bool allocated = false;
        if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS) return FAILED;
        if (!allocated) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "DualDst alloc still infeasible after spill. %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    } else {
        if (state_.bufferManagerMap[coreLocation][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", allocBuffer->id);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SeqSchedule()
{
    UpdateIssueExecOrder();
    size_t pcIdx = 0;
    LOG_SCOPE_BEGIN(tGenSpillSchedule, Elements::Function, "SeqSchedule");
    while (pcIdx < state_.orderedOps.size()) {
        auto op = state_.orderedOps[pcIdx];
        APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", state_.GetOpInfo(op).c_str());
        if (state_.schedInfoMap[op].isAlloc) {
            if (ExecuteAllocIssue(op, pcIdx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed! %s", GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        }
        for (auto& outTensor : op->GetOOperands()) {
            MemoryType memType = outTensor->GetMemoryTypeOriginal();
            if (memType >= MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = outTensor->memoryrange.memId;
            state_.tensorOccupyMap[memId] = op;
        }
        if (RetireIssue(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        pcIdx += 1;
    }
    for (auto bufRef : state_.bufRefCount) {
        if (bufRef.second != 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first);
            return FAILED;
        }
    }
    LOG_SCOPE_END(tGenSpillSchedule);
    if (state_.InitBufRefCount(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    for (const auto &op : state_.orderedOps) {
        state_.schedInfoMap[op].isRetired = false;
    }
    // 更新依赖关系
    if (state_.depManager.InitDependencies(state_.orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    state_.depManager.PrintDependencies(state_.orderedOps);
    return SUCCESS;
}

void OoOScheduler::InitIssueQueuesAndBufferManager()
{
    // 设置比较函数用于OpQueue排序
    auto compareFunc = [this](Operation* a, Operation* b) {
        return state_.schedInfoMap[a].execOrder > state_.schedInfoMap[b].execOrder;
    };

    // 初始化
    for (auto coreLocation : state_.coreInitConfigs) {
        for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            issueQueues[coreLocation][static_cast<PipeType>(i)] = queue;
        }
    }

    state_.bufferManagerMap.clear();
    for (auto coreLocation : state_.coreInitConfigs) {
        if (coreLocation == CoreLocationType::AIV0 || coreLocation == CoreLocationType::AIV1) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB] = queue;
            state_.bufferManagerMap[coreLocation].insert({MemoryType::MEM_UB,
                    BufferPool(MemoryType::MEM_UB, state_.localMemSize[MemoryType::MEM_UB])});
            continue;
        }
        for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            state_.allocIssueQueue[coreLocation][static_cast<MemoryType>(i)] = queue;
            if (state_.localMemSize.find(static_cast<MemoryType>(i)) != state_.localMemSize.end()) {
                state_.bufferManagerMap[coreLocation].insert(
                    {static_cast<MemoryType>(i),
                        BufferPool(static_cast<MemoryType>(i), state_.localMemSize[static_cast<MemoryType>(i)])});
            }
        }
    }
}

void OoOScheduler::InitTensorCoreMap()
{
    // 不存在 no producer情况
    for (auto op : state_.orderedOps) {
        if (state_.schedInfoMap[op].isAlloc) {
            auto memId = op->GetOutputOperand(0)->memoryrange.memId;
            state_.tensorAllocMap[memId] = op;
        }
    }
}

void OoOScheduler::InitCoreConfig(const std::vector<Operation *> &opList)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        state_.coreInitConfigs = CORE_INIT_CONFIGS_HARDWARE_ONE;
    } else {
        state_.coreInitConfigs = CORE_INIT_CONFIGS_HARDWARE_TWO;
    }
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
    state_.schedInfoMap[op].viewOps = viewOps;
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
                    state_.GetOpInfo(op).c_str(), MemoryTypeToString(op->GetInputOperand(0)->GetMemoryTypeOriginal()).c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. OOperand memoryType is %s",
                state_.GetOpInfo(op).c_str(), MemoryTypeToString(op->GetOutputOperand(0)->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    state_.schedInfoMap[op].coreLocation = coreLocation;
    return SUCCESS;
}

Status OoOScheduler::InitOpEntry(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    if (op == nullptr) return FAILED;

    if (IsViewOp(*op)) {
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            state_.newOperations.push_back(op);
        }
        return SUCCESS;
    }
    if (state_.CheckOpBufferSize(op) != SUCCESS) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "%s[%d] CheckOpBufferSize failed! %s", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
            GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 初始化Operation属性到map
    int order = static_cast<int>(state_.orderedOps.size());
    state_.orderedOps.push_back(op);
    state_.schedInfoMap[op].execOrder = order;
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    state_.schedInfoMap[op].isAlloc = (op->GetOpcodeStr().find("ALLOC") != std::string::npos);
    state_.schedInfoMap[op].isRetired = false;
    state_.SetOpMemIds(op, {});

    // 初始化viewOps
    InitOpViewOps(op);

    // 初始化核属性
    if (InitOpCoreType(op, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation %s init coreType failed!", state_.GetOpInfo(op).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "issue: %s, coreType: %s",
        state_.GetOpInfo(op).c_str(), coreTypeToString(state_.schedInfoMap[op].coreLocation).c_str());
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation*>& opList, const std::unordered_map<Operation*,
    CoreLocationType>& opCoreMap, const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    state_.orderedOps.clear();
    state_.schedInfoMap.clear();
    state_.ClearAllOpMemIds();
    state_.localBufferMap.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    state_.localMemSize = CommonUtils::GetLocalMemorySize();
    if (fixCoreConfig.empty()) {
        InitCoreConfig(opList);
    } else {
        state_.coreInitConfigs = fixCoreConfig;
    }
    // 校验并初始化Operation
    for (const auto &op : opList) {
        if (InitOpEntry(op, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init issue failed!",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    state_.numTotalIssues = state_.orderedOps.size();

    if (state_.InitBufRefCount(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 初始化依赖关系
    if (state_.depManager.InitDependencies(state_.orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    state_.depManager.PrintDependencies(state_.orderedOps);
    if (state_.CheckAllocOp(state_.orderedOps) != SUCCESS) {
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
            dualDstEngine_.GetL02L0MXMap()[l0ATensor] = l0AMXTensor;
            dualDstEngine_.GetL02L0MXMap()[l0BTensor] = l0BMXTensor;
        }
    }
}

Status OoOScheduler::Schedule(
    const std::vector<Operation*>& opList,
    const std::unordered_map<Operation*, CoreLocationType>& opCoreMap,
    const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    struct EndGuard {
        OoOScheduler* self;
        bool success{false};
        ~EndGuard() { self->NotifyScheduleEnd(success); }
    } guard{this};
    if (opList.empty()) {
        guard.success = true;
        return SUCCESS;
    }

    PrintOpList(opList);
    if (Init(opList, opCoreMap, fixCoreConfig) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed!");
        return FAILED;
    }
    AllocWorkspaceGM(opList);
    // DualDst 融合(开关关闭 / 非 Mix 路径会内部直接返回)
    if (dualDstEngine_.RunDualDstFuse() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RunDualDstFuse failed!");
        return FAILED;
    }
    NotifyInitDDRBuffers();
    // 生成spill指令（顺序模拟阶段）
    if (SeqSchedule() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenSpillSchedule failed!");
        return FAILED;
    }
    // 模拟调度（乱序模拟阶段）
    if (ScheduleMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleMainLoop failed!");
        return FAILED;
    }
    if (CheckAndUpdateLifecycle() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAndUpdateLifecycle failed!");
        return FAILED;
    }
    UpdateL0MXMap(opList);
    constexpr int kL0mxAddrShiftBits = 4; // L0→L0MX 地址右移位数 (16 字节粒度)
    for (auto& entry : dualDstEngine_.GetL02L0MXMap()) {
        auto l0Tensor = entry.first;
        auto l0MXTensor = entry.second;
        int l0MemID = l0Tensor->memoryrange.memId;
        int l0MemMXID = l0MXTensor->memoryrange.memId;
        l0MXTensor->memoryrange =
            TileRange(state_.localBufferMap[l0MemID]->start >> kL0mxAddrShiftBits,
                      state_.localBufferMap[l0MemID]->end >> kL0mxAddrShiftBits, l0MemMXID);
    }
    PrintOpList(state_.newOperations);
    function_.SetStackWorkespaceSize(state_.workspaceOffset);
    function_.pipeEndTime = state_.pipeEndTime;
    guard.success = true;
    return SUCCESS;
}

void OoOScheduler::AllocWorkspaceGM(const std::vector<Operation *> &opList) {
    std::set<int> allocedRawmagic;
    for (auto &inCast : function_.GetIncast()) {
        allocedRawmagic.insert(inCast->tensor->GetRawMagic());
    }
    for (auto &outCast : function_.GetOutcast()) {
        allocedRawmagic.insert(outCast->tensor->GetRawMagic());
    }
    std::map<int, TileRange> rawMagicRange;
    std::map<int, int64_t> rawMagicOffset;
    for (auto &op : opList) {
        for (auto &iOperand : op->GetIOperands()) {
            if (allocedRawmagic.count(iOperand->tensor->GetRawMagic()) && rawMagicRange.count(iOperand->tensor->GetRawMagic())) {
                iOperand->memoryrange = rawMagicRange[iOperand->tensor->GetRawMagic()];
                iOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, rawMagicOffset[iOperand->tensor->GetRawMagic()]);
            } else if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                !allocedRawmagic.count(iOperand->tensor->GetRawMagic())) {
                allocedRawmagic.insert(iOperand->tensor->GetRawMagic());
                iOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
                iOperand->memoryrange =
                    TileRange(state_.workspaceOffset, state_.workspaceOffset + iOperand->tensor->GetRawDataSize(), iOperand->tensor->GetRawMagic());
                rawMagicOffset[iOperand->tensor->GetRawMagic()] = state_.workspaceOffset;
                state_.workspaceOffset += iOperand->tensor->GetRawDataSize();
                rawMagicRange[iOperand->tensor->GetRawMagic()] = iOperand->memoryrange;
                ddrKindMap_[iOperand->memoryrange.memId] = DDRBufferKind::FUNCTION_TEMP;
            }
        }
        for (auto &oOperand : op->GetOOperands()) {
            if (allocedRawmagic.count(oOperand->tensor->GetRawMagic()) && rawMagicRange.count(oOperand->tensor->GetRawMagic())) {
                oOperand->memoryrange = rawMagicRange[oOperand->tensor->GetRawMagic()];
                oOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, rawMagicOffset[oOperand->tensor->GetRawMagic()]);
            } else if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                !allocedRawmagic.count(oOperand->tensor->GetRawMagic())) {
                allocedRawmagic.insert(oOperand->tensor->GetRawMagic());
                oOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
                oOperand->memoryrange =
                    TileRange(state_.workspaceOffset, state_.workspaceOffset + oOperand->tensor->GetRawDataSize(), oOperand->tensor->GetRawMagic());
                rawMagicOffset[oOperand->tensor->GetRawMagic()] = state_.workspaceOffset;
                state_.workspaceOffset += oOperand->tensor->GetRawDataSize();
                rawMagicRange[oOperand->tensor->GetRawMagic()] = oOperand->memoryrange;
                ddrKindMap_[oOperand->memoryrange.memId] = DDRBufferKind::FUNCTION_TEMP;
            }
        }
    }
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType)
{
    return !state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].IsFull(
        state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]], false);
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType)
{
    std::vector<int> memIds = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = spillEngine_.GetSpillOp(memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] lastest write op.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return SUCCESS;
        }
    }
    std::vector<BufferAddrChange> changes;
    auto status =
        state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].CompactBufferSlices(state_.localBufferMap, changes);
    if (status == SUCCESS) {
        NotifyBufferRearrange(allocOp, memType, std::move(changes));
    }
    return status;
}

std::vector<std::vector<int>> OoOScheduler::GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill)
{
    return pool.GetSpillGroup(sizeNeedSpill);
}

std::vector<std::vector<int>> OoOScheduler::GetDualSpillGroup(
    BufferPool& poolA, BufferPool& poolB, size_t sizeNeedSpill)
{
    std::vector<std::vector<int>> result;
    auto bufsA = poolA.GetSortedAllocatedBufs();
    auto bufsB = poolB.GetSortedAllocatedBufs();

    size_t iA = 0;
    while (iA < bufsA.size()) {
        size_t startAddrA = poolA.ObtainStartAddr(iA, bufsA);
        if ((poolA.GetMemSize() - startAddrA) < sizeNeedSpill) {
            break;
        }
        size_t jA = poolA.UpdateIdx(iA, sizeNeedSpill, startAddrA, bufsA);
        if (iA == jA) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect idx for poolA allocatedBufs.");
            return result;
        }

        size_t iB = 0;
        while (iB < bufsB.size()) {
            size_t startAddrB = poolB.ObtainStartAddr(iB, bufsB);
            if (startAddrB != startAddrA) {
                if (startAddrB < startAddrA) { iB++; continue; }
                break;
            }
            if ((poolB.GetMemSize() - startAddrB) < sizeNeedSpill) {
                break;
            }
            size_t jB = poolB.UpdateIdx(iB, sizeNeedSpill, startAddrB, bufsB);
            if (iB == jB) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect idx for poolB allocatedBufs.");
                return result;
            }
            std::vector<int> combined;
            combined.reserve((jA - iA) + (jB - iB));
            for (size_t k = iA; k < jA; k++) combined.push_back(std::get<0>(bufsA[k]));
            for (size_t k = iB; k < jB; k++) combined.push_back(std::get<0>(bufsB[k]));
            result.push_back(std::move(combined));
            iB++;
        }
        iA++;
    }
    return result;
}

Status OoOScheduler::GetGroupNextUseTime(std::vector<int> group, Operation* allocOp,
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache)
{
    size_t minNextUseTime = INT_MAX;
    for (auto& memId : group) {
        Operation* spillOp = spillEngine_.GetSpillOp(memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
            return FAILED;
        }
        if (spillEngine_.IsBelongSpillBlackList(spillOp, allocOp)) {
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            minNextUseTime = std::min(minNextUseTime, nextUseTimeCache[memId]);
        } else {
            int nextUseTime = spillEngine_.GetBufNextUseTime(memId);
            if (nextUseTime == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseTime);
            minNextUseTime = std::min(minNextUseTime, static_cast<size_t>(nextUseTime));
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp)
{
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]];
    std::vector<int> spillGroup;
    std::vector<std::vector<int>> canSpillGroups;

    if (dualDstEngine_.IsDualDstAlloc(allocOp)) {
        DualDstAllocCtx ctx;
        if (dualDstEngine_.ResolveDualDstAllocCtx(allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "DualDst spill select: ResolveDualDstAllocCtx failed.");
            return {};
        }
        auto& poolA = state_.bufferManagerMap[ctx.coreA][MemoryType::MEM_UB];
        auto& poolB = state_.bufferManagerMap[ctx.coreB][MemoryType::MEM_UB];
        auto sortedA = poolA.GetAddrSortedBufs();
        auto sortedB = poolB.GetAddrSortedBufs();
        spillGroup.reserve(sortedA.size() + sortedB.size());
        spillGroup.insert(spillGroup.end(), sortedA.begin(), sortedA.end());
        spillGroup.insert(spillGroup.end(), sortedB.begin(), sortedB.end());
        canSpillGroups = GetDualSpillGroup(poolA, poolB, ctx.bufA->size);
    } else {
        auto coreType = state_.schedInfoMap[allocOp].coreLocation;
        auto& pool = state_.bufferManagerMap[coreType][allocBuffer->memType];
        spillGroup = pool.GetAddrSortedBufs();
        canSpillGroups = GetSpillGroup(pool, allocBuffer->size);
    }

    if (canSpillGroups.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return spillGroup;
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : canSpillGroups) {
        if (GetGroupNextUseTime(group, allocOp, groupNextUseTime, nextUseTimeCache) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "Get group next use time failed, begin spill all tensor.");
            return spillGroup;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return spillGroup;
    }
    spillGroup = canSpillGroups[groupSel];
    return spillGroup;
}

Status OoOScheduler::GenBufferSpill(Operation* allocOp, SpillContext& ctx)
{
    auto reqMemType = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->memType;
    auto reqSize = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->size;
    std::vector<int> spillGroup = SelectSpillBuffers(allocOp);
    if (spillGroup.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    ctx.spillMemIds = spillGroup;
    for (auto& memId : spillGroup) {
        if (spillEngine_.SpillBuffer(memId, allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill tensor[%d] for %s failed!",
                memId, state_.GetOpInfo(allocOp).c_str());
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
    }

    if (RearrangeBuffer(allocOp, reqMemType) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed for %s.", GetFormatBacktrace(*allocOp).c_str());
    }
    if (!HasEnoughBuffer(allocOp, reqMemType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed!");
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ApplySpillContext(SpillContext& ctx, Operation* allocOp)
{
    for (auto* copyoutOp : ctx.newCopyoutOps) {
        int spillMemid = copyoutOp->GetInputOperand(0)->memoryrange.memId;
        Operation *insertOp = nullptr;
        for (auto op : state_.newOperations) {
            auto &reqMemids = state_.GetOpMemIds(op);
            if (std::find(reqMemids.begin(), reqMemids.end(), spillMemid) != reqMemids.end()) {
                insertOp = op;
            }
        }
        if (insertOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Insert %s in newOperations failed.", state_.GetOpInfo(copyoutOp).c_str());
            return FAILED;
        }
        auto it = find(state_.newOperations.begin(), state_.newOperations.end(), insertOp);
        if (it != state_.newOperations.end()) {
            state_.newOperations.insert(it + 1, copyoutOp);
        }
    }
    state_.numTotalIssues += ctx.newNotRetiredCopyOutSize - ctx.deleteNotRetiredOpSize;
    MemoryType memType = allocOp->GetOutputOperand(0)->GetMemoryTypeOriginal();

    for (auto deleteOp : ctx.deleteAllocOps) {
        state_.allocIssueQueue[std::get<2>(deleteOp)][std::get<1>(deleteOp)].DeleteOp(std::get<0>(deleteOp));
    }
    for (auto &newAllocOp : ctx.newAllocOps) {
        state_.allocIssueQueue[state_.schedInfoMap[allocOp].coreLocation][memType].Insert(newAllocOp);
    }

    for (auto memId : ctx.spillMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    return SUCCESS;
}

Status OoOScheduler::PrintSpillFailedInfo(Operation* allocOp)
{
    auto memType = state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->memType;
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s",
        MemoryTypeToString(memType).c_str(), GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
    APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", state_.GetOpInfo(allocOp).c_str(),
        state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->size, GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
    for (auto& [memId, occupyOp] : state_.tensorOccupyMap) {
        if (state_.localBufferMap[memId]->memType != memType) {
            continue;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer: %s. %s", memId,
            state_.localBufferMap[memId]->size, state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end,
            state_.GetOpInfo(occupyOp).c_str(), GetFormatBacktrace(*occupyOp).c_str());
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
