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
 * \file scheduler.cpp
 * \brief
 */

#include "scheduler.h"
#include "buffer_rearrange.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

inline std::string coreTypeToString(OpCoreType coreType)
{
    switch (coreType) {
        case OpCoreType::AIV:
            return "AIV";
        case OpCoreType::AIC:
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

Operation* OoOScheduler::SkipViewChain(Operation* start, bool followProducers) {
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

int OoOScheduler::GetOOperandIdx(Operation* op, int curMemId) {
    for (size_t i = 0; i < op->GetOOperands().size(); i++) {
        if (op->GetOOperands()[i]->memoryrange.memId == curMemId) {
            return i;
        }
    }
    return -1;
}

Status OoOScheduler::PrintSpillFailedInfo(Operation* allocOp, bool isGenSpill)
{
    auto memType = localBufferMap[opReqMemIdsMap[allocOp][0]]->memType;
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s",
        MemoryTypeToString(memType).c_str(), GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
    APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", GetOpInfo(allocOp).c_str(),
        localBufferMap[opReqMemIdsMap[allocOp][0]]->size, GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
    auto corePair = opCoreLocationMap[allocOp];
    if (isGenSpill) {
        auto bufferSlices = bufferManagerMap[corePair.first][corePair.second][memType].GetBufferSlices();
        for (auto memId : bufferSlices) {
            auto occupyOp = GetBufLastWriteOp(allocOp, memId);
            if (occupyOp == nullptr) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write time.", memId);
                return FAILED;
            }
            APASS_LOG_ERROR_F(
                Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer:%s. %s", memId, 
                localBufferMap[memId]->size, localBufferMap[memId]->start, localBufferMap[memId]->end, 
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
                    localBufferMap[memId]->size, localBufferMap[memId]->start, localBufferMap[memId]->end, 
                    GetOpInfo(occupyOp).c_str(), GetFormatBacktrace(*occupyOp).c_str());
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree) {
    if (isFree) {
        int freeBufferSize = localBufferMap[memId]->size;
        oooCheck.bufferTotalUsage[bufferType] +=
            oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] -= freeBufferSize;
        oooCheck.lastClock[bufferType] = clock;
    } else {
        oooCheck.bufferTotalUsage[bufferType] +=
            oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] += localBufferMap[memId]->size;
        oooCheck.lastClock[bufferType] = clock;
        oooCheck.bufferMaxUsage[bufferType] =
            std::max(oooCheck.bufferMaxUsage[bufferType], oooCheck.bufferLastUsage[bufferType]);
    }
}

Status OoOScheduler::DelBufRefCount(const int memId)
{
    if (bufRefCount_.find(memId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount_[memId]--;
    if (bufRefCount_[memId] < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::PrintOpList(std::vector<Operation*> operations)
{
    APASS_LOG_INFO_F(Elements::Operation, "==================== OP_LIST =====================");
    bool needMark = false;
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(operations)) {
        needMark = true;
    }
    for (auto& op : operations) {
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

uint64_t OoOScheduler::ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype)
{
    uint64_t bytes = 0;
    if (shape.size() == DIM_FIVE) {
        bytes = BytesPerElement(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        bytes = CeilAlign(bytes, UB_BLOCK_SIZE);
    } else {
        uint64_t preDimSize = 1;
        uint64_t last2DimSize = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if ((shape.size() != 1) && (i < (shape.size() - LAST_TWO_DIM))) {
                preDimSize *= shape[i];
            } else {
                last2DimSize *= shape[i];
            }
        }
        bytes = preDimSize * CeilAlign(last2DimSize * BytesPerElement(dtype), UB_BLOCK_SIZE);
    }
    return bytes;
}

Status OoOScheduler::CheckAndUpdateLifecycle()
{
    for (const auto &op : orderedOps) {
        if (!opIsRetiredMap[op]) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", GetOpInfo(op).c_str(), GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (opIsAllocMap[op]) {
            op->GetOutputOperand(0)->memoryrange.lifeStart =
                localBufferMap[opReqMemIdsMap[op][0]]->startCycle;
            op->GetOutputOperand(0)->memoryrange.lifeEnd =
                localBufferMap[opReqMemIdsMap[op][0]]->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnCoreBlock(OpCoreType coreType, int idx, bool& didSpill)
{
    bool anyNotEmpty = false;
    for (auto& kv : allocIssueQueue[coreType][idx]) {
        if (!kv.second.Empty()) {
            anyNotEmpty = true;
            break;
        }
    }
    if (!anyNotEmpty) {
        return FAILED;
    }

    MemoryType spillMemType;
    if (!allocIssueQueue[coreType][idx][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!allocIssueQueue[coreType][idx][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        for (auto& memType : allocIssueQueue[coreType][idx]) {
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
    if (GenBufferSpill(allocIssueQueue[coreType][idx][spillMemType].Front()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    didSpill = true;
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock()
{
    bool didSpill = false;
    for (const auto& [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            if (SpillOnCoreBlock(coreType, idx, didSpill) != SUCCESS) {
                APASS_LOG_WARN_F(
                    Elements::Operation, "SpillOnBlock failed/skipped at idx: %d, coreType: %s", idx,
                    coreTypeToString(coreType).c_str());
            }
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
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
        return FAILED;
    }
    outTensor->memoryrange = TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
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
        if (localBufferMap.find(memId) == localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%d] %s --> %s.",
            memId, GetOpInfo(tensorOccupyMap[memType][memId]).c_str(), GetOpInfo(op).c_str());
        tensorOccupyMap[memType][memId] = op;
        outTensor->memoryrange =
            TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
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
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (auto& [pipeType, pipe] : issueQueues[coreType][idx]) {
                if (pipe.Empty() || pipe.busy) {
                    continue;
                }
                Operation* op = pipe.PopFront();
                //标注op的生命周期
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
                    APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRangeOp failed at idx: %d, coreType: %s. %s",
                        idx, coreTypeToString(coreType).c_str(), GetFormatBacktrace(*op).c_str());
                    return FAILED;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", GetOpInfo(op).c_str());
            }
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
        auto& corePair = opCoreLocationMap[op];
        auto& reqMemIds = opReqMemIdsMap[op];
        if (!bufferManagerMap[corePair.first][corePair.second][memType].IsFull(localBufferMap[reqMemIds[0]])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", GetOpInfo(op).c_str());
            if (bufferManagerMap[corePair.first][corePair.second][memType].Allocate(localBufferMap[reqMemIds[0]]) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", reqMemIds[0]);
                return FAILED;
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(memType, reqMemIds[0], false);
            }
            tensorOccupyMap[memType][reqMemIds[0]] = op;
            localBufferMap[reqMemIds[0]]->startCycle = clock;
            if (op->GetOutputOperand(0) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0]. %s", op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            newOperations_.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", GetOpInfo(op).c_str());
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed. %s", GetFormatBacktrace(*op).c_str());
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
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (auto& [memType, pipe] : allocIssueQueue[coreType][idx]) {
                if (pipe.Empty()) {
                    continue;
                }
                // 不断按顺序执行alloc指令，直到buffer被占满为止。
                if (ExecuteAllocIssue(commitCnt, memType, pipe) != SUCCESS) {
                    APASS_LOG_ERROR_F(
                        Elements::Operation, "ExecuteAllocIssue failed at idx: %d coreType: %s.", idx,
                        coreTypeToString(coreType).c_str());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(Operation* op)
{
    auto& reqMemIds = opReqMemIdsMap[op];
    for (auto memId : reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            auto corePair = tensorAllocCoreMap[memId];
            if (bufferManagerMap[corePair.first][corePair.second][localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(localBufferMap[memId]->memType, memId, true);
            }
            localBufferMap[memId]->retireCycle = clock;
            if (tensorOccupyMap[localBufferMap[memId]->memType].erase(localBufferMap[memId]->id) == 0) {
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

    auto& successors = GetSuccessors(op);
    auto& coreLocation = opCoreLocationMap[op];
    for (auto succOp : successors) {
        if (opIsRetiredMap[succOp]) {
            continue;
        }
        bool ready = true;
        auto &preds = GetPredecessors(succOp);
        for (auto predOp : preds) {
            if (!opIsRetiredMap[predOp]) {
                ready = false;
                break;
            }
        }
        if (ready) {
            issueQueues[coreLocation.first][coreLocation.second][opPipeTypeMap[succOp]].Insert(succOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d",
                GetOpInfo(succOp).c_str(), opExecOrderMap[succOp]);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireCoreIssue(OpCoreType coreType, int idx, uint64_t& commitCnt, int& nextCycle)
{
    for (auto& [pipeType, pipe] : issueQueues[coreType][idx]) {
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
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed at idx: %d coreType: %s! %s",
                    idx, coreTypeToString(coreType).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", pipe.curOpRetireCycle, GetOpInfo(pipe.curIssue).c_str());
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle)
{
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            if (RetireCoreIssue(coreType, idx, commitCnt, nextCycle) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed");
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue()
{
    // 初始化 Queue
    for (auto &op : orderedOps) {
        auto& coreLocation = opCoreLocationMap[op];
        auto coreType = coreLocation.first;
        auto idx = coreLocation.second;
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && GetPredecessors(op).empty()) {
            issueQueues[coreType][idx][opPipeTypeMap[op]].Insert(op);
        }
        if (opIsAllocMap[op]) {
            auto& reqMemIds = opReqMemIdsMap[op];
            if (!reqMemIds.empty()) {
                auto memType = localBufferMap[reqMemIds[0]]->memType;
                allocIssueQueue[coreType][idx][memType].Insert(op);
            }
        }
    }
}

Status OoOScheduler::ScheduleMainLoop()
{
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    numTotalIssues = orderedOps.size();
    uint64_t commitCnt = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        APASS_LOG_DEBUG_F(Elements::Operation, "     clock: %d", clock);
        // Retire Stage :
        // 检查现有pipe中的op是否执行完。如果op执行完，则将op标记为retired状态，将可以被释放的buffer释放掉，并唤醒后续已经就绪的op。
        // 完毕后更新整个pipe的状态。
        if (RetireIssueStage(commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed.");
            return FAILED;
        }
        // Buffer Allocation Stage :
        // 分配buffer。对于所有类型的buffer，按顺序执行alloc指令，并激活后续已经就绪的op。不断执行alloc直到buffer被占满为止。
        if (BufferAllocStage(commitCnt) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "BufferAllocStage failed.");
            return FAILED;
        }
        // Launch Stage ：检查idle的pipe中是否有已经就绪的指令。如果有，则执行该指令，并更新pipe的状态为busy。
        if (LaunchIssueStage(nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "LaunchIssueStage failed.");
            return FAILED;
        }
        if (numTotalIssues == commitCnt && nextCycle == -1) {
            isAllRetired = true;
            break;
        }
        // 如果nextCycle为-1，说明每个pipe都处于idle的状态，判断出现阻塞。需要spill调整内存
        if (nextCycle == -1) {
            if (SpillOnBlock() != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed.");
                return FAILED;
            }
        } else {
            clock = nextCycle;
        }
    }
    LOG_SCOPE_END(tScheduleMainLoop);
    return SUCCESS;
}

Status OoOScheduler::RetireIssue(Operation* op)
{
    opIsRetiredMap[op] = true;
    for (auto memId : opReqMemIdsMap[op]) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            // 加载时的核信息
            auto corePair = tensorAllocCoreMap[memId];
            if (bufferManagerMap[corePair.first][corePair.second][localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(Operation* op, size_t &pcIdx)
{
    if (localBufferMap.find(opReqMemIdsMap[op][0]) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", opReqMemIdsMap[op][0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = localBufferMap[opReqMemIdsMap[op][0]];
    auto corePair = opCoreLocationMap[op];
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].IsFull(allocBuffer)) {
        if (GenSpillOp(pcIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenSpillOp failed at ExecuteAllocIssue. %s", GetFormatBacktrace(*orderedOps[pcIdx]).c_str());
            return FAILED;
        }
    }
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
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
    if (InitBufRefCount() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
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
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
                IssueQueue queue;
                queue.SetCompareFunc(compareFunc);
                issueQueues[coreType][idx][static_cast<PipeType>(i)] = queue;
            }
        }
    }

    bufferManagerMap.clear();
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            if (coreType == OpCoreType::AIV) {
                IssueQueue queue;
                queue.SetCompareFunc(compareFunc);
                allocIssueQueue[coreType][idx][MemoryType::MEM_UB] = queue;
                bufferManagerMap[coreType][idx].insert({MemoryType::MEM_UB,
                        BufferPool(MemoryType::MEM_UB, localMemorySize[MemoryType::MEM_UB])});
                continue;
            }
            for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
                IssueQueue queue;
                queue.SetCompareFunc(compareFunc);
                allocIssueQueue[coreType][idx][static_cast<MemoryType>(i)] = queue;
                if (localMemorySize.find(static_cast<MemoryType>(i)) != localMemorySize.end()) {
                    bufferManagerMap[coreType][idx].insert(
                        {static_cast<MemoryType>(i),
                         BufferPool(static_cast<MemoryType>(i), localMemorySize[static_cast<MemoryType>(i)])});
                }
            }
        }
    }
}

void OoOScheduler::UpdateAllocMap(Operation* op, std::map<int, Operation*> &tensorAllocMap)
{
    for (auto outTensor : op->GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = op;
        }
    }
    for (auto inTensor : op->GetIOperands()) {
        if (inTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = inTensor->memoryrange.memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = op;
        }
    }
}

Status OoOScheduler::CheckAllocIssue()
{
    std::map<int, Operation*> tensorAllocMap;
    for (const auto &op : orderedOps) {
        if (opIsAllocMap[op]) {
            if (opReqMemIdsMap[op].size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "ALLOC[%d] reqMemIds size not equal to 0. %s",
                    op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            UpdateAllocMap(op, tensorAllocMap);
        }
    }
    for (const auto &op : orderedOps) {
        if (!opIsAllocMap[op]) {
            UpdateAllocMap(op, tensorAllocMap);
        }
    }
    for (auto tensorAlloc : tensorAllocMap) {
        if (!opIsAllocMap[tensorAlloc.second]) {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s Tensor[%d] is missing Alloc.",
                GetOpInfo(tensorAlloc.second).c_str(), tensorAlloc.first);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::InitLocalBuffer(LogicalTensorPtr oOperand, int memId)
{
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return SUCCESS;
    }
    if (static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()) !=
        ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->tensor->datatype)) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "InitLocalBuffer Failed at ShapeCeilAlign! "
            "Please ensure that the rawTensor[%d] shapes are aligned.",
            oOperand->GetRawMagic());
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] =
            std::make_shared<LocalBuffer>(memId, oOperand->tensor->GetRawDataSize(), oOperand->GetMemoryTypeOriginal());
    } else {
        localBufferMap[memId]->size =
            std::max(localBufferMap[memId]->size, static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()));
    }
    return SUCCESS;
}

void OoOScheduler::UpdateBufRefCount(Operation* op, LogicalTensorPtr tensor)
{
    int memId = tensor->memoryrange.memId;
    if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
        bufRefCount_[memId]++;
        opReqMemIdsMap[op].push_back(memId);
    }
}

Status OoOScheduler::InitBufRefCount()
{
    bufRefCount_.clear();
    depManager_.ClearDependencies();
    for (const auto &op : orderedOps) {
        opIsRetiredMap[op] = false;
        opReqMemIdsMap[op].clear();
        for (auto &tensor : op->GetIOperands()) {
            UpdateBufRefCount(op, tensor);
            int memId = tensor->memoryrange.memId;
            if (InitLocalBuffer(tensor, memId) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                return FAILED;
            }
        }
        for (auto &tensor : op->GetOOperands()) {
            UpdateBufRefCount(op, tensor);
            int memId = tensor->memoryrange.memId;
            if (InitLocalBuffer(tensor, memId) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap) {
    for (auto tensor : tensors) {
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        const auto& shape = tensor->tensor->GetRawShape();
        if (std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d <= 0; })) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "Dynamic axis detected in %s, "
                "OoOSchedule requires static rawShape.",
                tensor->Dump().c_str());
            return FAILED;
        }
        if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
            bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->tensor->GetRawDataSize();
            memIdMap.insert(tensor->memoryrange.memId);
        }
    }
    return SUCCESS;
}

std::string OoOScheduler::DumpOpInfo(Operation &op)
{
    std::ostringstream oss;
    oss << "OP: " << op.GetOpcodeStr().c_str() << "[" << op.GetOpMagic() << "] | ";
    oss << "Inputs: {";
    for (size_t i = 0; i < op.iOperand.size(); i++) {
        oss << "RawTensor[" << op.GetInputOperand(i)->tensor->GetRawMagic() << "] ";
        oss << op.iOperand[i]->tensor->DumpSSA(true, true);
        if (i != op.iOperand.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}"
        << " | ";
    oss << "Outputs: {";
    for (size_t i = 0; i < op.oOperand.size(); i++) {
        oss << "RawTensor[" << op.GetOutputOperand(i)->tensor->GetRawMagic() << "] ";
        oss << op.oOperand[i]->tensor->DumpSSA(true, true);
        if (i != op.oOperand.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

Status OoOScheduler::CheckOpBufferSize(Operation* op)
{
    std::map<MemoryType, int64_t> bufferSize;
    std::set<int> memIdMap;
    if (CalcBufferSize(op->GetIOperands(), bufferSize, memIdMap) != SUCCESS ||
        CalcBufferSize(op->GetOOperands(), bufferSize, memIdMap) != SUCCESS) {
        return FAILED;
    }
    for (auto& buffer : bufferSize) {
        if (localMemorySize.find(buffer.first) == localMemorySize.end()) {
            continue;
        }
        if (buffer.second <= localMemorySize[buffer.first]) {
            continue;
        }
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation, "Alloc tensor[%d] size[%ld] exceeds %s size[%ld]! %s",
                op->GetOutputOperand(0)->GetMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                localMemorySize[buffer.first], GetFormatBacktrace(*op).c_str());
            APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] producer info:", op->GetOutputOperand(0)->GetMagic());
            for (auto producer : op->GetOutputOperand(0)->GetProducers()) {
                if (producer == op) {
                    continue;
                }
                APASS_LOG_ERROR_F(Elements::Operation, "    %s.", DumpOpInfo(*producer).c_str());
            }
        } else {
            APASS_LOG_ERROR_F(
                Elements::Operation, "OP %s[%d] in/output total size[%ld] exceeds %s size[%ld]!",
                op->GetOpcodeStr().c_str(), op->GetOpMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                localMemorySize[buffer.first]);
            APASS_LOG_ERROR_F(Elements::Operation, "%s.", DumpOpInfo(*op).c_str());
        }
        return FAILED;
    }
    return SUCCESS;
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

void OoOScheduler::InitCoreConfig(const std::vector<Operation *> &operations)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(operations)) {
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

void OoOScheduler::InitOpViewOps(Operation* op) {
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

Status OoOScheduler::InitOpCoreType(Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
    if (op == nullptr) return FAILED;

    std::pair<OpCoreType, int> coreLocation;
    if (!opCoreMap.empty()) {
        coreLocation = opCoreMap.at(op);
    } else if (op->GetCoreType() == CoreType::AIC) {
        coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
    } else if (op->GetCoreType() == CoreType::AIV) {
        coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
    } else {
        // 对 ANY 类型进行处理
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
            coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            if (op->GetIOperands().size() == 0 || op->GetInputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
                coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
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

Status OoOScheduler::InitOpEntry(Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
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
    opReqMemIdsMap[op] = std::vector<int>();

    // 初始化viewOps
    InitOpViewOps(op);

    // 初始化核属性
    if (InitOpCoreType(op, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation %s init coreType failed!", GetOpInfo(op).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "issue: %s, coreType: %s, idx: %d",
        GetOpInfo(op).c_str(), coreTypeToString(opCoreLocationMap[op].first).c_str(), opCoreLocationMap[op].second);
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation *> &operations, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap,
    const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig)
{
    orderedOps.clear();
    opExecOrderMap.clear();
    opPipeTypeMap.clear();
    opIsAllocMap.clear();
    opIsRetiredMap.clear();
    opReqMemIdsMap.clear();
    opViewOpsMap.clear();
    opCoreLocationMap.clear();
    localBufferMap.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    localMemorySize = CommonUtils::GetLocalMemorySize();
    if (fixCoreConfig.empty()) {
        InitCoreConfig(operations);
    } else {
        CORE_INIT_CONFIGS = fixCoreConfig;
    }
    // 校验并初始化Operation
    depManager_.ClearDependencies();
    for (const auto &op : operations) {
        if (InitOpEntry(op, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init issue failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    numTotalIssues = orderedOps.size();

    if (InitBufRefCount() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 初始化依赖关系
    if (depManager_.InitDependencies(orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    depManager_.PrintDependencies(orderedOps);
    if (CheckAllocIssue() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocIssue failed!");
        return FAILED;
    }
    InitTensorCoreMap();
    // 初始化内存管理器
    InitIssueQueuesAndBufferManager();
    LOG_SCOPE_END(tInit);
    return SUCCESS;
}

Status OoOScheduler::Schedule(
    const std::vector<Operation*>& operations,
    const std::unordered_map<Operation*, std::pair<OpCoreType, int>>& opCoreMap,
    const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig)
{
    if (operations.empty()) {
        return SUCCESS;
    }
    PrintOpList(operations);
    if (Init(operations, opCoreMap, fixCoreConfig) != SUCCESS) {
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
    for (size_t i = 0; i < operations.size(); i++) {
        if (operations[i]->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
            auto l0MxOut = operations[i]->GetOOperands()[0];
            auto consOp = *l0MxOut->GetConsumers().begin();
            LogicalTensorPtr l0ATensor, l0BTensor, l0AMXTensor, l0BMXTensor;
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
    for (auto& entry : l02L0MXMap_) {
        auto l0Tensor = entry.first;
        auto l0MXTensor = entry.second;
        int l0MemID = l0Tensor->memoryrange.memId;
        int l0MemMXID = l0MXTensor->memoryrange.memId;
        l0MXTensor->memoryrange =
            TileRange(localBufferMap[l0MemID]->start >> 4, localBufferMap[l0MemID]->end >> 4, l0MemMXID);
    }
    PrintOpList(newOperations_);
    function_.SetStackWorkespaceSize(workspaceOffset);
    function_.pipeEndTime = pipeEndTime;
    return SUCCESS;
}

// UpdateRemainOpBufId函数不能直接用
Status OoOScheduler::UpdateMemId(int oldMemId, int newMemId)
{
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = 0;
    for (auto &op : orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        auto& reqMemIds = opReqMemIdsMap[op];
        for (auto& memId : reqMemIds) {
            if (memId == oldMemId) {
                memId = newMemId;
                bufRefCount_[oldMemId] -= 1;
                bufRefCount_[newMemId] += 1;
            }
        }
        for (auto &outTensor : op->GetOOperands()) {
            if (outTensor->memoryrange.memId == oldMemId) {
                outTensor->memoryrange.memId = newMemId;
            }
        }
    }
    if (bufRefCount_[oldMemId] != 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "oldMemId %d bufRefCount is not 0, UpdateMemId failed.", oldMemId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::UpdateMoveOpAttr(Operation& moveOp, Operation& occupyOp)
{
    if (moveOp.GetOpcode() == Opcode::OP_COPY_IN && occupyOp.GetOpcode() == Opcode::OP_COPY_IN) {
        moveOp.SetOpAttribute(occupyOp.GetOpAttribute()->Clone());
        moveOp.inParamLocation_ = occupyOp.inParamLocation_;
        moveOp.SetIOpAttrOffset(0, occupyOp.GetIOpAttrOffset(0));
    } else if (moveOp.GetOpcode() == Opcode::OP_ADDS) {
        moveOp.SetAttr(OpAttributeKey::scalar, Element(DataType::DT_UINT64, 0));
        if (moveOp.GetIOperands()[0]->tensor->rawshape.back() == 1) {
            std::vector<bool> attrIn{true};
            moveOp.SetAttr(OpAttributeKey::inputCombineAxis, attrIn);
        }
        if (moveOp.GetOOperands()[0]->tensor->rawshape.back() == 1) {
            std::vector<bool> attrOut{true};
            moveOp.SetAttr(OpAttributeKey::outputCombineAxis, attrOut);
        }
    }
    if (occupyOp.GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        moveOp.UpdateInternalSubgraphID(occupyOp.GetInternalSubgraphID());
        moveOp.SetAIVCore(occupyOp.GetAIVCore());
    }
}

void OoOScheduler::ProcessMoveIssue(Operation* moveOp, Operation* allocOp, MemoryType memType, int oldMemId, int newMemId) 
{
    opReqMemIdsMap[moveOp] = {oldMemId, newMemId};
    opIsRetiredMap[moveOp] = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Add MOVEOP: %s.", GetOpInfo(moveOp).c_str());
    tensorOccupyMap[memType][newMemId] = moveOp;
    // 更新moveIssue的相关信息
    auto occupyOp = tensorOccupyMap[memType][oldMemId];
    depManager_.AddDependency(occupyOp, moveOp);
    // 更新执行序
    SetExecOrder(moveOp, GetExecOrder(allocOp));
    InsertOrdered(moveOp);
    // 找出moveFromTensor的所有consumer中未执行的, 并改变图的连接
    UpdateReloadIssueDepend(moveOp, occupyOp, oldMemId);
}

Status OoOScheduler::FindMoveFromTensor(
    Operation& occupyOp, int oldMemId, MemoryType memType, bool& rearrangeUBBF16, LogicalTensorPtr& moveFromTensor)
{
    for (auto outTensorPtr : occupyOp.GetOOperands()) {
        if (outTensorPtr->memoryrange.memId == oldMemId) {
            moveFromTensor = outTensorPtr;
            break;
        }
    }
    if (moveFromTensor == nullptr) {
        APASS_LOG_WARN_F(
            Elements::Tensor, "Cannot find tensor(memId: %d) according to tensorOccupyMap, GenRearrangeCopyOp failed",
            oldMemId);
        return FAILED;
    }
    // 如果moveFrom Tensor是UB且数据类型为bf16, rearrange失败
    if (memType == MemoryType::MEM_UB && moveFromTensor->Datatype() == DataType::DT_BF16) {
        APASS_LOG_ERROR_F(
            Elements::Tensor, "Cannot rearrange UB tensor with datatype bf16, do schedulemainloop spill.");
        rearrangeUBBF16 = true;
    }
    return SUCCESS;
}

Status OoOScheduler::GetMoveOpInTensor(
    Opcode moveOpcode, Operation& occupyOp, LogicalTensorPtr& inTensor, LogicalTensorPtr& moveFromTensor)
{
    if (moveOpcode == Opcode::OP_COPY_IN) {
        if (occupyOp.GetOpcode() != Opcode::OP_COPY_IN) {
            APASS_LOG_WARN_F(Elements::Operation, "Occupy op is not COPY_IN, GetMoveOpInTensor failed.");
            return FAILED;
        }
        inTensor = occupyOp.GetIOperands()[0];
        if (inTensor == nullptr || inTensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_WARN_F(Elements::Tensor, "inTensor is illegal, GetMoveOpInTensor failed.");
            return FAILED;
        }
    } else {
        inTensor = moveFromTensor;
    }
    return SUCCESS;
}

Status OoOScheduler::GenRearrangeCopyOp(Operation* allocOp, MemoryType memType, int oldMemId, int &newMemId, bool &rearrangeUBBF16) {
    if (memType != MemoryType::MEM_L1 && memType != MemoryType::MEM_UB) {
        APASS_LOG_WARN_F(Elements::Tensor, "Unexpected rearrange tensor memory type found, GenRearrangeCopyOp failed.");
        return FAILED;
    }
    Opcode moveOpcode = memType == MemoryType::MEM_L1 ? Opcode::OP_COPY_IN : Opcode::OP_ADDS;
    auto occupyOp = tensorOccupyMap[memType][oldMemId];
    LogicalTensorPtr moveFromTensor{nullptr};
    if (FindMoveFromTensor(*occupyOp, oldMemId, memType, rearrangeUBBF16, moveFromTensor) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "GenRearrangeCopyOp failed at FindMoveFromTensor.");
        return FAILED;
    }
    if (rearrangeUBBF16) {
        return SUCCESS;
    }
    LogicalTensorPtr moveToTensor =
        std::make_shared<LogicalTensor>(function_, moveFromTensor->Datatype(), moveFromTensor->shape);
    // 给moveToTensor分配memId和创建新的localbuffer
    if (UpdateTensorAttr(moveToTensor, memType, moveFromTensor, oldMemId) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "GenRearrangeCopyOp failed at UpdateTensorAttr.");
        return FAILED;
    }
    newMemId = moveToTensor->memoryrange.memId;
    LogicalTensorPtr inTensor{nullptr};
    if (GetMoveOpInTensor(moveOpcode, *occupyOp, inTensor, moveFromTensor) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "GenRearrangeCopyOp failed at GetMoveOpInTensor.");
        return FAILED;
    }
    auto &moveOp = (moveOpcode == Opcode::OP_COPY_IN && occupyOp->GetOpcode() == Opcode::OP_COPY_IN) ?
        occupyOp->CloneOperation(function_, {inTensor}, {moveToTensor}) :
        function_.AddRawOperation(moveOpcode, {inTensor}, {moveToTensor});
    moveToTensor->tensor->rawshape = inTensor->tensor->rawshape;
    newOperations_.push_back(&moveOp);
    // UpdateMoveOpAttr
    UpdateMoveOpAttr(moveOp, *occupyOp);
    // 设置coreLocation
    opCoreLocationMap[&moveOp] = tensorAllocCoreMap[newMemId];
    // 处理issue & 改变图连接
    ProcessMoveIssue(&moveOp, allocOp, memType, oldMemId, newMemId);
    // 更新memId
    if (UpdateMemId(oldMemId, newMemId) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "GenRearrangeCopyOp failed at UpdateMemId. %s", GetFormatBacktrace(moveOp).c_str());
        return FAILED;
    }
    // Free oldMemId
    auto corePair = tensorAllocCoreMap[oldMemId];
    bufferManagerMap[corePair.first][corePair.second][memType].Free(oldMemId);
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(memType, oldMemId, true);
    }
    localBufferMap[oldMemId]->retireCycle = clock;
    if (tensorOccupyMap[memType].erase(oldMemId) == 0) {
        APASS_LOG_WARN_F(Elements::Tensor, "Erase tensor[%d] failed", oldMemId);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool& bufferManager)
{
    auto moveToBufferPtr = localBufferMap[newMemId];
    if (bufferManager.ModifyBufferRange(moveToBufferPtr, offset) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "UpdateRange failed at ModifyBufferRange.");
        return FAILED;
    }
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(memType, newMemId, false);
    }
    tensorOccupyMap[memType][newMemId]->GetOOperands()[0]->memoryrange =
        TileRange(offset, offset + moveToBufferPtr->size, newMemId);
    localBufferMap[newMemId]->startCycle = clock;
    return SUCCESS;
}

Status OoOScheduler::RearrangeBuffers(Operation* op, bool isGenSpillStage, bool &rearrangeUBBF16)
{
    LocalBufferPtr allocBuffer = localBufferMap[opReqMemIdsMap[op][0]];
    auto corePair = opCoreLocationMap[op];
    BufferPool &bufferManager = bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType];
    auto rearrangeScheme = GetRearrangeScheme(bufferManager, allocBuffer->size);
    if (rearrangeScheme.cost == INT_MAX) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffers failed at GetRearrangeScheme.");
        return FAILED;
    }
    // 修改tensor对应的localbuffer
    for (auto& [memId, offset] : rearrangeScheme.orderedMoveTo) {
        auto targetBufferPtr = localBufferMap[memId];
        if (rearrangeScheme.moveFrom[memId] != targetBufferPtr->start ||
            rearrangeScheme.memSizeMap[memId] != targetBufferPtr->size) {
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "MemId %d localBuffer and rearrangeScheme range donot match, RearrangeBuffers failed.", memId);
            return FAILED;
        }
        Operation* occupyOp = GetSpillIssue(op, memId, isGenSpillStage);
        if (occupyOp == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "OccupyOp is nullptr, RearrangeBuffers failed. %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (IsViewOp(*occupyOp) || occupyOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            APASS_LOG_WARN_F(Elements::Operation, "Target rearrange tensor(memId: %d)'s occupy op is %d %s, RearrangeBuffers failed. %s",
                memId, op->GetOpMagic(), op->GetOpcodeStr().c_str(), GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        // GenSpillStage阶段的内存整理不需要插入搬运节点
        // ScheduleMainLoop阶段如果是alloc占有的tensor不需要插入搬运节点
        if (isGenSpillStage || occupyOp->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            if (bufferManager.ModifyBufferRange(targetBufferPtr, offset) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Tensor, "RearrangeBuffers failed at ModifyBufferRange.");
                return FAILED;
            }
        } else {
            int newMemId = INT_MAX;
            if (GenRearrangeCopyOp(op, allocBuffer->memType, memId, newMemId, rearrangeUBBF16) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffers failed at GenRearrangeCopyOp.");
                return FAILED;
            }
            if (rearrangeUBBF16) {
                return FAILED;
            }
            // 更新moveToTensor的localbuffer和bufferslice range
            if (UpdateRange(newMemId, offset, allocBuffer->memType, bufferManager) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffers failed at UpdateRange.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
