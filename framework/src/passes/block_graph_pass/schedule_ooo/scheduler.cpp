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

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

constexpr int64_t MAX_L0A_SIZE = 64 * 1024;
constexpr int64_t MAX_L0C_SIZE = 128 * 1024;
constexpr int64_t MAX_BT_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_QUANT_PRE_SIZE = 1 * 2048;
constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

inline std::string coreTypeToString(OpCoreType coreType) {
    switch (coreType) {
        case OpCoreType::AIV: return "AIV";
        case OpCoreType::AIC: return "AIC";
        default: return "MEM_UNKNOWN";
    }
}

inline bool IsMixGraph(const std::vector<Operation*> &operations) {
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

inline Operation* SkipViewChain(Operation* start, bool followProducers) {
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

IssueEntry::IssueEntry(Operation &op, uint64_t issueId)
    : tileOp(op), id(issueId), execOrder(issueId), type(RescheduleUtils::GetOpPipeType(&op)) {
    if (tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
        isAlloc = true;
    }
    for (auto iOperand : op.GetIOperands()) {
        for (auto pre : iOperand->GetProducers()) {
            while (IsViewOp(*pre) && pre->GetOutputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                viewOps.push_back(pre);
                pre = *(pre->GetInputOperand(0)->GetProducers().begin());
            }
        }
    }
}

int IssueEntry::GetOOperandIdx(int curMemId) {
    for (size_t i = 0; i < tileOp.GetOOperands().size(); i++) {
        if (tileOp.GetOOperands()[i]->memoryrange.memId == curMemId) {
            return i;
        }
    }
    return -1;
}

void IssueEntry::Clear() {
    isRetired = false;
    predecessors.clear();
    successors.clear();
    reqMemIds.clear();
}

void IssueEntry::UpdateTensorInput(std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const {
    for (size_t index = 0; index < tileOp.GetIOperands().size(); index++) {
        UpdateTensorInputForOperand(index, spillSrcIssue, tensor);
    }
}

void IssueEntry::UpdateTensorInputForOperand(size_t index, std::shared_ptr<IssueEntry> &spillSrcIssue,
    LogicalTensorPtr tensor) const {
    for (auto &inOp : tileOp.GetIOperands()[index]->GetProducers()) {
        if (IsViewOp(*inOp)) {
            Operation* op = SkipViewChain(inOp, true);
            UpdateTensorInputForView(*op, spillSrcIssue, tensor);
        } else if (inOp == &(spillSrcIssue->tileOp)) {
            tileOp.UpdateInputOperand(index, tensor);
        }
    }
}

void IssueEntry::UpdateTensorInputForView(Operation& op,
    std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const {
    bool hit = false;
    for (auto it : op.GetInputOperand(0)->GetProducers()) {
        if (it == &(spillSrcIssue->tileOp)) {
            hit = true;
            op.UpdateInputOperand(0, tensor);
            break;
        }
    }
    if (!hit) return;
    // 向后刷该View链路上的MemId
    for (Operation* p = &op; p != nullptr && IsViewOp(*p); ) {
        p->GetOutputOperand(0)->memoryrange.memId = tensor->memoryrange.memId;
        auto consumers = p->GetOutputOperand(0)->GetConsumers();
        if (consumers.empty()) break;
        p = *consumers.begin();
    }
}

std::string IssueEntry::GetOpInfo() {
    return tileOp.GetOpcodeStr() + "[" + std::to_string(tileOp.GetOpMagic()) + "]";
}

Status OoOScheduler::PrintSpillFailedInfo(IssueEntryPtr allocIssue) {
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s", 
        MemoryTypeToString(localBufferMap[allocIssue->reqMemIds[0]]->memType).c_str(), GetFormatBacktrace(allocIssue->tileOp).c_str());
    if (localBufferMap[allocIssue->reqMemIds[0]]->memType == MemoryType::MEM_L1) {
        APASS_LOG_ERROR_F(Elements::Operation, "A5 is not support L1 spill.");
    }
    if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s alloc buffer size: %lu. %s", allocIssue->GetOpInfo().c_str(), 
            localBufferMap[allocIssue->reqMemIds[0]]->size, GetFormatBacktrace(allocIssue->tileOp).c_str());
    }
    auto bufferSlices = bufferManagerMap[allocIssue->coreLocation.first][allocIssue->coreLocation.second][localBufferMap[allocIssue->reqMemIds[0]]->memType].GetBufferSlices();
    for (auto memId : bufferSlices) {
        auto occupyIssue = GetBufLastWriteIssue(allocIssue, memId);
        if (occupyIssue == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write time.", memId);
            return FAILED;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "%s, range[%lu, %lu], Tensor[%d] size: %lu.", occupyIssue->GetOpInfo().c_str(), 
            localBufferMap[memId]->start, localBufferMap[memId]->end, memId, localBufferMap[memId]->size);
    }
    return SUCCESS;
}

void OoOScheduler::PrintSpillFailedInfo(IssueEntryPtr allocIssue, MemoryType bufferType) {
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s", MemoryTypeToString(bufferType).c_str(), GetFormatBacktrace(allocIssue->tileOp).c_str());
    if (bufferType == MemoryType::MEM_L1) {
        APASS_LOG_ERROR_F(Elements::Operation, "A5 is not support L1 spill.");
    }
    if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
        APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", allocIssue->GetOpInfo().c_str(),
            localBufferMap[allocIssue->reqMemIds[0]]->size, GetFormatBacktrace(allocIssue->tileOp).c_str());
    }
    if (tensorOccupyMap.find(bufferType) != tensorOccupyMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
        for (auto occupyIssue : tensorOccupyMap[bufferType]) {
            APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d], size: %lu, range[%lu, %lu], last writer:%s. %s", occupyIssue.first, localBufferMap[occupyIssue.first]->size,
                localBufferMap[occupyIssue.first]->start, localBufferMap[occupyIssue.first]->end, 
                occupyIssue.second->GetOpInfo().c_str(), GetFormatBacktrace(occupyIssue.second->tileOp).c_str());
        }
    }
}

void OoOScheduler::PrintDependencies() {
    for (const auto &issue : issueEntries) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%s, latency: %d.", issue->GetOpInfo().c_str(), issue->tileOp.GetLatency());
        for (const auto &preId : issue->predecessors) {
            auto pre = issueEntryMap[preId];
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", pre->GetOpInfo().c_str());
        }
        for (const auto &succId : issue->successors) {
            auto successor = issueEntryMap[succId];
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", successor->GetOpInfo().c_str());
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "\n");
    }
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

Status OoOScheduler::DelBufRefCount(const int memId) {
    if (bufRefCount.find(memId) == bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount[memId]--;
    if (bufRefCount[memId] < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::PrintOpList(std::vector<Operation *> operations) {
    APASS_LOG_INFO_F(Elements::Operation, "==================== OP_LIST =====================");
    for (auto &op : operations) {
        bool isCubeComponent = op->HasAttribute(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube);
        if (!isCubeComponent) {
            op->SetAIVCore(AIVCore::AIV0);
        }
        if (!op->oOperand.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d], range[%zu, %zu]", op->GetOpcodeStr().c_str(), 
                op->GetOpMagic(), op->oOperand[0]->memoryrange.start, op->oOperand[0]->memoryrange.end);
        } else {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic()); 
        }
    }
}

void OoOScheduler::InsertIssueEntries(IssueEntryPtr insertIssue) {
    auto it = issueEntries.begin();
    for (; it != issueEntries.end(); it++) {
        if ((*it)->execOrder >= insertIssue->execOrder) {
            break;
        }
    }
    auto insertPos = issueEntries.insert(it++, insertIssue);
    for (auto adjustIt = insertPos + 1; adjustIt != issueEntries.end(); adjustIt++) {
        if ((*adjustIt)->execOrder >= insertIssue->execOrder) {
            (*adjustIt)->execOrder++;
        }
    }
}

void OoOScheduler::UpdateIssueExecOrder() {
    for (size_t idx = 0; idx < issueEntries.size(); idx++) {
        issueEntries[idx]->execOrder = idx;
    }
}

uint64_t OoOScheduler::ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype) {
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

Status OoOScheduler::CheckAndUpdateLifecycle() {
    for (const auto &issue : issueEntries) {
        if (!issue->isRetired) { 
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", issue->GetOpInfo().c_str(), GetFormatBacktrace(issue->tileOp).c_str()); 
            return FAILED; 
        }
        if (issue->isAlloc) {
            issue->tileOp.GetOutputOperand(0)->memoryrange.lifeStart =
                localBufferMap[issue->reqMemIds[0]]->startCycle;
            issue->tileOp.GetOutputOperand(0)->memoryrange.lifeEnd =
                localBufferMap[issue->reqMemIds[0]]->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnCoreBlock(OpCoreType coreType, int idx) {
    MemoryType spillMemType;
    if (!allocIssueQueue[coreType][idx][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510  && !allocIssueQueue[coreType][idx][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        for (auto memType: allocIssueQueue[coreType][idx]) {
            if (memType.second.Empty()) {
                continue;
            }
            PrintSpillFailedInfo(memType.second.Front(), memType.first);
        }
        APASS_LOG_WARN_F(Elements::Operation, "Buffer[L1/L0A/B/C] is Full. Possible causes: incorrect memory reuse, memory fragmentation, lack alloc. "
            "Please check tile shape, OOO spill failed info and alloc location.");
        return FAILED;
    }
    if (GenBufferSpill(allocIssueQueue[coreType][idx][spillMemType].Front()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock() {
    bool flag = false;
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            if (SpillOnCoreBlock(coreType, idx) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "SpillOnBlock failed at idx: %d, coreType: %s", idx, coreTypeToString(coreType).c_str());
            } else {
                flag = true;
            }
        }
    }
    if (!flag) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at all coreType.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AllocViewTensorMemRange(Operation &operation) {
    auto outTensor = operation.GetOOperands()[0];
    int memId = outTensor->memoryrange.memId;
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
        return FAILED;
    }
    outTensor->memoryrange =
            TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
    return SUCCESS;
}

Status OoOScheduler::AllocTensorMemRange(IssueEntryPtr issue) {
    for (auto& op : issue->viewOps) {
        if (!IsViewOp(*op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "op[%s] is not OP_VIEW.", op->GetOpMagic());
            return FAILED;
        }
        if (AllocViewTensorMemRange(*op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AllocViewTensorMemRange failed.");
            return FAILED;
        }
    }
    for (auto& outTensor : issue->tileOp.GetOOperands()) {
        MemoryType memType = outTensor->GetMemoryTypeOriginal();
        if (memType == MemoryType::MEM_DEVICE_DDR) {
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
                            MemoryTypeToString(memType).c_str(), GetFormatBacktrace(issue->tileOp).c_str());
            return FAILED;
        }
        if (localBufferMap.find(memId) == localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%u] %s --> %s.", 
            memId, tensorOccupyMap[memType][memId]->GetOpInfo().c_str(), issue->GetOpInfo().c_str());
        tensorOccupyMap[memType][memId] = issue;
        outTensor->memoryrange =
            TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
    }
    return SUCCESS;
}

void OoOScheduler::HandleViewOp(IssueEntryPtr issue) {
    for (auto& op : issue->viewOps) {
        if (std::find(newOperations_.begin(), newOperations_.end(), op) != newOperations_.end()) {
            continue;
        }
        newOperations_.emplace_back(op);
    }
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle) {
    // issue from all pipes
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (auto &[pipeType, pipe] : issueQueues[coreType][idx]) {
                if (pipe.Empty() || pipe.busy) {
                    continue;
                }
                IssueEntryPtr issue = pipe.PopFront();
                pipe.busy = true;
                pipe.curIssue = issue;
                pipe.curOpRetireCycle = clock + issue->tileOp.GetLatency();
                oooCheck.pipeUsageCount[pipeType] += issue->tileOp.GetLatency();
                HandleViewOp(issue);
                newOperations_.emplace_back(&(issue->tileOp));
                if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                    nextCycle = pipe.curOpRetireCycle;
                }
                if (AllocTensorMemRange(issue) != SUCCESS) {
                    APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRange failed at idx: %d, coreType: %s. %s",
                        idx, coreTypeToString(coreType).c_str(), GetFormatBacktrace(issue->tileOp).c_str());
                    return FAILED;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", issue->GetOpInfo().c_str());
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType, IssueQueue &pipe) {
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        IssueEntryPtr issue = pipe.Front();
        auto corePair = issue->coreLocation;
        if (!bufferManagerMap[corePair.first][corePair.second][memType].IsFull(localBufferMap[issue->reqMemIds[0]])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", issue->GetOpInfo().c_str());
            if (bufferManagerMap[corePair.first][corePair.second][memType].Allocate(localBufferMap[issue->reqMemIds[0]]) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", issue->reqMemIds[0]); 
                return FAILED; 
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(memType, issue->reqMemIds[0], false);
            }
            tensorOccupyMap[memType][issue->reqMemIds[0]] = issue;
            localBufferMap[issue->reqMemIds[0]]->startCycle = clock;
            if (issue->tileOp.GetOutputOperand(0) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0]. %s", issue->tileOp.GetOpMagic(), GetFormatBacktrace(issue->tileOp).c_str());
                return FAILED;
            }
            newOperations_.push_back(&(issue->tileOp));
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", issue->GetOpInfo().c_str());
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { 
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed. %s", GetFormatBacktrace(issue->tileOp).c_str()); 
                return FAILED; 
            }
        } else {
            canAlloc = false;
            break;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t &commitCnt) {
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (auto& [memType, pipe] : allocIssueQueue[coreType][idx]) {
                if (pipe.Empty()) {
                    continue;
                }
                // 不断按顺序执行alloc指令，直到buffer被占满为止。
                if (ExecuteAllocIssue(commitCnt, memType, pipe) != SUCCESS) {
                    APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed at idx: %d coreType: %s.", idx, coreTypeToString(coreType).c_str());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(IssueEntryPtr issue) {
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId); 
            return FAILED; 
        }
        if (bufRefCount[memId] == 0) {
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

Status OoOScheduler::RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt) {
    commitCnt++;
    issue->isRetired = true;
    if (FreeBuffer(issue) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBuffer failed. %s", GetFormatBacktrace(issue->tileOp).c_str());
        return FAILED;
    }

    for (auto succId : issue->successors) {
        auto succ = issueEntryMap[succId];
        if (succ->isRetired) {
            continue;
        }
        bool ready = true;
        for (auto predId : succ->predecessors) {
            auto pred = issueEntryMap[predId];
            if (!pred->isRetired) {
                ready = false;
            }
        }
        if (ready) {
            auto corePair = issue->coreLocation;
            issueQueues[corePair.first][corePair.second][succ->type].Insert(succ);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d", succ->GetOpInfo().c_str(), succ->execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireCoreIssue(OpCoreType coreType, int idx, uint64_t& commitCnt, int& nextCycle) {
    for (auto& [pipeType, pipe] : issueQueues[coreType][idx]) {
        (void)pipeType;
        if (!pipe.busy) {
            continue;
        }
        if (pipe.curOpRetireCycle <= clock) {   // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            IssueEntryPtr issue = pipe.curIssue;
            pipe.busy = false;
            pipe.curIssue = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", issue->GetOpInfo().c_str());
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed at idx: %d coreType: %s! %s",
                    idx, coreTypeToString(coreType).c_str(), GetFormatBacktrace(issue->tileOp).c_str());
                return FAILED;
            }
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%ld]: %s", pipe.curOpRetireCycle, pipe.curIssue->GetOpInfo().c_str());
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle) {
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            if (RetireCoreIssue(coreType, idx, commitCnt, nextCycle) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed");
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue() {
    // 初始化 Queue
    for (auto &issue : issueEntries) {
        auto coreType = issue->coreLocation.first;
        auto idx = issue->coreLocation.second;
        if (USE_LESS_OPS.find(issue->tileOp.GetOpcode()) != USE_LESS_OPS.end() && issue->predecessors.empty()) {
            issueQueues[coreType][idx][issue->type].Insert(issue);
        }
        if (issue->isAlloc) {
            auto memType = localBufferMap[issue->reqMemIds[0]]->memType;
            allocIssueQueue[coreType][idx][memType].Insert(issue);
        }
    }
}

bool OoOScheduler::IsInissueEntries(Operation* op) {
    for (auto &issue : issueEntries) {
        if (issue->tileOp.GetOpMagic() == op->GetOpMagic()) {
            return true;
        }
    }
    return false;
}

Status OoOScheduler::ScheduleMainLoop() {
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    numTotalIssues = issueEntries.size();
    uint64_t commitCnt = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        APASS_LOG_DEBUG_F(Elements::Operation, "     clock: %d", clock);
        // Retire Stage : 检查现有pipe中的op是否执行完。如果op执行完，则将op标记为retired状态，将可以被释放的buffer释放掉，并唤醒后续已经就绪的op。
        // 完毕后更新整个pipe的状态。
        if (RetireIssueStage(commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed.");
            return FAILED;
        }
        // Buffer Allocation Stage : 分配buffer。对于所有类型的buffer，按顺序执行alloc指令，并激活后续已经就绪的op。不断执行alloc直到buffer被占满为止。
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

Status OoOScheduler::RetireIssue(IssueEntryPtr issue) {
    issue->isRetired = true;
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId); 
            return FAILED; 
        }
        if (bufRefCount[memId] == 0) {
        // 加载时的核信息
            auto corePair =  tensorAllocCoreMap[memId];
            if (bufferManagerMap[corePair.first][corePair.second][localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId); 
                return FAILED; 
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(IssueEntryPtr issue, size_t &pcIdx) {
    if (localBufferMap.find(issue->reqMemIds[0]) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", issue->reqMemIds[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];
    auto corePair = issue->coreLocation;
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].IsFull(allocBuffer)) {
        if (GenSpillOp(allocBuffer, pcIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenSpillOp failed at ExecuteAllocIssue. %s", GetFormatBacktrace(issueEntries[pcIdx]->tileOp).c_str());
            return FAILED;
        }
    }
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%u] failed.", allocBuffer->id); 
        return FAILED; 
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule() {
    UpdateIssueExecOrder();
    size_t pcIdx = 0;
    LOG_SCOPE_BEGIN(tGenSpillSchedule, Elements::Function, "GenSpillSchedule");
    while (pcIdx < issueEntries.size()) {
        auto issue = issueEntries[pcIdx];
        APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", issue->GetOpInfo().c_str());
        if (issue->isAlloc) {
            if (ExecuteAllocIssue(issue, pcIdx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed! %s", GetFormatBacktrace(issue->tileOp).c_str()); 
                return FAILED;
            }
        }
        if (RetireIssue(issue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(issue->tileOp).c_str()); 
            return FAILED;
        }
        pcIdx += 1;
    }
    for (auto bufRef : bufRefCount) {
        if (bufRef.second != 0) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first); 
            return FAILED; 
        }
    }
    LOG_SCOPE_END(tGenSpillSchedule);
    InitBufRefCount();
    // 更新依赖关系
    if (InitDependencies() != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!"); 
        return FAILED; 
    }
    return SUCCESS;
}

void OoOScheduler::InitIssueQueuesAndBufferManager() {
    // 初始化
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
                issueQueues[coreType][idx][static_cast<PipeType>(i)] = IssueQueue();
            }
        }
    }

    bufferManagerMap.clear();
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (size_t i = 0; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
                allocIssueQueue[coreType][idx][static_cast<MemoryType>(i)] = IssueQueue();
                if (localMemorySize.find(static_cast<MemoryType>(i)) != localMemorySize.end()) {
                    bufferManagerMap[coreType][idx].insert({static_cast<MemoryType>(i),
                        BufferPool(static_cast<MemoryType>(i), localMemorySize[static_cast<MemoryType>(i)])});
                }
            }
        }
    }
}

void OoOScheduler::UpdateAllocMap(IssueEntryPtr issue, std::map<int, IssueEntryPtr> &tensorAllocMap) {
    for (auto outTensor : issue->tileOp.GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = issue;
        }
    }
    for (auto inTensor : issue->tileOp.GetIOperands()) {
        if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = inTensor->memoryrange.memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = issue;
        }
    }
}

Status OoOScheduler::CheckAllocIssue() {
    std::map<int, IssueEntryPtr> tensorAllocMap;
    for (const auto &issue : issueEntries) {
        if (issue->isAlloc) {
            if (issue->reqMemIds.size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "ALLOC[%d] reqMemIds size not equal to 0. %s", 
                    issue->tileOp.GetOpMagic(), GetFormatBacktrace(issue->tileOp).c_str());
                return FAILED;
            }
            UpdateAllocMap(issue, tensorAllocMap);
        }
    }
    for (const auto &issue : issueEntries) {
        if (!issue->isAlloc) {
            UpdateAllocMap(issue, tensorAllocMap);
        }
    }
    for (auto tensorAlloc : tensorAllocMap) {
        if (!tensorAlloc.second->isAlloc) {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s Tensor[%d] is missing Alloc.", 
                tensorAlloc.second->GetOpInfo().c_str(), tensorAlloc.first);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] = std::make_shared<LocalBuffer>(
            memId, ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->Datatype()), oOperand->GetMemoryTypeOriginal());
    } else {
        localBufferMap[memId]->size =
            std::max(localBufferMap[memId]->size, ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->Datatype()));
    }
}

void OoOScheduler::UpdateBufRefCount(IssueEntryPtr issue, LogicalTensorPtr tensor) {
    int memId = tensor->memoryrange.memId;
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        bufRefCount[memId]++;
        issue->reqMemIds.push_back(memId);
    }
}

void OoOScheduler::InitBufRefCount() {
    bufRefCount.clear();
    for (const auto &issue : issueEntries) {
        issue->Clear();
        for (auto &tensor : issue->tileOp.GetIOperands()) {
            UpdateBufRefCount(issue, tensor);
            int memId = tensor->memoryrange.memId;
            InitLocalBuffer(tensor, memId);
        }
        for (auto &tensor : issue->tileOp.GetOOperands()) {
            UpdateBufRefCount(issue, tensor);
            int memId = tensor->memoryrange.memId;
            InitLocalBuffer(tensor, memId);
        }
    }
}

Status OoOScheduler::InitAllocDependencies(IssueEntryPtr issue, std::unordered_map<int, IssueEntryPtr> &tensor2AllocMap) {
    for (auto &tensor : issue->tileOp.GetOOperands()) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            if (tensor2AllocMap.find(memId) == tensor2AllocMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc.", memId);
                return FAILED;
            }
            AddDependency(tensor2AllocMap[memId], issue, true);
        }
    }
    return SUCCESS;
}

void OoOScheduler::AddDependency(IssueEntryPtr preIssue, IssueEntryPtr postIssue, bool isAlloc) {
    if (preIssue == nullptr || postIssue == nullptr) {
        return;
    }
    if (isAlloc || (!preIssue->isAlloc && !postIssue->isAlloc)) {
        preIssue->successors.insert(postIssue->id);
        postIssue->predecessors.insert(preIssue->id);
    }
}

void OoOScheduler::FindDependencies(IssueEntryPtr issue, std::unordered_map<Operation*, IssueEntryPtr> &op2IssueEntryMap) {
    for (auto &producer : issue->tileOp.ProducerOps()) {
        if (IsViewOp(*producer)) {
            for (auto viewProducer : producer->ProducerOps()) {
                Operation* lastView = SkipViewChain(viewProducer, true);
                Operation* realProd = (lastView != nullptr) ? *lastView->ProducerOps().begin() : viewProducer;
                auto viewProdIssue = op2IssueEntryMap[realProd];
                AddDependency(viewProdIssue, issue, false);
            }
        } else {
            auto prodIssue = op2IssueEntryMap[producer];
            AddDependency(prodIssue, issue, false);
        }
    }
    for (auto &consumer : issue->tileOp.ConsumerOps()) {
        if (IsViewOp(*consumer)) {
            for (auto viewConsumer : consumer->ConsumerOps()) {
                Operation* lastView = SkipViewChain(viewConsumer, false);
                Operation* realCon = (lastView != nullptr) ? *lastView->ConsumerOps().begin() : viewConsumer;
                auto viewConIssue = op2IssueEntryMap[realCon];
                AddDependency(issue, viewConIssue, false);
            }
        }
    }
}

Status OoOScheduler::InitDependencies() {
    std::unordered_map<Operation*, IssueEntryPtr> op2IssueEntryMap;
    std::unordered_map<int, IssueEntryPtr> tensor2AllocMap;
    for (const auto &issue : issueEntries) {
        issue->predecessors.clear();
        issue->successors.clear();
        op2IssueEntryMap[&(issue->tileOp)] = issue;
        if (issue->isAlloc) {
            if (issue->tileOp.GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", issue->tileOp.GetOpMagic());
                return FAILED;
            }
            int memId = issue->tileOp.GetOutputOperand(0)->memoryrange.memId;
            tensor2AllocMap[memId] = issue;
            continue;
        }
    }
    for (const auto &issue : issueEntries) {
        if (!issue->isAlloc) {
            FindDependencies(issue, op2IssueEntryMap);
            if (InitAllocDependencies(issue, tensor2AllocMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
                return FAILED;
            }
        }
    }
    PrintDependencies();
    return SUCCESS;
}

void OoOScheduler::CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap) {
    for (auto tensor : tensors) {
        if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
            bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->GetDataSize();
            memIdMap.insert(tensor->memoryrange.memId);
        }
    }
}

std::string OoOScheduler::dumpOpInfo(Operation &op) {
    std::ostringstream oss;
    oss << "OP: " << op.GetOpcodeStr().c_str() << "[" << op.GetOpMagic() << "] | ";
    oss << "Inputs: {";
    for (size_t i = 0; i < op.iOperand.size(); i++) {
        oss << "Tensor[" << op.GetInputOperand(i)->GetMagic() << "] ";
        oss << op.iOperand[i]->DumpSSA(false, true, true);
        if (i != op.iOperand.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}" << " | ";
    oss << "Outputs: {";
    for (size_t i = 0; i < op.oOperand.size(); i++) {
        oss << "Tensor[" << op.GetOutputOperand(i)->GetMagic() << "] ";
        oss << op.oOperand[i]->DumpSSA(false, true, true);
        if (i != op.oOperand.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

Status OoOScheduler::CheckOpBufferSize(Operation *op) {
    std::map<MemoryType, int64_t> bufferSize;
    std::set<int> memIdMap;
    CalcBufferSize(op->GetIOperands(), bufferSize, memIdMap);
    CalcBufferSize(op->GetOOperands(), bufferSize, memIdMap);
    for (auto &buffer : bufferSize) {
        if (localMemorySize.find(buffer.first) == localMemorySize.end()) {
            continue;
        }
        if (buffer.second <= localMemorySize[buffer.first]) {
            continue;
        }
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation, "Alloc tensor[%d] size[%d] exceeds %s size[%d]! %s", 
                op->GetOutputOperand(0)->GetMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(), 
                localMemorySize[buffer.first], GetFormatBacktrace(*op).c_str());
            APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] producer info:", op->GetOutputOperand(0)->GetMagic());
            for (auto producer : op->GetOutputOperand(0)->GetProducers()) {
                if (producer == op) {
                    continue;
                }
                APASS_LOG_ERROR_F(Elements::Operation, "    %s.", dumpOpInfo(*producer).c_str());
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "OP %s[%d] in/output total size[%ld] exceeds %s size[%d]!", 
                op->GetOpcodeStr().c_str(), op->GetOpMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                localMemorySize[buffer.first]);
            APASS_LOG_ERROR_F(Elements::Operation, "%s.", dumpOpInfo(*op).c_str());
        }
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::InitTensorCoreMap() {
    // 不存在 no producer情况
    for (auto issue : issueEntries) {
        if (issue->isAlloc) {
            auto memId = issue->tileOp.GetOutputOperand(0)->memoryrange.memId;
            tensorAllocCoreMap[memId] = issue->coreLocation;
        }
    }
}

Status OoOScheduler::InitIssueCoreType(IssueEntryPtr issue, Operation* op,
    const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
    if (!opCoreMap.empty()) {
        issue->coreLocation = opCoreMap.at(op);
        return SUCCESS;
    }
    if (op->GetCoreType() == CoreType::AIC) {
        issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
        return SUCCESS;
    }
    if (op->GetCoreType() == CoreType::AIV) {
        issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
        return SUCCESS;
    }
    // 对 ANY 类型进行处理
    if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
        return SUCCESS;
    }
    if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
        issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
        return SUCCESS;
    }
    if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        if (op->GetIOperands().size() == 0 || op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
            return SUCCESS;
        }
        if (op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
            return SUCCESS;
        }
        if (op->GetInputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
            issue->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
            return SUCCESS;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. IOperand memoryType is %s",
            issue->GetOpInfo().c_str(), MemoryTypeToString(op->GetInputOperand(0)->GetMemoryTypeOriginal()).c_str());
    }
    APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. OOperand memoryType is %s",
        issue->GetOpInfo().c_str(), MemoryTypeToString(op->GetOutputOperand(0)->GetMemoryTypeOriginal()).c_str());
    return FAILED;
}

void OoOScheduler::InitCoreConfig(const std::vector<Operation *> &operations) {
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(operations)) {
        CORE_INIT_CONFIGS = CORE_INIT_CONFIGS_HARDWARE_ONE;
    } else {
        CORE_INIT_CONFIGS = CORE_INIT_CONFIGS_HARDWARE_TWO;
    }
}

Status OoOScheduler::InitIssueEntry(Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
    if (IsViewOp(*op)) {
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            newOperations_.push_back(op);
        }
        return SUCCESS;
    }
    if (CheckOpBufferSize(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] CheckOpBufferSize failed! %s",
            op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    // 核属性的初始化
    auto issue = std::make_shared<IssueEntry>(*op, issueId);
    issueEntryMap[issueId++] = issue;
    if (issue == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "IssueEntry %s, %d init failed! %s",
            op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    issueEntries.emplace_back(issue);
    if (InitIssueCoreType(issue, op, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IssueEntry %s init coreType failed!", issue->GetOpInfo().c_str());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "issue: %s, coreType: %s, idx: %d",
            issue->GetOpInfo().c_str(), coreTypeToString(issue->coreLocation.first).c_str(), issue->coreLocation.second);
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation *> &operations, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap,
    const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig) {
    issueEntries.clear();
    localBufferMap.clear();
    depthCache_.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    InitMemorySize();
    if (fixCoreConfig.empty()) {
        InitCoreConfig(operations);
    } else {
        CORE_INIT_CONFIGS = fixCoreConfig;
    }
    // 校验并初始化issueEntry
    for (const auto &op : operations) {
        if (InitIssueEntry(op, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init issue failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    numTotalIssues = issueEntries.size();

    InitBufRefCount();
    // 初始化issueEntry，构建依赖关系
    if (InitDependencies() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
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

void OoOScheduler::InitMemorySize() {
    localMemorySize = {
        {MemoryType::MEM_L0A, MAX_L0A_SIZE},
        {MemoryType::MEM_L0C, MAX_L0C_SIZE},
        {MemoryType::MEM_BT, MAX_BT_SIZE},
        {MemoryType::MEM_FIX, MAX_FIX_SIZE},
        {MemoryType::MEM_FIX_QUANT_PRE, MAX_FIX_QUANT_PRE_SIZE},
    };
    localMemorySize.insert({MemoryType::MEM_UB,
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB)});
    localMemorySize.insert({MemoryType::MEM_L1,
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1)});
    localMemorySize.insert({MemoryType::MEM_L0B,
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B)});
    localMemorySize.insert({MemoryType::MEM_FIX_QUANT_PRE,
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_FIX_QUANT_PRE)});
}

Status OoOScheduler::Schedule(const std::vector<Operation *> &operations, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap,
    const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig) {
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
    PrintOpList(newOperations_);
    function_.SetStackWorkespaceSize(workspaceOffset);
    return SUCCESS;
}

// UpdateRemainOpBufId函数不能直接用
Status OoOScheduler::UpdateMemId(int oldMemId, int newMemId) {
    if (bufRefCount.find(oldMemId) == bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]", oldMemId);
        return FAILED;
    }
    bufRefCount[newMemId] = 0;
    for (auto &issue : issueEntries) {
        if (issue->isRetired) {
            continue;
        }
        for (auto &memId : issue->reqMemIds) {
            if (memId == oldMemId) {
                memId = newMemId;
                bufRefCount[oldMemId] -= 1;
                bufRefCount[newMemId] += 1;
            }
        }
        for (auto &outTensor : issue->tileOp.GetOOperands()) {
            if (outTensor->memoryrange.memId == oldMemId) {
                outTensor->memoryrange.memId = newMemId;
            }
        }
    }
    if (bufRefCount[oldMemId] != 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "oldMemId %d bufRefCount is not 0, UpdateMemId failed.", oldMemId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp) {
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

void OoOScheduler::ProcessMoveIssue(IssueEntryPtr moveIssuePtr, IssueEntryPtr AllocIssue, MemoryType memType, int oldMemId, int newMemId) {
    issueEntryMap[issueId++] = moveIssuePtr;
    moveIssuePtr->reqMemIds = {oldMemId, newMemId};
    moveIssuePtr->isRetired = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Add MOVEOP: %s.", moveIssuePtr->GetOpInfo().c_str());
    tensorOccupyMap[memType][newMemId] = moveIssuePtr;
    // 更新moveIssue的相关信息
    auto occupyIssuePtr = tensorOccupyMap[memType][oldMemId];
    moveIssuePtr->predecessors.insert(occupyIssuePtr->id);
    occupyIssuePtr->successors.insert(moveIssuePtr->id);
    // 更新执行序
    moveIssuePtr->execOrder = AllocIssue->execOrder;
    InsertIssueEntries(moveIssuePtr);
    // 找出moveFromTensor的所有consumer中未执行的, 并改变图的连接
    UpdateReloadIssueDepend(moveIssuePtr, occupyIssuePtr, oldMemId);
}

Status OoOScheduler::FindMoveFromTensor(Operation &occupyOp, int oldMemId, MemoryType memType, bool &rearrangeUBBF16, LogicalTensorPtr &moveFromTensor) {
    for (auto outTensorPtr : occupyOp.GetOOperands()) {
        if (outTensorPtr->memoryrange.memId == oldMemId) {
            moveFromTensor = outTensorPtr;
            break;
        }
    }
    if (moveFromTensor == nullptr) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor(memId: %d) according to tensorOccupyMap, GenRearrangeCopyOp failed", oldMemId);
        return FAILED;
    }
    // 如果moveFrom Tensor是UB且数据类型为bf16, rearrange失败
    if (memType == MemoryType::MEM_UB && moveFromTensor->Datatype() == DataType::DT_BF16) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot rearrange UB tensor with datatype bf16, do schedulemainloop spill.");
        rearrangeUBBF16 = true;
    }
    return SUCCESS;
}

Status OoOScheduler::GetMoveOpInTensor(Opcode moveOpcode, Operation &occupyOp, LogicalTensorPtr &inTensor, LogicalTensorPtr &moveFromTensor) {
    if (moveOpcode == Opcode::OP_COPY_IN) {
        if (occupyOp.GetOpcode() != Opcode::OP_COPY_IN) {
            APASS_LOG_WARN_F(Elements::Operation, "Occupy op is not COPY_IN, GetMoveOpInTensor failed.");
            return FAILED;
        }
        inTensor = occupyOp.GetIOperands()[0];
        if (inTensor == nullptr || inTensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_WARN_F(Elements::Tensor, "inTensor is illegal, GetMoveOpInTensor failed.");
            return FAILED;
        }
    } else {
        inTensor = moveFromTensor;
    }
    return SUCCESS;
}

Status OoOScheduler::GenRearrangeCopyOp(IssueEntryPtr AllocIssue, MemoryType memType, int oldMemId, int &newMemId, bool &rearrangeUBBF16) {
    if (memType != MemoryType::MEM_L1 && memType != MemoryType::MEM_UB) {
        APASS_LOG_WARN_F(Elements::Tensor, "Unexpected rearrange tensor memory type found, GenRearrangeCopyOp failed.");
        return FAILED;
    }
    Opcode moveOpcode = memType == MemoryType::MEM_L1 ? Opcode::OP_COPY_IN : Opcode::OP_ADDS;
    auto &occupyOp = tensorOccupyMap[memType][oldMemId]->tileOp;
    LogicalTensorPtr moveFromTensor{nullptr};
    if (FindMoveFromTensor(occupyOp, oldMemId, memType, rearrangeUBBF16, moveFromTensor) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "GenRearrangeCopyOp failed at FindMoveFromTensor.");
        return FAILED;
    }
    if (rearrangeUBBF16) {
        return SUCCESS;
    }
    LogicalTensorPtr moveToTensor = std::make_shared<LogicalTensor>(function_, moveFromTensor->Datatype(), moveFromTensor->shape);
    // 给moveToTensor分配memId和创建新的localbuffer
    if (UpdateTensorAttr(moveToTensor, memType, moveFromTensor, oldMemId) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "GenRearrangeCopyOp failed at UpdateTensorAttr.");
        return FAILED;
    }
    newMemId = moveToTensor->memoryrange.memId;
    LogicalTensorPtr inTensor{nullptr};
    if (GetMoveOpInTensor(moveOpcode, occupyOp, inTensor, moveFromTensor) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "GenRearrangeCopyOp failed at GetMoveOpInTensor.");
        return FAILED;
    }
    auto &moveOp = (moveOpcode == Opcode::OP_COPY_IN && occupyOp.GetOpcode() == Opcode::OP_COPY_IN) ?
        occupyOp.CloneOperation(function_, {inTensor}, {moveToTensor}) :
        function_.AddRawOperation(moveOpcode, {inTensor}, {moveToTensor});
    moveToTensor->tensor->rawshape = inTensor->tensor->rawshape;
    newOperations_.push_back(&moveOp);
    // UpdateMoveOpAttr & 创建moveop的issueEntry
    UpdateMoveOpAttr(moveOp, occupyOp);
    IssueEntryPtr moveIssuePtr = std::make_shared<IssueEntry>(moveOp, issueId);
    moveIssuePtr->coreLocation = tensorAllocCoreMap[newMemId];
    // 处理issue & 改变图连接
    ProcessMoveIssue(moveIssuePtr, AllocIssue, memType, oldMemId, newMemId);
    // 更新memId
    if (UpdateMemId(oldMemId, newMemId) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "GenRearrangeCopyOp failed at UpdateMemId.", GetFormatBacktrace(moveOp).c_str());
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

Status OoOScheduler::UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool &bufferManager) {
    auto moveToBufferPtr = localBufferMap[newMemId];
    if (bufferManager.ModifyBufferRange(moveToBufferPtr, offset) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "UpdateRange failed at ModifyBufferRange.");
        return FAILED;
    }
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(memType, newMemId, false);
    }
    tensorOccupyMap[memType][newMemId]->tileOp.GetOOperands()[0]->memoryrange =
        TileRange(offset, offset + moveToBufferPtr->size, newMemId);
    localBufferMap[newMemId]->startCycle = clock;
    return SUCCESS;
}

Status OoOScheduler::RearrangeBuffers(IssueEntryPtr issue, bool isGenSpillStage, bool &rearrangeUBBF16) {
    LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];
    auto corePair = issue->coreLocation;
    BufferPool &bufferManager = bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType];
    auto rearrangeScheme = GetRearrangeScheme(bufferManager, allocBuffer->size);
    if (rearrangeScheme.cost == INT_MAX) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffers failed at GetRearrangeScheme.");
        return FAILED;
    }
    // 修改tensor对应的localbuffer
    for (auto &[memId, offset] : rearrangeScheme.orderedMoveTo) {
        auto targetBufferPtr = localBufferMap[memId];
        if (rearrangeScheme.moveFrom[memId] != targetBufferPtr->start ||
            rearrangeScheme.memSizeMap[memId] != targetBufferPtr->size) {
            APASS_LOG_WARN_F(Elements::Tensor, "MemId %d localBuffer and rearrangeScheme range donot match, RearrangeBuffers failed.", memId);
            return FAILED;
        }
        IssueEntryPtr occupyIssuePtr = GetSpillIssue(issue, memId, isGenSpillStage);
        if (occupyIssuePtr == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "OccupyIssue is nullptr, RearrangeBuffers failed. %s", GetFormatBacktrace(issue->tileOp).c_str());
            return FAILED;
        }
        if (IsViewOp(occupyIssuePtr->tileOp) || occupyIssuePtr->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
            APASS_LOG_WARN_F(Elements::Operation, "Target rearrange tensor(memId: %d)'s occupy op is %d %s, RearrangeBuffers failed. %s",
                memId, issue->tileOp.GetOpMagic(), issue->tileOp.GetOpcodeStr().c_str(), GetFormatBacktrace(issue->tileOp).c_str());
            return FAILED;
        }
        // GenSpillStage阶段的内存整理不需要插入搬运节点
        // ScheduleMainLoop阶段如果是alloc占有的tensor不需要插入搬运节点
        if (isGenSpillStage || occupyIssuePtr->tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
            if (bufferManager.ModifyBufferRange(targetBufferPtr, offset) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Tensor, "RearrangeBuffers failed at ModifyBufferRange.");
                return FAILED;
            }
        } else {
            int newMemId = INT_MAX;
            if (GenRearrangeCopyOp(issue, allocBuffer->memType, memId, newMemId, rearrangeUBBF16) != SUCCESS) {
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