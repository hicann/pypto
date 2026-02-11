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
 * \file spill_buffer.cpp
 * \brief
 */

#include "scheduler.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;

OoOSchedulerCheck::SpillInfo OoOScheduler::RecordSpillInfo(MemoryType bufferType, int memId,
    LocalBufferPtr allocBuffer, LogicalTensorPtr spillOutTensor, bool needCopyOut) {
    OoOSchedulerCheck::SpillInfo spillInfo;
    spillInfo.spillType = bufferType;
    spillInfo.bufferCurrUsage = oooCheck.bufferLastUsage[bufferType];
    spillInfo.spillTensorSize = localBufferMap[memId]->size;
    spillInfo.spillTensorMagic = spillOutTensor->GetMagic();
    spillInfo.triggerTensorSize = allocBuffer->size;
    int allocOccupied = 0;
    for (const auto &pair : tensorOccupyMap[bufferType]) {
        if (pair.second->isAlloc) {
            allocOccupied += localBufferMap[pair.first]->size;
        }
    }
    spillInfo.allocOccupiedSize = allocOccupied;
    if (needCopyOut) {
        auto dtype = spillOutTensor->tensor->datatype;
        spillInfo.spillCopyoutSize = std::accumulate(spillOutTensor->shape.begin(), spillOutTensor->shape.end(),
            1, std::multiplies<int64_t>()) * BytesOf(dtype);
    } else {
        spillInfo.spillCopyoutSize = 0;
    }
    return spillInfo;
}

int OoOScheduler::GetBufNextUseOrder(IssueEntryPtr issue, int curMemId) {
    auto it = std::find_if(issueEntries.begin(), issueEntries.end(), [issue, curMemId](const IssueEntryPtr a) {
        return a && a->execOrder > issue->execOrder &&
            std::find(a->reqMemIds.begin(), a->reqMemIds.end(), curMemId) != a->reqMemIds.end();
    });
    return (it != issueEntries.end()) ? (*it)->execOrder : -1;
}

int OoOScheduler::GetBufLastUseOrder(IssueEntryPtr issue, int curMemId) {
    auto targetIt = std::find(issueEntries.begin(), issueEntries.end(), issue);
    if (targetIt == issueEntries.end()) {
        return -1;
    }
    for (auto it = std::make_reverse_iterator(targetIt); it != issueEntries.rend(); it++) {
        IssueEntryPtr curIssue = *it;
        if (curIssue && curIssue->execOrder < issue->execOrder && std::find(curIssue->reqMemIds.begin(),
            curIssue->reqMemIds.end(), curMemId) != curIssue->reqMemIds.end()) {
            return curIssue->execOrder;
        }
    }
    return -1;
}

IssueEntryPtr OoOScheduler::GetBufLastWriteIssue(IssueEntryPtr issue, int curMemId) {
    auto targetIt = std::find(issueEntries.begin(), issueEntries.end(), issue);
    if (targetIt == issueEntries.end()) {
        return nullptr;
    }
    for (auto it = std::make_reverse_iterator(targetIt); it != issueEntries.rend(); it++) {
        IssueEntryPtr curIssue = *it;
        if (curIssue == nullptr || curIssue->execOrder >= issue->execOrder) {
            continue;
        }
        for (auto& outTensor : curIssue->tileOp.GetOOperands()) {
            if (outTensor->memoryrange.memId == curMemId) {
                return curIssue;
            }
        }
    }
    return nullptr;
}

Status OoOScheduler::UpdateTensorAttr(
    LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId) {
    tensor->SetMemoryTypeToBe(memType);
    tensor->SetMemoryTypeOriginal(memType);
    tensor->oriShape = spillTensor->oriShape;
    tensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    tensor->tensor->rawshape = spillTensor->tensor->rawshape;
    if (memType == MEM_DEVICE_DDR) {
        if (localBufferMap.find(spillMemId) == localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap.", spillMemId);
            return FAILED;
        }
        tensor->memoryrange =
            TileRange(workspaceOffset, workspaceOffset + localBufferMap[spillMemId]->size, workspaceMemId++);
        workspaceOffset += localBufferMap[spillMemId]->size;
    } else {
        int rawMagic = tensor->GetRawTensor()->GetRawMagic();
        tensor->memoryrange.memId = rawMagic;
        localBufferMap[rawMagic] = std::make_shared<LocalBuffer>(
            rawMagic, tensor->tensor->GetRawDataSize(), tensor->GetMemoryTypeOriginal());
        if (localBufferMap[rawMagic] == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Init Tensor[%d] localBuffer failed.", rawMagic);
            return FAILED;
        }
        tensorAllocCoreMap[tensor->memoryrange.memId] = tensorAllocCoreMap[spillTensor->memoryrange.memId];
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOpInternalSubgraphID(Operation &op, IssueEntryPtr issue) {
    if (issue->tileOp.GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(issue->tileOp.GetInternalSubgraphID());
        op.SetAIVCore(issue->tileOp.GetAIVCore());
    }
}

void OoOScheduler::UpdateOpAttr(
    Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset, IssueEntryPtr spillIssue) {
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(spillTensor->GetMemoryTypeOriginal(),
            OpImmediate::Specified(offset), OpImmediate::Specified(spillTensor->GetShape()),
            OpImmediate::Specified(spillTensor->GetRawTensor()->GetDynRawShape())));
    } else if (op.GetOpcodeStr().find("ALLOC") == std::string::npos) {
        if (spillIssue->tileOp.GetOpcode() == Opcode::OP_COPY_IN) {
            op.SetOpAttribute(spillIssue->tileOp.GetOpAttribute()->Clone());
            op.inParamLocation_ = spillIssue->tileOp.inParamLocation_;
        } else {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset),
                spillTensor->GetMemoryTypeOriginal(), OpImmediate::Specified(spillTensor->GetShape()),
                OpImmediate::Specified(spillTensor->tensor->GetDynRawShape())));
        }
    }
    op.UpdateLatency(opLatency);
    UpdateOpInternalSubgraphID(op, spillIssue);
}

void OoOScheduler::ReplaceTensorMemId(IssueEntryPtr &issue, int oldMemId, int newMemId) {
    for (auto memId : issue->reqMemIds) {
        if (memId == oldMemId) {
            std::replace(issue->reqMemIds.begin(), issue->reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto &outTensor : issue->tileOp.GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
        }
    }
}

Status OoOScheduler::UpdateRemainOpBufId(int oldMemId, int newMemId) {
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = bufRefCount_[oldMemId] + TWO_ISSUE;
    bufRefCount_[oldMemId] = 0;
    for (auto& issue : issueEntries) {
        if (issue->isRetired) {
            continue;
        }
        ReplaceTensorMemId(issue, oldMemId, newMemId);
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateReloadIssueDepend(IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue, int spillMemId) {
    for (auto& succId : spillIssue->successors) {
        auto succ = issueEntryMap[succId];
        if (!succ->isRetired && (std::count(succ->reqMemIds.begin(), succ->reqMemIds.end(), spillMemId) > 0)) {
            reloadCopyin->successors.insert(succ->id);
            if (succ->predecessors.erase(spillIssue->id) == 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "Erase issueEntry %s failed. %s", spillIssue->GetOpInfo().c_str(), GetFormatBacktrace(spillIssue->tileOp).c_str());
                return FAILED;
            }
            succ->predecessors.insert(reloadCopyin->id);
            if (reloadCopyin->tileOp.GetOutputOperand(0) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s cannot find oOperand[0]. %s", reloadCopyin->GetOpInfo().c_str(), GetFormatBacktrace(reloadCopyin->tileOp).c_str());
                return FAILED;
            }
            succ->UpdateTensorInput(spillIssue, reloadCopyin->tileOp.GetOutputOperand(0));
        }
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin,
    IssueEntryPtr spillIssue, int spillMemId, IssueEntryPtr allocIssue) {
    int allocOOperandsSize = reloadAlloc->tileOp.GetOOperands().size();
    int copyInOOperandsSize = reloadCopyin->tileOp.GetOOperands().size();
    if (allocOOperandsSize != 1 || copyInOOperandsSize != 1 || reloadAlloc->tileOp.GetOutputOperand(0) != reloadCopyin->tileOp.GetOutputOperand(0)) {
        APASS_LOG_ERROR_F(Elements::Operation, "oOperands expected 1. %s and %s should share the same oOperand[0]. %s", reloadAlloc->GetOpInfo().c_str(), reloadCopyin->GetOpInfo().c_str(), GetFormatBacktrace(reloadAlloc->tileOp).c_str());
        return FAILED;
    }
    auto outTensor = reloadAlloc->tileOp.GetOutputOperand(0);
    reloadAlloc->reqMemIds = {outTensor->memoryrange.memId};
    reloadAlloc->successors.insert(reloadCopyin->id);
    reloadCopyin->reqMemIds = {outTensor->memoryrange.memId};
    reloadCopyin->predecessors.insert(reloadAlloc->id);
    int bufNextUseOrder = GetBufNextUseOrder(allocIssue, spillMemId);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use order failed.", spillMemId);
        return FAILED;
    }
    reloadAlloc->execOrder = bufNextUseOrder++;
    InsertIssueEntries(reloadAlloc);
    reloadCopyin->execOrder = bufNextUseOrder;
    InsertIssueEntries(reloadCopyin);
    if (UpdateReloadIssueDepend(reloadCopyin, spillIssue, spillMemId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateReloadIssueDepend failed. %s", GetFormatBacktrace(reloadCopyin->tileOp).c_str());
        return FAILED;
    }
    if (UpdateRemainOpBufId(spillMemId, reloadAlloc->reqMemIds[0])) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainOpBufId failed. %s", GetFormatBacktrace(reloadAlloc->tileOp).c_str());
        return FAILED;
    }
    for (auto& issue : issueEntries) {
        if (issue->isRetired || issue->isAlloc) {
            continue;
        }
        auto predecessors = issue->predecessors;
        for (auto predId : predecessors) {
            auto predecessor = issueEntryMap[predId];
            if (predecessor->isAlloc && std::find(predecessor->reqMemIds.begin(), predecessor->reqMemIds.end(),
                spillMemId) != predecessor->reqMemIds.end()) {
                issue->predecessors.erase(predId);
                issue->predecessors.insert(reloadAlloc->id);
            }
        }
    }
    numTotalIssues += TWO_ISSUE;
    return SUCCESS;
}

Status OoOScheduler::CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor,
    LogicalTensorPtr spillTensor, IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues) {
    MemoryType memType = spillTensor->GetMemoryTypeOriginal();
    // 创建将spill搬出数据搬回OP_COPY_IN的tensor
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(
            function_, spillTensor->Datatype(), spillTensor->shape, spillTensor->Format());
    if (localTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create local tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(localTensor, memType, spillTensor, -1) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "UpdateTensorAttr local tensor failed!");
        return FAILED;
    }
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    // 创建spill搬出数据搬回OP_COPY_IN/OP_ALLOC
    Opcode allocOp = memType == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    auto &spillAllocOp = function_.AddRawOperation(allocOp, {}, {localTensor});
    auto &spillCopyInOp = (spillIssue->tileOp.GetOpcode() == Opcode::OP_COPY_IN) ?
        spillIssue->tileOp.CloneOperation(function_, {spillOutTensor}, {localTensor}) :
        function_.AddRawOperation(Opcode::OP_COPY_IN, {spillOutTensor}, {localTensor});

    if (spillIssue->tileOp.GetOpcode() == Opcode::OP_COPY_IN) {
        spillCopyInOp.SetIOpAttrOffset(0, spillIssue->tileOp.GetIOpAttrOffset(0));
    }
    UpdateOpAttr(spillAllocOp, 1, localTensor, {}, spillIssue);
    UpdateOpAttr(spillCopyInOp, DEFAULT_LATENCY, localTensor, spillOutTensor->GetOffset(), spillIssue);

    // 初始化OP_COPY_IN/OP_ALLOC的issueEntry
    IssueEntryPtr spillAllocInst = std::make_shared<IssueEntry>(spillAllocOp, issueId);
    spillAllocInst->coreLocation = spillIssue->coreLocation;
    issueEntryMap[issueId++] = spillAllocInst;
    IssueEntryPtr spillInInst = std::make_shared<IssueEntry>(spillCopyInOp, issueId);
    spillInInst->coreLocation = spillIssue->coreLocation;
    issueEntryMap[issueId++] = spillInInst;
    if (spillAllocInst == nullptr || spillInInst == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Create OP_COPY_IN/OP_ALLOC issueEntry failed!");
        return FAILED;
    }
    reloadIssues.first = spillAllocInst;
    reloadIssues.second = spillInInst;
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_ALLOC: %s.", spillAllocInst->GetOpInfo().c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_IN: %s.", spillInInst->GetOpInfo().c_str());
    return SUCCESS;
}

Status OoOScheduler::SpillInBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, MemoryType bufferType,
    bool isGenSpill) {
    IssueEntryPtr reloadCopyin = nullptr;
    IssueEntryPtr reloadAlloc = nullptr;
    std::pair<IssueEntryPtr, IssueEntryPtr> reloadIssues = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(spillInfo.ddrTensor_, spillInfo.spillTensor_, spillInfo.spillIssue_,
        reloadIssues) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillReloadIssue failed!");
        return FAILED;
    }
    reloadAlloc = reloadIssues.first;
    reloadCopyin = reloadIssues.second;
    if (UpdateReloadIssueInfo(reloadAlloc, reloadCopyin, spillInfo.spillIssue_, spillInfo.spillMemId_,
        allocIssue) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateReloadIssueInfo failed!");
        return FAILED;
    }
    auto corePair = allocIssue->coreLocation;
    if (!isGenSpill) {
        allocIssueQueue[corePair.first][corePair.second][bufferType].Insert(reloadAlloc);
    }
    if (bufferManagerMap[corePair.first][corePair.second][bufferType].Free(spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::CreateSpillCopyout(IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor,
    int spillMemId, IssueEntryPtr &spillCopyout) {
    // 创建spill搬出所需的DDR rawtensor/tensor
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(spillTensor->Datatype(), spillTensor->tensor->rawshape,
        TileOpFormat::TILEOP_ND, "WorkspaceGm", SYMBOL_STACK_BASE);
    if (ddrRawTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR raw tensor failed!");
        return FAILED;
    }
    std::vector<int64_t> offset(spillTensor->GetShape().size(), 0);
    offset.front() = workspaceOffset;

    LogicalTensorPtr ddrTensor = std::make_shared<LogicalTensor>(function_, ddrRawTensor, offset, spillTensor->GetShape());
    if (ddrTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(ddrTensor, MEM_DEVICE_DDR, spillTensor, spillMemId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "UpdateTensorAttr DDR tensor failed!");
        return FAILED;
    }

    // 创建spill搬出所需的DDR OP_COPY_OUT
    Operation &spillOutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {spillTensor}, {ddrTensor});
    UpdateOpAttr(spillOutOp, DEFAULT_LATENCY, spillTensor, offset, spillIssue);

    // 创建spill搬出数据OP_COPY_OUT的issueEntry
    spillCopyout = std::make_shared<IssueEntry>(spillOutOp, issueId);
    issueEntryMap[issueId++] = spillCopyout;
    if (spillCopyout == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Create OP_COPY_OUT issueEntry failed! %s", GetFormatBacktrace(spillOutOp).c_str());
        return FAILED;
    }
    spillCopyout->reqMemIds = {spillMemId};
    spillCopyout->predecessors.insert(spillIssue->id);
    spillIssue->successors.insert(spillCopyout->id);
    spillCopyout->isRetired = true;
    spillCopyout->coreLocation = spillIssue->coreLocation;
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_OUT: %s.", spillCopyout->GetOpInfo().c_str());
    return SUCCESS;
}

Status OoOScheduler::SpillOutBuffer(SpillInfo &spillInfo, IssueEntryPtr issue, size_t &pcIdx, bool isGenSpill) {
    if (spillInfo.spillIssue_->tileOp.GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        IssueEntryPtr spillCopyout = nullptr;
        if (CreateSpillCopyout(spillInfo.spillIssue_, spillInfo.spillTensor_, spillInfo.spillMemId_,
            spillCopyout) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillCopyout failed! %s", GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
            return FAILED;
        }
        int bufLastUseOrder = GetBufLastUseOrder(issue, spillInfo.spillMemId_);
        if (bufLastUseOrder == -1) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last used order.", spillInfo.spillMemId_);
            return FAILED;
        }
        spillCopyout->execOrder = bufLastUseOrder + 1;
        InsertIssueEntries(spillCopyout);
        if (isGenSpill) {
            pcIdx++;
            numTotalIssues++;
        } else {
            newOperations_.push_back(&(spillCopyout->tileOp));
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s", spillCopyout->GetOpInfo().c_str());
        }
        spillInfo.ddrTensor_ = spillCopyout->tileOp.GetOutputOperand(0);
    } else {
        spillInfo.ddrTensor_ = spillInfo.spillIssue_->tileOp.GetInputOperand(0);
    }
    return SUCCESS;
}

Status OoOScheduler::GetSpillTensor(IssueEntryPtr spillIssue, int spillMemId, LogicalTensorPtr &spillTensor) {
    int spillTensorIdx = spillIssue->GetOOperandIdx(spillMemId);
    if (spillTensorIdx == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in op's oOperand.", spillMemId);
        return FAILED;
    }
    spillTensor = spillIssue->tileOp.GetOutputOperand(spillTensorIdx);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op cannot find oOperand[%d]. %s", spillTensorIdx, GetFormatBacktrace(spillIssue->tileOp).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SpillBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    if (SpillOutBuffer(spillInfo, allocIssue, pcIdx, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOutBuffer failed. %s", GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
        return FAILED;
    }
    // Healthcheck record - spill info
    if (oooCheck.doHealthCheck) {
        oooCheck.spillInfoVec.emplace_back(
            RecordSpillInfo(allocBuffer->memType, spillInfo.spillMemId_, allocBuffer, spillInfo.ddrTensor_,
                spillInfo.spillIssue_->tileOp.GetOpcodeStr().find("COPY_IN") == std::string::npos));
    }
    if (SpillInBuffer(spillInfo, allocIssue, allocBuffer->memType, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillInBuffer failed. %s", GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
        return FAILED;
    }
    if (!isGenSpill) {
        // Healthcheck record - update buffer usage statistics
        if (oooCheck.doHealthCheck) {
            UpdateBufferUsage(allocBuffer->memType, spillInfo.spillMemId_, true);
        }
        localBufferMap[spillInfo.spillMemId_]->retireCycle = clock;
        if (tensorOccupyMap[allocBuffer->memType].erase(spillInfo.spillMemId_) == 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", spillInfo.spillMemId_);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FindAssembleWithSpillTensor(SpillInfo &spillInfo, std::vector<IssueEntryPtr> &assembleList) {
    for (auto producer : spillInfo.spillTensor_->GetProducers()) {
        if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            APASS_LOG_ERROR_F(Elements::Operation, 
                "All producer of Tensor[%d] must be assemble, now has %s[%d].", 
                spillInfo.spillTensor_->GetMagic(), producer->GetOpcodeStr(), producer->GetOpMagic());
            return FAILED;
        }
        for (auto issue : issueEntries) {
            if (&(issue->tileOp) == producer) {
                assembleList.push_back(issue);
                break;
            }
        }
    }
    return SUCCESS;
}

int64_t OoOScheduler::CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset) {
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }

    int64_t result = 0;
    int64_t stride = 1;
    // 从最低维到最高维计算
    for (size_t i = shape.size(); i > 0; --i) {
        result += offset[i - 1] * stride;
        if (i > 0) {
            stride *= shape[i - 1];
        }
    }
    return result;
}

LogicalTensorPtr OoOScheduler::CreateAssemblePartTensor(LogicalTensorPtr iOperand, LogicalTensorPtr assembleTensor,
    SpillInfo &spillInfo, std::shared_ptr<AssembleOpAttribute> assembleAttr) {
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(
        function_, iOperand->Datatype(), iOperand->shape, iOperand->Format());
    localTensor->SetMemoryTypeToBe(assembleTensor->GetMemoryTypeToBe());
    localTensor->SetMemoryTypeOriginal(assembleTensor->GetMemoryTypeOriginal());
    localTensor->oriShape = iOperand->shape;
    localTensor->tensor = assembleTensor->tensor;
    localTensor->memoryrange.memId = assembleTensor->memoryrange.memId;
    localTensor->UpdateDynValidShape(spillInfo.spillTensor_->GetDynValidShape());
    localTensor->offset = assembleAttr->GetToOffset();
    tensorAllocCoreMap[localTensor->memoryrange.memId] = tensorAllocCoreMap[iOperand->memoryrange.memId];
    return localTensor;
}

Status OoOScheduler::SpillParticalBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, IssueEntryPtr assemble, 
    LogicalTensorPtr assembleTensor, bool &isFirst) {
    auto iOperand = assemble->tileOp.GetInputOperand(0);
    auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(assemble->tileOp.GetOpAttribute());
    LogicalTensorPtr localTensor = CreateAssemblePartTensor(iOperand, assembleTensor, spillInfo, assembleAttr);
    int bufNextUseOrder = GetBufNextUseOrder(allocIssue, spillInfo.spillMemId_);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get Tensor[%d] next use order failed.", spillInfo.spillMemId_);
        return FAILED;
    }
    if (isFirst) {
        // alloc
        Opcode allocOp = assembleTensor->GetMemoryTypeToBe() == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
        auto &spillAllocOp = function_.AddRawOperation(allocOp, {}, {localTensor});
        spillAllocOp.UpdateLatency(1);
        UpdateOpInternalSubgraphID(spillAllocOp, allocIssue);
        IssueEntryPtr spillAllocInst = std::make_shared<IssueEntry>(spillAllocOp, issueId);
        issueEntryMap[issueId++] = spillAllocInst;
        spillAllocInst->reqMemIds = {assembleTensor->memoryrange.memId};
        spillAllocInst->execOrder = bufNextUseOrder++;
        spillAllocInst->coreLocation = allocIssue->coreLocation;
        InsertIssueEntries(spillAllocInst);
        isFirst = false;
    }
    // copyin
    std::vector<int64_t> offset(iOperand->GetShape().size(), 0);
    int64_t gmRelatOffset = CalcWorkspaceOffset(assembleTensor->GetShape(), assembleAttr->GetToOffset());
    if (gmRelatOffset == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "CalcWorkspaceOffset failed.");
        return FAILED;
    }
    offset.front() = gmRelatOffset + spillInfo.ddrTensor_->GetOffset().front() + spillInfo.ddrTensor_->GetOffset().front();
    auto &spillCopyInOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {localTensor});
    spillCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset),
                iOperand->GetMemoryTypeOriginal(), OpImmediate::Specified(iOperand->GetShape()),
                OpImmediate::Specified(assembleTensor->tensor->GetDynRawShape())));
    spillCopyInOp.UpdateLatency(DEFAULT_LATENCY);
    IssueEntryPtr spillInInst = std::make_shared<IssueEntry>(spillCopyInOp, issueId);
    issueEntryMap[issueId++] = spillInInst;
    spillInInst->reqMemIds = {assembleTensor->memoryrange.memId};
    spillInInst->execOrder = bufNextUseOrder++;
    InsertIssueEntries(spillInInst);
    // assemble
    auto &assembleOp = function_.AddRawOperation(Opcode::OP_ASSEMBLE, {localTensor}, {assembleTensor});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(assembleAttr->GetFrom(), 
        assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset(), assembleAttr->GetFromDynValidShape()));
    assembleOp.UpdateLatency(1);
    IssueEntryPtr assembleInst = std::make_shared<IssueEntry>(assembleOp, issueId);
    issueEntryMap[issueId++] = assembleInst;
    assembleInst->reqMemIds = {assembleTensor->memoryrange.memId, assembleTensor->memoryrange.memId};
    assembleInst->execOrder = bufNextUseOrder;
    InsertIssueEntries(assembleInst);
    return SUCCESS;
}

Status OoOScheduler::UpdateAssembleBuffer(SpillInfo &spillInfo, LocalBufferPtr allocBuffer, 
    LogicalTensorPtr assembleTensor) {
    auto corePair = tensorAllocCoreMap[allocBuffer->id];
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].Free(spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    if (UpdateRemainOpBufId(spillInfo.spillMemId_, assembleTensor->memoryrange.memId)) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainOpBufId failed.");
        return FAILED;
    }
    bufRefCount_[assembleTensor->memoryrange.memId] = 0;
    for (auto issue: issueEntries) {
        if (issue->isRetired) {
            continue;
        }
        for (auto memId : issue->reqMemIds) {
            if (memId == assembleTensor->memoryrange.memId) {
                bufRefCount_[assembleTensor->memoryrange.memId]++;
            }
        } 
    }
    InitDependencies();
    return SUCCESS;
}

Status OoOScheduler::SpillAssembleBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    if (SpillOutBuffer(spillInfo, allocIssue, pcIdx, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOutBuffer failed.");
        return FAILED;
    }

    LogicalTensorPtr assembleTensor = std::make_shared<LogicalTensor>(function_, 
        spillInfo.spillTensor_->Datatype(), spillInfo.spillTensor_->shape, spillInfo.spillTensor_->Format());
    if (assembleTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Create assemble tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(assembleTensor, allocBuffer->memType, spillInfo.spillTensor_, -1) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateTensorAttr local tensor failed!");
        return FAILED;
    }
    for (auto &succId : spillInfo.spillIssue_->successors) {
        auto succ = issueEntryMap[succId];
        if (!succ->isRetired && 
            (std::count(succ->reqMemIds.begin(), succ->reqMemIds.end(), spillInfo.spillMemId_) > 0)) {
            succ->UpdateTensorInput(spillInfo.spillIssue_, assembleTensor);
        }
    }
    std::vector<IssueEntryPtr> assembleList;
    FindAssembleWithSpillTensor(spillInfo, assembleList);
    Operation *memIdAlloc = nullptr;
    for (auto assemble : assembleList) {
        for (auto producer : assemble->tileOp.ProducerOps()) {
            if (producer->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                memIdAlloc = producer;
            }
        }
    }
    bool isFirst = true;
    for (auto assemble : assembleList) {
        if (assemble->isRetired) {
            if (isFirst) {
                memIdAlloc->UpdateOutputOperand(0, assemble->tileOp.GetOutputOperand(0));
            }
            SpillParticalBuffer(spillInfo, allocIssue, assemble, assembleTensor, isFirst);
        } else {
            assemble->tileOp.ReplaceOutput(assembleTensor, spillInfo.spillTensor_);
        }
    }
    if(UpdateAssembleBuffer(spillInfo, allocBuffer, assembleTensor) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::GetSpillInfo(IssueEntryPtr allocIssue, int spillMemId, bool isGenSpill,
    SpillInfo &spillInfo) {
    auto spillIssue = GetSpillIssue(allocIssue, spillMemId, isGenSpill);
    if (spillIssue == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", spillMemId);
        return FAILED;
    }
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s GetSpillTensor failed! %s", spillIssue->GetOpInfo().c_str(), GetFormatBacktrace(spillIssue->tileOp).c_str());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Begin spill %s tensor[%d].", spillIssue->GetOpInfo().c_str(), spillMemId);
    LogicalTensorPtr ddrTensor = nullptr;
    spillInfo.ddrTensor_ = ddrTensor;
    spillInfo.spillTensor_ = spillTensor;
    spillInfo.spillIssue_ = spillIssue;
    spillInfo.spillMemId_ = spillMemId;
    return SUCCESS;
}

Status OoOScheduler::SpillMultiBuffer(IssueEntryPtr allocIssue, std::vector<int> spillGroup, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    for (auto &spillMemId : spillGroup) {
        SpillInfo spillInfo;
        if (GetSpillInfo(allocIssue, spillMemId, isGenSpill, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed. %s", GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
            return FAILED;
        }
        if (spillInfo.spillIssue_->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (SpillAssembleBuffer(spillInfo, allocIssue, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillAssembleBuffer[%d] failed.", spillMemId);
                return FAILED;
            }
        } else {
            if (SpillBuffer(spillInfo, allocIssue, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed. %s", spillMemId, GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::FindFilterLtags(IssueEntryPtr allocIssue, std::set<IssueEntryPtr> &filterLtags) {
    auto dstIssueList = allocIssue->successors;
    auto dstIssue = issueEntryMap[*dstIssueList.begin()];
    if(COPY_IN_OPS.find(dstIssue->tileOp.GetOpcode()) != COPY_IN_OPS.end()){
        for (auto &dstIssueId : dstIssue->successors) {
            auto dstIssue_level0 = issueEntryMap[dstIssueId];
            for (auto &inIssueId : dstIssue_level0->predecessors) {
                auto inIssue = issueEntryMap[inIssueId];
                filterLtags.insert(inIssue);
            }
        }
    }
    for (auto &dstIssueId : dstIssueList) {
        auto dstIssue_level1 = issueEntryMap[dstIssueId];
        for (auto &inIssueId : dstIssue_level1->predecessors) {
            auto inIssue = issueEntryMap[inIssueId];
            filterLtags.insert(inIssue);
        }
    }
}

bool OoOScheduler::CheckMachineAndL1(IssueEntryPtr spillIssue, IssueEntryPtr allocIssue) {
    auto spillOp = spillIssue->tileOp.GetOpcodeStr();
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && allocIssue->tileOp.GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        (spillOp.find("L0C_COPY_L1") != std::string::npos || spillOp.find("UB_COPY_L1") != std::string::npos)) {
        return false;
    }
    return true;
}

bool OoOScheduler::IsBelongSpillBlackList(IssueEntryPtr spillIssue, IssueEntryPtr issue) {
    std::set<IssueEntryPtr> filterLtags;
    FindFilterLtags(issue, filterLtags);
    if (spillIssue->isAlloc || filterLtags.count(spillIssue) != 0 || !CheckMachineAndL1(spillIssue, issue)) {
        return true;
    }
    return false;
}

IssueEntryPtr OoOScheduler::GetSpillIssue(IssueEntryPtr allocIssue, int memId, bool isGenSpill) {
    if (isGenSpill) {
        return GetBufLastWriteIssue(allocIssue, memId);
    }
    return tensorOccupyMap[localBufferMap[allocIssue->reqMemIds[0]]->memType][memId];
}

Status OoOScheduler::GetGroupNextUseOrder(std::vector<int> group, IssueEntryPtr allocIssue,
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill) {
    std::vector<size_t> bufNextUseTime;
    for (auto& memId : group) {
        IssueEntryPtr spillIssue = GetSpillIssue(allocIssue, memId, isGenSpill);
        if (spillIssue == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }
        if (IsBelongSpillBlackList(spillIssue, allocIssue)) {
            // 存在非法memId时将该group排除
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            bufNextUseTime.push_back(nextUseTimeCache[memId]);
        } else {
            int nextUseOrder = GetBufNextUseOrder(allocIssue, memId);
            if (nextUseOrder == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseOrder);
            bufNextUseTime.push_back(static_cast<size_t>(nextUseOrder));
        }
    }
    if (bufNextUseTime.empty()) {
        groupNextUseTime.push_back(-1);
    } else {
        groupNextUseTime.push_back(*std::min_element(bufNextUseTime.begin(), bufNextUseTime.end()));
    }
    return SUCCESS;
}

bool OoOScheduler::CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType) {
    if (tensors.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "CanAllocateAll tensors is empty.");
        return true;
    }
    auto corePair = tensorAllocCoreMap[tensors[0]->id];
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals = bufferManagerMap[corePair.first][corePair.second][memType].FindFreeIntervals();
    for (auto tensor : tensors) {
        bool canAlloc = false;
        std::pair<uint64_t, uint64_t> newInterval;
        uint64_t allocInterval;
        uint64_t allocAddrStart;
        for (auto &interval : freeIntervals) {
            if (interval.first < tensor->size) {
                continue;
            }
            uint64_t addrStart = interval.second.begin()->first;
            uint64_t addrEnd = interval.second.begin()->second;
            interval.second.erase(addrStart);
            newInterval = {addrStart + tensor->size, addrEnd};
            allocInterval = interval.first;
            allocAddrStart = addrStart;
            canAlloc = true;
            break;
        }
        if (!canAlloc) {
            return false;
        }
        freeIntervals[newInterval.second - newInterval.first].insert(newInterval);
        freeIntervals[allocInterval].erase(allocAddrStart);
        if (freeIntervals[allocInterval].empty()) {
            freeIntervals.erase(allocInterval);
        }
    }
    return true;
}

int OoOScheduler::GetMemidAllocPriority(int memId) {
    for (auto issue : issueEntries) {
        if (!issue->isAlloc) {
            continue;
        }
        if (issue->reqMemIds[0] == memId) {
            return issue->execOrder;
        }
    }
    return -1;
}

bool OoOScheduler::HasEnoughBuffer(IssueEntryPtr allocIssue, MemoryType memType) {
    std::vector<LocalBufferPtr> tensors;
    std::vector<int> memIds;
    if (allocIssue->tileOp.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s must only have one ooperand.", GetFormatBacktrace(allocIssue->tileOp).c_str());
        return false;
    }
    for (auto &dstIssueId : allocIssue->successors) {
        if (&(issueEntryMap[dstIssueId]->tileOp) != 
            *(allocIssue->tileOp.GetOutputOperand(0)->GetProducers().begin())) {
            continue;
        }
        for (auto &memId : issueEntryMap[dstIssueId]->reqMemIds) {
            if (localBufferMap[memId]->memType != memType) {
                continue;
            }
            auto corePair = tensorAllocCoreMap[memId];
            if (bufferManagerMap[corePair.first][corePair.second][memType].isAllocate(memId)) {
                continue;
            }
            if (std::count(memIds.begin(), memIds.end(), memId) == 0) {
                memIds.push_back(memId);
            }
        }
    }
    std::sort(memIds.begin(), memIds.end(), [&](int a, int b) {
        int priorA = GetMemidAllocPriority(a);
        int priorB = GetMemidAllocPriority(b);
        return priorA < priorB;
    });
    for (auto memId : memIds) {
        tensors.push_back(localBufferMap[memId]);
    }
    return CanAllocateAll(tensors, memType);
}

Status OoOScheduler::SelectSpillBuffers(LocalBufferPtr allocBuffer, IssueEntryPtr allocIssue,
    std::vector<int> &spillGroup, bool isGenSpill) {
    // 查找出可以spill 单个或多个tensor的集合
    std::vector<std::vector<int>> canSpillGroups;
    auto corePair = allocIssue->coreLocation;
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].GetSpillGroup(allocBuffer->size, canSpillGroups) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetSpillGroup failed.");
        return FAILED;
    }
    if (canSpillGroups.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill.");
        return FAILED;
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto &group : canSpillGroups) {
        if (GetGroupNextUseOrder(group, allocIssue, groupNextUseTime, nextUseTimeCache, isGenSpill) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "GetGroupNextUseOrder failed.");
            return FAILED;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill.");
        return FAILED;
    }
    spillGroup = canSpillGroups[groupSel];
    return SUCCESS;
}

Status OoOScheduler::RearrangeBuffer(MemoryType memType, std::pair<OpCoreType, int> corePair) {
    std::vector<int> memIds = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto allocIssue = tensorOccupyMap[memType][memId];
        if (allocIssue->tileOp.GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return FAILED;
        }
    }
    return bufferManagerMap[corePair.first][corePair.second][memType].CompactBufferSlices(localBufferMap);
}

Status OoOScheduler::GenBufferSpill(IssueEntryPtr allocIssue) {
    std::vector<int> spillGroup;
    bool spillFailed = false;
    if (SelectSpillBuffers(localBufferMap[allocIssue->reqMemIds[0]], allocIssue, spillGroup, false) != SUCCESS) {
        spillFailed = true;
    }
    size_t temp = 1;
    if (spillFailed) {
        auto corePair = allocIssue->coreLocation;
        MemoryType memType = localBufferMap[allocIssue->reqMemIds[0]]->memType;
        std::vector<int> memIds = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();
        for (auto memId : memIds) {
            auto spillIssue = tensorOccupyMap[memType][memId];
            if (!CheckMachineAndL1(spillIssue, allocIssue) || IsViewOp(spillIssue->tileOp) || spillIssue->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
                continue;
            }
            if (spillIssue->tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
                continue;
            }
            SpillInfo spillInfo;
            if (GetSpillInfo(allocIssue, memId, false, spillInfo) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed. %s", GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
                return FAILED;
            }
            if (SpillBuffer(spillInfo, allocIssue, temp, localBufferMap[allocIssue->reqMemIds[0]], false) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed. %s", memId, GetFormatBacktrace(spillInfo.spillIssue_->tileOp).c_str());
                return FAILED;
            }
        }
        // Alloc内存整理
        if (RearrangeBuffer(memType, corePair) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed at GenBufferSpill. %s", GetFormatBacktrace(allocIssue->tileOp).c_str());
        }
        if (!HasEnoughBuffer(allocIssue, memType)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(allocIssue->tileOp).c_str());
            PrintSpillFailedInfo(allocIssue, memType);
            return FAILED;
        }
    } else {
        if (SpillMultiBuffer(allocIssue, spillGroup, temp, localBufferMap[allocIssue->reqMemIds[0]], false) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiBuffer failed! %s", GetFormatBacktrace(allocIssue->tileOp).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillOp(LocalBufferPtr allocBuffer, size_t &pcIdx) {
    APASS_LOG_DEBUG_F(Elements::Operation, "START: SPILL tensor.");
    if (allocBuffer->memType != MemoryType::MEM_L1 && allocBuffer->memType != MemoryType::MEM_UB) {
        if (PrintSpillFailedInfo(issueEntries[pcIdx]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed; Please check the PrintSpillFailedInfo method.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
        return FAILED;
    }
    // 选择最晚被使用的spill 单个或多个tensor	
    std::vector<int> spillGroup;	
    SelectSpillBuffers(allocBuffer, issueEntries[pcIdx], spillGroup, true);
    if (spillGroup.empty()) {	
        MemoryType memType = allocBuffer->memType;
        auto corePair = issueEntries[pcIdx]->coreLocation;
        std::vector<int> memIds = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();
        for (auto memId : memIds) {	
            auto spillIssue = GetBufLastWriteIssue(issueEntries[pcIdx], memId);	
            if (!CheckMachineAndL1(spillIssue, issueEntries[pcIdx]) || spillIssue->tileOp.GetOpcode() == Opcode::OP_VIEW || spillIssue->tileOp.GetOpcode() == Opcode::OP_VIEW_TYPE || spillIssue->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
                continue;
            }
            if (spillIssue->tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
                continue;
            }
            SpillInfo spillInfo;
            if (GetSpillInfo(issueEntries[pcIdx], memId, true, spillInfo) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed; Please check the GetSpillInfo method.");
                return FAILED;
            }
            if (SpillBuffer(spillInfo, issueEntries[pcIdx], pcIdx, allocBuffer, true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed.", memId);
                return FAILED;
            }
        }
        if (!HasEnoughBuffer(issueEntries[pcIdx], memType)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed!");
            if (PrintSpillFailedInfo(issueEntries[pcIdx]) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed; Please check the PrintSpillFailedInfo method.");
                return FAILED;
            }
            return FAILED;
        }
    } else {
        if (SpillMultiBuffer(issueEntries[pcIdx], spillGroup, pcIdx, allocBuffer, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiBuffer failed!");
            return FAILED;
        }
    }	
    APASS_LOG_DEBUG_F(Elements::Operation, "END: SPILL tensor.");	
    return SUCCESS;	
}

} // namespace npu::tile_fwk
