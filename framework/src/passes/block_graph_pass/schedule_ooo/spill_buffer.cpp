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

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;

static bool IsAnyDimSmaller(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] < b[i]) {
            return true;
        }
    }
    return false;
}

bool OoOScheduler::IsSupportedPartialWriteProducer(const Operation &op) const
{
    return op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_L0C_TO_L1 ||
        IsAllocOpCode(op.GetOpcode());
}

bool OoOScheduler::IsReusableCopyInOp(const Operation &op) const
{
    if (!OpcodeManager::Inst().IsBoundaryIn(op.GetOpcode())) {
        return false;
    }
    auto input = op.GetInputOperand(0);
    auto output = op.GetOutputOperand(0);
    if (input == nullptr || output == nullptr) {
        return false;
    }
    auto inputMem = input->GetMemoryTypeOriginal();
    auto outputMem = output->GetMemoryTypeOriginal();
    return inputMem == MemoryType::MEM_DEVICE_DDR &&
        (outputMem == MemoryType::MEM_L1 || outputMem == MemoryType::MEM_UB);
}

Status OoOScheduler::GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t> &toOffset,
    std::vector<SymbolicScalar> &toDynOffset, std::vector<SymbolicScalar> &fromDynValidShape) const
{
    if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto attr = std::static_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid AssembleOpAttribute.");
            return FAILED;
        }
        toOffset = attr->GetToOffset();
        toDynOffset = attr->GetToDynOffset();
        fromDynValidShape = attr->GetFromDynValidShape();
        return SUCCESS;
    } else if (producerOp->GetOpcode() == Opcode::OP_L0C_TO_L1) {
        auto attr = std::static_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid CopyOpAttribute.");
            return FAILED;
        }
        auto iOperand = producerOp->GetInputOperand(0);
        for (const auto &offsetImm : attr->GetToOffset()) {
            if (!offsetImm.IsSpecified() || !offsetImm.GetSpecifiedValue().ConcreteValid()) {
                APASS_LOG_ERROR_F(Elements::Operation, "L0C_TO_L1 replay only supports static concrete toOffset.");
                return FAILED;
            }
            toOffset.push_back(static_cast<int64_t>(offsetImm.GetSpecifiedValue()));
        }
        fromDynValidShape = iOperand->GetDynValidShape();
        if (fromDynValidShape.empty() && !attr->GetToDynValidShape().empty()) {
            fromDynValidShape = OpImmediate::ToSpecified(attr->GetToDynValidShape());
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "Unsupported producer opcode in SpillParticalBuffer.");
    return FAILED;
}

Operation* OoOScheduler::FindAllocForAssembleProducers(const std::vector<Operation*> &assembleOps) const
{
    for (auto *op : assembleOps) {
        if (op->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (auto *producer : op->ProducerOps()) {
            if (producer->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                return producer;
            }
        }
    }
    return nullptr;
}

bool OoOScheduler::HasNZHorizontalSlice(const std::vector<Operation*> &assembleOps) const
{
    for (auto *op : assembleOps) {
        if (op->GetOpcode() != Opcode::OP_L0C_TO_L1) {
            continue;
        }
        int64_t isNZ = 0;
        op->GetAttr(OpAttributeKey::copyIsNZ, isNZ);
        if (isNZ == 0) {
            continue;
        }
        auto attr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (attr == nullptr) {
            continue;
        }
        for (const auto &offsetImm : attr->GetToOffset()) {
            if (!offsetImm.IsSpecified() || !offsetImm.GetSpecifiedValue().ConcreteValid()) {
                return true; // dynamic offset, conservatively reject
            }
        }
        // NZ format: offset[0] is the M (row) dimension; non-zero means horizontal slice
        const auto &toOffset = attr->GetToOffset();
        if (!toOffset.empty() && static_cast<int64_t>(toOffset[0].GetSpecifiedValue()) != 0) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Reject spill-assemble for %s[%d]: NZ horizontal slice (toOffset[0]=%lld).",
                op->GetOpcodeStr().c_str(), op->GetOpMagic(),
                static_cast<long long>(static_cast<int64_t>(toOffset[0].GetSpecifiedValue())));
            return true;
        }
    }
    return false;
}

int OoOScheduler::GetBufNextUseOrder(Operation* op, int curMemId) {
    (void)op;
    for (size_t i = 0; i < orderedOps.size(); ++i) {
        auto& a = orderedOps[i];
        if (opIsRetiredMap[a]) continue;
        auto& reqMemIds = GetOpMemIds(a);
        if (std::find(reqMemIds.begin(), reqMemIds.end(), curMemId) != reqMemIds.end()) {
            return opExecOrderMap[a];
        }
    }
    return -1;
}

int OoOScheduler::GetBufLastUseOrder(Operation* op, int curMemId) {
    auto targetIt = std::find(orderedOps.begin(), orderedOps.end(), op);
    if (targetIt == orderedOps.end()) {
        return -1;
    }
    int execOrder = opExecOrderMap[op];
    for (auto it = std::make_reverse_iterator(targetIt); it != orderedOps.rend(); it++) {
        Operation* curOp = *it;
        if (curOp && opExecOrderMap[curOp] < execOrder) {
            auto& reqMemIds = GetOpMemIds(curOp);
            if (std::find(reqMemIds.begin(), reqMemIds.end(), curMemId) != reqMemIds.end()) {
                return opExecOrderMap[curOp];
            }
        }
    }
    return -1;
}

Operation* OoOScheduler::GetBufLastWriteOp(Operation* op, int curMemId) {
    auto targetIt = std::find(orderedOps.begin(), orderedOps.end(), op);
    if (targetIt == orderedOps.end()) {
        return nullptr;
    }
    int execOrder = opExecOrderMap[op];
    for (auto it = std::make_reverse_iterator(targetIt); it != orderedOps.rend(); it++) {
        Operation* curOp = *it;
        if (curOp == nullptr || opExecOrderMap[curOp] >= execOrder) {
            continue;
        }
        for (auto& outTensor : curOp->GetOOperands()) {
            if (outTensor->memoryrange.memId == curMemId) {
                return curOp;
            }
        }
    }
    return nullptr;
}

Status OoOScheduler::UpdateTensorAttr(
    LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId)
{
    tensor->SetMemoryTypeToBe(memType);
    tensor->SetMemoryTypeOriginal(memType);
    tensor->oriShape = spillTensor->oriShape;
    tensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    tensor->tensor->rawshape = spillTensor->tensor->rawshape;
    if (memType == MEM_DEVICE_DDR) {
        if (localBufferMap_.find(spillMemId) == localBufferMap_.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap_.", spillMemId);
            return FAILED;
        }
        tensor->memoryrange =
            TileRange(workspaceOffset, workspaceOffset + localBufferMap_[spillMemId]->size, workspaceMemId++);
        workspaceOffset += localBufferMap_[spillMemId]->size;
    } else {
        int rawMagic = tensor->GetRawTensor()->GetRawMagic();
        tensor->memoryrange.memId = rawMagic;
        localBufferMap_[rawMagic] =
            std::make_shared<LocalBuffer>(rawMagic, tensor->tensor->GetRawDataSize(), tensor->GetMemoryTypeOriginal());
        if (localBufferMap_[rawMagic] == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Init Tensor[%d] localBuffer failed.", rawMagic);
            return FAILED;
        }
        tensorAllocCoreMap[tensor->memoryrange.memId] = tensorAllocCoreMap[spillTensor->memoryrange.memId];
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp) {
    if (srcOp->GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(srcOp->GetInternalSubgraphID());
        op.SetAIVCore(srcOp->GetAIVCore());
    }
}

void OoOScheduler::UpdateOpIsCube(Operation &op, Operation* srcOp)
{
    if (srcOp->HasAttribute(OpAttributeKey::isCube)) {
        op.SetAttribute(OpAttributeKey::isCube, srcOp->GetBoolAttribute(OpAttributeKey::isCube));
    }
}

// A5 中：L1 发生 spill 时， copy_in的rawshape属性来源于实际spill的tensor，offset属性来源于spill op
void OoOScheduler::GetActualSpillInfo(Operation* spillOp, std::pair<LogicalTensorPtr, Operation*>& actualInfo)
{
    auto actualSpillTensor = spillOp->GetInputOperand(0);
    auto copyOp = spillOp;
    Operation* actualSpillOp = nullptr;
    for (auto &op : depManager_.GetPredecessors(spillOp)) {
        if (!opIsAllocMap[op]) {
            actualSpillOp = op;
        }
    }
    if (spillOp->GetOpcode() == Opcode::OP_RESHAPE && actualSpillOp) {
        actualSpillTensor = actualSpillOp->GetInputOperand(0);
        copyOp = actualSpillOp;
        for (auto &preOp : depManager_.GetPredecessors(actualSpillOp)) {
            if (!opIsAllocMap[preOp]) {
                actualSpillOp = preOp;
            }
        }
    }
    if (actualSpillOp->GetOpcodeStr().find("UB_COPY_ND2NZ") != std::string::npos) {
        // UB_COPY_ND2NZ 没有 CopyOpAttribute, 故只能从 UB_COPY_L1 上获取
        actualSpillTensor = actualSpillOp->GetInputOperand(0);
    }
    actualInfo = std::make_pair(actualSpillTensor, copyOp);
}

void OoOScheduler::UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor,
    std::vector<int64_t> offset, Operation* spillOp, bool isSpecialL1)
{
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(spillTensor->GetMemoryTypeOriginal(),
            OpImmediate::Specified(offset), OpImmediate::Specified(spillTensor->GetShape()),
            OpImmediate::Specified(spillTensor->GetRawTensor()->GetDynRawShape())));
    } else if (op.GetOpcodeStr().find("ALLOC") == std::string::npos) {
        if (IsReusableCopyInOp(*spillOp)) {
            op.SetOpAttribute(spillOp->GetOpAttribute()->Clone());
            op.inParamLocation_ = spillOp->inParamLocation_;
        } else if (!isSpecialL1) {
            int64_t isNZ = 0;
            spillOp->GetAttr(OpAttributeKey::copyIsNZ, isNZ);
            op.SetAttr(OpAttributeKey::copyIsNZ, isNZ);
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), spillTensor->GetMemoryTypeOriginal(),
                OpImmediate::Specified(spillTensor->GetShape()),
                OpImmediate::Specified(spillTensor->tensor->GetDynRawShape())));
        } else {
            std::pair<LogicalTensorPtr, Operation*> actualInfo;
            GetActualSpillInfo(spillOp, actualInfo);
            auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(actualInfo.second->GetOpAttribute());
            if (attr == nullptr) {
                APASS_LOG_INFO_F(Elements::Tensor, "Op %s attribute is nullptr", GetOpInfo(actualInfo.second).c_str());
                return;
            }
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                attr->GetFromOffset(), spillTensor->GetMemoryTypeOriginal(),
                OpImmediate::Specified(spillTensor->GetShape()),
                OpImmediate::Specified(actualInfo.first->tensor->GetDynRawShape())));
        }
    }
    op.UpdateLatency(opLatency);
}

void OoOScheduler::ReplaceViewOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId)
{
    std::vector<Operation*> viewConsumers;
    for (auto* consumer : startTensor->GetConsumers()) {
        if (IsViewOp(*consumer)) {
            viewConsumers.push_back(consumer);
        }
    }

    while (!viewConsumers.empty()) {
        Operation* viewOp = viewConsumers.back();
        viewConsumers.pop_back();
        auto viewOutTensor = viewOp->GetOutputOperand(0);
        if (viewOutTensor == nullptr) {
            continue;
        }
        if (viewOutTensor->memoryrange.memId == oldMemId) {
            viewOutTensor->memoryrange.memId = newMemId;
        }
        for (auto* consumer : viewOutTensor->GetConsumers()) {
            if (IsViewOp(*consumer)) {
                viewConsumers.push_back(consumer);
            }
        }
    }
}

void OoOScheduler::ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId)
{
    auto& reqMemIds = GetOpMemIds(op);
    for (auto memId : reqMemIds) {
        if (memId == oldMemId) {
            std::replace(reqMemIds.begin(), reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
            ReplaceViewOpChainMemId(outTensor, oldMemId, newMemId);
        }
    }
}

Status OoOScheduler::UpdateRemainOpBufId(int oldMemId, int newMemId) {
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]. ", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = bufRefCount_[oldMemId] + TWO_ISSUE;
    bufRefCount_[oldMemId] = 0;
    for (auto& op : orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        ReplaceTensorMemId(op, oldMemId, newMemId);
    }
    return SUCCESS;
}

void OoOScheduler::UpdateTensorInputFor(Operation* targetOp, Operation* spillSrcOp, LogicalTensorPtr tensor) {
    for (size_t index = 0; index < targetOp->GetIOperands().size(); index++) {
        UpdateTensorInputForOperand(targetOp, index, spillSrcOp, tensor);
    }
}

void OoOScheduler::UpdateTensorInputForOperand(Operation* targetOp, size_t index, Operation* spillSrcOp, LogicalTensorPtr tensor) {
    for (auto &inOp : targetOp->GetIOperands()[index]->GetProducers()) {
        if (IsViewOp(*inOp)) {
            Operation* op = SkipViewChain(inOp, true);
            UpdateTensorInputForView(*op, spillSrcOp, tensor);
        } else if (inOp == spillSrcOp) {
            targetOp->UpdateInputOperand(index, tensor);
        }
    }
}

void OoOScheduler::UpdateTensorInputForView(Operation& op, Operation* spillSrcOp, LogicalTensorPtr tensor) {
    bool hit = false;
    for (auto it : op.GetInputOperand(0)->GetProducers()) {
        if (it == spillSrcOp) {
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

Status OoOScheduler::UpdateReloadIssueDepend(Operation* reloadCopyin, Operation* spillOp, int spillMemId) {
    auto& successors = depManager_.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (!opIsRetiredMap[succOp]) {
            auto& reqMemIds = GetOpMemIds(succOp);
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
                depManager_.InsertSuccessor(reloadCopyin, succOp);
                if (depManager_.RemovePredecessor(succOp, spillOp) == 0) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Erase op %s failed. %s", GetOpInfo(spillOp).c_str(), GetFormatBacktrace(*spillOp).c_str());
                    return FAILED;
                }
                depManager_.InsertPredecessor(succOp, reloadCopyin);
                if (reloadCopyin->GetOutputOperand(0) == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation, "%s cannot find oOperand[0]. %s", GetOpInfo(reloadCopyin).c_str(), GetFormatBacktrace(*reloadCopyin).c_str());
                    return FAILED;
                }
                UpdateTensorInputFor(succOp, spillOp, reloadCopyin->GetOutputOperand(0));
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::InsertOrdered(Operation* insertOp) {
    int execOrder = opExecOrderMap[insertOp];
    auto it = orderedOps.begin();
    for (; it != orderedOps.end(); it++) {
        if (opExecOrderMap[*it] >= execOrder) {
            break;
        }
    }
    auto insertPos = orderedOps.insert(it, insertOp);
    // 更新后续元素的execOrder
    for (auto adjustIt = insertPos + 1; adjustIt != orderedOps.end(); adjustIt++) {
        if (opExecOrderMap[*adjustIt] >= execOrder) {
            opExecOrderMap[*adjustIt]++;
        }
    }
}

Status OoOScheduler::UpdateReloadIssueInfo(Operation* reloadAlloc, Operation* reloadCopyin, Operation* spillOp, int spillMemId, Operation* allocOp) {
    SetOpMemIds(reloadAlloc, {reloadAlloc->GetOutputOperand(0)->memoryrange.memId});
    depManager_.AddAllocDependency(reloadAlloc, reloadCopyin);
    SetOpMemIds(reloadCopyin, {reloadAlloc->GetOutputOperand(0)->memoryrange.memId});
    int bufNextUseOrder = GetBufNextUseOrder(allocOp, spillMemId);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use order failed.", spillMemId);
        return FAILED;
    }
    // 初始化 reloadAlloc 和 reloadCopyin 的 map 属性
    opIsAllocMap[reloadAlloc] = true;
    opIsAllocMap[reloadCopyin] = false;
    opIsRetiredMap[reloadAlloc] = false;
    opIsRetiredMap[reloadCopyin] = false;
    opPipeTypeMap[reloadAlloc] = RescheduleUtils::GetOpPipeType(reloadAlloc);
    opPipeTypeMap[reloadCopyin] = RescheduleUtils::GetOpPipeType(reloadCopyin);

    opExecOrderMap[reloadAlloc] = bufNextUseOrder++;
    InsertOrdered(reloadAlloc);
    opExecOrderMap[reloadCopyin] = bufNextUseOrder;
    InsertOrdered(reloadCopyin);
    opCoreLocationMap[reloadAlloc] = opCoreLocationMap[allocOp];
    opCoreLocationMap[reloadCopyin] = opCoreLocationMap[allocOp];
    UpdateOpInternalSubgraphID(*reloadAlloc, allocOp);
    UpdateOpInternalSubgraphID(*reloadCopyin, allocOp);
    UpdateOpIsCube(*reloadCopyin, *(depManager_.GetSuccessors(spillOp).begin()));
    if (UpdateReloadIssueDepend(reloadCopyin, spillOp, spillMemId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainOpBufId(spillMemId, GetOpMemIds(reloadAlloc)[0])) {
        return FAILED;
    }
    for (auto& op : orderedOps) {
        if (opIsRetiredMap[op] || opIsAllocMap[op]) {
            continue;
        }
        auto predecessors = depManager_.GetPredecessors(op);
        for (auto predOp : predecessors) {
            if (opIsAllocMap[predOp]) {
                auto& predReqMemIds = GetOpMemIds(predOp);
                if (std::find(predReqMemIds.begin(), predReqMemIds.end(), spillMemId) != predReqMemIds.end()) {
                    depManager_.RemovePredecessor(op, predOp);
                    depManager_.InsertPredecessor(op, reloadAlloc);
                }
            }
        }
    }

    numTotalIssues += TWO_ISSUE;
    return SUCCESS;
}

Status OoOScheduler::CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor,
    LogicalTensorPtr spillTensor, Operation* spillOp, std::pair<Operation*, Operation*> &reloadOps,  bool isSpecialL1)
{
    MemoryType memType = spillTensor->GetMemoryTypeOriginal();
    // 创建将spill搬出数据搬回OP_COPY_IN的tensor
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(
            function_, spillTensor->Datatype(), spillTensor->shape, spillTensor->Format());
    if (UpdateTensorAttr(localTensor, memType, spillTensor, -1) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "UpdateTensorAttr local tensor failed!");
        return FAILED;
    }
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    // 创建spill搬出数据搬回OP_COPY_IN/OP_ALLOC
    Opcode allocOp = memType == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    auto &spillAllocOp = function_.AddRawOperation(allocOp, {}, {localTensor});
    auto &spillCopyInOp = IsReusableCopyInOp(*spillOp) ?
        spillOp->CloneOperation(function_, {spillOutTensor}, {localTensor}) :
        function_.AddRawOperation(Opcode::OP_COPY_IN, {spillOutTensor}, {localTensor});

    if (IsReusableCopyInOp(*spillOp)) {
        spillCopyInOp.SetIOpAttrOffset(0, spillOp->GetIOpAttrOffset(0));
    }
    // 设置ODO copy_in op offset
    UpdateOpAttr(spillAllocOp, 1, localTensor, {}, spillOp, isSpecialL1);
    UpdateOpAttr(spillCopyInOp, DEFAULT_LATENCY, localTensor, spillOutTensor->GetOffset(), spillOp, isSpecialL1);
    // DDR->COPY_IN->spillTensor 场景不标记 copy_in_mode
    if (!IsReusableCopyInOp(*spillOp) && UpdateCopyInMode(spillCopyInOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyInMode failed!");
        return FAILED;
    }
    // 初始化Operation属性到map
    Operation* allocOpPtr = &spillAllocOp;
    Operation* copyInOpPtr = &spillCopyInOp;

    // 初始化 spillAllocOp
    opExecOrderMap[allocOpPtr] = static_cast<int>(orderedOps.size()) - 1;
    opPipeTypeMap[allocOpPtr] = RescheduleUtils::GetOpPipeType(allocOpPtr);
    opIsAllocMap[allocOpPtr] = true;
    opIsRetiredMap[allocOpPtr] = false;
    SetOpMemIds(allocOpPtr, {});

    depManager_.RegisterOp(allocOpPtr);
    InitOpViewOps(allocOpPtr);

    // 初始化 spillCopyInOp
    opExecOrderMap[copyInOpPtr] = static_cast<int>(orderedOps.size()) - 1;
    opPipeTypeMap[copyInOpPtr] = RescheduleUtils::GetOpPipeType(copyInOpPtr);
    opIsAllocMap[copyInOpPtr] = false;
    opIsRetiredMap[copyInOpPtr] = false;
    SetOpMemIds(copyInOpPtr, {});

    depManager_.RegisterOp(copyInOpPtr);
    InitOpViewOps(copyInOpPtr);

    reloadOps.first = allocOpPtr;
    reloadOps.second = copyInOpPtr;
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_ALLOC: %s. ", GetOpInfo(allocOpPtr).c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_IN: %s. ", GetOpInfo(copyInOpPtr).c_str());
    return SUCCESS;
}

Status OoOScheduler::UpdateReshapeDependAndBuf(Operation* allocOp, SpillInfo &spillInfo, LogicalTensorPtr reshapeTensor) {
    auto& coreLocation = opCoreLocationMap[allocOp];
    // 依赖 reqmemId
    if (bufferManagerMap[coreLocation][spillInfo.spillTensor_->GetMemoryTypeOriginal()].Free(
            spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    if (UpdateRemainOpBufId(spillInfo.spillMemId_, reshapeTensor->memoryrange.memId)) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainOpBufId failed.");
        return FAILED;
    }
    bufRefCount_[reshapeTensor->memoryrange.memId] = 0;
    for (auto op: orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        auto& reqMemIds = GetOpMemIds(op);
        for (auto memId : reqMemIds) {
            if (memId == reshapeTensor->memoryrange.memId) {
                bufRefCount_[reshapeTensor->memoryrange.memId]++;
            }
        }
    }
    depManager_.InitDependencies(orderedOps, false);
    return SUCCESS;
}

LogicalTensorPtr OoOScheduler::CreateReshapeL1Tensor(LogicalTensorPtr iOperand, LogicalTensorPtr reshapeTensor)
{
    LogicalTensorPtr newTensor =
        std::make_shared<LogicalTensor>(function_, iOperand->Datatype(), iOperand->shape, iOperand->Format());
    newTensor->SetMemoryTypeToBe(iOperand->GetMemoryTypeToBe());
    newTensor->SetMemoryTypeOriginal(iOperand->GetMemoryTypeOriginal());
    newTensor->oriShape = iOperand->shape;
    newTensor->tensor = reshapeTensor->tensor;
    newTensor->memoryrange.memId = reshapeTensor->memoryrange.memId;
    newTensor->UpdateDynValidShape(iOperand->GetDynValidShape());
    newTensor->offset = iOperand->GetOffset();
    tensorAllocCoreMap[newTensor->memoryrange.memId] = tensorAllocCoreMap[iOperand->memoryrange.memId];
    return newTensor;
}

Status OoOScheduler::SpillReshapeParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, LogicalTensorPtr reshapeTensor, bool isGenSpill) {
    auto iOperand = spillInfo.spillOp_->GetInputOperand(0);
    LogicalTensorPtr newTensor = CreateReshapeL1Tensor(iOperand, reshapeTensor);
    int bufNextUseOrder = GetBufNextUseOrder(allocOp, spillInfo.spillMemId_);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get Tensor[%d] next use order failed.", spillInfo.spillMemId_);
        return FAILED;
    }
    // 创建 alloc
    auto& spillAllocOp = function_.AddRawOperation(Opcode::OP_L1_ALLOC, {}, {newTensor});
    spillAllocOp.UpdateLatency(1);
    auto spillAllocOpPtr = UpdateIssueAttr(spillAllocOp, {reshapeTensor->memoryrange.memId}, allocOp, bufNextUseOrder, isGenSpill);
    // 创建 copyin
    Operation* preOp = nullptr;
    auto& predecessors = depManager_.GetPredecessors(spillInfo.spillOp_);
    for (auto predOp : predecessors) {
        if (!opIsAllocMap[predOp]) {
            preOp = predOp;
        }
    }
    if (preOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "preOp is nullptr");
        return FAILED;
    }
    if (preOp->GetOpcode() != Opcode::OP_COPY_IN && preOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
            preOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation, "The preOp of reshape is not COPY_IN/UB_COPY_L1/L0C_COPY_L1");
        return FAILED;
    }
    auto &spillCopyInOp = (preOp->GetOpcode() == Opcode::OP_COPY_IN) ?
        preOp->CloneOperation(function_, {spillInfo.ddrTensor_}, {newTensor}) :
        function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {newTensor});
    if (preOp->GetOpcode() == Opcode::OP_COPY_IN) {
        spillCopyInOp.SetIOpAttrOffset(0, preOp->GetIOpAttrOffset(0));
    }
    UpdateOpAttr(spillCopyInOp, DEFAULT_LATENCY, newTensor, spillInfo.ddrTensor_->GetOffset(), preOp,
        spillInfo.isSpecialL1_);
    auto spillCopyInOpPtr = UpdateIssueAttr(spillCopyInOp, {reshapeTensor->memoryrange.memId}, allocOp, bufNextUseOrder, isGenSpill);
    UpdateOpIsCube(spillCopyInOp, spillInfo.spillOp_);
    // A5 下 DDR->COPY_IN->L1->RESHAPE->L1 场景不标记 copy_in_mode
    if (preOp->GetOpcode() != Opcode::OP_COPY_IN && UpdateCopyInMode(spillCopyInOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyInMode failed");
        return FAILED;
    }
    // 创建 reshape
    auto& reshapeOp = function_.AddRawOperation(Opcode::OP_RESHAPE, {newTensor}, {reshapeTensor});
    UpdateOpIsCube(reshapeOp, spillInfo.spillOp_);
    reshapeOp.UpdateLatency(1);
    auto reshapeOpPtr = UpdateIssueAttr(reshapeOp, {reshapeTensor->memoryrange.memId, reshapeTensor->memoryrange.memId}, allocOp, bufNextUseOrder, isGenSpill);
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_ALLOC: %s. ", GetOpInfo(spillAllocOpPtr).c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_IN: %s. ", GetOpInfo(spillCopyInOpPtr).c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_RESHAPE: %s. ", GetOpInfo(reshapeOpPtr).c_str());
    return SUCCESS;
}

Status OoOScheduler::SpillInReshapeBuffer(SpillInfo &spillInfo, Operation* allocOp, bool isGenSpill) {
    LogicalTensorPtr reshapeTensor = std::make_shared<LogicalTensor>(function_,
        spillInfo.spillTensor_->Datatype(), spillInfo.spillTensor_->shape, spillInfo.spillTensor_->Format());
    if (reshapeTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Create reshape tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(reshapeTensor, spillInfo.spillTensor_->GetMemoryTypeOriginal(), spillInfo.spillTensor_, -1) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateTensorAttr reshape tensor failed!");
        return FAILED;
    }
    auto &successors = depManager_.GetSuccessors(spillInfo.spillOp_);
    for (auto succOp : successors) {
        if (!opIsRetiredMap[succOp]) {
            auto& reqMemIds = GetOpMemIds(succOp);
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillInfo.spillMemId_) > 0) {
                UpdateTensorInputFor(succOp, spillInfo.spillOp_, reshapeTensor);
            }
        }
    }
    if (SpillReshapeParticalBuffer(spillInfo, allocOp, reshapeTensor, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillReshapeParticalBufferOp failed!");
        return FAILED;
    }
    // 依赖关系 memId
    if (UpdateReshapeDependAndBuf(allocOp, spillInfo, reshapeTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateReshapeDependAndBufOp failed!");
        return FAILED;
    }

    return SUCCESS;
}

Status OoOScheduler::SpillInBuffer(SpillInfo &spillInfo, Operation* allocOp, MemoryType bufferType,
    bool isGenSpill) {
    if (spillInfo.isSpecialL1_ && spillInfo.spillOp_->GetOpcodeStr().find("RESHAPE") != std::string::npos) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Start to spill-reshape special L1 in A5.");
        if (SpillInReshapeBuffer(spillInfo, allocOp, isGenSpill) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillInReshapeBuffer failed!");
            return FAILED;
        }
        return SUCCESS;
    }
    Operation* reloadCopyin = nullptr;
    Operation* reloadAlloc = nullptr;
    std::pair<Operation*, Operation*> reloadOps = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(spillInfo.ddrTensor_, spillInfo.spillTensor_, spillInfo.spillOp_,
        reloadOps, spillInfo.isSpecialL1_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillReloadIssue failed!");
        return FAILED;
    }
    reloadAlloc = reloadOps.first;
    reloadCopyin = reloadOps.second;
    if (UpdateReloadIssueInfo(reloadAlloc, reloadCopyin, spillInfo.spillOp_, spillInfo.spillMemId_,
        allocOp) != SUCCESS) {
        reloadOps.first->SetAsDeleted();
        reloadOps.second->SetAsDeleted();
        function_.EraseOperations();
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateReloadIssueInfo failed!");
        return FAILED;
    }
    auto coreLocation = opCoreLocationMap[allocOp];
    if (!isGenSpill) {
        opExecOrderMap[allocIssueQueue[coreLocation][bufferType].Front()] > opExecOrderMap[reloadAlloc] ?
        allocIssueQueue[coreLocation][bufferType].ForceInsertAfterFirst(reloadAlloc) :
        allocIssueQueue[coreLocation][bufferType].Insert(reloadAlloc);
    }
    if (bufferManagerMap[coreLocation][bufferType].Free(spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::CreateSpillCopyout(Operation* spillOp, LogicalTensorPtr spillTensor,
    int spillMemId, Operation* &spillCopyoutOp, const SpillInfo &spillInfo) {
    // 创建spill搬出所需的DDR rawtensor/tensor
    // A5 中 L1-spill 在 spillIssue 为 L0C_L1 时, 需要设置 L0C_COPY_OUT 搬出的 DDR 的 dtype 与需要搬入的 L1 一致,
    // 其余情况和实际搬出 tensor 的 dtype 一致 L0C----L0C_COPY_L1------L1 L0C->L0C_COPY_OUT->DDR->L1_COPY_IN
    std::shared_ptr<RawTensor> ddrRawTensor =
        (spillInfo.spillOp_ != spillOp && spillTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) ?
        std::make_shared<RawTensor>(spillInfo.spillTensor_->Datatype(), spillTensor->tensor->rawshape,
        TileOpFormat::TILEOP_ND, "WorkspaceGm") :
        std::make_shared<RawTensor>(spillTensor->Datatype(), spillTensor->tensor->rawshape,
        TileOpFormat::TILEOP_ND, "WorkspaceGm");
    if (ddrRawTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR raw tensor failed!");
        return FAILED;
    }
    std::vector<int64_t> offset = spillTensor->GetOffset();

    LogicalTensorPtr ddrTensor =
        std::make_shared<LogicalTensor>(function_, ddrRawTensor, offset, spillTensor->GetShape());
    if (ddrTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR tensor failed!");
        return FAILED;
    }
    // workspaceOffset 会在 UpdateTensorAttr 中被更新，但 UpdateOpAttr 需要使用当前的 workspaceOffset
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceOffset);
    if (UpdateTensorAttr(ddrTensor, MEM_DEVICE_DDR, spillTensor, spillMemId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "UpdateTensorAttr DDR tensor failed!");
        return FAILED;
    }

    // 创建spill搬出所需的DDR OP_COPY_OUT
    Operation &spillOutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {spillTensor}, {ddrTensor});
    spillCopyoutOp = &spillOutOp;
    UpdateOpAttr(spillOutOp, DEFAULT_LATENCY, spillTensor, offset, spillOp, spillInfo.isSpecialL1_);
    if (spillInfo.spillOp_ != spillOp && spillTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        // L0C_L1 上的 op_attr_scale_value 属性迁移至 L0C_COPY_OUT 上
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        if (spillInfo.spillOp_->GetAttr(OpAttributeKey::scaleValue, scaleValue)) {
            spillOutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Transfer scaleValue %s from L0C_COPY_L1 to L0C_COPY_OUT",
                std::to_string(scaleValue.GetUnsignedData()).c_str());
        }
    }
    // 设置 spillCopyoutOp 的属性
    SetOpMemIds(spillCopyoutOp, {spillMemId});
    depManager_.AddDependency(spillOp, spillCopyoutOp);
    opIsRetiredMap[spillCopyoutOp] = true;
    opIsAllocMap[spillCopyoutOp] = false;
    opPipeTypeMap[spillCopyoutOp] = RescheduleUtils::GetOpPipeType(spillCopyoutOp);
    opViewOpsMap[spillCopyoutOp] = std::vector<Operation*>();
    UpdateOpIsCube(*spillCopyoutOp, *(depManager_.GetSuccessors(spillOp).begin()));
    for (auto preOp : depManager_.GetPredecessors(spillOp)) {
        if (opIsAllocMap[preOp]) {
            opCoreLocationMap[spillCopyoutOp] = opCoreLocationMap[preOp];
            UpdateOpInternalSubgraphID(*spillCopyoutOp, preOp);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_OUT: %s. ", GetOpInfo(spillCopyoutOp).c_str());
    return SUCCESS;
}

Status OoOScheduler::UpdateCopyOutMode(Operation& copyOutOp)
{
    // L0C spill 搬出 DDR 时统一按 ND 布局写出；普通 L1 spill 保持原有架构相关模式。
    auto input = copyOutOp.GetInputOperand(0);
    if (!input) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "CopyOutOp %s[%d] does not have inputOperand", copyOutOp.GetOpcodeStr().c_str(),
            copyOutOp.GetOpMagic());
        return FAILED;
    }
    if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        copyOutOp.SetAttribute(OpAttributeKey::copyIsNZ, 0);
    } else if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 &&
               input->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        copyOutOp.SetAttribute(OpAttributeKey::copyOutMode, static_cast<int64_t>(Matrix::CopyOutMode::ND2ND));
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateCopyInMode(Operation& copyInOp)
{
    // A5 上 L1_COPY_IN 设置为 ND_NZ, A2/A3 上 L1_COPY_IN 设置为 ND_ND
    auto output = copyInOp.GetOutputOperand(0);
    if (!output) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "CopyInOp %s[%d] does not have outputOperand", copyInOp.GetOpcodeStr().c_str(),
            copyInOp.GetOpMagic());
        return FAILED;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (copyInOp.GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            copyInOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
        }
    } else {
        if (copyInOp.GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            copyInOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2ND));
        }
    }
    return SUCCESS;
}

Status OoOScheduler::HandleReshapeSpillPath(SpillInfo &spillInfo, Operation* &actualSpillOp,
    LogicalTensorPtr &actualSpillTensor, bool &isFinish, bool isGenSpill, size_t &pcIdx)
{
    if (actualSpillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        spillInfo.ddrTensor_ = actualSpillOp->GetInputOperand(0);
        isFinish = true;
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill out finish in A5: DDR->copy_in->L1->reshape->L1");
        return SUCCESS;
    }
    if (actualSpillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        actualSpillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOp is Reshape, preop: %s, ioperand of L1: %s",
            GetOpInfo(actualSpillOp).c_str(),
            MemoryTypeToString(actualSpillOp->GetInputOperand(0)->GetMemoryTypeOriginal()).c_str());
        return FAILED;
    }
    // RESHAPE 路径下，若 actualSpillOp 是 UB_COPY_L1 / L0C_COPY_L1
    // 且 input.shape 任一维 < output.shape，同样走 small-shape 多生产者搬出逻辑。
    if (TryCreateSpillCopyoutForSmallShape(spillInfo, actualSpillOp, isFinish, isGenSpill, pcIdx) != SUCCESS) {
        return FAILED;
    }
    if (isFinish) {
        return SUCCESS;
    }
    actualSpillTensor = actualSpillOp->GetInputOperand(0);
    for (auto &preOp : depManager_.GetPredecessors(actualSpillOp)) {
        if (!opIsAllocMap[preOp]) {
            actualSpillOp = preOp;
        }
    }
    return SUCCESS;
}

void OoOScheduler::HandleUBCopyNZ2ND(Operation* &actualSpillOp, LogicalTensorPtr &actualSpillTensor)
{
    if (actualSpillOp->GetOpcodeStr().find("UB_COPY_ND2NZ") != std::string::npos) {
        actualSpillTensor = actualSpillOp->GetInputOperand(0);
        for (auto &preOp : depManager_.GetPredecessors(actualSpillOp)) {
            if (!opIsAllocMap[preOp]) {
                actualSpillOp = preOp;
            }
        }
    }
}

Status OoOScheduler::InsertCopyoutToOperations(Operation* actualSpillOp, Operation* spillCopyoutOp)
{
    if (depManager_.GetSuccessors(actualSpillOp).empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "actualSpillOp %s has no successor", GetOpInfo(actualSpillOp).c_str());
        return FAILED;
    }
    auto spillOpIt = std::find(newOperations_.begin(), newOperations_.end(),
        *(depManager_.GetSuccessors(actualSpillOp).begin()));
    if (spillOpIt != newOperations_.end()) {
        size_t pos = std::distance(newOperations_.begin(), spillOpIt);
        newOperations_.insert(newOperations_.begin() + pos + 1, spillCopyoutOp);
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert op: %s.", GetOpInfo(spillCopyoutOp).c_str());
    }
    return SUCCESS;
}

Status OoOScheduler::CreateSpecialL1Copyout(SpillInfo &spillInfo, Operation* &spillCopyoutOp, int &bufLastUseOrder,
    bool &isFinish, bool isGenSpill, size_t &pcIdx)
{
    auto spillOp = spillInfo.spillOp_;
    auto preTensor = spillOp->GetInputOperand(0);
    if (spillOp->GetOpcode() != Opcode::OP_RESHAPE && preTensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB && preTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation, "spillOp %s is not COPY_IN/UB_COPY_L1/UB_COPY_L1/RESHAPE in A5 L1 spill", GetOpInfo(spillOp).c_str());
        return FAILED;
    }
    auto actualSpillTensor = preTensor;
    Operation* actualSpillOp = nullptr;
    for (auto &preOp : depManager_.GetPredecessors(spillOp)) {
        if (!opIsAllocMap[preOp]) {
            actualSpillOp = preOp;
        }
    }
    if (spillOp->GetOpcode() == Opcode::OP_RESHAPE) {
        if (HandleReshapeSpillPath(spillInfo, actualSpillOp, actualSpillTensor, isFinish, isGenSpill, pcIdx) !=
            SUCCESS) {
            return FAILED;
        }
        if (isFinish) {
            return SUCCESS;
        }
    }
    if (actualSpillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        APASS_LOG_ERROR_F(Elements::Operation, "A5 does not support the COPY_IN-actualSpillOp. ");
        return FAILED;
    }
    // A5 L1 small-shape spill: spillInfo.spillOp_ 为 UB_COPY_L1 / L0C_COPY_L1 且其 input.shape 任一维 < output.shape，
    // 说明 L1 是被多个生产者拼接式写入，需为每个生产者分别插 COPY_OUT 落到同一块 DDR。
    if (TryCreateSpillCopyoutForSmallShape(spillInfo, spillOp, isFinish, isGenSpill, pcIdx) != SUCCESS) {
        return FAILED;
    }
    if (isFinish) {
        return SUCCESS;
    }

    HandleUBCopyNZ2ND(actualSpillOp, actualSpillTensor);
    if (CreateSpillCopyout(actualSpillOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, spillCopyoutOp, spillInfo) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillCopyout failed for specialL1 spill! %s", GetFormatBacktrace(*spillOp).c_str());
        return FAILED;
    }
    if (InsertCopyoutToOperations(actualSpillOp, spillCopyoutOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertCopyoutToOperations failed");
        return FAILED;
    }
    bufLastUseOrder = opExecOrderMap[spillOp];
    return SUCCESS;
}

Status OoOScheduler::TryCreateSpillCopyoutForSmallShape(SpillInfo &spillInfo, Operation* candidateOp, bool &isFinish,
    bool isGenSpill, size_t &pcIdx)
{
    if (candidateOp == nullptr) {
        return SUCCESS;
    }
    auto inOpd = candidateOp->GetInputOperand(0);
    auto outOpd = candidateOp->GetOutputOperand(0);
    if (inOpd == nullptr || outOpd == nullptr) {
        return SUCCESS;
    }
    auto inMem = inOpd->GetMemoryTypeOriginal();
    if (inMem != MemoryType::MEM_UB && inMem != MemoryType::MEM_L0C) {
        return SUCCESS;
    }
    if (!IsAnyDimSmaller(inOpd->GetShape(), outOpd->GetShape())) {
        return SUCCESS;
    }
    if (CreateSpillCopyoutForSmallShape(spillInfo, outOpd, isFinish, isGenSpill, pcIdx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "CreateSpillCopyoutForSmallShape failed! %s",
            GetFormatBacktrace(*spillInfo.spillOp_).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::CreateSpillCopyoutForSmallShape(SpillInfo &spillInfo, LogicalTensorPtr l1Tensor, bool &isFinish,
    bool isGenSpill, size_t &pcIdx)
{
    LogicalTensorPtr ddrTensor;
    if (BuildSmallShapeDdrTensor(spillInfo, l1Tensor, ddrTensor) != SUCCESS) {
        return FAILED;
    }
    for (auto* producerOp : l1Tensor->GetProducers()) {
        if (producerOp == nullptr || IsAllocOpCode(producerOp->GetOpcode())) {
            continue;
        }
        if (CreateSmallShapeCopyout(spillInfo, producerOp, ddrTensor, isGenSpill, pcIdx) !=
            SUCCESS) {
            return FAILED;
        }
    }
    isFinish = true;
    return SUCCESS;
}

Status OoOScheduler::BuildSmallShapeDdrTensor(SpillInfo &spillInfo, LogicalTensorPtr l1Tensor,
    LogicalTensorPtr &ddrTensor)
{
    // 建一整块 DDR tensor（属性参考 l1Tensor — 常规场景为 spillInfo.spillTensor_，
    // RESHAPE 场景为 RESHAPE 上一层 op 的 output，即真正承载数据的 L1 tensor）
    if (l1Tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "SmallShape spill: l1Tensor is null!");
        return FAILED;
    }
    auto ddrRawTensor = std::make_shared<RawTensor>(
        l1Tensor->Datatype(), l1Tensor->tensor->rawshape,
        TileOpFormat::TILEOP_ND, "WorkspaceGm");
    if (ddrRawTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR raw tensor failed in SmallShape spill!");
        return FAILED;
    }
    ddrTensor = std::make_shared<LogicalTensor>(
        function_, ddrRawTensor, l1Tensor->GetOffset(), l1Tensor->GetShape());
    if (ddrTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR tensor failed in SmallShape spill!");
        return FAILED;
    }
    ddrTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceOffset);
    if (UpdateTensorAttr(ddrTensor, MEM_DEVICE_DDR, l1Tensor, spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "UpdateTensorAttr DDR tensor failed in SmallShape spill!");
        return FAILED;
    }
    spillInfo.ddrTensor_ = ddrTensor;
    return SUCCESS;
}

Status OoOScheduler::ResolveSmallShapeActualSpill(Operation* producerOp,
    Operation* &actualOp, LogicalTensorPtr &actualTensor)
{
    auto producerIn = producerOp->GetInputOperand(0);
    if (producerIn == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "SmallShape spill: producer %s has null input.", GetOpInfo(producerOp).c_str());
        return FAILED;
    }
    auto producerInMem = producerIn->GetMemoryTypeOriginal();
    if (producerInMem != MemoryType::MEM_UB && producerInMem != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "SmallShape spill: producer %s input memType is %s, expect UB/L0C.",
            GetOpInfo(producerOp).c_str(), MemoryTypeToString(producerInMem).c_str());
        return FAILED;
    }
    actualOp = producerOp;
    actualTensor = producerIn;
    if (producerInMem == MemoryType::MEM_UB) {
        // 强制两层：producer -> UB_COPY_ND2NZ -> UB1
        Operation* prevOp = nullptr;
        for (auto &preOp : depManager_.GetPredecessors(producerOp)) {
            if (!opIsAllocMap[preOp]) {
                prevOp = preOp;
            }
        }
        if (prevOp == nullptr || prevOp->GetOpcode() != Opcode::OP_UB_COPY_ND2NZ) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "SmallShape spill: UB-producer %s does not have UB_COPY_ND2NZ predecessor.",
                GetOpInfo(producerOp).c_str());
            return FAILED;
        }
        actualOp = prevOp;
        actualTensor = prevOp->GetInputOperand(0);
    }
    if (actualTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "SmallShape spill: actualTensor is null for producer %s.", GetOpInfo(producerOp).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ConfigSmallShapeCopyoutAttrs(Operation &copyOutOp, Operation* producerOp,
    LogicalTensorPtr actualTensor)
{
    // 用 producerOp 取 tooffset 属性
    auto producerL1Out = producerOp->GetOutputOperand(0);
    if (producerL1Out == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "SmallShape spill: producer %s has null L1 output.", GetOpInfo(producerOp).c_str());
        return FAILED;
    }
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
    if (attr == nullptr) {
        APASS_LOG_INFO_F(Elements::Tensor, "Op %s attribute is nullptr", GetOpInfo(producerOp).c_str());
        return FAILED;
    }
    copyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        actualTensor->GetMemoryTypeOriginal(),
        attr->GetToOffset(),
        OpImmediate::Specified(actualTensor->GetShape()),
        OpImmediate::Specified(producerL1Out->tensor->GetDynRawShape())));
    copyOutOp.UpdateLatency(DEFAULT_LATENCY);
    if (actualTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        if (producerOp->GetAttr(OpAttributeKey::scaleValue, scaleValue)) {
            copyOutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
        }
    }
    return SUCCESS;
}

// 场景1 actualTensor(UB)->actualOp(UB_COPY_ND2NZ)->UB->producerOp UB_COPY_L1->L1
// 场景2 actualTensor(L0C)->actualOp L0C_COPY_L1->L1
Status OoOScheduler::CreateSmallShapeCopyout(SpillInfo &spillInfo, Operation* producerOp,
    LogicalTensorPtr ddrTensor, bool isGenSpill, size_t &pcIdx)
{
    Operation* actualOp = nullptr;
    LogicalTensorPtr actualTensor;
    if (ResolveSmallShapeActualSpill(producerOp, actualOp, actualTensor) != SUCCESS) {
        return FAILED;
    }
    Operation &copyOutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {actualTensor}, {ddrTensor});
    if (ConfigSmallShapeCopyoutAttrs(copyOutOp, producerOp, actualTensor) != SUCCESS) {
        return FAILED;
    }
    for (auto preOp : depManager_.GetPredecessors(actualOp)) {
        if (!opIsAllocMap[preOp]) {
            depManager_.AddDependency(preOp, &copyOutOp);
            for (auto allocOp : depManager_.GetPredecessors(preOp)) {
                if (opIsAllocMap[allocOp]) {
                    opCoreLocationMap[&copyOutOp] = opCoreLocationMap[allocOp];
                    UpdateOpInternalSubgraphID(copyOutOp, allocOp);
                }
            }
        }
    }
    SetOpMemIds(&copyOutOp, {spillInfo.spillMemId_});
    opIsRetiredMap[&copyOutOp] = true;
    opIsAllocMap[&copyOutOp] = false;
    opPipeTypeMap[&copyOutOp] = RescheduleUtils::GetOpPipeType(&copyOutOp);
    opViewOpsMap[&copyOutOp] = std::vector<Operation*>();
    if (UpdateCopyOutMode(copyOutOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyOutMode failed in SmallShape spill!");
        return FAILED;
    }
    int order = opExecOrderMap[actualOp];
    SetExecOrder(&copyOutOp, order + 1);
    InsertOrdered(&copyOutOp);
    if (isGenSpill) {
        pcIdx++;
        numTotalIssues++;
    } else {
        auto spillOpIt = std::find(newOperations_.begin(), newOperations_.end(), actualOp);
        if (spillOpIt != newOperations_.end()) {
            size_t pos = std::distance(newOperations_.begin(), spillOpIt);
            newOperations_.insert(newOperations_.begin() + pos + 1, &copyOutOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "SmallShape spill insert op: %s.", GetOpInfo(&copyOutOp).c_str());
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOutBuffer(SpillInfo &spillInfo, Operation* op, size_t &pcIdx, bool isGenSpill) {
    if (spillInfo.spillOp_->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        spillInfo.ddrTensor_ = spillInfo.spillOp_->GetInputOperand(0);
        return SUCCESS;
    }
    Operation* spillCopyoutOp = nullptr;
    int bufLastUseOrder = -1;
    if (spillInfo.isSpecialL1_) {
        // actualSpillOp 为 copy_in
        bool isFinish = false;
        if (CreateSpecialL1Copyout(spillInfo, spillCopyoutOp, bufLastUseOrder, isFinish, isGenSpill, pcIdx) !=
            SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpecialL1 CreateSpillCopyout failed!");
            return FAILED;
        }
        if (isFinish) {
            return SUCCESS;
        }
    } else {
        if (CreateSpillCopyout(spillInfo.spillOp_, spillInfo.spillTensor_, spillInfo.spillMemId_,
        spillCopyoutOp, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillCopyout failed! %s", GetFormatBacktrace(*spillInfo.spillOp_).c_str());
            return FAILED;
        }
        bufLastUseOrder = GetBufLastUseOrder(op, spillInfo.spillMemId_);
    }
    if (bufLastUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last used order.", spillInfo.spillMemId_);
        return FAILED;
    }
    if (UpdateCopyOutMode(*spillCopyoutOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyOutMode failed!");
        return FAILED;
    }
    SetExecOrder(spillCopyoutOp, bufLastUseOrder + 1);
    InsertOrdered(spillCopyoutOp);
    if (isGenSpill) {
        pcIdx++;
        numTotalIssues++;
    } else if (std::find(newOperations_.begin(), newOperations_.end(), spillCopyoutOp) == newOperations_.end()) {
        newOperations_.push_back(spillCopyoutOp);
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert op: %s.", GetOpInfo(spillCopyoutOp).c_str());
    }
    spillInfo.ddrTensor_ = spillCopyoutOp->GetOutputOperand(0);
    return SUCCESS;
}

Status OoOScheduler::GetSpillTensor(Operation* spillOp, int spillMemId, LogicalTensorPtr &spillTensor) {
    int spillTensorIdx = GetOOperandIdx(spillOp, spillMemId);
    if (spillTensorIdx == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in op's oOperand.", spillMemId);
        return FAILED;
    }
    spillTensor = spillOp->GetOutputOperand(spillTensorIdx);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op cannot find oOperand[%d]. %s", spillTensorIdx, GetFormatBacktrace(*spillOp).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SpillBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    if (SpillOutBuffer(spillInfo, allocOp, pcIdx, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOutBuffer failed. %s", GetFormatBacktrace(*spillInfo.spillOp_).c_str());
        return FAILED;
    }
    NotifySpill(spillInfo, allocBuffer);
    if (SpillInBuffer(spillInfo, allocOp, allocBuffer->memType, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillInBuffer failed. %s", GetFormatBacktrace(*spillInfo.spillOp_).c_str());
        return FAILED;
    }
    if (!isGenSpill) {
        NotifyBufferFreed(allocBuffer->memType, spillInfo.spillMemId_);
        localBufferMap_[spillInfo.spillMemId_]->retireCycle = clock;
        if (tensorOccupyMap[allocBuffer->memType].erase(spillInfo.spillMemId_) == 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", spillInfo.spillMemId_);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FindAssembleWithSpillTensor(SpillInfo &spillInfo, std::vector<Operation*> &assembleOps) {
    for (auto producer : spillInfo.spillTensor_->GetProducers()) {
        if (!IsSupportedPartialWriteProducer(*producer)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "All producer of Tensor[%d] must be assemble/l0c_to_l1/alloc, now has %s[%d].",
                spillInfo.spillTensor_->GetMagic(), producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
            return FAILED;
        }
        if (IsAllocOpCode(producer->GetOpcode())) {
            continue;
        }
        for (auto op : orderedOps) {
            if (op == producer) {
                assembleOps.push_back(op);
                break;
            }
        }
    }
    return SUCCESS;
}

LogicalTensorPtr OoOScheduler::CreateAssemblePartTensor(
    LogicalTensorPtr iOperand, LogicalTensorPtr assembleTensor, const std::vector<int64_t> &toOffset)
{
    LogicalTensorPtr localTensor =
        std::make_shared<LogicalTensor>(function_, iOperand->Datatype(), iOperand->shape, iOperand->Format());
    localTensor->SetMemoryTypeToBe(assembleTensor->GetMemoryTypeToBe());
    localTensor->SetMemoryTypeOriginal(assembleTensor->GetMemoryTypeOriginal());
    localTensor->oriShape = iOperand->shape;
    localTensor->tensor = assembleTensor->tensor;
    localTensor->memoryrange.memId = assembleTensor->memoryrange.memId;
    localTensor->UpdateDynValidShape(iOperand->GetDynValidShape());
    localTensor->offset = toOffset;
    tensorAllocCoreMap[localTensor->memoryrange.memId] = tensorAllocCoreMap[assembleTensor->memoryrange.memId];
    return localTensor;
}

Operation* OoOScheduler::UpdateIssueAttr(Operation &newOp, std::vector<int> memIds, Operation* allocOp, int &bufNextUseOrder, bool isGenSpill) {
    UpdateOpInternalSubgraphID(newOp, allocOp);
    auto& coreLocation = opCoreLocationMap[allocOp];
    Operation* newOpPtr = &newOp;

    // 初始化Operation属性到map
    int order = bufNextUseOrder++;
    opExecOrderMap[newOpPtr] = order;
    opPipeTypeMap[newOpPtr] = RescheduleUtils::GetOpPipeType(newOpPtr);
    opIsAllocMap[newOpPtr] = (newOp.GetOpcodeStr().find("ALLOC") != std::string::npos);
    opIsRetiredMap[newOpPtr] = false;
    SetOpMemIds(newOpPtr, memIds);

    depManager_.RegisterOp(newOpPtr);
    opViewOpsMap[newOpPtr] = std::vector<Operation*>();
    opCoreLocationMap[newOpPtr] = coreLocation;
    InitOpViewOps(newOpPtr);

    // 插入orderedOps
    InsertOrdered(newOpPtr);

    if (opIsAllocMap[newOpPtr] && !isGenSpill) {
        opExecOrderMap[allocIssueQueue[coreLocation][newOp.GetOutputOperand(0)->GetMemoryTypeOriginal()].Front()] > opExecOrderMap[newOpPtr] ?
        allocIssueQueue[coreLocation][newOp.GetOutputOperand(0)->GetMemoryTypeOriginal()].ForceInsertAfterFirst(newOpPtr) :
        allocIssueQueue[coreLocation][newOp.GetOutputOperand(0)->GetMemoryTypeOriginal()].Insert(newOpPtr);
    }
    return newOpPtr;
}

Status OoOScheduler::SpillParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, Operation* producerOp,
    LogicalTensorPtr assembleTensor, bool &isFirst, bool isGenSpill)
{
    auto iOperand = producerOp->GetInputOperand(0);
    std::vector<int64_t> toOffset;
    std::vector<SymbolicScalar> toDynOffset, fromDynValidShape;
    if (GetPartialWriteReplayAttr(producerOp, toOffset, toDynOffset, fromDynValidShape) != SUCCESS) {
        return FAILED;
    }
    LogicalTensorPtr localTensor = CreateAssemblePartTensor(iOperand, assembleTensor, toOffset);
    int bufNextUseOrder = GetBufNextUseOrder(allocOp, spillInfo.spillMemId_);
    if (bufNextUseOrder == -1) return FAILED;
    if (isFirst) {
        Opcode allocOpCode = assembleTensor->GetMemoryTypeToBe() == MemoryType::MEM_UB ?
            Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
        auto &spillAllocOp = function_.AddRawOperation(allocOpCode, {}, {localTensor});
        spillAllocOp.UpdateLatency(1);
        UpdateIssueAttr(spillAllocOp, {assembleTensor->memoryrange.memId}, allocOp, bufNextUseOrder, isGenSpill);
        isFirst = false;
        numTotalIssues++;
    }
    // copyin
    auto& spillCopyInOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {localTensor});
    int64_t isNZ = 0;
    producerOp->GetAttr(OpAttributeKey::copyIsNZ, isNZ);
    spillCopyInOp.SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    spillCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(toOffset), localTensor->GetMemoryTypeOriginal(), OpImmediate::Specified(spillInfo.ddrTensor_->GetShape()),
        OpImmediate::Specified(assembleTensor->tensor->GetDynRawShape())));
    spillCopyInOp.UpdateLatency(DEFAULT_LATENCY);
    if (UpdateCopyInMode(spillCopyInOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyInMode failed.");
        return FAILED;
    }
    UpdateIssueAttr(spillCopyInOp, {assembleTensor->memoryrange.memId}, allocOp, bufNextUseOrder, isGenSpill);
    UpdateOpIsCube(spillCopyInOp, spillInfo.spillOp_);
    // assemble
    auto &assembleOp = function_.AddRawOperation(Opcode::OP_ASSEMBLE, {localTensor}, {assembleTensor});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
        localTensor->GetMemoryTypeOriginal(), toOffset, toDynOffset, fromDynValidShape));
    assembleOp.SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    assembleOp.UpdateLatency(1);
    UpdateIssueAttr(assembleOp, {assembleTensor->memoryrange.memId, assembleTensor->memoryrange.memId},
        allocOp, bufNextUseOrder, isGenSpill);
    UpdateOpIsCube(assembleOp, spillInfo.spillOp_);
    numTotalIssues += TWO_ISSUE;
    return SUCCESS;
}

Status OoOScheduler::UpdateAssembleBuffer(SpillInfo &spillInfo, LocalBufferPtr allocBuffer,
    LogicalTensorPtr assembleTensor) {
    auto coreLocation = tensorAllocCoreMap[allocBuffer->id];
    if (bufferManagerMap[coreLocation][allocBuffer->memType].Free(spillInfo.spillMemId_) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    if (UpdateRemainOpBufId(spillInfo.spillMemId_, assembleTensor->memoryrange.memId)) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainOpBufId failed.");
        return FAILED;
    }
    bufRefCount_[assembleTensor->memoryrange.memId] = 0;
    for (auto op: orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        auto& reqMemIds = GetOpMemIds(op);
        for (auto memId : reqMemIds) {
            if (memId == assembleTensor->memoryrange.memId) {
                bufRefCount_[assembleTensor->memoryrange.memId]++;
            }
        }
    }
    depManager_.InitDependencies(orderedOps, false);
    return SUCCESS;
}

Status OoOScheduler::RejectIfNZHorizontalSlice(SpillInfo &spillInfo, std::vector<Operation*> &assembleOps)
{
    if (FindAssembleWithSpillTensor(spillInfo, assembleOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FindAssembleWithSpillTensor failed.");
        return FAILED;
    }
    if (HasNZHorizontalSlice(assembleOps)) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Spill-assemble rejected for Tensor[%d]: NZ horizontal slice not supported by DMA.",
            spillInfo.spillTensor_->GetMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ReplayPartialWriteProducers(SpillInfo &spillInfo, Operation* allocOp,
    LogicalTensorPtr assembleTensor, const std::vector<Operation*> &assembleOps, bool isGenSpill)
{
    Operation *memIdAlloc = FindAllocForAssembleProducers(assembleOps);
    bool isAllocHandoffDone = false;
    bool isFirst = true;
    for (auto producerOp : assembleOps) {
        if (opIsRetiredMap[producerOp]) {
            if (!isAllocHandoffDone && memIdAlloc != nullptr &&
                producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                memIdAlloc->UpdateOutputOperand(0, producerOp->GetInputOperand(0));
                isAllocHandoffDone = true;
            }
            if (SpillParticalBuffer(spillInfo, allocOp, producerOp, assembleTensor, isFirst, isGenSpill) != SUCCESS) {
                return FAILED;
            }
        } else {
            producerOp->ReplaceOutput(assembleTensor, spillInfo.spillTensor_);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillAssembleBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    std::vector<Operation*> assembleOps;
    if (RejectIfNZHorizontalSlice(spillInfo, assembleOps) != SUCCESS) {
        return FAILED;
    }
    if (SpillOutBuffer(spillInfo, allocOp, pcIdx, isGenSpill) != SUCCESS) {
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
    for (auto &succOp : depManager_.GetSuccessors(spillInfo.spillOp_)) {
        if (!opIsRetiredMap[succOp] &&
            (std::count(GetOpMemIds(succOp).begin(), GetOpMemIds(succOp).end(), spillInfo.spillMemId_) > 0)) {
            UpdateTensorInputFor(succOp, spillInfo.spillOp_, assembleTensor);
        }
    }
    if (ReplayPartialWriteProducers(spillInfo, allocOp, assembleTensor, assembleOps, isGenSpill) != SUCCESS) {
        return FAILED;
    }
    if (UpdateAssembleBuffer(spillInfo, allocBuffer, assembleTensor) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::GetSpillInfo(Operation* allocOp, int spillMemId, bool isGenSpill,
    SpillInfo &spillInfo) {
    auto spillOp = GetSpillIssue(allocOp, spillMemId, isGenSpill);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", spillMemId);
        return FAILED;
    }
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillOp, spillMemId, spillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s GetSpillTensor failed! %s", GetOpInfo(spillOp).c_str(), GetFormatBacktrace(*spillOp).c_str());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Begin spill op %s tensor[%d]!", GetOpInfo(spillOp).c_str(), spillMemId);
    LogicalTensorPtr ddrTensor = nullptr;
    spillInfo.ddrTensor_ = ddrTensor;
    spillInfo.spillTensor_ = spillTensor;
    spillInfo.spillOp_ = spillOp;
    spillInfo.spillMemId_ = spillMemId;
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        spillInfo.isSpecialL1_ = true;
    }
    return SUCCESS;
}

bool OoOScheduler::IsPartialWrite(const Operation &op) const
{
    const auto opcode = op.GetOpcode();
    if (opcode != Opcode::OP_ASSEMBLE && opcode != Opcode::OP_L0C_TO_L1) {
        return false;
    }
    if (opcode == Opcode::OP_L0C_TO_L1 &&
        Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        return false;
    }
    const auto &inShape = op.GetInputOperand(0)->shape;
    const auto &outShape = op.GetOutputOperand(0)->shape;
    const size_t dims = std::min(inShape.size(), outShape.size());
    for (size_t i = 0; i < dims; ++i) {
        if (outShape[i] > inShape[i]) {
            return true;
        }
    }
    return false;
}

Status OoOScheduler::SpillMultiBuffer(Operation* allocOp, std::vector<int> spillGroup, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    for (auto &spillMemId : spillGroup) {
        SpillInfo spillInfo;
        if (GetSpillInfo(allocOp, spillMemId, isGenSpill, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed. %s", GetFormatBacktrace(*spillInfo.spillOp_).c_str());
            return FAILED;
        }
        if (spillInfo.spillOp_->GetOpcode() == Opcode::OP_ASSEMBLE &&
            Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
            allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos) {
                APASS_LOG_ERROR_F(Elements::Operation, "Failed to spill %d in L1 spill. SpillIssue is assemble op.", spillMemId);
                return FAILED;
        }
        if (allocBuffer->memType == MemoryType::MEM_L0C) {
            if (SpillL0CBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillL0CBuffer[%d] failed. %s", spillMemId,
                    GetFormatBacktrace(*spillInfo.spillOp_).c_str());
                return FAILED;
            }
        } else if (IsPartialWrite(*spillInfo.spillOp_)) {
            if (SpillAssembleBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillAssembleBuffer[%d] failed.", spillMemId);
                return FAILED;
            }
        } else {
            if (SpillBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed. %s", spillMemId, GetFormatBacktrace(*spillInfo.spillOp_).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags) {
    auto dstOpList = depManager_.GetSuccessors(allocOp);
    auto dstOp = *dstOpList.begin();
    if (COPY_IN_OPS.find(dstOp->GetOpcode()) != COPY_IN_OPS.end()) {
        for (auto &dstOpId : depManager_.GetSuccessors(dstOp)) {
            auto dstOp_level0 = dstOpId;
            for (auto &inOp : depManager_.GetPredecessors(dstOp_level0)) {
                filterLtags.insert(inOp);
            }
        }
    }
    for (auto &dstOp_level1 : dstOpList) {
        for (auto &inOp : depManager_.GetPredecessors(dstOp_level1)) {
            filterLtags.insert(inOp);
        }
    }
}

bool OoOScheduler::CheckMachineAndL1(Operation* spillOp, Operation* allocOp) {
    if (!spillOp->GetInputOperand(0)) {
        APASS_LOG_WARN_F(Elements::Tensor, "CheckMachineAndL1: spillOp %s has no inputOperand.", GetOpInfo(spillOp).c_str());
        return false;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos && spillOp->GetOpcodeStr().find("RESHAPE") == std::string::npos &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        return false;
    }
    return true;
}

bool OoOScheduler::IsBelongSpillBlackList(Operation* spillOp, Operation* op) {
    std::set<Operation*> filterLtags;
    FindFilterLtags(op, filterLtags);
    if (opIsAllocMap[spillOp] || filterLtags.count(spillOp) != 0 || !CheckMachineAndL1(spillOp, op)) {
        return true;
    }
    return false;
}

Operation* OoOScheduler::GetSpillIssue(Operation* allocOp, int memId, bool isGenSpill) {
    if (isGenSpill) {
        return GetBufLastWriteOp(allocOp, memId);
    }
    return tensorOccupyMap[localBufferMap_[GetOpMemIds(allocOp)[0]]->memType][memId];
}

Status OoOScheduler::GetGroupNextUseOrder(std::vector<int> group, Operation* allocOp,
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill) {
    std::vector<size_t> bufNextUseTime;
    for (auto& memId : group) {
        Operation* spillOp = GetSpillIssue(allocOp, memId, isGenSpill);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }
        if (IsBelongSpillBlackList(spillOp, allocOp)) {
            // 存在非法memId时将该group排除
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            bufNextUseTime.push_back(nextUseTimeCache[memId]);
        } else {
            int nextUseOrder = GetBufNextUseOrder(allocOp, memId);
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

bool OoOScheduler::CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType)
{
    if (tensors.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "CanAllocateAll input tensors is empty.");
        return true;
    }
    auto coreLocation = tensorAllocCoreMap[tensors[0]->id];
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals =
        bufferManagerMap[coreLocation][memType].FindFreeIntervals();
    for (auto tensor : tensors) {
        bool canAlloc = false;
        std::pair<uint64_t, uint64_t> newInterval;
        uint64_t allocInterval;
        uint64_t allocAddrStart;
        for (auto& interval : freeIntervals) {
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
    for (auto op : orderedOps) {
        if (!opIsAllocMap[op]) {
            continue;
        }
        auto& reqMemIds = GetOpMemIds(op);
        if (!reqMemIds.empty() && reqMemIds[0] == memId) {
            return opExecOrderMap[op];
        }
    }
    return -1;
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType) {
    std::vector<LocalBufferPtr> tensors;
    std::vector<int> memIds;
    if (allocOp->GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s must only have one ooperand.", GetFormatBacktrace(*allocOp).c_str());
        return false;
    }
    for (auto &succOp : depManager_.GetSuccessors(allocOp)) {
        if (succOp != *(allocOp->GetOutputOperand(0)->GetProducers().begin())) {
            continue;
        }
        for (auto &memId : GetOpMemIds(succOp)) {
            if (localBufferMap_[memId]->memType != memType) {
                continue;
            }
            auto coreLocation = tensorAllocCoreMap[memId];
            if (bufferManagerMap[coreLocation][memType].isAllocate(memId)) {
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
        tensors.push_back(localBufferMap_[memId]);
    }
    return CanAllocateAll(tensors, memType);
}

Status OoOScheduler::SelectSpillBuffers(LocalBufferPtr allocBuffer, Operation* allocOp,
    std::vector<int> &spillGroup, bool isGenSpill) {
    // 查找出可以spill 单个或多个tensor的集合
    std::vector<std::vector<int>> canSpillGroups;
    auto coreLocation = opCoreLocationMap[allocOp];
    auto memType = allocBuffer->memType;
    if (bufferManagerMap[coreLocation][memType].GetSpillGroup(allocBuffer->size, canSpillGroups) != SUCCESS) {
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
        if (GetGroupNextUseOrder(group, allocOp, groupNextUseTime, nextUseTimeCache, isGenSpill) != SUCCESS) {
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

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType, CoreLocationType coreLocation,
    bool isGenSpill)
{
    std::vector<int> memIds = bufferManagerMap[coreLocation][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = GetSpillIssue(allocOp, memId, isGenSpill);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return FAILED;
        }
    }
    return bufferManagerMap[coreLocation][memType].CompactBufferSlices(localBufferMap_);
}

Status OoOScheduler::SpillAllBuffer(Operation* allocOp, size_t &pcIdx, bool isGenSpill, LocalBufferPtr allocBuffer) {
    MemoryType memType = allocBuffer->memType;
    auto coreLocation = opCoreLocationMap[allocOp];
    std::vector<int> memIds = bufferManagerMap[coreLocation][memType].GetAddrSortedBufs();

    for (auto memId : memIds) {
        Operation* spillOp = isGenSpill ? GetBufLastWriteOp(allocOp, memId) : tensorOccupyMap[memType][memId];
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }

        if (spillOp->GetOpcodeStr().find("ALLOC") != std::string::npos || !CheckMachineAndL1(spillOp, allocOp) ||
            IsViewOp(*spillOp) || spillOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            continue;
        }

        SpillInfo spillInfo;
        if (GetSpillInfo(allocOp, memId, isGenSpill, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed. %s",
                GetFormatBacktrace(*spillInfo.spillOp_).c_str());
            return FAILED;
        }

        if (allocBuffer->memType == MemoryType::MEM_L0C) {
            if (SpillL0CBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillL0CBuffer[%d] failed. %s", memId,
                    GetFormatBacktrace(*spillInfo.spillOp_).c_str());
                return FAILED;
            }
        } else if (SpillBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed. %s", memId, GetFormatBacktrace(*spillInfo.spillOp_).c_str());
            return FAILED;
        }
    }

    // Alloc内存整理
    if (RearrangeBuffer(allocOp, memType, coreLocation, isGenSpill) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed at SpillAllBuffer. %s", GetFormatBacktrace(*allocOp).c_str());
    }

    if (!HasEnoughBuffer(allocOp, memType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp, isGenSpill) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed; Please check the PrintSpillFailedInfo method.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        return FAILED;
    }

    return SUCCESS;
}

Status OoOScheduler::GenBufferSpill(Operation* allocOp) {
    std::vector<int> spillGroup;
    bool spillFailed = false;
    if (SelectSpillBuffers(localBufferMap_[GetOpMemIds(allocOp)[0]], allocOp, spillGroup, false) != SUCCESS) {
        spillFailed = true;
    }
    size_t temp = 1;
    if (spillFailed) {
        return SpillAllBuffer(allocOp, temp, false, localBufferMap_[GetOpMemIds(allocOp)[0]]);
    } else {
        if (SpillMultiBuffer(allocOp, spillGroup, temp, localBufferMap_[GetOpMemIds(allocOp)[0]], false) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiBuffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillOp(size_t& pcIdx)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "START: SPILL tensor!");
    LocalBufferPtr allocBuffer = localBufferMap_[GetOpMemIds(orderedOps[pcIdx])[0]];
    if (allocBuffer->memType != MemoryType::MEM_L1 && allocBuffer->memType != MemoryType::MEM_UB &&
        allocBuffer->memType != MemoryType::MEM_L0C) {
        if (PrintSpillFailedInfo(orderedOps[pcIdx], true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed; Please check the PrintSpillFailedInfo method.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "No spillable buffer found. Only UB/L1/L0C support OoO spill. "
            "Please check tile shape and OOO spill failed info.");
        return FAILED;
    }
    // 选择最晚被使用的spill 单个或多个tensor
    std::vector<int> spillGroup;
    SelectSpillBuffers(allocBuffer, orderedOps[pcIdx], spillGroup, true);
    if (spillGroup.empty()) {
        return SpillAllBuffer(orderedOps[pcIdx], pcIdx, true, allocBuffer);
    } else {
        if (SpillMultiBuffer(orderedOps[pcIdx], spillGroup, pcIdx, allocBuffer, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiBuffer failed!");
            return FAILED;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "END: SPILL tensor!");
    return SUCCESS;
}

void OoOScheduler::CollectL0CConsumers(LogicalTensorPtr spillTensor, std::vector<Operation*> &consumers)
{
    // L0C CopyOut/DDR consumer 默认已 retired，spill 只重连未 retired 的 UB/L1 consumer。
    for (auto* c : spillTensor->GetConsumers()) {
        if (c == nullptr || opIsRetiredMap[c]) {
            continue;
        }
        auto output = c->GetOutputOperand(0);
        if (output == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "L0C spill: skip consumer %s without output operand.",
                GetOpInfo(c).c_str());
            continue;
        }
        auto outMem = output->GetMemoryTypeOriginal();
        if (outMem == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (outMem != MemoryType::MEM_UB && outMem != MemoryType::MEM_L1) {
            APASS_LOG_WARN_F(Elements::Operation,
                "L0C spill: skip consumer %s with output memType %s.",
                GetOpInfo(c).c_str(), MemoryTypeToString(outMem).c_str());
            continue;
        }
        consumers.push_back(c);
    }
    std::sort(consumers.begin(), consumers.end(), [this](Operation* a, Operation* b) {
        return opExecOrderMap[a] < opExecOrderMap[b];
    });
}

Status OoOScheduler::CreateL0CSpillOut(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx, bool isGenSpill)
{
    Operation* spillCopyoutOp = nullptr;
    if (CreateSpillCopyout(spillInfo.spillOp_, spillInfo.spillTensor_, spillInfo.spillMemId_,
        spillCopyoutOp, spillInfo) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C CreateSpillCopyout failed! %s",
            GetFormatBacktrace(*spillInfo.spillOp_).c_str());
        return FAILED;
    }
    int bufLastUseOrder = GetBufLastUseOrder(allocOp, spillInfo.spillMemId_);
    if (bufLastUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor,
            "L0C spill: cannot find Tensor[%d] last used order.", spillInfo.spillMemId_);
        return FAILED;
    }
    if (UpdateCopyOutMode(*spillCopyoutOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: UpdateCopyOutMode failed!");
        return FAILED;
    }
    SetExecOrder(spillCopyoutOp, bufLastUseOrder + 1);
    InsertOrdered(spillCopyoutOp);
    if (isGenSpill) {
        pcIdx++;
        numTotalIssues++;
    } else if (std::find(newOperations_.begin(), newOperations_.end(), spillCopyoutOp) == newOperations_.end()) {
        newOperations_.push_back(spillCopyoutOp);
    }
    spillInfo.ddrTensor_ = spillCopyoutOp->GetOutputOperand(0);
    return SUCCESS;
}

std::vector<int> OoOScheduler::FilterOutMemId(const std::vector<int> &memIds, int dropId)
{
    std::vector<int> out;
    out.reserve(memIds.size());
    for (auto m : memIds) {
        if (m != dropId) {
            out.push_back(m);
        }
    }
    return out;
}

void OoOScheduler::MoveDependencies(Operation* from, Operation* to)
{
    auto preds = depManager_.GetPredecessors(from); // 拷贝，避免迭代时被改
    auto succs = depManager_.GetSuccessors(from);
    for (auto* pred : preds) {
        depManager_.RemoveSuccessor(pred, from);
        depManager_.RemovePredecessor(from, pred);
        depManager_.InsertSuccessor(pred, to);
        depManager_.InsertPredecessor(to, pred);
    }
    for (auto* succ : succs) {
        depManager_.RemovePredecessor(succ, from);
        depManager_.RemoveSuccessor(from, succ);
        depManager_.InsertPredecessor(succ, to);
        depManager_.InsertSuccessor(to, succ);
    }
}

void OoOScheduler::EraseSchedulerSideMaps(Operation* op)
{
    opExecOrderMap.erase(op);
    opPipeTypeMap.erase(op);
    opIsAllocMap.erase(op);
    opIsRetiredMap.erase(op);
    opCoreLocationMap.erase(op);
    opViewOpsMap.erase(op);
    opReqMemIdsMap.erase(op);
    inOutOperandsCache_.erase(op);
}

Status OoOScheduler::ReplaceL0CConsumer(Operation* oldCons, const SpillInfo &spillInfo)
{
    auto T_dst = oldCons->GetOutputOperand(0);
    if (T_dst == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: consumer %s has no output operand.",
            GetOpInfo(oldCons).c_str());
        return FAILED;
    }
    auto& newOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {T_dst});
    // spill DDR 是完整 NZ2ND 输出，L1 reload 需要 ND2NZ，UB reload 按普通 GM->UB 处理。
    newOp.SetAttr(OpAttributeKey::copyIsNZ, static_cast<int64_t>(0));
    newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(spillInfo.ddrTensor_->GetOffset()),
        T_dst->GetMemoryTypeOriginal(),
        OpImmediate::Specified(T_dst->GetShape()),
        OpImmediate::Specified(T_dst->tensor->GetDynRawShape())));
    newOp.UpdateLatency(DEFAULT_LATENCY);
    if (T_dst->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        // L0C spill reload 到 L1 属于特殊 DDR->L1 路径，需要显式恢复 NZ 布局。
        newOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
    } else if (UpdateCopyInMode(newOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: UpdateCopyInMode failed.");
        return FAILED;
    }
    Operation* newPtr = &newOp;
    opExecOrderMap[newPtr] = opExecOrderMap[oldCons];
    opPipeTypeMap[newPtr] = RescheduleUtils::GetOpPipeType(newPtr);
    opIsAllocMap[newPtr] = false;
    opIsRetiredMap[newPtr] = false;
    opCoreLocationMap[newPtr] = opCoreLocationMap[oldCons];
    UpdateOpInternalSubgraphID(newOp, oldCons);
    SetOpMemIds(newPtr, FilterOutMemId(GetOpMemIds(oldCons), spillInfo.spillMemId_));
    depManager_.RegisterOp(newPtr);
    InitOpViewOps(newPtr);
    MoveDependencies(oldCons, newPtr);
    if (spillInfo.ddrTensor_->GetProducers().empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: DDR tensor has no producer.");
        return FAILED;
    }
    depManager_.AddDependency(*spillInfo.ddrTensor_->GetProducers().begin(), newPtr);
    std::replace(orderedOps.begin(), orderedOps.end(), oldCons, newPtr);
    APASS_LOG_DEBUG_F(Elements::Operation, "L0C spill: replace %s with %s.",
        GetOpInfo(oldCons).c_str(), GetOpInfo(newPtr).c_str());
    oldCons->SetAsDeleted();
    EraseSchedulerSideMaps(oldCons);
    return SUCCESS;
}

Status OoOScheduler::FreeL0CAfterSpill(SpillInfo &spillInfo, LocalBufferPtr allocBuffer)
{
    if (localBufferMap_.find(spillInfo.spillMemId_) != localBufferMap_.end()) {
        localBufferMap_[spillInfo.spillMemId_]->retireCycle = clock;
    }
    NotifyBufferFreed(allocBuffer->memType, spillInfo.spillMemId_);
    if (tensorOccupyMap[allocBuffer->memType].erase(spillInfo.spillMemId_) == 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "L0C spill: erase tensor[%d] from occupy map failed.",
            spillInfo.spillMemId_);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::RecomputeBufRefCount(int memId)
{
    bufRefCount_[memId] = 0;
    for (auto* op : orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        auto& reqMemIds = GetOpMemIds(op);
        bufRefCount_[memId] += std::count(reqMemIds.begin(), reqMemIds.end(), memId);
    }
}

Status OoOScheduler::SpillL0CBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "START: SPILL L0C tensor[%d]!", spillInfo.spillMemId_);
    std::vector<Operation*> consumers;
    CollectL0CConsumers(spillInfo.spillTensor_, consumers);
    if (CreateL0CSpillOut(spillInfo, allocOp, pcIdx, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: CreateL0CSpillOut failed.");
        return FAILED;
    }
    NotifySpill(spillInfo, allocBuffer);
    for (auto* cons : consumers) {
        if (ReplaceL0CConsumer(cons, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: ReplaceL0CConsumer failed for %s.",
                GetOpInfo(cons).c_str());
            return FAILED;
        }
    }
    RecomputeBufRefCount(spillInfo.spillMemId_);
    function_.EraseOperations(false, false);
    auto coreLocation = opCoreLocationMap[allocOp];
    if (bufferManagerMap[coreLocation][allocBuffer->memType].Free(spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "L0C spill: bufferManager free tensor[%d] failed.",
            spillInfo.spillMemId_);
        return FAILED;
    }
    if (!isGenSpill && FreeL0CAfterSpill(spillInfo, allocBuffer) != SUCCESS) {
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "END: SPILL L0C tensor[%d]!", spillInfo.spillMemId_);
    return SUCCESS;
}

} // namespace npu::tile_fwk
