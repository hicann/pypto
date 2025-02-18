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
 * \file common_operation_eliminate.cpp
 * \brief
 */

#include "common_operation_eliminate.h"
#include <unordered_map>
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_check/common_operation_eliminate_checker.h"

namespace npu::tile_fwk {
Status CommonOperationEliminate::RunOnFunction(Function &function) {
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (OpAlreadyExist(op)) {
            op->SetAsDeleted();
        }
    }
    function.EraseOperations();
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    return SUCCESS;
}

Status CommonOperationEliminate::PreCheck(Function &function) {
    CommonOperationEliminateChecker checker;
    return checker.DoPreCheck(function);
}

Status CommonOperationEliminate::PostCheck(Function &function) {
    CommonOperationEliminateChecker checker;
    return checker.DoPostCheck(function);
}

Operation *CommonOperationEliminate::OperationExist(Operation *operation) {
    auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
    auto &outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
    OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
    bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
    bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
    bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
    if (inputCheck && calcTypeCheck && outputCheck) { // copy from L1 to L0
        return nullptr;
    }
    if (operation->GetOpcode() == Opcode::OP_VIEW) {
        return nullptr;
    }
    if (operation->GetBoolAttribute(OpAttributeKey::dontTouch)) {
        return nullptr;
    }
    if (operationCache_.count(operation->ComputeHash()) != 0) {
        return operationCache_[operation->ComputeHash()];
    }
    operationCache_.insert({operation->ComputeHash(), operation});
    return nullptr;
}

void CommonOperationEliminate::UpdateView(ViewOpAttribute *viewOpAttribute,
                                          const std::shared_ptr<LogicalTensor> oldtensor,
                                          const std::shared_ptr<LogicalTensor> newtensor) const {
    auto &fromOffset = viewOpAttribute->GetFromOffset();
    for (size_t j = 0; j < fromOffset.size(); j++) {
        fromOffset[j] -= oldtensor->offset[j] - newtensor->offset[j];
    }
}

void CommonOperationEliminate::UpdateCopy(CopyOpAttribute *copyOpAttribute,
                                          const std::shared_ptr<LogicalTensor> oldtensor,
                                          const std::shared_ptr<LogicalTensor> newtensor) const {
    if (!copyOpAttribute->IsCopyOut()) {
        auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
        (void)memType;
        for (size_t j = 0; j < fromOffset.size(); j++) {
            fromOffset[j] -= oldtensor->offset[j] - newtensor->offset[j];
        }
        copyOpAttribute->SetFromOffset(fromOffset);
    }
}

bool CommonOperationEliminate::OpAlreadyExist(Operation *op) {
    auto existOp = OperationExist(op);
    if (existOp == nullptr || op->GetOOperands().size() == 0 || existOp->GetOOperands().size() == 0) {
        return false;
    }
    if (op->GetOOperands().front()->shape != existOp->GetOOperands().front()->shape) {
        return false;
    }
    LogicalTensors oldtensors(op->GetOOperands().begin(), op->GetOOperands().end());
    LogicalTensors newtensors(existOp->GetOOperands().begin(), existOp->GetOOperands().end());
    if (oldtensors.size() != newtensors.size()) {
        return false;
    }
    for (auto oldtensor : oldtensors) {
        if (oldtensor->nodetype == NodeType::OUTCAST) {
            return false;
        }
    }
    for (size_t i = 0; i < oldtensors.size(); i++) {
        auto oldtensor = oldtensors[i];
        auto newtensor = newtensors[i];
        if (oldtensor->GetConsumers().size() == 0) {
            continue;
        }
        if (newtensor->GetMagic() == oldtensor->GetMagic()) {
            ALOG_DEBUG_F("In CommonOperationEliminate, Operation %d is marked as redundant.", op->GetOpMagic());
            continue;
        }
        auto consumers = oldtensor->GetConsumers();
        for (auto &cur : consumers) {
            cur->ReplaceInput(newtensor, oldtensor);
            if (cur->GetOpAttribute() == nullptr) {
                continue;
            }
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(cur->GetOpAttribute().get())) {
                // VIEW操作的offset要相应被修改。
                UpdateView(viewOpAttribute, oldtensor, newtensor);
                continue;
            }
            if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(cur->GetOpAttribute().get())) {
                // CopyIn操作的offset要相应被修改。
                UpdateCopy(copyOpAttribute, oldtensor, newtensor);
                continue;
            }
        }
        oldtensor->GetConsumers().clear();
    }
    ALOG_DEBUG_F("In CommonOperationEliminate, Operation %d is marked as redundant.", op->GetOpMagic());
    return true;
}
}  // namespace npu::tile_fwk