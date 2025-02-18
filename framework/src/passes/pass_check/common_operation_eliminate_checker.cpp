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
 * \file common_operation_eliminate_checker.cpp
 * \brief
 */

#include "common_operation_eliminate_checker.h"

namespace npu {
namespace tile_fwk {
Status CommonOperationEliminateChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for CommonOperationEliminate.");
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpAttribute() != nullptr) {
            size_t fromOffsetSize = -1;
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op->GetOpAttribute().get())) {
                auto &fromOffset = viewOpAttribute->GetFromOffset();
                fromOffsetSize = fromOffset.size();
            } else if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(op->GetOpAttribute().get())) {
                if (copyOpAttribute->IsCopyOut()) {
                    continue;
                }
                auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
                (void)memType;
                fromOffsetSize = fromOffset.size();
            } else {
                continue;
            }
            auto& ioperands = op->GetIOperands();
            if (ioperands.size() != 1) {
                ALOG_ERROR_F("View or Copy_In Operation %d with not one input operand.", op->GetOpMagic());
                return FAILED;
            }
            if (ioperands.front()->offset.size() != fromOffsetSize) {
                ALOG_ERROR_F("View or Copy_In Operation %d with mismatch input offset shape.", op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status CommonOperationEliminateChecker::DoPostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for CommonOperationEliminate.");
    operationCache_.clear();
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (OpAlreadyExist(op)) {
            ALOG_ERROR_F("Redundant Operation %d still exist after CommonOperationEliminate.", op->GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

bool CommonOperationEliminateChecker::OpAlreadyExist(Operation *op) {
    auto existOp = OperationExist(op);
    if (existOp == nullptr || op->GetOOperands().size() == 0 || existOp->GetOOperands().size() == 0) {
        return false;
    }
    if (op->GetOOperands().front()->shape == existOp->GetOOperands().front()->shape) {
        auto oldtensor = op->GetOOperands().front();
        if (oldtensor->GetConsumers().size() == 0) {
            return false;
        }
        auto newtensor = existOp->GetOOperands().front();
        if (newtensor->GetMagic() == oldtensor->GetMagic()) {
            ALOG_DEBUG_F("In CommonOperationEliminate, Operation %d is marked as redundant.", op->GetOpMagic());
            return true;
        }
        ALOG_DEBUG_F("In CommonOperationEliminate, Operation %d is marked as redundant.", op->GetOpMagic());
        return true;
    }
    return false;
}

Operation *CommonOperationEliminateChecker::OperationExist(Operation *operation) {
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
    if (operationCache_.count(operation->ComputeHash()) == 0) {
        operationCache_.insert({operation->ComputeHash(), operation});
        return nullptr;
    }
    return operationCache_[operation->ComputeHash()];
}
} // namespace tile_fwk
} // namespace npu