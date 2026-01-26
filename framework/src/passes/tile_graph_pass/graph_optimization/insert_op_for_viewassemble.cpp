/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file insert_op_for_viewassemble.cpp
 * \brief
 */
#include "insert_op_for_viewassemble.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InsertOpForViewAssemble"

namespace npu {
namespace tile_fwk {
void InsertOpForViewAssemble::InsertViewAssemble(Function &function, Operation *viewOp, Operation *assembleOp) {
    auto &moveOutTensorPtr = viewOp->GetOOperands()[0];
    LogicalTensor ddrTensor(function, moveOutTensorPtr->Datatype(), moveOutTensorPtr->GetShape());
    ddrTensor.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    LogicalTensor moveInTensor(function, moveOutTensorPtr->Datatype(), moveOutTensorPtr->GetShape());
    moveInTensor.SetMemoryTypeBoth(moveOutTensorPtr->GetMemoryTypeOriginal(), true);
    LogicalTensorPtr ddrTensorPtr = std::make_shared<LogicalTensor>(std::move(ddrTensor));
    LogicalTensorPtr moveInTensorPtr = std::make_shared<LogicalTensor>(std::move(moveInTensor));
    std::vector<int64_t> offset(moveOutTensorPtr->GetShape().size(), 0);
    std::vector<SymbolicScalar> dynOffset(moveOutTensorPtr->GetShape().size(), 0);
    Operation &assemble = function.AddRawOperation(Opcode::OP_ASSEMBLE, {moveOutTensorPtr}, {ddrTensorPtr});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(moveOutTensorPtr->GetMemoryTypeOriginal(), 
                                                                  offset, 
                                                                  dynOffset, 
                                                                  moveOutTensorPtr->GetDynValidShape()));
    Operation &view = function.AddRawOperation(Opcode::OP_VIEW, {ddrTensorPtr}, {moveInTensorPtr});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, 
                                                          moveOutTensorPtr->GetMemoryTypeOriginal(), 
                                                          dynOffset, 
                                                          moveOutTensorPtr->GetDynValidShape()));
    assembleOp->ReplaceInput(moveInTensorPtr, moveOutTensorPtr);
}

Status InsertOpForViewAssemble::InsertCopy(Function &function, Operation *viewOp, Operation *assOp) {
    if (assOp->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        assOp->GetIOperands()[0]->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        APASS_LOG_INFO_F(Elements::Operation, "Set Assemble Op[%d] iOperand MEM_DDR.", assOp->GetOpMagic());
    } else if (assOp->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
            assOp->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        APASS_LOG_INFO_F(Elements::Operation, "Insert Copy For View[%d], Assemble[%d].", viewOp->GetOpMagic(), assOp->GetOpMagic());
        InsertViewAssemble(function, viewOp, assOp);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Assemble inTensor %d memory type is unexpected, InsertCopy failed.",
                          assOp->GetIOperands()[0]->GetMagic());
        return FAILED;
    }
    return SUCCESS;
}

bool InsertOpForViewAssemble::NeedInsertCopy(LogicalTensorPtr &assembleOut) {
    bool isNeedInsert = false;
    for (auto &assOp : assembleOut->GetProducers()) {
        auto &prodOp = *assOp->GetIOperands()[0]->GetProducers().begin();
        if (prodOp->GetOpcode() != Opcode::OP_VIEW) {
            isNeedInsert = true;
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] producerOp %s[%d] is not viewOp.", assOp->GetOpMagic(), prodOp->GetOpcodeStr().c_str(), prodOp->GetOpMagic());
            continue;
        }
        recordOpPair_.push_back(std::make_pair(prodOp, assOp));
        if (isNeedInsert) continue;
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(assOp->GetOpAttribute());
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(prodOp->GetOpAttribute());
        if (assembleAttr == nullptr || viewAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "View or Assemble attribute is nullptr, NeedInsertCopy Failed.");
            return false;
        }
        if (assembleAttr->GetToOffset() != viewAttr->GetFromOffset()) {
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetToOffset is not equal to viewOp[%d] GetFromOffset.", assOp->GetOpMagic(), prodOp->GetOpMagic());
            isNeedInsert = true;
            continue;
        }
        if (assembleAttr->GetToDynOffset().size() != viewAttr->GetFromDynOffset().size()) {
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetToDynOffset size is not equal to viewOp[%d] GetFromDynOffset size.", assOp->GetOpMagic(), prodOp->GetOpMagic());
            isNeedInsert = true;
            continue;
        }
        for (size_t i = 0; i < assembleAttr->GetToDynOffset().size(); i++) {
            if (assembleAttr->GetToDynOffset()[i].Dump() != viewAttr->GetFromDynOffset()[i].Dump()) {
                APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetToDynOffset value is not equal to viewOp[%d] GetFromDynOffset value.", assOp->GetOpMagic(), prodOp->GetOpMagic());
                isNeedInsert = true;
                break;
            }
        }
        auto viewIn = prodOp->GetIOperands()[0];
        auto inShape = viewIn->GetShape();
        auto outShape = assembleOut->GetShape();
        if (inShape.size() != outShape.size()) {
            APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetShape size is not equal to viewOp[%d] GetShape size.", assOp->GetOpMagic(), prodOp->GetOpMagic());
            isNeedInsert = true;
            continue;
        }
        for (size_t i = 0; i < inShape.size(); i++) {
            if (inShape[i] != outShape[i]) {
                APASS_LOG_INFO_F(Elements::Operation, "assOp[%d] GetShape value is not equal to viewOp[%d] GetShape value.", assOp->GetOpMagic(), prodOp->GetOpMagic());
                isNeedInsert = true;
                break;
            }
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "isNeedInsert value is %d.", isNeedInsert);
    return isNeedInsert;
}

Status InsertOpForViewAssemble::JudgedViewAssemble(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto &prodOp = *op.GetIOperands()[0]->GetProducers().begin();
        if (prodOp->GetOpcode() == Opcode::OP_VIEW && assembleOutSet_.find(op.GetOOperands()[0]) == assembleOutSet_.end() &&
            prodOp->GetIOperands()[0]->GetMemoryTypeOriginal() == prodOp->GetOOperands()[0]->GetMemoryTypeOriginal()) {
            assembleOutSet_.insert(op.GetOOperands()[0]);
            APASS_LOG_INFO_F(Elements::Operation, "assembleOutSet_ insert oOperand %d", op.GetOOperands()[0]->GetMagic());
        }
    }
    for (auto assembleOut : assembleOutSet_) {
        recordOpPair_.clear();
        if (NeedInsertCopy(assembleOut)) {
            for (auto &opPair : recordOpPair_) {
                auto viewOp = opPair.first;
                auto assOp = opPair.second;
                if (InsertCopy(function, viewOp, assOp) == FAILED) {
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status InsertOpForViewAssemble::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start InsertOpForViewAssemble");
    if (JudgedViewAssemble(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "JudgedViewAssemble Failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End InsertOpForViewAssemble");
    return SUCCESS;
}
}
}