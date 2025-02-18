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
 * \file generate_move_op.cpp
 * \brief
 */

#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_check/generate_move_op_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"

namespace npu::tile_fwk {
Status GenerateMoveOp::RunOnFunction(Function &function) {
    ASLOGI("===> Start GenerateMoveOp");
    Status status = CreateMoveOp(function);
    if(status != SUCCESS) {return status;}
    ASLOGI("===> End GenerateMoveOp");
    return SUCCESS;
}

Status GenerateMoveOp::PreCheck(Function &function) {
    GenerateMoveOpChecker checker;
    return checker.DoPreCheck(function);
}

Status GenerateMoveOp::PostCheck(Function &function) {
    GenerateMoveOpChecker checker;
    return checker.DoPostCheck(function);
}

bool GenerateMoveOp::HasSpecificConsumer(const Operation &op) const {
    auto viewResult = op.GetOOperands()[0];
    auto consumersCopy = viewResult->GetConsumers();

    for (auto childOp : consumersCopy) {
        if (childOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST ||
            childOp->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

Status GenerateMoveOp::CreateMoveOpForView(Operation &op) const {
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    bool isGmInput = op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    bool isGmOutput = op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    if (isGmInput) {
        //case1: VIEW转copyIn
        if (isGmOutput && HasSpecificConsumer(op)) {
            return SUCCESS;
        }
        if ((!isGmOutput)) {
            op.SetOpCode(Opcode::OP_COPY_IN);
            SetCopyAttr(op,viewOpAttribute);
        }
    }else if(op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
        //case2: VIEW转L0A/L0AT
        auto isTrans = (op.HasAttr("op_attr_l1_to_l0_transpose")) ? op.GetBoolAttribute("op_attr_l1_to_l0_transpose") : 0;
        if(isTrans) {
            op.SetOpCode(Opcode::OP_L1_TO_L0_AT);
        }else {
            op.SetOpCode(Opcode::OP_L1_TO_L0A);
        }
        SetCopyAttr(op,viewOpAttribute);
    }else if(op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
        //case3: VIEW转L0B/L0BT
       auto isTrans = (op.HasAttr("op_attr_l1_to_l0_transpose")) ? op.GetBoolAttribute("op_attr_l1_to_l0_transpose") : 0;
        if(isTrans) {
            op.SetOpCode(Opcode::OP_L1_TO_L0_BT);
        }else {
            op.SetOpCode(Opcode::OP_L1_TO_L0B);
        }
        SetCopyAttr(op,viewOpAttribute);
    }else {
        //case4: VIEW转其他搬运op
        auto from = op.iOperand.front()->GetMemoryTypeOriginal();
        auto to = op.oOperand.front()->GetMemoryTypeOriginal();
        if (from == to) {
            return SUCCESS;
        }
        Status status = SetOpcodeByMemPath(op,from,to);
        if(status != SUCCESS) {return status;}
        SetCopyAttr(op,viewOpAttribute);
    }
    return SUCCESS;
}
void GenerateMoveOp::SetCopyAttr(Operation &op,ViewOpAttribute *viewOpAttribute) const {
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()),
        viewOpAttribute->GetTo(), OpImmediate::Specified(op.oOperand.front()->shape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(viewOpAttribute->GetToDynValidShape())
    );
    op.GetOOperands()[0]->UpdateDynValidShape(viewOpAttribute->GetToDynValidShape());
    op.SetOpAttribute(copyAttr);
}

Status GenerateMoveOp::SetOpcodeByMemPath(Operation &op,MemoryType from,MemoryType to) const {
    std::pair<MemoryType,MemoryType> memPathPair = {from,to};
    auto it = platformPathMap.find(memPathPair);
    if (it == platformPathMap.end()) {
        ALOG_ERROR_F("No memory path found from %s to %s for operation %s[%d].",
            BriefMemoryTypeToString(from).c_str(),
            BriefMemoryTypeToString(to).c_str(),
            op.GetOpcodeStr().c_str(),
            op.GetOpMagic());
        return FAILED;
    }
    auto opcodeFindByPath = it->second;
    op.SetOpCode(opcodeFindByPath);
    return SUCCESS;
}

void GenerateMoveOp::CreateMoveOpForAssemble(Operation &op) const {
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
    auto ASSEMBLE_in = op.iOperand.front();
    auto parentOp = *ASSEMBLE_in->GetProducers().begin();
    if (op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR ||
        op.oOperand.front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        parentOp->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT || parentOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        return;
    }
    op.SetOpCode(Opcode::OP_COPY_OUT);
    if (assembleOpAttribute->GetFrom() != ASSEMBLE_in->GetMemoryTypeOriginal()) {
        ALOG_WARN_F(" Assemble op from Attr is different from iOperand, opmagic: %d, do force setting.", op.opmagic);
    }
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(ASSEMBLE_in->GetMemoryTypeOriginal(),
        OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()),
        OpImmediate::Specified(op.iOperand.front()->shape),
        OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(op.iOperand.front()->GetDynValidShape())));
}

Status GenerateMoveOp::CreateMoveOpForConvert(Operation &op) const {
    auto convertOpAttribute = dynamic_cast<ConvertOpAttribute *>(op.GetOpAttribute().get());
    auto [from, to] = convertOpAttribute->GetConvertPath();
    Status status = SetOpcodeByMemPath(op,from,to);
    if(status != SUCCESS) {return status;}
    auto childOp = *op.oOperand.front()->GetConsumers().begin();
    op.UpdateSubgraphID(childOp->GetSubgraphID());
    return SUCCESS;
}

Status GenerateMoveOp::CreateMoveOp(Function &function) const {
    for (auto &op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_ASSEMBLE_SSA:
            case Opcode::OP_ASSEMBLE: {
                CreateMoveOpForAssemble(op);
                break;
            }
            case Opcode::OP_VIEW: {
                Status status = CreateMoveOpForView(op);
                if(status != SUCCESS) {return status;}
                break;
            }
            case Opcode::OP_CONVERT: {
                Status createMoveOpForConvert = CreateMoveOpForConvert(op);
                if(createMoveOpForConvert != SUCCESS) {return createMoveOpForConvert;}
                break;
            }
            case Opcode::OP_DUPLICATE: {
                op.SetOpCode(Opcode::OP_COPY_OUT); //将duplicate转化为copyout
                std::vector<OpImmediate> newOffset;
                for (size_t i = 0; i < op.iOperand.front()->shape.size(); i++) {
                    newOffset.push_back(OpImmediate::Specified(SymbolicScalar(0)));
                }
                op.SetOpAttribute(std::make_shared<CopyOpAttribute>(op.iOperand.front()->GetMemoryTypeOriginal(),
                    newOffset, OpImmediate::Specified(op.iOperand.front()->shape),
                    OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape())));
                break;
            }
            default: break;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
