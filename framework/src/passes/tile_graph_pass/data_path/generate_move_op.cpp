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
constexpr int64_t INNER_PAD_VALUE = 32;
constexpr int64_t OUTER_PAD_VALUE = 16;
const Offset ZERO_OFFSET = {0, 0};

int64_t GenerateMoveOp::PadUB(int64_t dim, int64_t padValue) {
    ASSERT (padValue >0);
    return (dim + padValue - 1) / padValue * padValue;
}

Status GenerateMoveOp::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start GenerateMoveOp");
    Status status = CreateMoveOp(function);
    if(status != SUCCESS) {return status;}
    ALOG_INFO_F("===> End GenerateMoveOp");
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

Status GenerateMoveOp::CreateMoveOpForView(Function &function, Operation &op) const {
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
        if(op.GetOpcode() == Opcode::OP_UB_COPY_L1) {
            ProcessUB2L1(function, op);
        }
        if(op.GetOpcode() == Opcode::OP_L0C_TO_L1) {
            SetL0C2L1CopyAttr(op, op.GetOOperands()[0]->GetShape(), OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()), OpImmediate::Specified(ZERO_OFFSET));
        } else {
            SetCopyAttr(op,viewOpAttribute);
        }
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

void GenerateMoveOp::SetL0C2L1CopyAttr(Operation &op, const Shape &realShape,
    const std::vector<OpImmediate> &fromOffset, const std::vector<OpImmediate> &toOffset) const {
    std::vector<SymbolicScalar> validShape;
    for (auto dim : realShape) {
        SymbolicScalar scal = SymbolicScalar(dim);
        validShape.push_back(scal);
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        fromOffset,
        op.oOperand.front()->GetMemoryTypeOriginal(),
        OpImmediate::Specified(realShape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(validShape)
    );
    copyAttr->SetToOffset(toOffset);
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
    auto inputMemtype = op.iOperand.front()->GetMemoryTypeOriginal();
    auto outputMemtype = op.oOperand.front()->GetMemoryTypeOriginal();
    if (inputMemtype == MemoryType::MEM_L0C && outputMemtype == MemoryType::MEM_L1) {
        SetOpcodeByMemPath(op, inputMemtype, outputMemtype);
        SetL0C2L1CopyAttr(op, op.GetIOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET), OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()));
        return;
    }
    if (inputMemtype == MemoryType::MEM_DEVICE_DDR || outputMemtype != MemoryType::MEM_DEVICE_DDR ||
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

Status GenerateMoveOp::CreateMoveOpForConvert(Function &function, Operation &op) const {
    auto convertOpAttribute = dynamic_cast<ConvertOpAttribute *>(op.GetOpAttribute().get());
    auto [from, to] = convertOpAttribute->GetConvertPath();
    Status status = SetOpcodeByMemPath(op,from,to);
    if(op.GetOpcode() == Opcode::OP_UB_COPY_L1) {
        ProcessUB2L1(function, op);
    }
    if (op.GetOpcode() == Opcode::OP_L0C_TO_L1) {
        SetL0C2L1CopyAttr(op, op.GetOOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET), OpImmediate::Specified(ZERO_OFFSET));
    }
    if(status != SUCCESS) {return status;}
    auto childOp = *op.oOperand.front()->GetConsumers().begin();
    op.UpdateSubgraphID(childOp->GetSubgraphID());
    return SUCCESS;
}
void GenerateMoveOp::ProcessUB2L1(Function &function, Operation &op) const {
    //插入UB2L1节点（NZ2NZ)，并设置UBcopyL1的NZ属性
    op.SetAttribute(OP_ATTR_PREFIX + "is_nz", 1);
    auto inputTensor = op.iOperand.front();
    if(inputTensor->Format() == TileOpFormat::TILEOP_ND) {
        //新建一块logcialtensor
        std::shared_ptr<LogicalTensor> ubNdTensor = inputTensor;
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(ubNdTensor->Datatype(), ubNdTensor->GetShape(), TileOpFormat::TILEOP_NZ);
        std::vector<int64_t> newoffset(inputTensor->GetShape().size(),0);
        std::shared_ptr<LogicalTensor> ubNzTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newoffset, inputTensor->shape, inputTensor->GetDynValidShape());
        //ND转NZ时shape对齐
        auto innerIndex = ubNzTensor->shape.size() - 2; // matmul高轴
        auto outerIndex = ubNzTensor->shape.size() - 1;  // matmul低轴
        ubNzTensor->shape[innerIndex] = GenerateMoveOp::PadUB(ubNzTensor->shape[innerIndex], INNER_PAD_VALUE/BytesOf(ubNdTensor->Datatype()));
        ubNzTensor->shape[outerIndex] = GenerateMoveOp::PadUB(ubNzTensor->shape[outerIndex], OUTER_PAD_VALUE);
        std::vector<int64_t> rawshape_new = ubNdTensor->tensor->rawshape;
        rawshape_new[innerIndex] = GenerateMoveOp::PadUB(ubNzTensor->tensor->rawshape[innerIndex], INNER_PAD_VALUE/BytesOf(ubNdTensor->Datatype()));
        rawshape_new[outerIndex] = GenerateMoveOp::PadUB(ubNzTensor->tensor->rawshape[outerIndex], OUTER_PAD_VALUE);
        ubNzTensor->tensor->UpdateRawShape(rawshape_new);
        ubNzTensor->SetMemoryTypeBoth(MemoryType::MEM_UB);
        //插入UB2UB节点（ND2NZ)
        auto &ub2ub = function.AddRawOperation(Opcode::OP_UB_COPY_ND2NZ, {inputTensor}, {ubNzTensor});
        ub2ub.UpdateSubgraphID(op.GetSubgraphID());

        //图重连
        op.iOperand = {ubNzTensor};
        inputTensor->RemoveConsumer(op);
        ubNzTensor->AddConsumer(op);
    }
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
                Status status = CreateMoveOpForView(function, op);
                if(status != SUCCESS) {return status;}
                break;
            }
            case Opcode::OP_CONVERT: {
                Status createMoveOpForConvert = CreateMoveOpForConvert(function, op);
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
