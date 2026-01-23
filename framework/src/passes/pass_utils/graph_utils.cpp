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
 * \file graph_utils.cpp
 * \brief
 */

#include "graph_utils.h"

namespace npu {
namespace tile_fwk {
void GraphUtils::SetDynShape(Operation *newOp, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    if (outDynShape.empty()) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(newOp);
    } else {
        for (size_t i = 0; i < newOp->GetOOperands().size(); ++i) {
            newOp->GetOOperands()[i]->UpdateDynValidShape(outDynShape[i]);
        }
    }
}

Operation &GraphUtils::AddDynOperation(Function &function, const Opcode opCode, LogicalTensors iOperands,
                                       const LogicalTensors &oOperands, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddOperation(opCode, iOperands, oOperands);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddDynRawOperation(Function &function, const Opcode opCode, LogicalTensors iOperands,
                                          const LogicalTensors &oOperands, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddRawOperation(opCode, iOperands, oOperands);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddViewOperation(Function &function, const ViewOp &view, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = AddDynOperation(function, Opcode::OP_VIEW, {view.input}, {view.output}, outDynShape);
    SetViewAttr(function, newOp, view);
    return newOp;
}

Operation &GraphUtils::AddAssembleOperation(Function &function, const AssembleOp &assemble, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {assemble.input}, {assemble.output});
    if (assemble.originOp != nullptr) {
        newOp.SetScopeId(assemble.originOp->GetScopeId());
        newOp.CopyAttrFrom(*assemble.originOp, "");
    }
    SetAssembleAttr(newOp, assemble);
    SetDynShape(&newOp, outDynShape);
    return newOp;
}

Operation &GraphUtils::AddReshapeOperation(Function &function, const LogicalTensorPtr iOperand, const LogicalTensorPtr &oOperand,
    const ReshapeOp &reshapeOp, const std::vector<SymbolicScalar> &outDynShape) {
    auto &newOp = function.AddOperation(Opcode::OP_RESHAPE, {iOperand}, {oOperand});
    if (reshapeOp.originOpPtr != nullptr) {
        newOp.SetScopeId(reshapeOp.originOpPtr->GetScopeId());
        newOp.CopyAttrFrom(*reshapeOp.originOpPtr, "");
    }
    if (outDynShape.empty()) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(&newOp);
        std::vector<SymbolicScalar> validShape;
        if (!newOp.GetAttr(OP_ATTR_PREFIX + "validShape", validShape) || validShape.empty()) {
            newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", oOperand->GetDynValidShape());
        }
    } else {
        newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", outDynShape);
        oOperand->UpdateDynValidShape(outDynShape);
    }
    return newOp;
}

void GraphUtils::SetCopyInAttr(Operation &op, const CopyInOutOp &copy) {
    auto copyAttr = std::make_shared<CopyOpAttribute>(copy.Offset, copy.from, copy.shape, copy.rawShape, copy.fromDynValidShape);
    op.SetOpAttribute(copyAttr);
}

void GraphUtils::SetCopyOutAttr(Operation &op, const CopyInOutOp &copy) {
    auto copyAttr = std::make_shared<CopyOpAttribute>(copy.from, copy.Offset, copy.shape, copy.rawShape, copy.fromDynValidShape);
    op.SetOpAttribute(copyAttr);
}

Operation &GraphUtils::AddCopyInOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddOperation(Opcode::OP_COPY_IN, {copy.input}, {copy.output});
    SetCopyInAttr(newOp, copy);
    SetDynShape(&newOp, outDynShape);
    newOp.UpdateSubgraphID(copy.output->subGraphID);
    return newOp;
}

Operation &GraphUtils::AddCopyOutOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape) {
    auto &newOp = function.AddOperation(Opcode::OP_COPY_OUT, {copy.input}, {copy.output});
    SetCopyOutAttr(newOp, copy);
    SetDynShape(&newOp, outDynShape);
    newOp.UpdateSubgraphID(copy.input->subGraphID);
    return newOp;
}

void GraphUtils::CopyDynStatus(const LogicalTensorPtr &dstTensor, const LogicalTensorPtr &srcTensor) {
    dstTensor->UpdateDynValidShape(srcTensor->GetDynValidShape());
}

void GraphUtils::UpdateViewAttr(Function &function, Operation &op) {
    LogicalTensorPtr input = op.GetIOperands().front();
    LogicalTensorPtr output = op.GetIOperands().front();
    auto viewAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (function.IsFromInCast(input) || function.IsFromOutCast(output)) {
        if (viewAttribute->GetFromDynOffset().empty()) {
            std::vector<int64_t> fromOffset = viewAttribute->GetFromOffset();
            std::vector<SymbolicScalar> fromDynOffset = SymbolicScalar::FromConcrete(fromOffset);
            viewAttribute->SetFromOffset(fromOffset, fromDynOffset);
        }
    }
}

void GraphUtils::SetViewAttr(Function &function, Operation &op, const ViewOp &view) {
    std::vector<SymbolicScalar> toDynShape = view.output->GetDynValidShape();
    auto viewAttribute =std::make_shared<ViewOpAttribute>(view.fromOffset);
    viewAttribute->SetToDynValidShape(toDynShape);
    viewAttribute->SetToType(view.toType);
    op.SetOpAttribute(viewAttribute);
    UpdateViewAttr(function, op);
}

void GraphUtils::SetAssembleAttr(Operation &op, const AssembleOp &assemble) {
    auto assembleOpAttribute = std::make_shared<AssembleOpAttribute>(assemble.from, assemble.toOffset);
    auto fromValidShape = assemble.input->GetDynValidShape();
    assembleOpAttribute->SetFromDynValidShape(fromValidShape);
    op.SetOpAttribute(assembleOpAttribute);
}
}
}