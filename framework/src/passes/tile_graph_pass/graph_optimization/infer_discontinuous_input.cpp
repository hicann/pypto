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
 * \file infer_memory_conflict.cpp
 * \brief
 */

#include "infer_discontinuous_input.h"
#include <queue>
#include "passes/pass_log/pass_log.h"
#include "passes/pass_check/infer_discontinuous_input_checker.h"

#define MODULE_NAME "InferDiscontinuousInput"

namespace npu {
namespace tile_fwk {
Status InferDiscontinuousInput::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferDiscontinuousInput for function [%s].", function.GetRawName().c_str());
    Init(function);
    if (InferFromIncast() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Infer INCAST and OUTCAST address failed.");
        return FAILED;
    }
    if (InsertTensorCopy(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Insert copy op failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End InferDiscontinuousInput for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

std::vector<std::pair<LogicalTensorPtr, Operation *>> GetInplacedTileTensors(LogicalTensorPtr targetTensor) {
    std::unordered_set<Opcode> inplaceNodes{
        Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_RESHAPE, Opcode::OP_INDEX_OUTCAST};
    std::vector<std::pair<LogicalTensorPtr, Operation *>> inplacedTensor;
    for (auto &producer : targetTensor->GetProducers()) {
        if (inplaceNodes.count(producer->GetOpcode()) == 0) {
            continue;
        }
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            auto consumerOp = *(targetTensor->GetConsumers().begin());
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                continue;
            }
            const int index = 2;
            inplacedTensor.emplace_back(std::make_pair(producer->GetInputOperand(index), producer));
            continue;
        }
        for (auto &inputTensor : producer->GetIOperands()) {
            inplacedTensor.emplace_back(std::make_pair(inputTensor, producer));
        }
    }
    return inplacedTensor;
}

inline int64_t ShapeToSize(Shape &shapes) {
    int64_t sz = 1;
    for (int dimValue : shapes) {
        sz *= dimValue;
    }
    return sz;
}

inline bool VecEqual(Offset &vec1, Offset &vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    for (size_t i = 0; i < vec1.size(); i++) {
        if (vec1[i] != vec2[i]) {
            return false;
        }
    }
    return true;
}

inline bool PerfectOffsetOverlap(std::vector<int> &rawTensorIds, std::vector<Shape> &rawShapes,
                                 std::vector<Shape> &shapes, std::vector<Offset> &offsets,
                                 std::vector<Offset> &offsetTos)
{
    std::unordered_map<int, Offset> rawIdToRawOffset;
    std::unordered_map<int, int64_t> rawEmptySize;
    std::unordered_map<int, int64_t> rawValueSize;
    for (int i = 0; i < static_cast<int>(rawTensorIds.size()); i++) {
        int rawId = rawTensorIds[i];
        Offset rawOffset(rawShapes[i].size(), 0);
        for (size_t dim = 0; dim < rawShapes[i].size(); dim++) {
            rawOffset[dim] = offsetTos[i][dim] - offsets[i][dim];
        }
        if (rawIdToRawOffset.count(rawId) == 0) {
            rawIdToRawOffset[rawId] = rawOffset;
        } else {
            if (!VecEqual(rawIdToRawOffset[rawId], rawOffset)) {
                return false;
            }
        }
        if (rawEmptySize.count(rawId) == 0) {
            rawEmptySize[rawId] = ShapeToSize(rawShapes[i]);
        }
        rawValueSize[rawId] += ShapeToSize(shapes[i]);
    }
    for (const auto &rawPr : rawEmptySize) {
        if (rawPr.second != rawValueSize[rawPr.first]) {
            return false;
        }
    }
    return true;
}

inline bool IsTraceableView(Operation* cur) {
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(cur->GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        return false;
    }
    for (auto &fromOffset : viewOpAttribute->GetFromDynOffset()) {
        if (fromOffset.IsSymbol()) {
            return false;
        }
        if (fromOffset.IsExpression()) {
            return false;
        }
    }
    for (auto &dynShape : viewOpAttribute->GetToDynValidShape()) {
        if (dynShape.IsSymbol()) {
            return false;
        }
        if (dynShape.IsExpression()) {
            return false;
        }
    }
    return true;
}

inline bool NoViewConflict(const std::vector<std::pair<LogicalTensorPtr, Operation*>> &inplaceTensors)
{
    std::vector<Operation*> viewOps(inplaceTensors.size(), nullptr);
    for (size_t i = 0; i < inplaceTensors.size(); i++) {
        auto tensor = inplaceTensors[i].first;
        for (auto &producer : tensor->GetProducers()) {
            if (producer->GetOpcode() == Opcode::OP_VIEW) {
                viewOps[i] = producer;
            }
        }
    }
    for (size_t i = 0; i < inplaceTensors.size(); i++) {
        if (viewOps[i] == nullptr) {
            continue;
        }
        // dynamic view check
        if (!IsTraceableView(viewOps[i])) {
            return false;
        }
        // incast outcast Check
        for (auto &producerTensor : viewOps[i]->GetIOperands()) {
            if (producerTensor->nodetype != NodeType::LOCAL) {
                return false;
            }
        }
    }
    return true;
}

inline std::vector<size_t> GetInputTileConflict(
    const std::vector<std::pair<LogicalTensorPtr, Operation*>> &inplaceTensors) {
    std::vector<int> rawTensorMagics;
    std::vector<Shape> rawShapes;
    std::vector<Shape> shapes;
    std::vector<Offset> offsets;
    std::vector<Offset> offsetTos;

    bool assembleCheck = true;
    for (const auto &pr : inplaceTensors) {
        if (pr.first->GetMemoryTypeOriginal() != pr.second->GetOOperands()[0]->GetMemoryTypeOriginal()) {
            assembleCheck = false;
            break;
        }
        std::shared_ptr<AssembleOpAttribute> attr =
            std::dynamic_pointer_cast<AssembleOpAttribute>(pr.second->GetOpAttribute());
        if (attr == nullptr) {
            assembleCheck = false;
            break;
        }
        offsetTos.push_back(attr->GetToOffset());
        rawTensorMagics.push_back(pr.first->GetRawMagic());
        rawShapes.push_back(pr.first->GetRawTensor()->GetRawShape());
        shapes.push_back(pr.first->GetShape());
        offsets.push_back(pr.first->GetOffset());
    }
    std::vector<size_t> copyIdx;
    if (!assembleCheck) {
        return {};
    }
    if (!(PerfectOffsetOverlap(rawTensorMagics, rawShapes, shapes, offsets, offsetTos) &&
            NoViewConflict(inplaceTensors)) &&
        inplaceTensors.size() > 1) {
        for (size_t i = 0; i < inplaceTensors.size(); i++) {
            copyIdx.push_back(i);
        }
    }

    return copyIdx;
}

std::vector<std::pair<LogicalTensorPtr, Operation *>> InferDiscontinuousInput::FilterCopyScenes(
    const std::vector<std::pair<LogicalTensorPtr, Operation*>> &inplaceTensors) {
    std::vector<std::pair<LogicalTensorPtr, Operation *>> needInsertCopys;
    if (inplaceTensors.empty()) {
        return needInsertCopys;
    }
    auto copyIdx = GetInputTileConflict(inplaceTensors);
    for (auto idx : copyIdx) {
        needInsertCopys.push_back(inplaceTensors[idx]);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Input tensor [%d] conflit.", inplaceTensors[idx].first->GetMagic());
    }
    return needInsertCopys;
}

void InferDiscontinuousInput::Init(Function &function) {
    auto opList = function.Operations().DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        opInputDegree_.emplace(opList[i], opList[i]->ProducerOps().size());
        for (auto outTensor : opList[i]->GetOOperands()) {
            tensorProducers_[outTensor] = outTensor->GetProducers().size();
        }
    }
}

// 从INCAST出发，按DFS做前向推导
Status InferDiscontinuousInput::InferFromIncast() {
    std::queue<Operation *> procOpQueue;
    for (auto &opInputDegree : opInputDegree_) {
        if (opInputDegree.second == 0) {
            procOpQueue.push(opInputDegree.first);
        }
    }
    std::set<Operation *> visitedOps;
    while (!procOpQueue.empty()) {
        auto currentOp = procOpQueue.front();
        procOpQueue.pop();
        visitedOps.insert(currentOp);
        for (auto outOp : currentOp->ConsumerOps()) {
            opInputDegree_[outOp]--;
            if (opInputDegree_[outOp] == 0) {
                procOpQueue.push(outOp);
            }
        }
        for (auto &outputTensor : currentOp->GetOOperands()) {
            tensorProducers_[outputTensor]--;
            std::vector<std::pair<LogicalTensorPtr, Operation *>> filterdTensor;
            if (tensorProducers_[outputTensor] != 0) {
                continue;
            }
            auto inplacedTensor = GetInplacedTileTensors(outputTensor);
            filterdTensor = FilterCopyScenes(inplacedTensor);
            insertCopys_.emplace(outputTensor, filterdTensor);
        }
    }
    return SUCCESS;
}
void InferDiscontinuousInput::InsertViewOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    auto &insertViewOp = function.AddRawOperation(Opcode::OP_VIEW, {iOperand}, {oOperand});
    insertViewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(iOperand->GetOffset(),
        oOperand->GetMemoryTypeOriginal(), iOperand->GetDynOffset(), iOperand->GetDynValidShape()));
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert view op [%d].", insertViewOp.GetOpMagic());
}
void InferDiscontinuousInput::InsertAssembleOp(
    Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    auto &insertAssembleOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
    insertAssembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(iOperand->GetMemoryTypeOriginal(),
        oOperand->GetOffset(), oOperand->GetDynOffset(), oOperand->GetDynValidShape()));
     APASS_LOG_DEBUG_F(Elements::Operation, "Insert assemble op [%d].", insertAssembleOp.GetOpMagic());
}

void InferDiscontinuousInput::InsertCopyOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    if ((iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) &&
        (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(iOperand->Datatype(),
            iOperand->GetShape(), iOperand->Format());
        Offset newOffset(iOperand->GetShape().size(), 0);
        LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(
            function, newRawTensor, newOffset, iOperand->GetShape(), iOperand->GetDynValidShape());
        newTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        newTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
        InsertViewOp(function, iOperand, newTensor);
        InsertAssembleOp(function, newTensor, oOperand);
        return;
    }
    if ((iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) &&
        (oOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR)) {
        InsertViewOp(function, iOperand, oOperand);
        return;
    }
    if ((iOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) &&
        (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
        InsertAssembleOp(function, iOperand, oOperand);
        return;
    }
    std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(iOperand->Datatype(),
        iOperand->GetShape(), iOperand->Format());
    Offset newOffset(iOperand->GetShape().size(), 0);
    LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(
        function, newRawTensor, newOffset, iOperand->GetShape(), iOperand->GetDynValidShape());
    newTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    newTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    InsertAssembleOp(function, iOperand, newTensor);
    InsertViewOp(function, newTensor, oOperand);
}

inline void DDRTensorAssignUB(Function &function, std::map<LogicalTensorPtr, std::set<Operation *>> insertedNodes) {
    auto opList = function.Operations().DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        Operation *currOp = opList[i];
        if (currOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (LogicalTensorPtr ioperand : currOp->GetIOperands()) {
            if (ioperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            auto &inOp = *ioperand->GetProducers().begin();
            if (ioperand->GetProducers().size() != 1 || inOp->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto viewOut = inOp->GetOOperands().front();
            auto outShape = viewOut->GetShape();
            bool isDynAxis = false;
            for (size_t dim = 0; dim < outShape.size(); dim++) {
                if (outShape[dim] < 0) {
                    insertedNodes[ioperand].insert(currOp);
                    isDynAxis = true;
                    break;
                }
            }
            if (isDynAxis) {
                continue;;
            }
            ioperand->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            ioperand->SetMemoryTypeToBe(MemoryType::MEM_UB);
            insertedNodes[ioperand].insert(currOp);
        }
    }
}
Status InferDiscontinuousInput::InsertTensorCopy(Function &function) {
    std::map<LogicalTensorPtr, std::set<Operation *>> insertedNodes;
    DDRTensorAssignUB(function, insertedNodes);
    for (auto &copyInserts : insertCopys_) {
        auto &inplaceNodes = copyInserts.second;
        for (auto &inplaceNode : inplaceNodes) {
            auto &inputTensor = inplaceNode.first;
            if (insertedNodes.find(inputTensor) != insertedNodes.end()) {
                if (insertedNodes[inputTensor].count(inplaceNode.second) != 0U) {
                    continue;
                }
            }
            insertedNodes[inputTensor].insert(inplaceNode.second);
            std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(inputTensor->Datatype(),
                inputTensor->GetShape(), inputTensor->Format());
            Offset newOffset(inputTensor->GetShape().size(), 0);
            LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newOffset,
                inputTensor->GetShape(), inputTensor->GetDynValidShape());
            LogicalTensorPtr customTensor = inplaceNode.second->GetOOperands()[0];
            newTensor->SetMemoryTypeOriginal(customTensor->GetMemoryTypeOriginal(), true);
            newTensor->SetMemoryTypeToBe(newTensor->GetMemoryTypeOriginal());
            InsertCopyOp(function, inputTensor, newTensor);
            inputTensor->RemoveConsumer(inplaceNode.second);
            inplaceNode.second->ReplaceInput(newTensor, inputTensor);
        }
    }
    return SUCCESS;
}

Status InferDiscontinuousInput::PostCheck(Function &function) {
    InferDisContinuousInputChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace tile_fwk
} // namespace npu
