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
 * \file remove_redundant_assemble.cpp
 * \brief
 */

#include "remove_redundant_assemble.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
std::vector<OpImmediate> SumOffset(const std::vector<OpImmediate> offset1, const std::vector<OpImmediate> offset2) {
    std::vector<OpImmediate> res;
    for (size_t i = 0; i < offset1.size(); i++) {
        res.push_back(offset1[i] + offset2[i]);
    }
    return res;
}

// 当前op为Copy Out时，需要将后继Assemble上的offset累加到当前op的CopyOpAttr上
void UpdateCopyOutAttr(Operation &op, Operation &opNext) {
    auto opAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto opNextAttr = std::static_pointer_cast<AssembleOpAttribute>(opNext.GetOpAttribute());
    if (opNextAttr->GetToDynOffset().size() != 0) {
        if (op.GetOpcode() != Opcode::OP_COPY_OUT) {
            opAttr->SetToOffset(OpImmediate::Specified(opNextAttr->GetToDynOffset()));
        } else {
            opAttr->SetToOffset(SumOffset(OpImmediate::Specified(opNextAttr->GetToDynOffset()), opAttr->GetToOffset()));
        }
    }
    opAttr->SetRawShape(OpImmediate::Specified(op.GetOOperands().front()->tensor->GetDynRawShape()));
}

bool CalculateNewRawShape(
    const std::vector<int64_t> &newShape, const std::vector<int64_t> &oriRawShape, std::vector<int64_t> &newRawShape) {
    newRawShape.resize(newShape.size());
    size_t diff = oriRawShape.size() - newShape.size();
    std::copy(oriRawShape.begin() + diff, oriRawShape.end(), newRawShape.begin());
    int64_t newShapeSize = 1;
    if (newRawShape.size() > 1) {
        newShapeSize =
            std::accumulate(newRawShape.begin() + 1, newRawShape.end(), INT64_C(1), std::multiplies<int64_t>());
    }
    int64_t oriShapeSize =
        std::accumulate(oriRawShape.begin(), oriRawShape.end(), INT64_C(1), std::multiplies<int64_t>());
    if (newShapeSize == 0 || oriShapeSize % newShapeSize != 0) {
        APASS_LOG_INFO_F(Elements::Function, "Cannot calculate NewRawShape as the dimension is not divisible.");
        return false;
    }
    newRawShape[0] = oriShapeSize / newShapeSize;
    return true;
}

void RemoveRedundantAssemble::HandleForAssembleFromInOut(Function &function, Operation &assembleOp,
    std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const {
    LogicalTensorPtr inOrOutTensor = nullptr;
    if (function.IsFromInCast(assembleOp.iOperand[0]) || function.IsFromOutCast(assembleOp.iOperand[0])) {
        inOrOutTensor = assembleOp.iOperand[0];            
    }
    if (inOrOutTensor == nullptr) {
        return;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Find incast or outcast, tensor magic: %d, raw magic: %d.", inOrOutTensor->magic, inOrOutTensor->GetRawMagic());
    for (const auto &producer : producersBackup) {
        producer->oOperand[0]->tensor = inOrOutTensor->tensor;
        for (auto &cons : producer->oOperand[0]->GetConsumers()) {
            if (cons->GetOpcode() == Opcode::OP_RESHAPE && cons->oOperand[0]->tensor->actualRawmagic != -1) {
                APASS_LOG_DEBUG_F(Elements::Operation, "consumer[%d] is OP_RESHAPE.", cons->GetOpMagic());
                cons->oOperand[0]->tensor->actualRawmagic = inOrOutTensor->GetRawMagic();
            }
        }
    }
}

void GetDynOffsetBeforeReshape(const std::vector<SymbolicScalar> &oriOffset, const std::vector<int64_t> &oriShape,
    const std::vector<int64_t> &newShape, std::vector<SymbolicScalar> &newOffset) {
    // 计算原始shape的步长（stride）
    if (oriShape.size() != oriOffset.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "OriShape and oriOffset size mismatch.");
        return;
    }
    size_t oriSize = oriOffset.size();
    size_t newSize = newShape.size();
    std::vector<int64_t> oriStride(oriShape.size());
    int64_t currentStride = 1;
    for (int i = oriSize - 1; i >= 0; --i) {
        oriStride[i] = currentStride;
        currentStride *= oriShape[i];
    }
    // 计算原始偏移量对应的线性索引
    SymbolicScalar linearIndex = oriOffset[0] * SymbolicScalar(oriStride[0]);
    for (size_t i = 1; i < oriOffset.size(); ++i) {
        linearIndex = linearIndex + oriOffset[i] * SymbolicScalar(oriStride[i]);
    }

    // 计算新shape的步长
    std::vector<int64_t> newStride(newSize);
    currentStride = 1;
    for (int i = newSize - 1; i >= 0; --i) {
        newStride[i] = currentStride;
        currentStride *= newShape[i];
    }

    // 根据线性索引计算新的偏移量
    newOffset.resize(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        newOffset[i] = linearIndex / SymbolicScalar(newStride[i]);
        linearIndex = linearIndex % SymbolicScalar(newStride[i]);
    }
}

std::vector<int64_t> removeAllOnes(const std::vector<int64_t> &vec) {
    std::vector<int64_t> result = vec;
    result.erase(std::remove(result.begin(), result.end(), 1), result.end());
    return result;
}

bool MatchReshapePattern(const LogicalTensorPtr &reshapeInput, const LogicalTensorPtr &reshapeOutput) {
    auto inputShape = reshapeInput->GetShape();
    auto outputShape = reshapeOutput->GetShape();
    if (std::max(inputShape.size(), outputShape.size()) < 3) {
        return false;
    }
    return removeAllOnes(inputShape) == removeAllOnes(outputShape);
}

/*
删除冗余的VIEW处理场景:
Brfore：
VIEW -> RESHAPE -> COPYIN
                -> COPYIN
After：
RESHAPE -> COPYIN
        -> COPYIN
*/
Status RemoveRedundantAssemble::ProcessView(Function &function) const {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) continue;
        auto &reshapeOp = op;
        if (!MatchReshapePattern(reshapeOp.GetIOperands().front(), reshapeOp.GetOOperands().front())) continue;
        auto producers = reshapeOp.GetIOperands().front()->GetProducers();
        if (producers.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "No producers found for RESHAPE op's input %d.", reshapeOp.GetOpMagic());
            continue;;
        }
        auto producerOp = *producers.begin();
        if (producerOp == nullptr || producers.size() != 1 || producerOp->GetOpcode() != Opcode::OP_VIEW) continue;
        auto viewInput = producerOp->GetIOperands().front();
        for (auto reshapeConsumer : reshapeOp.GetOOperands().front()->GetConsumers()) {
            if (reshapeConsumer->GetOpcode() != Opcode::OP_COPY_IN) return SUCCESS;
        }
        auto opAttr = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
        if (opAttr == nullptr) {
            APASS_LOG_INFO_F(Elements::Operation, "Op %d Attribute is nullptr.", producerOp->GetOpMagic());
            continue;;
        }
        auto &offset = opAttr->GetFromDynOffset();
        std::vector<int64_t> newRawShape = reshapeOp.GetOOperands().front()->shape;
        if (!CalculateNewRawShape(reshapeOp.GetOOperands().front()->shape, viewInput->tensor->GetRawShape(), newRawShape)) return SUCCESS;
        std::vector<SymbolicScalar> newDynOffset;
        GetDynOffsetBeforeReshape(offset, viewInput->shape, newRawShape, newDynOffset);
        APASS_LOG_DEBUG_F(Elements::Operation, "Process View[%d] Tensor[%d]: newRawshape: %s, newOffset: %s.",
            producerOp->GetOpMagic(), reshapeOp.GetOOperands().front()->GetMagic(), IntVecToStr(newRawShape).c_str(),
            IntVecToStr(newDynOffset).c_str());
        for (auto copyIn : reshapeOp.GetOOperands().front()->GetConsumers()) {
            auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
            if (copyAttr == nullptr) {
                APASS_LOG_INFO_F(Elements::Operation, "CopyIn Op %d Attribute is nullptr.", copyIn->GetOpMagic());
                continue;
            }
            auto oriCopyOffset = copyAttr->GetFromOffset();
            std::vector<OpImmediate> newOffset = OpImmediate::Specified(newDynOffset);
            for (size_t i = 0; i < oriCopyOffset.size(); i++) {
                newOffset[i] = newOffset[i] + oriCopyOffset[i];
            }
            copyAttr->SetFromOffset(newOffset);
            copyAttr->SetRawShape(OpImmediate::Specified(newRawShape));
        }
        reshapeOp.GetOOperands().front()->shape = newRawShape;
        reshapeOp.GetOOperands().front()->tensor->UpdateRawShape(newRawShape);
        reshapeOp.ReplaceIOperand(0, viewInput);
        producerOp->SetAsDeleted();
    }
    return SUCCESS;
}

// large , small
bool RemoveViewMultiReshapePattern(const LogicalTensorPtr &reshapeInput, const LogicalTensorPtr &reshapeOutput) {
    auto longerRawShape = reshapeInput->GetRawTensor()->GetRawShape();
    auto shorterRawShape = reshapeOutput->GetRawTensor()->GetRawShape();

    // 确保longerRawShape是维度数更大的一个
    if (longerRawShape.size() < shorterRawShape.size()) {
        std::swap(longerRawShape, shorterRawShape);
    }

    // 维度数一致或太小（小于1）则失败
    if (longerRawShape.size() == shorterRawShape.size() || std::min(longerRawShape.size(), shorterRawShape.size()) < 1) {
        return false;
    }

    // 检查总元素数量是否一致
    auto longerRawShapeSize = reshapeInput->GetRawTensor()->GetRawShapeSize();
    auto shorterRawShapeSize = reshapeOutput->GetRawTensor()->GetRawShapeSize();
    if (longerRawShapeSize != shorterRawShapeSize) {
        return false;
    }

    // 检查维度为1的条件
    auto haveOne = longerRawShape[0] == 1 || longerRawShape[1] == 1;
    return haveOne && longerRawShape[0] * longerRawShape[1] == shorterRawShape[0];
}

/*
拷贝一个RESHAPE
Before:
RESHAPE -> VIEW -> RESHAPE
        -> COPYIN
        -> COPYIN

After:
RESHAPE -> VIEW -> RESHAPE
RESHAPE -> COPYIN
        -> COPYIN
*/
Status ProcessReshape(Function &function, Operation *&operation) {
    auto iOperand = operation->iOperand[0];
    auto oOperand = operation->oOperand[0];
    if (oOperand == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Null output operands detected while iterating over the output operands of the operation [%d].%s",
            operation->opmagic, GetFormatBacktrace(operation).c_str());
        return FAILED;
    }
    if (iOperand == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "Null input operands detected while iterating over the input operands of the operation [%d].%s",
            operation->opmagic, GetFormatBacktrace(operation).c_str());
        return FAILED;
    }
    auto consumers = oOperand->GetConsumers();
    for (auto &consumer : consumers) {
        if (consumer == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Null consumer detected while iterating over the consumers of the output operand [%d].", oOperand->magic);
            return FAILED;
        }
        if (consumer->GetOpcode() == Opcode::OP_COPY_IN) {
            continue;
        }
        auto dst = oOperand->Clone(function, true);
        if (dst == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Clone failed for output operand [%d].", oOperand->magic);
            return FAILED;
        }
        consumer->ReplaceInput(dst, oOperand);
        auto &newReshapeOp = function.AddRawOperation(Opcode::OP_RESHAPE, {iOperand}, {dst});
        const std::shared_ptr<OpAttribute> oriReshapeAttr = operation->GetOpAttribute();
        if (oriReshapeAttr != nullptr) {
            newReshapeOp.SetOpAttribute(oriReshapeAttr);
        }
    }
    return SUCCESS;
}

/*
拷贝一个RESHAPE和删除冗余RESHAPE
Before:
RESHAPE1 -> VIEW -> RESHAPE2
         -> COPYIN
         -> COPYIN

After:
RESHAPE2
RESHAPE -> COPYIN
        -> COPYIN
*/
Status RemoveViewMultiReshape(Function &function) {
    for (auto op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        auto firstReshape = op;
        if (!RemoveViewMultiReshapePattern(
                firstReshape->GetIOperands().front(), firstReshape->GetOOperands().front())) {
            continue;
        }
        auto consumer = firstReshape->GetOOperands().front()->GetConsumers();
        for (auto consumerOp : consumer) {
            if (consumerOp->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto viewConsumers = consumerOp->GetOOperands().front()->GetConsumers();
            auto viewConsumerOp = *viewConsumers.begin();
            if (viewConsumerOp == nullptr || viewConsumers.size() != 1 ||
                viewConsumerOp->GetOpcode() != Opcode::OP_RESHAPE) {
                continue;
            }
            if (ProcessReshape(function, firstReshape) != SUCCESS) {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "ProcessReshape failed. %s", GetFormatBacktrace(firstReshape).c_str());
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "Match RemoveViewMultiReshape pattern %d -> %d -> %d",
                firstReshape->GetOpMagic(), consumerOp->GetOpMagic(), viewConsumerOp->GetOpMagic());

            auto oriRawShape = viewConsumerOp->GetIOperands().front()->GetRawTensor()->GetRawShape();
            Shape newShape;
            std::remove_copy_if(oriRawShape.begin(), oriRawShape.end(), std::back_inserter(newShape),
                [](const auto &e) { return e == 1; });
            viewConsumerOp->GetOOperands().front()->GetRawTensor()->UpdateRawShape(newShape);
            viewConsumerOp->ReplaceIOperand(0, firstReshape->GetIOperands().front());
            firstReshape->SetAsDeleted();
            consumerOp->SetAsDeleted();
        }
    }
    return SUCCESS;
}

/*
生效场景:
Assemble拆分了最高轴，认为可以透传，不需要拷贝，前序在ExpandFunction中做了判断，属性NeedCopy=false
Copy_Out --> tensor(GM) --> Reshape --> oriBackUp [16, 16] --> Assemble(offset, dynOffset) --> OCAST(offset, dynOffset) [16, 64]
因此需要: 重新计算Reshape输入的RawShape, offset, dynOffset
*/
Status HandleDynOffsetForReshape(
    Operation &assembleOp, const std::set<Operation *, LogicalTensor::CompareOp> &producers) {
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<int64_t> newRawShape;
    auto opAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp.GetOpAttribute());
    if (opAttr == nullptr) return FAILED;
    auto &dynOffset = opAttr->GetToDynOffset();
    if (dynOffset.empty()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] does not have DynOffset attributes", 
            assembleOp.GetOpcodeStr().c_str(), assembleOp.GetOpMagic());
        return SUCCESS;
    }
    if (producers.size() != 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] has multiple producer operations, size: %zu", 
            assembleOp.GetOpcodeStr().c_str(), assembleOp.GetOpMagic(), producers.size());
        return SUCCESS;
    }
    auto producer = *(producers.begin());
    if (producer->GetOpcode() != Opcode::OP_RESHAPE) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Producer op:%s[%d] is not Reshape", 
            producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
        return SUCCESS;
    }
    auto &assembleOutShape = assembleOp.GetOOperands()[0]->tensor->rawshape;
    if (!CalculateNewRawShape(producer->GetIOperands()[0]->shape, assembleOutShape, newRawShape)) {
        return SUCCESS;
    }
    GetDynOffsetBeforeReshape(dynOffset, assembleOutShape, newRawShape, newDynOffset);
    APASS_LOG_DEBUG_F(Elements::Operation, "Process Assemble %d Tensor[%d]: newRawshape: %s, newOffset: %s.",
        assembleOp.GetOpMagic(), producer->GetIOperands()[0]->GetMagic(), IntVecToStr(newRawShape).c_str(),
        IntVecToStr(newDynOffset).c_str());
    for (auto copyOut : producer->GetIOperands()[0]->GetProducers()) {
        if (!IsCopyOut(copyOut->GetOpcode())) return SUCCESS;
        const std::shared_ptr<OpAttribute> &attr = copyOut->GetOpAttribute();
        if (attr == nullptr) return FAILED;
        std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
        auto oriCopyOffset = copyAttr->GetToOffset();
        std::vector<OpImmediate> newOffset = OpImmediate::Specified(newDynOffset);
        for (size_t i = 0; i < oriCopyOffset.size(); i++) {
            newOffset[i] = newOffset[i] + oriCopyOffset[i];
        }
        copyAttr->SetRawShape(OpImmediate::Specified(newRawShape));
        copyAttr->SetToOffset(newOffset);
    }
    producer->GetIOperands()[0]->tensor->UpdateRawShape(newRawShape);
    return SUCCESS;
}

/* 将某个op的输入是expected的替换为newTensor并刷新Producer、Consumer关系 */
void SubstituteInput(Operation &op, LogicalTensorPtr &expected, LogicalTensorPtr &newTensor) {
    for (auto &input : op.iOperand) {
        if (input == expected) {
            newTensor->AddConsumer(op);
            input->RemoveConsumer(op);
            input = newTensor;
        }
    }
}

bool RemoveRedundantAssemble::IsCandidateAssembleOp(Function &function, Operation &op) const {
    if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    auto &output = op.GetOOperands().front();
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR || op.IsDeleted()) {
        return false;
    }
    for (auto &prod : function.FindProducers(op)) {
        if (prod->GetOpcode() != Opcode::OP_VIEW) {
            if (prod->GetOpcode() == Opcode::OP_HUB) {
                return false;
            }
            return true;
        }
    }
    return false;
}

void RemoveRedundantAssemble::HandleForReshapeToOutcast(Function &function) const {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            auto dstRT = op.GetOOperands().front()->tensor;
            auto srcRT = op.GetIOperands().front()->tensor;
            if (function.outIncastLinkMap.count(dstRT) && function.outIncastLinkMap[dstRT] == srcRT) {
                continue;
            }
            if (function.IsFromOutCast(op.GetOOperands()[0])) {
                // INPUT --> RESHAPE --> OCAST
                if (op.GetIOperands()[0]->tensor->actualRawmagic != -1) {
                    // 说明输入也来自于reshape，需要找到指向的raw tensor，并更新其actual raw
                    int inputActualRawId = op.GetIOperands()[0]->tensor->actualRawmagic;
                    auto inputRaw = function.GetTensorMap().GetRawTensorByRawMagic(inputActualRawId);
                    inputRaw->actualRawmagic = op.GetOOperands()[0]->GetRawMagic();
                }
                op.GetIOperands()[0]->tensor->actualRawmagic = op.GetOOperands()[0]->GetRawMagic();
            }
        }
    }
}

/*
                /--> Assemble1-1(self) --> Tensor
op1 --> tensor1 ---> Assemble1-2 --> OCAST
                \--> Reshape --> tensor2
*/
/*
                /--> Assemble1-1(self) --> Tensor
op1 --> tensor1 ---> Assemble1-2 --> OCAST

                /--> Assemble2-1 --> Tensor
op2 --> tensor2 ---> Assemble2-2 --> OCAST
*/
void RemoveRedundantAssemble::HandleForAssembleToOutcast(Function &function, Operation& assembleOp,
    std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const {
    int outCastMagic = -1;
    if (function.IsFromOutCast(assembleOp.oOperand[0]) && assembleOp.oOperand[0]->nodetype == NodeType::OUTCAST) {
        outCastMagic = assembleOp.oOperand[0]->GetMagic();
    }
    if (outCastMagic != -1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Find outCastMagic: %d.", outCastMagic);
        for (auto &producer : producersBackup) {
            producer->oOperand[0]->SetMagic(outCastMagic);
            producer->oOperand[0]->nodetype = NodeType::OUTCAST;
        }
    }
}

void RemoveRedundantAssemble::HanldeForMultiAssemble(Function &function, std::unordered_set<Operation *>& concurrentAssembles) const {
    LogicalTensorPtr replaceTensor = nullptr;
    for (auto &assemble : concurrentAssembles) {
        if (function.IsFromInCast(assemble->iOperand[0]) || function.IsFromOutCast(assemble->iOperand[0])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d]'s iOperand comes from Incast or Outcast", 
                assemble->GetOpcodeStr().c_str(), assemble->GetOpMagic());
            replaceTensor = assemble->iOperand[0];
            break;
        } else if (function.IsFromInCast(assemble->oOperand[0]) || function.IsFromOutCast(assemble->oOperand[0])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d]'s oOperand comes from Incast or Outcast", 
                assemble->GetOpcodeStr().c_str(), assemble->GetOpMagic());
            replaceTensor = assemble->oOperand[0];
            break;
        }
    }
    for (auto& assemble : concurrentAssembles) {
        if (replaceTensor == nullptr) replaceTensor = assemble->oOperand[0];
        auto &input = assemble->GetIOperands().front();
        auto &output = assemble->GetOOperands().front();
        input->tensor = replaceTensor->tensor;
        output->tensor = replaceTensor->tensor;
    }
}

Status RemoveRedundantAssemble::HanldeForSingleAssemble(Function &function, LogicalTensorPtr input, LogicalTensorPtr output, Operation &op) const {
    auto producersBackup = input->GetProducers();
    auto &consumers = input->GetConsumers();
    LogicalTensorPtr oriOutputBackUp = nullptr;
    for (auto &cons : consumers) {
        if (cons->GetOpcode() != Opcode::OP_ASSEMBLE) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Change the connection relationship of non assemble op:[%d]. %s",
                cons->GetOpMagic(), cons->GetOpcodeStr().c_str());
            cons->iOperand[0] = output;
            cons->iOperand[0]->AddConsumer(cons);
            continue;
        }
        cons->SetAsDeleted();
        for (auto &producer : producersBackup) {
            oriOutputBackUp = producer->oOperand[0]; // producer --> oriOutputBackUp(input) --> op
            producer->ReplaceOutput(output, oriOutputBackUp);
            output->isSubGraphBoundary = true;
            if (!IsCopyOut(producer->GetOpcode())) continue;
            APASS_LOG_DEBUG_F(Elements::Operation, "The producer op:[%d] is copyOut, update its CopyOpAttr. %s",
                producer->GetOpMagic(), producer->GetOpcodeStr().c_str());
            UpdateCopyOutAttr(*producer, *cons);
        }
    }
    HandleForAssembleFromInOut(function, op, producersBackup);
    HandleForAssembleToOutcast(function, op, producersBackup);
    if (HandleDynOffsetForReshape(op, producersBackup) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "HandleDynOffsetForReshape for op:[%d] failed. %s", op.GetOpMagic(),
            op.GetOpcodeStr().c_str());
        return FAILED;
    }
    return SUCCESS;
}

/*
    Producer1 -->
                 \
    Producer2 --> input --> consAssemble -----> Tensor1 --> Op1
                    \---> consumer ------> Tensor2 --> Op2
    will be modified to:
    Producer1 -->
                 \
    Producer2 --> Tensor1 --> Op1
                    \---> Consumer ------> Tensor2 --> Op2
*/
Status RemoveRedundantAssemble::DeleteRedundantAssemble(Function &function) const {
    for (auto &op : function.Operations()) {
        if (!IsCandidateAssembleOp(function, op)) {
            continue;
        }
        auto &input = op.GetIOperands().front();
        auto &output = op.GetOOperands().front();
        auto &consumers = input->GetConsumers();
        std::unordered_set<Operation *> concurrentAssembles;
        for (auto &cons : consumers) {
            if (cons->GetOpcode() == Opcode::OP_ASSEMBLE) {
                concurrentAssembles.emplace(cons);
            }
        }
        if (concurrentAssembles.size() > 1) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] has %zu parallel assemble op", 
                op.GetOpcodeStr().c_str(), op.GetOpMagic(), concurrentAssembles.size());
            HanldeForMultiAssemble(function, concurrentAssembles);
        } else {
            if (HanldeForSingleAssemble(function, input, output, op) != SUCCESS) return FAILED;
        }
    }
    if (RemoveViewMultiReshape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "RemoveViewMultiReshape failed.");
        return FAILED;
    }
    if (ProcessView(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessView failed.");
        return FAILED;
    }
    function.EraseOperations(false);
    HandleForReshapeToOutcast(function);
    return SUCCESS;
}
} // namespace npu::tile_fwk