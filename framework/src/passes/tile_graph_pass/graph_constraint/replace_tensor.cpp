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
 * \file replace_tensor.cpp
 * \brief
 */

#include "replace_tensor.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "ReplaceTensor"

namespace npu {
namespace tile_fwk {
bool ReplaceTensor::CheckAddrConflict(const Operation& op) {
    auto tensorIn = op.GetIOperands().front();
    auto tensorOut = op.GetOOperands().front();
    if (tensorIn->GetRawMagic() != tensorOut->GetRawMagic() && tensorIn->GetRawTensor()->memoryId != tensorOut->GetRawTensor()->memoryId) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s op[%d] invalid or conflict. tensorIn magic: %d, rawMagic: %d, tensorOut magic: %d, rawMagic: %d",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), tensorIn->GetMagic(), tensorIn->GetRawMagic(), tensorOut->GetMagic(), tensorOut->GetRawMagic());
        return true;
    }
    return false;
}

/*
用于校验assemble节点的输入输出是否存在冲突
注意，若输入由OP_INDEX_OUTCAST构造，则不会出现冲突
*/
bool ReplaceTensor::CheckIndexProducer(const Operation& op) {
    for (const auto &producer : op.ProducerOps()) {
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            return true;
        }
    }
    return false;
}

bool ReplaceTensor::CheckAssembleConflict(const Operation& op) {
    if (!CheckIndexProducer(op) && CheckAddrConflict(op)) {
        return true;
    }
    return false;
}

/*
用于校验index_outcast节点的输入输出是否存在冲突
注意，若输出后接assemble节点，则不会出现冲突
*/
bool ReplaceTensor::CheckIndexOutcastConflict(const Operation& op) {
    int index = 2;
    auto indexIn = op.GetInputOperand(index);
    auto indexOut = op.GetOOperands().front();
    auto &inOp = *indexIn->GetProducers().begin();
    auto &outOp = *indexOut->GetConsumers().begin();
    if (inOp == nullptr && outOp == nullptr) {
        return false;
    }
    if (indexIn->GetRawMagic() != indexOut->GetRawMagic()) {
        return true;
    }
    return false;
}

/*
用于校验reshape节点的输入输出是否存在冲突
需要校验的场景：
    shape输入输出的rawtensor除了首轴之外都一致
*/
bool ReplaceTensor::CheckReshapeConflict(const Operation& op) {
    if (op.GetBoolAttribute(OP_ATTR_PREFIX + "isInplace"))
        return false;
    if (CheckAddrConflict(op)) {
        return true;
    }
    return false;
}

/* 
用于校验a_mulacc_b节点的输入输出是否存在冲突
*/
bool ReplaceTensor::CheckAMulAccBConflict(const Operation& op) {
    int index = 2;
    auto tensorIn = op.GetInputOperand(index);
    auto tensorOut = op.GetOOperands().front();
    auto &inOp = *tensorIn->GetProducers().begin();
    auto &outOp = *tensorOut->GetConsumers().begin();
    if (inOp == nullptr && outOp == nullptr) {
        return false;
    }
    if (tensorIn->GetRawMagic() != tensorOut->GetRawMagic()) {
        return true;
    }
    return false;
}

/*
用于校验remote_reduce节点的输入输出是否存在冲突
*/
bool ReplaceTensor::CheckRemoteReduceConflict(const Operation& op) {
    int index = 0;
    auto reduceIn = op.GetInputOperand(index);
    auto reduceOut = op.GetOOperands().front();
    auto &inOp = *reduceIn->GetProducers().begin();
    auto &outOp = *reduceOut->GetConsumers().begin();
    if (inOp == nullptr && outOp == nullptr) {
        return false;
    }
    if (reduceIn->GetRawMagic() != reduceOut->GetRawMagic()) {
        return true;
    }
    return false;
}

Status ReplaceTensor::InplaceCheck(Function &function) {
    struct OpValidator {
        std::function<bool(const Operation&)> validate;
        std::string opName;
        uint64_t inputCount;
        uint64_t outputCount;
    };
    std::unordered_map<Opcode, OpValidator> opValidators = {
        { Opcode::OP_VIEW, { [this](const Operation &op) { return this->CheckAddrConflict(op); }, "View", 1UL, 1UL } },
        { Opcode::OP_ASSEMBLE, { [this](const Operation &op) { return this->CheckAssembleConflict(op); }, "Assemble", 1UL, 1UL } },
        { Opcode::OP_INDEX_OUTCAST, { [this](const Operation &op) { return this->CheckIndexOutcastConflict(op); }, "Index_Outcast", 3UL, 1UL } },
        { Opcode::OP_RESHAPE, { [this](const Operation &op) { return this->CheckReshapeConflict(op); }, "Reshape", 1UL, 1UL } },
        { Opcode::OP_A_MULACC_B, { [this](const Operation &op) { return this->CheckAMulAccBConflict(op); }, "A_Mulacc_B", 3UL, 1UL } },
        { Opcode::OP_REMOTE_REDUCE, { [this](const Operation &op) { return this->CheckRemoteReduceConflict(op); }, "Remote_Reduce", 1UL, 1UL } },
    };
    for (const auto &op : function.Operations()) {
        auto it = opValidators.find(op.GetOpcode());
        if (it != opValidators.end()) {
            const auto &validator = it->second;
            if (op.GetInputOperandSize() != validator.inputCount || 
                op.GetOutputOperandSize() != validator.outputCount ||
                validator.validate(op)) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s op[%d] invalid or conflict.", validator.opName.c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool ReplaceTensor::CheckInplace(const Operation &op) {
    if (inplaceOpSet.find(op.GetOpcode()) != inplaceOpSet.end()) {
        return true;
    }
    return false;
}


bool ReplaceTensor::HasSameConsecutive(Operation &op) {
    for (auto &nextOp : op.ConsumerOps()) {
        if (nextOp->GetOpcode() == op.GetOpcode()) {
            return true;
        }
    }
    return false;
}
Status ReplaceTensor::PreCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ReplaceTensor.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Loopcheck failed before PreGraph; Please check whether there is a loop.");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] is not partitioned; Please check subGraphIDs. %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) &&
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        if (HasSameConsecutive(op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has the same Opcode child op; Plese check child ops. %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "unmatched input output memory type for reshape opmagic: %d, input mem type: %s, output mem type: %s; Please check the input ans output.", 
                op.opmagic,
                MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ReplaceTensor success.");
    return SUCCESS;
}

Status ReplaceTensor::PostCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor.");
    if (InplaceCheck(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InplaceCheck failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor success.");
    return SUCCESS;
}

void ReplaceTensor::UniteTensor(Function &function, UnionFind &uf) {
    for (const auto &op : function.Operations()) {
        if (CheckInplace(op)) {
            if (inplaceOpMap.find(op.GetOpcode()) != inplaceOpMap.end()) {
                for (const auto &pair : inplaceOpMap.at(op.GetOpcode())) {
                    uf.Unite(op.GetInputOperand(pair.first), op.GetOutputOperand(pair.second));
                    APASS_LOG_INFO_F(Elements::Operation, "Unite %s op[%d] iOperand %d and oOperand %d.",
                                     op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands()[0]->GetMagic(), op.GetOOperands()[0]->GetMagic());
                }
            } else {
                uf.Unite(op.GetIOperands().front(), op.GetOOperands().front());
                APASS_LOG_INFO_F(Elements::Operation, "Unite %s op[%d] iOperand %d and oOperand %d.",
                                     op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands()[0]->GetMagic(), op.GetOOperands()[0]->GetMagic());
            }
        }
        if (op.GetOpcode() == Opcode::OP_COPY_OUT && op.HasAttribute(OpAttributeKey::inplaceIdx)) {
            uf.Unite(op.GetIOperands()[op.GetIntAttribute(OpAttributeKey::inplaceIdx)], op.GetOOperands().front());
        }
    }
}

Status ReplaceTensor::FindBaseTensor(Function &function, LogicalTensorPtr &baseTensor, LogicalTensors &group) {
    std::unordered_set<LogicalTensorPtr> groupTensor;
    LogicalTensors orderGroup;
    for (const auto &curTensor : group) {
        groupTensor.insert(curTensor);
        if (function.IsFromInCast(curTensor) || function.IsFromOutCast(curTensor)) {
            if (baseTensor == nullptr) {
                baseTensor = curTensor;
                APASS_LOG_INFO_F(Elements::Tensor, "Set base Tensor %d", curTensor->GetMagic());
            } else if (baseTensor->Symbol() != curTensor->Symbol() && 
                       baseTensor->GetRawTensor()->memoryId != curTensor->GetRawTensor()->memoryId && 
                       baseTensor->tensor->actualRawmagic != curTensor->tensor->actualRawmagic) {
                APASS_LOG_ERROR_F(Elements::Tensor, "baseTensor %d and curTensor %d has conflict.", 
                                      baseTensor->GetMagic(), curTensor->GetMagic());
                return FAILED;
            } else if (function.IsFromInCast(curTensor)) {
                baseTensor = curTensor;
                APASS_LOG_INFO_F(Elements::Tensor, "Set base Tensor %d", curTensor->GetMagic());
            }
        }
    }
    for (auto &op : function.Operations()) {
        for (auto &ioperand : op.GetIOperands()) {
            if (groupTensor.erase(ioperand)) {
                orderGroup.push_back(ioperand);
            }
        }
        for (auto &ooperand : op.GetOOperands()) {
            if (groupTensor.erase(ooperand)) {
                orderGroup.push_back(ooperand);
            }
        }
        if (groupTensor.empty()) {
            break;
        }
    }
    if (baseTensor == nullptr) {
        baseTensor = orderGroup.front();
        int64_t baseShape = abs(baseTensor->tensor->GetRawDataSize());
        for (auto &curTensor : orderGroup) {
            int64_t curShape = abs(curTensor->tensor->GetRawDataSize());
            if (curShape > baseShape) {
                APASS_LOG_INFO_F(Elements::Tensor, "Replace curTensor %d size %d to baseTensor %d size %d.",
                                curTensor->GetMagic(), curShape, baseTensor->GetMagic(), baseShape);
                baseTensor = curTensor;
                baseShape = curShape;
            }
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardView(Operation *op, LogicalTensorPtr &rootTensor, Function &function) {
    if (ForUpdateView(op) == FAILED) {
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    op->GetOOperands()[0]->tensor = rootTensor->tensor;
    forRoots.push(op->GetOOperands()[0]);
    if (!function.IsFromOutCast(op->GetOOperands()[0])) {
        function.UpdateLinkMap(op->GetOOperands()[0], op->GetIOperands()[0]);
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardReshape(Operation *op, LogicalTensorPtr &rootTensor, Function &function) {
    processedOp.insert(op->GetOpMagic());
    (void) function;
    if (function.IsFromOutCast(op->GetOOperands()[0])) {
        APASS_LOG_INFO_F(Elements::Operation, "OP_RESHAPE %d oOperand is OutCast, Skip inplace.", op->GetOpMagic());
        return SUCCESS;
    }
    op->GetOOperands()[0]->tensor->actualRawmagic = rootTensor->GetRawMagic();
    forRoots.push(op->GetOOperands()[0]);
    return SUCCESS;
}

Status ReplaceTensor::ForwardInplaceOp(Operation *op, LogicalTensorPtr &rootTensor, Function &function) {
    auto reusePairs = inplaceOpMap.at(op->GetOpcode());
    for (const auto &reusePair : reusePairs) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        auto tensorIn = op->GetIOperands()[inputIdx];
        auto tensorOut = op->GetOOperands()[outputIdx];
        if (tensorIn != rootTensor) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(), rootTensor->GetMagic());
            return SUCCESS;
        }
        processedOp.insert(op->GetOpMagic());
        if (function.IsFromInCast(tensorIn) && function.IsFromOutCast(tensorOut)) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is incast, tensorOut %d is outcast.",
                            op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(), tensorOut->GetMagic());
            return SUCCESS;
        }
        tensorOut->tensor = tensorIn->tensor;
        forRoots.push(tensorOut);
        tensorOut->UpdateOffset(tensorIn->GetOffset());
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardViewType(Operation *op, LogicalTensorPtr &rootTensor) {
    auto viewTypeIn = op->GetIOperands()[0];
    auto viewTypeOut = op->GetOOperands()[0];
    if (viewTypeIn != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_VIEW_TYPE %d rootTensor %d is not same as viewTypeIn %d.", op->GetOpMagic(), rootTensor->GetMagic(), viewTypeIn->GetMagic());
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    viewTypeOut->tensor->actualRawmagic = viewTypeIn->GetRawMagic();
    if(AdjustOffsetAndRawShape(viewTypeIn, viewTypeOut) == FAILED) {
        return FAILED;
    }
    forRoots.push(viewTypeOut);
    return SUCCESS;
}

bool isInplaceAssemble(Operation *op) {
    auto assembleIn = op->GetIOperands()[0];
    for (auto inOp : assembleIn->GetProducers()) {
        if (inplaceOpSet.find(inOp->GetOpcode()) != inplaceOpSet.end()) {
            return true;
        }
    }
    return false;
}

bool isMultiAssemble(Operation *op) {
    auto assembleIn = op->GetIOperands()[0];
    for (auto outOp : assembleIn->GetConsumers()) {
        if (outOp->GetOpMagic() != op->GetOpMagic() && outOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            return true;
        }
    }
    return false;
}

Status ReplaceTensor::ForwardAssemble(Operation *op, LogicalTensorPtr &rootTensor) {
    auto assembleIn = op->GetIOperands()[0];
    auto assembleOut = op->GetOOperands()[0];
    if (assembleIn != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_ASSEMBLE %d rootTensor %d is not same as viewTypeIn %d.", op->GetOpMagic(), rootTensor->GetMagic(), assembleIn->GetMagic());
        return FAILED;
    }
    if (isInplaceAssemble(op) || isMultiAssemble(op)) {
        auto &inOp = *(assembleIn)->GetProducers().begin();
        processedOp.insert(op->GetOpMagic());
        forRoots.push(assembleOut);
        if (inOp != nullptr && inOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            APASS_LOG_INFO_F(Elements::Operation, "OP_ASSEMBLE %d parentOp is OP_INDEX_OUTCAST %d, skip replace tensor.", op->GetOpMagic(), inOp->GetOpMagic());
            return SUCCESS;
        }
        assembleOut->tensor = assembleIn->tensor;
        return SUCCESS;
    } else {
        backRoots.push(assembleOut);
        return SUCCESS;
    }
}

Status ReplaceTensor::ForwardCopyOut(Operation *op, LogicalTensorPtr &rootTensor, Function &function) {
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (inTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                            op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (function.IsFromInCast(inTensor) && function.IsFromOutCast(outTensor)) {
        APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] input tensor %d is Incast, output tensor %d is OutCast",
                            op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), outTensor->GetMagic());
                        return SUCCESS; 
    }
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    outTensor->tensor = rootTensor->tensor;
    outTensor->UpdateOffset(rootTensor->GetOffset());
    forRoots.push(outTensor);
    return SUCCESS;
}

Status ReplaceTensor::BackwardReshape(Operation *op, LogicalTensorPtr &rootTensor) {
    processedOp.insert(op->GetOpMagic());
    op->GetIOperands()[0]->tensor->actualRawmagic = rootTensor->GetRawMagic();
    backRoots.push(op->GetIOperands()[0]);
    return SUCCESS;
}

Status ReplaceTensor::BackwardInplaceOp(Operation *op, LogicalTensorPtr &rootTensor) {
    auto reusePairs = inplaceOpMap.at(op->GetOpcode());
    for (const auto &reusePair : reusePairs) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        auto tensorIn = op->GetIOperands()[inputIdx];
        auto tensorOut = op->GetOOperands()[outputIdx];
        if (tensorOut != rootTensor) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(), rootTensor->GetMagic());
            return SUCCESS;
        }
        processedOp.insert(op->GetOpMagic());
        tensorIn->tensor = tensorOut->tensor;
        backRoots.push(tensorIn);
        tensorOut->UpdateOffset(tensorIn->GetOffset());
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardView(Operation *op, LogicalTensorPtr &rootTensor) {
    auto viewIn = op->GetIOperands()[0];
    auto viewOut = op->GetOOperands()[0];
    (void) rootTensor;
    processedOp.insert(op->GetOpMagic());
    backRoots.push(viewIn);
    viewIn->tensor = viewOut->tensor;
    return SUCCESS;
}

Status ReplaceTensor::BackwardViewType(Operation *op, LogicalTensorPtr &rootTensor) {
    auto viewTypeIn = op->GetIOperands()[0];
    auto viewTypeOut = op->GetOOperands()[0];
    if (viewTypeOut != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_VIEW_TYPE %d rootTensor %d is not same as viewTypeOut %d.", op->GetOpMagic(), rootTensor->GetMagic(), viewTypeOut->GetMagic());
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    viewTypeIn->tensor->actualRawmagic = viewTypeOut->GetRawMagic();
    if (AdjustOffsetAndRawShape(viewTypeOut, viewTypeIn) == FAILED) {
        return FAILED;
    }
    backRoots.push(viewTypeIn);
    return SUCCESS;
}

Status ReplaceTensor::BackwardAssemble(Operation *op, LogicalTensorPtr &rootTensor) {
    auto &inOp = *(op->GetIOperands()[0])->GetProducers().begin();
    backRoots.push(op->GetIOperands()[0]);
    processedOp.insert(op->GetOpMagic());
    if (inOp != nullptr && inOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        APASS_LOG_INFO_F(Elements::Operation, "OP_ASSEMBLE %d parent op is OP_INDEX_OUTCAST %d, skip inplace.", op->GetOpMagic(), inOp->GetOpMagic());
        return SUCCESS;
    }
    if (BackUpdateAssemble(op) == FAILED) {
        return FAILED;
    }
    op->GetIOperands()[0]->tensor = rootTensor->tensor;

    if (op->GetIOperands()[0]->GetConsumers().size() > 1) {
        forRoots.push(op->GetIOperands()[0]);
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardProcess(Function &function) {
    while (!forRoots.empty()) {
        auto rootTensor = forRoots.front();
        forRoots.pop();
        for (auto &consumerOp : rootTensor->GetConsumers()) {
            if (processedOp.find(consumerOp->GetOpMagic()) != processedOp.end()) {
                continue;
            }
            if (consumerOp->GetOpcode() == Opcode::OP_VIEW) {
                if (ForwardView(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (ForwardAssemble(consumerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                if (ForwardReshape(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
                if (ForwardViewType(consumerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (inplaceOpMap.find(consumerOp->GetOpcode()) != inplaceOpMap.end()) {
                if (ForwardInplaceOp(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_COPY_OUT && consumerOp->HasAttribute(OpAttributeKey::inplaceIdx)) {
                if (ForwardCopyOut(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else {
                continue;
            }
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardProcess() {
    while (!backRoots.empty()) {
        auto rootTensor = backRoots.front();
        backRoots.pop();
        for (auto &producerOp : rootTensor->GetProducers()) {
            if (processedOp.find(producerOp->GetOpMagic()) != processedOp.end()) {
                continue;
            }
            if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (BackwardAssemble(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_VIEW) {
                if (BackwardView(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                if (BackwardReshape(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (inplaceOpMap.find(producerOp->GetOpcode()) != inplaceOpMap.end()) {
                if (BackwardInplaceOp(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
                if (BackwardViewType(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            }
            else {
                continue;
            }
        }
    }
    return SUCCESS;
}

LogicalTensorPtr ReplaceTensor::FindReplaceSource(Function &function, Operation &op, std::unordered_map<Operation *, LogicalTensorPtr> &visited) {
    if (visited.count(&op) > 0) {
        return visited.at(&op);
    }
    auto inplaceIdx = op.GetIntAttribute(OpAttributeKey::inplaceIdx);
    ASSERT(inplaceIdx >= 0 && inplaceIdx < static_cast<int>(op.GetIOperands().size()));
    auto inplaceIOperand = op.GetInputOperand(inplaceIdx);
    LogicalTensorPtr res = nullptr;
    for (auto producer : inplaceIOperand->GetProducers()) {
        if (!producer->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        auto tmp = FindReplaceSource(function, *producer, visited);
        if (res == nullptr) {
            res = tmp;
        } else {
            ASSERT(res == tmp); // inplace路径应总是交汇于同一起点
        }
    }
    if (res == nullptr) {
        res = inplaceIOperand; // 向前没有inplace了，自己就是起点
    }
    visited.emplace(&op, res);
    return res;
}

Status ReplaceTensor::RefactorViewConnectForReplace(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===> Start RefactorViewConnectForInplace.");
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (op.GetInputOperand(0)->GetRawTensor() == op.GetOutputOperand(0)->GetRawTensor()) {
            op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        }
    }
    std::unordered_map<Operation *, LogicalTensorPtr> visited;
    for (Operation &op : function.Operations()) {
        if (!op.HasAttribute(OpAttributeKey::inplaceIdx) || visited.count(&op) > 0) {
            continue;
        }
        FindReplaceSource(function, op, visited);
    }

    for (auto &[op, srcTensor] : visited) {
        if (op->GetOpcode() != Opcode::OP_VIEW) { // 仅重构View连接
            continue;
        }
        auto inplaceIdx = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
        ASSERT(inplaceIdx == 0);
        auto iOperand = op->GetInputOperand(inplaceIdx);
        auto oOperand = op->GetOutputOperand(0);
        if (iOperand == srcTensor) { // 开头的VIEW不需要插入NOP来控制顺序
            continue;
        }
        ASSERT(iOperand->GetRawTensor() == srcTensor->GetRawTensor());
        ASSERT(oOperand->GetRawTensor() == srcTensor->GetRawTensor());
        op->ReplaceIOperand(0, srcTensor);
        // 含inplace语义，都为同一个RawTensor
        auto nopOutput = std::make_shared<LogicalTensor>(function, srcTensor->GetRawTensor(),
            Offset(srcTensor->GetOffset().size()), srcTensor->GetShape(), NodeType::LOCAL);
        nopOutput->SetMemoryTypeBoth(oOperand->GetMemoryTypeOriginal());
        auto &nop = function.AddRawOperation(Opcode::OP_NOP, {iOperand, oOperand}, {nopOutput});
        nop.SetAttribute(OpAttributeKey::inplaceIdx, 0); // 期望上设成任何一个都可以，因为来源一致
        nop.UpdateSubgraphID(op->GetSubgraphID());
        auto consumers = oOperand->GetConsumers(); // deep copy
        for (auto consumer : consumers) {
            if (consumer->GetOpcode() == Opcode::OP_NOP || !consumer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                continue;
            }
            consumer->ReplaceIOperand(consumer->GetIntAttribute(OpAttributeKey::inplaceIdx), nopOutput);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End RefactorViewConnectForInplace.");
    return SUCCESS;
}

void ReplaceTensor::ProcessHubAssembleOp(Function &function, Operation &hubOp, Operation &assembleOp, 
                             std::shared_ptr<LogicalTensor> hubInput, std::shared_ptr<LogicalTensor> hubOutput) {
    auto assembleInput = assembleOp.GetIOperands()[0];
    auto assembleOutput = assembleOp.GetOOperands()[0];
    if (assembleInput.get() != hubOutput.get()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Assemble input[%d] is not HUB output[%d], chain may be broken",
                    assembleInput->GetMagic(), hubOutput->GetMagic());
        return;
    }
    bool isExactOutcast = false;
    auto outcasts = function.GetOutcast();
    for (auto &outcast : outcasts) {
        if (outcast.get() == assembleOutput.get()) {
            isExactOutcast = true;
            break;
        }
    }
    if (!isExactOutcast) {
        APASS_LOG_WARN_F(Elements::Operation, "Assemble[%d] output is not exact outcast, skip HUB memory reuse processing.", assembleOp.GetOpMagic());
        return;    
    }
    APASS_LOG_INFO_F(Elements::Operation, "Found exact HUB-ASSEMBLE-OUTCAST chain: HUB[%d] -> ASSEMBLE[%d] -> OUTCAST[%d]",
                hubOp.GetOpMagic(), assembleOp.GetOpMagic(), assembleOutput->GetMagic());
    auto hubInputMemType = hubInput->GetMemoryTypeOriginal();
    auto hubOutputMemType = hubOutput->GetMemoryTypeOriginal();
    auto assembleOutputMemType = assembleOutput->GetMemoryTypeOriginal();
    if (hubInputMemType != hubOutputMemType || hubInputMemType != assembleOutputMemType) {
        APASS_LOG_WARN_F(Elements::Tensor, "Memory type mismatch: HUB input=%d, HUB output=%d, ASSEMBLE output=%d",
                    hubInputMemType, hubOutputMemType, assembleOutputMemType);
        return;
    }
    hubInput->tensor = assembleOutput->tensor;
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(assembleOp.GetOpAttribute().get());
    hubInput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    hubOutput->tensor = assembleOutput->tensor;
    hubOutput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    APASS_LOG_INFO_F(Elements::Tensor, "Complete memory reuse established: all tensors share HUB input[%d] memory", hubInput->GetMagic());
}

Status ReplaceTensor::ProcessHubOp(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_HUB) {
            continue;
        }
        auto hubInput = op.GetIOperands()[0];   // HUB 的输入 tensor
        auto hubOutput = op.GetOOperands()[0];  // HUB 的输出 tensor
        for (auto consumerOp : hubOutput->GetConsumers()) {
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                ProcessHubAssembleOp(function, op, *consumerOp, hubInput, hubOutput);
            }
        }
        for (auto producerOp : hubInput->GetProducers()) {
            if (!OpcodeManager::Inst().IsCopyOut(producerOp->GetOpcode())) {
                continue;
            }
            auto copyAttr = dynamic_cast<CopyOpAttribute*>(producerOp->GetOpAttribute().get());
            if (copyAttr == nullptr) {
                APASS_LOG_INFO_F(Elements::Operation, "Copy Op %d Attribute is nullptr.", producerOp->GetOpMagic());
                continue;
            }
            auto attrOffset = copyAttr->GetToOffset(); // OpImm
            auto tensorOffset = OpImmediate::Specified(hubInput->GetTensorOffset());
            std::vector<OpImmediate> newOffset;
            for (size_t i = 0; i < attrOffset.size(); i++) {
                newOffset.push_back(attrOffset[i] + tensorOffset[i]);
            }
            copyAttr->SetToOffset(newOffset);
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===> Start ReplaceTensor.");
    LogicalTensors rec;
    for (const auto &op : function.Operations()) {
        for (const auto &inTensor : op.GetIOperands()) {
            rec.emplace_back(inTensor);
        }
        for (const auto &outTensor : op.GetOOperands()) {
            rec.emplace_back(outTensor);
        }
    }
    UnionFind uf(rec);
    UniteTensor(function, uf);
    std::vector<LogicalTensors> tensorGroups = uf.GetGroups();
    LogicalTensorPtr baseTensor;
    for (auto &group : tensorGroups) {
        baseTensor = nullptr;
        if (group.size() == 1) {
            continue;
        }
        if (FindBaseTensor(function, baseTensor, group) == FAILED || baseTensor == nullptr) {
            return FAILED;
        }
        backRoots.push(baseTensor);
        forRoots.push(baseTensor);
        while (!forRoots.empty() || !backRoots.empty()) {
            if (BackwardProcess() == FAILED || ForwardProcess(function) == FAILED) {
                return FAILED;
            }
        }
    }
    if (RefactorViewConnectForReplace(function) == FAILED) {
        return FAILED;
    }
    if (ProcessHubOp(function) == FAILED) {
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End ReplaceTensor.");
    return SUCCESS;
}

Status ReplaceTensor::AdjustOffsetAndRawShape(LogicalTensorPtr &fromView, LogicalTensorPtr &toView) const {
    auto fromType = fromView->tensor->datatype;
    auto toType = toView->tensor->datatype;
    auto inEntry = viewTypeTable.find(fromType);
    auto outEntry = viewTypeTable.find(toType);
    if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
        return FAILED;
    }
    int inSize = inEntry->second;
    int outSize = outEntry->second;
    int ratio = inSize > outSize ? inSize / outSize : outSize / inSize;
    bool isExpand = inSize > outSize;
    std::vector<int64_t> fromOffset = fromView->GetOffset();
    std::vector<int64_t> toOffset(fromOffset.size(), 0);
    std::vector<int64_t> inShape = fromView->GetRawTensor()->rawshape;
    std::vector<int64_t> outShape(inShape.size(), 0);
    for (size_t i = 0; i < fromOffset.size(); ++i) {
        if (i != fromOffset.size() - 1) {
            toOffset[i] = fromOffset[i];
            outShape[i] = inShape[i];
            continue;
        }
        if (isExpand) {
            toOffset[i] = fromOffset[i] * ratio;
            outShape[i] = inShape[i] * ratio;
        } else {
            if (fromOffset[i] % ratio != 0 || inShape[i] % ratio != 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "ViewType Offset is not Even.");
                return FAILED;
            }
            toOffset[i] = fromOffset[i] / ratio;
            outShape[i] = inShape[i] / ratio;
        }
    }
    toView->UpdateOffset(toOffset);
    toView->GetRawTensor()->rawshape = outShape;
    return SUCCESS;
}

Status ReplaceTensor::ForUpdateView(Operation *op) {
    auto viewAttr = dynamic_cast<ViewOpAttribute *>(op->GetOpAttribute().get());
    auto viewIn = op->GetIOperands()[0];
    auto viewOut = op->GetOOperands()[0];
    std::vector<int64_t> inputOffset = viewIn->GetOffset();
    if (viewAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ReplaceTensor::ForUpdateView: View op %d Attribute is nullptr.", op->GetOpMagic());
        return FAILED;
    }
    std::vector<int64_t> viewOpOffset = viewAttr->GetFrom();
    auto inputDynOffset = viewIn->GetDynOffset();
    if (inputDynOffset.empty()) {
        inputDynOffset = std::vector<SymbolicScalar>(inputOffset.size(), 0);
    }
    auto attrDynOffset = viewAttr->GetFromDynOffset();
    std::vector<SymbolicScalar> outTensorOffset;
    for (size_t i = 0; i < inputOffset.size(); i++) {
        viewOpOffset[i] = inputOffset[i] + viewOpOffset[i];
        if (attrDynOffset.size() == inputOffset.size()) {
            attrDynOffset[i] = inputDynOffset[i] + attrDynOffset[i];
        }
    }
    viewAttr->SetFromOffset(viewOpOffset, viewAttr->GetFromDynOffset());
    TensorOffset newOffset(viewOpOffset, attrDynOffset);
    viewOut->UpdateOffset(newOffset);
    return SUCCESS;
}

Status ReplaceTensor::BackUpdateAssemble(Operation *op) {
    auto assembleIn = op->GetIOperands()[0];
    auto assembleOut = op->GetOOperands()[0];
    auto assAttr = dynamic_cast<AssembleOpAttribute *>(op->GetOpAttribute().get());
    if (assAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ReplaceTensor::BackUpdateAssemble: Assemble op %d Attribute is nullptr.", op->GetOpMagic());
        return FAILED;
    }
    std::vector<int64_t> assOffset = assAttr->GetToOffset();
    std::vector<int64_t> outOffset = assembleOut->GetOffset();
    auto outDynOffset = assembleOut->GetDynOffset();
    auto assDynOffset = assAttr->GetToDynOffset();
    std::vector<SymbolicScalar> inTensorOffset;
    if (outDynOffset.empty()) {
        outDynOffset = std::vector<SymbolicScalar>(outOffset.size(), 0);
    }
    for (size_t i = 0; i < outOffset.size(); i++) {
        assOffset[i] = outOffset[i] + assOffset[i];
        if (assDynOffset.size() == outOffset.size()) {
            assDynOffset[i] = outDynOffset[i] + assDynOffset[i];
        }
    }
    assAttr->SetToOffset(assOffset, assAttr->GetToDynOffset());
    TensorOffset newOffset(assOffset, assDynOffset);
    assembleIn->UpdateOffset(newOffset);
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu