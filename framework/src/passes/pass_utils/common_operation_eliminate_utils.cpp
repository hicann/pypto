/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_operation_eliminate_utils.cpp
 * \brief utils of common operation elimination
 */

#include "common_operation_eliminate_utils.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "CommonOperationEliminateUtils"

namespace npu::tile_fwk {
void CommonOperationEliminateUtils::SortedProducer(std::vector<Operation*>& sortedProducers) const
{
    std::sort(sortedProducers.begin(), sortedProducers.end(), [](const Operation* op1, const Operation* op2) {
        const auto& iOp1 = op1->GetIOperands();
        const auto& iOp2 = op2->GetIOperands();
        size_t minLen = std::min(iOp1.size(), iOp2.size());
        for (size_t i = 0; i < minLen; ++i) {
            LogicalTensorPtr ptr1 = iOp1[i];
            LogicalTensorPtr ptr2 = iOp2[i];
            if (ptr1 != ptr2) {
                return ptr1 < ptr2;
            }
        }
        if (iOp1.size() != iOp2.size()) {
            return iOp1.size() < iOp2.size();
        }
        std::stringstream ss1, ss2;
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op1->GetOpcode())) {
            ss1 << " attr: [" << attr << " : " << op1->DumpAttr(attr) << "]";
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op2->GetOpcode())) {
            ss2 << " attr: [" << attr << " : " << op2->DumpAttr(attr) << "]";
        }
        return ss1.str() < ss2.str();
    });
}

void CommonOperationEliminateUtils::CollectProducerInfo(
    const std::vector<Operation*>& sortedProducers, const LogicalTensorPtr& curTensor,
    std::vector<std::string>& opStrList, std::stringstream& ss) const
{
    for (const auto& op : sortedProducers) {
        if (op == nullptr) {
            continue;
        }
        ss.str("");
        ss.clear();
        ss << op->GetOpcodeStr(true);
        for (const auto& iOperands : op->GetIOperands()) {
            if (iOperands == nullptr || iOperands->tensor == nullptr) {
                continue;
            }
            ss << "[i";
            ss << "$" << iOperands->tensor->DumpSSA(false, false);
            ss << iOperands->DumpType();
            ss << "(";
            for (size_t i = 0; i < iOperands->offset.size(); ++i) {
                ss << iOperands->offset[i];
                if (i != iOperands->offset.size() - 1) {
                    ss << ", ";
                }
            }
            if (curTensor && !curTensor->GetDynValidShape().empty()) {
                std::string shapeStr;
                for (size_t i = 0; i < curTensor->GetDynValidShape().size(); i++) {
                    shapeStr += curTensor->GetDynValidShape()[i].Dump();
                }
                ss << "[" << shapeStr << "]";
            }
            ss << ")";
            ss << "]";
        }
        if (op->GetOpAttribute() != nullptr) {
            ss << " " << op->GetOpAttribute()->Dump();
        }
        if (!op->DumpAttr().empty()) {
            ss << " " << op->DumpAttr();
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        ss << "id" << op->GetSubgraphID();
        opStrList.emplace_back(ss.str());
    }
    ss.str("");
    ss.clear();
    for (const auto& str : opStrList) {
        ss << str;
    }
}

unsigned long CommonOperationEliminateUtils::ComputeHash(
    const std::vector<Operation*>& producers, LogicalTensorPtr curTensor) const
{
    std::vector<std::string> opStrList;
    std::stringstream ss;
    std::vector<Operation*> sortedProducers = producers;
    SortedProducer(sortedProducers);
    CollectProducerInfo(sortedProducers, curTensor, opStrList, ss);
    std::hash<std::string> hasher;
    return hasher(ss.str());
}

Status CommonOperationEliminateUtils::EliminateCommonOperation(Function& function)
{
    CommonOperationEliminateUtils commonOperationEliminateUtils;
    return commonOperationEliminateUtils.Process(function);
}

Status CommonOperationEliminateUtils::Process(Function& function)
{
    std::vector<LogicalTensorPtr> sequence;
    auto tensorProducerMap = GetTensorProducers(function, sequence);
    std::unordered_set<Operation*> cacheProducers;
    for (auto& orderedTensor : sequence) {
        auto& producerGroup = tensorProducerMap[orderedTensor];
        if (producerGroup.empty() ||
            !TensorProducersMerge(function, orderedTensor, cacheProducers, tensorProducerMap)) {
            continue;
        }
        for (auto op : producerGroup) {
            if (op == nullptr) {
                continue;
            }
            if (!cacheProducers.count(op)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Operation[%d] was set as deleted.", op->GetOpMagic());
                op->SetAsDeleted();
            }
        }
    }
    function.EraseOperations();
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Eliminate dead operation failed in CommonOperationEliminateUtils.");
        return FAILED;
    }
    return SUCCESS;
}

std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminateUtils::GetTensorProducers(
    Function& function, std::vector<LogicalTensorPtr>& sequence)
{
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> tensorProducerMap;
    std::unordered_set<int> visitedTensors;
    auto allOps = function.Operations(true).DuplicatedOpList();
    for (const auto& op : allOps) {
        if (op == nullptr) {
            continue;
        }
        auto& outputTensors = op->GetOOperands();
        for (const auto& tensor : outputTensors) {
            if (tensor == nullptr || visitedTensors.count(tensor->GetMagic())) {
                continue;
            }
            visitedTensors.insert(tensor->GetMagic());
            for (const auto& producer : tensor->GetProducers()) {
                if (producer == nullptr) {
                    APASS_LOG_ERROR_F(
                        Elements::Operation, "Producer operation nullptr for Tensor[%d].", tensor->GetMagic());
                    continue;
                }
                if (tensorProducerMap.count(tensor) == 0) {
                    sequence.push_back(tensor);
                }
                tensorProducerMap[tensor].push_back(producer);
            }
        }
    }
    return tensorProducerMap;
}

std::pair<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminateUtils::TensorHashExist(
    const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
    const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap)
{
    const std::vector<Operation*>& producers = tensorProducerMap.find(orderedTensor)->second;
    for (auto operation : producers) {
        if (operation == nullptr) {
            continue;
        }
        auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
        auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
        OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
        bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
        bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
        bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
        if (inputCheck && calcTypeCheck && outputCheck) { // copy from L1 to L0
            return {nullptr, {}};
        }
        if (operation->GetOpcode() == Opcode::OP_VIEW) { // work with GraphPartition processing logic
            return {nullptr, {}};
        }
        if (operation->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            return {nullptr, {}};
        }
    }
    uint64_t groupHash = ComputeHash(producers, orderedTensor);
    if (hashCache_.count(groupHash) != 0) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Tensor[%d] are marked as hash already existed tensor.", orderedTensor->GetMagic());
        return hashCache_[groupHash];
    }
    hashCache_.emplace(groupHash, std::make_pair(orderedTensor, producers));
    if (orderedTensor == nullptr) {
        return {nullptr, {}};
    }
    for (auto producer : orderedTensor->GetProducers()) {
        if (producer != nullptr) {
            cacheProducers.insert(producer);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] hash already existed.", orderedTensor->GetMagic());
    return {nullptr, {}};
}

void CommonOperationEliminateUtils::UpdateView(
    ViewOpAttribute* viewOpAttribute, const std::shared_ptr<LogicalTensor> oldTensor,
    const std::shared_ptr<LogicalTensor> newTensor) const
{
    auto& fromOffset = viewOpAttribute->GetFromOffset();
    for (size_t j = 0; j < fromOffset.size(); j++) {
        fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
    }
}

void CommonOperationEliminateUtils::UpdateCopy(
    CopyOpAttribute* copyOpAttribute, const std::shared_ptr<LogicalTensor> oldTensor,
    const std::shared_ptr<LogicalTensor> newTensor) const
{
    if (!copyOpAttribute->IsCopyOut()) {
        auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
        (void)memType;
        for (size_t j = 0; j < fromOffset.size(); j++) {
            fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
        }
        copyOpAttribute->SetFromOffset(fromOffset);
    }
}

void CommonOperationEliminateUtils::UpdateConnection(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor)
{
    auto consumers = oldTensor->GetConsumers();
    for (auto& cur : consumers) {
        if (cur == nullptr) {
            continue;
        }
        cur->ReplaceInput(newTensor, oldTensor);
        auto attrPtr = cur->GetOpAttribute().get();
        if (attrPtr == nullptr) {
            continue;
        }
        if (cur->GetOpcode() == Opcode::OP_VIEW) {
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(attrPtr)) {
                UpdateView(viewOpAttribute, oldTensor, newTensor);
                continue;
            }
        } else if (cur->GetOpcode() == Opcode::OP_COPY_IN) {
            if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(attrPtr)) {
                UpdateCopy(copyOpAttribute, oldTensor, newTensor);
                continue;
            }
        }
    }
}

bool CommonOperationEliminateUtils::TensorProducersMerge(
    Function& function, const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
    const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap)
{
    auto& producers = tensorProducerMap.at(orderedTensor);
    if (producers.empty()) {
        return false;
    }
    auto existOp = TensorHashExist(orderedTensor, cacheProducers, tensorProducerMap);
    if (existOp.first == nullptr || orderedTensor == nullptr || existOp.second.empty()) {
        return false;
    }
    if (orderedTensor->shape != existOp.first->shape) {
        return false;
    }
    if (orderedTensor->tensor->GetDataType() != existOp.first->tensor->GetDataType()) {
        return false;
    }
    LogicalTensorPtr oldTensor = orderedTensor;
    LogicalTensorPtr newTensor = existOp.first;
    if (FunctionUtils::GetNodeType(*oldTensor, function) == NodeType::OUTCAST) {
        return false;
    }
    if (producers.size() == existOp.second.size()) {
        bool allSame = true;
        for (size_t i = 0; i < existOp.second.size() && allSame; i++) {
            allSame = (producers[i] == existOp.second[i]);
        }
        if (allSame) {
            return false;
        }
    }
    if (newTensor->GetConsumers().size() == 0 || oldTensor->GetConsumers().size() == 0) {
        return false;
    }
    UpdateConnection(oldTensor, newTensor);
    oldTensor->GetConsumers().clear();
    APASS_LOG_DEBUG_F(
        Elements::Operation, "In CommonOperationEliminateUtils, Tensor[%d] and producersgroup are marked as redundant.",
        oldTensor->GetMagic());
    return true;
}
} // namespace npu::tile_fwk
