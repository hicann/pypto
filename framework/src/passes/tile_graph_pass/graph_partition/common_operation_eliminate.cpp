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
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "CommonOperationEliminate"

namespace npu::tile_fwk {
static std::unordered_map<uint64_t, std::pair<LogicalTensorPtr, std::vector<Operation*>>> hashCache;
void SortedProducer(std::vector<Operation*>& sortedProducers) {
    std::sort(sortedProducers.begin(), sortedProducers.end(),
        [](const Operation* op1, const Operation* op2) {
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
            for (const auto &attr : OpcodeManager::Inst().GetAttrs(op1->GetOpcode())) {
                ss1 << " attr: [" << attr << " : " << op1->DumpAttr(attr) << "]";
            }
            for (const auto &attr : OpcodeManager::Inst().GetAttrs(op2->GetOpcode())) {
                ss2 << " attr: [" << attr << " : " << op2->DumpAttr(attr) << "]";
            }
        return ss1.str() < ss2.str();
    });
}

unsigned long ComputeHash(const std::vector <Operation*>& producers, LogicalTensorPtr curTensor) {
    std::vector<std::string> opStrList;
    std::stringstream ss;
    std::vector<Operation*> sortedProducers = producers;
    SortedProducer(sortedProducers);
    for (const auto& op: sortedProducers) {
        if (op == nullptr) {
            continue;
        }
        ss.str(""), ss.clear();
        ss << op->GetOpcodeStr(true);
        for (const auto& iOperands: op->GetIOperands()) {
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
        for (const auto &attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        ss << "id" << op->GetSubgraphID();
        opStrList.emplace_back(ss.str());
    }
    ss.str(""), ss.clear();
    for (const auto& str: opStrList) {
        ss << str;
    }
    std::hash<std::string> hasher;
    return hasher(ss.str());
}

Status CommonOperationEliminate::RunOnFunction(Function &function) {
    auto tensorProducerMap = GetTensorProducers(function);
    std::unordered_set<Operation*> cacheProducers;
    for (auto& tensorProducerPair: tensorProducerMap) {
        auto& producerGroup = tensorProducerPair.second;
        if (producerGroup.empty() || !TensorProducersMerge(tensorProducerPair, cacheProducers)) {
            continue;
        }
        for (auto op: producerGroup) {
            if (op == nullptr) continue;
            if (!cacheProducers.count(op)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Operation[%d] was set as deleted.", op->GetOpMagic());
                op->SetAsDeleted();
            }
        }
    }
    function.EraseOperations();
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    hashCache.clear();
    return SUCCESS;
}

Status CommonOperationEliminate::PreCheck(Function &function) {
    CommonOperationEliminateChecker checker;
    return checker.DoPreCheck(function);
}

std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminate::GetTensorProducers(Function &function) {
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> tensorProducerMap;
    std::unordered_set<int> visitedTensors;
    auto allOps = function.Operations(true).DuplicatedOpList();
    for (const auto& op: allOps) {
        if (op == nullptr) {
            continue;
        }
        auto& outputTensors = op->GetOOperands();
        for (const auto& tensor: outputTensors) {
            if (tensor == nullptr || visitedTensors.count(tensor->GetMagic())) {
                continue;
            }
            visitedTensors.insert(tensor->GetMagic());
            for (const auto& pro: tensor->GetProducers()) {
                if (pro != nullptr) {
                    tensorProducerMap[tensor].push_back(pro);
                }
            }
        }
    }
    return tensorProducerMap;
}

std::pair<LogicalTensorPtr, std::vector<Operation*>>  CommonOperationEliminate::TensorHashExist(const std::pair<LogicalTensorPtr, std::vector<Operation*>>& tensorProducersPair, std::unordered_set<Operation*>& cacheProducers) {
    const std::vector<Operation*>& producers = tensorProducersPair.second;
    for (auto operation: producers) {
        if (operation == nullptr) {
            continue;
        }
        auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
        auto &outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
        OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
        bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
        bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
        bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
        if (inputCheck && calcTypeCheck && outputCheck) { // copy from L1 to L0
            return {nullptr, {}};
        }
        if (operation->GetOpcode() == Opcode::OP_VIEW) { //配合GraphPartition处理逻辑
            return {nullptr, {}};
        }
        if (operation->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            return {nullptr, {}};
        }
    }
    uint64_t groupHash = ComputeHash(producers, tensorProducersPair.first);
    if (hashCache.count(groupHash) != 0){
        APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] are marked as hash already existed tensor.", tensorProducersPair.first->GetMagic());
        return hashCache[groupHash];
    }
    hashCache.emplace(groupHash, tensorProducersPair);
    if (tensorProducersPair.first == nullptr) {
        return {nullptr, {}};
    }
    for (auto producer: tensorProducersPair.first->GetProducers()) {
        if (producer != nullptr) {
            cacheProducers.insert(producer);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] hash already existed.", tensorProducersPair.first->GetMagic());
    return {nullptr, {}};
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

void CommonOperationEliminate::UpdateConnection(LogicalTensorPtr oldtensor,  LogicalTensorPtr newtensor) {
    auto consumers = oldtensor->GetConsumers();
    for (auto &cur : consumers) {
        if (cur == nullptr) {
            continue;
        }
        cur->ReplaceInput(newtensor, oldtensor);
        auto attptr = cur->GetOpAttribute().get();
        if (attptr == nullptr) {
            continue;
        }
        if (cur->GetOpcode() == Opcode::OP_VIEW) {
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(attptr)) {
                UpdateView(viewOpAttribute, oldtensor, newtensor);
                continue;
            }
        } else if (cur->GetOpcode() == Opcode::OP_COPY_IN) { 
            if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(attptr)) {
                UpdateCopy(copyOpAttribute, oldtensor, newtensor);
                continue;
            }
        }
    }
}

bool CommonOperationEliminate::TensorProducersMerge(const std::pair<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerPair, std::unordered_set<Operation*>& cacheProducers) {
    auto& producers = tensorProducerPair.second;  
    if (producers.empty()) {
        return false;
    }
    auto existOp = TensorHashExist(tensorProducerPair, cacheProducers);
    if (existOp.first == nullptr || tensorProducerPair.first == nullptr || existOp.second.empty()) {
        return false;
    }
    if (tensorProducerPair.first->shape != existOp.first->shape) {
        return false;
    }
    if (tensorProducerPair.first->tensor->GetDataType() != existOp.first->tensor->GetDataType()) {
        return false;
    }
    LogicalTensorPtr oldtensor = tensorProducerPair.first;
    LogicalTensorPtr newtensor = existOp.first;
    if (oldtensor->nodetype == NodeType::OUTCAST) {
        return false;
    }
    if (tensorProducerPair.second.size() == existOp.second.size()) {
        bool allSame = true;
        for (size_t i = 0; i < existOp.second.size() && allSame; i++) {
            allSame = (tensorProducerPair.second[i] == existOp.second[i]);
        }
        if (allSame) {
            return false;
        }
    }
    if (newtensor->GetConsumers().size() == 0 || oldtensor->GetConsumers().size() == 0) {
        return false;
    }
    UpdateConnection(oldtensor, newtensor);
    oldtensor->GetConsumers().clear();
    APASS_LOG_DEBUG_F(Elements::Operation, "In CommonOperationEliminate, Tensor[%d] and producersgroup are marked as redundant.", oldtensor->GetMagic());
    return true;
}  
}// namespace npu::tile_fwk