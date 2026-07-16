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
 * \file process_atomic.cpp
 * \brief Process atomic operations including ReduceAcc and AtomicRMW
 */

#include "process_atomic.h"
#include "passes/pass_check/process_atomic_checker.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk_op.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_log/pass_log.h"
#include <set>

#define MODULE_NAME "ProcessAtomic"

namespace npu {
namespace tile_fwk {

Status ProcessAtomic::PreCheck(Function& function)
{
    ProcessAtomicChecker checker;
    return checker.DoPreCheck(function);
}

Status ProcessAtomic::PostCheck(Function& function)
{
    ProcessAtomicChecker checker;
    return checker.DoPostCheck(function);
}

Status ProcessAtomic::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start ProcessAtomic.");
    if (CheckAtomicRMWUnsupportedMode(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Unsupported AtomicRMW mode detected.");
        return FAILED;
    }
    bool hasReduceAccCascade = false;
    if (EliminateVecDupBranch(function, hasReduceAccCascade) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate VecDup branch failed.");
        return FAILED;
    }
    if (EliminateReduceAcc(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate ReduceAcc failed.");
        return FAILED;
    }
    if (EliminateAtomicRMW(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate AtomicRMW failed.");
        return FAILED;
    }
    if (hasReduceAccCascade) {
        Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
            return status;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End ProcessAtomic.");
    return SUCCESS;
}

Status ProcessAtomic::CheckAtomicRMWUnsupportedMode(Function& function)
{
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            int rmwModeValue = op.GetIntAttribute(OpAttributeKey::rmwMode);
            AtomicRMWMode rmwMode = static_cast<AtomicRMWMode>(rmwModeValue);
            if (rmwMode == AtomicRMWMode::MAX || rmwMode == AtomicRMWMode::MIN) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Op[%d] AtomicRMW mode '%s' is not supported yet. "
                    "Currently only ADD mode is supported. Please use ADD mode or wait for future support.%s",
                    op.GetOpMagic(), (rmwMode == AtomicRMWMode::MAX ? "MAX" : "MIN"), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::EliminateReduceAcc(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            APASS_LOG_INFO_F(Elements::Operation, "ATOMIC_ADD, opmagic: %d", op.GetOpMagic());
            auto reduceOut = op.GetOOperands().front();
            reduceOut->GetProducers().clear();

            for (const auto& input : op.GetIOperands()) {
                auto producersBackup = input->GetProducers();
                for (auto& produceCopyOutOp : producersBackup) {
                    produceCopyOutOp->ReplaceOOperand(0, reduceOut);
                    produceCopyOutOp->SetAttribute(RMW_MODE_ATTR_ADD, 1);
                }
            }
            op.SetAsDeleted();
            APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic());
        }
    }
    function.EraseOperations(true);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Eliminate dead operation failed for ReduceAcc in CommonOperationEliminate.");
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::EliminateAtomicRMW(Function& function)
{
    std::vector<Operation*> atomicRmwOps;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwOps.emplace_back(&op);
        }
    }
    if (PrepareAtomicRMWSharedInputs(function, atomicRmwOps) != SUCCESS) {
        return FAILED;
    }
    for (auto* op : atomicRmwOps) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        if (ProcessSingleAtomicRMW(*op) != SUCCESS) {
            return FAILED;
        }
    }
    function.EraseOperations(true);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Eliminate dead operation failed for AtomicRMW in CommonOperationEliminate.");
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::ProcessSingleAtomicRMW(Operation& op)
{
    APASS_LOG_INFO_F(Elements::Operation, "ATOMIC_RMW, opmagic: %d", op.GetOpMagic());

    auto rmwOut = op.GetOOperands().front();
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    if (assembleAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] missing AssembleOpAttribute; Cannot eliminate.",
                          op.GetOpMagic());
        return FAILED;
    }

    auto& rmwOffset = assembleAttr->GetToOffset();
    auto& rmwDynOffset = assembleAttr->GetToDynOffset();

    int rmwModeValue = op.GetIntAttribute(OpAttributeKey::rmwMode);
    AtomicRMWMode rmwMode = static_cast<AtomicRMWMode>(rmwModeValue);
    std::string rmwAttrKey = GetRmwAttrKey(rmwMode);
    if (rmwAttrKey.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] has invalid rmwMode value %d.", op.GetOpMagic(), rmwModeValue);
        return FAILED;
    }

    for (const auto& input : op.GetIOperands()) {
        auto producersBackup = input->GetProducers();
        for (auto& producerOp : producersBackup) {
            if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE || producerOp->GetOpcode() == Opcode::OP_ASSEMBLE_SSA) {
                if (ProcessAssembleProducer(*producerOp, rmwOut, rmwMode, rmwOffset, rmwDynOffset) != SUCCESS) {
                    return FAILED;
                }
            }
        }
    }

    op.SetAsDeleted();
    APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
    return SUCCESS;
}

bool ProcessAtomic::HasAssembleProducer(const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* producerOp : input->GetProducers()) {
        if (producerOp == nullptr) {
            continue;
        }
        if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE || producerOp->GetOpcode() == Opcode::OP_ASSEMBLE_SSA) {
            return true;
        }
    }
    return false;
}

bool ProcessAtomic::HasConsumerExcept(const std::shared_ptr<LogicalTensor>& input, const Operation& op) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* consumerOp : input->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->GetOpMagic() != op.GetOpMagic()) {
            return true;
        }
    }
    return false;
}

Status ProcessAtomic::PrepareAtomicRMWSharedInputs(Function& function,
                                                   const std::vector<Operation*>& atomicRmwOps) const
{
    for (auto* op : atomicRmwOps) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        auto inputsBackup = op->GetIOperands();
        for (const auto& input : inputsBackup) {
            if (PrepareExclusiveAtomicInput(function, *op, input) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Prepare shared input failed for AtomicRMW op[%d].",
                                  op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

std::shared_ptr<LogicalTensor> ProcessAtomic::PrepareExclusiveAtomicInput(
    Function& function, Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr || !HasConsumerExcept(input, atomicOp) || !HasAssembleProducer(input)) {
        return input;
    }

    auto producersBackup = input->GetProducers();
    auto clonedInput = input->Clone(function, true);
    if (clonedInput == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Clone atomic input tensor[%d] failed.", input->GetMagic());
        return nullptr;
    }
    atomicOp.ReplaceInput(clonedInput, input);

    for (auto* producerOp : producersBackup) {
        if (producerOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Null producer detected for atomic input tensor[%d].",
                              input->GetMagic());
            return nullptr;
        }
        if (producerOp->GetOpcode() != Opcode::OP_ASSEMBLE && producerOp->GetOpcode() != Opcode::OP_ASSEMBLE_SSA) {
            continue;
        }
        auto& clonedProducer = producerOp->CloneOperation(function, producerOp->GetIOperands(),
                                                          producerOp->GetOOperands());
        clonedProducer.UpdateSubgraphID(producerOp->GetSubgraphID());
        clonedProducer.SetCoreType(producerOp->GetCoreType());
        clonedProducer.SetSpan(producerOp->GetSpan());
        clonedProducer.SetScopeInfo(producerOp->GetScopeInfo());
        if (producerOp->GetOpAttribute() != nullptr) {
            clonedProducer.SetOpAttribute(producerOp->GetOpAttribute()->Clone());
        }
        clonedProducer.ReplaceOutput(clonedInput, input);
    }
    return clonedInput;
}

std::string ProcessAtomic::GetRmwAttrKey(AtomicRMWMode mode)
{
    switch (mode) {
        case AtomicRMWMode::ADD:
            return RMW_MODE_ATTR_ADD;
        case AtomicRMWMode::MAX:
            return RMW_MODE_ATTR_MAX;
        case AtomicRMWMode::MIN:
            return RMW_MODE_ATTR_MIN;
        default:
            return "";
    }
}

Status ProcessAtomic::ProcessAssembleProducer(Operation& producerOp, std::shared_ptr<LogicalTensor> rmwOut,
                                              AtomicRMWMode rmwMode, const std::vector<int64_t>& rmwOffset,
                                              const std::vector<SymbolicScalar>& rmwDynOffset)
{
    std::string rmwAttrKey = GetRmwAttrKey(rmwMode);
    if (CheckAndSetRmwAttr(producerOp, rmwMode, rmwAttrKey) != SUCCESS) {
        return FAILED;
    }

    producerOp.ReplaceOOperand(0, rmwOut);

    auto producerAssembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp.GetOpAttribute());
    if (producerAssembleAttr != nullptr) {
        AccumulateAssembleOffset(producerAssembleAttr, rmwOffset, rmwDynOffset);
    }
    return SUCCESS;
}

Status ProcessAtomic::CheckAndSetRmwAttr(Operation& producerOp, AtomicRMWMode rmwMode, const std::string& rmwAttrKey)
{
    bool hasAdd = producerOp.HasAttr(RMW_MODE_ATTR_ADD);
    bool hasMax = producerOp.HasAttr(RMW_MODE_ATTR_MAX);
    bool hasMin = producerOp.HasAttr(RMW_MODE_ATTR_MIN);
    if (!hasAdd && !hasMax && !hasMin) {
        producerOp.SetAttribute(rmwAttrKey, 1L);
        return SUCCESS;
    }

    bool attrConflict = (rmwMode == AtomicRMWMode::ADD && !hasAdd) || (rmwMode == AtomicRMWMode::MAX && !hasMax) ||
                        (rmwMode == AtomicRMWMode::MIN && !hasMin);

    if (attrConflict) {
        std::string existingAttrType;
        if (hasAdd)
            existingAttrType = "atomic_add";
        else if (hasMax)
            existingAttrType = "atomic_max";
        else if (hasMin)
            existingAttrType = "atomic_min";

        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] rmwMode conflict: producer assemble op already has '%s' attribute, "
                          "but current wants to set '%s'. Cannot set different rmwMode to the same assemble op.",
                          producerOp.GetOpMagic(), existingAttrType.c_str(), rmwAttrKey.c_str());
        return FAILED;
    }
    return SUCCESS;
}

void ProcessAtomic::AccumulateAssembleOffset(std::shared_ptr<AssembleOpAttribute> producerAttr,
                                             const std::vector<int64_t>& rmwOffset,
                                             const std::vector<SymbolicScalar>& rmwDynOffset)
{
    auto& producerOffset = producerAttr->GetToOffset();
    auto& producerDynOffset = producerAttr->GetToDynOffset();

    if (producerOffset.size() == rmwOffset.size()) {
        for (size_t i = 0; i < producerOffset.size(); ++i) {
            producerOffset[i] += rmwOffset[i];
        }
    }
    if (producerDynOffset.empty()) {
        for (size_t i = 0; i < rmwDynOffset.size(); ++i) {
            producerDynOffset.push_back(producerOffset[i] + rmwDynOffset[i]);
        }
    } else {
        for (size_t i = 0; i < producerDynOffset.size(); ++i) {
            producerDynOffset[i] = producerDynOffset[i] + rmwDynOffset[i];
        }
    }
}

void ProcessAtomic::CollectReduceAccUpstream(Operation& op, std::set<int>& visited,
                                             std::vector<Operation*>& result) const
{
    if (visited.count(op.GetOpMagic()) > 0 || op.IsDeleted()) {
        return;
    }
    visited.insert(op.GetOpMagic());
    if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
        result.push_back(&op);
        return;
    }
    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted() || producer->GetOpcode() == Opcode::OP_ATOMIC_RMW) {
                continue;
            }
            CollectReduceAccUpstream(*producer, visited, result);
        }
    }
}

Status ProcessAtomic::TraceBackAndRemoveVecDup(Function& function, Operation& op, std::set<int>& visited,
                                               bool& anyRemoved)
{
    if (visited.count(op.GetOpMagic()) > 0 || op.IsDeleted()) {
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        return SUCCESS;
    }
    visited.insert(op.GetOpMagic());

    if (op.GetOpcode() == Opcode::OP_A_MUL_B || op.GetOpcode() == Opcode::OP_A_MULACC_B) {
        if (RemoveVecDupBranchFromCubeOp(op, anyRemoved) != SUCCESS) {
            return FAILED;
        }
    }

    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted()) {
                continue;
            }
            if (TraceBackAndRemoveVecDup(function, *producer, visited, anyRemoved) != SUCCESS) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::RemoveVecDupBranchFromCubeOp(Operation& cubeOp, bool& anyRemoved)
{
    auto inputsBackup = cubeOp.GetIOperands();
    for (const auto& input : inputsBackup) {
        auto producersBackup = input->GetProducers();
        for (auto* producer : producersBackup) {
            if (producer == nullptr || producer->IsDeleted()) {
                continue;
            }
            if ((producer->GetOpcode() == Opcode::OP_ASSEMBLE || producer->GetOpcode() == Opcode::OP_ASSEMBLE_SSA) &&
                IsVecDupAssembleInput(*producer)) {
                input->RemoveConsumer(&cubeOp);
                cubeOp.EraseInput(input);
                anyRemoved = true;
                break;
            }
        }
    }
    return SUCCESS;
}

bool ProcessAtomic::IsVecDupAssembleInput(const Operation& assembleOp) const
{
    for (const auto& input : assembleOp.GetIOperands()) {
        for (auto* producer : input->GetProducers()) {
            if (producer != nullptr && producer->GetOpcode() == Opcode::OP_VEC_DUP) {
                return true;
            }
        }
    }
    return false;
}

Status ProcessAtomic::EliminateVecDupBranch(Function& function, bool& hasReduceAccCascade)
{
    std::vector<Operation*> reduceAccOps;
    std::set<int> collectVisited;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            CollectReduceAccUpstream(op, collectVisited, reduceAccOps);
        }
    }
    hasReduceAccCascade = !reduceAccOps.empty();
    if (reduceAccOps.empty()) {
        return SUCCESS;
    }

    bool anyRemoved = false;
    std::set<int> traceVisited;
    for (auto* reduceAccOp : reduceAccOps) {
        if (TraceBackAndRemoveVecDup(function, *reduceAccOp, traceVisited, anyRemoved) != SUCCESS) {
            return FAILED;
        }
    }

    if (!anyRemoved) {
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Function, "EliminateVecDupBranch removed VecDup assemble input branch.");
    function.EraseOperations(true);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate dead operation failed for VecDup branch.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
