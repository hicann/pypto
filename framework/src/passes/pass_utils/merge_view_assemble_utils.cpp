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
 * \file merge_view_assemble_utils.cpp
 * \brief utils of view and assemble operation merging
 */

#include "merge_view_assemble_utils.h"
#include "interface/operation/attribute.h"
#include "passes/pass_utils/dead_operation_eliminate.h"

namespace npu::tile_fwk {

Status MergeViewAssembleUtils::MergeViewAssemble(Function &function) {
    MergeViewAssembleUtils MergeViewAssembleUtils;
    Status status = MergeViewAssembleUtils.Process(function);
    return status;
}

Status MergeViewAssembleUtils::Process(Function &function) {
    Status status = Initialize();
    if (status != SUCCESS) {   
        ALOG_ERROR("MergeViewAssembleUtils initialization failed.");
        return status;
    }
    status = ProcessOperations(function);
    if (status != SUCCESS) {   
        ALOG_ERROR("Processing operations failed.");
        return status;
    }
    status = CleanUp(function);
    if (status != SUCCESS) {   
        ALOG_ERROR("Cleanup phase failed.");
        return status;
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::Initialize() {
    visitedOp_.clear();
    viewOpToAppend_.clear();
    assembleOpToAppend_.clear();
    return SUCCESS;
}

Status MergeViewAssembleUtils::ProcessOperations(Function &function) {
    for (auto &op : function.Operations()) {
        if (visitedOp_.count(op.GetOpMagic()) != 0) {
            continue;
        }
        Status processStatus = SUCCESS;
        std::vector<Operation *> chain;
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            processStatus = MergeViewChain(function, op, chain);
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            processStatus = MergeAssembleChain(function, op, chain);
        }
        if (processStatus != SUCCESS) {
            ALOG_ERROR("ProcessOperations failed.");
            return processStatus;
        }
    }
    Status status = AppendMergedViewOperations(function);
    if (status != SUCCESS) {
        ALOG_ERROR("AppendMergedViewOperations phase failed.");
        return status;
    }
    status = AppendMergedAssembleOperations(function);
    if (status != SUCCESS) {
        ALOG_ERROR("AppendMergedAssembleOperations phase failed.");
        return FAILED;
    }
    return status;
}

Status MergeViewAssembleUtils::AppendMergedViewOperations(Function &function) {
    /* Process View ops first to avoid View output being cleared in View-Assemble scenarios */
    for (auto &viewOp : viewOpToAppend_) {
        auto attr = std::make_shared<ViewOpAttribute>(viewOp.offset, viewOp.toType, viewOp.dynOffset,
                     viewOp.dynValidShape);
        if (!attr) {
            ALOG_ERROR("Failed to create ViewOpAttribute.");
            return FAILED;
        }
        auto &mergedViewOp = function.AddRawOperation(Opcode::OP_VIEW, {viewOp.input}, {viewOp.output});
        mergedViewOp.SetOpAttribute(attr);
        // 继承op_attr_copy_in_mode属性
        if (viewOp.hasCopyInMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_mode", viewOp.copyInModeValue); 
        }   
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);    
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::AppendMergedAssembleOperations(Function &function) {
    for (const auto &assembleOp : assembleOpToAppend_) {
        auto attr = std::make_shared<AssembleOpAttribute>(assembleOp.offset, assembleOp.dynOffset);
        if (!attr) {
            return FAILED;
        }
        auto &mergedAssembleOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {assembleOp.input}, {assembleOp.output});
        mergedAssembleOp.SetOpAttribute(attr);
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::CleanUp(Function &function) {
    Status status = EraseRedundantAssemble(function);
    if (status != SUCCESS)
    {   
        ALOG_ERROR("EraseRedundantAssemble failed.");
        return status;
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false);
    return SUCCESS;
}

Status MergeViewAssembleUtils::MergeViewChain(
    Function &function, Operation &operation, std::vector<Operation *> &chain) {
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    // 1. 初始化操作链
    InitOperationChain(operation, chain);

    // 2. 处理消费者链
    auto consumers = function.FindConsumers(operation);
    bool chainEnd = true;
    Status status = ProcessConsumerChain(function, consumers, chain, chainEnd);
    if (status != SUCCESS) {
        return status;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        return ProcessChainEnd(function, chain);
    }

    return SUCCESS;
}

void MergeViewAssembleUtils::InitOperationChain(Operation &operation, std::vector<Operation *> &chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeViewAssembleUtils::ProcessConsumerChain(
    Function &function,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
    std::vector<Operation *> &chain,
    bool &chainEnd)
{
    if (consumers.empty()) {
        return SUCCESS;
    }
    Operation *currentOp = chain.back();
    auto currentViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(currentOp->GetOpAttribute());
    if (!currentViewAttr) {
        ALOG_ERROR("Failed to get current view attribute.");
        return FAILED;
    }
    MemoryType currentMemType = currentViewAttr->GetTo();
    for (auto &op : consumers) {
        if (!op) {
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
            if (viewOpAttribute == nullptr) {
                ALOG_ERROR("View operation has null viewOpAttribute.");
                return FAILED;
            }
            auto memoryTo = viewOpAttribute->GetTo();
            // 根据新的合并原则判断是否可以合并
            bool canMerge = false;
            if (currentMemType == MemoryType::MEM_UNKNOWN || currentMemType == memoryTo) {
                // 1.unknown memType 可以向它之后的view合并 2.相同memType的view可以合并
                canMerge = true;
            } 
            if (canMerge) {
                chainEnd = false;
                Status status = MergeViewChain(function, *op, chain);
                if (status != SUCCESS) {
                    return status;
                }
                chain.pop_back();
            } else {
                chainEnd = true;
            }
        }
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::ProcessChainEnd(
    Function &function,
    std::vector<Operation *> &chain)
{
    // 1. 验证链的有效性
    Operation *startOp = chain.front();
    Operation *endOp = chain.back();
    if (startOp->iOperand.empty()) {
        ALOG_ERROR("First operation in chain has no input operands.");
        return FAILED;
    }
    if (endOp->oOperand.empty()) {
        ALOG_ERROR("Last operation in chain has no output operands.");
        return FAILED;
    }
    auto &startTensor = startOp->iOperand.front();
    auto &endTensor = endOp->oOperand.front();
    if (!startTensor) {
        ALOG_ERROR("Null input tensor found for first operation in chain.");
        return FAILED;
    }
    if (!endTensor) {
        ALOG_ERROR("Null output tensor found for last operation in chain.");
        return FAILED;
    }
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<SymbolicScalar> newDynValidShape;
    Status status = CalculateMergedOffsets(chain, newOffset, newDynOffset, newDynValidShape);
    if (status != SUCCESS) {
        return status;
    }
    // 记录合并操作
    RecordMergedViewOperation(endOp, startTensor, endTensor, newOffset, newDynOffset, newDynValidShape);

    // 清理链尾
    endOp->oOperand.clear();
    function.GetTensorMap().Erase(endTensor);
    return SUCCESS;
}

Status MergeViewAssembleUtils::CalculateMergedOffsets(const std::vector<Operation *> &chain, std::vector<int64_t> &newOffset,
    std::vector<SymbolicScalar> &newDynOffset, std::vector<SymbolicScalar> &newDynValidShape) {
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto &view = chain[i];
        if (!view) {
            ALOG_ERROR("Null view operation in chain.");
            return FAILED;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
        if (!viewOpAttribute) {
            ALOG_ERROR("Failed to get ViewOpAttribute.");
            return FAILED;
        }
        if (i == 0) {
            newOffset = viewOpAttribute->GetFromOffset();
            newDynOffset = viewOpAttribute->GetFromDynOffset();
            if (!viewOpAttribute->GetToDynValidShape().empty()) {
                newDynValidShape = viewOpAttribute->GetToDynValidShape();
            }
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, viewOpAttribute->GetFromOffset(), viewOpAttribute->GetFromDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
        if (!viewOpAttribute->GetToDynValidShape().empty()) {
            newDynValidShape = viewOpAttribute->GetToDynValidShape();
            continue;
        }
        newDynValidShape = GetViewValidShape(newDynValidShape, viewOpAttribute->GetFromOffset(),
            viewOpAttribute->GetFromDynOffset(), view->GetOOperands()[0]->GetShape());
    }
    return SUCCESS;
}

void MergeViewAssembleUtils::RecordMergedViewOperation(Operation* lastViewOp, const std::shared_ptr<LogicalTensor> &startTensor,
    const std::shared_ptr<LogicalTensor> &endTensor, const std::vector<int64_t> &newOffset,
    const std::vector<SymbolicScalar> &newDynOffset, const std::vector<SymbolicScalar> &newDynValidShape) {
    // 获取最后一个VIEW的属性
    auto lastViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(lastViewOp->GetOpAttribute());
    if (!lastViewAttr) {
        return;
    }
    // 获取特定的 op_attr_copy_in_mode 属性
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    // 清理消费者关系
    endTensor->GetProducers().clear();
    // 记录合并op
    viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue)});
}

Status MergeViewAssembleUtils::MergeAssembleChain(Function &function, Operation &operation, std::vector<Operation *> &chain) {
    // 1. 初始化操作链
    InitAssembleChain(operation, chain);

    // 2. 处理消费者
    bool chainEnd = false;
    bool hasAssembleConsumer = false;
    if (assembleWithoutAssembleConsumer_.count(operation.opmagic) == 0) {
        auto consumers = function.FindConsumers(operation);
        chainEnd = consumers.empty();
        Status status = ProcessAssembleConsumers(function, consumers, chain, chainEnd, hasAssembleConsumer);
        if (status != SUCCESS) {
            return status;
        }
        if(!hasAssembleConsumer){
            assembleWithoutAssembleConsumer_.insert(operation.opmagic);
        }
    } else {
        chainEnd = true;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        Status status = ProcessAssembleChainEnd(function, chain, operation);
        if (status != SUCCESS) {
            return status;
        }
    }

    chain.pop_back();
    return SUCCESS;
}

void MergeViewAssembleUtils::InitAssembleChain(
    Operation &operation,
    std::vector<Operation *> &chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeViewAssembleUtils::ProcessAssembleConsumers(
    Function &function,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
    std::vector<Operation *> &chain,
    bool &chainEnd, bool &hasAssembleConsumer)
{
    if (consumers.empty()) {
        return SUCCESS;
    }
    for (auto &op : consumers) {
        if (!op) {
            ALOG_ERROR("Null consumer operation found.");
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            hasAssembleConsumer = true;
            Status status = MergeAssembleChain(function, *op, chain);
            if (status != SUCCESS) { 
                ALOG_ERROR("Run MergeAssembleChain failed.");
                return status;
            }
            continue;
        }
        chainEnd = true;
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::ProcessAssembleChainEnd(
    Function &function,
    std::vector<Operation *> &chain,
    Operation &operation)
{
    // 验证链有效性
    if (chain.front()->iOperand.empty() || chain.back()->oOperand.empty()) {
        ALOG_ERROR("Invalid chain operations.");
        return FAILED;
    }
    auto &startTensor = chain.front()->iOperand.front();
    auto &endTensor = chain.back()->oOperand.front();
    if (!startTensor || !endTensor) {
        ALOG_ERROR("Null tensor found in chain.");
        return FAILED;
    }
    // 计算合并offset
    auto [newOffset, newDynOffset] = CalculateAssembleOffsets(chain, startTensor->offset.size());
    // 4. 记录并清理
    RecordAssembleOperation(startTensor, endTensor, newOffset, newDynOffset);
    function.GetTensorMap().Erase(endTensor);
    operation.SetAsDeleted();

    return SUCCESS;
}

std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> MergeViewAssembleUtils::CalculateAssembleOffsets(
    const std::vector<Operation *> &chain, size_t offsetSize) {
    std::vector<int64_t> newOffset(offsetSize, 0);
    std::vector<SymbolicScalar> newDynOffset;
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto &assemble = chain[i];
        if (!assemble) {
            return {};
        }
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(assemble->GetOpAttribute());
        if (!assembleOpAttribute) {
            return {};
        }
        if (i == 0) {
            newOffset = assembleOpAttribute->GetToOffset();
            newDynOffset = assembleOpAttribute->GetToDynOffset();
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, assembleOpAttribute->GetToOffset(), assembleOpAttribute->GetToDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
    }
    return {newOffset, newDynOffset};
}

void MergeViewAssembleUtils::RecordAssembleOperation(const std::shared_ptr<LogicalTensor> &input,
    const std::shared_ptr<LogicalTensor> &output, const std::vector<int64_t> &offset,
    const std::vector<SymbolicScalar> &dynOffset) {
    assembleOpToAppend_.emplace_back(AssembleOp{input, output, offset, dynOffset});
}

Status MergeViewAssembleUtils::EraseRedundantAssemble(Function &function) const {
    std::unordered_set<Operation *> redundantAssembles;
    for (auto &op : function.Operations(false)) {
        if (op.GetOpcode() !=  Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.iOperand.empty()) {
            ALOG_ERROR("Assemble operation with no input operands.");
            return FAILED;
        }
        if (op.iOperand.front()->GetProducers().empty()) {
            redundantAssembles.emplace(&op);
        }
    }
    for (auto &ele : redundantAssembles) {
        if (!ele) {
            continue;
        }
        ele->SetAsDeleted();
    }
    function.EraseOperations(true, false);
    return SUCCESS;
}
} // namespace npu::tile_fwk
