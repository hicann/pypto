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
 * \file mix_call_operation_builder.CPP
 * \brief
 */

#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_call_operation_builder.h"

namespace npu {
namespace tile_fwk {
Status MixCallOperationBuilder::CreateCallOps(Function& rootFunc, const std::vector<Operation*>& originalCallOps,
                                              Function* originalMixFunc,
                                              const std::vector<InternalComponentInfo>& components,
                                              const std::vector<uint64_t>& newProgramIDs,
                                              SubgraphToFunction& subgraphToFunction,
                                              std::vector<Function*>& newFunctions,
                                              const std::vector<InternalDependencyInfo>& internalDeps)
{
    ALOG_INFO_F("Creating call operations for %zu original call ops and %zu components",
                originalCallOps.size(), components.size());
    std::vector<CallOpCreationInfo> callOpInfos;
    // 处理所有指向同构originalMixFunc的originalCallOps
    for (auto* originalCallOp : originalCallOps) {
        // 为每个原始callOp分配唯一的wrapId
        uint64_t wrapId = nextWrapId_++;
        ALOG_DEBUG_F("Assigning wrapId=%lu for original callOp %d", wrapId, originalCallOp->GetOpMagic());
        for (size_t i = 0; i < components.size(); i++) {
            CallOpCreationInfo info;
            info.leafFunc = newFunctions[i];
            info.newProgramID = newProgramIDs[i];
            info.componentIndex = i;
            info.originalCallOp = originalCallOp;
            info.wrapId = wrapId;
            auto status = CreateCallOpInRootFunction(rootFunc, *info.leafFunc, info.newProgramID, info.componentIndex,
                                                     info.originalCallOp, originalMixFunc, subgraphToFunction, info);
            if (status != SUCCESS) {
                ALOG_ERROR_F("Failed to create call op for component %zu", info.componentIndex);
                return FAILED;
            }
            if (!info.createdCallOp) {
                ALOG_ERROR_F("Created call op is null for component %zu", info.componentIndex);
                return FAILED;
            }
            // 记录这个call op的信息
            callOpInfos.push_back(info);
            ALOG_DEBUG_F("Created call op %d in info for component %zu (wrapId=%lu)",
                         info.createdCallOp->GetOpMagic(), info.componentIndex, wrapId);
        }
    }
    // 现在统一处理内部依赖（按wrapId分组）
    ProcessAllInternalDependencies(rootFunc, callOpInfos, internalDeps);
    ALOG_INFO_F("Successfully created %zu call operations with internal dependencies",
                callOpInfos.size());
    return SUCCESS;
}

Status MixCallOperationBuilder::CreateCallOpInRootFunction(Function& rootFunc,
                                                           Function& leafFunc,
                                                           uint64_t newProgramID,
                                                           uint64_t componentIndex,
                                                           Operation* originalCallOp,
                                                           Function* originalMixFunc,
                                                           SubgraphToFunction& subgraphToFunction,
                                                           CallOpCreationInfo& info)
{
    ALOG_DEBUG_F("Creating callOp in root function for leaf: %s, programID=%d, component=%d, wrapId=%lu",
                 leafFunc.GetRawName().c_str(), newProgramID, componentIndex, info.wrapId);
    auto originalCallAttr = dynamic_cast<CallOpAttribute*>(originalCallOp->GetOpAttribute().get());
    if (originalCallAttr == nullptr) {
        ALOG_ERROR_F("Original callOp %d has no CallOpAttribute", originalCallOp->GetOpMagic());
        return FAILED;
    }
    // 获取原始callOp的operands
    auto originalIOperands = originalCallOp->GetIOperands();
    auto originalOOperands = originalCallOp->GetOOperands();
    ALOG_DEBUG_F("Original callOp %d has %zu iOperands, %zu oOperands",
                 originalCallOp->GetOpMagic(), originalIOperands.size(), originalOOperands.size());
    auto originalIncasts = originalMixFunc->GetIncast();
    auto originalOutcasts = originalMixFunc->GetOutcast();
    ALOG_DEBUG_F("Original mix function %s has %zu incasts, %zu outcasts",
                 originalMixFunc->GetRawName().c_str(), originalIncasts.size(), originalOutcasts.size());
    // 从invokeInfo获取incast和outcast参数信息
    const auto& invokeInfo = subgraphToFunction.subFuncInvokeInfos[componentIndex];
    // 构建新的operands列表
    std::vector<LogicalTensorPtr> newIOperands;
    std::vector<LogicalTensorPtr> newOOperands;
    // 用于跟踪已经处理过的tensor
    std::set<LogicalTensorPtr> processedTensors;
    // 处理直接外部依赖的IOperands
    ALOG_INFO_F("===> FindIOperandsAndOOperands start.");
    FindNewIOperandsInOriginalIncast(originalIOperands, originalIncasts, invokeInfo, newIOperands, processedTensors);
    // 处理直接外部依赖的OOperands
    FindNewOOperandsInOriginalOutcast(originalOOperands, originalOutcasts, invokeInfo, newOOperands, processedTensors);
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();
    // 处理传播依赖的IOperands和OOperands
    FindNewIOperandsAndOOperandsInPropagateInOutcast(originalIOperands, originalOOperands,
                                                     originalIncasts, originalOutcasts,
                                                     actualIncasts, actualOutcasts,
                                                     newIOperands, newOOperands, processedTensors);
    ALOG_INFO_F("===> FindIOperandsAndOOperands end.");
    auto& callOp = rootFunc.AddRawOperation(Opcode::OP_CALL, newIOperands, newOOperands, false);
    ALOG_INFO_F("Created operands for new callOp %d: %zu inputs, %zu outputs",
                callOp.GetOpMagic(), newIOperands.size(), newOOperands.size());
    // 寻找新call op的IOpAttrOffset和OOpAttrOffset
    FindIOpAttrOffsetAndOOpAttrOffset(leafFunc, invokeInfo, info.iOffsets, info.oOffsets, originalMixFunc);
    callOp.SetOpOffset(info.iOffsets, info.oOffsets);
    SetCallOpAttribute(leafFunc, callOp, originalCallOp, originalCallAttr,
                       newProgramID, componentIndex, subgraphToFunction, info);
    // 将创建的call op记录到info中
    info.createdCallOp = &callOp;
    ALOG_INFO_F("Successfully created callOp %d in root function for programID=%d, leaf=%s",
                callOp.GetOpMagic(), newProgramID, leafFunc.GetRawName().c_str());
    return SUCCESS;
}

void MixCallOperationBuilder::FindNewIOperandsInOriginalIncast(
    const std::vector<LogicalTensorPtr>& originalIOperands,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalIncasts,
    const SubfuncInvokeInfoTy& invokeInfo,
    std::vector<LogicalTensorPtr>& newIOperands,
    std::set<LogicalTensorPtr>& processedTensors) const
{
    // 为incast构建新的iOperands
    for (const auto& incastParam : invokeInfo.GetIncastTensorParamList()) {
        int tensorMagic = incastParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedTensors.insert(incastParam.tensor);
            ALOG_DEBUG_F("  Found: tensor magic=%d -> original iOperand[%d] (tensor magic=%d)",
                         tensorMagic, originalIndex, originalIOperands[originalIndex]->magic);
        }
    }
    // 为global tensor输入构建新的iOperands
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic == -1 || tensorParam.isOutputToGM) {
            continue;
        }
        int tensorMagic = tensorParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedTensors.insert(tensorParam.tensor);
            ALOG_DEBUG_F("  Found: global input tensor magic=%d -> original iOperand[%d]",
                         tensorMagic, originalIndex);
        }
    }
}


void MixCallOperationBuilder::FindNewOOperandsInOriginalOutcast(
    const std::vector<LogicalTensorPtr>& originalOOperands,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalOutcasts,
    const SubfuncInvokeInfoTy& invokeInfo,
    std::vector<LogicalTensorPtr>& newOOperands,
    std::set<LogicalTensorPtr>& processedTensors) const
{
    // 为outcast构建新的oOperands
    for (const auto& outcastParam : invokeInfo.GetOutcastTensorParamList()) {
        int tensorMagic = outcastParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOOperands[originalIndex]);
            processedTensors.insert(outcastParam.tensor);
            ALOG_DEBUG_F("  Found: tensor magic=%d -> original oOperand[%d] (tensor magic=%d)",
                         tensorMagic, originalIndex, originalOOperands[originalIndex]->magic);
        }
    }
    // 为global tensor输出构建新的oOperands
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic == -1 || tensorParam.tensor == nullptr || !tensorParam.isOutputToGM) {
            continue;
        }
        int tensorMagic = tensorParam.tensor->magic;
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOOperands[originalIndex]);
            processedTensors.insert(tensorParam.tensor);
            ALOG_DEBUG_F("  Found: global output tensor magic=%d -> original oOperand[%d]",
                         tensorMagic, originalIndex);
        }
    }
}


void MixCallOperationBuilder::FindNewIOperandsAndOOperandsInPropagateInOutcast(
    const std::vector<LogicalTensorPtr>& originalIOperands,
    const std::vector<LogicalTensorPtr>& originalOOperands,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalIncasts,
    const std::vector<std::shared_ptr<LogicalTensor>>& originalOutcasts,
    const std::vector<std::shared_ptr<LogicalTensor>>& actualIncasts,
    const std::vector<std::shared_ptr<LogicalTensor>>& actualOutcasts,
    std::vector<LogicalTensorPtr>& newIOperands,
    std::vector<LogicalTensorPtr>& newOOperands,
    std::set<LogicalTensorPtr>& processedTensors) const
{
    // 处理传播的incast
    for (const auto& incast : actualIncasts) {
        // 检查是否已经在之前的列表中处理过
        if (processedTensors.count(incast) > 0) {
            ALOG_DEBUG_F("  Propagated incast tensor magic=%d already processed, skipping", incast->magic);
            continue;
        }
        int tensorMagic = incast->magic;
        ALOG_DEBUG_F("  Checking propagated incast tensor magic=%d", tensorMagic);
        int originalIndex = FindTensorIndexInList(tensorMagic, originalIncasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalIOperands.size())) {
            newIOperands.push_back(originalIOperands[originalIndex]);
            processedTensors.insert(incast);
            ALOG_DEBUG_F("    Found: propagated incast tensor magic=%d -> original iOperand[%d]",
                         tensorMagic, originalIndex);
        }
    }
    // 处理传播的outcast
    for (const auto& outcast : actualOutcasts) {
        if (processedTensors.count(outcast) > 0) {
            ALOG_DEBUG_F("  Propagated outcast tensor magic=%d already processed, skipping", outcast->magic);
            continue;
        }
        int tensorMagic = outcast->magic;
        ALOG_DEBUG_F("  Checking propagated outcast tensor magic=%d", tensorMagic);
        int originalIndex = FindTensorIndexInList(tensorMagic, originalOutcasts);
        if (originalIndex >= 0 && originalIndex < static_cast<int>(originalOOperands.size())) {
            newOOperands.push_back(originalOOperands[originalIndex]);
            processedTensors.insert(outcast);
            ALOG_DEBUG_F("    Found: propagated outcast tensor magic=%d -> original oOperand[%d]",
                         tensorMagic, originalIndex);
        }
    }
}

// 辅助函数：在tensor列表中查找指定magic的tensor索引
int MixCallOperationBuilder::FindTensorIndexInList(int tensorMagic,
                                                   const std::vector<LogicalTensorPtr>& tensorList) const
{
    for (size_t i = 0; i < tensorList.size(); ++i) {
        if (tensorList[i] != nullptr && tensorList[i]->magic == tensorMagic) {
            return static_cast<int>(i);
        }
    }
    return -1;
}


void MixCallOperationBuilder::FindIOpAttrOffsetAndOOpAttrOffset(
    Function& leafFunc,
    const SubfuncInvokeInfoTy& invokeInfo,
    std::vector<int>& iOffsets,
    std::vector<int>& oOffsets,
    Function* originalMixFunc) const
{
    // 清空offset向量
    iOffsets.clear();
    oOffsets.clear();
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();

    ALOG_INFO_F("===> FindIOpAttrOffsetAndOOpAttrOffset start.");
    ALOG_DEBUG_F("Leaf function %s has %zu actual incasts, %zu actual outcasts after dependency propagation",
                 leafFunc.GetRawName().c_str(), actualIncasts.size(), actualOutcasts.size());
    std::set<LogicalTensorPtr> processedTensors;
    // 处理直接参数（在原始invokeInfo中能找到的）
    ExtractInfo extractInfo{iOffsets, oOffsets, processedTensors};
    // 使用invokeInfo中预先构造的incast信息
    if (!FindIOpAttrOffsetFromIncast(invokeInfo, leafFunc, extractInfo)) {
        return;
    }
    // 使用invokeInfo中的outcast信息
    if (!FindOOpAttrOffsetFromOutcast(invokeInfo, leafFunc, extractInfo)) {
        return;
    }
    // 使用invokeInfo中的global tensor信息
    if (!FindIOOpAttrOffsetGlobalTensor(invokeInfo, leafFunc, extractInfo)) {
        return;
    }
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    if (!FindIOpAttrOffsetFromActualIncasts(actualIncasts, extractInfo, originalMixFunc)) {
        return;
    }
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    if (!FindOOpAttrOffsetFromActualOutcasts(actualOutcasts, extractInfo, originalMixFunc)) {
        return;
    }
    ALOG_INFO_F("===> FindIOpAttrOffsetAndOOpAttrOffset end.");
}


bool MixCallOperationBuilder::FindIOpAttrOffsetFromIncast(const SubfuncInvokeInfoTy& invokeInfo,
                                                          Function& leafFunc,
                                                          ExtractInfo& extractInfo) const
{
    // 使用invokeInfo中预先构造的incast信息
    for (const auto& in : invokeInfo.GetIncastTensorParamList()) {
        if (in.opMagic == -1) { // 跳过无效的incast
            continue;
        }
        int offset = GetOffsetFromOp(in.opMagic, in.operandIdx, leafFunc, false);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to get offset for incast (op=%d, idx=%d)!",
                         in.opMagic, in.operandIdx);
            continue;
        }
        extractInfo.iOffsets.push_back(offset);
        extractInfo.processedTensors.insert(in.tensor);
        ALOG_DEBUG_F("Incast (op=%d, idx=%d) -> original offset=%d",
                     in.opMagic, in.operandIdx, offset);
    }
    return true;
}

bool MixCallOperationBuilder::FindOOpAttrOffsetFromOutcast(const SubfuncInvokeInfoTy& invokeInfo,
                                                           Function& leafFunc,
                                                           ExtractInfo& extractInfo) const
{
    // 使用invokeInfo中预先构造的outcast信息
    for (const auto& out : invokeInfo.GetOutcastTensorParamList()) {
        if (out.opMagic == -1) {
            continue;
        }
        int offset = GetOffsetFromOp(out.opMagic, out.operandIdx, leafFunc, true);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to get offset for outcast (op=%d, idx=%d)!",
                         out.opMagic, out.operandIdx);
            continue;
        }
        extractInfo.oOffsets.push_back(offset);
        extractInfo.processedTensors.insert(out.tensor);
        ALOG_DEBUG_F("Outcast (op=%d, idx=%d) -> original offset=%d",
                     out.opMagic, out.operandIdx, offset);
    }
    return true;
}

// 统一的offset获取函数
int MixCallOperationBuilder::GetOffsetFromOp(int opMagic, int operandIdx,
                                             Function& leafFunc, bool isOutput) const
{
    auto operations = leafFunc.Operations(false);
    for (auto& op : operations) {
        if (op.GetOpMagic() != opMagic) {
            continue;
        }
        if (isOutput) {
            if (operandIdx >= 0 && static_cast<size_t>(operandIdx) < op.GetOOperands().size()) {
                int offset = op.GetOOpAttrOffset(operandIdx);
                return offset;
            }
        } else {
            if (operandIdx >= 0 && static_cast<size_t>(operandIdx) < op.GetIOperands().size()) {
                int offset = op.GetIOpAttrOffset(operandIdx);
                return offset;
            }
        }
    }
    
    ALOG_WARN_F("Could not find offset for op %d idx %d (isOutput=%d)",
                opMagic, operandIdx, isOutput);
    return -1;
}

bool MixCallOperationBuilder::FindIOOpAttrOffsetGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo,
                                                             Function& leafFunc,
                                                             ExtractInfo& extractInfo) const
{
    for (const auto &tensor : invokeInfo.GetTensorParamList()) {
        if (tensor.opMagic == -1) {
            continue;
        }
        int offset = GetOffsetFromOp(tensor.opMagic, tensor.operandIdx, leafFunc, tensor.isOutputToGM);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to get offset for global tensor (op=%d, idx=%d, isOutput=%d)!",
                         tensor.opMagic, tensor.operandIdx, tensor.isOutputToGM);
            continue;
        }
        if (tensor.isOutputToGM) {
            extractInfo.oOffsets.push_back(offset);
            ALOG_DEBUG_F("Global tensor -> Outcast: opmagic=%d, idx=%d -> oOpAttrOffset=%d",
                         tensor.opMagic, tensor.operandIdx, offset);
        } else {
            extractInfo.iOffsets.push_back(offset);
            ALOG_DEBUG_F("Global tensor -> Incast: opmagic=%d, idx=%d -> iOpAttrOffset=%d",
                         tensor.opMagic, tensor.operandIdx, offset);
        }
        extractInfo.processedTensors.insert(tensor.tensor);
    }
    return true;
}

bool MixCallOperationBuilder::FindIOpAttrOffsetFromActualIncasts(
    const std::vector<std::shared_ptr<LogicalTensor>> &actualIncasts,
    ExtractInfo& extractInfo,
    Function* originalMixFunc) const
{
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    for (const auto& incast : actualIncasts) {
        if (incast == nullptr || extractInfo.processedTensors.count(incast) > 0) {
            continue;
        }

        // 这是传播依赖添加的参数，需要特殊处理
        // 在原始Mix function中查找这个tensor的offset
        int offset = FindOriginalOffsetInMixFunction(incast, originalMixFunc);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to find offset for propagated incast tensor %d!", incast->GetRawMagic());
            return false;  // 直接报错返回
        }
        extractInfo.iOffsets.push_back(offset);
        extractInfo.processedTensors.insert(incast);
        ALOG_DEBUG_F("Extracted propagated incast: tensor rawmagic = %d, offset = %d",
                     incast->GetRawMagic(), offset);
    }
    return true;
}

bool MixCallOperationBuilder::FindOOpAttrOffsetFromActualOutcasts(
    const std::vector<std::shared_ptr<LogicalTensor>> &actualOutcasts,
    ExtractInfo& extractInfo,
    Function* originalMixFunc) const
{
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    for (const auto& outcast : actualOutcasts) {
        if (outcast == nullptr || extractInfo.processedTensors.count(outcast) > 0) {
            continue;
        }
        auto shape = outcast->GetShape();
        if (shape.empty()) {
            ALOG_ERROR_F("Propagated outcast tensor %d has empty shape!", outcast->GetRawMagic());
            return false;
        }
        int offset = FindOriginalOffsetInMixFunction(outcast, originalMixFunc);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to find offset for propagated outcast tensor %d!", outcast->GetRawMagic());
            return false;  // 直接报错返回
        }
        extractInfo.oOffsets.push_back(offset);
        extractInfo.processedTensors.insert(outcast);
        ALOG_DEBUG_F("Extracted propagated outcast: tensor rawmagic = %d -> original offset = %d",
                        outcast->GetRawMagic(), offset);
    }
    return true;
}

int MixCallOperationBuilder::FindOriginalOffsetInMixFunction(LogicalTensorPtr tensor,
                                                             Function* originalMixFunc) const
{
    if (tensor == nullptr || originalMixFunc == nullptr) {
        ALOG_ERROR_F("Tensor or function is nullptr in FindOriginalOffsetInMixFunction");
        return -1;
    }
    int rawMagic = tensor->GetRawMagic();
    ALOG_DEBUG_F("Finding original offset for tensor raw magic=%d in function %s",
                 rawMagic, originalMixFunc->GetRawName().c_str());
    auto operations = originalMixFunc->Operations(false);
    for (auto& op : operations) {
        if (op.IsNOP()) continue;
        auto iOperands = op.GetIOperands();
        for (size_t i = 0; i < iOperands.size(); i++) {
            auto inputTensor = iOperands[i];
            if (inputTensor.get() == tensor.get()) {
                int offset = op.GetIOpAttrOffset(i);
                if (offset != -1) {
                    return offset;
                }
            }
        }
        auto oOperands = op.GetOOperands();
        for (size_t i = 0; i < oOperands.size(); i++) {
            auto outputTensor = oOperands[i];
            if (outputTensor.get() == tensor.get()) {
                int offset = op.GetOOpAttrOffset(i);
                if (offset != -1) {
                    return offset;
                }
            }
        }
    }
    ALOG_ERROR_F("Tensor raw magic=%d not found in function %s operations",
                 rawMagic, originalMixFunc->GetRawName().c_str());
    return -1;
}

// 设置新创建的callOp的属性
void MixCallOperationBuilder::SetCallOpAttribute(Function& leafFunc,
                                                 Operation& callOp,
                                                 Operation* originalCallOp,
                                                 CallOpAttribute* originalCallAttr,
                                                 uint64_t newProgramID,
                                                 uint64_t componentIndex,
                                                 SubgraphToFunction& subgraphToFunction,
                                                 CallOpCreationInfo& info)
{
    if (originalCallAttr == nullptr) {
        ALOG_ERROR_F("Original callOp %d has no CallOpAttribute", originalCallOp->GetOpMagic());
        return;
    }
    // 使用原mix callOp的argList
    auto argList = originalCallAttr->GetArgList();
    std::map<int, SymbolicScalar> outIndexToExpr;
    leafFunc.GetOutcastSymbolicExpr(outIndexToExpr);
    // 创建CallOpAttribute（使用从原始CallOp提取的argList）
    auto callAttr = leafFunc.CreateCallOpAttribute(argList, outIndexToExpr);
    auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(callAttr);
    if (callOpAttr != nullptr) {
        callOpAttr->wrapId = info.wrapId;
        ALOG_DEBUG_F("Set wrapId=%lu to callOp attribute for programID=%d (from original callOp %d)",
                     info.wrapId, newProgramID, originalCallOp->GetOpMagic());
    }
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(newProgramID);
    if (componentIndex < subgraphToFunction.subFuncInvokeInfos.size()) {
        callOp.SetSubFuncInvokeInfo(subgraphToFunction.subFuncInvokeInfos[componentIndex]);
    }
    if (callOpAttr != nullptr && callOpAttr->invokeInfo_ != nullptr) {
        callOpAttr->invokeInfo_->UpdateProgramSubgraphId(newProgramID);
    }
    subgraphToFunction.SetSemanticLabel(leafFunc.GetProgramOp(), callOp);
    ALOG_DEBUG_F("Created callOp %d: %zu arg blocks (from original callOp %d), %zu input offsets, %zu output offsets",
                 callOp.GetOpMagic(), argList.size(), originalCallOp->GetOpMagic(),
                 info.iOffsets.size(), info.oOffsets.size());
}

// 添加内部依赖的depend operands
void MixCallOperationBuilder::ProcessAllInternalDependencies(
    Function& rootFunc,
    const std::vector<CallOpCreationInfo>& callOpInfos,
    const std::vector<InternalDependencyInfo>& internalDeps) const
{
    if (internalDeps.empty()) {
        ALOG_DEBUG_F("No internal dependencies to process");
        return;
    }
    // 按wrapId分组call op信息
    std::unordered_map<uint64_t, std::vector<const CallOpCreationInfo*>> wrapIdToInfos;
    for (const auto& info : callOpInfos) {
        wrapIdToInfos[info.wrapId].push_back(&info);
    }
    ALOG_INFO_F("Processing internal dependencies for %zu wrap groups", wrapIdToInfos.size());
    // 为每个wrap组处理内部依赖
    for (const auto& [wrapId, infos] : wrapIdToInfos) {
        ProcessInternalDependenciesForWrap(rootFunc, infos, internalDeps, wrapId);
    }
}

// 为单个wrap组处理内部依赖
void MixCallOperationBuilder::ProcessInternalDependenciesForWrap(
    Function& rootFunc,
    const std::vector<const CallOpCreationInfo*>& infos,
    const std::vector<InternalDependencyInfo>& internalDeps,
    uint64_t wrapId) const
{
    // 构建scope索引到call op的映射
    std::unordered_map<int, Operation*> componentToCallOp;
    for (const auto& info : infos) {
        if (info->createdCallOp) {
            componentToCallOp[info->componentIndex] = info->createdCallOp;
            ALOG_DEBUG_F("Wrap %lu: map component %d -> call op %d",
                         wrapId, info->componentIndex, info->createdCallOp->GetOpMagic());
        }
    }
    // 如果没有内部依赖，直接返回
    if (internalDeps.empty()) {
        ALOG_DEBUG_F("Wrap %lu: No internal dependencies to add", wrapId);
        return;
    }
    ALOG_INFO_F("Wrap %lu: Processing %zu internal dependencies", wrapId, internalDeps.size());
    
    // 处理每个内部依赖
    for (const auto& dep : internalDeps) {
        int srcComp = dep.srcComp;
        int dstComp = dep.dstComp;
        auto srcIt = componentToCallOp.find(srcComp);
        auto dstIt = componentToCallOp.find(dstComp);
        if (srcIt == componentToCallOp.end() || dstIt == componentToCallOp.end()) {
            // 这个wrap中可能不包含这个依赖的所有scope
            continue;
        }
        Operation* producerCallOp = srcIt->second;
        Operation* consumerCallOp = dstIt->second;
        // 生成tensor key
        const char* scopeType = (dep.compType == ComponentType::C_SCOPE) ? "C" : "V";
        std::string tensorKey = "depend_" + std::to_string(wrapId) + "_" +
                                scopeType + std::to_string(srcComp) + "_to_" +
                                scopeType + std::to_string(dstComp);
        // 创建新的dummy tensor
        LogicalTensorPtr dependTensor = std::make_shared<LogicalTensor>(
            rootFunc,                   // 属于root function
            DataType::DT_INT8,             // 最小数据类型
            Shape({1}),                 // 最小形状 (标量)
            TileOpFormat::TILEOP_ND,    // 默认格式
            tensorKey,                  // tensor名称
            NodeType::LOCAL             // 节点类型
        );
        dependTensor->AddProducer(producerCallOp);
        dependTensor->AddConsumer(consumerCallOp);
        consumerCallOp->AddDependOperand(dependTensor);
        ALOG_INFO_F("Wrap %lu: component %d -> component %d "
                    "(call op %d -> %d, tensor magic=%d, has %zu producers, %zu consumers)",
                    wrapId, srcComp, dstComp,
                    producerCallOp->GetOpMagic(), consumerCallOp->GetOpMagic(),
                    dependTensor->magic, dependTensor->GetProducers().size(), dependTensor->GetConsumers().size());
    }
}
}
}