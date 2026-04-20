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
 * \file supernode_graph_builder.cpp
 * \brief
 */

#include "supernode_graph_builder.h"
#include <iostream>
#include <deque>
#include <algorithm>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SuperNodeGraphBuilder"

namespace npu::tile_fwk {

uint64_t OperationGraphInfo::GetHash(const Operation* op) const
{
    std::string hashString;
    hashString.append(op->GetOpcodeStr());
    for (const auto& tensor : op->GetIOperands()) {
        hashString.append("IOperand-");
        hashString.append(std::to_string(tensor->GetMemoryTypeOriginal()));
        hashString.append(std::to_string(tensor->tensor->datatype));
        for (const auto& tensorDim : tensor->shape) {
            hashString.append(std::to_string(tensorDim));
            hashString.append("-");
        }
    }
    for (const auto& tensor : op->GetOOperands()) {
        hashString.append("OOperand-");
        hashString.append(std::to_string(tensor->GetMemoryTypeOriginal()));
        hashString.append(std::to_string(tensor->tensor->datatype));
        for (const auto& tensorDim : tensor->shape) {
            hashString.append(std::to_string(tensorDim));
            hashString.append("-");
        }
    }
    return std::hash<std::string>{}(hashString);
}

std::vector<int32_t> OperationGraphInfo::GetSameLevelOpIdx(int32_t opIdx, Opcode opLabel) const
{
    if (opIdx < 0 || opIdx >= static_cast<int32_t>(opList_.size()) || opList_[opIdx]->GetOOperands().empty()) {
        return {};
    }
    std::vector<int32_t> res;
    std::shared_ptr<LogicalTensor> output = opList_[opIdx]->GetOOperands()[0];
    for (const auto& parentOpPtr : output->GetProducers()) {
        if (parentOpPtr->GetOpcode() == opLabel) {
            int32_t parentOpMagic = parentOpPtr->GetOpMagic();
            if (magic2Idx_.count(parentOpMagic) > 0) {
                int32_t targetIdx = magic2Idx_.at(parentOpPtr->GetOpMagic());
                res.push_back(targetIdx);
            }
        }
    }
    return res;
}

bool OperationGraphInfo::CoreTypeMergeable(const std::set<OpCoreType>& coreTypes) const
{
    if (coreTypes.size() == 1 && (*coreTypes.begin() == OpCoreType::AICPU || *coreTypes.begin() == OpCoreType::HUB)) {
        return false;
    }
    if (useCVMixPartition_ || coreTypes.size() == 1) {
        return true;
    }
    const size_t maxSeperateCoreNum = 2;
    if (coreTypes.size() > maxSeperateCoreNum) {
        return false;
    }
    if (coreTypes.size() == maxSeperateCoreNum) {
        auto firstType = *coreTypes.begin();
        auto secondType = *(++coreTypes.begin());
        if (firstType == OpCoreType::AICPU || secondType == OpCoreType::AICPU) {
            return false;
        }
        if (firstType == OpCoreType::HUB || secondType == OpCoreType::HUB) {
            return false;
        }
        if (firstType == OpCoreType::ANY || secondType == OpCoreType::ANY) {
            return true;
        }
    }
    return false;
}

inline int32_t FindParent(std::vector<int32_t>& parent, int32_t i)
{
    if (i < 0 || i >= static_cast<int32_t>(parent.size())) {
        APASS_LOG_ERROR_F(Elements::Operation, "Call FindParent with illegal parameter %d.", i);
        return -1;
    }
    if (parent[i] == i) {
        return i;
    }
    std::vector<int32_t> searchPath;
    int32_t currIdx = i;
    while (parent[currIdx] != currIdx) {
        searchPath.push_back(currIdx);
        currIdx = parent[currIdx];
        if (currIdx < 0 || currIdx >= static_cast<int32_t>(parent.size())) {
            APASS_LOG_ERROR_F(Elements::Operation, "Find illegal parameter %d in FindParent.", currIdx);
            return -1;
        }
        if (searchPath.size() > (parent.size() + 1)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Find loop in FindParent.");
            return -1;
        }
    }
    for (auto parentIdx : searchPath) {
        parent[parentIdx] = currIdx;
    }
    return currIdx;
}

inline std::string GetOpCoreTypeStr(OpCoreType coreType)
{
    std::map<OpCoreType, std::string> coreTypeStr{
        {OpCoreType::AIC, "AIC"},
        {OpCoreType::AIV, "AIV"},
        {OpCoreType::AICPU, "AICPU"},
        {OpCoreType::HUB, "HUB"},
        {OpCoreType::GMATOMIC, "GMATOMIC"}};
    if (coreTypeStr.count(coreType) == 0) {
        return "UNKNOWN";
    } else {
        return coreTypeStr[coreType];
    }
}

Status NodeGraphInfo::MergeSrcToDstIsland(
    const std::shared_ptr<OperationGraphInfo> operationGraphInfo, std::vector<int32_t>& parent, int32_t src,
    int32_t dst)
{
    int32_t srcParent = FindParent(parent, src);
    int32_t dstParent = FindParent(parent, dst);
    if (srcParent == -1 || dstParent == -1) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "Merge node in the disjoint set failed.%s",
            GetFormatBacktrace(*(operationGraphInfo->opList_[src])).c_str());
        return FAILED;
    }
    std::set<OpCoreType> coreTypes{
        operationGraphInfo->opCoreType_[src], operationGraphInfo->opCoreType_[dst],
        operationGraphInfo->opCoreType_[srcParent], operationGraphInfo->opCoreType_[dstParent]};
    bool isAICPUandVIEW = false;
    isAICPUandVIEW = isAICPUandVIEW || (operationGraphInfo->opCoreType_[src] == OpCoreType::AICPU &&
                                        operationGraphInfo->opList_[dst]->GetOpcode() == Opcode::OP_VIEW);
    isAICPUandVIEW = isAICPUandVIEW || (operationGraphInfo->opCoreType_[dst] == OpCoreType::AICPU &&
                                        operationGraphInfo->opList_[src]->GetOpcode() == Opcode::OP_VIEW);
    bool isAICPUandAssemble = false;
    // HUB只能和View/Assemble在一张子图中
    bool hubWithViewAssemble = false;
    const size_t maxCoreTypesNum = 2;
    if (coreTypes.size() == maxCoreTypesNum && coreTypes.count(OpCoreType::HUB) > 0) {
        auto srcOpCoreType = operationGraphInfo->opCoreType_[src];
        auto dstOpCoreType = operationGraphInfo->opCoreType_[dst];
        auto srcOpCode = operationGraphInfo->opList_[src]->GetOpcode();
        auto dstOpCode = operationGraphInfo->opList_[dst]->GetOpcode();
        if (srcOpCoreType != OpCoreType::HUB) {
            hubWithViewAssemble = (srcOpCode == Opcode::OP_VIEW || srcOpCode == Opcode::OP_ASSEMBLE);
        }
        if (dstOpCoreType != OpCoreType::HUB) {
            hubWithViewAssemble = (dstOpCode == Opcode::OP_VIEW || dstOpCode == Opcode::OP_ASSEMBLE);
        }
    }
    isAICPUandAssemble = isAICPUandAssemble || (operationGraphInfo->opCoreType_[src] == OpCoreType::AICPU &&
                                                operationGraphInfo->opList_[dst]->GetOpcode() == Opcode::OP_ASSEMBLE);
    isAICPUandAssemble = isAICPUandAssemble || (operationGraphInfo->opCoreType_[dst] == OpCoreType::AICPU &&
                                                operationGraphInfo->opList_[src]->GetOpcode() == Opcode::OP_ASSEMBLE);
    if ((!hubWithViewAssemble) && (!isAICPUandVIEW) && (!isAICPUandAssemble) &&
        (!operationGraphInfo->CoreTypeMergeable(coreTypes))) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "Try to merge operations with different OpCoreType in building SuperNode.");
        std::vector<int> mergeIdxs{src, srcParent, dst, dstParent};
        for (int mergeIdx : mergeIdxs) {
            auto& mergeOp = operationGraphInfo->opList_[mergeIdx];
            APASS_LOG_ERROR_F(
                Elements::Operation, "%s [opMagic: %d] [opCoreType: %s].%s", mergeOp->GetOpcodeStr().c_str(),
                mergeOp->GetOpMagic(), GetOpCoreTypeStr(operationGraphInfo->opCoreType_[mergeIdx]).c_str(),
                GetFormatBacktrace(*mergeOp).c_str());
        }
        return FAILED;
    }
    parent[srcParent] = dstParent;
    return SUCCESS;
}

std::vector<int32_t> NodeInnerExpand(
    const std::shared_ptr<OperationGraphInfo> operationGraphInfo, std::vector<int32_t>& nodeOps)
{
    std::vector<int32_t> frontBackVisitedOp;
    int32_t minOpIdx = static_cast<int32_t>(operationGraphInfo->opList_.size());
    int32_t maxOpIdx = -1;
    for (int32_t opIdx : nodeOps) {
        minOpIdx = opIdx < minOpIdx ? opIdx : minOpIdx;
        maxOpIdx = opIdx > maxOpIdx ? opIdx : maxOpIdx;
    }
    std::unordered_set<int32_t> frontVisitedOp;
    std::vector<int32_t> frontVisitStack(nodeOps);
    while (frontVisitStack.size() > 0) {
        int32_t opIdx = frontVisitStack.back();
        frontVisitStack.pop_back();
        if (frontVisitedOp.count(opIdx) > 0) {
            continue;
        }
        frontVisitedOp.insert(opIdx);
        for (int32_t nextOpIdx : operationGraphInfo->outGraph_[opIdx]) {
            if (nextOpIdx <= maxOpIdx) {
                frontVisitStack.push_back(nextOpIdx);
            }
        }
    }
    std::unordered_set<int32_t> backVisitedOp;
    std::vector<int32_t> backVisitStack(nodeOps);
    while (backVisitStack.size() > 0) {
        int32_t opIdx = backVisitStack.back();
        backVisitStack.pop_back();
        if (backVisitedOp.count(opIdx) > 0) {
            continue;
        }
        if (frontVisitedOp.count(opIdx) > 0) {
            frontBackVisitedOp.push_back(opIdx);
        }
        backVisitedOp.insert(opIdx);
        for (int32_t prevOpIdx : operationGraphInfo->inGraph_[opIdx]) {
            if (prevOpIdx >= minOpIdx) {
                backVisitStack.push_back(prevOpIdx);
            }
        }
    }
    return frontBackVisitedOp;
}

Status NodeGraphInfo::AvoidLoop(
    const std::shared_ptr<OperationGraphInfo> operationGraphInfo, std::vector<int32_t>& parent,
    std::vector<std::vector<int32_t>>& node2Op, bool& updated)
{
    std::vector<Operation*>& opList = operationGraphInfo->opList_;
    std::vector<int32_t> parentToNodes(opList.size(), -1);
    updated = false;
    node2Op.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(opList.size()); i++) {
        int32_t currParent = FindParent(parent, i);
        if (currParent == -1) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "Find parent in the union set failed.%s",
                GetFormatBacktrace(*(operationGraphInfo->opList_[i])).c_str());
            return FAILED;
        }
        if (currParent == i) {
            parentToNodes[i] = node2Op.size();
            node2Op.push_back(std::vector<int32_t>());
        }
    }
    for (int32_t i = 0; i < static_cast<int32_t>(operationGraphInfo->opList_.size()); i++) {
        int32_t currParent = FindParent(parent, i);
        if (currParent == -1) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "Find parent in the union set failed.%s",
                GetFormatBacktrace(*(operationGraphInfo->opList_[i])).c_str());
            return FAILED;
        }
        int32_t nodeIdx = parentToNodes[currParent];
        node2Op[nodeIdx].push_back(i);
    }
    for (size_t nodeIdx = 0; nodeIdx < node2Op.size(); nodeIdx++) {
        std::vector<int32_t> expandNode = NodeInnerExpand(operationGraphInfo, node2Op[nodeIdx]);
        if (expandNode.size() == node2Op[nodeIdx].size() || expandNode.empty()) {
            continue;
        }
        updated = true;
        for (size_t opIdx = 1; opIdx < expandNode.size(); opIdx++) {
            if (MergeSrcToDstIsland(operationGraphInfo, parent, expandNode[0], expandNode[opIdx]) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "Build the disjoint set failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status NodeGraphInfo::Build(
    const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
    const std::vector<std::pair<int32_t, int32_t>>& mergePair, bool markIsCube)
{
    std::vector<Operation*>& opList = operationGraphInfo->opList_;
    std::vector<int32_t> parent(opList.size());
    for (size_t i = 0; i < opList.size(); i++) {
        parent[i] = i;
    }
    for (auto& pr : mergePair) {
        if (MergeSrcToDstIsland(operationGraphInfo, parent, pr.first, pr.second) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Build the disjoint set failed.");
            return FAILED;
        }
    }
    bool updated = true;
    while (updated) {
        updated = false;
        if (AvoidLoop(operationGraphInfo, parent, node2Op_, updated) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Avoid loop in building node failed");
            return FAILED;
        }
    }
    BuildNodeMapping(operationGraphInfo);
    BuildInOutGraph(operationGraphInfo);
    SetNodeCoreTypeAndMergeable(operationGraphInfo, markIsCube);
    return SUCCESS;
}

bool NodeGraphInfo::GetNodeMergeable(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, int32_t nodeIdx)
{
    bool isMergeable =
        !(node2Op_[nodeIdx].size() == 1 &&
          operationGraphInfo->opList_[node2Op_[nodeIdx][0]]->GetOpcode() == Opcode::OP_RESHAPE &&
          ((nodeInGraph_[nodeIdx].size() > 1 && nodeOutGraph_[nodeIdx].size() > 1) ||
           (nodeInGraph_[nodeIdx].size() > 1 && nodeOutGraph_[nodeIdx].empty()) ||
           (nodeInGraph_[nodeIdx].empty() && nodeOutGraph_[nodeIdx].size() > 1)));
    for (auto opIdx : node2Op_[nodeIdx]) {
        if (operationGraphInfo->opList_[opIdx]->GetScopeId() != -1 && !operationGraphInfo->opList_[opIdx]->GetAllowCrossScopeMerge()) {
            isMergeable = false;
        }
    }
    return isMergeable;
}

Status NodeGraphInfo::BuildInOutGraph(const std::shared_ptr<OperationGraphInfo> operationGraphInfo)
{
    nodeInGraph_.assign(node2Op_.size(), std::set<int32_t>());
    nodeOutGraph_.assign(node2Op_.size(), std::set<int32_t>());
    nodeInGraphList_.assign(node2Op_.size(), std::vector<int32_t>());
    nodeOutGraphList_.assign(node2Op_.size(), std::vector<int32_t>());
    for (size_t i = 0; i < node2Op_.size(); i++) {
        std::vector<int32_t>& currNode = node2Op_[i];
        for (int32_t opIdx : currNode) {
            for (int32_t publisherOpIdx : operationGraphInfo->inGraph_[opIdx]) {
                int32_t publisherNodeIdx = op2Node_[publisherOpIdx];
                if (publisherNodeIdx != static_cast<int32_t>(i)) {
                    nodeInGraph_[i].insert(publisherNodeIdx);
                    nodeOutGraph_[publisherNodeIdx].insert(i);
                }
            }
        }
    }
    for (size_t i = 0; i < node2Op_.size(); i++) {
        nodeInGraphList_[i].insert(nodeInGraphList_[i].begin(), nodeInGraph_[i].begin(), nodeInGraph_[i].end());
        nodeOutGraphList_[i].insert(nodeOutGraphList_[i].begin(), nodeOutGraph_[i].begin(), nodeOutGraph_[i].end());
    }
    return SUCCESS;
}

void NodeGraphInfo::SetNodeCoreTypeAndMergeable(
    const std::shared_ptr<OperationGraphInfo> operationGraphInfo, bool markIsCube)
{
    nodeCoreType_.resize(node2Op_.size());
    nodeMergeable_.resize(node2Op_.size());
    for (size_t i = 0; i < node2Op_.size(); i++) {
        nodeCoreType_[i] = OpCoreType::AIV;
        for (int32_t opIdx : node2Op_[i]) {
            if (operationGraphInfo->opCoreType_[opIdx] != OpCoreType::ANY) {
                nodeCoreType_[i] = operationGraphInfo->opCoreType_[opIdx];
                break;
            }
        }
        nodeMergeable_[i] = GetNodeMergeable(operationGraphInfo, i);
        if (!markIsCube) {
            continue;
        }
        bool isCube = false;
        for (auto j : node2Op_[i]) {
            if (operationGraphInfo->opCoreType_[j] == OpCoreType::AIC) {
                isCube = true;
                break;
            }
        }
        for (auto j : node2Op_[i]) {
            operationGraphInfo->opList_[j]->SetAttribute(OpAttributeKey::isCube, isCube);
        }
    }
}

int32_t NodeGraphInfo::GetNodeCycle(int32_t nodeIdx) const
{
    if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(nodeCycles_.size())) {
        return 0;
    }
    return nodeCycles_[nodeIdx];
}

Status SuperNodeGraphBuilder::BuildOpGraph(const std::vector<Operation*>& opList)
{
    operationInfo_ = std::make_shared<OperationGraphInfo>();
    if (operationInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create OperationInfo failed.");
        return FAILED;
    }
    operationInfo_->opList_ = opList;
    operationInfo_->inGraph_.resize(opList.size());
    operationInfo_->outGraph_.resize(opList.size());
    operationInfo_->opHashList_.resize(opList.size());
    operationInfo_->opCoreType_.resize(opList.size());
    operationInfo_->useCVMixPartition_ = useCVMixPartition_;
    for (size_t i = 0; i < opList.size(); i++) {
        operationInfo_->magic2Idx_[opList[i]->GetOpMagic()] = i;
    }
    for (size_t i = 0; i < opList.size(); i++) {
        for (const auto& input : opList[i]->GetIOperands()) {
            for (const auto& parentOpPtr : input->GetProducers()) {
                if (operationInfo_->magic2Idx_.count(parentOpPtr->GetOpMagic()) == 0) {
                    continue;
                }
                int32_t operationInIdx = operationInfo_->magic2Idx_[parentOpPtr->GetOpMagic()];
                operationInfo_->inGraph_[i].insert(operationInIdx);
                operationInfo_->outGraph_[operationInIdx].insert(i);
            }
        }
    }
    for (size_t i = 0; i < opList.size(); i++) {
        operationInfo_->opHashList_[i] = operationInfo_->GetHash(opList[i]);
        operationInfo_->opCoreType_[i] = OpcodeManager::Inst().GetCoreType(opList[i]->GetOpcode());
    }
    return SUCCESS;
}

inline bool IsL0cToL1MoveOp(Operation* op)
{
    return (op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE) &&
           op->GetOOperands().size() > 0 && op->GetIOperands().size() > 0 &&
           op->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
           op->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1;
}

inline bool SuperNodeGraphBuilder::L1CopyInCombine(
    const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList, int32_t i,
    std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (IsL0cToL1MoveOp(opList[i])) {
        for (auto outNode : operationInfo->outGraph_[i]) {
            mergePair.emplace_back(outNode, i);
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Combine %d and %d(outNode) for L1 CopyIn in building SuperNode.",
                opList[i]->GetOpMagic(), opList[outNode]->GetOpMagic());
        }
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Combine %d and %d(inNode) for L1 CopyIn in building SuperNode.",
                opList[i]->GetOpMagic(), opList[inNode]->GetOpMagic());
        }
        return true;
    }
    if (opList[i]->GetOOperands().size() > 0 && opList[i]->GetIOperands().size() > 0 &&
        opList[i]->GetIOperands()[0]->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        (opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
         opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_BT ||
         opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_FIX_QUANT_PRE)) {
        for (auto outNode : operationInfo->outGraph_[i]) {
            mergePair.emplace_back(outNode, i);
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Combine %d and %d for L1 CopyIn in building SuperNode.", opList[i]->GetOpMagic(),
                opList[outNode]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::ConvertCombine(
    const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList, int32_t i,
    std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    const std::unordered_set<MemoryType> AICmem{
        MemoryType::MEM_L0C, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B};
    const std::unordered_set<MemoryType> AIVmem{MemoryType::MEM_UB};
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (opList[i]->GetOpcode() != Opcode::OP_CONVERT) {
        return false;
    }
    std::shared_ptr<ConvertOpAttribute> attr =
        std::static_pointer_cast<ConvertOpAttribute>(opList[i]->GetOpAttribute());
    if (attr == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "Convert Op %d has no ConvertOpAttribute.", opList[i]->GetOpMagic());
        return true;
    }
    std::pair<MemoryType, MemoryType> convertPath = attr->GetConvertPath();
    bool isAICtoAIV = (AICmem.count(convertPath.first) > 0 && AIVmem.count(convertPath.second) > 0);
    bool isAIVtoAIC = (AIVmem.count(convertPath.first) > 0 && AICmem.count(convertPath.second) > 0);
    if (isAICtoAIV || isAIVtoAIC) {
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
        }
        return true;
    }
    for (auto inNode : operationInfo->inGraph_[i]) {
        mergePair.emplace_back(inNode, i);
    }
    for (auto outNode : operationInfo->outGraph_[i]) {
        mergePair.emplace_back(outNode, i);
    }
    return true;
}

inline bool SuperNodeGraphBuilder::AssembleCombine(
    const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList, int32_t i,
    std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // assemble 特殊处理, assemble到local tensor，需要将这些assemble统一island
    if (opList[i]->GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (opList[i]->GetOOperands().empty()) {
            return false;
        }
        if (AssembleToCopyoutScene(opList[i])) {
            // 在GenerateMoveOp中需要转换为CopyOut的Assemble, 参考CopyOutCombine处理
            return CopyOutCombine(operationInfo, opList, i, mergePair, true);
        }
        // assmemble和其输入绑定
        if (operationInfo->inGraph_[i].size() > 0) {
            mergePair.emplace_back(i, *(operationInfo->inGraph_[i].begin()));
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Combine %d and %d for Assemble in building SuperNode.", opList[i]->GetOpMagic(),
                opList[*(operationInfo->inGraph_[i].begin())]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::CopyOutCombine(
    const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList, int32_t i,
    std::vector<std::pair<int32_t, int32_t>>& mergePair, bool assembleScene)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    std::vector<int32_t> candidateOpMagic;
    // 所有的copyout操作与其输入绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_OUT || assembleScene) {
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Combine %d and %d for CopyOut in building SuperNode.",
                opList[inNode]->GetOpMagic(), opList[i]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::CopyInCombine(
    const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList, int32_t i,
    std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // 所有的copyin操作与其输出绑定
    if ((OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_IN ||
         OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_LOCAL) &&
        operationInfo->outGraph_[i].size() > 0) {
        mergePair.emplace_back(i, *(operationInfo->outGraph_[i].begin()));
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Combine %d and %d for CopyIn in building SuperNode.", opList[i]->GetOpMagic(),
            opList[*(operationInfo->outGraph_[i].begin())]->GetOpMagic());
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::MulAccCombine(
    const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*>& opList, int32_t i,
    std::vector<std::pair<int32_t, int32_t>>& mergePair)
{
    if (i < 0 || i >= static_cast<int32_t>(opList.size())) {
        return false;
    }
    // MulAcc需要与其输入mul绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
        for (auto inOp : operationInfo->inGraph_[i]) {
            if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                mergePair.emplace_back(i, inOp);
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "Combine %d and %d for MulAcc in building SuperNode.", opList[i]->GetOpMagic(),
                    opList[inOp]->GetOpMagic());
            }
        }
        for (auto outOp : operationInfo->outGraph_[i]) {
            if (opList[outOp]->GetOpcode() == Opcode::OP_VIEW && !opList[outOp]->GetOOperands().empty() &&
                opList[outOp]->GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
                mergePair.emplace_back(i, outOp);
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "Combine MatMul %d and View %d in building SuperNode.",
                    opList[i]->GetOpMagic(), opList[outOp]->GetOpMagic());
            }
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::AssembleToCopyoutScene(Operation* op)
{
    auto assembleIn = op->iOperand.front();
    auto parentOp = *assembleIn->GetProducers().begin();
    if (op->iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR ||
        op->oOperand.front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        parentOp->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT || parentOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        return false;
    }
    return true;
}

inline void PropagateScopeInfo(std::vector<Operation*>& opList)
{
    for (size_t i = 0; i < opList.size(); i++) {
        int targetScope = opList[i]->GetScopeId();
        if (targetScope == DEFAULT_SCOPE_ID) {
            continue;
        }
        for (auto& consumer : opList[i]->ConsumerOps()) {
            if (consumer->GetScopeId() == -1 && consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                consumer->SetScopeInfo(opList[i]->GetScopeInfo());
            }
        }
        for (auto& producer : opList[i]->ProducerOps()) {
            if (producer->GetScopeId() == -1 && producer->GetOpcode() == Opcode::OP_VIEW) {
                producer->SetScopeInfo(opList[i]->GetScopeInfo());
            }
        }
    }
}

Status SuperNodeGraphBuilder::BuildSuperNodeGraph()
{
    std::vector<Operation*>& opList = operationInfo_->opList_;
    if (opList.size() != operationInfo_->inGraph_.size() || opList.size() != operationInfo_->outGraph_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation inGraph and outGraph have not been initialized.");
        return FAILED;
    }
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    PropagateScopeInfo(opList);
    for (size_t i = 0; i < opList.size(); i++) {
        if (ConvertCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (L1CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (AssembleCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (CopyOutCombine(operationInfo_, opList, i, mergePair, false)) {
            continue;
        }
        if (CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (MulAccCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
    }
    superNodeInfo_ = std::make_shared<NodeGraphInfo>();
    if (superNodeInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create SuperNodeInfo failed.");
        return FAILED;
    }
    if (superNodeInfo_->Build(operationInfo_, mergePair, true) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Build SuperNodeInfo Failed.");
        return FAILED;
    }
    return ProcessScopeMerge();
}

void NodeGraphInfo::BuildNodeMapping(const std::shared_ptr<OperationGraphInfo> operationGraphInfo)
{
    int32_t numNodes = static_cast<int32_t>(node2Op_.size());
    op2Node_.resize(operationGraphInfo->opList_.size());
    nodeScope_.assign(numNodes, Operation::ScopeInfo());
    nodeCycles_.assign(numNodes, 0);
    for (int32_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        for (int32_t opIdx : node2Op_[nodeIdx]) {
            op2Node_[opIdx] = nodeIdx;
            const auto& scopeInfo = operationGraphInfo->opList_[opIdx]->GetScopeInfo();
            if (scopeInfo.scopeId != -1) {
                nodeScope_[nodeIdx] = scopeInfo;
            }
            nodeCycles_[nodeIdx] += operationGraphInfo->opList_[opIdx]->GetLatency();
        }
    }
}

SuperNodeGraphBuilder::ScopeCollectResult SuperNodeGraphBuilder::CollectScopeInfo(int32_t numNodes)
{
    ScopeCollectResult result;
    for (int32_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        const auto& scopeInfo = superNodeInfo_->nodeScope_[nodeIdx];
        if (scopeInfo.scopeId == -1) {
            continue;
        }
        result.scope2Nodes[scopeInfo.scopeId].push_back(nodeIdx);
        for (int32_t opIdx : superNodeInfo_->node2Op_[nodeIdx]) {
            result.scopeCoreTypes[scopeInfo.scopeId].insert(operationInfo_->opCoreType_[opIdx]);
        }
        if (scopeInfo.allowParallelMerge) {
            result.scopeAllowParallel[scopeInfo.scopeId] = true;
        }
    }
    return result;
}

Status SuperNodeGraphBuilder::ValidateScopeCoreTypes(
    int32_t scopeId, const std::unordered_set<OpCoreType>& coreTypes, bool isCVMix,
    std::map<int32_t, int32_t>& scopeToCvFuseId)
{
    bool hasAic = coreTypes.count(OpCoreType::AIC) > 0;
    bool hasAiv = coreTypes.count(OpCoreType::AIV) > 0;
    if (!hasAic || !hasAiv) {
        return SUCCESS;
    }
    if (isCVMix) {
        scopeToCvFuseId[scopeId] = nextCvFuseId_++;
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(
        Elements::Function, "Cannot mix cube and vector op on a CV separate platform, scopeId=%d", scopeId);
    return FAILED;
}

Status SuperNodeGraphBuilder::CheckAndMergeScopes(
    const ScopeCollectResult& scopeInfo, std::vector<int32_t>& snParent, bool& needRebuild,
    std::map<int32_t, int32_t>& scopeToCvFuseId)
{
    bool isCVMix = GraphUtils::IsCVMixPlatform();
    for (auto& [scopeId, coreTypes] : scopeInfo.scopeCoreTypes) {
        if (ValidateScopeCoreTypes(scopeId, coreTypes, isCVMix, scopeToCvFuseId) != SUCCESS) {
            return FAILED;
        }
        bool allowParallel =
            scopeInfo.scopeAllowParallel.count(scopeId) > 0 && scopeInfo.scopeAllowParallel.at(scopeId);
        const auto& nodes = scopeInfo.scope2Nodes.at(scopeId);
        if (allowParallel) {
            int32_t firstNode = -1;
            int32_t p1 = -1;
            for (int32_t nodeIdx : nodes) {
                if (firstNode == -1) {
                    firstNode = nodeIdx;
                    p1 = FindParent(snParent, firstNode);
                } else {
                    int32_t p2 = FindParent(snParent, nodeIdx);
                    snParent[p2] = p1;
                    APASS_LOG_DEBUG_F(
                        Elements::Operation, "Combine %d and %d for ScopeMerge(parallel) scopeId=%d in building SuperNode.",
                        operationInfo_->opList_[superNodeInfo_->node2Op_[nodeIdx][0]]->GetOpMagic(),
                        operationInfo_->opList_[superNodeInfo_->node2Op_[firstNode][0]]->GetOpMagic(),
                        scopeId);
                    needRebuild = true;
                }
            }
            continue;
        }
        for (int32_t nodeIdx : nodes) {
            int32_t p1 = FindParent(snParent, nodeIdx);
            for (int32_t outNodeIdx : superNodeInfo_->nodeOutGraph_[nodeIdx]) {
                if (superNodeInfo_->nodeScope_[outNodeIdx].scopeId == scopeId) {
                    int32_t p2 = FindParent(snParent, outNodeIdx);
                    snParent[p2] = p1;
                    APASS_LOG_DEBUG_F(
                        Elements::Operation, "Combine %d and %d for ScopeMerge scopeId=%d in building SuperNode.",
                        operationInfo_->opList_[superNodeInfo_->node2Op_[outNodeIdx][0]]->GetOpMagic(),
                        operationInfo_->opList_[superNodeInfo_->node2Op_[nodeIdx][0]]->GetOpMagic(),
                        scopeId);
                    needRebuild = true;
                }
            }
        }
    }
    return SUCCESS;
}

void SuperNodeGraphBuilder::RebuildSuperNodes(std::vector<int32_t>& snParent, int32_t numNodes)
{
    std::vector<int32_t> parentToNewNode(numNodes, -1);
    std::vector<std::vector<int32_t>> newNode2Op;

    for (int32_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        int32_t p = FindParent(snParent, nodeIdx);
        if (parentToNewNode[p] == -1) {
            parentToNewNode[p] = static_cast<int32_t>(newNode2Op.size());
            newNode2Op.push_back({});
        }
        for (int32_t opIdx : superNodeInfo_->node2Op_[nodeIdx]) {
            newNode2Op[parentToNewNode[p]].push_back(opIdx);
        }
    }

    superNodeInfo_->node2Op_ = std::move(newNode2Op);
    superNodeInfo_->BuildNodeMapping(operationInfo_);
    superNodeInfo_->BuildInOutGraph(operationInfo_);
    superNodeInfo_->SetNodeCoreTypeAndMergeable(operationInfo_, false);
}

void SuperNodeGraphBuilder::ApplyCvFuseIds(
    const std::map<int32_t, int32_t>& scopeToCvFuseId, const std::map<int32_t, std::vector<int32_t>>& scope2Nodes)
{
    for (const auto& [scopeId, cvFuseId] : scopeToCvFuseId) {
        auto it = scope2Nodes.find(scopeId);
        if (it == scope2Nodes.end()) continue;
        for (int32_t nodeIdx : it->second) {
            for (int32_t opIdx : superNodeInfo_->node2Op_[nodeIdx]) {
                operationInfo_->opList_[opIdx]->scopeInfo_.SetCvFuseId(cvFuseId);
            }
        }
    }
}

Status SuperNodeGraphBuilder::ProcessScopeMerge()
{
    int32_t numNodes = static_cast<int32_t>(superNodeInfo_->node2Op_.size());
    auto scopeInfo = CollectScopeInfo(numNodes);

    std::vector<int32_t> snParent(numNodes);
    for (int32_t i = 0; i < numNodes; i++) {
        snParent[i] = i;
    }

    bool needRebuild = false;
    std::map<int32_t, int32_t> scopeToCvFuseId;
    Status ret = CheckAndMergeScopes(scopeInfo, snParent, needRebuild, scopeToCvFuseId);
    if (ret != SUCCESS) {
        return ret;
    }

    if (needRebuild) {
        RebuildSuperNodes(snParent, numNodes);
        scopeInfo = CollectScopeInfo(static_cast<int32_t>(superNodeInfo_->node2Op_.size()));
    }

    if (GraphUtils::IsCVMixPlatform()) {
        ApplyCvFuseIds(scopeToCvFuseId, scopeInfo.scope2Nodes);
    }
    return SUCCESS;
}

uint64_t SuperNodeGraphBuilder::CombineHash(const uint64_t h1, const uint64_t h2) const
{
    const uint64_t kMul = 0x9ddfea08eb382d69ULL;
    uint64_t a = (h1 ^ h2) * kMul;
    a ^= (a >> 47);
    uint64_t b = (h2 ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
}

std::vector<std::pair<int32_t, int32_t>> SuperNodeGraphBuilder::GetReduceNodeMergePair() const
{
    std::unordered_set<Opcode> reduceType{Opcode::OP_PAIRMAX, Opcode::OP_PAIRMIN, Opcode::OP_PAIRSUM};
    std::vector<Operation*>& opList = operationInfo_->opList_;
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    for (size_t i = 0; i < opList.size(); i++) {
        if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
            for (auto inOp : operationInfo_->inGraph_[i]) {
                if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                    mergePair.emplace_back(i, inOp);
                    APASS_LOG_DEBUG_F(
                        Elements::Operation, "Combine %d and %d for MulAcc in building ReduceNode.",
                        opList[i]->GetOpMagic(), opList[inOp]->GetOpMagic());
                }
            }
            continue;
        }
        if (reduceType.count(opList[i]->GetOpcode()) > 0 && operationInfo_->outGraph_[i].size() == 1 &&
            opList[i]->GetOpcode() == opList[*(operationInfo_->outGraph_[i].begin())]->GetOpcode()) {
            mergePair.emplace_back(i, *(operationInfo_->outGraph_[i].begin()));
            APASS_LOG_DEBUG_F(
                Elements::Operation, "Combine %d and %d for Reduce AIV Operation in building ReduceNode.",
                opList[i]->GetOpMagic(), opList[*(operationInfo_->outGraph_[i].begin())]->GetOpMagic());
        }
    }
    return mergePair;
}

Status SuperNodeGraphBuilder::BuildReduceNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo)
{
    std::vector<uint64_t> reduceNodeHashListFront(reduceNodeInfo->node2Op_.size(), 0);
    std::vector<uint64_t> reduceNodeHashListBack(reduceNodeInfo->node2Op_.size(), 0);
    std::vector<uint64_t> reduceNodeHashList(reduceNodeInfo->node2Op_.size(), 0);
    if (operationInfo_->opHashList_.size() != reduceNodeInfo->op2Node_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation number mismatch in OperationInfo and ReduceNodeInfo.");
        return FAILED;
    }
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        reduceNodeHashList[i] = 0;
        for (int32_t opInNode : reduceNodeInfo->node2Op_[i]) {
            reduceNodeHashList[i] = CombineHash(reduceNodeHashList[i], operationInfo_->opHashList_[opInNode]);
        }
    }

    for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
        int32_t nodeIdx = reduceNodeInfo->op2Node_[i];
        if (reduceNodeHashListFront[nodeIdx] != 0) {
            continue;
        }
        reduceNodeHashListFront[nodeIdx] = reduceNodeHashList[nodeIdx];
        for (int32_t j : reduceNodeInfo->nodeInGraph_[nodeIdx]) {
            reduceNodeHashListFront[nodeIdx] =
                CombineHash(reduceNodeHashListFront[nodeIdx], reduceNodeHashListFront[j]);
        }
    }

    for (int32_t i = static_cast<int32_t>(operationInfo_->opList_.size()) - 1; i >= 0; i--) {
        int32_t nodeIdx = reduceNodeInfo->op2Node_[i];
        if (reduceNodeHashListBack[nodeIdx] != 0) {
            continue;
        }
        reduceNodeHashListBack[nodeIdx] = reduceNodeHashList[nodeIdx];
        for (int32_t j : reduceNodeInfo->nodeOutGraph_[nodeIdx]) {
            reduceNodeHashListBack[nodeIdx] = CombineHash(reduceNodeHashListBack[nodeIdx], reduceNodeHashListBack[j]);
        }
    }

    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        reduceNodeHashList[i] = CombineHash(reduceNodeHashListFront[i], reduceNodeHashListBack[i]);
    }
    reduceNodeInfo->nodeHashList_ = reduceNodeHashList;
    return SUCCESS;
}

Status SuperNodeGraphBuilder::BuildBalanceOpHash(std::vector<uint64_t>& opHashList)
{
    std::vector<std::pair<int32_t, int32_t>> mergePair = GetReduceNodeMergePair();
    std::shared_ptr<NodeGraphInfo> reduceNodeInfo = std::make_shared<NodeGraphInfo>();
    if (reduceNodeInfo == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create ReduceNodeInfo failed.");
        return FAILED;
    }
    reduceNodeInfo->Build(operationInfo_, mergePair, false);
    BuildReduceNodeHash(reduceNodeInfo);
    std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        if (reduceNodeInfo->node2Op_[i].size() == 1) {
            opHashListFrontBack[reduceNodeInfo->node2Op_[i][0]] = reduceNodeInfo->nodeHashList_[i];
            continue;
        }
        std::vector<int32_t>& localOps = reduceNodeInfo->node2Op_[i];
        std::unordered_map<int32_t, uint64_t> localFront;
        std::unordered_map<int32_t, uint64_t> localBack;
        for (size_t localIdx = 0; localIdx < localOps.size(); localIdx++) {
            int32_t localOpIdx = localOps[localIdx];
            localFront[localOpIdx] = operationInfo_->opHashList_[localOpIdx];
            for (int32_t publisherOpIdx : operationInfo_->inGraph_[localOpIdx]) {
                if (localFront.count(publisherOpIdx) > 0) {
                    localFront[localOpIdx] = CombineHash(localFront[localOpIdx], localFront[publisherOpIdx]);
                }
            }
        }
        for (int32_t localIdx = static_cast<int32_t>(localOps.size()) - 1; localIdx >= 0; localIdx--) {
            int32_t localOpIdx = localOps[localIdx];
            localBack[localOpIdx] = operationInfo_->opHashList_[localOpIdx];
            for (int32_t consumerOpIdx : operationInfo_->outGraph_[localOpIdx]) {
                if (localBack.count(consumerOpIdx) > 0) {
                    localBack[localOpIdx] = CombineHash(localBack[localOpIdx], localBack[consumerOpIdx]);
                }
            }
        }
        for (size_t localIdx = 0; localIdx < localOps.size(); localIdx++) {
            int32_t localOpIdx = localOps[localIdx];
            uint64_t localHash = CombineHash(localFront[localOpIdx], localBack[localOpIdx]);
            opHashListFrontBack[localOpIdx] = CombineHash(reduceNodeInfo->nodeHashList_[i], localHash);
        }
    }
    opHashList = opHashListFrontBack;
    return SUCCESS;
}

Status SuperNodeGraphBuilder::BuildHashValues()
{
    std::vector<uint64_t> opHashList;
    if (useReduceBalanceHash_) {
        if (BuildBalanceOpHash(opHashList) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "BuildBalanceOpHash failed.");
            return FAILED;
        }
    } else {
        std::vector<uint64_t> opHashListFront(operationInfo_->opList_.size(), 0);
        std::vector<uint64_t> opHashListBack(operationInfo_->opList_.size(), 0);
        std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
        for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
            opHashListFront[i] = operationInfo_->opHashList_[i];
            for (int32_t j : operationInfo_->inGraph_[i]) {
                opHashListFront[i] = CombineHash(opHashListFront[i], opHashListFront[j]);
            }
        }
        for (int32_t i = static_cast<int32_t>(operationInfo_->opList_.size() - 1); i >= 0; i--) {
            opHashListBack[i] = operationInfo_->opHashList_[i];
            for (int32_t j : operationInfo_->outGraph_[i]) {
                std::set<OpCoreType> coreTypes{operationInfo_->opCoreType_[i], operationInfo_->opCoreType_[j]};
                if (!operationInfo_->CoreTypeMergeable(coreTypes)) {
                    continue;
                }
                opHashListBack[i] = CombineHash(opHashListBack[i], opHashListBack[j]);
            }
        }
        for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
            opHashListFrontBack[i] = CombineHash(opHashListFront[i], opHashListBack[i]);
        }
        opHashList.swap(opHashListFrontBack);
    }
    if (superNodeInfo_->op2Node_.size() != operationInfo_->opList_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation number mismatch in SuperNodeInfo and OperationInfo.");
        return FAILED;
    }
    int32_t numNode = superNodeInfo_->node2Op_.size();
    superNodeInfo_->nodeHashList_.resize(numNode);
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->nodeHashList_[i] = 0;
        for (int32_t opIdx : superNodeInfo_->node2Op_[i]) {
            superNodeInfo_->nodeHashList_[i] = CombineHash(superNodeInfo_->nodeHashList_[i], opHashList[opIdx]);
        }
    }
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[i]].push_back(i);
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
