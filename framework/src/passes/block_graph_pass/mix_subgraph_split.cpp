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
 * \file mix_subgraph_split.cpp
 * \brief
 */

#include "passes/block_graph_pass/mix_subgraph_split.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/utils/id_gen.h"

namespace npu {
namespace tile_fwk {
RawSymbolicScalarPtr ReplaceExpression(const RawSymbolicScalarPtr &expr,
                                        const RawSymbolicScalarPtr &src,
                                        const RawSymbolicScalarPtr &dst);

int GetStartIndex(const std::vector<Operation *> &opList, Operation* startOp) {
    // 找到起始op的索引
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] == startOp) {
            return i;
        }
    }
    ALOG_DEBUG_F("Start op %d not found in sequence", startOp->GetOpMagic());
    return -1;
}

Status MixSubgraphSplit::GatherSubGraphInfo(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::set<uint64_t> &mixSubgraphIDsToDelete, std::vector<Operation*> &callOpsToDelete) {
    // 获取所有CallOps
    auto rootFunc = function.rootFunc_;
    auto callOps = rootFunc->GetCallopList();
    std::unordered_map<uint64_t, std::vector<Operation*>> programIDToCallOps;
    // 构建programID到callOps的映射
    for (auto* callOp : callOps) {
        if (callOp == nullptr || callOp->IsDeleted()) {
            continue;
        }
        auto callAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
        if (callAttr != nullptr && callAttr->invokeInfo_ != nullptr) {
            uint64_t programID = callAttr->invokeInfo_->GetProgramId();
            programIDToCallOps[programID].push_back(callOp);
        }
    }
    for (auto &program : rootFunc->programs_) {
        if (program.second != nullptr && IsMixSubgraph(*program.second)) {
            auto components = AnalyzeInternalComponents(*program.second);
            if (components.size() > 1) {
                // 获取该programID对应的原始callOps
                std::vector<Operation*> originalCallOps;
                auto it = programIDToCallOps.find(program.first);
                if (it != programIDToCallOps.end()) {
                    originalCallOps = it->second;
                    callOpsToDelete.insert(callOpsToDelete.end(),
                                            originalCallOps.begin(), originalCallOps.end());
                }
                mixSubgraphs.push_back({program.first, program.second, components, originalCallOps});
                mixSubgraphIDsToDelete.insert(program.first);
                ALOG_INFO_F("Found mix subgraph: programID=%d, components=%d", program.first, components.size());
            }
        }
    }
    return SUCCESS;
}

Status MixSubgraphSplit::CalculateSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::set<uint64_t> &mixSubgraphIDsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap) {
    // 计算最终的programID分配
    auto rootFunc = function.rootFunc_;
    size_t originalCount = rootFunc->programs_.size();
    size_t deleteCount = mixSubgraphIDsToDelete.size();
    size_t newSubgraphCount = 0;
    for (const auto& mixInfo : mixSubgraphs) {
        newSubgraphCount += mixInfo.components.size();
    }
    size_t finalCount = originalCount - deleteCount + newSubgraphCount;
    ALOG_INFO_F("Program count: original= %d, delete=%d, new=%d, final=%d", originalCount, deleteCount, newSubgraphCount, finalCount);
    // 构建programID重映射表
    uint64_t nextProgramID = 0; // 从0开始重新分配连续ID

    // 首先映射保留的子图（非Mix子图）
    for (auto &program : rootFunc->programs_) {
        if (mixSubgraphIDsToDelete.find(program.first) == mixSubgraphIDsToDelete.end()) {
            programIDRemap[program.first] = nextProgramID++;
            ALOG_DEBUG_F("Remap preserved program: %d ->  %d", program.first, programIDRemap[program.first]);
        }
    }
    // 然后为新创建的子图分配连续的ID
    for (const auto& mixInfo : mixSubgraphs) {
        std::vector<uint64_t> newProgramIDs;
        for (size_t i = 0; i < mixInfo.components.size(); ++i) {
            newProgramIDs.push_back(nextProgramID++);
        }
        mixSubgraphNewIDs[mixInfo.programID] = newProgramIDs;
        ALOG_INFO_F("Allocated new programIDs for mix subgraph %d.", mixInfo.programID);
    }
    return SUCCESS;
}

Status MixSubgraphSplit::ExecuteSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::vector<Operation*> callOpsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap) {
    // 执行实际的拆分
    std::vector<MixSubgraphSplitResult> splitResults;
    auto rootFunc = function.rootFunc_;
    for (const auto& mixInfo : mixSubgraphs) {
        auto newProgramIDs = mixSubgraphNewIDs[mixInfo.programID];
        auto status = ProcessLeafFunction(*rootFunc, mixInfo.programID, mixInfo.function, mixInfo.components, newProgramIDs, mixInfo.originalCallOps, splitResults);
        if (status != SUCCESS) {
            ALOG_ERROR_F("ProcessLeafFunction failed for programID=%d.", mixInfo.programID);
            return status;
        }
    }

    // 删除原始Mix子图的callOp
    DeleteOriginalMixCallOps(*rootFunc, callOpsToDelete);
    ALOG_INFO_F("Found %d mix subgraphs to split", mixSubgraphs.size());

    // 应用拆分结果并重新映射所有programID
    auto status = ApplySplitResultsWithRemap(function, splitResults, programIDRemap, mixSubgraphNewIDs);
    if (status != SUCCESS) {
        ALOG_ERROR_F("ApplySplitResultsWithRemap failed.");
        return status;
    }
    return SUCCESS;
}

Status MixSubgraphSplit::RunOnFunction(Function &function) {
    ALOG_INFO_F("===============================================================> Start MixSubgraphSplit.");
    // 获取rootFunc和programs
    auto rootFunc = function.rootFunc_;
    if (rootFunc == nullptr) {
        ALOG_ERROR_F("Get root function failed.");
        return FAILED;
    }

    auto& programs = rootFunc->programs_;
    if (programs.empty()) {
        ALOG_INFO_F("No leaf function found, jump MixSubgraphSplit.");
        return SUCCESS;
    }

    // 收集所有需要拆分的Mix子图及其内部组件信息
    std::vector<MixSubgraphInfo> mixSubgraphs;
    std::set<uint64_t> mixSubgraphIDsToDelete;
    std::vector<Operation*> callOpsToDelete;
    if (GatherSubGraphInfo(function, mixSubgraphs, mixSubgraphIDsToDelete, callOpsToDelete) != SUCCESS) {
        ALOG_ERROR_F("GatherSubGraphInfo failed");
        return FAILED;
    }
    ALOG_INFO_F("Found %d leaf function to process", programs.size());

    if (mixSubgraphs.empty()) {
        ALOG_INFO_F("No mix subgraph found, jump MixSubgraphSplit.");
        return SUCCESS;
    }
    std::unordered_map<uint64_t, std::vector<uint64_t>> mixSubgraphNewIDs;
    std::unordered_map<uint64_t, uint64_t> programIDRemap;
    if (CalculateSplit(function, mixSubgraphs, mixSubgraphIDsToDelete, mixSubgraphNewIDs, programIDRemap) != SUCCESS) {
        ALOG_ERROR_F("CalculateSplit failed");
        return FAILED;
    }
    if (ExecuteSplit(function, mixSubgraphs, callOpsToDelete, mixSubgraphNewIDs, programIDRemap) != SUCCESS) {
        ALOG_ERROR_F("ExecuteSplit failed");
        return FAILED;
    }
    ALOG_INFO_F("===============================================================> Finish MixSubgraphSplit.");
    return SUCCESS;
}

bool MixSubgraphSplit::IsMixSubgraph(Function& function) const {
    auto operations = function.Operations(false);
    for (size_t idx = 0; idx < operations.size(); idx++) {
        auto& op = operations[idx];
        if (op.IsNOP()) continue;
        // 只要有一个op有有效的internalSubgraphID，就认为是Mix子图
        int internalSubgraphID = op.GetInternalSubgraphID();
        if (internalSubgraphID > 0) {
            ALOG_DEBUG_F("Function %s identified as mix subgraph: op %s has internalSubgraphID=%d",
                        function.GetRawName().c_str(),
                        op.GetOpcodeStr().c_str(),
                        internalSubgraphID);
            return true;
        }
    }
    ALOG_DEBUG_F("Function %s is not a mix subgraph: no ops with internalSubgraphID",
            function.GetRawName().c_str());
    return false;
}

std::map<int, std::vector<Operation*>> MixSubgraphSplit::GroupOperationsByExistingInternalID(Function& mixSubgraphFunc,
                                                                                                    std::vector<Operation*>& unassignedOps) const {
    std::map<int, std::vector<Operation*>> internalIDToOperations;
    auto operationViewer = mixSubgraphFunc.Operations(false);
    for (size_t idx = 0; idx < operationViewer.size(); idx++) {
        auto& op = operationViewer[idx];
        if (op.IsNOP()) continue;
        int internalSubgraphID = op.GetInternalSubgraphID();
        if (internalSubgraphID >= 0) {
            internalIDToOperations[internalSubgraphID].push_back(&op);
            ALOG_DEBUG_F("Operation %s assigned to internalID=%d",
                        op.GetOpcodeStr().c_str(), internalSubgraphID);
        } else {
            // 没有有效的internalSubgraphID，收集到未分配列表
            unassignedOps.push_back(&op);
        }
    }
    ALOG_INFO_F("Grouped operations by existing internalSubgraphID into %d groups, %d unassigned", internalIDToOperations.size(), unassignedOps.size());
    return internalIDToOperations;
}

void MixSubgraphSplit::ProcessUnassignedOperations(
    std::vector<Operation*>& unassignedOps,
    std::map<int, std::vector<Operation*>>& componentsByInternalID,
    Function& mixSubgraphFunc) const {
    // 预先构建op到组件的映射表，并分析现有组件的coreType
    std::unordered_map<Operation*, int> opToComponentMap;
    for (const auto& [internalID, operations] : componentsByInternalID) {
        for (auto* op : operations) {
            opToComponentMap[op] = internalID;
        }
    }
    std::vector<Operation*> remainingOps;
    // 处理数据移动op(基于OpCalcType)和同步op
    for (auto* op : unassignedOps) {
        bool merged = false;
        // 通过OpcodeManager获取OpCalcType
        OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(op->GetOpcode());
        ALOG_DEBUG_F("Analyzing operation %s with calcType=%d",
                    op->GetOpcodeStr().c_str(), static_cast<int>(opCalcType));
        if (opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN) {
            // 应该与输出tensor的消费者在同一个组件
            merged = MergeMoveInOperation(op, componentsByInternalID, opToComponentMap);
        }
        else if (opCalcType == OpCalcType::MOVE_OUT) {
            // 应该与输入tensor的生产者在同一个组件
            merged = MergeMoveOutOperation(op, componentsByInternalID, opToComponentMap);
        }
        // 处理同步节点
        else if (IsSyncOperation(op)) {
            merged = MergeSyncOperation(op, componentsByInternalID, opToComponentMap, mixSubgraphFunc);
        }
        if (!merged) {
            remainingOps.push_back(op);
            ALOG_DEBUG_F("Operation %s (calcType=%d) not merged by data movement pattern",
                       op->GetOpcodeStr().c_str(), static_cast<int>(opCalcType));
        }
    }
    if (!remainingOps.empty()) {
        ALOG_ERROR_F("Found %d unexpected unassigned operations after first step:", remainingOps.size());
    }
}

// 合并MOVE_IN类型的op
bool MixSubgraphSplit::MergeMoveInOperation(
    Operation* op,
    std::map<int, std::vector<Operation*>>& componentsByInternalID,
    std::unordered_map<Operation*, int>& opToComponentMap) const {
    ALOG_DEBUG_F("Processing MOVE_IN operation %s", op->GetOpcodeStr().c_str());
    // 主要策略：与输出tensor的消费者合并
    for (auto& outputTensor : op->GetOOperands()) {
        for (auto* consumer : outputTensor->GetConsumers()) {
            auto it = opToComponentMap.find(consumer);
            if (it != opToComponentMap.end()) {
                int targetComponent = it->second;
                op->UpdateInternalSubgraphID(targetComponent);
                componentsByInternalID[targetComponent].push_back(op);
                opToComponentMap[op] = targetComponent;
                ALOG_DEBUG_F("Merged MOVE_IN operation %s to component %d (serves consumer %s)",
                            op->GetOpcodeStr().c_str(), targetComponent,
                            consumer->GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

// 合并MOVE_OUT类型的op
bool MixSubgraphSplit::MergeMoveOutOperation(
    Operation* op,
    std::map<int, std::vector<Operation*>>& componentsByInternalID,
    std::unordered_map<Operation*, int>& opToComponentMap) const {
    ALOG_DEBUG_F("Processing MOVE_OUT operation %s", op->GetOpcodeStr().c_str());
    // 主要策略：与输入tensor的生产者合并
    for (auto& inputTensor : op->GetIOperands()) {
        for (auto* producer : inputTensor->GetProducers()) {
            auto it = opToComponentMap.find(producer);
            if (it != opToComponentMap.end()) {
                int targetComponent = it->second;
                op->UpdateInternalSubgraphID(targetComponent);
                componentsByInternalID[targetComponent].push_back(op);
                opToComponentMap[op] = targetComponent;
                ALOG_DEBUG_F("Merged MOVE_OUT operation %s to component %d (serves producer %s)",
                           op->GetOpcodeStr().c_str(), targetComponent,
                           producer->GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

Operation* MixSubgraphSplit::FindPreviousOpInSequence(Operation* op, Function& mixSubgraphFunc) const {
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] == op && i > 0) {
            return opList[i - 1];
        }
    }
    return nullptr;
}

Operation* MixSubgraphSplit::FindNextOpInSequence(Operation* op, Function& mixSubgraphFunc) const {
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] == op && i + 1 < opList.size()) {
            return opList[i + 1];
        }
    }
    return nullptr;
}

bool MixSubgraphSplit::MergeSyncPhase2(Operation* op, Function& mixSubgraphFunc, std::map<int, std::vector<Operation*>>& componentsByInternalID, std::unordered_map<Operation*, int>& opToComponentMap) const {
    Operation* targetOp = FindFirstOpBackward(op, mixSubgraphFunc,
        [](Operation* candidate) {
            return candidate->GetOpcode() == Opcode::OP_COPY_IN;
        });
    if (targetOp) {
        auto it = opToComponentMap.find(targetOp);
        if (it != opToComponentMap.end()) {
            op->UpdateInternalSubgraphID(it->second);
            componentsByInternalID[it->second].push_back(op);
            opToComponentMap[op] = it->second;
            ALOG_DEBUG_F("Merged PHASE2 %d to component %d via COPY_IN %d",
                        op->GetOpMagic(), it->second, targetOp->GetOpMagic());
            return true;
        }
    }
    ALOG_WARN_F("Failed to merge PHASE2 %d: no COPY_IN found backward", op->GetOpMagic());
    return false;
}

bool MixSubgraphSplit::MergeSyncPhase1(Operation* op, Function& mixSubgraphFunc, std::map<int, std::vector<Operation*>>& componentsByInternalID, std::unordered_map<Operation*, int>& opToComponentMap) const {
    Operation* targetOp = FindFirstOpForward(op, mixSubgraphFunc,
        [](Operation* candidate) {
            return candidate->GetOpcode() == Opcode::OP_COPY_IN;
        });
    if (targetOp) {
        auto it = opToComponentMap.find(targetOp);
        if (it != opToComponentMap.end()) {
            op->UpdateInternalSubgraphID(it->second);
            componentsByInternalID[it->second].push_back(op);
            opToComponentMap[op] = it->second;
            ALOG_DEBUG_F("Merged PHASE1 %d to component %d via COPY_IN %d",
                        op->GetOpMagic(), it->second, targetOp->GetOpMagic());
            return true;
        }
    }
    ALOG_WARN_F("Failed to merge PHASE1 %d: no COPY_IN found forward", op->GetOpMagic());
    return false;
}

bool MixSubgraphSplit::MergeSyncSrcDst(Operation* op, Operation* targetOp, std::map<int, std::vector<Operation*>>& componentsByInternalID, std::unordered_map<Operation*, int>& opToComponentMap) const {
    if (targetOp) {
        auto it = opToComponentMap.find(targetOp);
        if (it != opToComponentMap.end()) {
            op->UpdateInternalSubgraphID(it->second);
            componentsByInternalID[it->second].push_back(op);
            opToComponentMap[op] = it->second;
            ALOG_DEBUG_F("Merged %s %d to component %d via non-sync op %d",
                        op->GetOpcodeStr().c_str(), op->GetOpMagic(), it->second, targetOp->GetOpMagic());
            return true;
        }
    }
    ALOG_WARN_F("Failed to merge %s %d: no non-sync op found backward", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return false;
}

bool MixSubgraphSplit::MergeSyncOperation(Operation* op, std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                        std::unordered_map<Operation*, int>& opToComponentMap, Function& mixSubgraphFunc) const {
    Opcode opcode = op->GetOpcode();
    // OP_PHASE2: 往前找到第一个COPY_IN放到COPY_IN所在组
    if (opcode == Opcode::OP_PHASE2) {
        return MergeSyncPhase2(op, mixSubgraphFunc, componentsByInternalID, opToComponentMap);
    }

    // OP_PHASE1: 往后找到第一个COPY_IN放到COPY_IN所在组
    if (opcode == Opcode::OP_PHASE1) {
        return MergeSyncPhase1(op, mixSubgraphFunc, componentsByInternalID, opToComponentMap);
    }

    // OP_SYNC_SRC、OP_CV_SYNC_SRC: 往前找到第一个非同步op放到该op所在分组
    if (opcode == Opcode::OP_SYNC_SRC || opcode == Opcode::OP_CV_SYNC_SRC) {
        Operation* targetSrcOp = FindFirstOpBackward(op, mixSubgraphFunc, [this](Operation* candidate) { return !IsSyncOperation(candidate); });
        return MergeSyncSrcDst(op, targetSrcOp, componentsByInternalID, opToComponentMap);
    }

    // BAR类、OP_SYNC_DST、OP_CV_SYNC_DST: 往后找到第一个非同步op放到该op所在分组
    if (opcode == Opcode::OP_BAR_V || opcode == Opcode::OP_BAR_M ||
        opcode == Opcode::OP_BAR_ALL || opcode == Opcode::OP_SYNC_DST ||
        opcode == Opcode::OP_CV_SYNC_DST) {
        Operation* targetDstOp = FindFirstOpForward(op, mixSubgraphFunc, [this](Operation* candidate) { return !IsSyncOperation(candidate); });
        return MergeSyncSrcDst(op, targetDstOp, componentsByInternalID, opToComponentMap);
    }

    ALOG_ERROR_F("Unhandled sync operation type: %s %d",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
    return false;
}

Operation* MixSubgraphSplit::FindFirstOpForward(Operation* startOp, Function& mixSubgraphFunc, std::function<bool(Operation*)> predicate) const {
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    int startIndex = GetStartIndex(opList, startOp);
    if (startIndex == -1) {
        return nullptr;
    }

    // 向后搜索（向序列结束方向）
    for (int i = startIndex + 1; i < static_cast<int>(opList.size()); ++i) {
        Operation* candidate = opList[i];
        if (predicate(candidate)) {
            ALOG_DEBUG_F("Found target op %d at index %d (searching forward from %d)", candidate->GetOpMagic(), i, startIndex);
            return candidate;
        }
    }

    ALOG_DEBUG_F("No matching op found for op %d in forward direction", startOp->GetOpMagic());
    return nullptr;
}

Operation* MixSubgraphSplit::FindFirstOpBackward(Operation* startOp, Function& mixSubgraphFunc, std::function<bool(Operation*)> predicate) const {
    const auto& opList = mixSubgraphFunc.Operations(false).DuplicatedOpList();
    int startIndex = GetStartIndex(opList, startOp);
    if (startIndex == -1) {
        return nullptr;
    }

    // 向前搜索（向序列开始方向）
    for (int i = startIndex - 1; i >= 0; --i) {
        Operation* candidate = opList[i];
        if (predicate(candidate)) {
            ALOG_DEBUG_F("Found target op %d at index %d (searching backward from %d)", candidate->GetOpMagic(), i, startIndex);
            return candidate;
        }
    }

    ALOG_DEBUG_F("No matching op found for op %d in backward direction", startOp->GetOpMagic());
    return nullptr;
}

bool MixSubgraphSplit::IsInUnassignedOps(Operation* op, const std::vector<Operation*>& unassignedOps) const {
    return std::find(unassignedOps.begin(), unassignedOps.end(), op) != unassignedOps.end();
}

bool MixSubgraphSplit::IsSyncOperation(Operation* op) const {
    if (!op) {
        return false;
    }

    Opcode opcode = op->GetOpcode();

    // 同步操作类型列表
    return opcode == Opcode::OP_SYNC_SRC ||
           opcode == Opcode::OP_SYNC_DST ||
           opcode == Opcode::OP_CV_SYNC_SRC ||
           opcode == Opcode::OP_CV_SYNC_DST ||
           opcode == Opcode::OP_PHASE1 ||
           opcode == Opcode::OP_PHASE2 ||
           opcode == Opcode::OP_BAR_V ||
           opcode == Opcode::OP_BAR_M ||
           opcode == Opcode::OP_BAR_ALL;
}

AIVCore MixSubgraphSplit::FindConsumerVectorAIVCore(Operation* copyOp) const {
    if (copyOp == nullptr) {
        return AIVCore::UNSPECIFIED;
    }
    auto outputTensors = copyOp->GetOOperands();
    for (auto tensor : outputTensors) {
        if (tensor == nullptr) {
            continue;
        }
        for (auto* consumer : tensor->GetConsumers()) {
            if (consumer == nullptr) {
                continue;
            }
            AIVCore consumerCore = consumer->GetAIVCore();
            if (consumerCore != AIVCore::UNSPECIFIED) {
                ALOG_DEBUG_F("Found consumer op %s with AIVCore=%d",
                            consumer->GetOpcodeStr().c_str(), static_cast<int>(consumerCore));
                return consumerCore;
            }
        }
    }
    return AIVCore::UNSPECIFIED;
}

AIVCore MixSubgraphSplit::DetermineComponentAIVCore(const std::vector<Operation*>& operations) const
{
    if (operations.empty()) {
        return AIVCore::UNSPECIFIED;
    }

    // 检查第一个操作的isCube属性来确定组件类型
    bool isCubeComponent = operations[0]->HasAttr(OpAttributeKey::isCube) &&
                          operations[0]->GetAttr<bool>(OpAttributeKey::isCube);
    if (isCubeComponent) {
        // Cube组件：分析L0C_COPY_UB op的目标vector子图的AIVCore
        AIVCore targetAIVCore = AIVCore::UNSPECIFIED;
        for (auto* op : operations) {
            if (op->GetOpcode() == Opcode::OP_L0C_COPY_UB) {
                targetAIVCore = FindConsumerVectorAIVCore(op);
                if (targetAIVCore != AIVCore::UNSPECIFIED) {
                    int64_t subBlockIdx = (targetAIVCore == AIVCore::AIV0) ? 0 : 1;
                    op->SetAttr(OpAttributeKey::subBlockIdx, subBlockIdx);
                    ALOG_DEBUG_F("Set SUB_BLOCK_IDX=%ld for L0C_COPY_UB op %d", subBlockIdx, op->GetOpMagic());
                }
            }
        }
        ALOG_DEBUG_F("Component AIC determined by op %s", operations[0]->GetOpcodeStr().c_str());
        return AIVCore::UNSPECIFIED;  // Cube组件
    }

    // Vector组件：查找第一个非同步op的AIVCore
    for (auto* op : operations) {
        if (!IsSyncOperation(op)) {
            AIVCore opCore = op->GetAIVCore();
            if (opCore != AIVCore::UNSPECIFIED) {
                // 直接返回第一个非同步op的AIVCore
                ALOG_DEBUG_F("Component AIVCore determined by op %s: AIV%d",
                           op->GetOpcodeStr().c_str(),
                           (opCore == AIVCore::AIV0 ? 0 : 1));
                return opCore;
            }
        }
    }
    ALOG_ERROR_F("Cannot determine AIVCore for Vector component %d: all ops are sync or no AIVCore set",
                operations[0]->GetInternalSubgraphID());
    return AIVCore::UNSPECIFIED;
}

std::vector<InternalComponentInfo> MixSubgraphSplit::AnalyzeInternalComponents(Function& mixSubgraphFunc) const {
    std::vector<InternalComponentInfo> internalComponents;
    // 首先尝试使用现有的internalSubgraphID, 同时收集未分配的op
    std::vector<Operation*> unassignedOps;
    auto componentsByInternalID = GroupOperationsByExistingInternalID(mixSubgraphFunc, unassignedOps);
    // 如果没有有效的internalSubgraphID，使用启发式方法推导
    if (!unassignedOps.empty()) {
        ALOG_INFO_F("Found %d operations without internalSubgraphID, using heuristic analysis",
                   unassignedOps.size());
        ProcessUnassignedOperations(unassignedOps, componentsByInternalID, mixSubgraphFunc);
    }
    // 构建最终的InternalComponentInfo
    for (const auto& [internalID, operations] : componentsByInternalID) {
        std::string suffix = "_internal_" + std::to_string(internalID);
        // 确定AIVCore，UNSPECIFIED表示Cube或其他类型
        AIVCore aivCore = DetermineComponentAIVCore(operations);
        internalComponents.emplace_back(internalID, suffix, aivCore);
        internalComponents.back().operations = operations;
        ALOG_DEBUG_F("Internal component: internalSubgraphID=%d, operationCount=%d.", internalID, operations.size());
    }
    ALOG_INFO_F("Analyzed %d components", internalComponents.size());
    return internalComponents;
}

MixResourceType MixSubgraphSplit::GetMixResourceType(Function& mixFunc) const {
    bool hasAIV0 = false;
    bool hasAIV1 = false;
    auto operations = mixFunc.Operations(false);
    for (size_t idx = 0; idx < operations.size(); idx++) {
        auto& op = operations[idx];
        if (op.IsNOP()) continue;
        auto aivCore = op.GetAIVCore();
        if (aivCore == AIVCore::AIV0) {
            hasAIV0 = true;
        } else if (aivCore == AIVCore::AIV1) {
            hasAIV1 = true;
        }
        if (hasAIV0 && hasAIV1) {
            return MixResourceType::ONE_CUBE_TWO_VECTOR;
        }
    }
    if (hasAIV0 || hasAIV1) {
        return MixResourceType::ONE_CUBE_ONE_VECTOR;
    }
    return MixResourceType::UNKNOWN;
}

Status MixSubgraphSplit::ApplySplitResultsWithRemap(Function& function,
                                                    const std::vector<MixSubgraphSplitResult>& splitResults,
                                                    const std::unordered_map<uint64_t, uint64_t>& programIDRemap,
                                                    const std::unordered_map<uint64_t, std::vector<uint64_t>>& mixSubgraphNewIDs) {
    auto rootFunc = function.rootFunc_;
    if (rootFunc == nullptr) {
        return FAILED;
    }
    size_t originalCount = rootFunc->programs_.size();
    // 构建新的programs映射
    std::map<uint64_t, Function*> newPrograms;
    //添加保留的子图
    for (auto &program : rootFunc->programs_) {
        // 跳过要删除的Mix子图
        if (programIDRemap.find(program.first) != programIDRemap.end()) {
            uint64_t newID = programIDRemap.at(program.first);
            newPrograms[newID] = program.second;
            // 更新function的programID
            if (program.second != nullptr) {
                program.second->SetProgramId(newID);
                ALOG_DEBUG_F("Updated preserved program: oldID=%d -> newID=%d", program.first, newID);
            }
        }
    }
    // 添加新创建的子图
    for (const auto& result : splitResults) {
        auto newProgramIDs = mixSubgraphNewIDs.at(result.originalProgramID);
        for (size_t i = 0; i < result.newFunctions.size(); i++) {
            uint64_t newProgramID = newProgramIDs[i];
            Function* newFunc = result.newFunctions[i];
            if (newFunc != nullptr) {
                newPrograms[newProgramID] = newFunc;
                newFunc->SetProgramId(newProgramID);
                ALOG_DEBUG_F("Added new subgraph: programID=%d, function=%s",
                        newProgramID, newFunc->GetRawName().c_str());
            }
        }
    }
    // 更新rootFunc的programs
    rootFunc->programs_ = std::move(newPrograms);
    ALOG_INFO_F("Program mapping completed: original count=%d, new count=%d",
            originalCount, rootFunc->programs_.size());
    return SUCCESS;
}

Status MixSubgraphSplit::CreateCallOpInRootFunction(Function& rootFunc,
                                                    Function& leafFunc,
                                                    uint64_t newProgramID,
                                                    uint64_t componentIndex,
                                                    Operation* originalCallOp,
                                                    Function* originalMixFunc,
                                                    SubgraphToFunction& subgraphToFunction,
                                                    uint64_t wrapId,
                                                    std::vector<int>& iOffsets,
                                                    std::vector<int>& oOffsets) {
    ALOG_DEBUG_F("Creating callOp in root function for leaf: %s, programID=%d, component=%d, wrapId=%lu", leafFunc.GetRawName().c_str(), newProgramID, componentIndex, wrapId);
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
    // 1. 为incast构建新的iOperands
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
    // 2. 为global tensor输入构建新的iOperands
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
    // 3. 为outcast构建新的oOperands
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
    // 4. 为global tensor输出构建新的oOperands
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
    // 5. 处理传播依赖添加的参数
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();
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
    auto& callOp = rootFunc.AddRawOperation(Opcode::OP_CALL, newIOperands, newOOperands, false);
    ALOG_INFO_F("Created operands for new callOp %d: %zu inputs, %zu outputs",
            callOp.GetOpMagic(), newIOperands.size(), newOOperands.size());
    // 使用invokeInfo提取argList
    auto extractedArgList = ExtractArgListForLeafFunction(leafFunc, originalCallAttr, invokeInfo, iOffsets, oOffsets);
    ALOG_DEBUG_F("Created callOp %d: %zu arg blocks (from original callOp %d), %zu input offsets, %zu output offsets", callOp.GetOpMagic(), extractedArgList.size(), originalCallOp->GetOpMagic(), iOffsets.size(), oOffsets.size());
    std::map<int, SymbolicScalar> outIndexToExpr;
    leafFunc.GetOutcastSymbolicExpr(outIndexToExpr);
    // 创建CallOpAttribute（使用从原始CallOp提取的argList）
    auto callAttr = leafFunc.CreateCallOpAttribute(extractedArgList, outIndexToExpr);
    auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(callAttr);
    if (callOpAttr != nullptr) {
        callOpAttr->wrapId = wrapId;
        ALOG_DEBUG_F("Set wrapId=%lu to callOp attribute for programID=%d (from original callOp %d)", wrapId, newProgramID, originalCallOp->GetOpMagic());
    }
    callOp.SetOpAttribute(callAttr);
    callOp.SetOpOffset(iOffsets, oOffsets);
    callOp.UpdateSubgraphID(newProgramID);
    if (componentIndex < subgraphToFunction.subFuncInvokeInfos.size()) {
        callOp.SetSubFuncInvokeInfo(subgraphToFunction.subFuncInvokeInfos[componentIndex]);
    }
    subgraphToFunction.SetSemanticLabel(leafFunc.GetProgramOp(), callOp);
    if (callOpAttr != nullptr && callOpAttr->invokeInfo_ != nullptr) {
        callOpAttr->invokeInfo_->UpdateProgramSubgraphId(newProgramID);
    }
    ALOG_INFO_F("Successfully created callOp in root function for programID=%d, leaf=%s", newProgramID, leafFunc.GetRawName().c_str());
    return SUCCESS;
}

// 辅助函数：在tensor列表中查找指定magic的tensor索引
int MixSubgraphSplit::FindTensorIndexInList(int tensorMagic,
                                           const std::vector<LogicalTensorPtr>& tensorList) const {
    for (size_t i = 0; i < tensorList.size(); ++i) {
        if (tensorList[i] != nullptr && tensorList[i]->magic == tensorMagic) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool MixSubgraphSplit::ExtractArgListFromIncast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const {
    //使用invokeInfo中预先构造的incast信息
    for (const auto& in : invokeInfo.GetIncastTensorParamList()) {
        if (in.opMagic == -1) { // 有效的incast
            continue;
        }
        int offset = GetOffsetFromIncastParam(in, leafFunc);
        if (offset == -1) {
            continue;
        }
        // 根据tensor维度计算参数块长度
        int dim = in.shape.size();
        int blockLength = 1 + 4 * dim;
        std::vector<SymbolicScalar> argBlock;
        for (size_t i = 0; i < static_cast<size_t>(blockLength) && (offset + i) < originalLinearArgs.size(); i++) {
            argBlock.push_back(originalLinearArgs[offset + i]);
        }
        if (argBlock.size() == static_cast<size_t>(blockLength)) {
            extractInfo.extractedArgList.push_back(argBlock);
            extractInfo.iOffsets.push_back(extractInfo.currentOffset);
            extractInfo.currentOffset += blockLength;
            extractInfo.processedTensors.insert(in.tensor);
            ALOG_DEBUG_F("Incast (op=%d, idx=%d, dim=%d) -> offset=%d, length=%d -> extracted arg block",
                        in.opMagic, in.operandIdx, dim, offset, blockLength);
        }
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromOutcast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const {
    //使用invokeInfo中预先构造的outcast信息
    for (const auto& out : invokeInfo.GetOutcastTensorParamList()) {
        if (out.opMagic == -1) {
            continue;
        }
        int offset = GetOffsetFromOutcastParam(out, leafFunc);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to get offset for outcast (op=%d, idx=%d)!",
                        out.opMagic, out.operandIdx);
            continue;
        }
        // 根据tensor维度计算参数块长度
        int dim = out.shape.size();  // 从outcastParam中获取维度
        int blockLength = 1 + 4 * dim;
        // 提取完整的参数块
        std::vector<SymbolicScalar> argBlock;
        for (size_t i = 0; i < static_cast<size_t>(blockLength) && (offset + i) < originalLinearArgs.size(); i++) {
            argBlock.push_back(originalLinearArgs[offset + i]);
        }
        if (argBlock.size() == static_cast<size_t>(blockLength)) {
            extractInfo.extractedArgList.push_back(argBlock);
            extractInfo.oOffsets.push_back(extractInfo.currentOffset);
            extractInfo.currentOffset += blockLength;
            extractInfo.processedTensors.insert(out.tensor);
            ALOG_DEBUG_F("Outcast (op=%d, idx=%d, dim=%d) -> offset=%d, length=%d -> extracted arg block",
                        out.opMagic, out.operandIdx, dim, offset, blockLength);
        } else {
            ALOG_ERROR_F("Failed to extract complete arg block for outcast (op=%d, idx=%d): expected %d elements, got %zu",
                        out.opMagic, out.operandIdx, blockLength, argBlock.size());
            return false;
        }
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const {
    for (const auto &tensor : invokeInfo.GetTensorParamList()) {
        if (tensor.opMagic == -1) {
            continue;
        }
        int offset = GetOffsetFromTensorParam(tensor, leafFunc);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to get offset for global tensor (op=%d, idx=%d, isOutput=%d)!",
                        tensor.opMagic, tensor.operandIdx, tensor.isOutputToGM);
            continue;
        }
        // 根据tensor维度计算参数块长度
        int dim = tensor.shape.size(); // 从tensorParam中获取维度
        int blockLength = 1 + 4 * dim;

        // 提取完整的参数块
        std::vector<SymbolicScalar> argBlock;
        for (size_t i = 0; i < static_cast<size_t>(blockLength) && (offset + i) < originalLinearArgs.size(); i++) {
            argBlock.push_back(originalLinearArgs[offset + i]);
        }
        if (argBlock.size() == static_cast<size_t>(blockLength)) {
            extractInfo.extractedArgList.push_back(argBlock);
            ALOG_DEBUG_F("Incast arg block [offset=%d, length=%d]:", offset, blockLength);
            for (size_t j = 0; j < argBlock.size(); ++j) {
                const auto& arg = argBlock[j];
                if (arg.Raw()->IsImmediate()) {
                    auto imm = std::dynamic_pointer_cast<RawSymbolicImmediate>(arg.Raw());
                    if (imm) {
                        ALOG_DEBUG_F("  [%zu]: Immediate = %ld", j, imm->Immediate());
                    }
                } 
            }
            if (tensor.isOutputToGM) {
                extractInfo.oOffsets.push_back(extractInfo.currentOffset);
                ALOG_DEBUG_F("Global tensor -> Outcast: op=%d, idx=%d, dim=%d -> oOffset=%d", tensor.opMagic, tensor.operandIdx, dim, extractInfo.oOffsets.back());
            } else {
                extractInfo.iOffsets.push_back(extractInfo.currentOffset);
                ALOG_DEBUG_F("Global tensor -> Incast: op=%d, idx=%d, dim=%d -> iOffset=%d", tensor.opMagic, tensor.operandIdx, dim, extractInfo.iOffsets.back());
            }
            extractInfo.currentOffset += blockLength;
            extractInfo.processedTensors.insert(tensor.tensor);
            ALOG_DEBUG_F("Global tensor (op=%d, idx=%d, dim=%d, isOutput=%d) -> offset=%d, length=%d -> extracted arg block", tensor.opMagic, tensor.operandIdx, dim, tensor.isOutputToGM, offset, blockLength);
        } else {
            ALOG_ERROR_F("Failed to extract complete arg block for global tensor (op=%d, idx=%d): expected %d elements, got %zu", tensor.opMagic, tensor.operandIdx, blockLength, argBlock.size());
            return false;
        }
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromActualIncasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualIncasts, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const {
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    for (const auto& incast : actualIncasts) {
        if (incast == nullptr || extractInfo.processedTensors.count(incast) > 0) {
            continue;
        }

        // 这是传播依赖添加的参数，需要特殊处理
        ALOG_DEBUG_F("Processing propagated incast: tensor=%d", incast->GetRawMagic());

        // 在原始Mix function中查找这个tensor的offset
        int offset = FindOriginalOffsetInMixFunction(incast);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to find offset for propagated incast tensor %d!", incast->GetRawMagic());
            return false;  // 直接报错返回
        }
        auto shape = incast->GetShape();
        if (shape.empty()) {
            ALOG_ERROR_F("Propagated incast tensor %d has empty shape!", incast->GetRawMagic());
            return false;
        }
        int dim = shape.size();
        int blockLength = 1 + 4 * dim;
        std::vector<SymbolicScalar> argBlock;
        for (size_t i = 0; i < static_cast<size_t>(blockLength) && (offset + i) < originalLinearArgs.size(); i++) {
            argBlock.push_back(originalLinearArgs[offset + i]);
        }
        if (argBlock.size() == static_cast<size_t>(blockLength)) {
            extractInfo.extractedArgList.push_back(argBlock);
            ALOG_DEBUG_F("Incast arg block [offset=%d, length=%d]:", offset, blockLength);
            for (size_t j = 0; j < argBlock.size(); ++j) {
                const auto& arg = argBlock[j];
                if (arg.Raw()->IsImmediate()) {
                    auto imm = std::dynamic_pointer_cast<RawSymbolicImmediate>(arg.Raw());
                    if (imm) {
                        ALOG_DEBUG_F("  [%zu]: Immediate = %ld", j, imm->Immediate());
                    }
                } 
            }
            extractInfo.iOffsets.push_back(extractInfo.currentOffset); // 传播的incast
            extractInfo.currentOffset += blockLength;
            extractInfo.processedTensors.insert(incast);
            ALOG_DEBUG_F("Extracted propagated incast: tensor=%d, offset=%d",
                        incast->GetRawMagic(), offset);
        } else {
            ALOG_ERROR_F("Failed to extract complete arg block for propagated incast tensor %d: expected %d elements, got %zu",
                        incast->GetRawMagic(), blockLength, argBlock.size());
            return false;
        }
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromActualOutcasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualOutcasts, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const {
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    for (const auto& outcast : actualOutcasts) {
        if (outcast == nullptr || extractInfo.processedTensors.count(outcast) > 0) {
            continue;
        }
        ALOG_DEBUG_F("Processing propagated outcast: tensor=%d", outcast->GetRawMagic());
        auto shape = outcast->GetShape();
        if (shape.empty()) {
            ALOG_ERROR_F("Propagated outcast tensor %d has empty shape!", outcast->GetRawMagic());
            return false;
        }
        int offset = FindOriginalOffsetInMixFunction(outcast);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to find offset for propagated outcast tensor %d!", outcast->GetRawMagic());
            return false;  // 直接报错返回
        }
        std::vector<SymbolicScalar> argBlock;
        int dim = shape.size();
        int blockLength = 4 * dim + 1;
        for (size_t i = 0; i < static_cast<size_t>(blockLength) && (offset + i) < originalLinearArgs.size(); i++) {
            argBlock.push_back(originalLinearArgs[offset + i]);
        }
        if (argBlock.size() == static_cast<size_t>(blockLength)) {
            extractInfo.extractedArgList.push_back(argBlock);
            ALOG_DEBUG_F("Incast arg block [offset=%d, length=%d]:", offset, blockLength);
            for (size_t j = 0; j < argBlock.size(); ++j) {
                const auto& arg = argBlock[j];
                if (arg.Raw()->IsImmediate()) {
                    auto imm = std::dynamic_pointer_cast<RawSymbolicImmediate>(arg.Raw());
                    if (imm) {
                        ALOG_DEBUG_F("  [%zu]: Immediate = %ld", j, imm->Immediate());
                    }
                } 
            }
            extractInfo.oOffsets.push_back(extractInfo.currentOffset); // 传播的outcast
            extractInfo.currentOffset += blockLength;
            extractInfo.processedTensors.insert(outcast);
            ALOG_DEBUG_F("Extracted propagated outcast: tensor=%d, offset=%d, dim=%d",
                        outcast->GetRawMagic(), offset, dim);
        } else {
            ALOG_ERROR_F("Failed to extract complete arg block for propagated outcast tensor %d: expected %d elements, got %zu",
                        outcast->GetRawMagic(), blockLength, argBlock.size());
            return false;
        }
    }
    return true;
}

void MixSubgraphSplit::DisplayArg(const std::vector<SymbolicScalar>& originalLinearArgs) const {
    ALOG_DEBUG_F("Original linear args count: %zu", originalLinearArgs.size());
    for (size_t i = 0; i < originalLinearArgs.size(); ++i) {
        const auto& arg = originalLinearArgs[i];
        // 根据SymbolicScalar的类型打印具体值
        if (arg.Raw()->IsImmediate()) {
            auto imm = std::dynamic_pointer_cast<RawSymbolicImmediate>(arg.Raw());
            if (imm) {
                ALOG_DEBUG_F("  Arg[%zu]: Immediate = %ld", i, imm->Immediate());
            }
        } 
    }
}

std::vector<std::vector<SymbolicScalar>> MixSubgraphSplit::ExtractArgListForLeafFunction(
        Function& leafFunc,
        CallOpAttribute* originalCallAttr,
        const SubfuncInvokeInfoTy& invokeInfo,
        std::vector<int>& iOffsets,
        std::vector<int>& oOffsets) const {
    auto originalLinearArgs = originalCallAttr->GetLinearArgList();
    std::vector<std::vector<SymbolicScalar>> extractedArgList;
    DisplayArg(originalLinearArgs);
    // 清空offset向量
    iOffsets.clear();
    oOffsets.clear();
    int currentOffset = COA_INDEX_BASE;
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();

    ALOG_DEBUG_F("Leaf function %s has %zu actual incasts, %zu actual outcasts after dependency propagation",
                leafFunc.GetRawName().c_str(), actualIncasts.size(), actualOutcasts.size());
    //处理直接参数（在原始invokeInfo中能找到的）
    std::set<LogicalTensorPtr> processedTensors;

    ExtractInfo extractInfo{extractedArgList, iOffsets, oOffsets, currentOffset, processedTensors};
    //使用invokeInfo中预先构造的incast信息
    if (!ExtractArgListFromIncast(invokeInfo, leafFunc, originalLinearArgs, extractInfo)) {
        return {};
    }
    //使用invokeInfo中的outcast信息
    if (!ExtractArgListFromOutcast(invokeInfo, leafFunc, originalLinearArgs, extractInfo)) {
        return {};
    }
    // 使用invokeInfo中的global tensor信息
    if (!ExtractArgListFromGlobalTensor(invokeInfo, leafFunc, originalLinearArgs, extractInfo)) {
        return {};
    }
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    if (!ExtractArgListFromActualIncasts(actualIncasts, originalLinearArgs, extractInfo)) {
        return {};
    }
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    if (!ExtractArgListFromActualOutcasts(actualOutcasts, originalLinearArgs, extractInfo)) {
        return {};
    }

    ALOG_INFO_F("Extracted %zu arg blocks for leaf function %s",
                extractedArgList.size(), leafFunc.GetRawName().c_str());
    return extractedArgList;
}

Status MixSubgraphSplit::SetOffsetsToLeafFunction(Function& leafFunc, const std::vector<int>& iOffsets, const std::vector<int> &oOffsets, const SubfuncInvokeInfoTy& invokeInfo) {
    ALOG_DEBUG_F("Setting offsets to leaf function %s, %zu iOffsets, %zu oOffsets",
                leafFunc.GetRawName().c_str(), iOffsets.size(), oOffsets.size());
    int iOffsetIndex = 0;
    int oOffsetIndex = 0;
    // 设置incast的offset
    for (const auto& in : invokeInfo.GetIncastTensorParamList()) {
        if (in.opMagic != -1 && iOffsetIndex < static_cast<int>(iOffsets.size())) {
            if (SetOffsetToOpByMagic(in.opMagic, in.operandIdx, iOffsets[iOffsetIndex], leafFunc, false)) {
                ALOG_DEBUG_F("Set offset %d to op %d input[%d] for incast",
                            iOffsets[iOffsetIndex], in.opMagic, in.operandIdx);
            }
            iOffsetIndex++;
        }
    }
    // 设置global tensor作为输入的offset（使用iOffsets）
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic != -1 && !tensorParam.isOutputToGM) {
            int offset = iOffsets[iOffsetIndex];
            if (SetOffsetToOpByMagic(tensorParam.opMagic, tensorParam.operandIdx, offset, leafFunc, false)) {
                ALOG_DEBUG_F("Set global tensor input offset %d to op %d input[%d] for tensor %d (iOffsetIndex = %d)",
                            offset, tensorParam.opMagic, tensorParam.operandIdx,
                            tensorParam.tensor->GetRawMagic(), iOffsetIndex);
            } 
            iOffsetIndex++;
        }
    }
    // 设置outcast的offset
    for (const auto& out : invokeInfo.GetOutcastTensorParamList()) {
        if (out.opMagic != -1 && oOffsetIndex < static_cast<int>(oOffsets.size())) {
            if (SetOffsetToOpByMagic(out.opMagic, out.operandIdx, oOffsets[oOffsetIndex], leafFunc, true)) {
                ALOG_DEBUG_F("Set offset %d to op %d output[%d] for outcast",
                            oOffsets[oOffsetIndex], out.opMagic, out.operandIdx);
            }
            oOffsetIndex++;
        }
    }
    // 设置global tensor作为输出的offset（使用oOffsets）
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic != -1 && tensorParam.isOutputToGM) {
            int offset = oOffsets[oOffsetIndex];
            if (SetOffsetToOpByMagic(tensorParam.opMagic, tensorParam.operandIdx, offset, leafFunc, true)) {
                ALOG_DEBUG_F("Set global tensor output offset %d to op %d output[%d] for tensor %d (oOffsetIndex = %d)",
                            offset, tensorParam.opMagic, tensorParam.operandIdx,
                            tensorParam.tensor->GetRawMagic(), oOffsetIndex);
            } 
            oOffsetIndex++;
        }
    }
    ALOG_INFO_F("Successfully set all offsets to leaf function %s(input: %d offsets, output: %d offsets)",
                leafFunc.GetRawName().c_str(), iOffsetIndex, oOffsetIndex);
    return SUCCESS;
}

// 根据opMagic直接设置offset到指定的op，并更新CopyOpAttribute中的表达式
bool MixSubgraphSplit::SetOffsetToOpByMagic(int opMagic, int operandIdx, int offset,
                                            Function& leafFunc, bool isOutput) const {
    if (opMagic == -1 || operandIdx < 0) {
        ALOG_ERROR_F("Invalid opMagic=%d or operandIdx=%d", opMagic, operandIdx);
        return false;
    }

    // 在leaf function中查找指定opMagic的op
    auto operations = leafFunc.Operations(false);
    for (auto& op : operations) {
        if (op.GetOpMagic() == opMagic) {
            if (isOutput) {
            // 设置输出offset
            if (static_cast<size_t>(operandIdx) < op.GetOOperands().size()) {
                op.SetOOpAttrOffset(operandIdx, offset);
                ALOG_DEBUG_F("Set OOpAttrOffset %d for op %d output[%d]",
                            offset, opMagic, operandIdx);
                // 如果是Copy操作，还需要更新CopyOpAttribute中的表达式
                UpdateCopyOpAttributeExpressions(&op, offset, isOutput);
                return true;
            } else {
                ALOG_ERROR_F("Operand index %d out of range for op %d outputs (max=%zu)",
                            operandIdx, opMagic, op.GetOOperands().size());
                return false;
            }
            } else {
                // 设置输入offset
                if (static_cast<size_t>(operandIdx) < op.GetIOperands().size()) {
                    op.SetIOpAttrOffset(operandIdx, offset);
                    ALOG_DEBUG_F("Set IOpAttrOffset %d for op %d input[%d]",
                                offset, opMagic, operandIdx);
                    // 如果是Copy操作，还需要更新CopyOpAttribute中的表达式
                    UpdateCopyOpAttributeExpressions(&op, offset, isOutput);
                    return true;
                } else {
                    ALOG_ERROR_F("Operand index %d out of range for op %d inputs (max=%zu)",
                                operandIdx, opMagic, op.GetIOperands().size());
                    return false;
                }
            }
        }
    }
    ALOG_WARN_F("Failed to find op with magic=%d in leaf function", opMagic);
    return false;
}

// 更新CopyOpAttribute中的表达式
void MixSubgraphSplit::UpdateCopyOpAttributeExpressions(Operation* op, int newOffset, bool isOutput) const {
    if (op == nullptr) {
        return;
    }
    // 只处理Copy相关的op
    Opcode opcode = op->GetOpcode();
    if (!IsCopyIn(opcode) &&  !IsCopyOut(opcode)) {
        return;
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (copyAttr == nullptr) {
        ALOG_DEBUG_F("Op %d is not a CopyOp or has no CopyOpAttribute", op->GetOpMagic());
        return;
    }
    // 创建新的offset值（RawSymbolicImmediate）
    auto newOffsetValue = RawSymbolicImmediate::Create(newOffset);
    auto newOffsetScalar = std::static_pointer_cast<RawSymbolicScalar>(newOffsetValue);
    ALOG_DEBUG_F("Updating CopyOpAttribute expressions for op %d, newOffset=%d, isOutput=%s",
                op->GetOpMagic(), newOffset, isOutput ? "true" : "false");
    if (isOutput) {
        // 处理CopyOut: 更新toOffset
        auto toOffsets = copyAttr->GetToOffset();
        UpdateOffsetExpressions(toOffsets, newOffsetScalar);
        copyAttr->SetToOffset(toOffsets);
        ALOG_DEBUG_F("Updated CopyOut op %d toOffset expressions", op->GetOpMagic());
    } else {
        // 处理CopyIn: 更新fromOffset
        auto fromOffsets = copyAttr->GetFromOffset();
        UpdateOffsetExpressions(fromOffsets, newOffsetScalar);
        copyAttr->SetFromOffset(fromOffsets);
        ALOG_DEBUG_F("Updated CopyIn op %d fromOffset expressions", op->GetOpMagic());
    }
}

// 更新offset表达式
void MixSubgraphSplit::UpdateOffsetExpressions(std::vector<OpImmediate>& offsets,
                                            const RawSymbolicScalarPtr& newOffsetValue) const {
    for (size_t i = 0; i < offsets.size(); ++i) {
        SymbolicScalar& ss = offsets[i].GetSpecifiedValue();
        RawSymbolicScalarPtr ssRaw = ss.Raw();
        const size_t NUM_THREE = 3;
        if (ssRaw->IsExpression()) {
            std::vector<RawSymbolicScalarPtr> rawList = ssRaw->GetExpressionOperandList();
            if (rawList.size() >= NUM_THREE) {
                RawSymbolicScalarPtr replaced = ReplaceExpression(ssRaw, rawList[2], newOffsetValue);
                offsets[i] = OpImmediate(SymbolicScalar(replaced));
                ALOG_DEBUG_F("Replaced offset expression at index %zu", i);
            }
        }
    }
}

int MixSubgraphSplit::FindOriginalOffsetInMixFunction(LogicalTensorPtr tensor) const {
    if (tensor == nullptr) {
        ALOG_ERROR_F("Tensor is nullptr in FindOriginalOffsetInMixFunction");
        return -1;
    }

    ALOG_DEBUG_F("Finding original offset for tensor %d in mix function", tensor->GetRawMagic());

    // 检查producers
    auto producers = tensor->GetProducers();
    for (auto* producer : producers) {
        if (producer == nullptr) {
            continue;
        }

        auto oOperands = producer->GetOOperands();
        for (size_t i = 0; i < oOperands.size(); i++) {
            if (oOperands[i] == tensor) {
                int offset = producer->GetOOpAttrOffset(i);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found offset %d for tensor %d via producer op %d output[%zu]",
                                offset, tensor->GetRawMagic(), producer->GetOpMagic(), i);
                    return offset;
                }
            }
        }
    }

    // 检查consumers
    auto consumers = tensor->GetConsumers();
    for (auto* consumer : consumers) {
        if (consumer == nullptr) {
            continue;
        }

        auto iOperands = consumer->GetIOperands();
        for (size_t i = 0; i < iOperands.size(); i++) {
            if (iOperands[i] == tensor) {
                int offset = consumer->GetIOpAttrOffset(i);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found offset %d for tensor %d via consumer op %d input[%zu]",
                                offset, tensor->GetRawMagic(), consumer->GetOpMagic(), i);
                    return offset;
                }
            }
        }
    }

    ALOG_ERROR_F("Failed to find original offset for tensor %d in mix function (checked %zu producers, %zu consumers)",
                tensor->GetRawMagic(), producers.size(), consumers.size());
    return -1;
}

int MixSubgraphSplit::GetOffsetFromIncastParam(const SubfuncInvokeInfoTy::IncastParamPackTy& incastParam, Function& leafFunc) const {
    // 通过opMagic在leafFunc中查找对应的操作
    auto operations = leafFunc.Operations(false);
    for (auto& op : operations) {
        if (op.GetOpMagic() == incastParam.opMagic) {
            // 检查输入操作数
            if (incastParam.operandIdx >= 0 && static_cast<size_t>(incastParam.operandIdx) < op.GetIOperands().size()) {
                int offset = op.GetIOpAttrOffset(incastParam.operandIdx);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found input offset %d for incast (op=%d, idx=%d)",
                                offset, incastParam.opMagic, incastParam.operandIdx);
                    return offset;
                } else {
                    ALOG_WARN_F("Input offset is -1 for incast (op=%d, idx=%d)",
                                incastParam.opMagic, incastParam.operandIdx);
                }
            } else {
                ALOG_WARN_F("Invalid operand index %d for op %d (max=%zu)",
                            incastParam.operandIdx, incastParam.opMagic, op.GetIOperands().size());
            }
            break;
        }
    }
    ALOG_WARN_F("Could not find op %d for incast", incastParam.opMagic);
    return -1;
}

int MixSubgraphSplit::GetOffsetFromOutcastParam(const SubfuncInvokeInfoTy::OutcastParamPackTy& outcastParam, Function& leafFunc) const {
    // 通过opMagic在leafFunc中查找对应的操作
    auto operations = leafFunc.Operations(false);
    for (auto& op : operations) {
        if (op.GetOpMagic() == outcastParam.opMagic) {
            // 检查输出操作数
            if (outcastParam.operandIdx >= 0 && static_cast<size_t>(outcastParam.operandIdx) < op.GetOOperands().size()) {
                int offset = op.GetOOpAttrOffset(outcastParam.operandIdx);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found output offset %d for outcast (op=%d, idx=%d)",
                                offset, outcastParam.opMagic, outcastParam.operandIdx);
                    return offset;
                } else {
                    ALOG_WARN_F("Output offset is -1 for outcast (op=%d, idx=%d)",
                                outcastParam.opMagic, outcastParam.operandIdx);
                }
            } else {
                ALOG_WARN_F("Invalid operand index %d for op %d (max=%zu)",
                            outcastParam.operandIdx, outcastParam.opMagic, op.GetOOperands().size());
            }
            break;
        }
    }
    ALOG_WARN_F("Could not find op %d for outcast", outcastParam.opMagic);
    return -1;
}

int MixSubgraphSplit::GetOffsetFromTensorParam(const SubfuncInvokeInfoTy::TensorParamPackTy& tensorParam, Function& leafFunc) const {
    // 对于global tensor，根据isOutputToGM判断是输入还是输出
    if (tensorParam.isOutputToGM) {
        // 作为输出处理
        auto operations = leafFunc.Operations(false);
        for (auto& op : operations) {
            if (op.GetOpMagic() != tensorParam.opMagic) {
                continue;
            }
            if (tensorParam.operandIdx >= 0 && static_cast<size_t>(tensorParam.operandIdx) < op.GetOOperands().size()) {
                int offset = op.GetOOpAttrOffset(tensorParam.operandIdx);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found output offset %d for global tensor (op=%d, idx=%d)",
                                offset, tensorParam.opMagic, tensorParam.operandIdx);
                    return offset;
                }
            }
            break;
        }
    } else {
        // 作为输入处理
        auto operations = leafFunc.Operations(false);
        for (auto& op : operations) {
            if (op.GetOpMagic() != tensorParam.opMagic) {
                continue;
            }
            if (tensorParam.operandIdx >= 0 && static_cast<size_t>(tensorParam.operandIdx) < op.GetIOperands().size()) {
                int offset = op.GetIOpAttrOffset(tensorParam.operandIdx);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found input offset %d for global tensor (op=%d, idx=%d)",
                                offset, tensorParam.opMagic, tensorParam.operandIdx);
                    return offset;
                }
            }
            break;
        }
    }

    ALOG_WARN_F("Could not find offset for global tensor (op=%d, idx=%d, isOutput=%d)",
                tensorParam.opMagic, tensorParam.operandIdx, tensorParam.isOutputToGM);
    return -1;
}

Function* MixSubgraphSplit::CreateSplitLeafFunction(Function& rootFunc,
                                                    Function& originalMixFunc,
                                                    const InternalComponentInfo& component,
                                                    uint64_t newProgramID,
                                                    uint64_t i,
                                                    SubgraphToFunction& subgraphToFunction) {
    // 创建新的function名称
    std::string leafName = originalMixFunc.GetRawName() + "_leaf" + std::to_string(i);
    ALOG_DEBUG_F("Add leafFunction %s", leafName.c_str());
    // 手动创建function对象
    auto funcMagicName = leafName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    auto newFunc = std::make_shared<Function>(Program::GetInstance(), funcMagicName, leafName, &rootFunc);
    // 设置function类型
    newFunc->SetFunctionType(FunctionType::STATIC);
    newFunc->SetGraphType(GraphType::BLOCK_GRAPH);

    std::vector<std::shared_ptr<Operation>> programOps;
    // 获取原始Mix子图的所有op（按原始顺序）
    auto originalOps = originalMixFunc.Operations(false).DuplicatedOpList();
    // 按原始顺序筛选属于当前component的op
    for (auto *originalOp : originalOps) {
        if (originalOp->IsNOP()) {
            continue;
        }
        // 检查这个op是否属于当前component
        bool belongsToComponent = false;
        for (auto* compOp : component.operations) {
            if (compOp == originalOp) {
                belongsToComponent = true;
                break;
            }
        }

        if (belongsToComponent) {
            programOps.push_back(originalOp->shared_from_this());
            ALOG_DEBUG_F("Added op %d to leaf function %s (original order preserved)",
                        originalOp->GetOpMagic(), leafName.c_str());
        }
    }
    // 验证顺序正确性
    ALOG_DEBUG_F("Leaf function %s has %zu ops in original order",
                leafName.c_str(), programOps.size());
    newFunc->SetProgramOp(programOps);
    // 创建并设置LeafFuncAttribute
    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    // 设置aivCore属性
    leafAttr->aivCore = component.aivCore;
    newFunc->SetLeafFuncAttribute(leafAttr);
    newFunc->UpdateBelongToThis();
    newFunc->SetProgramId(newProgramID);
    // 复制参数配置
    newFunc->paramConfigs_ = originalMixFunc.paramConfigs_;
    ALOG_DEBUG_F("Called UpdateBelongToThis for new function: %s", leafName.c_str());
    newFunc->ComputeHash();
    FunctionHash funcHash = newFunc->GetFunctionHash();
    ALOG_DEBUG_F("Function %s computed hash: %lu", leafName.c_str(), funcHash);
    Program::GetInstance().GetFunctionCache().Insert(funcHash, *newFunc);
    ALOG_DEBUG_F("Inserted new function %s into function cache with hash %lu",
                leafName.c_str(), funcHash);
    // 注册到program的function map中
    auto* resultFunc = newFunc.get();
    Program::GetInstance().InsertFuncToFunctionMap(funcMagicName, newFunc);
    subgraphToFunction.InsertParameter(i, *resultFunc);
    ALOG_DEBUG_F("Created leaf function: %s.", leafName.c_str());
    return resultFunc;
}

void MixSubgraphSplit::DisplayComponents(const std::vector<InternalComponentInfo>& components) {
    for (size_t i = 0; i < components.size(); i++) {
        const auto& component = components[i];
        ALOG_DEBUG_F("Component[%zu]: internalID=%d, suffix=%s, aivCore=%d, operations=%zu",
                    i, component.internalSubgraphID, component.suffix.c_str(),
                    static_cast<int>(component.aivCore), component.operations.size());
        // 打印component中的operations
        for (size_t j = 0; j < component.operations.size(); j++) {
            auto* op = component.operations[j];
            if (op != nullptr) {
                ALOG_DEBUG_F("  Operation[%zu]: magic=%d, opcode=%s, internalSubgraphID=%d",
                            j, op->GetOpMagic(), op->GetOpcodeStr().c_str(), op->GetInternalSubgraphID());
            }
        }
    }
}

SubgraphToFunction MixSubgraphSplit::InitSubgraphToFunction(const std::vector<InternalComponentInfo>& components) {
    SubgraphToFunction subgraphToFunction;
    subgraphToFunction.nLIST.resize(components.size());
    subgraphToFunction.subFuncInvokeInfos.resize(components.size());
    // 初始化nList
    for (size_t compIndex = 0; compIndex < components.size(); compIndex++) {
        const auto& component = components[compIndex];
        // 将 Operation* 转换为 std::shared_ptr<Operation>
        std::vector<std::shared_ptr<Operation>> sharedOperations;
        for (auto* op : component.operations) {
            sharedOperations.push_back(op->shared_from_this());
        }
        subgraphToFunction.nLIST[compIndex] = sharedOperations;
        subgraphToFunction.subFuncInvokeInfos[compIndex] = SubfuncInvokeInfoTy();
    }
    return subgraphToFunction;
}

void MixSubgraphSplit::InOutCastRecord(SubgraphToFunction &subgraphToFunction, Function* originalMixFunc) {
    for (int i = 0; i < static_cast<int>(subgraphToFunction.nLIST.size()); i++) {
        for (size_t j = 0; j < subgraphToFunction.nLIST[i].size(); j++) {
            for (size_t k = 0; k < subgraphToFunction.nLIST[i][j]->GetIOperands().size(); k++) {
                subgraphToFunction.RecordEsgIncast(*originalMixFunc, i, j, k);
            }
            for (size_t k = 0; k < subgraphToFunction.nLIST[i][j]->GetOOperands().size(); k++) {
                subgraphToFunction.RecordEsgOutcast(*originalMixFunc, i, j, k);
            }
        }
    }
    // 完成所有记录
    for (auto& invokeInfo : subgraphToFunction.subFuncInvokeInfos) {
        invokeInfo.DoFinishRecord();
    }
    for (size_t i = 0; i < subgraphToFunction.subFuncInvokeInfos.size(); i++) {
        subgraphToFunction.subFuncInvokeInfos[i].ConstructActualInvokeParam(i);
    }
}


Status MixSubgraphSplit::GenNewFunctions(Function& rootFunc, Function* originalMixFunc,
                                        const std::vector<InternalComponentInfo>& components,
                                        const std::vector<uint64_t>& newProgramIDs,
                                        SubgraphToFunction& subgraphToFunction,
                                        std::vector<Function*>& newFunctions) {
    for (size_t i = 0; i < components.size(); i++) {
        Function* newFunc = CreateSplitLeafFunction(rootFunc, *originalMixFunc, components[i], newProgramIDs[i], i, subgraphToFunction);
        if (newFunc == nullptr) {
            return FAILED;
        }
        newFunctions.push_back(newFunc);
    }
    return SUCCESS;
}

Status MixSubgraphSplit::CreateCallOps(Function& rootFunc, const std::vector<Operation*>& originalCallOps,
                                        Function* originalMixFunc,
                                        const std::vector<InternalComponentInfo>& components,
                                        const std::vector<uint64_t>& newProgramIDs,
                                        SubgraphToFunction& subgraphToFunction,
                                        std::vector<Function*>& newFunctions) {
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
            callOpInfos.push_back(info);
        }
    }
    // 先创建所有callOp但不设置offset到leaf function中的op
    std::vector<Operation*> newCallOps;
    for (auto& info : callOpInfos) {
        auto status = CreateCallOpInRootFunction(rootFunc, *info.leafFunc, info.newProgramID, info.componentIndex,
                                                        info.originalCallOp, originalMixFunc, subgraphToFunction, info.wrapId, info.iOffsets, info.oOffsets);
        (void)status;
    }
    // 在所有callOp都创建完成后，统一设置offset到leaffunction中的op
    for (size_t i = 0; i < callOpInfos.size(); ++i) {
        const auto& info = callOpInfos[i];
        auto* leafFunc = info.leafFunc;
        // 设置offset到leaf function中的op
        auto status = SetOffsetsToLeafFunction(*leafFunc, info.iOffsets, info.oOffsets,
                                                subgraphToFunction.subFuncInvokeInfos[info.componentIndex]);
        (void)status;
    }
    return SUCCESS;
}

Status MixSubgraphSplit::SetMixIdResourceType(std::vector<Function*> &newFunctions, uint64_t mixId, MixResourceType resourceType) {
    for (size_t i = 0; i < newFunctions.size(); i++) {
        auto leafAttr = newFunctions[i]->GetLeafFuncAttribute();
        if (leafAttr == nullptr) {
            ALOG_ERROR_F("LeafFuncAttribute not set for new function");
            return FAILED;
        }
        // 设置mixId和resourceType
        leafAttr->mixId = mixId;
        leafAttr->mixResourceType = resourceType;
        ALOG_DEBUG_F("Set mixId=%lu to leaf function %s", mixId, newFunctions[i]->GetRawName().c_str());
    }
    return SUCCESS;
}

Status MixSubgraphSplit::ProcessLeafFunction(Function& rootFunc,
                                            uint64_t programID,
                                            Function* originalMixFunc,
                                            const std::vector<InternalComponentInfo>& components,
                                            const std::vector<uint64_t>& newProgramIDs,
                                            const std::vector<Operation*>& originalCallOps,
                                            std::vector<MixSubgraphSplitResult>& splitResults) {
    ALOG_INFO_F("Processing mix subgraph: programID=%d, callOps=%d, components=%d", programID, originalCallOps.size(), components.size());
    ALOG_DEBUG_F("=== Component Details ===");
    DisplayComponents(components);
    // 获取资源类型
    MixResourceType resourceType = GetMixResourceType(*originalMixFunc);
    ALOG_DEBUG_F("Mix resource type: %d for programID=%d", static_cast<int>(resourceType), programID);
    // 为这个mix leafFunction分配唯一的mixId
    uint64_t mixId = nextMixId_++;
    ALOG_DEBUG_F("Assigning mixId=%lu for original mix function programID=%d", mixId, programID);
    SubgraphToFunction subgraphToFunction = InitSubgraphToFunction(components);
    // 步骤1：分析组件间直接依赖关系
    ALOG_INFO_F("Analyzing component dependencies...");
    auto directDeps = AnalyzeComponentDependencies(*originalMixFunc);
    // 步骤2：计算依赖传递闭包
    ALOG_INFO_F("Computing dependency closure...");
    auto dependencyClosure = ComputeDependencyClosure(directDeps);
    // 步骤3：记录直接的incast/outcast
    InOutCastRecord(subgraphToFunction, originalMixFunc);
    // 步骤4：为每个组件创建leaf function并插入参数
    std::vector<Function*> newFunctions;
    if (GenNewFunctions(rootFunc, originalMixFunc, components, newProgramIDs, subgraphToFunction, newFunctions) != SUCCESS) {
        return FAILED;
    }
    // 步骤5：传播外部依赖（基于传递闭包）
    ALOG_INFO_F("Propagating external dependencies...");
    PropagateExternalDependencies(newFunctions, dependencyClosure, subgraphToFunction);
    // 步骤6：为每个原始CallOp创建一组新的callOp, 每个原始callOp使用不同的wrapId
    if (CreateCallOps(rootFunc, originalCallOps, originalMixFunc, components, newProgramIDs, subgraphToFunction, newFunctions) != SUCCESS) {
        return FAILED;
    }
    // 步骤7：复制InferParamIndex信息到所有新子图
    if (CopyInferParamIndexInfo(originalMixFunc, newFunctions) != SUCCESS) {
        return FAILED;
    }
    // 步骤8：设置wrapId和resourceType
    if (SetMixIdResourceType(newFunctions, mixId, resourceType) != SUCCESS) {
        return FAILED;
    }
    // 保存拆分结果
    MixSubgraphSplitResult result;
    result.originalProgramID = programID;
    result.originalFunction = originalMixFunc;
    result.newProgramIDs = newProgramIDs;
    result.newFunctions = newFunctions;
    result.components = components;
    splitResults.push_back(result);
    ALOG_INFO_F("Successfully split mix subgraph programID=%d: created %d program instances and %d callOps", programID, newFunctions.size(), originalCallOps.size() * components.size());
    return SUCCESS;
}

// 分析component间依赖
std::unordered_map<int, std::vector<int>> MixSubgraphSplit::AnalyzeComponentDependencies(Function& mixFunc) const {
    std::unordered_map<int, std::vector<int>> dependencies;
    // 分析CV子图间依赖
    auto operationViewer = mixFunc.Operations(false);
    for (size_t idx = 0; idx < operationViewer.size(); idx++) {
        auto& op = operationViewer[idx];
        if (op.IsNOP()) continue;
        int producerInternalID = op.GetInternalSubgraphID();
        // 分析该op的输出tensor的消费者
        for (size_t k = 0; k < op.GetOOperands().size(); k++) {
            auto oOperand = op.GetOOperands()[k];
            if (oOperand == nullptr) continue;

            auto consumers = oOperand->GetConsumers();
            for (auto* consumer : consumers) {
                if (consumer == nullptr) continue;

                int consumerID = consumer->GetInternalSubgraphID();
                // 记录子图间依赖（忽略同一组件内的依赖）
                if (producerInternalID != consumerID) {
                    dependencies[producerInternalID].push_back(consumerID);
                }
            }
        }
    }
    return dependencies;
}

void MixSubgraphSplit::PropagateIncastDependencies(const std::vector<Function*>& leafFunctions,
                                                const std::unordered_map<int, std::vector<LogicalTensorPtr>> &directIncasts,
                                                const std::unordered_map<int, std::set<int>>& dependencyClosure,
                                                const SubgraphToFunction& subgraphToFunction) const {
    for (const auto& pair : directIncasts) {
        auto it = dependencyClosure.find(pair.first);
        if (it == dependencyClosure.end()) {
            continue;
        }
        // 从subgraphToFunction获取完整的incast参数信息
        std::vector<SimpleIncastParam> incastParams;
        if (pair.first >= 0 && pair.first < static_cast<int>(subgraphToFunction.subFuncInvokeInfos.size())) {
            const auto& invokeInfo = subgraphToFunction.subFuncInvokeInfos[pair.first];
            // 收集所有incast参数
            for (const auto& incast : invokeInfo.GetIncastTensorParamList()) {
                // 创建IncastParam对象，包含完整的参数信息
                incastParams.emplace_back(incast.tensor, incast.opMagic, incast.operandIdx);
                ALOG_DEBUG_F("Collected incast: tensor=%d, opMagic=%d, operandIdx=%d",
                            incast.tensor->GetRawMagic(), incast.opMagic, incast.operandIdx);
            }
            // 从tensorParamList中收集作为输入的tensor
            for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
                if (!tensorParam.isOutputToGM) {
                    incastParams.emplace_back(tensorParam.tensor, tensorParam.opMagic, tensorParam.operandIdx);
                    ALOG_DEBUG_F("Collected tensor param (input): tensor=%d, opMagic=%d, operandIdx=%d",
                                tensorParam.tensor->GetRawMagic(), tensorParam.opMagic, tensorParam.operandIdx);
                }
            }
        }
        for (int targetComp : it->second) {
            if (targetComp >= 0 && targetComp < static_cast<int>(leafFunctions.size())) {
                PropagateIncastToLeafFunction(leafFunctions[targetComp], pair.first, incastParams);
            }
        }
    }
}

void MixSubgraphSplit::PropagateOutcastDependencies(const std::vector<Function*>& leafFunctions,
                                                    const std::unordered_map<int, std::vector<LogicalTensorPtr>> &directOutcasts,
                                                    const std::unordered_map<int, std::set<int>>& dependencyClosure,
                                                    const SubgraphToFunction& subgraphToFunction) const {
    for (const auto& pair : directOutcasts) {
        // 从subgraphToFunction获取完整的outcast参数信息
        std::vector<SimpleOutcastParam> outcastParams;
        if (pair.first >= 0 && pair.first < static_cast<int>(subgraphToFunction.subFuncInvokeInfos.size())) {
            const auto& invokeInfo = subgraphToFunction.subFuncInvokeInfos[pair.first];
            // 收集所有outcast参数
            for (const auto& outcast : invokeInfo.GetOutcastTensorParamList()) {
                outcastParams.emplace_back(outcast.tensor, outcast.opMagic, outcast.operandIdx);
                ALOG_DEBUG_F("Collected outcast: tensor=%d, opMagic=%d, operandIdx=%d",
                            outcast.tensor->GetRawMagic(), outcast.opMagic, outcast.operandIdx);
            }
            // 从tensorParamList中收集作为输出的tensor
            for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
                if (tensorParam.isOutputToGM) {
                    outcastParams.emplace_back(tensorParam.tensor, tensorParam.opMagic, tensorParam.operandIdx);
                    ALOG_DEBUG_F("Collected tensor param (output): tensor=%d, opMagic=%d, operandIdx=%d",
                                tensorParam.tensor->GetRawMagic(), tensorParam.opMagic, tensorParam.operandIdx);
                }
            }
        }
        for (const auto& [sourceComp, deps] : dependencyClosure) {
            if (deps.count(pair.first)) {
                if (sourceComp >= 0 && sourceComp < static_cast<int>(leafFunctions.size())) {
                    PropagateOutcastToLeafFunction(leafFunctions[sourceComp], pair.first, outcastParams);
                }
            }
        }
    }
}

void MixSubgraphSplit::PropagateExternalDependencies(
    const std::vector<Function*>& leafFunctions,
    const std::unordered_map<int, std::set<int>>& dependencyClosure,
    const SubgraphToFunction& subgraphToFunction) const {
    // 收集所有leaf function的直接外部依赖
    std::unordered_map<int, std::vector<LogicalTensorPtr>> directIncasts;
    std::unordered_map<int, std::vector<LogicalTensorPtr>> directOutcasts;
    // 第一遍：收集直接的外部依赖
    for (size_t i = 0; i < leafFunctions.size(); i++) {
        auto* leafFunc = leafFunctions[i];
        if (leafFunc == nullptr) continue;

        const auto& incasts = leafFunc->GetIncast();
        if (!incasts.empty()) {
            directIncasts[i] = incasts;
        }

        const auto& outcasts = leafFunc->GetOutcast();
        if (!outcasts.empty()) {
            directOutcasts[i] = outcasts;
        }
    }
    // 第二遍：传播incast依赖（外部输入依赖向前传播）
    PropagateIncastDependencies(leafFunctions, directIncasts, dependencyClosure, subgraphToFunction);

    // 第三遍：传播outcast依赖（外部输出依赖向后传播）
    PropagateOutcastDependencies(leafFunctions, directOutcasts, dependencyClosure, subgraphToFunction);
}

// 传播incast到目标leaf function
void MixSubgraphSplit::PropagateIncastToLeafFunction(
    Function* targetLeafFunc,
    int sourceComp,
    const std::vector<SimpleIncastParam>& incastParams) const {
    if (targetLeafFunc == nullptr) return;
    // 获取当前leaf function已有的incast
    auto existingIncasts = targetLeafFunc->GetIncast();
    std::set<int> existingMagicSet;
    for (const auto& incast : existingIncasts) {
        if (incast != nullptr) {
            existingMagicSet.insert(incast->magic);
        }
    }

    for (const auto& param : incastParams) {
        // 检查是否已经存在相同magic的tensor
        if (existingMagicSet.find(param.tensor->magic) != existingMagicSet.end()) {
            continue;
        }
        // 使用完整的参数信息传递incast
        targetLeafFunc->AppendIncast(param.tensor, param.opMagic, param.operandIdx);
        existingMagicSet.insert(param.tensor->magic);
        ALOG_DEBUG_F("Propagated incast: component %d -> component %d (tensor %d, opMagic %d, operandIdx %d)",
                    sourceComp, targetLeafFunc->GetProgramId(), param.tensor->GetRawMagic(), param.opMagic, param.operandIdx);
    }
}

// 传播outcast到源leaf function（共享tensor）
void MixSubgraphSplit::PropagateOutcastToLeafFunction(
    Function* sourceLeafFunc,
    int targetComp,
    const std::vector<SimpleOutcastParam>& outcastParams) const {
    if (sourceLeafFunc == nullptr) return;
    // 获取当前leaf function已有的outcast
    auto existingOutcasts = sourceLeafFunc->GetOutcast();
    std::set<int> existingMagicSet;
    for (const auto& outcast : existingOutcasts) {
        if (outcast != nullptr) {
            existingMagicSet.insert(outcast->magic);
        }
    }

    for (const auto& param : outcastParams) {
        // 检查是否已经存在相同magic的tensor
        if (existingMagicSet.find(param.tensor->magic) != existingMagicSet.end()) {
            continue;
        }
        // 使用完整的参数信息传递outcast
        sourceLeafFunc->AppendOutcast(param.tensor, param.opMagic, param.operandIdx);
        existingMagicSet.insert(param.tensor->magic);
        ALOG_DEBUG_F("Propagated outcast: component %d -> component %d (tensor %d, opMagic %d, operandIdx %d)",
                    targetComp, sourceLeafFunc->GetProgramId(),
                    param.tensor->GetRawMagic(), param.opMagic, param.operandIdx);
    }
}

void MixSubgraphSplit::BroadcastDependencyClosure(std::set<int> &deps_i, std::set<int> &newDeps,
                                                std::unordered_map<int, std::set<int>> &closure,
                                                bool &changed, int i) const {
    for (int j : deps_i) {
        // 如果j有传递依赖k，把k也加入i的依赖
        if (closure.count(j)) {
            for (int k : closure[j]) {
                if (newDeps.insert(k).second) {
                    changed = true;
                    ALOG_DEBUG_F("Iteration: %d -> %d -> %d, added %d -> %d",
                                i, j, k, i, k);
                }
            }
        }
    }
}

// 计算依赖传递闭包
std::unordered_map<int, std::set<int>> MixSubgraphSplit::ComputeDependencyClosure(
    const std::unordered_map<int, std::vector<int>>& directDeps) const {
    std::unordered_map<int, std::set<int>> closure;
    // 步骤1：初始化直接依赖
    // 确保所有组件都在closure中，即使没有出边
    int maxComponent = 0;
    for (const auto& [component, deps] : directDeps) {
        closure[component] = std::set<int>(deps.begin(), deps.end());
        if (component > maxComponent) {
            maxComponent = component;
        }
        for (int dep : deps) {
            if (dep > maxComponent) {
                maxComponent = dep;
            }
        }
    }

    // 确保所有组件索引都在closure中
    for (int i = 0; i <= maxComponent; i++) {
        closure[i]; // 确保存在，即使没有依赖关系
    }
    // 步骤2：使用Floyd-Warshall算法计算传递闭包
    bool changed;
    int iteration = 0;
    do {
        changed = false;
        iteration++;
        for (auto& [i, deps_i] : closure) {
            std::set<int> newDeps = deps_i;
            // 对于i的每个直接依赖j
            BroadcastDependencyClosure(deps_i, newDeps, closure, changed, i);
            deps_i = std::move(newDeps);
        }
        ALOG_DEBUG_F("After iteration %d, changed: %s", iteration, changed ? "true" : "false");
    } while (changed);
    return closure;
}

Status MixSubgraphSplit::CopyInferParamIndexInfo(Function* originalMixFunc,
                                                const std::vector<Function*>& newFunctions) const {
    // 获取原Mix子图的完整符号表
    const auto& originalDynParamTable = originalMixFunc->GetDynParamTable();

    // 为每个新子图继承相同的符号表
    for (auto* newFunc : newFunctions) {
        // 使用InsertDynParam方法逐个复制dynParam
        for (const auto& [dim, info] : originalDynParamTable) {
            DynParamInfo copiedInfo = info;
            newFunc->InsertDynParam(dim, copiedInfo);
        }
        ALOG_DEBUG_F("Copied %zu dyn param entries to function: %s",
                    originalDynParamTable.size(), newFunc->GetRawName().c_str());
    }
    return SUCCESS;
}

void MixSubgraphSplit::DeleteOriginalMixCallOps(Function& rootFunc, const std::vector<Operation*>& callOpsToDelete) {
    if (callOpsToDelete.empty()) {
        return;
    }
    for (auto* callOp : callOpsToDelete) {
        if (callOp != nullptr && !callOp->IsDeleted()) {
            callOp->SetAsDeleted();
            ALOG_DEBUG_F("Deleted callOp with magic=%d", callOp->GetOpMagic());
        }
    }
    // 执行实际删除
    rootFunc.EraseOperations(false);
    ALOG_INFO_F("Deleted %d original mix subgraph callOps", callOpsToDelete.size());
}
}
}