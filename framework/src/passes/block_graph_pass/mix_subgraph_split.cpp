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
    auto rootFunc = function.rootFunc_;
    auto callOps = rootFunc->GetCallopList();
    
    // 按哈希值分组callOp
    std::unordered_map<FunctionHash, std::vector<Operation*>> hashToCallOps;  
    for (auto* callOp : callOps) {
        if (callOp == nullptr || callOp->IsDeleted()) {
            continue;
        }
        auto callAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
        if (callAttr != nullptr) {
            FunctionHash calleeHash = callAttr->GetCalleeHash();
            hashToCallOps[calleeHash].push_back(callOp);
        }
    }
    // 为每个哈希值查找对应的cacheFunction并判断是否为Mix子图
    for (auto& pair : hashToCallOps) {
        const FunctionHash& calleeHash = pair.first;
        std::vector<Operation*>& callOpList = pair.second;
        if (callOpList.empty()) {
            continue;
        }
        // 从全局缓存中获取function
        auto cacheValue = Program::GetInstance().TryHitCahce(calleeHash);
        Function* cacheFunc = cacheValue->GetFunction();
        // 检查是否是Mix子图
        if (!IsMixSubgraph(*cacheFunc)) {
            continue;
        }
        // 分析内部组件
        auto components = AnalyzeInternalComponents(*cacheFunc);
        // 确定programID（仅在当前function中查找）
        uint64_t localProgramID = INVALID_PROGRAM_ID;
        bool isInCurrentFunc = false;
        // 检查这个function是否在当前function的programs中
        for (const auto& program : rootFunc->programs_) {
            if (program.second == cacheFunc) {
                localProgramID = program.first;
                isInCurrentFunc = true;   
                break;
            }
        }
        // 添加到待处理列表
        mixSubgraphs.push_back(MixSubgraphInfo(
            localProgramID,
            cacheFunc,
            components,
            callOpList,
            calleeHash,  
            isInCurrentFunc
        ));
        // 如果需要删除当前function中的原leaffunction
        if (isInCurrentFunc) {
            mixSubgraphIDsToDelete.insert(localProgramID);
        }          
        callOpsToDelete.insert(callOpsToDelete.end(),
                                callOpList.begin(), callOpList.end());
        ALOG_INFO_F("Found mix subgraph: local=%d, programID=%lu, callOps=%zu, components=%zu",
                isInCurrentFunc, localProgramID, callOpList.size(), components.size());
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
        if (mixInfo.isLocalFunction) {
            // 只有本地function的拆分才会在当前function中添加新program
            newSubgraphCount += mixInfo.components.size();
        }
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
    // 为新创建的子图分配连续的ID
    for (const auto& mixInfo : mixSubgraphs) {
        if (!mixInfo.isLocalFunction) {
            continue; // 跳过跨function
        }
        std::vector<uint64_t> newProgramIDs;
        for (size_t i = 0; i < mixInfo.components.size(); ++i) {
            newProgramIDs.push_back(nextProgramID++);
        }
        mixSubgraphNewIDs[mixInfo.programID] = newProgramIDs;
        ALOG_INFO_F("Allocated %zu new programIDs for local mix subgraph %lu", 
                    newProgramIDs.size(), mixInfo.programID);
    }
    return SUCCESS;
}

Status MixSubgraphSplit::ExecuteSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::vector<Operation*> callOpsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap) {
    // 执行实际的拆分
    std::vector<MixSubgraphSplitResult> splitResults;
    auto rootFunc = function.rootFunc_;
    for (const auto& mixInfo : mixSubgraphs) {
        std::vector<uint64_t> newProgramIDs;
        if (mixInfo.isLocalFunction) {
             // 本地function：使用预分配的programID
            auto it = mixSubgraphNewIDs.find(mixInfo.programID);
            if (it != mixSubgraphNewIDs.end()) {
                newProgramIDs = it->second;
            } 
        } else {
            // 跨function：创建虚拟的programID（不添加到programs中）
            static uint64_t tempIDBase = 0xFFFFFFFF00000000ULL;
            for (size_t i = 0; i < mixInfo.components.size(); ++i) {
                newProgramIDs.push_back(tempIDBase++);
            }
        }
        ProcessLeafFunction(*rootFunc, mixInfo.programID, mixInfo.function, mixInfo.components, 
                            newProgramIDs, mixInfo.originalCallOps, splitResults);
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
    // 处理数据移动op(基于OpCalcType)
    std::vector<Operation*> syncOps;
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
        if (!merged && IsSyncOperation(op)) {
            // 同步op先收集起来，稍后处理
            syncOps.push_back(op);
            ALOG_DEBUG_F("Deferring sync operation %s %d", 
                        op->GetOpcodeStr().c_str(), op->GetOpMagic());    
        } else if (!merged) {
            remainingOps.push_back(op);
        }
    }

    // 处理同步op（此时数据移动op已经分配）
    for (auto* syncOp : syncOps) {
        bool merged = MergeSyncOperation(syncOp, componentsByInternalID, opToComponentMap, mixSubgraphFunc);
        if (!merged) {
            remainingOps.push_back(syncOp);
            ALOG_DEBUG_F("Sync operation %s %d not merged", 
                        syncOp->GetOpcodeStr().c_str(), syncOp->GetOpMagic());
        }
    }
    // 报告未分配的op
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
        } else {
            // 目标op存在但尚未分配
            ALOG_DEBUG_F("Cannot merge %s %d: target op %d exists but not yet assigned (will retry later)",
                        op->GetOpcodeStr().c_str(), op->GetOpMagic(), targetOp->GetOpMagic());
            return false;
        }
    }
    ALOG_WARN_F("Failed to merge %s %d: no non-sync operation found in search direction",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
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

    bool isCubeComponent = false;
    Operation* firstNonSyncOp = nullptr;
    
    for (auto* op : operations) {
        if (!IsSyncOperation(op)) {
            firstNonSyncOp = op;
            break;
        }
    }
    
    // 使用第一个非同步op的isCube属性
    isCubeComponent = firstNonSyncOp->HasAttribute(OpAttributeKey::isCube) &&
                     firstNonSyncOp->GetBoolAttribute(OpAttributeKey::isCube);
    
    ALOG_DEBUG_F("Component type determined by first non-sync op %s (magic=%d): isCube=%d",
                firstNonSyncOp->GetOpcodeStr().c_str(),
                firstNonSyncOp->GetOpMagic(),
                isCubeComponent);
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
        if (result.originalProgramID == INVALID_PROGRAM_ID) {
            ALOG_DEBUG_F("Skip cross-function result: originalProgramID=INVALID");
            continue;
        }
        auto it = mixSubgraphNewIDs.find(result.originalProgramID);
        if (it == mixSubgraphNewIDs.end()) {
            ALOG_WARN_F("No programIDs found for result with originalProgramID=%lu", 
                          result.originalProgramID);    
            continue;
        }
        const auto& newProgramIDs = it->second;
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
                                                    CallOpCreationInfo& info) {
    ALOG_DEBUG_F("Creating callOp in root function for leaf: %s, programID=%d, component=%d, wrapId=%lu", leafFunc.GetRawName().c_str(), newProgramID, componentIndex, info.wrapId);
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
    auto extractedArgList = ExtractArgListForLeafFunction(leafFunc, originalCallAttr, invokeInfo, info.iOffsets, info.oOffsets, originalMixFunc);
    ALOG_DEBUG_F("Created callOp %d: %zu arg blocks (from original callOp %d), %zu input offsets, %zu output offsets", callOp.GetOpMagic(), extractedArgList.size(), originalCallOp->GetOpMagic(), info.iOffsets.size(), info.oOffsets.size());
    std::map<int, SymbolicScalar> outIndexToExpr;
    leafFunc.GetOutcastSymbolicExpr(outIndexToExpr);
    // 创建CallOpAttribute（使用从原始CallOp提取的argList）
    auto callAttr = leafFunc.CreateCallOpAttribute(extractedArgList, outIndexToExpr);
    auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(callAttr);
    if (callOpAttr != nullptr) {
        callOpAttr->wrapId = info.wrapId;
        ALOG_DEBUG_F("Set wrapId=%lu to callOp attribute for programID=%d (from original callOp %d)", info.wrapId, newProgramID, originalCallOp->GetOpMagic());
    }
    callOp.SetOpAttribute(callAttr);
    callOp.SetOpOffset(info.iOffsets, info.oOffsets);
    callOp.UpdateSubgraphID(newProgramID);
    if (componentIndex < subgraphToFunction.subFuncInvokeInfos.size()) {
        callOp.SetSubFuncInvokeInfo(subgraphToFunction.subFuncInvokeInfos[componentIndex]);
    }
    subgraphToFunction.SetSemanticLabel(leafFunc.GetProgramOp(), callOp);
    if (callOpAttr != nullptr && callOpAttr->invokeInfo_ != nullptr) {
        callOpAttr->invokeInfo_->UpdateProgramSubgraphId(newProgramID);
    }
    // 将创建的call op记录到info中
    info.createdCallOp = &callOp;
    ALOG_INFO_F("Successfully created callOp %d in root function for programID=%d, leaf=%s", 
                callOp.GetOpMagic(), newProgramID, leafFunc.GetRawName().c_str());
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

bool MixSubgraphSplit::ExtractArgListFromIncast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, ExtractInfo& extractInfo) const {
    //使用invokeInfo中预先构造的incast信息
    for (const auto& in : invokeInfo.GetIncastTensorParamList()) {
        if (in.opMagic == -1) { // 有效的incast
            continue;
        }
        int offset = GetOffsetFromIncastParam(in, leafFunc);
        if (offset == -1) {
            continue;
        }       
        extractInfo.iOffsets.push_back(offset);
        extractInfo.processedTensors.insert(in.tensor);
        ALOG_DEBUG_F("Incast (op=%d, idx=%d) -> original offset=%d",
                    in.opMagic, in.operandIdx, offset);
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromOutcast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, ExtractInfo& extractInfo) const {
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
        extractInfo.oOffsets.push_back(offset);
        extractInfo.processedTensors.insert(out.tensor);
        ALOG_DEBUG_F("Outcast (op=%d, idx=%d) -> original offset=%d",
                    out.opMagic, out.operandIdx, offset);
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, ExtractInfo& extractInfo) const {
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
        if (tensor.isOutputToGM) {
            extractInfo.oOffsets.push_back(offset);
            ALOG_DEBUG_F("Global tensor -> Outcast: op=%d, idx=%d -> oOffset=%d", 
                        tensor.opMagic, tensor.operandIdx, offset);
        } else {
            extractInfo.iOffsets.push_back(offset);
            ALOG_DEBUG_F("Global tensor -> Incast: op=%d, idx=%d -> iOffset=%d", 
                        tensor.opMagic, tensor.operandIdx, offset);
        }
        extractInfo.processedTensors.insert(tensor.tensor);
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromActualIncasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualIncasts, ExtractInfo& extractInfo, Function* originalMixFunc) const {
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    for (const auto& incast : actualIncasts) {
        if (incast == nullptr || extractInfo.processedTensors.count(incast) > 0) {
            continue;
        }

        // 这是传播依赖添加的参数，需要特殊处理
        ALOG_DEBUG_F("Processing propagated incast: tensor=%d", incast->GetRawMagic());

        // 在原始Mix function中查找这个tensor的offset
        int offset = FindOriginalOffsetInMixFunction(incast, originalMixFunc);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to find offset for propagated incast tensor %d!", incast->GetRawMagic());
            return false;  // 直接报错返回
        }
        extractInfo.iOffsets.push_back(offset);
        extractInfo.processedTensors.insert(incast);
        ALOG_DEBUG_F("Extracted propagated incast: tensor=%d, offset=%d",
                    incast->GetRawMagic(), offset);
    }
    return true;
}

bool MixSubgraphSplit::ExtractArgListFromActualOutcasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualOutcasts, ExtractInfo& extractInfo, Function* originalMixFunc) const {
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
        int offset = FindOriginalOffsetInMixFunction(outcast, originalMixFunc);
        if (offset == -1) {
            ALOG_ERROR_F("Failed to find offset for propagated outcast tensor %d!", outcast->GetRawMagic());
            return false;  // 直接报错返回
        }
        extractInfo.oOffsets.push_back(offset);
        extractInfo.processedTensors.insert(outcast);
            ALOG_DEBUG_F("Propagated outcast tensor %d -> original offset=%d",
                    outcast->GetRawMagic(), offset);
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
        std::vector<int>& oOffsets,
        Function* originalMixFunc) const {
    auto originalArgList = originalCallAttr->GetArgList();
    // 清空offset向量
    iOffsets.clear();
    oOffsets.clear();
    // 获取传播依赖后的实际incast/outcast
    auto actualIncasts = leafFunc.GetIncast();
    auto actualOutcasts = leafFunc.GetOutcast();

    ALOG_DEBUG_F("Leaf function %s has %zu actual incasts, %zu actual outcasts after dependency propagation",
                leafFunc.GetRawName().c_str(), actualIncasts.size(), actualOutcasts.size());
    //处理直接参数（在原始invokeInfo中能找到的）
    std::set<LogicalTensorPtr> processedTensors;

    ExtractInfo extractInfo{iOffsets, oOffsets, processedTensors};
    //使用invokeInfo中预先构造的incast信息
    if (!ExtractArgListFromIncast(invokeInfo, leafFunc, extractInfo)) {
        return {};
    }
    //使用invokeInfo中的outcast信息
    if (!ExtractArgListFromOutcast(invokeInfo, leafFunc, extractInfo)) {
        return {};
    }
    // 使用invokeInfo中的global tensor信息
    if (!ExtractArgListFromGlobalTensor(invokeInfo, leafFunc, extractInfo)) {
        return {};
    }
    // 然后处理传播依赖添加的参数（在actualIncasts中但不在InvokeInfo中）
    if (!ExtractArgListFromActualIncasts(actualIncasts, extractInfo, originalMixFunc)) {
        return {};
    }
    // 处理传播依赖添加的outcast参数（在actualOutcasts中但不在InvokeInfo中）
    if (!ExtractArgListFromActualOutcasts(actualOutcasts, extractInfo, originalMixFunc)) {
        return {};
    }
    return originalArgList;
}

Status MixSubgraphSplit::SetOffsetsToLeafFunction(Function& leafFunc, const std::vector<int>& iOffsets, const std::vector<int> &oOffsets, const SubfuncInvokeInfoTy& invokeInfo) {
    ALOG_DEBUG_F("Setting offsets to leaf function %s, %zu iOffsets, %zu oOffsets",
                leafFunc.GetRawName().c_str(), iOffsets.size(), oOffsets.size());
    // 获取该leaf function的magic映射
    auto mapIt = leafFuncMagicMaps_.find(&leafFunc);
    const auto* magicMap = (mapIt != leafFuncMagicMaps_.end()) ? 
                          &mapIt->second.originalToClonedMagic : nullptr;
    int iOffsetIndex = 0;
    int oOffsetIndex = 0;
    // 设置incast的offset
    for (const auto& in : invokeInfo.GetIncastTensorParamList()) {
        if (in.opMagic != -1 && iOffsetIndex < static_cast<int>(iOffsets.size())) {
            int opMagicToFind = in.opMagic;
            if (magicMap) {
                auto it = magicMap->find(in.opMagic);
                if (it != magicMap->end()) {
                    opMagicToFind = it->second;
                    ALOG_DEBUG_F("Mapping incast op magic %d -> %d", in.opMagic, opMagicToFind);
                }
            }
            
            if (SetOffsetToOpByMagic(opMagicToFind, in.operandIdx, iOffsets[iOffsetIndex], leafFunc, false)) {
                ALOG_DEBUG_F("Set offset %d to op %d input[%d] for incast",
                            iOffsets[iOffsetIndex], in.opMagic, in.operandIdx);
            }
            iOffsetIndex++;
        }
    }
    // 设置global tensor作为输入的offset（使用iOffsets）
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic != -1 && !tensorParam.isOutputToGM) {
            int opMagicToFind = tensorParam.opMagic;
            if (magicMap) {
                auto it = magicMap->find(tensorParam.opMagic);
                if (it != magicMap->end()) {
                    opMagicToFind = it->second;
                    ALOG_DEBUG_F("Mapping global input op magic %d -> %d", tensorParam.opMagic, opMagicToFind);
                }
            }
            
            int offset = iOffsets[iOffsetIndex];
            if (SetOffsetToOpByMagic(opMagicToFind, tensorParam.operandIdx, offset, leafFunc, false)) {
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
            int opMagicToFind = out.opMagic;
            if (magicMap) {
                auto it = magicMap->find(out.opMagic);
                if (it != magicMap->end()) {
                    opMagicToFind = it->second;
                    ALOG_DEBUG_F("Mapping outcast op magic %d -> %d", out.opMagic, opMagicToFind);
                }
            }
            
            if (SetOffsetToOpByMagic(opMagicToFind, out.operandIdx, oOffsets[oOffsetIndex], leafFunc, true)) {
                ALOG_DEBUG_F("Set offset %d to op %d output[%d] for outcast",
                            oOffsets[oOffsetIndex], out.opMagic, out.operandIdx);
            }
            oOffsetIndex++;
        }
    }
    // 设置global tensor作为输出的offset（使用oOffsets）
    for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
        if (tensorParam.opMagic != -1 && tensorParam.isOutputToGM) {
            int opMagicToFind = tensorParam.opMagic;
            if (magicMap) {
                auto it = magicMap->find(tensorParam.opMagic);
                if (it != magicMap->end()) {
                    opMagicToFind = it->second;
                    ALOG_DEBUG_F("Mapping global output op magic %d -> %d", tensorParam.opMagic, opMagicToFind);
                }
            }
            
            int offset = oOffsets[oOffsetIndex];
            if (SetOffsetToOpByMagic(opMagicToFind, tensorParam.operandIdx, offset, leafFunc, true)) {
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

int MixSubgraphSplit::FindOriginalOffsetInMixFunction(LogicalTensorPtr tensor, Function* originalMixFunc) const {
    if (tensor == nullptr || originalMixFunc == nullptr) {
        ALOG_ERROR_F("Tensor or function is nullptr in FindOriginalOffsetInMixFunction");
        return -1;
    }
    int rawMagic = tensor->GetRawMagic();
    ALOG_DEBUG_F("Finding original offset for tensor raw magic=%d in function %s", 
            rawMagic, originalMixFunc->GetRawName().c_str());
    auto operations = originalMixFunc->Operations(false);
    for (auto& op : operations) {
        if (op.IsNOP()) {
            continue;
        }
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

int MixSubgraphSplit::GetOffsetFromIncastParam(const SubfuncInvokeInfoTy::IncastParamPackTy& incastParam, Function& leafFunc) const {
    // 首先尝试通过映射表查找
    auto mapIt = leafFuncMagicMaps_.find(&leafFunc);
    if (mapIt == leafFuncMagicMaps_.end()) {
        ALOG_ERROR_F("No magic mapping found for leaf function %s", 
                    leafFunc.GetRawName().c_str());
        return -1;
    }
    const auto& magicMap = mapIt->second.originalToClonedMagic;
    auto magicIt = magicMap.find(incastParam.opMagic);
    if (magicIt == magicMap.end()) {
        ALOG_ERROR_F("No magic mapping for original magic %d in leaf function %s (incast)",
                    incastParam.opMagic, leafFunc.GetRawName().c_str());
        return -1;
    }    
    int mappedMagic = magicIt->second;
    ALOG_DEBUG_F("Mapping original magic %d to %d for leaf function", 
                incastParam.opMagic, mappedMagic);
    // 用映射后的magic查找在leafFunc中查找对应的op
    return GetOffsetFromOp(mappedMagic, incastParam.operandIdx, leafFunc, false);
}

int MixSubgraphSplit::GetOffsetFromOutcastParam(const SubfuncInvokeInfoTy::OutcastParamPackTy& outcastParam, Function& leafFunc) const {
    // 获取该leaf function的magic映射
    auto mapIt = leafFuncMagicMaps_.find(&leafFunc);
    if (mapIt == leafFuncMagicMaps_.end()) {
        ALOG_ERROR_F("No magic mapping found for leaf function %s", 
                    leafFunc.GetRawName().c_str());
        return -1;
    }
    const auto& magicMap = mapIt->second.originalToClonedMagic;
    // 查找对应的新magic
    auto magicIt = magicMap.find(outcastParam.opMagic);
    if (magicIt == magicMap.end()) {
        ALOG_ERROR_F("No magic mapping for original magic %d in leaf function %s (outcast)",
                    outcastParam.opMagic, leafFunc.GetRawName().c_str());
        return -1;
    }
    int mappedMagic = magicIt->second;
    ALOG_DEBUG_F("Mapping outcast op magic %d -> %d", outcastParam.opMagic, mappedMagic);
    // 使用映射后的magic查找offset
    return GetOffsetFromOp(mappedMagic, outcastParam.operandIdx, leafFunc, true);
}

// 统一的offset获取函数
int MixSubgraphSplit::GetOffsetFromOp(int opMagic, int operandIdx, 
                                     Function& leafFunc, bool isOutput) const {
    auto operations = leafFunc.Operations(false);
    for (auto& op : operations) {
        if (op.GetOpMagic() == opMagic) {
            if (isOutput) {
                if (operandIdx >= 0 && static_cast<size_t>(operandIdx) < op.GetOOperands().size()) {
                    int offset = op.GetOOpAttrOffset(operandIdx);
                    if (offset != -1) {
                        ALOG_DEBUG_F("Found output offset %d for op %d idx %d", 
                                    offset, opMagic, operandIdx);
                    }
                    return offset;
                }
            } else {
                if (operandIdx >= 0 && static_cast<size_t>(operandIdx) < op.GetIOperands().size()) {
                    int offset = op.GetIOpAttrOffset(operandIdx);
                    if (offset != -1) {
                        ALOG_DEBUG_F("Found input offset %d for op %d idx %d", 
                                    offset, opMagic, operandIdx);
                    }
                    return offset;
                }
            }
        }
    }
    
    ALOG_WARN_F("Could not find offset for op %d idx %d (isOutput=%d)", 
                opMagic, operandIdx, isOutput);
    return -1;
}

int MixSubgraphSplit::GetOffsetFromTensorParam(const SubfuncInvokeInfoTy::TensorParamPackTy& tensorParam, Function& leafFunc) const {
    // 获取leafFunc对应的magic映射
    auto it = leafFuncMagicMaps_.find(&leafFunc);
    if (it == leafFuncMagicMaps_.end()) {
        ALOG_ERROR_F("No magic mapping found for leaf function");
        return -1;
    }
    
    const auto& magicMap = it->second.originalToClonedMagic;
    
    // 查找对应的新magic
    auto magicIt = magicMap.find(tensorParam.opMagic);
    if (magicIt == magicMap.end()) {
        ALOG_ERROR_F("No magic mapping for original magic %d in function", tensorParam.opMagic);
        return -1;
    }
    
    int newMagic = magicIt->second;
    // 对于global tensor，根据isOutputToGM判断是输入还是输出
    if (tensorParam.isOutputToGM) {
        // 作为输出处理
        auto operations = leafFunc.Operations(false);
        for (auto& op : operations) {
            if (op.GetOpMagic() != newMagic) {
                continue;
            }
            if (tensorParam.operandIdx >= 0 && static_cast<size_t>(tensorParam.operandIdx) < op.GetOOperands().size()) {
                int offset = op.GetOOpAttrOffset(tensorParam.operandIdx);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found output offset %d for global tensor (op=%d, idx=%d)",
                                offset, newMagic, tensorParam.operandIdx);
                    return offset;
                }
            }
            break;
        }
    } else {
        // 作为输入处理
        auto operations = leafFunc.Operations(false);
        for (auto& op : operations) {
            if (op.GetOpMagic() != newMagic) {
                continue;
            }
            if (tensorParam.operandIdx >= 0 && static_cast<size_t>(tensorParam.operandIdx) < op.GetIOperands().size()) {
                int offset = op.GetIOpAttrOffset(tensorParam.operandIdx);
                if (offset != -1) {
                    ALOG_DEBUG_F("Found input offset %d for global tensor (op=%d, idx=%d)",
                                offset, newMagic, tensorParam.operandIdx);
                    return offset;
                }
            }
            break;
        }
    }

    ALOG_WARN_F("Could not find offset for global tensor (op=%d, idx=%d, isOutput=%d)",
                newMagic, tensorParam.operandIdx, tensorParam.isOutputToGM);
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
    std::unordered_map<int, int> magicMap; // 原始magic -> 新magic
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
            // 判断是否为同步op
            bool isSyncOp = IsSyncOperation(originalOp);
            if (isSyncOp) {
                // 对于同步op，直接使用原始op的shared_ptr
                std::shared_ptr<Operation> opPtr = originalOp->shared_from_this();
                programOps.push_back(opPtr);
                int originalMagic = originalOp->GetOpMagic();
                magicMap[originalMagic] = originalMagic;
                ALOG_DEBUG_F("Reuse sync op %d in leaf function %s",
                        originalMagic, leafName.c_str());
            } else {
                // 对于非同步op，进行克隆
                auto iOperands = originalOp->GetIOperands();
                auto oOperands = originalOp->GetOOperands();
                Operation& clonedOp = originalOp->CloneOperation(*newFunc, iOperands, oOperands);
                // 记录映射关系
                int originalMagic = originalOp->GetOpMagic();
                int clonedMagic = clonedOp.GetOpMagic();
                magicMap[originalMagic] = clonedMagic;
                // 复制offset信息
                for (size_t idx = 0; idx < iOperands.size(); ++idx) {
                    int offset = originalOp->GetIOpAttrOffset(idx);
                    if (offset != -1) {
                        clonedOp.SetIOpAttrOffset(idx, offset);
                    }
                }
                for (size_t idx = 0; idx < oOperands.size(); ++idx) {
                    int offset = originalOp->GetOOpAttrOffset(idx);
                    if (offset != -1) {
                        clonedOp.SetOOpAttrOffset(idx, offset);
                    }
                }
                programOps.push_back(clonedOp.shared_from_this());
                ALOG_DEBUG_F("Cloned op %d to leaf function %s (original order preserved)",
                                originalOp->GetOpMagic(), leafName.c_str());
            }
        }
    }
    // 保存映射关系
    LeafFuncMagicMap leafMap;
    leafMap.leafFunc = newFunc.get();
    leafMap.originalToClonedMagic = std::move(magicMap);
    leafFuncMagicMaps_[newFunc.get()] = std::move(leafMap);
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
    ALOG_DEBUG_F("Function %s computed hash: %lu", leafName.c_str(), funcHash.GetHash());
    Program::GetInstance().GetFunctionCache().Insert(funcHash, *newFunc);
    ALOG_DEBUG_F("Inserted new function %s into function cache with hash %lu",
                leafName.c_str(), funcHash.GetHash());
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
    std::unordered_set<Function*> processedLeafFuncs;
    // 在所有callOp都创建完成后，统一设置offset到leaffunction中的op
    for (size_t i = 0; i < callOpInfos.size(); ++i) {
        const auto& info = callOpInfos[i];
        auto* leafFunc = info.leafFunc;
        
        // 每个leaf function只设置一次offsets
        if (processedLeafFuncs.find(leafFunc) != processedLeafFuncs.end()) {
            continue;
        }
        
        // 设置offset到leaf function中的op
        auto status = SetOffsetsToLeafFunction(*leafFunc, info.iOffsets, info.oOffsets,
                                                subgraphToFunction.subFuncInvokeInfos[info.componentIndex]);
        if (status != SUCCESS) {
            ALOG_ERROR_F("Failed to set offsets for leaf function %s", 
                        leafFunc->GetRawName().c_str());
            return FAILED;
        }
        
        processedLeafFuncs.insert(leafFunc);
        ALOG_DEBUG_F("Set offsets for leaf function %s (component %d)", 
                    leafFunc->GetRawName().c_str(), info.componentIndex);
    }
    ALOG_INFO_F("Successfully created %zu call operations with internal dependencies", 
               callOpInfos.size());
    return SUCCESS;
}

// 添加内部依赖的depend operands
void MixSubgraphSplit::ProcessAllInternalDependencies(
    Function& rootFunc,
    const std::vector<CallOpCreationInfo>& callOpInfos,
    const std::vector<InternalDependencyInfo>& internalDeps) const {
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
void MixSubgraphSplit::ProcessInternalDependenciesForWrap(
    Function& rootFunc,
    const std::vector<const CallOpCreationInfo*>& infos,
    const std::vector<InternalDependencyInfo>& internalDeps,
    uint64_t wrapId) const {
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
        LogicalTensorPtr dependTensor;
        dependTensor = std::make_shared<LogicalTensor>(
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
                wrapId,
                srcComp, dstComp,
                producerCallOp->GetOpMagic(), consumerCallOp->GetOpMagic(),
                dependTensor->magic,
                dependTensor->GetProducers().size(),
                dependTensor->GetConsumers().size());
    }
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

// 每个mixLeafFunction的处理
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

    MixResourceType resourceType = GetMixResourceType(*originalMixFunc);
    ALOG_DEBUG_F("Mix resource type: %d for programID=%d", static_cast<int>(resourceType), programID);
    
    // 从原始mix函数获取isUnderDynamicFunction属性
    bool isUnderDynamicFunction = originalMixFunc->IsUnderDynamicFunction();
    ALOG_DEBUG_F("Original mix function isUnderDynamicFunction: %s for programID=%d", 
                 isUnderDynamicFunction ? "true" : "false", programID);
    
    uint64_t mixId = nextMixId_++;
    ALOG_DEBUG_F("Assigning mixId=%lu for original mix function programID=%d", mixId, programID);
   
    SubgraphToFunction subgraphToFunction = InitSubgraphToFunction(components);

    // 步骤1：记录直接的incast/outcast(Mix子图整体与外部的依赖)
    ALOG_INFO_F("Step 1: Recording direct incast/outcast...");
    InOutCastRecord(subgraphToFunction, originalMixFunc);
    // 步骤2：分析组件间直接依赖（scope与scope之间的依赖）
    ALOG_INFO_F("Step 2: Analyzing inter-component dependencies...");
    auto directDeps = AnalyzeComponentDependencies(*originalMixFunc);
    // 步骤3：计算依赖传递闭包
    ALOG_INFO_F("Step 3: Computing dependency closure...");
    auto dependencyClosure = ComputeDependencyClosure(directDeps);    
    // 步骤4：为每个scope创建leaf function
    std::vector<Function*> newFunctions;
    if (GenNewFunctions(rootFunc, originalMixFunc, components, newProgramIDs, subgraphToFunction, newFunctions) != SUCCESS) {
        return FAILED;
    }

    // 设置每个新建leaf function的isUnderDynamicFunction属性
    ALOG_INFO_F("Setting isUnderDynamicFunction for %zu new leaf functions", newFunctions.size());
    for (auto* leafFunc : newFunctions) {
        if (leafFunc) {
            leafFunc->SetUnderDynamicFunction(isUnderDynamicFunction);
            ALOG_DEBUG_F("Set isUnderDynamicFunction=%s for leaf function programID=%d",
                         isUnderDynamicFunction ? "true" : "false", leafFunc->GetProgramId());
        }
    }
    
    // 步骤5：计算所有依赖（包括外部依赖和内部依赖）
    ALOG_INFO_F("Step 5: Computing all dependencies...");
    // 5.1：提取外部依赖（从subgraphToFunction）
    std::unordered_map<int, std::vector<SimpleIncastParam>> allIncasts;
    std::unordered_map<int, std::vector<SimpleOutcastParam>> allOutcasts;
    ExtractExternalDependencies(subgraphToFunction, allIncasts, allOutcasts);
    // 5.2：基于传递闭包传播外部依赖到内部scope
    PropagateExternalDependenciesWithClosure(allIncasts, allOutcasts, dependencyClosure);
    
    // 构建scope类型映射
    std::unordered_map<int, ComponentType> componentTypes;
    for (size_t i = 0; i < components.size(); i++) {
        componentTypes[i] = DetermineComponentType(components[i]);
    }
    // 5.3：添加内部同类型scope之间的依赖（只收集C-C、V-V的依赖）
    std::vector<InternalDependencyInfo> internalDeps;
    CollectInternalDependencies(dependencyClosure, internalDeps, componentTypes);
    
    // 步骤6：消除冗余依赖
    ALOG_INFO_F("Step 6: Eliminating redundant dependencies...");
    EliminateRedundantDependencies(allIncasts, allOutcasts, internalDeps);
    // 步骤7：应用最终的依赖到leaf functions（外部依赖）
    ALOG_INFO_F("Step 7: Applying final dependencies to leaf functions...");
    ApplyFinalDependencies(newFunctions, allIncasts, allOutcasts);
    
    // 为每个原始CallOp创建一组新的callOp, 每个原始callOp使用不同的wrapId（包含dummyTensor依赖）
    if (CreateCallOps(rootFunc, originalCallOps, originalMixFunc, components, newProgramIDs, subgraphToFunction, newFunctions, internalDeps) != SUCCESS) {
        return FAILED;
    }
    
    // 复制InferParamIndex信息到所有新子图
    if (CopyInferParamIndexInfo(originalMixFunc, newFunctions) != SUCCESS) {
        return FAILED;
    }
    // 设置wrapId和resourceType
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

ComponentType MixSubgraphSplit::DetermineComponentType(const InternalComponentInfo& component) const 
{
    if (component.operations.empty()) {
        ALOG_WARN_F("Empty component, cannot determine type");
        return ComponentType::UNKNOWN;
    }
    // 遍历所有非同步op，查找isCube属性
    for (auto* op : component.operations) {
        // 跳过同步op
        if (IsSyncOperation(op)) {
            ALOG_DEBUG_F("Skipping sync op %d (opcode=%s)", 
                        op->GetOpMagic(), op->GetOpcodeStr().c_str());
            continue;
        }
        // 检查isCube属性
        if (op->HasAttribute(OpAttributeKey::isCube)) {
            bool isCube = op->GetBoolAttribute(OpAttributeKey::isCube);
            if (isCube) {
                ALOG_DEBUG_F("Component %zu determined as C_SCOPE (non-sync op %d has isCube=true)", 
                            component.suffix, op->GetOpMagic());
                return ComponentType::C_SCOPE;
            }
        }
        ALOG_DEBUG_F("Component %zu determined as V_SCOPE (non-sync op %d has isCube=false or no isCube attr)", 
                    component.suffix, op->GetOpMagic());
        return ComponentType::V_SCOPE;
    }
    // 如果所有操作都是同步操作
    ALOG_DEBUG_F("Component %zu has only sync operations (%zu ops)", 
                component.suffix, component.operations.size());
    return ComponentType::UNKNOWN;
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
                if (consumer->GetSubgraphID() != op.GetSubgraphID()) continue;
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

// 从subgraphToFunction中提取外部依赖
void MixSubgraphSplit::ExtractExternalDependencies(
    const SubgraphToFunction& subgraphToFunction,
    std::unordered_map<int, std::vector<SimpleIncastParam>>& allIncasts,
    std::unordered_map<int, std::vector<SimpleOutcastParam>>& allOutcasts) const {
    for (size_t i = 0; i < subgraphToFunction.subFuncInvokeInfos.size(); i++) {
        const auto& invokeInfo = subgraphToFunction.subFuncInvokeInfos[i];
        // 提取incast
        for (const auto& incast : invokeInfo.GetIncastTensorParamList()) {
            allIncasts[i].emplace_back(incast.tensor, incast.opMagic, incast.operandIdx);
        }
        // 提取global tensor作为输入
        for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
            if (!tensorParam.isOutputToGM) {
                allIncasts[i].emplace_back(tensorParam.tensor, tensorParam.opMagic, tensorParam.operandIdx);                
            }
        }
        // 提取outcast
        for (const auto& outcast : invokeInfo.GetOutcastTensorParamList()) {
            allOutcasts[i].emplace_back(outcast.tensor, outcast.opMagic, outcast.operandIdx);
        }
        // 提取global tensor作为输出
        for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
            if (tensorParam.isOutputToGM) {
                allOutcasts[i].emplace_back(tensorParam.tensor, tensorParam.opMagic, tensorParam.operandIdx);
            }
        }
    }
}

// 传播外部依赖到内部scope（基于传递闭包）
void MixSubgraphSplit::PropagateExternalDependenciesWithClosure(
    std::unordered_map<int, std::vector<SimpleIncastParam>>& allIncasts,
    std::unordered_map<int, std::vector<SimpleOutcastParam>>& allOutcasts,
    const std::unordered_map<int, std::set<int>>& dependencyClosure) const {    
    // 基于传递闭包传播依赖
    for (const auto& [sourceComp, targets] : dependencyClosure) {
        // 传播incast：source的incast传播给所有依赖它的target
        auto incastIt = allIncasts.find(sourceComp);
        if (incastIt != allIncasts.end()) {
            for (int targetComp : targets) {
                if (targetComp >= 0) {
                    for (const auto& incastParam : incastIt->second) {
                        if (!ContainsIncast(allIncasts[targetComp], incastParam.tensor)) {
                            allIncasts[targetComp].push_back(incastParam);
                        }
                    }
                }
            }
        }
        // 传播outcast：target的outcast反向传播给所有source
        for (int targetComp : targets) {
            auto outcastIt = allOutcasts.find(targetComp);
            if (outcastIt != allOutcasts.end()) {
                for (const auto& outcastParam : outcastIt->second) {
                    if (!ContainsOutcast(allOutcasts[sourceComp], outcastParam.tensor)) {
                        allOutcasts[sourceComp].push_back(outcastParam);
                    }
                }
            }
        }
    }
}

// 检查incast列表中是否包含指定的tensor
bool MixSubgraphSplit::ContainsIncast(
    const std::vector<SimpleIncastParam>& incasts,
    LogicalTensorPtr tensor) const {
    
    if (!tensor) return false;
    
    for (const auto& incast : incasts) {
        if (incast.tensor == tensor) {
            return true;
        }
    }
    
    return false;
}

// 检查outcast列表中是否包含指定的tensor
bool MixSubgraphSplit::ContainsOutcast(
    const std::vector<SimpleOutcastParam>& outcasts,
    LogicalTensorPtr tensor) const {
    
    if (!tensor) return false;
    
    for (const auto& outcast : outcasts) {
        if (outcast.tensor == tensor) {
            return true;
        }
    }
    
    return false;
}

// 添加内部scope之间的依赖（包括直接依赖和传递依赖）
void MixSubgraphSplit::CollectInternalDependencies(
    const std::unordered_map<int, std::set<int>>& dependencyClosure,
    std::vector<InternalDependencyInfo>& internalDeps,
    const std::unordered_map<int, ComponentType>& componentTypes) const {    
    // 遍历传递闭包中的每个依赖关系
    for (const auto& [srcComp, dstComps] : dependencyClosure) {
        ComponentType srcType = componentTypes.at(srcComp);
        if (srcType == ComponentType::UNKNOWN) continue;
        for (int dstComp : dstComps) {
            // 跳过自依赖（scope依赖自己）
            if (srcComp == dstComp) {
                ALOG_DEBUG_F("Skip self-dependency: component %d -> %d", srcComp, dstComp);
                continue;    
            }
            ComponentType dstType = componentTypes.at(dstComp);
            // 只添加同类型scope间的依赖（C-C、V-V）   
            if (srcType == dstType && dstType != ComponentType::UNKNOWN) {
                // 添加这两个组件间的tensor依赖
                InternalDependencyInfo depInfo(srcComp, dstComp, srcType);
                internalDeps.push_back(depInfo);
                ALOG_DEBUG_F("Added internal dependency: component %d (%s) -> component %d (%s)",
                           srcComp, srcType == ComponentType::C_SCOPE ? "C" : "V",
                           dstComp, dstType == ComponentType::C_SCOPE ? "C" : "V");
            }
        }
    } 
    ALOG_INFO_F("Collected %zu internal dependencies between same-type components", 
               internalDeps.size());        
}  

// 消除冗余依赖
void MixSubgraphSplit::EliminateRedundantDependencies(
    std::unordered_map<int, std::vector<SimpleIncastParam>>& allIncasts,
    std::unordered_map<int, std::vector<SimpleOutcastParam>>& allOutcasts,
    const std::vector<InternalDependencyInfo>& internalDeps) const {
    ALOG_INFO_F("Eliminating redundant dependencies...");
    // 消除冗余incast
    EliminateRedundantIncasts(allIncasts, internalDeps);
    // 消除冗余outcast
    EliminateRedundantOutcasts(allOutcasts, internalDeps);
}   

// 消除冗余incast
void MixSubgraphSplit::EliminateRedundantIncasts(
    std::unordered_map<int, std::vector<SimpleIncastParam>>& allIncasts,
    const std::vector<InternalDependencyInfo>& internalDeps) const {
    // 步骤1：按tensor分组，找出每个外部tensor被传播到了哪些组件
    std::unordered_map<LogicalTensorPtr, std::set<int>> tensorToComponents;
    for (const auto& [compId, incasts] : allIncasts) {
        for (const auto& incast : incasts) {
            if (incast.tensor) {
                tensorToComponents[incast.tensor].insert(compId);
            }
        }
    }
    // 步骤2：构建同类型依赖的快速查找表
    std::unordered_map<int, std::set<int>> sameTypeReachable;
    for (const auto& dep : internalDeps) {
        sameTypeReachable[dep.srcComp].insert(dep.dstComp);
    }
    // 步骤3：对于每个tensor，找出所有接收它的组件，分析哪些是冗余的
    for (const auto& item : tensorToComponents) {
        const auto& tensor = item.first;
        const auto& compSet = item.second;
        if (compSet.size() <= 1) continue;  
        std::vector<int> comps(compSet.begin(), compSet.end());
        std::vector<bool> isRedundant(comps.size(), false);
        for (size_t i = 0; i < comps.size(); i++) {
            int compB = comps[i];
            // 检查是否存在其他同类型组件可以到达compB
            for (size_t j = 0; j < comps.size(); j++) {
                if (i == j) continue;
                int compA = comps[j];
                // 直接查找：compA是否可以通过同类型依赖到达compB
                auto reachableIt = sameTypeReachable.find(compA);
                if (reachableIt != sameTypeReachable.end() && 
                    reachableIt->second.find(compB) != reachableIt->second.end()) {
                    // compA可以通过同类型scope依赖到达compB
                    isRedundant[i] = true;
                    ALOG_DEBUG_F("  Component %d's incast is redundant: "
                                "can be obtained from component %d via same-type dependency",
                                compB, compA);
                    break;
                }
            }
        }
        // 删除冗余incast
        for (size_t i = 0; i < comps.size(); i++) {
            if (isRedundant[i]) {
                int compId = comps[i];
                auto& incasts = allIncasts[compId];       
                auto newEnd = std::remove_if(incasts.begin(), incasts.end(),
                    [tensor](const SimpleIncastParam& param) {
                        return param.tensor == tensor;
                    });
                incasts.erase(newEnd, incasts.end());
                ALOG_DEBUG_F("Removed redundant incast for tensor %d from component %d",
                           tensor->GetRawMagic(), compId);
            }
        }
    }
}
    
void MixSubgraphSplit::EliminateRedundantOutcasts(
    std::unordered_map<int, std::vector<SimpleOutcastParam>>& allOutcasts,  
    const std::vector<InternalDependencyInfo>& internalDeps) const {
    // 步骤1：按tensor分组，找出每个输出tensor来自哪些组件
    std::unordered_map<LogicalTensorPtr, std::set<int>> tensorFromComponents;
    
    for (const auto& [compId, outcasts] : allOutcasts) {
        for (const auto& outcast : outcasts) {
            if (outcast.tensor) {
                tensorFromComponents[outcast.tensor].insert(compId);
            }
        }
    }
    // 步骤2：构建同类型依赖的快速查找表
    std::unordered_map<int, std::set<int>> sameTypeReachable;
    for (const auto& dep : internalDeps) {
        sameTypeReachable[dep.srcComp].insert(dep.dstComp);
    }
    // 步骤3：对于每个tensor，分析哪些scope的outcast是冗余的
    for (const auto& item : tensorFromComponents) {
        const auto& tensor = item.first;
        const auto& compSet = item.second;
        if (compSet.size() <= 1) continue;
        std::vector<int> comps(compSet.begin(), compSet.end());
        std::vector<bool> isRedundant(comps.size(), false);
        for (size_t i = 0; i < comps.size(); i++) {
            int compA = comps[i];  // 被检查的源组件
            // 检查compA是否可以通过同类型依赖将tensor传递给其他组件
            for (size_t j = 0; j < comps.size(); j++) {
                if (i == j) continue;
                int compB = comps[j];  // 可能的目标组件
                // 直接查找：compA是否可以通过同类型依赖到达compB
                auto reachableIt = sameTypeReachable.find(compA);
                if (reachableIt != sameTypeReachable.end() && 
                    reachableIt->second.find(compB) != reachableIt->second.end()) {
                    // compA可以通过同类型scope依赖到达compB
                    isRedundant[i] = true;
                    ALOG_DEBUG_F("  Component %d's outcast (tensor %d) is redundant: "
                               "tensor can be output via component %d via same-type dependency",
                               compA, tensor->GetRawMagic(), compB);
                    break;
                }
            }
        }
        // 删除冗余outcast
        for (size_t i = 0; i < comps.size(); i++) {
            if (isRedundant[i]) {
                int compId = comps[i];
                auto& outcasts = allOutcasts[compId];
                
                auto newEnd = std::remove_if(outcasts.begin(), outcasts.end(),
                    [tensor](const SimpleOutcastParam& param) {
                        return param.tensor == tensor;
                    });
                
                outcasts.erase(newEnd, outcasts.end());                
                ALOG_DEBUG_F("Removed redundant outcast: tensor %d from component %d",
                           tensor->GetRawMagic(), compId);
            }
        }
    }
}

void MixSubgraphSplit::ApplyFinalDependencies(
    const std::vector<Function*>& newFunctions,
    const std::unordered_map<int, std::vector<SimpleIncastParam>>& allIncasts,
    const std::unordered_map<int, std::vector<SimpleOutcastParam>>& allOutcasts) const {
    ALOG_INFO_F("Applying final dependencies to %zu leaf functions", newFunctions.size());
    for (size_t i = 0; i < newFunctions.size(); i++) {
        Function* leafFunc = newFunctions[i];
        if (!leafFunc) continue;        
        // 应用incast依赖
        auto incastIt = allIncasts.find(i);
        if (incastIt != allIncasts.end()) {
            ApplyIncastDependencies(leafFunc, i, incastIt->second);
        } 
        // 应用outcast依赖
        auto outcastIt = allOutcasts.find(i);
        if (outcastIt != allOutcasts.end()) {
            ApplyOutcastDependencies(leafFunc, i, outcastIt->second);
        } 
    }
}

// 应用incast依赖
void MixSubgraphSplit::ApplyIncastDependencies(
    Function* leafFunc,
    int componentId,
    const std::vector<SimpleIncastParam>& incastParams) const {
    if (!leafFunc) return;
    // 获取当前已有的incast，用于去重
    const auto& existingIncasts = leafFunc->GetIncast();
    std::unordered_set<uint32_t> existingMagicSet;
    
    for (const auto& tensor : existingIncasts) {
        if (tensor) {
            existingMagicSet.insert(tensor->magic);
        }
    }
    for (const auto& param : incastParams) {
        if (!param.tensor) {
            ALOG_WARN_F("Component %d: Null tensor in incast params, skipping", componentId);
            continue;
        } 
        // 检查是否已经存在相同tensor（按magic）
        if (existingMagicSet.find(param.tensor->magic) != existingMagicSet.end()) {
            ALOG_DEBUG_F("Component %d: Tensor %d already in incast list, skipping",
                        componentId, param.tensor->GetRawMagic());
            continue;
        }
        // 添加新的incast
        leafFunc->AppendIncast(param.tensor, param.opMagic, param.operandIdx);   
        existingMagicSet.insert(param.tensor->magic);
        ALOG_DEBUG_F("Component %d: Added incast - tensor %d (opMagic=%d, operandIdx=%d)",
                    componentId, param.tensor->GetRawMagic(), 
                    param.opMagic, param.operandIdx);
    }
}

// 应用outcast依赖
void MixSubgraphSplit::ApplyOutcastDependencies(
    Function* leafFunc,
    int componentId,
    const std::vector<SimpleOutcastParam>& outcastParams) const {
    
    if (!leafFunc) return;
    // 获取当前已有的outcast，用于去重
    const auto& existingOutcasts = leafFunc->GetOutcast();
    std::unordered_set<uint32_t> existingMagicSet;
    
    for (const auto& tensor : existingOutcasts) {
        if (tensor) {
            existingMagicSet.insert(tensor->magic);
        }
    }
    for (const auto& param : outcastParams) {
        if (!param.tensor) {
            ALOG_WARN_F("Component %d: Null tensor in outcast params, skipping", componentId);
             continue;
        }
        
        // 检查是否已经存在相同tensor（按magic）
        if (existingMagicSet.find(param.tensor->magic) != existingMagicSet.end()) {
            ALOG_DEBUG_F("Component %d: Tensor %d already in outcast list, skipping",
                        componentId, param.tensor->GetRawMagic());
            continue;
        }
        
        // 添加新的outcast
        leafFunc->AppendOutcast(param.tensor, param.opMagic, param.operandIdx);
        existingMagicSet.insert(param.tensor->magic);
        ALOG_DEBUG_F("Component %d: Added outcast - tensor %d (opMagic=%d, operandIdx=%d)",
                    componentId, param.tensor->GetRawMagic(), 
                    param.opMagic, param.operandIdx);
    }
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