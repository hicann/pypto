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
 * \file mix_dependency_analyzer.CPP
 * \brief
 */

#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_dependency_analyzer.h"

namespace npu {
namespace tile_fwk {
void MixDependencyAnalyzer::InitSubgraphToFunction(const std::vector<InternalComponentInfo>& components) {
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
}

void MixDependencyAnalyzer::InOutCastRecord(Function* originalMixFunc) {
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

std::unordered_map<int, std::set<int>> MixDependencyAnalyzer::AnalyzeComponentDependencies(Function &mixFunc) {
    std::unordered_map<int, std::set<int>> dependencies;
    // 分析子图的所有的op
    for (auto &op : mixFunc.Operations(false)) {
        if (op.IsNOP()) {
            continue;
        }
        int producerInternalID = op.GetInternalSubgraphID();
        for (size_t k = 0; k < op.GetOOperands().size(); k++) {
            auto oOperand = op.GetOOperands()[k];
            if (oOperand == nullptr) {
                continue;
            }
            // 分析该op的输出tensor的消费者
            auto consumers = oOperand->GetConsumers();
            for (auto* consumer : consumers) {
                if (consumer == nullptr) {
                    continue;
                }
                // 要求消费者也在同一个subgraph中
                if (consumer->GetSubgraphID() != op.GetSubgraphID()) {
                    continue;
                }
                // 记录进一步切分后的依赖关系
                int consumerID = consumer->GetInternalSubgraphID();
                if (producerInternalID != consumerID) {
                    dependencies[producerInternalID].insert(consumerID);
                }
            }
        }
    }
    // 用于记录mixsplit间的依赖关系
    return dependencies;
}

void MixDependencyAnalyzer::InitDependencies(std::unordered_map<int, std::set<int>> &dependencies) {
    // 记录最大的mixsplit id
    maxComponent = 0;
    for (const auto &pair : dependencies) {
        if (pair.first > maxComponent) {
            maxComponent = pair.first;
        }
        for (int dep : pair.second) {
            if (dep > maxComponent) {
                maxComponent = dep;
            }
        }
    }
    // 确保所有组件索引都在closure中
    for (int i = 0; i <= maxComponent; i++) {
        dependencies[i]; // 确保存在，即使没有依赖关系
    }
}

void MixDependencyAnalyzer::WarshallAlgorithm(std::vector<std::vector<bool>> &matrix) {
    size_t n = matrix.size();
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] = matrix[i][j] || (matrix[i][k] && matrix[k][j]);
            }
        }
    }
}

void MixDependencyAnalyzer::UpdateDependencies(std::unordered_map<int, std::set<int>> &dependencies) {
    size_t n = dependencies.size();
    std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n, false));
    for (const auto& pair : dependencies) {
        int fromId = pair.first;
        for (int toId : pair.second) {
            matrix[fromId][toId] = true;
        }
    }
    WarshallAlgorithm(matrix);
    for (size_t fromId = 0; fromId < n; ++fromId) {
        for (size_t toId = 0; toId < n; ++toId) {
            if (matrix[fromId][toId]) {
                dependencies[fromId].insert(toId);
            }
        }
    }
}

void MixDependencyAnalyzer::ComputeDependencyClosure(std::unordered_map<int, std::set<int>> &dependencies) {
    // 步骤1：初始化直接依赖
    InitDependencies(dependencies);
    // 步骤2：使用Warshall算法将邻接矩阵转换为可达矩阵，时间复杂度为O(n^3)，空间复杂度为O(n^2)
    UpdateDependencies(dependencies);
}

void MixDependencyAnalyzer::ExtractExternalDependencies(const std::vector<SubfuncInvokeInfoTy> &subFuncInvokeInfos) {
    for (size_t i = 0; i < subFuncInvokeInfos.size(); i++) {
        const auto& invokeInfo = subFuncInvokeInfos[i];
        // 提取incast
        for (const auto& incast : invokeInfo.GetIncastTensorParamList()) {
            allIncasts[i].emplace_back(incast.tensor, incast.opMagic, incast.operandIdx);
        }
        // 提取outcast
        for (const auto& outcast : invokeInfo.GetOutcastTensorParamList()) {
            allOutcasts[i].emplace_back(outcast.tensor, outcast.opMagic, outcast.operandIdx);
        }
        // 提取global tensor作为输入输出
        for (const auto& tensorParam : invokeInfo.GetTensorParamList()) {
            if (tensorParam.isOutputToGM) {
                allOutcasts[i].emplace_back(tensorParam.tensor, tensorParam.opMagic, tensorParam.operandIdx);
            } else {
                allIncasts[i].emplace_back(tensorParam.tensor, tensorParam.opMagic, tensorParam.operandIdx);
            }
        }
    }
}

// 检查incast列表中是否包含指定的tensor
bool MixDependencyAnalyzer::ContainsTensor(const std::vector<SimpleTensorParam> &tensors, const LogicalTensorPtr &tensor) const {
    for (const auto& incast : tensors) {
        if (incast.tensor == tensor) {
            return true;
        }
    }
    return false;
}

void MixDependencyAnalyzer::PropagateIncastDependencies(const std::set<int> &targets, const std::vector<SimpleTensorParam> &tensorParams) {
    for (int targetComp : targets) {
        for (const auto& incastParam : tensorParams) {
            if (!ContainsTensor(allIncasts[targetComp], incastParam.tensor)) {
                allIncasts[targetComp].push_back(incastParam);
            }
        }
    }
}

void MixDependencyAnalyzer::PropagateOutcastDependencies(int targetComp, int sourceComp) {
    auto outcastIt = allOutcasts.find(targetComp);
    if (outcastIt != allOutcasts.end()) {
        for (const auto& outcastParam : outcastIt->second) {
            if (!ContainsTensor(allOutcasts[sourceComp], outcastParam.tensor)) {
                allOutcasts[sourceComp].push_back(outcastParam);
            }
        }
    }
}

void MixDependencyAnalyzer::PropagateExternalDependenciesWithClosure(const std::unordered_map<int, std::set<int>> &dependencyClosure) {
    // 基于传递闭包传播依赖
    for (const auto &[sourceComp, targets] : dependencyClosure) {
        // 传播incast：source的incast传播给所有依赖它的target
        auto incastIt = allIncasts.find(sourceComp);
        if (incastIt != allIncasts.end()) {
            PropagateIncastDependencies(targets, incastIt->second);
        }
        // 传播outcast：target的outcast反向传播给所有source
        for (int targetComp : targets) {
            PropagateOutcastDependencies(targetComp, sourceComp);
        }
    }
}

void MixDependencyAnalyzer::CollectInternalDependencies(const std::unordered_map<int, std::set<int>> &dependencyClosure,
                                                        const std::vector<InternalComponentInfo> &components) {
    // 遍历传递闭包中的每个依赖关系
    for (const auto& [srcComp, dstComps] : dependencyClosure) {
        ComponentType srcType = components[srcComp].componentType;
        if (srcType == ComponentType::UNKNOWN) {
            continue;
        }
        for (int dstComp : dstComps) {
            // 跳过自依赖（scope依赖自己）
            if (srcComp == dstComp) {
                ALOG_DEBUG_F("Skip self-dependency: component %d -> %d", srcComp, dstComp);
                continue;    
            }
            ComponentType dstType = components[dstComp].componentType;
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

void MixDependencyAnalyzer::ObtainMinAdjMatrix(std::vector<std::vector<bool>> &matrix) {
    size_t n = matrix.size();
    std::vector<std::vector<bool>> matrixCopy = matrix;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                matrix[i][j] = false;
                continue;
            }
            bool isRedundant = false;
            for (size_t k = 0; k < n; ++k) {
                if (k != i && k != j && matrixCopy[i][k] && matrixCopy[k][j]) {
                    isRedundant = true;
                    break;
                }
            }
            matrix[i][j] = isRedundant ? false : true;
        }
    }
}

void MixDependencyAnalyzer::EliminateRedundantOuterDeps(const std::vector<std::vector<bool>> &innerDeps,
                                                        std::unordered_map<int, std::vector<SimpleTensorParam>> &allTensors) {
    // 初始化，构造tensor到compId的映射
    std::set<int> isRedundant;
    std::vector<bool> outerDeps(maxComponent + 1, false);
    std::unordered_map<LogicalTensorPtr, std::set<int>> tensorToComponents;
    for (const auto &[compId, incasts] : allTensors) {
        for (const auto& incast : incasts) {
            if (incast.tensor) {
                tensorToComponents[incast.tensor].insert(compId);
            }
        }
    }
    for (const auto &pair : tensorToComponents) {
        isRedundant.clear();
        // 用于记录当前的连接关系
        for (int i = 0; i <= maxComponent; ++i) {
            outerDeps[i] = false;
        }
        for (const auto &compId : pair.second) {
            outerDeps[compId] = true;
        }
        // 若tensor可达i且i可达j，则移除tensor到j的可达关系
        for (int i = 0; i <= maxComponent; ++i) {
            if (!outerDeps[i]) {
                continue;
            }
            for (int j = 0; j <= maxComponent; ++j) {
                if (i == j) {
                    continue;
                }
                // 可以证明，若j可达k且i可达j，则必然有i可达k，所以可以原地移除
                if (outerDeps[j] && innerDeps[i][j]) {
                    outerDeps[j] = false;
                    isRedundant.insert(j);
                }
            }
        }
        // 删除冗余incast
        for (const auto &compId : isRedundant) {
            auto& tensors = allTensors[compId];       
            auto newEnd = std::remove_if(tensors.begin(), tensors.end(),
                [&](const SimpleTensorParam& param) {
                    return param.tensor == pair.first;
                });
            tensors.erase(newEnd, tensors.end());
            ALOG_DEBUG_F("Removed redundant incast for tensor %d from component %d",
                        pair.first->GetRawMagic(), compId);
        }
    }
}

std::vector<std::vector<bool>> MixDependencyAnalyzer::Transpose(const std::vector<std::vector<bool>> &matrix) {
    size_t n = matrix.size();
    std::vector<std::vector<bool>> ret(n, std::vector<bool>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ret[j][i] = matrix[i][j];
        }
    }
    return ret;
}

void MixDependencyAnalyzer::EliminateRedundantInnerDeps(std::vector<std::vector<bool>> &innerDeps) {
    ObtainMinAdjMatrix(innerDeps);
    std::vector<InternalDependencyInfo> internalDepsCopy = internalDeps;
    internalDeps.clear();
    for (const auto &dep : internalDepsCopy) {
        if (innerDeps[dep.srcComp][dep.dstComp]) {
            internalDeps.push_back(dep);
        }
    }
}

void MixDependencyAnalyzer::EliminateRedundantDependencies() {
    ALOG_INFO_F("Eliminating redundant dependencies...");
    // 生成内部依赖的可达阵
    std::vector<std::vector<bool>> innerDeps(maxComponent + 1, std::vector<bool>(maxComponent + 1, false));
    for (const auto& dep : internalDeps) {
        innerDeps[dep.srcComp][dep.dstComp] = true;
    }
    // 消除冗余incast
    EliminateRedundantOuterDeps(innerDeps, allIncasts);
    // 消除冗余outcast
    EliminateRedundantOuterDeps(Transpose(innerDeps), allOutcasts);
    // 消除冗余内部依赖
    EliminateRedundantInnerDeps(innerDeps);
}

void MixDependencyAnalyzer::ProcessDependencyAnalyzer(const AnalyzerInput &input, AnalyzerOutput &output) {
    Reset();
    InitSubgraphToFunction(input.components);
    // 步骤1：记录直接的incast/outcast(Mix子图整体与外部的依赖)
    ALOG_INFO_F("Step 1: Recording direct incast/outcast...");
    InOutCastRecord(input.originalMixFunc);
    // 步骤2：分析组件间直接依赖（scope与scope之间的依赖）
    ALOG_INFO_F("Step 2: Analyzing inter-component dependencies...");
    auto directDeps = AnalyzeComponentDependencies(*input.originalMixFunc);
    // 步骤3：计算依赖传递闭包
    ALOG_INFO_F("Step 3: Computing dependency closure...");
    ComputeDependencyClosure(directDeps);    
    // 步骤4：计算所有依赖（包括外部依赖和内部依赖）
    ALOG_INFO_F("Step 4: Computing all dependencies...");
    // 4.1：提取外部依赖（从subgraphToFunction）
    ExtractExternalDependencies(subgraphToFunction.subFuncInvokeInfos);
    // 4.2：基于传递闭包传播外部依赖到内部scope
    PropagateExternalDependenciesWithClosure(directDeps);
    // 4.3：添加内部同类型scope之间的依赖（只收集C-C、V-V的依赖）
    CollectInternalDependencies(directDeps, input.components);
    // 步骤5：消除冗余依赖
    ALOG_INFO_F("Step 5: Eliminating redundant dependencies...");
    EliminateRedundantDependencies();
    output.subgraphToFunction = subgraphToFunction;
    output.internalDeps = internalDeps;
    output.allIncasts = allIncasts;
    output.allOutcasts = allOutcasts;
}

void MixDependencyAnalyzer::Reset() {
    internalDeps.clear();
    allIncasts.clear();
    allOutcasts.clear();
}
}
}