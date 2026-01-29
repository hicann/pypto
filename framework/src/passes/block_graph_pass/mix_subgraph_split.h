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
 * \file mix_subgraph_split.h
 * \brief 将Mix子图拆分为多个独立的Cube和Vector子图，并重新分配subgraphID
 */

#ifndef PASS_MIX_SUBGRAPH_SPLIT_H
#define PASS_MIX_SUBGRAPH_SPLIT_H

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_internal_components_analyzer.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_dependency_analyzer.h"
#include "passes/block_graph_pass/mix_subgraph_split/function_clone.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_call_operation_builder.h"
#include <unordered_map>
#include <set>
#include <vector>

namespace npu {
namespace tile_fwk {
// 单个Mix子图的拆分结果
struct MixSubgraphSplitResult {
    uint64_t originalProgramID; // 原Mix子图的programID
    Function* originalFunction; // 原Mix子图函数
    std::vector<uint64_t> newProgramIDs; // 新创建子图的programID列表
    std::vector<Function*> newFunctions; // 新创建leafFunction列表
    std::vector<InternalComponentInfo> components; // 内部组件信息
    std::vector<Operation*> originalCallOps; // 所有同构的原始callOp
};

// Mix子图及其内部scope信息
struct MixSubgraphInfo {
    uint64_t programID;
    Function* function;
    std::vector<InternalComponentInfo> components;
    std::vector<Operation*> originalCallOps;
    FunctionHash hashValue;
    bool isLocalFunction;
    MixSubgraphInfo(uint64_t pid, Function* func, std::vector<InternalComponentInfo> comp,
                    std::vector<Operation*> ops, FunctionHash hash, bool isLocal)
        : programID(pid), function(func), components(std::move(comp)),
          originalCallOps(std::move(ops)), hashValue(hash), isLocalFunction(isLocal) {}
};

class MixSubgraphSplit : public Pass {
public:
    MixSubgraphSplit() : Pass("MixSubgraphSplit"), nextMixId_(0) {
        SetSupportedArches({NPUArch::DAV_3510});
    }
    ~MixSubgraphSplit() override = default;

    Status RunOnFunction(Function &function) override;

private:
    // 子模块
    MixInternalComponentsAnalyzer componentsAnalyzer_;
    MixDependencyAnalyzer dependencyAnalyzer_;
    MixCallOperationBuilder callOpBuilder_;
    
    void DisplayComponents(const std::vector<InternalComponentInfo>& components);
    Status GenNewFunctions(Function& rootFunc,
                            Function* originalMixFunc, 
                            const std::vector<InternalComponentInfo>& components,
                            const std::vector<uint64_t>& newProgramIDs,
                            SubgraphToFunction& subgraphToFunction,
                            std::vector<Function*>& newFunctions);
    Status SetMixIdResourceType(std::vector<Function*> &newFunctions, uint64_t mixId, MixResourceType resourceType);
                                      
    // 应用拆分结果到全局programs
    Status ApplySplitResultsWithRemap(Function& function,
                                     const std::vector<MixSubgraphSplitResult>& splitResults,
                                     const std::unordered_map<uint64_t, uint64_t>& programIDRemap,
                                     const std::unordered_map<uint64_t, std::vector<uint64_t>>& mixSubgraphNewIDs);

    // 清理函数
    void DeleteOriginalMixCallOps(Function& rootFunc, const std::vector<Operation*>& callOpsToDelete);

    // 辅助函数
    // 检查是否为需要拆分的Mix子图
    bool IsMixSubgraph(Function& leafFunc) const;
    MixResourceType GetMixResourceType(Function& mixFunc) const;

    Status GatherSubGraphInfo(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::set<uint64_t> &mixSubgraphIDsToDelete, std::vector<Operation*> &callOpsToDelete);
    Status CalculateSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::set<uint64_t> &mixSubgraphIDsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap);
    Status ExecuteSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::vector<Operation*> callOpsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap);
    // 处理单个leaf function
    Status ProcessLeafFunction(Function& rootFunc,
                              uint64_t programID,
                              Function* originalMixFunc,
                              const std::vector<InternalComponentInfo>& components,
                              const std::vector<uint64_t>& newProgramIDs,
                              const std::vector<Operation*>& originalCallOps,
                              std::vector<MixSubgraphSplitResult>& splitResults);

    void ApplyFinalDependencies(
        const std::vector<Function*>& newFunctions,
        const std::unordered_map<int, std::vector<SimpleTensorParam>>& allIncasts,
        const std::unordered_map<int, std::vector<SimpleTensorParam>>& allOutcasts) const;
    
    void ApplyIncastDependencies(
        Function* leafFunc,
        int componentId,
        const std::vector<SimpleTensorParam>& incastParams) const;
        
    void ApplyOutcastDependencies(
        Function* leafFunc,
        int componentId,
        const std::vector<SimpleTensorParam>& outcastParams) const;

    uint64_t nextMixId_;
    static constexpr uint64_t INVALID_PROGRAM_ID = static_cast<uint64_t>(-1);
};

} // namespace tile_fwk
} // namespace npu

#endif // PASS_MIX_SUBGRAPH_SPLIT_H