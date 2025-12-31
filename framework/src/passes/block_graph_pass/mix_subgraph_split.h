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
#include <unordered_map>
#include <set>
#include <vector>

namespace npu {
namespace tile_fwk {
// Mix子图内部独立子图的信息
struct InternalComponentInfo {
    int internalSubgraphID;  // mix子图内部的子图ID(cube/vector组件ID)
    std::vector<Operation*> operations; // 包含的op
    std::string suffix;
    AIVCore aivCore;
    
    InternalComponentInfo(int id, const std::string& suf = "")
        : internalSubgraphID(id), suffix(suf), aivCore(AIVCore::UNSPECIFIED) {}
        
    InternalComponentInfo(int id, const std::string& suf, AIVCore aiv)
        : internalSubgraphID(id), suffix(suf), aivCore(aiv) {}
};

// 单个Mix子图的拆分结果
struct MixSubgraphSplitResult {
    uint64_t originalProgramID; // 原Mix子图的programID
    Function* originalFunction; // 原Mix子图函数
    std::vector<uint64_t> newProgramIDs; // 新创建子图的programID列表
    std::vector<Function*> newFunctions; // 新创建leafFunction列表
    std::vector<InternalComponentInfo> components; // 内部组件信息
    std::vector<Operation*> originalCallOps; // 所有同构的原始callOp
};

// Wrap属性结构体
struct WrapAttributes {
    uint64_t wrapId;
    MixResourceType resourceType;
};

struct SimpleIncastParam {
    LogicalTensorPtr tensor;
    int opMagic;
    int operandIdx;

    SimpleIncastParam(LogicalTensorPtr t, int magic, int idx)
        : tensor(t), opMagic(magic), operandIdx(idx) {}
};

struct SimpleOutcastParam {
    LogicalTensorPtr tensor;
    int opMagic;
    int operandIdx;

    SimpleOutcastParam(LogicalTensorPtr t, int magic, int idx)
        : tensor(t), opMagic(magic), operandIdx(idx) {}
};

// Mix子图及其内部组件信息
struct MixSubgraphInfo {
    uint64_t programID;
    Function* function;
    std::vector<InternalComponentInfo> components;
    std::vector<Operation*> originalCallOps;
};

struct ExtractInfo {
    std::vector<std::vector<SymbolicScalar>> &extractedArgList;
    std::vector<int>& iOffsets;
    std::vector<int>& oOffsets;
    int& currentOffset;
    std::set<LogicalTensorPtr>& processedTensors;
};

struct CallOpCreationInfo {
    Function* leafFunc;
    uint64_t newProgramID;
    size_t componentIndex;
    Operation* originalCallOp;
    uint64_t wrapId;
    std::vector<int> iOffsets;
    std::vector<int> oOffsets;
};

class MixSubgraphSplit : public Pass {
public:
    MixSubgraphSplit() : Pass("MixSubgraphSplit"), nextWrapId_(0), nextMixId_(0) {
        SetSupportedArches({NPUArch::DAV_3510});
    }
    ~MixSubgraphSplit() override = default;

    Status RunOnFunction(Function &function) override;

private:
    void DisplayComponents(const std::vector<InternalComponentInfo>& components);
    SubgraphToFunction InitSubgraphToFunction(const std::vector<InternalComponentInfo>& components);
    void InOutCastRecord(SubgraphToFunction &subgraphToFunction, Function* originalMixFunc);
    Status GenNewFunctions(Function& rootFunc,
                            Function* originalMixFunc, 
                            const std::vector<InternalComponentInfo>& components,
                            const std::vector<uint64_t>& newProgramIDs,
                            SubgraphToFunction& subgraphToFunction,
                            std::vector<Function*>& newFunctions);
    Status CreateCallOps(Function& rootFunc, 
                        const std::vector<Operation*>& originalCallOps, 
                        Function* originalMixFunc,  
                        const std::vector<InternalComponentInfo>& components,
                        const std::vector<uint64_t>& newProgramIDs,
                        SubgraphToFunction& subgraphToFunction,
                        std::vector<Function*>& newFunctions);
    Status SetMixIdResourceType(std::vector<Function*> &newFunctions, uint64_t mixId, MixResourceType resourceType);
    // 处理单个leaf function
    Status ProcessLeafFunction(Function& rootFunc,
                              uint64_t programID,
                              Function* originalMixFunc,
                              const std::vector<InternalComponentInfo>& components,
                              const std::vector<uint64_t>& newProgramIDs,
                              const std::vector<Operation*>& originalCallOps,
                              std::vector<MixSubgraphSplitResult>& splitResults);

    // 检查是否为需要拆分的Mix子图
    bool IsMixSubgraph(Function& leafFunc) const;

    // 分析Mix子图内部的独立子图结构
    std::vector<InternalComponentInfo> AnalyzeInternalComponents(Function& mixSubgraphFunc) const;

    // op分组函数
    std::map<int, std::vector<Operation*>> GroupOperationsByExistingInternalID(
        Function& mixSubgraphFunc, std::vector<Operation*>& unassignedOps) const;
    
    // 处理未分组的op
    void ProcessUnassignedOperations(std::vector<Operation*>& unassignedOps,
                                   std::map<int, std::vector<Operation*>>& componentsByInternalID,
                                   Function& mixSubgraphFunc) const;

    // op合并函数
    bool MergeMoveInOperation(Operation* op,
                             std::map<int, std::vector<Operation*>>& componentsByInternalID,
                             std::unordered_map<Operation*, int>& opToComponentMap) const;
                             
    bool MergeMoveOutOperation(Operation* op,
                              std::map<int, std::vector<Operation*>>& componentsByInternalID,
                              std::unordered_map<Operation*, int>& opToComponentMap) const;
             
    bool MergeSyncPhase2(Operation* op, Function& mixSubgraphFunc, std::map<int, std::vector<Operation*>>& componentsByInternalID, std::unordered_map<Operation*, int>& opToComponentMap) const;
    bool MergeSyncPhase1(Operation* op, Function& mixSubgraphFunc, std::map<int, std::vector<Operation*>>& componentsByInternalID, std::unordered_map<Operation*, int>& opToComponentMap) const;
    bool MergeSyncSrcDst(Operation* op, Operation* targetOp, std::map<int, std::vector<Operation*>>& componentsByInternalID, std::unordered_map<Operation*, int>& opToComponentMap) const;
    bool MergeSyncOperation(Operation* op,
                           std::map<int, std::vector<Operation*>>& componentsByInternalID,
                           std::unordered_map<Operation*, int>& opToComponentMap,
                           Function& mixSubgraphFunc) const;
    // 搜索函数
    Operation* FindFirstOpForward(Operation* startOp,
                                 Function& mixSubgraphFunc,
                                 std::function<bool(Operation*)> predicate) const;

    Operation* FindFirstOpBackward(Operation* startOp,
                                  Function& mixSubgraphFunc,
                                  std::function<bool(Operation*)> predicate) const;

    // 创建拆分后的Leaf function
    Function* CreateSplitLeafFunction(Function& rootFunc,
                                    Function& originalMixFunc,
                                    const InternalComponentInfo& component,
                                    uint64_t newProgramID,
                                    uint64_t componentIndex,
                                    SubgraphToFunction& subgraphToFunction);

    // 在root function中创建call op
    Status CreateCallOpInRootFunction(Function& rootFunc,
                                     Function& leafFunc,
                                     uint64_t newProgramID,
                                     uint64_t componentIndex,
                                     Operation* originalCallOp,
                                     Function* originalMixFunc,  
                                     SubgraphToFunction& subgraphToFunction,
                                     uint64_t wrapId,
                                     std::vector<int>& iOffsets,
                                     std::vector<int>& oOffsets);
    
    int FindTensorIndexInList(int tensorMagic, const std::vector<LogicalTensorPtr>& tensorList) const;

    // 依赖分析函数
    std::unordered_map<int, std::vector<int>> AnalyzeComponentDependencies(Function& mixFunc) const;
    
    void BroadcastDependencyClosure(std::set<int> &deps_i, std::set<int> &newDeps, std::unordered_map<int, std::set<int>> &closure, bool &changed, int i) const;
    std::unordered_map<int, std::set<int>> ComputeDependencyClosure(const std::unordered_map<int, std::vector<int>>& directDeps) const;

    void PropagateIncastDependencies(const std::vector<Function*>& leafFunctions,
                                    const std::unordered_map<int, std::vector<LogicalTensorPtr>> &directIncasts,
                                    const std::unordered_map<int, std::set<int>>& dependencyClosure,   
                                    const SubgraphToFunction& subgraphToFunction) const;
    void PropagateOutcastDependencies(const std::vector<Function*>& leafFunctions,
                                    const std::unordered_map<int, std::vector<LogicalTensorPtr>> &directOutcasts,
                                    const std::unordered_map<int, std::set<int>>& dependencyClosure,   
                                    const SubgraphToFunction& subgraphToFunction) const;
    void PropagateExternalDependencies(const std::vector<Function*>& leafFunctions,
                                      const std::unordered_map<int, std::set<int>>& dependencyClosure,
                                      const SubgraphToFunction& subgraphToFunction) const;
                                      
    void PropagateIncastToLeafFunction(Function* targetLeafFunc, int sourceComp,
                                      const std::vector<SimpleIncastParam>& incastParams) const;
                                      
    void PropagateOutcastToLeafFunction(Function* sourceLeafFunc, int targetComp,
                                       const std::vector<SimpleOutcastParam>& outcastTensors) const;

    // 应用拆分结果到全局programs
    Status ApplySplitResultsWithRemap(Function& function,
                                     const std::vector<MixSubgraphSplitResult>& splitResults,
                                     const std::unordered_map<uint64_t, uint64_t>& programIDRemap,
                                     const std::unordered_map<uint64_t, std::vector<uint64_t>>& mixSubgraphNewIDs);

    // 动态参数处理
    Status CopyInferParamIndexInfo(Function* originalMixFunc,
                                  const std::vector<Function*>& newFunctions) const;
                                  
    Status CopyDynParamFromOps(Function* newFunc) const;

    // 清理函数
    void DeleteOriginalMixCallOps(Function& rootFunc, const std::vector<Operation*>& callOpsToDelete);

    // 辅助函数
    MixResourceType GetMixResourceType(Function& mixFunc) const;
    AIVCore FindConsumerVectorAIVCore(Operation* copyOp) const;
    AIVCore DetermineComponentAIVCore(const std::vector<Operation*>& operations) const;
    bool IsSyncOperation(Operation* op) const;
    bool IsInUnassignedOps(Operation* op, const std::vector<Operation*>& unassignedOps) const;
    Operation* FindPreviousOpInSequence(Operation* op, Function& mixSubgraphFunc) const;
    Operation* FindNextOpInSequence(Operation* op, Function& mixSubgraphFunc) const;
    void DisplayArg(const std::vector<SymbolicScalar>& originalLinearArgs) const;
    
    bool ExtractArgListFromIncast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const;
    bool ExtractArgListFromOutcast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const;
    bool ExtractArgListFromGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const;
    bool ExtractArgListFromActualIncasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualIncasts, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const; 
    bool ExtractArgListFromActualOutcasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualOutcasts, std::vector<SymbolicScalar> &originalLinearArgs, ExtractInfo& extractInfo) const;
    // 参数提取函数
    std::vector<std::vector<SymbolicScalar>> ExtractArgListForLeafFunction(
        Function& leafFunc,
        CallOpAttribute* originalCallAttr,
        const SubfuncInvokeInfoTy& invokeInfo,
        std::vector<int>& iOffsets,
        std::vector<int>& oOffsets) const;

    int FindOriginalOffsetInMixFunction(LogicalTensorPtr tensor) const;

    int GetOffsetFromIncastParam(const SubfuncInvokeInfoTy::IncastParamPackTy& incastParam, Function& leafFunc) const;
    int GetOffsetFromOutcastParam(const SubfuncInvokeInfoTy::OutcastParamPackTy& outcastParam, Function& leafFunc) const;
    int GetOffsetFromTensorParam(const SubfuncInvokeInfoTy::TensorParamPackTy& tensorParam, Function& leafFunc) const;
    Status SetOffsetsToLeafFunction(Function& leafFunc, const std::vector<int>& iOffsets, const std::vector<int> &oOffsets, const SubfuncInvokeInfoTy& invokeInfo);
    bool SetOffsetToOpByMagic(int opMagic, int operandIdx, int offset, Function& leafFunc, bool isOutput) const;
    void UpdateCopyOpAttributeExpressions(Operation* op, int newOffset, bool isOutput) const;
    void UpdateOffsetExpressions(std::vector<OpImmediate>& offsets, const RawSymbolicScalarPtr& newOffsetValue) const;

    Status GatherSubGraphInfo(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::set<uint64_t> &mixSubgraphIDsToDelete, std::vector<Operation*> &callOpsToDelete);
    Status CalculateSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::set<uint64_t> &mixSubgraphIDsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap);
    Status ExecuteSplit(Function &function, std::vector<MixSubgraphInfo> &mixSubgraphs, std::vector<Operation*> callOpsToDelete, std::unordered_map<uint64_t, std::vector<uint64_t>> &mixSubgraphNewIDs, std::unordered_map<uint64_t, uint64_t> &programIDRemap);

    uint64_t nextWrapId_;
    uint64_t nextMixId_;
};

} // namespace tile_fwk
} // namespace npu

#endif // PASS_MIX_SUBGRAPH_SPLIT_H