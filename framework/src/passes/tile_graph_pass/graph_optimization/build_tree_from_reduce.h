/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file build_tree_from_reduce.h
 * \brief Convert atomic-add assemble fan-in into an explicit balanced add tree.
 */

#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/operation/opcode.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

inline constexpr const char* COMPUTE_DETERMINISM_LEVEL = "compute_determinism_level";

class BuildTreeFromReducePass : public Pass {
public:
    BuildTreeFromReducePass() : Pass("BuildTreeFromReduce") {}
    ~BuildTreeFromReducePass() override = default;

private:
    static constexpr uint64_t ADD_TILE_BUFFER_COUNT = 4;
    static constexpr int64_t MIN_SPLITTABLE_DIM = 2;

    using OperationRanks = std::unordered_map<const Operation*, size_t>;

    struct TreeRewriteInfo {
        std::vector<Operation*> addedAddOps;
        std::vector<Operation*> retainedAssembles;
    };

    struct SplitRegion {
        Offset inputOffset;
        Shape shape;
        Offset outputOffset;
        std::vector<SymbolicScalar> inputDynOffset;
        std::vector<SymbolicScalar> outputDynOffset;
        std::vector<SymbolicScalar> dynValidShape;
    };

    Status RunOnFunction(Function& function) override;

    // Candidate discovery and grouping.
    static bool IsTreeAssembleOpcode(Opcode opcode);
    static bool IsAtomicAddCandidate(const Operation& op);
    static bool TensorDependsOn(const LogicalTensorPtr& tensor, const LogicalTensorPtr& dependency,
                                std::unordered_set<const LogicalTensor*>& visited);
    static bool HasAtomicTargetReadDependency(const Operation& candidate, const LogicalTensorPtr& output);
    static bool IsSameDynShape(const std::vector<SymbolicScalar>& lhs, const std::vector<SymbolicScalar>& rhs);
    static bool IsSameAssembleRegion(const Operation& lhs, const Operation& rhs);
    static bool CanBuildOneTree(const Operation& lhs, const Operation& rhs);

    // Large-input split planning.
    static bool TryGetStaticTensorBytes(const LogicalTensorPtr& tensor, uint64_t& bytes);
    static bool HasOnlyImmediateDynOffset(const std::vector<SymbolicScalar>& dynOffset);
    static Offset AddStaticOffset(const Offset& baseOffset, const Offset& concreteOffset);
    static void NormalizeStaticDynOffset(Offset& staticOffset, std::vector<SymbolicScalar>& dynOffset);
    static std::vector<SymbolicScalar> AddStaticOffsetToDynOffset(const std::vector<SymbolicScalar>& baseDynOffset,
                                                                  const Offset& concreteOffset);
    static std::vector<SymbolicScalar> MakeSliceDynValidShape(const LogicalTensorPtr& input, const Offset& inputOffset,
                                                              const std::vector<SymbolicScalar>& inputDynOffset,
                                                              const Shape& sliceShape);
    static bool ValidateSplitRegionSource(const Operation& anchor, const LogicalTensorPtr& input, uint64_t& inputBytes);
    static std::vector<int64_t> DeriveAddVecTile(const Operation& anchor, const LogicalTensorPtr& lhs, uint64_t ubSize);
    static bool TryBuildSplitRegions(const Operation& anchor, std::array<SplitRegion, 2>& regions);

    // Add-tree graph rewrite.
    static LogicalTensorPtr CreateSliceView(Function& function, const LogicalTensorPtr& input,
                                            const SplitRegion& region, const Operation& anchor, size_t index);
    static void UpdateAssembleRegion(Operation& assemble, const SplitRegion& region);
    static void MergeAtomicSemanticAttrsToRetained(const std::vector<Operation*>& assembles);
    static size_t GetInputProducerRank(const Operation& assemble, const OperationRanks& ranks);
    static LogicalTensorPtr CreateAdd(Function& function, const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs,
                                      const Operation& anchor, size_t index, TreeRewriteInfo& rewriteInfo);
    static void BuildTree(Function& function, std::vector<Operation*>& assembles, TreeRewriteInfo& rewriteInfo);
    static void BuildSplitTrees(Function& function, std::vector<Operation*>& assembles,
                                const std::array<SplitRegion, 2>& regions, TreeRewriteInfo& rewriteInfo);

    // Memory repair after graph rewrite.
    template <typename ConvertInserterT>
    static Status FinalizeAddedAddMemory(ConvertInserterT& inserter, const std::vector<Operation*>& addedAddOps);

    template <typename ConvertInserterT>
    static Status FinalizeRetainedAssemblesMemory(ConvertInserterT& inserter,
                                                  const std::vector<Operation*>& retainedAssembles);

    static void CollectNewOperations(Function& function, const std::unordered_set<Operation*>& existingOps,
                                     std::vector<Operation*>& addedOps);
    static Status CheckAddedAddOps(const std::vector<Operation*>& addedAddOps);

    template <typename ConvertInserterT>
    static Status FinalizeBuildTreeMemory(Function& function, const TreeRewriteInfo& rewriteInfo,
                                          ConvertInserterT& inserter);

    static Status ProcessOutputGroup(Function& function, const LogicalTensorPtr& output, const OperationRanks& ranks,
                                     TreeRewriteInfo& rewriteInfo, bool& changed);
};

// Keep the historical C++ type name source-compatible while exposing the
// pass under its concise name in the pass registry and execution strategies.
using BuildTreeFromReduce = BuildTreeFromReducePass;

} // namespace npu::tile_fwk
