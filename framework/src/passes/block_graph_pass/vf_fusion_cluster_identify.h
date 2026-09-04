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
 * \file vf_fusion_cluster_identify.h
 * \brief Identify VF fusion clusters.
 */

#ifndef VF_FUSION_CLUSTER_IDENTIFY_H
#define VF_FUSION_CLUSTER_IDENTIFY_H

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_common_defs.h"

namespace npu::tile_fwk {

constexpr size_t VF_CLUSTER_SIZE_LIMIT = 32;

class AncestorBits {
public:
    void Build(const std::vector<std::vector<size_t>>& producers, const std::vector<size_t>& topoOrder);
    bool IsAncestor(size_t descendantIndex, size_t ancestorIndex) const;
    size_t Size() const { return bits_.size(); }

private:
    static constexpr size_t BITSET_WORD_BITS = 64;

    static void SetBit(std::vector<uint64_t>& bits, size_t bitIndex);

    std::vector<std::vector<uint64_t>> bits_;
};

class VFFusionClusterIdentify : public Pass {
public:
    VFFusionClusterIdentify() : Pass("VFFusionClusterIdentify") { SetSupportedArches({NPUArch::DAV_3510}); }
    ~VFFusionClusterIdentify() override = default;

    Status RunOnFunction(Function& function) override;

private:
    friend class VFFusionClusterIdentifyTestAccessor;

    // Graph topology context: op list + producer/consumer adjacency tables.
    struct GraphContext {
        std::vector<Operation*> ops;
        std::unordered_map<Operation*, size_t> opToIndex;
        std::vector<std::vector<size_t>> producers;
        std::vector<std::vector<size_t>> consumers;
    };

    // Ready node for priority queue scheduling: opIndex + topological orderIndex.
    struct ReadyNode {
        size_t opIndex;
        size_t orderIndex;
    };

    struct ReadyNodeCompare {
        bool operator()(const ReadyNode& lhs, const ReadyNode& rhs) const { return lhs.orderIndex > rhs.orderIndex; }
    };

    // Schedule block: a contiguous range of ops with the same clusterId in scheduleOrder_.
    struct ScheduleBlock {
        std::vector<size_t> opIndices;
        std::vector<Operation*> ops;
    };

    // Schedule group: ops merged for group-level topological sort, ordered by orderIndex.
    struct ScheduleGroup {
        std::vector<Operation*> ops;
        size_t orderIndex;
    };

    // Replaces a contiguous range in scheduleOrder_. Ops before firstPosition and after lastPosition keep both
    // their relative order and their cached positions, so only this window needs validation and index refresh.
    struct ScheduleWindow {
        size_t firstPosition{0};
        size_t lastPosition{0};
        std::vector<Operation*> ops;
    };

    Status ProcessLeafFunction(Function& function, int& nextClusterId);
    GraphContext BuildGraph(Function& function) const;
    bool BuildDfsPriorityTopoOrder(const GraphContext& graph, std::vector<size_t>& topoOrder) const;
    void SortOpIndicesByMagic(const GraphContext& graph, std::vector<size_t>& opIndices) const;
    bool ReleaseConsumers(const GraphContext& graph, size_t producerIndex, std::vector<size_t>& remainingProducers,
                          std::vector<size_t>& releasedConsumers) const;
    std::vector<Operation*> BuildScheduledList(const GraphContext& graph) const;
    bool BuildScheduleGroups(const GraphContext& graph, std::vector<ScheduleGroup>& groups,
                             std::unordered_map<Operation*, size_t>& opToGroup, size_t& clusterGroupNum) const;
    bool BuildScheduleGroupGraph(const GraphContext& graph, const std::unordered_map<Operation*, size_t>& opToGroup,
                                 std::vector<std::unordered_set<size_t>>& groupConsumers,
                                 std::vector<size_t>& inDegree) const;
    std::vector<Operation*> EmitScheduleGroups(const std::vector<ScheduleGroup>& groups,
                                               const std::vector<std::unordered_set<size_t>>& groupConsumers,
                                               std::vector<size_t> inDegree) const;
    void RefreshScheduleIndex();
    bool RefreshScheduleIndex(size_t firstPosition, size_t lastPosition);

    std::vector<int> GetInputClusterIds(const GraphContext& graph, size_t opIndex) const;
    bool CanMergeConsumerWithInputClusters(const GraphContext& graph, size_t consumerIndex,
                                           const std::vector<int>& inputClusterIds,
                                           const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                           const AncestorBits& ancestorBits,
                                           std::vector<int>& mergeableClusterIds) const;
    bool CanMergeMultiInputClusters(const GraphContext& graph, size_t consumerIndex,
                                    const std::vector<int>& inputClusterIds,
                                    const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                    const AncestorBits& ancestorBits, std::vector<int>& mergeableClusterIds) const;
    bool CanMergeToCluster(const GraphContext& graph, size_t consumerIndex, int clusterId,
                           const std::unordered_map<int, std::vector<Operation*>>& clusters,
                           const AncestorBits& ancestorBits) const;
    bool CanMergeClusterIntoTarget(const GraphContext& graph, size_t consumerIndex, int targetClusterId,
                                   int inputClusterId, const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                   const AncestorBits& ancestorBits) const;
    bool IsAdjustableAdjacent(const GraphContext& graph, const std::unordered_set<size_t>& candidateOpIndices,
                              const AncestorBits& ancestorBits) const;
    bool VerifyScheduleTopology(const GraphContext& graph, const std::vector<Operation*>& schedule) const;
    bool ValidateScheduleWindowOps(const ScheduleWindow& window) const;
    bool ValidateScheduleWindow(const GraphContext& graph, const ScheduleWindow& window) const;
    bool TryBuildCompactedScheduleWindow(const GraphContext& graph,
                                         const std::unordered_set<size_t>& candidateOpIndices,
                                         const AncestorBits& ancestorBits, ScheduleWindow& window) const;
    bool CanPlaceBlockBeforeCandidates(const ScheduleBlock& block, const std::unordered_set<size_t>& candidateOpIndices,
                                       const AncestorBits& ancestorBits) const;
    bool CanPlaceBlockAfterCandidates(const ScheduleBlock& block, const std::unordered_set<size_t>& candidateOpIndices,
                                      const AncestorBits& ancestorBits) const;
    bool VerifyVfClusterContiguity(const std::vector<Operation*>& schedule) const;
    bool ApplyValidatedScheduleWindow(const ScheduleWindow& window);
    bool CompactScheduleForCluster(const GraphContext& graph, const std::vector<Operation*>& clusterOps,
                                   const AncestorBits& ancestorBits);
    bool GetScheduleWindowRange(const GraphContext& graph, const std::unordered_set<size_t>& candidateOpIndices,
                                size_t& firstPosition, size_t& lastPosition) const;
    bool BuildScheduleBlock(const GraphContext& graph, size_t& scheduleIndex, size_t lastPosition,
                            ScheduleBlock& block) const;
    bool ClassifyScheduleBlock(const ScheduleBlock& block, const std::unordered_set<size_t>& candidateOpIndices,
                               const AncestorBits& ancestorBits, std::vector<ScheduleBlock>& beforeBlocks,
                               std::vector<ScheduleBlock>& afterBlocks) const;
    void AssembleScheduleWindow(const std::vector<ScheduleBlock>& beforeBlocks,
                                const std::vector<Operation*>& candidateOps,
                                const std::vector<ScheduleBlock>& afterBlocks, ScheduleWindow& window) const;
    bool HasDirectDependencyFromCluster(const GraphContext& graph, size_t consumerIndex,
                                        const std::vector<Operation*>& clusterOps) const;
    std::vector<size_t> GetClusterOpIndices(const GraphContext& graph, const std::vector<Operation*>& clusterOps) const;
    bool AssignClusterToOp(Operation& op, std::unordered_map<int, std::vector<Operation*>>& clusters,
                           int& nextClusterId);
    bool InitializeLeafSchedule(const GraphContext& graph, std::vector<size_t>& topoOrder, AncestorBits& ancestorBits);
    Status ProcessFusableOps(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                             const AncestorBits& ancestorBits,
                             std::unordered_map<int, std::vector<Operation*>>& clusters, int& nextClusterId);
    Status MergeConsumerIntoClusters(const GraphContext& graph, size_t opIndex,
                                     const std::vector<int>& mergeableClusterIds, const AncestorBits& ancestorBits,
                                     std::unordered_map<int, std::vector<Operation*>>& clusters);
    bool MergeClusters(int targetClusterId, int inputClusterId,
                       std::unordered_map<int, std::vector<Operation*>>& clusters);
    void DissolveSingletonClusters(std::unordered_map<int, std::vector<Operation*>>& clusters);
    void ResetGeneratedClusterIds(const GraphContext& graph);

    bool IsFusableOp(const Operation& op) const;
    bool IsUserScopedOp(const Operation& op) const;
    bool IsVfFusionCluster(const Operation& op) const;
    void SetVfFusionCluster(Operation& op, bool isVfFusionCluster);

    std::vector<Operation*> scheduleOrder_;
    std::unordered_map<Operation*, size_t> schedulePosition_;
    std::unordered_set<const Operation*> vfFusionClusterOps_;
};

} // namespace npu::tile_fwk

#endif // VF_FUSION_CLUSTER_IDENTIFY_H
