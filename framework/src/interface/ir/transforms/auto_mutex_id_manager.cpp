/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/auto_mutex_id_manager.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/transforms/base/mutator.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

using npu::tile_fwk::ExternalError;

namespace {

constexpr int kMutexIdCount = 32;

class PlaceholderResolver final : public IRMutator {
public:
    explicit PlaceholderResolver(const std::vector<std::pair<VarPtr, int>>& replacements)
    {
        for (const auto& [placeholder, value] : replacements) {
            placeholderValues_[placeholder.get()] = value;
        }
    }

protected:
    using IRMutator::VisitExpr_;

    ExprPtr VisitExpr_(const VarPtr& op) override
    {
        auto it = placeholderValues_.find(op.get());
        if (it == placeholderValues_.end()) {
            // Ordinary SSA Vars are not mutex placeholders and must remain unchanged.
            return IRMutator::VisitExpr_(op);
        }
        return std::make_shared<const ConstInt>(it->second, DataType::INT64, op->span_);
    }

private:
    std::unordered_map<const Var*, int> placeholderValues_;
};

std::string HexRange(int64_t start, int64_t end)
{
    std::ostringstream os;
    os << "[0x" << std::hex << start << ", 0x" << end << ")";
    return os.str();
}

} // namespace

class AutoMutexIdManager::Impl {
public:
    struct TileMutexId {
        VarPtr placeholder;
        std::vector<int> manualIds;

        bool IsAuto() const { return placeholder != nullptr; }
    };

    struct TileRecord {
        MemorySpace memorySpace;
        int64_t start;
        int64_t end;
        std::vector<int> manualIds;
        VarPtr placeholder;
        std::string groupName;
        Span span;

        bool IsAuto() const { return placeholder != nullptr; }
    };

    struct CandidateGroup {
        std::vector<VarPtr> autoPlaceholders;
        std::vector<int> manualIds;
    };

    struct AliasComponent {
        std::vector<size_t> tiles;
        MemorySpace memorySpace;
        int64_t start;
        int64_t end;
    };

    struct MutexJob {
        size_t component;
        std::vector<size_t> autoTiles;
        std::vector<int> candidates;
    };

    struct AllocationState {
        std::vector<AliasComponent> components;
        std::vector<size_t> componentByTile;
        std::vector<std::set<size_t>> conflictNeighbors;
        std::vector<std::set<int>> avoidedIdsByComponent;
        std::array<int, kMutexIdCount> tileUsageById{};
        std::array<int, kMutexIdCount> componentUsageById{};
        std::vector<std::optional<int>> assignedIdByComponent;
    };

    void CollectGroup(const std::vector<ExprPtr>& tiles, const MakeTuplePtr& mutexIds, const std::string& groupName)
    {
        Span span = tiles.empty() || tiles.front() == nullptr ? Span::Unknown() : tiles.front()->span_;
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexIds->elements_.size() == tiles.size(), span)
            << "mutex ID count must match TileGroup depth";

        // The outer tuple follows TileGroup depth; each row describes the IDs for the Tile at
        // the same position.
        for (size_t index = 0; index < tiles.size(); ++index) {
            auto tileMutexId = ParseTileMutexId(mutexIds->elements_[index], span);
            CollectTile(tiles[index], tileMutexId, groupName);
        }
    }

    void RecordOpConstraints(const std::vector<MakeTuplePtr>& tileIdGroups,
                             const std::vector<MakeTuplePtr>& candidateGroups)
    {
        if (!hasAuto_ || tileIdGroups.size() < 2) {
            return;
        }
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          tileIdGroups.size() == candidateGroups.size(), Span::Unknown())
            << "mutex tile-ID and candidate group counts must match";

        std::vector<CandidateGroup> groups;
        groups.reserve(tileIdGroups.size());
        for (size_t index = 0; index < tileIdGroups.size(); ++index) {
            groups.push_back(GetEffectiveCandidateGroup(tileIdGroups[index], candidateGroups[index]));
        }

        // Different Tile operands prefer distinct AUTO IDs. Manual candidates are soft
        // exclusions for AUTO placeholders in the other operand group.
        for (size_t leftIndex = 0; leftIndex < groups.size(); ++leftIndex) {
            const auto& left = groups[leftIndex];
            for (size_t rightIndex = leftIndex + 1; rightIndex < groups.size(); ++rightIndex) {
                const auto& right = groups[rightIndex];
                for (const auto& leftPlaceholder : left.autoPlaceholders) {
                    AddAvoidedIds(leftPlaceholder, right.manualIds);
                    for (const auto& rightPlaceholder : right.autoPlaceholders) {
                        AddAutoConflict(leftPlaceholder, rightPlaceholder);
                    }
                }
                for (const auto& rightPlaceholder : right.autoPlaceholders) {
                    AddAvoidedIds(rightPlaceholder, left.manualIds);
                }
            }
        }
    }

    ProgramPtr AssignMutexIds(const ProgramPtr& program)
    {
        if (!hasAuto_) {
            return program;
        }
        // Allocation is completed before mutation; the resolver then performs only registered
        // AUTO-placeholder-to-ConstInt substitutions.
        auto assignedIdByTile = AllocateMutexIds();
        std::vector<std::pair<VarPtr, int>> replacements;
        replacements.reserve(placeholderToTile_.size());
        for (size_t tileIndex = 0; tileIndex < tiles_.size(); ++tileIndex) {
            const auto& tile = tiles_[tileIndex];
            if (!tile.IsAuto()) {
                continue;
            }
            auto value = assignedIdByTile[tileIndex];
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, value.has_value(),
                                              tile.span)
                << "AUTO mutex Tile has no allocated ID";
            replacements.emplace_back(tile.placeholder, *value);
        }
        PlaceholderResolver resolver(replacements);
        return resolver.VisitProgram(program);
    }

    Span DiagnosticSpan() const { return diagnosticSpan_; }

private:
    bool IsPlaceholder(const ExprPtr& value) const
    {
        auto var = As<Var>(value);
        return var != nullptr && placeholderToTile_.find(var.get()) != placeholderToTile_.end();
    }

    CandidateGroup GetEffectiveCandidateGroup(const MakeTuplePtr& tileIds, const MakeTuplePtr& tileCandidates) const
    {
        bool hasExactIds = std::all_of(
            tileIds->elements_.begin(), tileIds->elements_.end(),
            [this](const ExprPtr& value) { return IsPlaceholder(value) || As<ConstInt>(value) != nullptr; });
        return ParseCandidateGroup(hasExactIds ? tileIds : tileCandidates);
    }

    CandidateGroup ParseCandidateGroup(const MakeTuplePtr& candidates) const
    {
        CandidateGroup group;
        for (const auto& candidate : candidates->elements_) {
            if (IsPlaceholder(candidate)) {
                group.autoPlaceholders.push_back(As<Var>(candidate));
                continue;
            }

            auto manualId = As<ConstInt>(candidate);
            Span candidateSpan = candidate == nullptr ? candidates->span_ : candidate->span_;
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, manualId != nullptr,
                                              candidateSpan)
                << "mutex candidate must be a registered AUTO placeholder or ConstInt";
            ValidateMutexId(manualId->value_, candidateSpan);
            group.manualIds.push_back(static_cast<int>(manualId->value_));
        }
        return group;
    }

    void AddAutoConflict(const VarPtr& left, const VarPtr& right)
    {
        size_t leftIndex = GetAutoTile(left);
        size_t rightIndex = GetAutoTile(right);
        if (leftIndex == rightIndex) {
            return;
        }
        if (leftIndex > rightIndex) {
            std::swap(leftIndex, rightIndex);
        }
        // Store an unordered Tile pair in canonical order so repeated op uses collapse naturally.
        autoConflicts_.emplace(leftIndex, rightIndex);
    }

    void AddAvoidedIds(const VarPtr& placeholder, const std::vector<int>& avoidedIds)
    {
        size_t tileIndex = GetAutoTile(placeholder);
        // These IDs are allocation preferences, not hard exclusions: exhaustion falls back to
        // the full candidate domain.
        auto& avoided = avoidedIds_[tileIndex];
        for (int mutexId : avoidedIds) {
            ValidateMutexId(mutexId, tiles_[tileIndex].span);
            avoided.insert(mutexId);
        }
    }

    TileMutexId ParseTileMutexId(const ExprPtr& value, const Span& fallbackSpan)
    {
        auto mutexIds = As<MakeTuple>(value);
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexIds != nullptr && !mutexIds->elements_.empty(), fallbackSpan)
            << "each Tile must provide a non-empty mutex-ID tuple";

        // A one-element Var row is the AUTO marker. Every other valid row is a list of fixed
        // manual ConstInt IDs.
        if (mutexIds->elements_.size() == 1) {
            auto placeholder = As<Var>(mutexIds->elements_.front());
            if (placeholder != nullptr) {
                return TileMutexId{placeholder, {}};
            }
        }

        std::vector<int> manualIds;
        manualIds.reserve(mutexIds->elements_.size());
        for (const auto& mutexIdExpr : mutexIds->elements_) {
            auto mutexId = As<ConstInt>(mutexIdExpr);
            Span mutexIdSpan = mutexIdExpr == nullptr ? fallbackSpan : mutexIdExpr->span_;
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, mutexId != nullptr,
                                              mutexIdSpan)
                << "manual mutex ID must be a ConstInt";
            ValidateMutexId(mutexId->value_, mutexIdSpan);
            manualIds.push_back(static_cast<int>(mutexId->value_));
        }
        return TileMutexId{nullptr, std::move(manualIds)};
    }

    void CollectTile(const ExprPtr& tile, const TileMutexId& tileMutexId, const std::string& groupName)
    {
        auto tileType = As<TileType>(tile->GetType());
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
            npu::tile_fwk::InternalError::PASS_INNER_ERROR,
            tileType != nullptr && tileType->memref_.has_value() && *tileType->memref_ != nullptr, tile->span_)
            << "AUTO mutex-ID planning requires every Tile to have a MemRef";
        auto memref = *tileType->memref_;
        auto address = As<ConstInt>(memref->addr_);
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, address != nullptr,
                                          tile->span_)
            << "AUTO mutex-ID planning requires a constant Tile address";
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
            npu::tile_fwk::InternalError::PASS_INNER_ERROR,
            memref->size_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                address->value_ <= std::numeric_limits<int64_t>::max() - static_cast<int64_t>(memref->size_),
            tile->span_)
            << "Tile address range overflows int64";
        auto previous = tileToIndex_.find(tile.get());
        if (previous != tileToIndex_.end()) {
            // One Tile can reach several mutex ops; keep one record and reject inconsistent
            // metadata rather than silently changing its allocation mode.
            const auto& record = tiles_[previous->second];
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
                npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                record.placeholder.get() == tileMutexId.placeholder.get() && record.manualIds == tileMutexId.manualIds,
                tile->span_)
                << "the same Tile was collected with inconsistent mutex settings";
            return;
        }

        if (tileMutexId.IsAuto()) {
            hasAuto_ = true;
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
                npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                placeholderToTile_.find(tileMutexId.placeholder.get()) == placeholderToTile_.end(),
                tileMutexId.placeholder->span_)
                << "AUTO mutex placeholder was reused by multiple Tiles";
        }

        size_t tileIndex = tiles_.size();
        tiles_.push_back(TileRecord{memref->memorySpace_, address->value_,
                                    address->value_ + static_cast<int64_t>(memref->size_), tileMutexId.manualIds,
                                    tileMutexId.placeholder, groupName, tile->span_});
        tileToIndex_[tile.get()] = tileIndex;
        if (tileMutexId.IsAuto()) {
            placeholderToTile_[tileMutexId.placeholder.get()] = tileIndex;
        }
    }

    size_t GetAutoTile(const VarPtr& placeholder) const
    {
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, placeholder != nullptr,
                                          Span::Unknown())
            << "AUTO mutex constraint target is null";
        auto it = placeholderToTile_.find(placeholder.get());
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          it != placeholderToTile_.end(), placeholder->span_)
            << "unknown AUTO mutex placeholder";
        // Constraints are keyed by placeholders, while allocation data is indexed by Tile.
        return it->second;
    }

    static void ValidateMutexId(int64_t mutexId, const Span& span)
    {
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexId >= 0 && mutexId < kMutexIdCount, span)
            << "mutex ID must be in [0, 31]";
    }

    std::vector<AliasComponent> BuildAliasComponents() const
    {
        std::vector<size_t> ordered(tiles_.size());
        std::iota(ordered.begin(), ordered.end(), 0);
        std::sort(ordered.begin(), ordered.end(), [this](size_t left, size_t right) {
            const auto& lhs = tiles_[left];
            const auto& rhs = tiles_[right];
            return std::tie(lhs.memorySpace, lhs.start, lhs.end, left) <
                   std::tie(rhs.memorySpace, rhs.start, rhs.end, right);
        });

        // Sweep sorted half-open ranges. Direct and transitive overlaps become one component,
        // whose Tiles must ultimately share one mutex ID.
        std::vector<AliasComponent> components;
        for (size_t tileIndex : ordered) {
            const auto& tile = tiles_[tileIndex];
            bool beginsNew = components.empty() || components.back().memorySpace != tile.memorySpace ||
                             tile.start >= components.back().end;
            if (beginsNew) {
                components.push_back(AliasComponent{{}, tile.memorySpace, tile.start, tile.end});
            }
            auto& component = components.back();
            component.tiles.push_back(tileIndex);
            component.end = std::max(component.end, tile.end);
        }
        return components;
    }

    void ValidateCrossingRanges(const std::vector<AliasComponent>& components)
    {
        for (const auto& component : components) {
            bool containsAuto = std::any_of(component.tiles.begin(), component.tiles.end(),
                                            [this](size_t index) { return tiles_[index].IsAuto(); });
            if (!containsAuto) {
                continue;
            }
            // Equal or nested ranges have an unambiguous shared owner. Crossing ranges only
            // partially overlap and are unsupported because they can trample each other.
            for (size_t leftPos = 0; leftPos < component.tiles.size(); ++leftPos) {
                const auto& left = tiles_[component.tiles[leftPos]];
                for (size_t rightPos = leftPos + 1; rightPos < component.tiles.size(); ++rightPos) {
                    const auto& right = tiles_[component.tiles[rightPos]];
                    bool overlaps = left.start < right.end && right.start < left.end;
                    bool contains = (left.start <= right.start && right.end <= left.end) ||
                                    (right.start <= left.start && left.end <= right.end);
                    if (overlaps && !contains) {
                        int64_t overlapStart = std::max(left.start, right.start);
                        int64_t overlapEnd = std::min(left.end, right.end);
                        FailInvalidOperation(
                            std::string("mutex_ids=\"auto\" does not support crossing Tile address ranges: ") +
                                "TileGroup '" + left.groupName + "' at line " + std::to_string(left.span.BeginLine()) +
                                " uses " + HexRange(left.start, left.end) + "; TileGroup '" + right.groupName +
                                "' at line " + std::to_string(right.span.BeginLine()) + " uses " +
                                HexRange(right.start, right.end) + "; overlap is " +
                                HexRange(overlapStart, overlapEnd) + "; possible buffer address trampling",
                            right.span);
                    }
                }
            }
        }
    }

    std::vector<size_t> GetManualTileIndices(const AliasComponent& component) const
    {
        std::vector<size_t> manualTiles;
        for (size_t tileIndex : component.tiles) {
            if (!tiles_[tileIndex].IsAuto()) {
                manualTiles.push_back(tileIndex);
            }
        }
        return manualTiles;
    }

    void ValidateManualTiles(const AliasComponent& component, const std::vector<size_t>& manualTiles)
    {
        // A manual Tile pins its whole overlap component. Such a pin must be a single ID, and all
        // manual Tiles in that component must agree on it.
        for (size_t tileIndex : manualTiles) {
            const auto& tile = tiles_[tileIndex];
            if (tile.manualIds.size() != 1) {
                FailInvalidArgument(
                    std::string("when address-overlapping Tiles include a manually configured Tile, that Tile ") +
                        "must have exactly one manual mutex ID; TileGroup '" + tile.groupName + "' at line " +
                        std::to_string(tile.span.BeginLine()) +
                        " has an overlapping Tile with multiple mutex IDs: mutex_ids=" + FormatIds(tile.manualIds),
                    tile.span);
            }
        }

        const auto& expectedTile = tiles_[manualTiles.front()];
        int expected = expectedTile.manualIds.front();
        for (size_t tileIndex : manualTiles) {
            const auto& tile = tiles_[tileIndex];
            if (tile.manualIds.front() != expected) {
                FailInvalidOperation(
                    std::string("manual Tiles connected through overlapping address ranges must use the same ") +
                        "mutex ID: TileGroup '" + expectedTile.groupName + "' at line " +
                        std::to_string(expectedTile.span.BeginLine()) + " uses " +
                        HexRange(expectedTile.start, expectedTile.end) + " with mutex_id=" + std::to_string(expected) +
                        "; TileGroup '" + tile.groupName + "' at line " + std::to_string(tile.span.BeginLine()) +
                        " uses " + HexRange(tile.start, tile.end) +
                        " with mutex_id=" + std::to_string(tile.manualIds.front()) + "; address-overlap component is " +
                        HexRange(component.start, component.end),
                    tile.span);
            }
        }
    }

    std::vector<int> GetCandidateIdsForAutoComponent(const AliasComponent& component)
    {
        auto manualTiles = GetManualTileIndices(component);
        if (manualTiles.empty()) {
            // With no manual pin, an AUTO component may choose from the entire hardware ID pool.
            std::vector<int> all(kMutexIdCount);
            std::iota(all.begin(), all.end(), 0);
            return all;
        }

        ValidateManualTiles(component, manualTiles);
        return {tiles_[manualTiles.front()].manualIds.front()};
    }

    static std::string FormatIds(const std::vector<int>& ids)
    {
        std::ostringstream os;
        os << '[';
        for (size_t index = 0; index < ids.size(); ++index) {
            if (index != 0) {
                os << ", ";
            }
            os << ids[index];
        }
        os << ']';
        return os.str();
    }

    void BuildComponentConstraints(AllocationState& state) const
    {
        // Lift per-Tile preferences to overlap components, which are the actual allocation units.
        state.conflictNeighbors.resize(state.components.size());
        for (const auto& [left, right] : autoConflicts_) {
            size_t leftComponent = state.componentByTile[left];
            size_t rightComponent = state.componentByTile[right];
            if (leftComponent == rightComponent) {
                // Aliasing already requires these Tiles to share one ID, so no distinct-ID
                // preference can be applied inside the component.
                continue;
            }
            state.conflictNeighbors[leftComponent].insert(rightComponent);
            state.conflictNeighbors[rightComponent].insert(leftComponent);
        }

        state.avoidedIdsByComponent.resize(state.components.size());
        for (const auto& [tileIndex, ids] : avoidedIds_) {
            auto& componentIds = state.avoidedIdsByComponent[state.componentByTile[tileIndex]];
            componentIds.insert(ids.begin(), ids.end());
        }
    }

    AllocationState BuildAllocationState()
    {
        AllocationState state;
        state.components = BuildAliasComponents();
        ValidateCrossingRanges(state.components);

        // Build the reverse Tile-to-component lookup used to lift per-Tile constraints to their
        // address-overlap allocation unit.
        state.componentByTile.resize(tiles_.size());
        for (size_t componentIndex = 0; componentIndex < state.components.size(); ++componentIndex) {
            for (size_t tileIndex : state.components[componentIndex].tiles) {
                state.componentByTile[tileIndex] = componentIndex;
            }
        }

        BuildComponentConstraints(state);

        // Account for fixed IDs before AUTO allocation so the load-balancing score includes
        // already occupied manual IDs.
        for (const auto& tile : tiles_) {
            if (!tile.IsAuto()) {
                for (int mutexId : tile.manualIds) {
                    ++state.tileUsageById[mutexId];
                }
            }
        }

        state.assignedIdByComponent.resize(state.components.size());
        return state;
    }

    std::vector<MutexJob> BuildAllocationJobs(const AllocationState& state)
    {
        std::vector<MutexJob> jobs;
        for (size_t componentIndex = 0; componentIndex < state.components.size(); ++componentIndex) {
            std::vector<size_t> autoTiles;
            for (size_t tileIndex : state.components[componentIndex].tiles) {
                if (tiles_[tileIndex].IsAuto()) {
                    autoTiles.push_back(tileIndex);
                }
            }
            if (!autoTiles.empty()) {
                // All AUTO Tiles in one overlap component are allocated together and receive the
                // same selected ID.
                jobs.push_back(MutexJob{componentIndex, std::move(autoTiles),
                                        GetCandidateIdsForAutoComponent(state.components[componentIndex])});
            }
        }
        return jobs;
    }

    static size_t EffectiveAvailableIdCount(const MutexJob& job, const AllocationState& state)
    {
        size_t count = std::count_if(job.candidates.begin(), job.candidates.end(), [&](int mutexId) {
            const auto& avoidedIds = state.avoidedIdsByComponent[job.component];
            return avoidedIds.find(mutexId) == avoidedIds.end();
        });
        // Avoided IDs are a preference. When every candidate is avoided, allocation falls back to the full domain.
        return count == 0 ? job.candidates.size() : count;
    }

    static void SortAllocationJobs(std::vector<MutexJob>& jobs, const AllocationState& state)
    {
        // Allocate the most constrained jobs first, then prefer highly connected and larger
        // components so later choices retain as much freedom as possible.
        auto priority = [&state](const MutexJob& job) {
            return std::make_tuple(job.candidates.size(), EffectiveAvailableIdCount(job, state),
                                   -static_cast<int>(state.conflictNeighbors[job.component].size()),
                                   -static_cast<int>(job.autoTiles.size()), job.component);
        };
        std::sort(jobs.begin(), jobs.end(), [&priority](const MutexJob& left, const MutexJob& right) {
            return priority(left) < priority(right);
        });
    }

    static int ChooseMutexId(const MutexJob& job, const AllocationState& state)
    {
        // The lexicographic score first honors avoidance preferences and already allocated
        // neighbors, then balances Tile/component usage, with the numeric ID as a stable tie-break.
        auto score = [&job, &state](int mutexId) {
            int neighborConflicts = 0;
            for (size_t neighbor : state.conflictNeighbors[job.component]) {
                neighborConflicts += state.assignedIdByComponent[neighbor] == mutexId;
            }
            return std::make_tuple(state.avoidedIdsByComponent[job.component].count(mutexId) != 0, neighborConflicts,
                                   state.tileUsageById[mutexId], state.componentUsageById[mutexId], mutexId);
        };
        return *std::min_element(job.candidates.begin(), job.candidates.end(),
                                 [&score](int left, int right) { return score(left) < score(right); });
    }

    std::vector<std::optional<int>> AllocateJobs(AllocationState& state, const std::vector<MutexJob>& jobs) const
    {
        std::vector<std::optional<int>> assignedIdByTile(tiles_.size());
        for (const auto& job : jobs) {
            int chosen = ChooseMutexId(job, state);
            state.assignedIdByComponent[job.component] = chosen;
            // Component members alias one address region and therefore share the chosen ID.
            for (size_t tileIndex : job.autoTiles) {
                assignedIdByTile[tileIndex] = chosen;
            }
            state.tileUsageById[chosen] += static_cast<int>(job.autoTiles.size());
            ++state.componentUsageById[chosen];
        }
        return assignedIdByTile;
    }

    std::vector<std::optional<int>> AllocateMutexIds()
    {
        auto state = BuildAllocationState();
        auto jobs = BuildAllocationJobs(state);
        SortAllocationJobs(jobs, state);
        return AllocateJobs(state, jobs);
    }

    void FailInvalidArgument(const std::string& message, const Span& span)
    {
        diagnosticSpan_ = span;
        PRO_IR_THROW(ValueError, ExternalError::INVALID_ARGUMENT) << message;
    }

    void FailInvalidOperation(const std::string& message, const Span& span)
    {
        diagnosticSpan_ = span;
        PRO_IR_THROW(ValueError, ExternalError::INVALID_OPERATION) << message;
    }

    std::vector<TileRecord> tiles_;
    std::unordered_map<const Expr*, size_t> tileToIndex_;
    std::unordered_map<const Var*, size_t> placeholderToTile_;
    std::set<std::pair<size_t, size_t>> autoConflicts_;
    std::unordered_map<size_t, std::set<int>> avoidedIds_;
    bool hasAuto_ = false;
    Span diagnosticSpan_ = Span::Unknown();
};

AutoMutexIdManager::AutoMutexIdManager() : impl_(std::make_unique<Impl>()) {}

AutoMutexIdManager::~AutoMutexIdManager() = default;

void AutoMutexIdManager::CollectGroup(const std::vector<ExprPtr>& tiles, const MakeTuplePtr& mutexIds,
                                      const std::string& groupName)
{
    impl_->CollectGroup(tiles, mutexIds, groupName);
}

void AutoMutexIdManager::RecordOpConstraints(const std::vector<MakeTuplePtr>& tileIdGroups,
                                             const std::vector<MakeTuplePtr>& candidateGroups)
{
    impl_->RecordOpConstraints(tileIdGroups, candidateGroups);
}

ProgramPtr AutoMutexIdManager::AssignMutexIds(const ProgramPtr& program) { return impl_->AssignMutexIds(program); }

Span AutoMutexIdManager::DiagnosticSpan() const { return impl_->DiagnosticSpan(); }

} // namespace ir
} // namespace pypto
