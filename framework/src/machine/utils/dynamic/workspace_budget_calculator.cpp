/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "machine/utils/dynamic/workspace_budget_calculator.h"

#include "interface/utils/common.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/error_code.h"

namespace {
constexpr uint64_t kAlignment32K = 32 * 1024;
}

namespace npu::tile_fwk {
namespace dynamic {

uint32_t EffectiveUnrollTimes(uint32_t unrollTimes)
{
    return unrollTimes > 0 ? unrollTimes : 1u;
}

uint32_t StitchUnitCapacityForRuntime(uint32_t stitchNumMax)
{
    return stitchNumMax;
}

static uint64_t CalcUnrolledRootBudget(uint64_t budget, uint32_t unrollTimes, uint64_t configMultiplier)
{
    const uint32_t unroll = EffectiveUnrollTimes(unrollTimes);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, configMultiplier > 0)
        << "Invalid configMultiplier: " << configMultiplier << ", must be greater than 0";
    if (static_cast<uint64_t>(unroll) >= configMultiplier) {
        return budget;
    }
    return AlignUp((budget + unroll - 1) / unroll, TENSOR_ADDR_ALIGNMENT) * configMultiplier;
}

static uint64_t MaxRawRootInnerWhenUnrollGeDepth(const WorkspaceDesc& desc, uint32_t stitchDepthK)
{
    (void)stitchDepthK;
    const uint64_t configMultiplier = desc.config.innerSpilledRecyclePeriod > 0
                                          ? desc.config.innerSpilledRecyclePeriod
                                          : static_cast<uint64_t>(stitchDepthK);
    uint64_t maxBudget = 0;
    for (const auto& root : desc.rootFuncDescList) {
        maxBudget = std::max(
            maxBudget, CalcUnrolledRootBudget(root.rootInnerSpilledRawMem, static_cast<uint32_t>(root.unroll), configMultiplier));
    }
    return maxBudget;
}

static uint64_t MaxRawExclusiveOutcastWhenUnrollGeDepth(const WorkspaceDesc& desc, uint32_t stitchDepthK)
{
    (void)stitchDepthK;
    const uint64_t configMultiplier =
        desc.config.unrollStitchCount > 0 ? desc.config.unrollStitchCount : static_cast<uint64_t>(stitchDepthK);
    uint64_t maxBudget = 0;
    for (const auto& root : desc.rootFuncDescList) {
        maxBudget = std::max(
            maxBudget,
            CalcUnrolledRootBudget(
                root.rootTotalExclusiveOutcastRawMem, static_cast<uint32_t>(root.unroll), configMultiplier));
    }
    return maxBudget;
}

static bool Uint64MulWouldOverflow(uint64_t lhs, uint64_t rhs)
{
    return lhs != 0 && rhs > UINT64_MAX / lhs;
}

uint32_t DeriveOutcastCacheDepth(
    const WorkspaceDesc& desc, uint32_t tensorStitchDepthK, uint32_t configuredStitchNumMax)
{
    if (tensorStitchDepthK == 0) {
        return 1;
    }
    uint32_t depth = 1;
    for (const auto& root : desc.rootFuncDescList) {
        const uint32_t unroll = EffectiveUnrollTimes(static_cast<uint32_t>(root.unroll));
        const uint64_t perRoot =
            (static_cast<uint64_t>(tensorStitchDepthK) + unroll - 1) / static_cast<uint64_t>(unroll);
        const uint32_t perRootDepth = static_cast<uint32_t>(std::min(perRoot, static_cast<uint64_t>(UINT32_MAX)));
        depth = std::max(depth, perRootDepth);
    }
    return std::min(depth, configuredStitchNumMax);
}

uint64_t EstimateCtrlFlowCacheSlottedBlockCount(uint64_t totalSlot, uint32_t outcastCacheDepth)
{
    return totalSlot * (static_cast<uint64_t>(outcastCacheDepth) + SLOTS_NEED_ALLOC_SIZE);
}

void BuildTensorWorkspaceFromDescriptor(
    WorkspaceDesc& desc, uint32_t stitchDepthK, uint32_t boundaryOutcastDepthOverride)
{
    (void)boundaryOutcastDepthOverride;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, stitchDepthK > 0)
        << "Invalid stitchDepthK: " << stitchDepthK << ", must be greater than 0";

    ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
        !Uint64MulWouldOverflow(desc.maxPerUnitRootInnerSpilledMem, stitchDepthK))
        << "maxPerUnitRootInnerSpilledMem * stitchDepthK would overflow";
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
        !Uint64MulWouldOverflow(desc.maxPerUnitRootTotalExclusiveOutcastMem, stitchDepthK))
        << "maxPerUnitRootTotalExclusiveOutcastMem * stitchDepthK would overflow";

    const uint64_t fromPerUnitRootInner = desc.maxPerUnitRootInnerSpilledMem * stitchDepthK;
    const uint64_t fromPerUnitExclusiveOutcast = desc.maxPerUnitRootTotalExclusiveOutcastMem * stitchDepthK;
    desc.maxRootInnerSpilledMem = std::max(fromPerUnitRootInner, MaxRawRootInnerWhenUnrollGeDepth(desc, stitchDepthK));
    desc.maxRootTotalExclusiveOutcastMem =
        std::max(fromPerUnitExclusiveOutcast, MaxRawExclusiveOutcastWhenUnrollGeDepth(desc, stitchDepthK));
    desc.devTaskBoundaryOutcastNum =
        desc.totalExclusiveOutcastSlot * SLOTS_NEED_ALLOC_SIZE + desc.totalAssembleOutcastSlot * SLOTS_NEED_ALLOC_SIZE;
    const uint64_t unrollStitchCount = desc.config.unrollStitchCount > 0 ? desc.config.unrollStitchCount : stitchDepthK;
    const uint64_t actualStitchCount = desc.config.actualStitchCount > 0 ? desc.config.actualStitchCount : stitchDepthK;
    desc.devTaskInnerTemporalOutcastNum =
        desc.totalAssembleOutcastSlot * std::min(unrollStitchCount, actualStitchCount);
}

uint64_t WorkspaceTotalFromDesc(
    const WorkspaceDesc& desc, uint32_t parallelism, uint64_t aicoreSpilled, uint64_t debugTotal)
{
    const uint64_t boundaryAndInnerTemporal = desc.devTaskBoundaryOutcastNum + desc.devTaskInnerTemporalOutcastNum;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
        !Uint64MulWouldOverflow(desc.maxStaticOutcastMem, boundaryAndInnerTemporal))
        << "maxStaticOutcastMem * boundaryAndInnerTemporalOutcastNum would overflow";
    const uint64_t staticOutcast = desc.maxStaticOutcastMem * boundaryAndInnerTemporal;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
        desc.maxRootInnerSpilledMem <= UINT64_MAX - desc.maxRootTotalExclusiveOutcastMem)
        << "tensor workspace inner sum would overflow";
    const uint64_t innerSum = desc.maxRootInnerSpilledMem + desc.maxRootTotalExclusiveOutcastMem;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, innerSum <= UINT64_MAX - staticOutcast)
        << "tensor workspace sum would overflow";
    const uint64_t raw = innerSum + staticOutcast;
    const uint64_t alignedRaw = AlignUp(raw, kAlignment32K);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !Uint64MulWouldOverflow(alignedRaw, parallelism))
        << "aligned workspace * parallelism would overflow";
    const uint64_t alignedParallel = alignedRaw * parallelism;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, alignedParallel <= UINT64_MAX - aicoreSpilled - debugTotal)
        << "encoded workspace total would overflow";
    return alignedParallel + aicoreSpilled + debugTotal;
}

StitchDepthConfig ResolveStitchDepthConfig(WorkspaceDesc& desc, const RuntimeWorkspaceConfig& runtimeCfg)
{
    StitchDepthConfig config;
    config.kEff = runtimeCfg.stitchNumMax;
    config.outcastCacheDepth = runtimeCfg.stitchNumMax;
    BuildTensorWorkspaceFromDescriptor(desc, config.kEff);
    config.stitchMaxFunctionNum = StitchUnitCapacityForRuntime(runtimeCfg.stitchNumMax);
    config.encodedWorkspaceSize =
        WorkspaceTotalFromDesc(desc, runtimeCfg.parallelism, runtimeCfg.aicoreSpilled, runtimeCfg.debugTotal);
    return config;
}

uint64_t WorkspaceBytesToKbCeil(uint64_t bytes)
{
    return bytes == 0 ? UINT64_C(0) : (bytes + 1023) / 1024;
}

} // namespace dynamic
} // namespace npu::tile_fwk
