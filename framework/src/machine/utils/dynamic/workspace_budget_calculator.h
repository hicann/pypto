/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "tilefwk/workspace_desc.h"

#include <cstdint>

namespace npu::tile_fwk::dynamic {

constexpr int32_t SLOTS_NEED_ALLOC_SIZE = 2;

struct RuntimeWorkspaceConfig {
    uint32_t stitchNumMax{0};
    uint32_t parallelism{1};
    uint64_t aicoreSpilled{0};
    uint64_t debugTotal{0};
};

struct StitchDepthConfig {
    uint32_t kEff{0};
    uint32_t outcastCacheDepth{0};
    uint32_t stitchMaxFunctionNum{0};
    uint64_t encodedWorkspaceSize{0};
};

uint32_t EffectiveUnrollTimes(uint32_t unrollTimes);

uint32_t StitchUnitCapacityForRuntime(uint32_t stitchNumMax);

uint32_t DeriveOutcastCacheDepth(
    const WorkspaceDesc& desc, uint32_t tensorStitchDepthK, uint32_t configuredStitchNumMax);

uint64_t EstimateCtrlFlowCacheSlottedBlockCount(uint64_t totalSlot, uint32_t outcastCacheDepth);

void BuildTensorWorkspaceFromDescriptor(
    WorkspaceDesc& desc, uint32_t stitchDepthK, uint32_t boundaryOutcastDepthOverride = 0);

uint64_t WorkspaceTotalFromDesc(
    const WorkspaceDesc& desc, uint32_t parallelism, uint64_t aicoreSpilled, uint64_t debugTotal);

StitchDepthConfig ResolveStitchDepthConfig(WorkspaceDesc& desc, const RuntimeWorkspaceConfig& runtimeCfg);

uint64_t WorkspaceBytesToKbCeil(uint64_t bytes);

} // namespace npu::tile_fwk::dynamic
