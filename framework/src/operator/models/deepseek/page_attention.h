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
 * \file page_attention.h
 * \brief
 */

#pragma once
#ifndef PAGE_ATTENTION
#define PAGE_ATTENTION

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {

void PageAttention(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false);

void PageAttentionWithImmScalar(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    std::vector<std::vector<int>>  &blockTable, std::vector<int> &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1, bool isNzFormat = false);

void PageAttentionWithManualUnroll(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes);

void PageAttentionHighThroughput(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1);
} // namespace npu::tile_fwk

#endif // MLA_PROLOG