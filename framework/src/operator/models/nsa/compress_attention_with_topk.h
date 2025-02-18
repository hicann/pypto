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
 * \file compress_attention_with_topk.h
 * \brief
 */

#pragma once
#ifndef COMPRESS_ATTENTION_WITH_TOPK_H
#define COMPRESS_ATTENTION_WITH_TOPK_H

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "operator/models/nsa/fused_compress_kv_select.h"

namespace npu::tile_fwk {

struct CmpTile {
    std::array<int64_t, TILE_CUBE_DIMS> c1Tile; // (m, M), (k, K), (n, N)
    std::array<int64_t, TILE_VEC_DIMS> v1Tile;
    std::array<int64_t, TILE_CUBE_DIMS> c2Tile; // (m, M), (k, K), (n, N)
    std::array<int64_t, TILE_VEC_DIMS> v2Tile;
};

struct CmpAttnTopkTile {
    std::vector<int64_t> topkTile;
    CmpTile cmpTile;
};

void CompressAttentionWithTopK(const Tensor &qNope, const Tensor &qRope, const Tensor &cmpKvCache,
    const Tensor &cmpKrCache, const Tensor &cmpBlockTable, const Tensor &actSeq, const Tensor &auxTensor,
    Tensor &cmpAttnOut, Tensor &topkRes, const int blockSize, const int cmpBlockSize, const int cmpStride,
    const int slcBlockSize, const float softmaxScale, const int n1, const int topk, const int front, const int near,
    CmpAttnTopkTile &tileConfig);

} // namespace npu::tile_fwk

#endif // COMPRESS_ATTENTION_WITH_TOPK_H
