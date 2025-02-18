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
 * \file fused_compress_kv_select.h
 * \brief
 */

#pragma once
#ifndef FUSED_COMPRESS_KV_SELECT_H
#define FUSED_COMPRESS_KV_SELECT_H

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk {

struct MlpRopeTile {
    std::array<int, SHAPE_DIM2> twoDim;
    std::array<int, SHAPE_DIM3> threeDim;
    std::array<int, SHAPE_DIM4> fourDim;
    std::array<int, SHAPE_DIM5> fiveDim;
};

struct MlpCmpTile {
    std::array<int, SHAPE_DIM3> transTileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, SHAPE_DIM3> v2TileShape;
};

struct AttnTile {
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape={16, 16};
};

struct CmpAttnTile {
    MlpRopeTile mlpRopeTile;
    MlpCmpTile mlpCmpTile;
    AttnTile attnTile;
    std::array<int, SHAPE_DIM2> castTile;
};

Tensor MlpSingleRope(const Tensor &x, const Tensor &cos, const Tensor &sin, MlpRopeTile &tileConfig);
Tensor MlpCompress(const Tensor &x, const Tensor &w1, const Tensor &w2, MlpCmpTile &tileConfig);

Tensor BatchMlpSingleRope(const Tensor &x, const Tensor &cos, const Tensor &sin, MlpRopeTile &tileConfig);
Tensor BatchMlpCompress(const Tensor &x, const Tensor &w1, const Tensor &w2, MlpCmpTile &tileConfig);

void FusedCompressKvSelectCompute(const Tensor &qNope, const Tensor &qRope, const Tensor &kvCache, const Tensor &krCache,
    const Tensor &cmpKvCache, const Tensor &cmpKrCache, const Tensor &blockTable, const Tensor &cmpBlockTable,
    const Tensor &actSeqLen, const Tensor &actCmpSeqLen, const Tensor &mlpWk1,
    const Tensor &mlpWk2, const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpAttnOut, Tensor &cmpAttnOut16,
    Tensor &cmpSoftmax, Tensor &fullK, Tensor &cmpK, Tensor &firstRope, Tensor &firstRopeInput, Tensor &topkRes, Tensor &topkInput, const int blockSize, const int cmpBlockSize, const int cmpStride, const float softmaxScale,
    const int n1, const int n2, CmpAttnTile &tileConfig);

void FusedCompressKvSelect(const Tensor &qNope, const Tensor &qRope, const Tensor &kvCache, const Tensor &krCache,
    const Tensor &cmpKvCache, const Tensor &cmpKrCache, const Tensor &blockTable, const Tensor &cmpBlockTable,
    const Tensor &actSeqLen, const Tensor &actCmpSeqLen, const Tensor &mlpWk1,
    const Tensor &mlpWk2, const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpAttnOut, Tensor &cmpAttnOut16,
    Tensor &cmpSoftmax, Tensor &fullK, Tensor &cmpK, Tensor &firstRope, Tensor &firstRopeInput, Tensor &topkRes, Tensor &topkInput, const int blockSize, const int cmpBlockSize, const int cmpStride, const float softmaxScale,
    const int n1, const int n2, CmpAttnTile &tileConfig);

} // namespace npu::tile_fwk

#endif // FUSED_COMPRESS_KV_SELECT_H
