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
 * \file win_attention.h
 * \brief
 */

#pragma once
#ifndef WIN_ATTENTION
#define WIN_ATTENTION

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk {

constexpr int NUM_9 = 9;

struct WinAttenTileShapeConfig {
    int gTile; // 由于没有处理尾块，当前仅支持因子切分
    int skvTile;
    std::array<int, TILE_VEC_DIMS> vNopeTileShape; // nope tileshape
    std::array<int, TILE_VEC_DIMS> vRopeTileShape; // rope tileshape
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
    std::array<int, TILE_VEC_DIMS> outTileShape; // 4-Dim output tile
};

void WinAttentionCompute(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig);

void WinAttentionComputeFlash(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig);

void WinAttentionDebugCompute(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig);

void WinAttention(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig);

void WinAttentionFlash(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig);

void WinAttentionDebug(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig);
} // namespace npu::tile_fwk

#endif // WIN_ATTENTION
