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
 * \file nsa_selected_attention.h
 * \brief
 */

#pragma once
#ifndef NSA_SELECTED_ATTENTION
#define NSA_SELECTED_ATTENTION

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"

#include "operator/models/nsa/slc_attn.h"
#include "operator/models/deepseek/gen_kv_slc.h"

namespace npu::tile_fwk {

struct SATileShapeConfig {
    std::array<int, TILE_VEC_DIMS> kvSlcV0TileShape;

    int gTile; // 由于没有处理尾块，当前仅支持因子切分
    int sKvTile;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

void SelectedAttentionCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig, bool debug=false);

void SelectedAttentionFlashCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig, bool debug=false);

void SelectedAttention(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig);

} // namespace npu::tile_fwk

#endif // NSA_SELECTED_ATTENTION
