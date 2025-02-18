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
 * \file DYNAMIC_NSA_COMMON.h
 * \brief
 */

#pragma once
#ifndef DYNAMIC_NSA_COMMON
#define DYNAMIC_NSA_COMMON

namespace npu::tile_fwk {
#define DEBUG_DUMP_TMP_IN_OUT 1 // debug模式，用于打印中间输出及设置临时输入

constexpr int NUM_65536 = 65536;
constexpr int SCATTER_UPADATE_DIM = -2;
constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_4 = 4;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_20 = 20;
constexpr int NUM_24 = 24;
constexpr int NUM_28 = 28;
constexpr int NUM_32 = 32;
constexpr int NUM_48 = 48;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_384 = 384;
constexpr int NUM_448 = 448;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;
constexpr int NUM_2048 = 2048;
constexpr int NUM_1536 = 1536;
constexpr int NUM_1792 = 1792;
constexpr int NUM_4096 = 4096;
constexpr int NUM_6144 = 6144;
constexpr int NUM_8192 = 8192;
constexpr int NUM_7168 = 7168;
constexpr float F_1 = 1.0;
constexpr float F_0 = 0.0;
constexpr float F_NEGA_1 = -1.0;
constexpr double DF_1E_20 = 1e-20;

enum GateMode {
    standard,
    simple
};

struct MlaTileConfig {
    int tileB = 8; // tileB is 8
    int tileS = 1;
    int tileBS = 8;
};

struct SaTileShapeConfig {
    int gTile; // 由于没有处理尾块，当前仅支持因子切分
    int sKvTile;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

struct IndexerTile {
    std::vector<int64_t> weightTile;
    std::array<int64_t, TILE_CUBE_DIMS> c1Tile; // (m, M), (k, K), (n, N)
    std::vector<int64_t> v1Tile;
    std::vector<int64_t> topkTile;
    std::vector<int64_t> addsTile;
};

struct IndexerTileShapeConfig {
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, SHAPE_DIM4> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, SHAPE_DIM4> v2TileShape;
};

struct RopeTileShapeConfig {
    std::array<int, SHAPE_DIM2> twoDim;
    std::array<int, SHAPE_DIM3> threeDim;
    std::array<int, SHAPE_DIM4> fourDim;
};

struct NSASimpleParams {
    int b;
    int s1;
    int s2;
    int n1;
    int n2;
    int h;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    int rope_dim;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int front;
    int near;
    int topk;
    std::string cacheMode;
    int blockSize;
    int winSize;
    int vHeadDim;
    int idx_n_heads;
    int idx_head_dim;
    float softmaxScale;
    float scoreScale;

    int idxHeadDim;
    int idxHeadNum;
    int blockNum;

    float eps;
    MlaTileConfig mlaTileCfg;
    SaTileShapeConfig salTileCfg;
    IndexerTile indexTileCfg;
    IndexerTileShapeConfig indexerTileConfigs;
    RopeTileShapeConfig ropeTileConfigs;
    static NSASimpleParams getCommonParams() {
        NSASimpleParams params;
        params.h = NUM_7168;
        params.q_lora_rank = NUM_1536;
        params.kv_lora_rank = NUM_512;
        params.qk_rope_head_dim = NUM_64;
        params.qk_nope_head_dim = NUM_128;
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.rope_dim = NUM_64;
        params.cmpBlockSize = NUM_32;
        params.cmpStride = NUM_16;
        params.slcBlockSize = NUM_64;
        params.front = NUM_1;
        params.near = NUM_2;
        params.topk = NUM_16;
        params.cacheMode = "BSND";
        params.blockSize = NUM_128;
        params.winSize = NUM_512;
        params.vHeadDim = NUM_128;
        params.eps = 1e-5f;
        params.idx_n_heads = NUM_64;
        params.idx_head_dim = NUM_128;
        params.softmaxScale = static_cast<float>(1.0 / sqrtf((params.kv_lora_rank + params.rope_dim)));
        params.scoreScale = (1.0f / sqrtf(params.idx_n_heads)) * (1.0f / sqrtf(params.idx_head_dim));
        return params;
    }

    static NSASimpleParams getDecodeParams() {
        NSASimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_1;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }

    static NSASimpleParams getMTPParams() {
        NSASimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_2;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }
};


} // namespace npu::tile_fwk

#endif // DYNAMIC_NSA_COMMON