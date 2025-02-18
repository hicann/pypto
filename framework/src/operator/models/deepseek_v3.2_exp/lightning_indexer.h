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
 * \file lightning_indexer.h
 * \brief
 */

#pragma once
#ifndef LIGHTNING_INDEXER_H
#define LIGHTNING_INDEXER_H

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "dsia_common.h"

namespace npu::tile_fwk {

constexpr int64_t MAX_LI_BATCH = 128;
constexpr int64_t MAX_LI_S1 = 4;
constexpr int64_t MAX_LI_S2 = 128 * 1024;
constexpr float AVOID_FP32_TO_FP16_OVERFLOW_SCALE = 1.0f / 2048;

struct LightningIndexerConfigs {
    // graph optimization params
    int mgCopyInUpperBound = 2 * 1024 * 1024;
    int pgUpperBound = 16 * 8192;
    std::map<int64_t, int64_t> cubeL1ReuseSetting = {
        {0, 8},
        {1, 8}
    };
    int vecMergeMode = 2;
    std::map<int64_t, int64_t> vecNBufferSetting = {
        {-1, 32}
    }; // set with max unrolls
    // stitch params
    int maxRecyclePeriod = 2048;
    int maxLoopNum = 2048;
    // tile params
    int64_t s1Tile;
    int64_t topkTile;
    std::array<int64_t, TILE_CUBE_DIMS> c1Tile; // (m, M), (k, K), (n, N)
    std::array<int64_t, TILE_CUBE_DIMS> c2Tile; // (m, M), (k, K), (n, N)
    // matmul relu fuse params
    Matrix::MatmulExtendParam extendParam;
};

void LightningIndexerTopkImpl(const Tensor &query, const Tensor &key, bool isQuant, const Tensor *qScale,
    const Tensor *kScale, const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes,
    const int selectedCount, IndexerTile tileConfig, const std::set<int> &unrollList = {64, 32, 16, 8, 4, 1},
    Tensor *tmpOut = nullptr, Tensor *topkValue = nullptr);

void LightningIndexerTopkQuant(const Tensor &query, const Tensor &key, const Tensor &qScale, const Tensor &kScale,
    const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes, const int selectedCount,
    IndexerTile tileConfig, const std::set<int> &unrollList = {64, 32, 16, 8, 4, 1});

void LightningIndexerTopk(const Tensor &query, const Tensor &key, const Tensor &qScale, const Tensor &kScale,
    const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes, const int selectedCount,
    IndexerTile tileConfig, const std::set<int> &unrollList = {64, 32, 16, 8, 4, 1});

void LightningIndexerImpl(const Tensor &idxQuery, const Tensor &idxQueryScale, const Tensor &idxKeyCache,
    const Tensor &idxKeyScale, const Tensor &idxWeight, const Tensor &actSeqKey, const Tensor &blockTable,
    const int selectedCount, Tensor &topkRes, LightningIndexerConfigs configs,
    const std::set<int> &unrollList = {32, 16, 8, 4, 1}, Tensor *firstMm = nullptr, Tensor *mmOut = nullptr,
    Tensor *topkValue = nullptr);

void LightningIndexer(const Tensor &idxQuery, const Tensor &idxQueryScale, const Tensor &idxKeyCache,
    const Tensor &idxKeyScale, const Tensor &idxWeight, const Tensor &actSeqKey, const Tensor &blockTable,
    const int selectedCount, Tensor &topkRes, LightningIndexerConfigs configs,
    const std::set<int> &unrollList = {32, 16, 8, 4, 1});

} // namespace npu::tile_fwk

#endif // LIGHTNING_INDEXER_H