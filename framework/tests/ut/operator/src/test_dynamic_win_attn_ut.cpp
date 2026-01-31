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
 * \file test_dynamic_win_attn_ut.cpp
 * \brief
 */
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/win_attention.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class DynamicTestWinAttenUt : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

constexpr int NUM_2 = 2;
constexpr int NUM_16 = 16;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;

template <typename T = npu::tile_fwk::float16>
void TestWinAttenUt(WinAttenTileShapeConfig& tileConfig) {

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    int b = NUM_2;
    int sQ = 1;
    int nQ = NUM_128;
    int nKV = 1;
    int sMax = NUM_1024;
    int dN = NUM_512;
    int dR = NUM_64;
    int blockSize = NUM_128;
    int windowSize = NUM_1024;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dN + dR)));

    int maxBlock = (sMax + blockSize - 1) / blockSize;
    std::vector<int64_t> qNopeShape = {b * sQ * nQ, dN};
    std::vector<int64_t> qRopeShape = {b * sQ * nQ, dR};
    std::vector<int64_t> vNopeCacheShape = {b * maxBlock * blockSize , nKV * dN};
    std::vector<int64_t> kRopeCacheShape = {b * maxBlock * blockSize , nKV * dR};
    std::vector<int64_t> attentionOutShape = {b ,sQ ,nQ, dN};
    std::vector<int64_t> blockTableShape = {b, maxBlock};

    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor vNopeCache(dType, vNopeCacheShape, "vNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "kRopeCache");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor attentionOut(DT_FP32, attentionOutShape, "attentionOut");

    WinAttention(qNope, vNopeCache, qRope, kRopeCache, nQ, nKV, blockTable, actSeqs, windowSize,
        blockSize, softmaxScale, attentionOut, tileConfig);
}

TEST_F(DynamicTestWinAttenUt, TestOnboardWinAttnTest_FP16_Test0) {
    WinAttenTileShapeConfig tileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    tileConfig.gTile = gTileSize;
    tileConfig.vNopeTileShape = {NUM_16, NUM_256};
    tileConfig.vRopeTileShape = {NUM_128, NUM_64};
    tileConfig.outTileShape = {NUM_16, NUM_256};
    tileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (s2Tile, dN+dR) -> (n1, s2Tile)
    tileConfig.v1TileShape = {NUM_16, NUM_256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTileSize, gTileSize, NUM_128, NUM_128, NUM_128, NUM_128}; // (n1, s2Tile) @ (s2Tile, dN) -> (n1, d)
    tileConfig.v2TileShape = {NUM_16, NUM_256}; // (n1, d)
    // WinConfig config;
    TestWinAttenUt<npu::tile_fwk::float16>(tileConfig);
}