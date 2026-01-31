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
 * \file test_dynamic_genAtten.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "operator/models/nsa/gen_Attention.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_128 = 128;
constexpr int NUM_512 = 512;

class GenAttnUtTest : public testing::Test {
public:
    void SetUp() override {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override { config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);}
protected:
    bool oriEnableAihacBackend = false;
};

struct GenAttenConfig {
    int batchSize = 0;
    int headNumSize = 0;
    int s1Size = 0;
    int dimSize = 0;
};

template<typename T = npu::tile_fwk::float16>
void genAtten(GenAttenTileShapeConfig &tileConfig) {
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    int B = NUM_16;
    int N = NUM_128;
    int S1 = 1;
    int D = NUM_512;
    DataType dType;
    if (std::is_same<T, float>::value) {
        dType = DT_FP32;
    } else {
        dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    }

    std::vector<int64_t> shape_cmpAtten = {B, S1, N, D};
    std::vector<int64_t> shape_selAtten = {B, S1, N, D};
    std::vector<int64_t> shape_winAtten = {B, S1, N, D};
    std::vector<int64_t> shape_gatingScore = {B, S1, N, NUM_3};
    std::vector<int64_t> shape_attentionOut = {B, S1, N, D};

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor selAtten(dType, shape_selAtten, "selAtten");
    Tensor winAtten(dType, shape_winAtten, "winAtten");
    Tensor gatingScore(dType, shape_gatingScore, "gatingScore");
    Tensor out_npu(dType, shape_attentionOut, "out_npu");

    std::vector<T>cmpAttenData(B * S1 * N * D);
    std::vector<T>selAttenData(B * S1 * N * D);
    std::vector<T>winAttenData(B * S1 * N * D);
    std::vector<T>gatingScoreData(B * S1 * N * NUM_3);
    std::vector<T>out_goldenData(B * S1 * N * D);

    GenAttention(cmpAtten, selAtten, winAtten, gatingScore, out_npu, tileConfig);
}

TEST_F(GenAttnUtTest, TestDynamicGenAttenTest_FP16_ut) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};
    genAtten<npu::tile_fwk::float16>(tileConfig);
}