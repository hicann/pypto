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
 * \file test_cache_manager.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/configs/config_manager.h"
#include "operator/models/deepseek/page_attention.h"
#define private public
#include "machine/cache_manager/cache_manager.h"
#undef private

namespace npu::tile_fwk {
class CacheManagerUnitTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
private:
    bool oriEnableAihacBackend = false;
};

TEST(CacheManagerUnitTest, test_init_case1) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    CacheManager cacheManager;
    EXPECT_EQ(cacheManager.Initialize(), true);
    EXPECT_EQ(cacheManager.MatchBinCache("112233"), false);
    EXPECT_EQ(cacheManager.RecoverTask("112233", nullptr), false);
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
}

TEST(CacheManagerUnitTest, test_init_case2) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    CacheManager cacheManager;
    EXPECT_EQ(cacheManager.Initialize(), true);
    EXPECT_EQ(cacheManager.Initialize(), true);
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
}

TEST(CacheManagerUnitTest, test_match_cache) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    CacheManager cacheManager;
    EXPECT_EQ(cacheManager.Initialize(), true);
    EXPECT_EQ(cacheManager.MatchBinCache("112233"), false);
    EXPECT_EQ(cacheManager.RecoverTask("112233", nullptr), false);
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
}

TEST(CacheManagerUnitTest, test_page_attention) {
    Program::GetInstance().Reset();
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {nTile, 64};

    std::vector<int> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int blockNum = 0;
    std::vector<int> seq(4, 128);
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    TileOpFormat kvFormat = TileOpFormat::TILEOP_ND;
    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache", kvFormat);
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache", kvFormat);
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope", kvFormat);
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");
    PageAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
                  tileConfig, 1, false);
    Function *lastFunc = Program::GetInstance().GetLastFunction();
    EXPECT_NE(lastFunc, nullptr);
    std::cout << "===hash of last func==" << lastFunc->GetFunctionHash().Data() << lastFunc->GetFunctionTypeStr() << std::endl;
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    CacheManager cacheManager;
    EXPECT_EQ(cacheManager.Initialize(), true);
    auto task = std::make_shared<MachineTask>(111, lastFunc);
    task->SetCacheKey(lastFunc->GetFunctionHash().Data());
    auto deviceAgentTask = std::make_unique<DeviceAgentTask>(task);
    cacheManager.SaveTaskFile(deviceAgentTask.get());
    EXPECT_EQ(cacheManager.MatchBinCache(task->GetCacheKey()), true);
    cacheManager.RecoverTask(lastFunc->GetFunctionHash().Data(), deviceAgentTask.get());
}
}