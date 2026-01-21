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
 * \file test_lightning_indexer.cpp
 * \brief
 */
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/lightning_indexer.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class TestLightningIndexerUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    void TearDown() override {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
};

void TestLightningIndexerTopkQuant(IndexerTile &tileConfig) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    const int b = 4;
    const int s1 = 2;
    const int n1 = 64;
    const int d = 128;
    const int blockNum = 1127;
    const int blockSize = 128;
    const int n2 = 1;
    const int maxBlockNum = 1024;
    const int selectedCount = 2048;

    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};

    Tensor staticQuery(DT_INT8, {b, s1, n1, d}, "staticQuery");
    Tensor staticKey(DT_INT8, {blockNum, blockSize, n2, d}, "staticKey");
    Tensor staticQScale(DT_FP16, {b, s1, n1, 1}, "staticQScale");
    Tensor staticKScale(DT_FP16, {blockNum, blockSize, n2, 1}, "staticKScale");
    Tensor staticWeights(DT_FP16, {b, s1, n1}, "staticWeights");
    Tensor staticActSeq(DT_INT32, {b}, "staticActSeq");
    Tensor staticBlockTable(DT_INT32, {b, maxBlockNum}, "staticBlockTable");
    Tensor staticTopkRes(DT_INT32, {b, s1, n2, selectedCount}, "staticTopkRes");
    Tensor staticTmpOut(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "staticTmpOut");
    Tensor staticTopkValue(DT_FP32, {b, s1, n2, selectedCount}, "staticTopkValue");

    Tensor query(DT_INT8, {-1, -1, n1, d}, "query");
    Tensor key(DT_INT8, {-1, blockSize, n2, d}, "key");
    Tensor qScale(DT_FP16, {-1, -1, n1, 1}, "qScale");
    Tensor kScale(DT_FP16, {-1, blockSize, n2, 1}, "kScale");
    Tensor weights(DT_FP16, {-1, -1, n1}, "weights");
    Tensor actSeq(DT_INT32, {-1}, "actSeq");
    Tensor blockTable(DT_INT32, {-1, -1}, "blockTable");

    auto symB = b;
    auto symS1 = s1;
    auto symMaxBlock = maxBlockNum;

    Tensor topkRes(DT_INT32, {symB, symS1, n2, selectedCount}, "topkRes");
    Tensor tmpOut(DT_FP32, {symB * symS1 * n2, symMaxBlock * blockSize}, "tmpOut");
    Tensor topkValue(DT_FP32, {symB, symS1, n2, selectedCount}, "topkValue");

    Tensor topkResGolden(DT_INT8, {b, s1, n2, selectedCount}, "topkResGolden");
    Tensor tmpGolden(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "tmpGolden");
    Tensor topkValueGolden(DT_FP32, {b, s1, n2, selectedCount}, "topkValueGolden");

    FUNCTION("IndexerTopk", {query, key, qScale, kScale, weights, actSeq, blockTable}, {topkRes, tmpOut, topkValue}) {
        LightningIndexerTopkImpl(query, key, true, &qScale, &kScale, weights, actSeq, blockTable, topkRes,
            selectedCount, tileConfig, unrollList, &tmpOut, &topkValue);
    }
}

void TestLightningIndexer(LightningIndexerConfigs &tileConfig) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    const int b = 4;
    const int s1 = 2;
    const int n1 = 64;
    const int d = 128;
    const int s2 = 64 * 1024;
    const int blockSize = 128;
    const int blockNum = s2 / blockSize * b;
    const int maxBlockNum = s2 / blockSize; // seq of each batch is equal in this case
    const int n2 = 1;
    const int selectedCount = 2048;

    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};

    Tensor query(DT_INT8, {b * s1, n1, d}, "query");
    Tensor qScale(DT_FP16, {b * s1, n1}, "qScale");
    Tensor key(DT_INT8, {blockNum, blockSize, n2, d}, "key");
    Tensor kScale(DT_FP16, {blockNum, blockSize, n2}, "kScale");
    Tensor weights(DT_FP16, {b * s1, n1}, "weights");
    Tensor actSeq(DT_INT32, {b}, "actSeq");
    Tensor blockTable(DT_INT32, {b, maxBlockNum}, "blockTable");

    Tensor firstMm(DT_FP32, {b * s1 * n1, maxBlockNum * blockSize}, "firstMm");
    Tensor mmOut(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "MmOut");
    Tensor topkValue(DT_FP32, {b * s1, n2, selectedCount}, "topkValue");
    Tensor topkRes(DT_INT32, {b * s1, n2, selectedCount}, "topkRes");

    FUNCTION("LightningIndexer", {query, qScale, key, kScale, weights, actSeq, blockTable},
        {topkRes, firstMm, mmOut, topkValue}) {
        LightningIndexerImpl(query, qScale, key, kScale, weights, actSeq, blockTable, selectedCount, topkRes,
            tileConfig, unrollList, &firstMm, &mmOut, &topkValue);
    }
}

// TestLightningIndexerUtest.lightning_indexer_b_4_s1_2_s2_64k_quant
TEST_F(TestLightningIndexerUtest, lightning_indexer_b_4_s1_2_s2_64k_quant) {
    LightningIndexerConfigs config;
    config.s1Tile = 2;
    config.topkTile = 8192;
    config.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.c2Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.extendParam.reluType = npu::tile_fwk::Matrix::ReLuType::ReLu;
    float scale = 2048.0;
    config.extendParam.scaleValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scale));

    TestLightningIndexer(config);
}