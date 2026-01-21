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
 * \file test_lightning_indexer_prolog.cpp
 * \brief
 */

#include "tilefwk/tilefwk_op.h"
#include "test_cost_macro.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"
#include "operator/models/deepseek_v3.2_exp/lightning_indexer_prolog.h"
#include "operator/models/deepseek_v3.2_exp/dsia_common.h"

using namespace npu::tile_fwk;

class DynamicLightningIndexerPrologUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

static IndexerShapeParams ReadParams(const RopeTileShapeConfig &ropeTileConfigs, const IndexerTileShapeConfig &indexerConfigs, int tileBS) {
    int paramsSize = 11;
    std::vector<int32_t> input_param(paramsSize);

    IndexerShapeParams params;

    params.b = NUM_28;
    params.seq = NUM_1;
    params.dim = NUM_7168;
    params.qLoraRank = NUM_1536;
    params.headDim = NUM_128;
    params.headNum = NUM_64;
    params.ropeHeadDim = NUM_64;
    params.blockNum = NUM_448;
    params.blockSize = NUM_128;
    params.nKV = NUM_1;
    params.s2 = NUM_2048;
    params.indexerTileConfigs = indexerConfigs;
    params.ropeTileConfigs = ropeTileConfigs;
    params.tileBS = tileBS;

    std::cout << "Read params:" << std::endl;
    std::cout << "s2=" << params.s2 << ", b=" << params.b << ", seq=" << params.seq << ", dim=" << params.dim
              << ", qLoraRank=" << params.qLoraRank << ", headDim=" << params.headDim << ", headNum=" << params.headNum
              << ", ropeHeadDim=" << params.ropeHeadDim << ", blockSize=" << params.blockSize
              << ", blockNum=" << params.blockNum << ", nKV=" << params.nKV << ", tileBS=" << params.tileBS << std::endl;

    return params;
}

void PerformanceConfig() {
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 4}});
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2 * 1024 * 1024);
}

TEST_F_WITH_COST(DynamicLightningIndexerPrologUtest, utest_lightning_indexer_prolog, 60) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 1, 64, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 16, 16, 256, 256, 128, 128}, // c1TileShape
        {1, 256, 128, 128 }, // v1TileShape
        { 16, 16, 256, 256, 128, 128}, // c2TileShape
        {1, 128, 128, 128 } // v2TileShape
    };

    auto params = ReadParams(ropeTileConfigs, indexerConfigs, -1);
    PerformanceConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    // inputs
    DataType dType = DT_BF16;

    int b = params.b;
    int seq = params.seq;
    int dim = params.dim;
    int qLoraRank = params.qLoraRank;
    int headDim = params.headDim;
    int headNum = params.headNum;
    int ropeHeadDim = params.ropeHeadDim;
    int blockNum = params.blockNum;
    int blockSize = params.blockSize;
    int nKV = params.nKV;
    int s2 = params.s2;

    TileOpFormat weightFormat = TileOpFormat::TILEOP_NZ;
    Tensor x(dType, {b, seq, dim}, "x");
    Tensor qr(dType, {b, seq, qLoraRank}, "qr");
    Tensor qW(dType, {qLoraRank, headNum * headDim}, "qW", weightFormat);
    Tensor kW(dType, {dim, headDim}, "kW", weightFormat);
    Tensor projW(dType, {dim, headNum}, "projW", weightFormat);
    Tensor lnW(dType, {headDim}, "lnW");
    Tensor lnBias(dType, {headDim}, "lnBias");
    Tensor cos(dType, {b, seq, ropeHeadDim}, "cos");
    Tensor sin(dType, {b, seq, ropeHeadDim}, "sin");
    Tensor kCache(dType, {blockNum, blockSize, nKV, headDim}, "kCache");
    Tensor kCacheIndex(DT_INT32, {b, seq}, "kCacheIndex");
    Tensor blockTable(DT_INT32, {b, s2 / blockSize}, "actSeqLen");
    IndexerPrologInput input{x, qr, qW, kW, projW, lnW, lnBias, cos, sin, kCache, kCacheIndex, blockTable};

    // outputs
    Tensor query(dType, {b * seq, headNum, headDim}, "qOut");
    Tensor weight(dType, {b * seq, headNum}, "weightOut");
    Tensor kCacheOut(dType, {blockNum, blockSize, nKV, headDim}, "kCacheOut");
    IndexerPrologOutput output{query, weight, kCacheOut};

    LightningIndexerProlog(input, output, params);
}
