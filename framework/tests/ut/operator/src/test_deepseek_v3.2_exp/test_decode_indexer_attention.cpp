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

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"
#include "operator/models/deepseek_v3.2_exp/decode_indexer_attention.h"
#include "operator/models/deepseek_v3.2_exp/dsia_common.h"


using namespace npu::tile_fwk;

class DecodeIndexerAttentionUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

void SetPreConfig() {
    config::SetHostOption(ONLY_CODEGEN, true);
}

struct TensorWithData {
    Tensor tensor;
    RawTensorDataPtr dataPtr = nullptr;
};

template <typename T>
TensorWithData CreateConstantDynamicOutputTensor(const std::vector<int64_t> &shape, DataType dType, std::string name, T value, const std::vector<SymbolicScalar> &dynamicShape = {}) {
    ASSERT(dynamicShape.size() == 0 || dynamicShape.size() == shape.size());

    if (dynamicShape.empty()) {
        Tensor dynamicT(dType, shape, name);
        RawTensorDataPtr data = RawTensorData::CreateConstantTensorData<T>(shape, dType, value);
        return TensorWithData{dynamicT, data};
    }

    Tensor dynamicT(dType, dynamicShape, name);
    RawTensorDataPtr data = RawTensorData::CreateConstantTensorData<T>(shape, dType, value);
    return TensorWithData{dynamicT, data};
}

Tensor CreateDynamicTensor(DataType dType, const std::vector<int64_t> &shape, std::string name, const std::vector<int> &dynamicAxises = {}) {
     std::vector<int64_t> dynamicShape = shape;
     for (int axis : dynamicAxises) {
        ASSERT(axis >= 0 && (size_t)axis < dynamicShape.size());
        dynamicShape[axis] = -1;
     }

     Tensor dynamicT(dType, dynamicShape, name, TileOpFormat::TILEOP_ND);
     return dynamicT;
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool isSmooth = false, bool nz = false>
void TestDecodeIndexerAttentionSTest(DSIASimpleParams &params) {
    SetPreConfig();

    int b = params.b;
    int s1 = params.s1;
    int n1 = params.n1;
    int n2 = params.n2;
    int h = params.h;
    int dn = params.kv_lora_rank;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int blockSize = params.blockSize;
    int idx_n_heads = params.idx_n_heads;
    int idx_head_dim = params.idx_head_dim;

    std::vector<int> kvCacheActSeqVec(b);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    params.blockNum = blockNum;
    std::cout << "========= blockNum " << blockNum << std::endl;
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    // 1. 设置shape
    // MlaProlog
    std::vector<int64_t> xShape = {b, s1, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, dn + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n1, qkNopeHeadDim, dn};
    std::vector<int64_t> cosShape = {b, s1, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {dn};
    std::vector<int64_t> kvLenShape = {b, s1};
    std::vector<int64_t> kvCacheShape = {blockNum, blockSize, n2, dn};
    std::vector<int64_t> krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};

    std::vector<int64_t> qNopeOutShape = {b * s1 , n1, dn};
    std::vector<int64_t> qRopeOutShape = {b * s1 , n1, qkRopeHeadDim};
    std::vector<int64_t> rmsResShape = {b, s1, params.q_lora_rank};

    std::vector<int64_t> wQbScaleShape = {1, n1 * qHeadDim};
    std::vector<int64_t> smoothCqShape{1, qLoraRank};

    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> actSeqsShape = {b};
    std::vector<int64_t> tmpTopkInputShape = {b, s1, n2, params.topk};

    std::vector<int64_t> saOutShape = {b, s1, n1, dn};
    std::vector<int64_t> gatherResShape = {b * s1 * params.topk, dn + qkRopeHeadDim};

    std::vector<int64_t> queryShape = {b, s1, idx_n_heads, idx_head_dim};
    std::vector<int64_t> keyShape = {blockNum, blockSize, n2, idx_head_dim};
    std::vector<int64_t> weightShape = {b, s1, idx_n_heads};

    Tensor dynamicX = CreateDynamicTensor(dType, xShape, "dynamicX", {0, 1});
    SymbolicScalar bSymbol = GetInputShape(dynamicX, 0);
    SymbolicScalar s1Symbol = GetInputShape(dynamicX, 1);

    Tensor wDq(dType, wDqShape, "wDq");
    Tensor wUqQr(dType, wUqQrShape, "wUqQr");
    const bool usePrefetch = true;
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }

    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr");
    Tensor wUk(dType, wUkShape, "wUk");
    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");
    Tensor dynamicCos = CreateDynamicTensor(dType, cosShape, "dynamicCos", {0, 1});
    Tensor dynamicSin = CreateDynamicTensor(dType, cosShape, "dynamicSin", {0, 1});
    Tensor dynamicCacheIndex = CreateDynamicTensor(DT_INT32, kvLenShape, "dynamicCacheIndex", {0, 1});
    Tensor kvCache = CreateDynamicTensor(dType, kvCacheShape, "kvCache", {0});
    Tensor krCache = CreateDynamicTensor(dType, krCacheShape, "krCache", {0});
    Tensor indexKCache = CreateDynamicTensor(dType, keyShape, "indexKCache", {0});
    Tensor dynamicBlockTable = CreateDynamicTensor(DT_INT32, blockTableShape, "dynamicBlockTable", {0, 1});
    Tensor dynamicActSeqs = CreateDynamicTensor(DT_INT32, actSeqsShape, "dynamicActSeqs", {0});
    SymbolicScalar maxBlockNumSymbol = GetInputShape(dynamicBlockTable, 1);

    auto weightFormat = TileOpFormat::TILEOP_NZ;
    Tensor qW(dType, Shape({qLoraRank, idx_n_heads * idx_head_dim}), "qW", weightFormat);
    Tensor kW(dType, Shape({h, idx_head_dim}), "kW", weightFormat);
    Tensor projW(dType, Shape({h, idx_n_heads}), "projW", weightFormat);
    Tensor lnW(dType, Shape({idx_head_dim}), "lnW");
    Tensor lnBias(dType, Shape({idx_head_dim}), "lnBias");

    Tensor dynamicTmpTopkInput = CreateDynamicTensor(DT_INT32, tmpTopkInputShape, "dynamicTmpTopkInput", {0, 1});

    // output
    auto dynamicSaOut = CreateConstantDynamicOutputTensor<T>(saOutShape, dType, "saOut", 0, {bSymbol, s1Symbol, n1, dn});
    auto dynamicGatherRes = CreateConstantDynamicOutputTensor<T>(gatherResShape, dType, "gatherRes", 0, {bSymbol * s1Symbol * params.topk, dn + qkRopeHeadDim});
    auto dynamicTmpRowSumOut = CreateConstantDynamicOutputTensor<float>({b * s1 * n2, maxBlockNumPerBatch * blockSize}, DT_FP32, "tmpRowSumOut", 0, {bSymbol * s1Symbol * n2, maxBlockNumSymbol * blockSize});
    auto dynamicTmpIndexerTopkRes = CreateConstantDynamicOutputTensor<int32_t>({b, s1, n2, params.topk}, DT_INT32, "tmpIndexerTopkRes", 0, {bSymbol, s1Symbol, n2, params.topk});

    auto rmsResOut = CreateConstantDynamicOutputTensor<T>(rmsResShape, dType, "rmsResOut", 0, {bSymbol, s1Symbol, params.q_lora_rank});
    auto queryOut = CreateConstantDynamicOutputTensor<T>(queryShape, dType, "queryOut", 0, {bSymbol, s1Symbol, idx_n_heads, idx_head_dim});
    auto weightOut = CreateConstantDynamicOutputTensor<T>(weightShape, dType, "weightOut", 0, {bSymbol, s1Symbol, idx_n_heads});

    auto qNopeOut = CreateConstantDynamicOutputTensor<T>(qNopeOutShape, dType, "qNopeOut", 0, {bSymbol * s1Symbol , n1, dn});
    auto qRopeOut = CreateConstantDynamicOutputTensor<T>(qRopeOutShape, dType, "qRopeOut", 0, {bSymbol * s1Symbol , n1, qkRopeHeadDim});

    MlaQuantInputs quantInputs;

    DecodeIndexerAttention(dynamicX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, dynamicSin, dynamicCos, dynamicCacheIndex, kvCache, krCache, quantInputs,
        dynamicBlockTable, dynamicActSeqs, qW, kW, projW, lnW, lnBias, indexKCache, dynamicSaOut.tensor,
        dynamicGatherRes.tensor, dynamicTmpTopkInput, dynamicTmpIndexerTopkRes.tensor, dynamicTmpRowSumOut.tensor,
        rmsResOut.tensor,queryOut.tensor,weightOut.tensor,qNopeOut.tensor,qRopeOut.tensor,
        params);
}

TEST_F(DecodeIndexerAttentionUtest, utest_decode_indexer_attention) {
    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    auto params = DSIASimpleParams::getDecodeParams();
    params.b = NUM_1;
    params.s2 = NUM_1024;
    params.topk = NUM_2048;
    params.cacheMode = "PA_BSND";

    RopeTileShapeConfig ropeTileConfigs = {
        {128, 128},
        {32, 128, 128},
        {1, 128, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        {16, 16, 128, 128, 128, 128}, // c1TileShape
        {128, 128, 128, 128}, // v1TileShape
        {16, 16, 128, 128, 128, 128}, // c2TileShape
        {128, 128, 128, 128}  // v2TileShape
    };

    SaTileShapeConfig saTileConfig;
    const int gTile = 128;  // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256};   // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256};                          // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128};                          // (n1, d)

    auto tileB = params.b;
    if (params.b == 24) {
        tileB = 8;
    }
    MlaTileConfig prologConfig = {tileB, 1};

    IndexerTile indexerTile;
    indexerTile.weightTile = {64, 128};
    indexerTile.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    indexerTile.v1Tile = {64, 128};
    indexerTile.topkTile = {1, 2048};
    indexerTile.addsTile = {1, 1, 1, 2048};

    params.salTileCfg = saTileConfig;
    params.mlaTileCfg = prologConfig;
    params.indexTileCfg = indexerTile;
    params.indexerTileConfigs = indexerConfigs;
    params.ropeTileConfigs = ropeTileConfigs;

    TestDecodeIndexerAttentionSTest<npu::tile_fwk::float16, npu::tile_fwk::float16, false, true>(params);
}