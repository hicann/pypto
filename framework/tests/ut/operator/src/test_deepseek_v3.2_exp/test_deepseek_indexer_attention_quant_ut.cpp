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
 * \file test_decode_sparse_attention_quant_ut.cpp
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
#include "operator/models/deepseek_v3.2_exp/deepseek_indexer_attention_quant.h"
#include "operator/models/deepseek_v3.2_exp/dsia_common.h"

using namespace npu::tile_fwk;

class DeepSeekIndexerAttentionQuantUTest : public testing::Test {
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

void SetQuantPreConfig() {
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_MODE, NUM_4);
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_2 * NUM_1024 * NUM_1024);
}

void TestDeepSeekIndexerAttentionQuantUTest(DSIASimpleParams &params) {
    SetQuantPreConfig();

    int b = params.b;
    int s1 = params.s1;
    int n1 = params.n1;
    int n2 = params.n2;
    int h = params.h;
    int dn = params.kv_lora_rank;
    int dr = params.rope_dim;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int blockSize = params.blockSize;
    int idx_n_heads = params.idx_n_heads;
    int idx_head_dim = params.idx_head_dim;
    int topk = params.topk;
    std::string layoutKey = params.cacheMode;

    std::vector<int32_t> kvCacheActSeqVec(b, 65536);
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    params.maxBlockNumPerBatch = maxBlockNumPerBatch;
    std::cout << "========= maxBlockNumPerBatch " << maxBlockNumPerBatch << std::endl;

    int blockNum = 0;
    for (auto curS : kvCacheActSeqVec) {
        blockNum += CeilDiv(curS, blockSize);
    }
    params.blockNum = blockNum;

    std::cout << "====input param==== b sq nq nkv dn dr blockNum blockSize topk: " 
                << b << " " << s1 << " " << n1 << " " << n2 << " " << dn << " " << dr << " " << blockNum << " " << blockSize << " " << topk 
                << std::endl;

    DataType dType = DT_BF16;
    DataType dTypeQuant = DT_INT8;

    constexpr int64_t b16C0Dim = 16;
    constexpr int64_t b8C0Dim = 32;
    constexpr int64_t nzFirstDim = 16;

    // 1. 设置shape
    std::vector<int64_t> xShape = {b, s1, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, dn + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n1, qkNopeHeadDim, dn};
    std::vector<int64_t> dequantScaleWUqqrShape = {n1 * qHeadDim, 1};
    std::vector<int64_t> cosShape = {b, s1, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {dn};
    std::vector<int64_t> kvLenShape = {b, s1};
    std::vector<int64_t> kvCacheShape = {blockNum, blockSize, n2, dn};
    std::vector<int64_t> krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
    std::vector<int64_t> kScaleCacheShape = {blockNum, blockSize, n2, 4};

    std::vector<int64_t> wIdxQbShape = {
        idx_n_heads * idx_head_dim / b8C0Dim, qLoraRank / nzFirstDim, nzFirstDim, b8C0Dim
    }; //nz
    std::vector<int64_t> wIdxQbScaleShape = {idx_n_heads * idx_head_dim, 1};
    std::vector<int64_t> wIdxKShape = {idx_head_dim / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}; //nz
    std::vector<int64_t> wIdxProjShape = {idx_n_heads / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}; //nz
    std::vector<int64_t> layernormGammaKShape = {idx_head_dim};
    std::vector<int64_t> layernormBetaKShape = {idx_head_dim};
    std::vector<int64_t> hadamardQShape = {idx_head_dim, idx_head_dim};
    std::vector<int64_t> hadamardKShape = {idx_head_dim, idx_head_dim};
    std::vector<int64_t> idxKCacheShape = {blockNum, blockSize, n2, idx_head_dim};
    std::vector<int64_t> idxKScaleCacheShape = {blockNum, blockSize, n2, 1};

    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> actualSeqShape = {b};

    std::vector<int64_t> qNopeShape = {b * s1, n1, dn};
    std::vector<int64_t> qRopeShape = {b * s1, n1, qkRopeHeadDim};
    std::vector<int64_t> rmsNormShape = {b * s1, qLoraRank};
    std::vector<int64_t> rmsNormScaleShape = {b * s1, 1};

    std::vector<int64_t> qInt8Shape = {b * s1, idx_n_heads, idx_head_dim};
    std::vector<int64_t> qScaleShape = {b * s1, idx_n_heads, 1};
    std::vector<int64_t> weightsShape = {b * s1, params.idx_n_heads};

    std::vector<int64_t> indexerTopkShape = {b, s1, n2, topk};
    std::vector<int64_t> indexerTopkTmpOutShape = {b * s1 * n2, maxBlockNumPerBatch * blockSize};

    std::vector<int64_t> saOutShape = {b, s1, n1, dn};

    // 2. 构造tensor
    Tensor dynamicX(dType, xShape, "x");
    Tensor wDq(dType, wDqShape, "wDq", TileOpFormat::TILEOP_NZ);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", TileOpFormat::TILEOP_NZ);
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", TileOpFormat::TILEOP_NZ);
    Tensor wUk(dType, wUkShape, "wUk", TileOpFormat::TILEOP_ND);
    Tensor dequantScaleWUqqr(DT_FP32, dequantScaleWUqqrShape, "dequantScaleWUqqrShape");
    Tensor gammaCq(dType, gammaCqShape, "gammaCq", TileOpFormat::TILEOP_ND);
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv", TileOpFormat::TILEOP_ND);
    Tensor dynamicCos(dType, cosShape, "cos");
    Tensor dynamicSin(dType, cosShape, "sin");
    Tensor kvCache(dTypeQuant, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor kScaleCache(DT_FP32, kScaleCacheShape, "kScaleCache");

    Tensor wIdxQb(dTypeQuant, wIdxQbShape, "wIdxQb", TileOpFormat::TILEOP_NZ);
    Tensor wIdxQbScale(DT_FP32, wIdxQbScaleShape, "wIdxQbScale", TileOpFormat::TILEOP_ND);
    Tensor wIdxK(dType, wIdxKShape, "wIdxK", TileOpFormat::TILEOP_NZ);
    Tensor wIdxProj(dType, wIdxProjShape, "wIdxProj", TileOpFormat::TILEOP_NZ);
    Tensor layernormGammaK(dType, layernormGammaKShape, "layernormGammaK");
    Tensor layernormBetaK(dType, layernormBetaKShape, "layernormBetaK");
    Tensor hadamardQ(dType, hadamardQShape, "hadamardQ");
    Tensor hadamardK(dType, hadamardKShape, "hadamardK");
    Tensor idxKCache(dTypeQuant, idxKCacheShape, "idxKCache");
    Tensor idxKScaleCache(DT_FP16, idxKScaleCacheShape, "idxKScaleCache");

    Tensor dynamicCacheIndex(DT_INT64, kvLenShape, "cacheIndex");
    Tensor dynamicBlockTable(DT_INT32, blockTableShape, "blockTable");

    Tensor dynamicActualSeqLengthsKey(DT_INT32, actualSeqShape, "actualSeqLengthsKey");

    Tensor dynamicSaOut(dType, saOutShape, "saOut");

    Tensor debugQNopeOut(dType, qNopeShape, "debugQNopeOut");
    Tensor debugQRopeOut(dType, qRopeShape, "debugQRopeOut");
    Tensor debugRmsNormOut(DT_INT8, rmsNormShape, "debugRmsNormOut");
    Tensor debugRmsNormScaleOut(DT_FP32, rmsNormScaleShape, "debugRmsNormScaleOut");

    Tensor debugQInt8Out(DT_INT8, qInt8Shape, "debugQInt8Out");
    Tensor debugQScaleOut(DT_FP16, qScaleShape, "debugQScaleOut");
    Tensor debugWeightsOut(DT_FP16, weightsShape, "debugWeightsOut");

    Tensor indexerTopkOut(DT_INT32, indexerTopkShape, "indexerTopkOut");
    Tensor indexerTopkValueOut(DT_FP32, indexerTopkShape, "indexerTopkValueOut");
    Tensor indexerTopkTmpOut(DT_FP32, indexerTopkTmpOutShape, "indexerTopkTmpOut");

    DiaQuantAttr attrs;
    attrs.rmsnormEpsilonCq = params.eps;
    attrs.rmsnormEpsilonCkv = params.eps;
    attrs.layernormEpsilonK = 1e-6f;
    attrs.attnSoftmaxScale = static_cast<float>(1.0 / sqrtf(dn + dr));
    attrs.selectedCount = topk;
    attrs.layeroutKey = layoutKey;
    attrs.layeroutQuery = "TND";

    DeepSeekIndexerAttentionQuant(dynamicX, wDq, wUqQr, wUk, wDkvKr,
                                  gammaCq, gammaCkv, dynamicCos, dynamicSin, dynamicCacheIndex,
                                  kvCache, krCache, kScaleCache, dequantScaleWUqqr,
                                  wIdxQb, wIdxQbScale, wIdxK, wIdxProj,
                                  layernormGammaK, layernormBetaK, hadamardQ, hadamardK,
                                  idxKCache, idxKScaleCache,
                                  dynamicActualSeqLengthsKey, dynamicBlockTable,
                                  dynamicSaOut, attrs, params,
                                  // debug
                                  debugQNopeOut, debugQRopeOut, debugRmsNormOut, debugRmsNormScaleOut,
                                  debugQInt8Out, debugQScaleOut, debugWeightsOut,
                                  indexerTopkOut, indexerTopkValueOut, indexerTopkTmpOut
                                  );
}

void test_common_ut(DSIASimpleParams params) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::CODEGEN_EXPRESSION_FUSION, true);
    params.topk = 2048;
    params.cacheMode = "PA_BSND";

    auto tileB = 4;
    auto tileS = 1;
    MlaTileConfig prologConfig = {tileB, tileS};

    IndexerTile indexerTile;
    indexerTile.weightTile = {64, 128};
    indexerTile.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    indexerTile.v1Tile = {64, 128};
    indexerTile.topkTile = {1, 4096};
    indexerTile.addsTile = {1, 1, 1, 4096};

    SaTileShapeConfig saTileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 2048; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128}; // (n1, d)

    params.mlaTileCfg = prologConfig;
    params.indexTileCfg = indexerTile;
    params.salTileCfg = saTileConfig;

    TestDeepSeekIndexerAttentionQuantUTest(params);
}

TEST_F(DeepSeekIndexerAttentionQuantUTest, 4B_mtp_ut) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    params.b = 4;
    params.s1 = 2;
    params.s2 = 128 * 1024;
    params.n1 = 128;
    params.n2 = 1;
    test_common_ut(params);
}
