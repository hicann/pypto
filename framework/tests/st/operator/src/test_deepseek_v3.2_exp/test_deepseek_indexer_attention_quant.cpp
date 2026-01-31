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
 * \file test_decode_sparse_attention_quant.cpp
 * \brief
 */

#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"
#include "test_common.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek_v3.2_exp/deepseek_indexer_attention_quant.h"
#include "operator/models/deepseek_v3.2_exp/dsia_common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tensor.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DeepSeekIndexerAttentionQuantSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {
void SetPreConfig() {
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, NUM_4}});
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_2 * NUM_1024 * NUM_1024);
}

void TestDeepSeekIndexerAttentionQuantSTest(DSIASimpleParams &params) {
    SetPreConfig();

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
    int blockNum = params.blockNum;
    int topk = params.topk;
    std::string layoutKey = params.cacheMode;

    std::cout << "====input param==== b sq nq nkv dn dr blockNum blockSize topk: "
                << b << " " << s1 << " " << n1 << " " << n2 << " " << dn << " " << dr << " " << blockNum << " " << blockSize << " " << topk
                << std::endl;

    std::vector<int32_t> kvCacheActSeqVec(b);
    readInput<int32_t>(GetGoldenDir() + "/actual_seq.bin", kvCacheActSeqVec);
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    params.maxBlockNumPerBatch = maxBlockNumPerBatch;
    std::cout << "========= maxBlockNumPerBatch " << maxBlockNumPerBatch << std::endl;

    DataType dType = DT_BF16;
    using T = npu::tile_fwk::bfloat16;
    DataType dTypeQuant = DT_INT8;
    using TQuant = int8_t;

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
    auto dynamicX = CreateTensorAndData<T>(xShape, dType, "x", "/x.bin", {0, 1});
    SymbolicScalar bSymbol = GetInputShape(dynamicX.tensor, 0);
    SymbolicScalar s1Symbol = GetInputShape(dynamicX.tensor, 1);
    auto wDq = CreateTensorAndData<T>(wDqShape, dType, "wDq", TileOpFormat::TILEOP_NZ, "/wDq.bin");
    auto wUqQr = CreateTensorAndData<TQuant>(wUqQrShape, dTypeQuant, "wUqQr", TileOpFormat::TILEOP_NZ, "/wUqQr.bin");
    auto wDkvKr = CreateTensorAndData<T>(wDkvKrShape, dType, "wDkvKr", TileOpFormat::TILEOP_NZ, "/wDkvKr.bin");
    auto wUk = CreateTensorAndData<T>(wUkShape, dType, "wUk", TileOpFormat::TILEOP_ND, "/wUk.bin");
    auto dequantScaleWUqqr = CreateTensorAndData<float>(dequantScaleWUqqrShape, DT_FP32, "dequantScaleWUqqrShape", "/w_qb_scale.bin");
    auto gammaCq = CreateTensorAndData<T>(gammaCqShape, dType, "gammaCq", TileOpFormat::TILEOP_ND, "/gamma_cq.bin");
    auto gammaCkv = CreateTensorAndData<T>(gammaCkvShape, dType, "gammaCkv", TileOpFormat::TILEOP_ND, "/gamma_ckv.bin");
    auto dynamicCos = CreateTensorAndData<T>(cosShape, dType, "cos", "/cos.bin", {0, 1});
    auto dynamicSin = CreateTensorAndData<T>(cosShape, dType, "sin", "/sin.bin", {0, 1});
    auto kvCache = CreateTensorAndData<TQuant>(kvCacheShape, dTypeQuant, "kvCache", "/kv_cache.bin", {0});
    auto krCache = CreateTensorAndData<T>(krCacheShape, dType, "krCache", "/kr_cache.bin", {0});
    auto kScaleCache = CreateTensorAndData<float>(kScaleCacheShape, DT_FP32, "kScaleCache", "/kv_quant_scale_cache.bin", {0});

    auto wIdxQb = CreateTensorAndData<TQuant>(wIdxQbShape, dTypeQuant, "wIdxQb", TileOpFormat::TILEOP_NZ, "/w_idx_qb_nz.bin");
    auto wIdxQbScale = CreateTensorAndData<float>(wIdxQbScaleShape, DT_FP32, "wIdxQbScale", TileOpFormat::TILEOP_ND, "/w_idx_qb_scale.bin");
    auto wIdxK = CreateTensorAndData<T>(wIdxKShape, dType, "wIdxK", TileOpFormat::TILEOP_NZ, "/w_idx_k_nz.bin");
    auto wIdxProj = CreateTensorAndData<T>(wIdxProjShape, dType, "wIdxProj", TileOpFormat::TILEOP_NZ, "/w_idx_proj_nz.bin");
    auto layernormGammaK = CreateTensorAndData<T>(layernormGammaKShape, dType, "layernormGammaK", "/ln_gamma.bin");
    auto layernormBetaK = CreateTensorAndData<T>(layernormBetaKShape, dType, "layernormBetaK", "/ln_beta.bin");
    auto hadamardQ = CreateTensorAndData<T>(hadamardQShape, dType, "hadamardQ", "/hadamard_q.bin");
    auto hadamardK = CreateTensorAndData<T>(hadamardKShape, dType, "hadamardK", "/hadamard_k.bin");
    auto idxKCache = CreateTensorAndData<TQuant>(idxKCacheShape, dTypeQuant, "idxKCache", "/k_cache.bin", {0});
    auto idxKScaleCache = CreateTensorAndData<npu::tile_fwk::float16>(idxKScaleCacheShape, DT_FP16, "idxKScaleCache", "/k_scale_cache.bin", {0});

    auto dynamicCacheIndex = CreateTensorAndData<int64_t>(kvLenShape, DT_INT64, "cacheIndex", "/kv_len.bin", {0, 1});
    auto dynamicBlockTable = CreateTensorAndData<int32_t>(blockTableShape, DT_INT32, "blockTable", "/block_table.bin", {0, 1});

    auto dynamicActualSeqLengthsKey = CreateTensorAndData<int32_t>(actualSeqShape, DT_INT32, "actualSeqLengthsKey", "/actual_seq.bin", {0});

    // 3. 构造output
    auto dynamicSaOut = CreateConstantDynamicOutputTensor<T>(saOutShape, dType, "saOut", 0, {bSymbol, s1Symbol, n1, dn});

    auto debugQNopeOut = CreateConstantDynamicOutputTensor<T>(qNopeShape, dType, "debugQNopeOut", 0, {bSymbol * s1Symbol, n1, dn});
    auto debugQRopeOut = CreateConstantDynamicOutputTensor<T>(qRopeShape, dType, "debugQRopeOut", 0, {bSymbol * s1Symbol, n1, dr});
    auto debugRmsNormOut = CreateConstantDynamicOutputTensor<int8_t>(rmsNormShape, DT_INT8, "debugRmsNormOut", 1, {bSymbol * s1Symbol, qLoraRank});
    auto debugRmsNormScaleOut = CreateConstantDynamicOutputTensor<float>(rmsNormScaleShape, DT_FP32, "debugRmsNormScaleOut", 0, {bSymbol * s1Symbol, 1});

    auto debugQInt8Out = CreateConstantDynamicOutputTensor<int8_t>(qInt8Shape, DT_INT8, "debugQInt8Out", 0, {bSymbol * s1Symbol, idx_n_heads, idx_head_dim});
    auto debugQScaleOut = CreateConstantDynamicOutputTensor<npu::tile_fwk::float16>(qScaleShape, DT_FP16, "debugQScaleOut", 0, {bSymbol * s1Symbol, idx_n_heads, 1});
    auto debugWeightsOut = CreateConstantDynamicOutputTensor<npu::tile_fwk::float16>(weightsShape, DT_FP16, "debugWeightsOut", 0, {bSymbol * s1Symbol, params.idx_n_heads});

    auto indexerTopkOut = CreateConstantDynamicOutputTensor<int32_t>(indexerTopkShape, DT_INT32, "indexerTopkOut", 0, {bSymbol, s1Symbol, n2, topk});
    auto indexerTopkValueOut = CreateConstantDynamicOutputTensor<float>(indexerTopkShape, DT_FP32, "indexerTopkValueOut", 0, {bSymbol, s1Symbol, n2, topk});
    auto indexerTopkTmpOut = CreateConstantDynamicOutputTensor<float>(indexerTopkTmpOutShape, DT_FP32, "indexerTopkTmpOut", 0, {bSymbol * s1Symbol * n2, maxBlockNumPerBatch * blockSize});

    // 4. golden
    auto qNopeGolden = GetGoldenVec<T>(qNopeShape, "/q_golden.bin");
    auto qRopeGolden = GetGoldenVec<T>(qRopeShape, "/q_rope_golden.bin");
    auto kNopeGolden = GetGoldenVec<int8_t>(kvCacheShape, "/kv_cache_golden.bin");
    auto kRopeGolden = GetGoldenVec<T>(krCacheShape, "/kr_cache_golden.bin");
    auto kNopeScaleGolden = GetGoldenVec<float>(kScaleCacheShape, "/kv_quant_scale_cache_golden.bin");
    auto rmsNormGolden = GetGoldenVec<int8_t>(rmsNormShape, "/rms_norm_golden.bin");
    auto rmsNormScaleGolden = GetGoldenVec<float>(rmsNormScaleShape, "/rms_norm_scale_golden.bin");
    auto qInt8Golden = GetGoldenVec<int8_t>(qInt8Shape, "/query_golden.bin");
    auto qScaleGolden = GetGoldenVec<npu::tile_fwk::float16>(qScaleShape, "/query_scale_golden.bin");
    auto kInt8Golden = GetGoldenVec<int8_t>(idxKCacheShape, "/idx_k_cache_golden.bin");
    auto kScaleGolden = GetGoldenVec<npu::tile_fwk::float16>(idxKScaleCacheShape, "/idx_k_scale_cache_golden.bin");
    auto weightsGolden = GetGoldenVec<npu::tile_fwk::float16>(weightsShape, "/weights_golden.bin");

    auto indexerTopkResGolden = GetGoldenVec<int32_t>(indexerTopkShape, "/topk_res.bin");
    auto indexerTopkValueGolden = GetGoldenVec<float>(indexerTopkShape, "/topk_value.bin");
    auto indexerTopkTmpOutGolden = GetGoldenVec<float>(indexerTopkTmpOutShape, "/tmp_out.bin");

    auto saOutResultGolden = GetGoldenVec<T>(saOutShape, "/attn_golden.bin");

    // 5.input/outputDataList
    std::vector<RawTensorDataPtr> outputDataList = {
        dynamicSaOut.dataPtr,
#if QUANT_DSIA_DEBUG == 1
        debugQNopeOut.dataPtr, debugQRopeOut.dataPtr, debugRmsNormOut.dataPtr, debugRmsNormScaleOut.dataPtr,
        debugQInt8Out.dataPtr, debugQScaleOut.dataPtr, debugWeightsOut.dataPtr,
        indexerTopkOut.dataPtr, indexerTopkValueOut.dataPtr, indexerTopkTmpOut.dataPtr
#endif
    };
    std::vector<RawTensorDataPtr> inputDataList = {
        dynamicX.dataPtr, wDq.dataPtr, wUqQr.dataPtr, wUk.dataPtr, wDkvKr.dataPtr,
        gammaCq.dataPtr, gammaCkv.dataPtr, dynamicCos.dataPtr, dynamicSin.dataPtr, kvCache.dataPtr,
        krCache.dataPtr, kScaleCache.dataPtr, dequantScaleWUqqr.dataPtr, wIdxQb.dataPtr, wIdxQbScale.dataPtr,
        wIdxK.dataPtr, wIdxProj.dataPtr, layernormGammaK.dataPtr, layernormBetaK.dataPtr, hadamardQ.dataPtr,
        hadamardK.dataPtr, idxKCache.dataPtr, idxKScaleCache.dataPtr, dynamicBlockTable.dataPtr,
        dynamicCacheIndex.dataPtr, dynamicActualSeqLengthsKey.dataPtr
    };

    DiaQuantAttr attrs;
    attrs.rmsnormEpsilonCq = params.eps;
    attrs.rmsnormEpsilonCkv = params.eps;
    attrs.layernormEpsilonK = 1e-6f;
    attrs.attnSoftmaxScale = static_cast<float>(1.0 / sqrtf(dn + dr));
    attrs.selectedCount = topk;
    attrs.layeroutKey = layoutKey;
    attrs.layeroutQuery = "TND";

    DeepSeekIndexerAttentionQuant(dynamicX.tensor, wDq.tensor, wUqQr.tensor, wUk.tensor, wDkvKr.tensor,
                                  gammaCq.tensor, gammaCkv.tensor, dynamicCos.tensor, dynamicSin.tensor, dynamicCacheIndex.tensor,
                                  kvCache.tensor, krCache.tensor, kScaleCache.tensor, dequantScaleWUqqr.tensor,
                                  wIdxQb.tensor, wIdxQbScale.tensor, wIdxK.tensor, wIdxProj.tensor,
                                  layernormGammaK.tensor, layernormBetaK.tensor, hadamardQ.tensor, hadamardK.tensor,
                                  idxKCache.tensor, idxKScaleCache.tensor,
                                  dynamicActualSeqLengthsKey.tensor, dynamicBlockTable.tensor,
                                  dynamicSaOut.tensor, attrs, params,
                                  // debug
                                  debugQNopeOut.tensor, debugQRopeOut.tensor, debugRmsNormOut.tensor, debugRmsNormScaleOut.tensor,
                                  debugQInt8Out.tensor, debugQScaleOut.tensor, debugWeightsOut.tensor,
                                  indexerTopkOut.tensor, indexerTopkValueOut.tensor, indexerTopkTmpOut.tensor
                                  );

    // MlaProlog
    uint64_t queryNopeOutBuffer = b * s1 * n1 * dn * BytesOf(dType);
    uint64_t queryRopeOutBuffer = b * s1 * n1 * dr * BytesOf(dType);
    uint64_t qNormResBuffer = b * s1 * qLoraRank * BytesOf(DT_INT8);
    uint64_t qNormScaleResBuffer = b * s1 * BytesOf(DT_FP32);
    // IndexerProlog
    uint64_t qInt8OutBuffer = b * s1 * idx_n_heads * idx_head_dim * BytesOf(DT_INT8);
    uint64_t qScaleOutBuffer = b * s1 * idx_n_heads * BytesOf(DT_FP16);
    uint64_t weightOutBuffer = b * s1 * idx_n_heads * BytesOf(DT_FP16);
    auto totalBuffer =
        queryNopeOutBuffer + queryRopeOutBuffer + qNormResBuffer + qNormScaleResBuffer +
        qInt8OutBuffer + qScaleOutBuffer + weightOutBuffer;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList,
                       DeviceLauncherConfig(totalBuffer));

    // mla prolog debug output
#if QUANT_DSIA_DEBUG == 1
    std::cout << "qNopeGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(qNopeGolden, (T*)debugQNopeOut.dataPtr->data(), 0.005f, 0, 1000));
    std::cout << "qRopeGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(qRopeGolden, (T*)debugQRopeOut.dataPtr->data(), 0.005f, 0, 1000));
    std::cout << "rmsNormGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmpAbsDelta(rmsNormGolden, (int8_t*)debugRmsNormOut.dataPtr->data(), 1.0f, 0));
    std::cout << "rmsNormScaleGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(rmsNormScaleGolden, (float*)debugRmsNormScaleOut.dataPtr->data(), 0.001f, 0, 1000));
#endif
    std::cout << "kNopeGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmpAbsDelta(kNopeGolden, (int8_t*)kvCache.dataPtr->data(), 1.0f, 0));
    std::cout << "kRopeGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(kRopeGolden, (T*)krCache.dataPtr->data(), 0.001f, 0, 1000));
    std::cout << "kNopeScaleGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(kNopeScaleGolden, (float*)kScaleCache.dataPtr->data(), 0.001f, 0, 1000));

    // indexer prolog debug output
#if QUANT_DSIA_DEBUG == 1
    std::cout << "qInt8Golden ====== " << std::endl;
    EXPECT_TRUE(resultCmpAbsDelta(qInt8Golden, (int8_t*)debugQInt8Out.dataPtr->data(), 1.0f, 0));
    std::cout << "qScaleGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(qScaleGolden, (npu::tile_fwk::float16*)debugQScaleOut.dataPtr->data(), 0.0005f, 0, 1000, false, true));
    std::cout << "weightsGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(weightsGolden, (npu::tile_fwk::float16*)debugWeightsOut.dataPtr->data(), 0.0005f, 0, 1000));
#endif
    std::cout << "kInt8Golden ====== " << std::endl;
    EXPECT_TRUE(resultCmpAbsDelta(kInt8Golden, (int8_t*)idxKCache.dataPtr->data(), 1.0f, 0));
    std::cout << "kScaleGolden ====== " << std::endl;
    EXPECT_TRUE(resultCmp(kScaleGolden, (npu::tile_fwk::float16*)idxKScaleCache.dataPtr->data(), 0.0001f, 0, 1000));

#if QUANT_DSIA_DEBUG == 1
    // indexer topk debug output
    // std::cout << "indexerTopkTmpOut result ====== " << std::endl;
    // EXPECT_TRUE(resultCmp(indexerTopkTmpOutGolden, (float *)indexerTopkTmpOut.dataPtr->data(), 0.01f, 0, 1000, true));
    std::cout << "indexerTopkValue result ====== " << std::endl;
    EXPECT_TRUE(resultCmp(indexerTopkValueGolden, (float *)indexerTopkValueOut.dataPtr->data(), 0.05f, 0, 1000, false));
    std::cout << "indexerTopkRes result ====== " << std::endl;
    EXPECT_TRUE(resultCmp4TopK(indexerTopkResGolden, (int32_t *)indexerTopkOut.dataPtr->data(), topk, 0.0005f));
    // EXPECT_TRUE(resultCmp(indexerTopkResGolden, (int32_t *)indexerTopkOut.dataPtr->data(), 0.0005f, 0, 1000, true));
#endif
    std::cout << "selectAtten result ====== " << std::endl;
    EXPECT_TRUE(resultCmp(saOutResultGolden, (T *)dynamicSaOut.dataPtr->data(), 0.005f, 0, 1000));
}

void test_common(DSIASimpleParams params) {
    int paramsSize = 6;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0];
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    params.blockNum = inputParams[5];
    params.topk = 2048;
    params.cacheMode = "PA_BSND";

    MlaTileConfig prologConfig;
    prologConfig.tileBS = 8;

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
    saTileConfig.c1TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {8, 2048}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128}; // (n1, d)

    params.mlaTileCfg = prologConfig;
    params.indexTileCfg = indexerTile;
    params.salTileCfg = saTileConfig;

    TestDeepSeekIndexerAttentionQuantSTest(params);
}

TEST_F(DeepSeekIndexerAttentionQuantSTest, 4B_mtp) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DeepSeekIndexerAttentionQuantSTest, 4B_mtp_perf) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DeepSeekIndexerAttentionQuantSTest, 32B) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common(params);
}

} // namespace
