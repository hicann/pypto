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
 * \file test_decode_sparse_attention.cpp
 * \brief
 */

#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"
#include "test_common.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek_v3.2_exp/decode_indexer_attention.h"
#include "operator/models/deepseek_v3.2_exp/dsia_common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tensor.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DecodeIndexerAttentionSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};
namespace {
void SetPreConfig() {
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
    int dr = params.rope_dim;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int blockSize = params.blockSize;
    int idx_n_heads = params.idx_n_heads;
    int idx_head_dim = params.idx_head_dim;

    std::vector<int> kvCacheActSeqVec(b);
    readInput<int>(GetGoldenDir() + "/kv_cache_actual_seq_len.bin", kvCacheActSeqVec);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    params.blockNum = blockNum;
    std::cout << "========= blockNum " << blockNum << std::endl;
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

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

    // 2. 构造tensor
    // MlaProlog
    // auto dynamicX = CreateTensorAndData<T>(xShape, dType, "x", "/x.bin");
    auto dynamicX = CreateTensorAndData<T>(xShape, dType, "x", "/x.bin", {0, 1});
    SymbolicScalar bSymbol = GetInputShape(dynamicX.tensor, 0);
    SymbolicScalar s1Symbol = GetInputShape(dynamicX.tensor, 1);

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    std::string nz_prefix = nz ? "/nz_" : "";
    auto wDq = CreateTensorAndData<T>(wDqShape, dType, "wDq", weightFormat, nz_prefix + "wDq.bin");
    auto wUqQr = CreateTensorAndData<wDtype>(wUqQrShape, dTypeQuant, "wUqQr", weightFormat, nz_prefix + "wUqQr.bin");
    const bool usePrefetch = true;
    if constexpr (usePrefetch) {
        wDq.tensor.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.tensor.SetCachePolicy(CachePolicy::PREFETCH, true);
    }

    auto wDkvKr = CreateTensorAndData<T>(wDkvKrShape, dType, "wDkvKr", weightFormat, nz_prefix + "wDkvKr.bin");
    auto wUk = CreateTensorAndData<T>(wUkShape, dType, "wUk", TileOpFormat::TILEOP_ND, "/wUk.bin");
    auto gammaCq = CreateTensorAndData<T>(gammaCqShape, dType, "gammaCq", TileOpFormat::TILEOP_ND, "/gamma_cq.bin");
    auto gammaCkv = CreateTensorAndData<T>(gammaCkvShape, dType, "gammaCkv", TileOpFormat::TILEOP_ND, "/gamma_ckv.bin");

    auto dynamicCos = CreateTensorAndData<T>(cosShape, dType, "cos", "/cos.bin", {0, 1});
    auto dynamicSin = CreateTensorAndData<T>(cosShape, dType, "sin", "/sin.bin", {0, 1});
    auto dynamicCacheIndex =
        CreateTensorAndData<int32_t>(kvLenShape,  DT_INT32, "cacheIndex", "/k_cache_index.bin", {0, 1});

    auto kvCache = CreateTensorAndData<T>(kvCacheShape, dType, "kvCache", "/kv_cache.bin", {0});
    auto krCache = CreateTensorAndData<T>(krCacheShape, dType, "krCache", "/kr_cache.bin", {0});

    auto indexKCache = CreateTensorAndData<T>(keyShape, dType, "kCache", "/idx_k_cache.bin", {0});

    // MlaProlog output
    // std::vector<SymbolicScalar> kvCacheDynamicOutShape = {GetInputShape(kvCache.tensor, 0), blockSize, n2, dn};
    // std::vector<SymbolicScalar> krCacheDynamicOutShape = {GetInputShape(krCache.tensor, 0), blockSize, n2, qkRopeHeadDim};

    auto dynamicBlockTable =
        CreateTensorAndData<int32_t>(blockTableShape, DT_INT32, "blockTable", "/block_table.bin", {0, 1});
    auto dynamicActSeqs =
        CreateTensorAndData<int32_t>(actSeqsShape, DT_INT32, "actSeqs", "/kv_cache_actual_seq_len.bin", {0});
    SymbolicScalar maxBlockNumSymbol = GetInputShape(dynamicBlockTable.tensor, 1);
    weightFormat = TileOpFormat::TILEOP_NZ;

    auto qW = CreateTensorAndData<T>({qLoraRank, idx_n_heads * idx_head_dim}, dType, "qW", weightFormat, "/wq_b_nz.bin");
    auto kW = CreateTensorAndData<T>({h, idx_head_dim}, dType, "kW", weightFormat, "/wk_nz.bin");
    auto projW = CreateTensorAndData<T>({h, idx_n_heads}, dType, "projW", weightFormat, "/weights_proj_nz.bin");
    auto lnW = CreateTensorAndData<T>({idx_head_dim}, dType, "lnW", "/weight_layer_norm.bin");
    auto lnBias = CreateTensorAndData<T>({idx_head_dim}, dType, "lnBias", "/bias_layer_norm.bin");

    auto dynamicTmpTopkInput =
        CreateTensorAndData<int32_t>(tmpTopkInputShape, DT_INT32, "tmpTopkInput", "/topk_res.bin", {0, 1});

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

    // 4. golden
    std::vector<T> golden3 = GetGoldenVec<T>(kvCacheShape, "/kv_cache_golden.bin");
    std::vector<T> golden4 = GetGoldenVec<T>(krCacheShape, "/kr_cache_golden.bin");
    std::vector<T> golden5 = GetGoldenVec<T>(rmsResShape, "/q_a_rms_norm.bin");

    std::vector<T> queryOutGolden = GetGoldenVec<T>(queryShape, "/query.bin");
    std::vector<T> weightsOutGolden = GetGoldenVec<T>(weightShape, "/weights.bin");

    std::vector<T> qNopeOutGolden = GetGoldenVec<T>(qNopeOutShape, "/q_golden.bin");
    std::vector<T> qRopeOutGolden = GetGoldenVec<T>(qRopeOutShape, "/q_rope_golden.bin");

    std::vector<T> saoutResultGolden = GetGoldenVec<T>(saOutShape, "/atten_out.bin");
    auto kCacheOutGolden = GetGoldenVec<T>(keyShape, "/key.bin");
    auto tmpRowSumOutGolden = GetGoldenVec<float>({b * s1 * n2, maxBlockNumPerBatch * blockSize}, "/tmp_out.bin");
    auto indexerTopkResGolden = GetGoldenVec<int32_t>({b, s1, n2, params.topk}, "/topk_res.bin");

    // 5.input/outputDataList
    std::vector<RawTensorDataPtr> outputDataList = {dynamicSaOut.dataPtr};
    std::vector<RawTensorDataPtr> inputDataList = {dynamicX.dataPtr, wDq.dataPtr, wUqQr.dataPtr, wUk.dataPtr, wDkvKr.dataPtr, gammaCq.dataPtr,
        gammaCkv.dataPtr, dynamicSin.dataPtr, dynamicCos.dataPtr, dynamicCacheIndex.dataPtr, kvCache.dataPtr, krCache.dataPtr, dynamicBlockTable.dataPtr, dynamicActSeqs.dataPtr, qW.dataPtr,
        kW.dataPtr, projW.dataPtr, lnW.dataPtr, lnBias.dataPtr, indexKCache.dataPtr};
    MlaQuantInputs quantInputs;

#if DSIA_DEBUG == 1
    // tmp out
    outputDataList.emplace_back(rmsResOut.dataPtr);
    outputDataList.emplace_back(queryOut.dataPtr);
    outputDataList.emplace_back(weightOut.dataPtr);
    outputDataList.emplace_back(qNopeOut.dataPtr);
    outputDataList.emplace_back(qRopeOut.dataPtr);

    outputDataList.emplace_back(dynamicTmpRowSumOut.dataPtr);
    outputDataList.emplace_back(dynamicTmpIndexerTopkRes.dataPtr);
    auto gatherResGolden = GetGoldenVec<T>(gatherResShape, "/k_slc.bin");
    outputDataList.emplace_back(dynamicGatherRes.dataPtr);
    // tmp in
    inputDataList.emplace_back(dynamicTmpTopkInput.dataPtr);
#endif // DSIA_DEBUG

    DecodeIndexerAttention(dynamicX.tensor, wDq.tensor, wUqQr.tensor, wUk.tensor, wDkvKr.tensor, gammaCq.tensor, gammaCkv.tensor, dynamicSin.tensor, dynamicCos.tensor, dynamicCacheIndex.tensor, kvCache.tensor, krCache.tensor, quantInputs,
        dynamicBlockTable.tensor, dynamicActSeqs.tensor, qW.tensor, kW.tensor, projW.tensor, lnW.tensor, lnBias.tensor, indexKCache.tensor, dynamicSaOut.tensor,
        dynamicGatherRes.tensor, dynamicTmpTopkInput.tensor, dynamicTmpIndexerTopkRes.tensor, dynamicTmpRowSumOut.tensor,
        rmsResOut.tensor,queryOut.tensor,weightOut.tensor,qNopeOut.tensor,qRopeOut.tensor,
        params);

#ifdef BUILD_WITH_CANN
    uint64_t queryNopeOutBuffer = b * s1 * n1 * dn * BytesOf(dType);
    uint64_t queryRopeOutBuffer = b * s1 * n1 * dr * BytesOf(dType);
    uint64_t rmsResBuffer = b * s1 * qLoraRank * BytesOf(dType);
    uint64_t queryOutBuffer = b * s1 * idx_n_heads * idx_head_dim * BytesOf(dType);
    uint64_t weightOutBuffer = b * s1 * idx_n_heads * BytesOf(dType);
    uint64_t gatherResBuffer = b * s1 * params.topk * (dn + qkRopeHeadDim) * BytesOf(dType);
    uint64_t indexerTopkResBuffer = b * s1 * n2 * params.topk * BytesOf(DT_INT32);
    auto totalBuffer = queryNopeOutBuffer + queryRopeOutBuffer + rmsResBuffer + queryOutBuffer + weightOutBuffer + gatherResBuffer + indexerTopkResBuffer;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList, DeviceLauncherConfig(totalBuffer)); // output list
    std::cout << "MlaProlog kv ====== " << std::endl;
    EXPECT_TRUE(resultCmpPrint<T>(golden3, (T *)kvCache.dataPtr->data(), 0.003f, 3));
    std::cout << "MlaProlog kr ====== " << std::endl;
    EXPECT_TRUE(resultCmpPrint<T>(golden4, (T *)krCache.dataPtr->data(), 0.003f, 3));
    std::cout << "kCacheOut ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(kCacheOutGolden, (T *)indexKCache.dataPtr->data(), 0.003f, 0, 1000, false, true, 0));

#if DSIA_DEBUG == 1
    std::cout << "tmpRowSumOut result ====== " << std::endl;
    EXPECT_TRUE(resultCmp(tmpRowSumOutGolden, (float *)dynamicTmpRowSumOut.dataPtr->data(), 0.01f, 0, 1000, false));
    std::cout << "indexerTopkRes result ====== " << std::endl;
    resultCmpPrint(indexerTopkResGolden, (int32_t *)dynamicTmpIndexerTopkRes.dataPtr->data(), 0.003f, 8);
    std::cout << "!!!!!!!!!!!!!!! dump topk_input_row_sum and topk_output" << std::endl;
    dynamicTmpRowSumOut.dataPtr->ToFile("topk_input_row_sum_npu.bin"); // build/tests/st
    dynamicTmpIndexerTopkRes.dataPtr->ToFile("topk_output_npu.bin");
    std::cout << "!!!!!!!!!!!!!!! replace gather_topk_input by topk_res_golden !!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "gatherRes result ====== " << std::endl;
    EXPECT_TRUE(resultCmp(gatherResGolden, (T *)dynamicGatherRes.dataPtr->data(), 0.0003f));
#endif
    std::cout << "selectAtten result ====== " << std::endl;
    EXPECT_TRUE(resultCmpPrint(saoutResultGolden, (T *)dynamicSaOut.dataPtr->data(), 0.0005f, 8));
#endif
}

template <typename T = npu::tile_fwk::float16>
void test_common(DSIASimpleParams params) {
    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0];                                         // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    params.topk = 2048;
    params.cacheMode = "PA_BSND";
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];
    std::cout << "=========== DecodeIndexerAttentionSTest: isQuant: " << isQuant << ", isSmooth: " << isSmooth << std::endl;

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

    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestDecodeIndexerAttentionSTest<T, int8_t, true, false>(params);
        } else {
            TestDecodeIndexerAttentionSTest<T, int8_t, false, false>(params);
        }
    } else {
        TestDecodeIndexerAttentionSTest<T, T, false, true>(params);
    }
}

TEST_F(DecodeIndexerAttentionSTest, mini) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common<npu::tile_fwk::bfloat16>(params);
}

TEST_F(DecodeIndexerAttentionSTest, 32B) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common<npu::tile_fwk::bfloat16>(params);
}

TEST_F(DecodeIndexerAttentionSTest, 24B) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common<npu::tile_fwk::bfloat16>(params);
}

TEST_F(DecodeIndexerAttentionSTest, 48B) {
    DSIASimpleParams params = DSIASimpleParams::getDecodeParams();
    test_common<npu::tile_fwk::bfloat16>(params);
}
} // namespace