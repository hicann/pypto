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
 * \file test_nsa_slc_attention.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class NSAUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        // skip pass, ut only execute model op code
        config::SetPassDefaultConfig(KEY_DISABLE_PASS, true);
    }

    void TearDown() override {}
};

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool isSmooth = false, bool nz = false, bool debug = false>
void TestNsa(const NSAV1SimpleParams &params, const MlaTileConfig &prologConfig, WinAttenTileShapeConfig &winAttntileConfig, SATileShapeConfig& saTileConfig,
    PostTileConfig& postConfig, CmpAttnTile &cmpTileConfig, std::string cacheMode = "PA_BSND") {
    float eps = params.eps;
    int b = params.b;
    int s1 = params.s1;
    int s2 = params.s2;
    int n1 = params.n1;
    int n2 = params.n2;
    int h = params.h;
    int v_dim = params.kv_lora_rank;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int smax = params.topk * params.slcBlockSize;
    int dn = v_dim;
    int dr = params.rope_dim;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int blockSize = params.blockSize;
    int winSize = params.winSize;
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;

    std::vector<int> kvCacheActSeqVec(b, s2);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    std::cout << "========= blockNum " << blockNum << std::endl;
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    int vHeadDim = params.vHeadDim;
    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    // 1. 设置shape
    // MlaProlog
    std::vector<int64_t> xShape = {b, s1, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, v_dim + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n1, qkNopeHeadDim, v_dim};
    std::vector<int64_t> cosShape = {b, s1, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {v_dim};
    std::vector<int64_t> kvLenShape = {b, s1};
    std::vector<int64_t> kvCacheShape = {b, n2, s2, v_dim};
    std::vector<int64_t> krCacheShape = {b, n2, s2, qkRopeHeadDim};
    std::vector<int64_t> kvCacheOutShape = {b, n2, s2, v_dim};
    std::vector<int64_t> krCacheOutShape = {b, n2, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        int blockNum2 = b * (s2 / blockSize);
        std::cout << "========= blockNum2 " << blockNum2 << std::endl;
        kvCacheShape = {blockNum, blockSize, n2, v_dim};
        krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
        kvCacheOutShape = {blockNum * blockSize, n2 * v_dim};
        krCacheOutShape = {blockNum * blockSize, n2 * qkRopeHeadDim};
    }
    std::vector<int64_t> wQbScaleShape = {1, n1 * qHeadDim};
    std::vector<int64_t> smoothCqShape{1, qLoraRank};
    std::vector<int64_t> qOutShape = {b, s1, n1, v_dim};
    std::vector<int64_t> qRopeOutShape = {b, s1, n1, qkRopeHeadDim};

    std::vector<int64_t> topkIndicesShape = {b, s1, topk - front - near};
    std::vector<int64_t> topkTensorShapeShape = {b, s1};
    std::vector<int64_t> kvNopeCacheShape = {int(blockNum * blockSize), n2 * dn};
    std::vector<int64_t> kRopeCacheShape = {int(blockNum * blockSize), n2 * dr};
    std::vector<int64_t> kvCacheActSeqShape = {b};
    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};

    std::vector<int64_t> slcActSeqsShape = {b, s1};
    std::vector<int64_t> qNopeShape = {b * s1 * n1, dn};
    std::vector<int64_t> qRopeShape = {b * s1 * n1, dr};
    std::vector<int64_t> kSlcShape = {b * s1 * n2 * smax, dn + dr};
    std::vector<int64_t> vSlcShape = {b * s1 * n2 * smax, dn};

    std::vector<int64_t> gateW1Shape = {h, 4 * h};
    std::vector<int64_t> gateW2Shape = {4 * h, 3 * n1};
    std::vector<int64_t> gateSimW1Shape = {h, 3 * n1};
    // std::vector<int64_t> gatingScoreShape = {b, s1, n1, 3};

    std::vector<int64_t> shape_cmpAtten = {b, s1, n1, v_dim};
    std::vector<int64_t> shape_selAtten = {b, s1, n1, v_dim};
    std::vector<int64_t> shape_winAtten = {b, s1, n1, v_dim};
    std::vector<int64_t> shape_attentionOut = {b, s1, n1, v_dim};

    // post: shape
    std::vector<int64_t> wUvShape = {n1, v_dim, vHeadDim};
    std::vector<int64_t> woShape = {n1 * vHeadDim, h};
    std::vector<int64_t> woScaleShape = {1, h};
    std::vector<int64_t> smoothWoShape = {1, n1 * vHeadDim};
    std::vector<int64_t> outShape = {b, s1, h};

    // 2. 构造tensor
    // MlaProlog
    Tensor x(dType, xShape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", weightFormat);
    const bool usePrefetch = true;
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", weightFormat);
    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");
    Tensor cos(dType, cosShape, "cos");
    Tensor sin(dType, cosShape, "sin");
    Tensor cacheIndex(DT_INT64, kvLenShape, "cacheIndex"); // int64
    Tensor kvCache(dType, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor wQbScale(DT_FP32, wQbScaleShape, "wQbScale");
    Tensor smoothCq(DT_FP32, smoothCqShape, "smoothCq");
    // MlaProlog output
    Tensor outputKvCache(dType, kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheOutShape, "outputKrCache");

    Tensor topkIndices(DT_INT32, topkIndicesShape, "topkTensor");
    Tensor topkTensorShape(DT_INT32, topkTensorShapeShape, "topkTensorShape");
    Tensor kvNopeCache(dType, kvNopeCacheShape, "kNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "vNopeCache");
    Tensor kvCacheActSeq(DT_INT32, kvCacheActSeqShape, "kvCacheActSeq");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");

    Tensor slcActSeqs(DT_INT32, slcActSeqsShape, "slcActSeqs");
    // Tensor qNope(dType, qNopeShape, "qNope");
    // Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kSlc(dType, kSlcShape, "kSlc");
    Tensor vSlc(dType, vSlcShape, "vSlc");

    Tensor gateW1(dType, gateW1Shape, "gateW1");
    Tensor gateW2(dType, gateW2Shape, "gateW2");
    Tensor gateSimW1(dType, gateSimW1Shape, "gateSimW1");
    // Tensor gatingScore(dType, gatingScoreShape, "gatingScore");

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor slcAttn(DT_FP32, shape_selAtten, "selAtten"); // fp32输入
    // Tensor winAtten(DT_FP32, shape_winAtten, "winAtten");

    Tensor kvSlcActSeqsMidOut(DT_INT32, slcActSeqsShape, "kvSlcActSeqsMidOut");
    Tensor attenOut(dType, shape_attentionOut, "attenOut");

    // post: Tensor
    Tensor wUv(dType, wUvShape, "wUv");
    Tensor wUvScale;
    Tensor smoothWUv;
    Tensor wo(dTypeQuant, woShape, "wo", weightFormat);
    Tensor woScale;
    Tensor smoothWo;
    Tensor postOut(dType, outShape, "postOut");

    int paramsSize = 10;
    std::vector<int> input_param(paramsSize);

    const int b_v2 = params.b;;
    const int dq = v_dim+ dr;
    const int dv = v_dim;
    const int cmpBlockSize = NUM_32;
    const int cmpStride = NUM_16;
    softmaxScale = static_cast<float>(1.0 / sqrtf((dq)));

    DataType qType = dType;
    DataType kType = dType;

    // Read actSeqLen_v2
    std::vector<int> actSeq(b_v2,s2);
    //    int blockNum = 0;
    for (auto s : actSeq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable_v2: (b_v2, maxBlockNum)
    int maxBlockNum = CeilDiv(s2, blockSize);

    // Read actCmpSeqLen_v2
    std::vector<int> actCmpSeq(b_v2, (s2-cmpBlockSize)/cmpStride+1);
    int cmpBlockNum = 0;
    for (auto s : actCmpSeq) {
        cmpBlockNum += CeilDiv(s, blockSize);
    }
    // cmpBlockTable_v2: (b_v2, maxCmpBlockNum)
    int maxCmpSeq = *(std::max_element(actCmpSeq.begin(), actCmpSeq.end()));
    int maxCmpBlockNum = CeilDiv(maxCmpSeq, blockSize);

    // Construct input tensors
    Tensor cmpKvCache_v2(kType, {cmpBlockNum * blockSize, n2 * dv}, "cmpKvCache_v2");
    Tensor cmpKrCache_v2(kType, {cmpBlockNum * blockSize, n2 * dr}, "cmpKrCache_v2");
    Tensor cmpBlockTable_v2(DT_INT32, {b_v2, maxCmpBlockNum}, "cmpBlockTable_v2");
    Tensor actSeqLen_v2(DT_INT32, {b_v2}, "actSeqLen_v2");
    Tensor actCmpSeqLen_v2(DT_INT32, {b_v2}, "actCmpSeqLen_v2");
    Tensor mlpWk1_v2(kType, {cmpBlockSize * dq, 2 * cmpBlockSize * dq}, "mlpWk1_v2");
    Tensor mlpWk2_v2(kType, {2 * cmpBlockSize * dq, dq}, "mlpWk2_v2");
    Tensor mlpCos_v2(kType, {b_v2, cmpBlockSize, dr}, "mlpCos_v2");
    Tensor mlpSin_v2(kType, {b_v2, cmpBlockSize, dr}, "mlpSin_v2");
    Tensor cmpAttn(DT_FP32, {b_v2 * s1 * n1, dv}, "cmpAttnOut");
    Tensor cmpAttn16(DT_FP16, {b, s1, n1, dv}, "cmpAttnOut16");
    Tensor cmpSoftmax(DT_FP32, {b_v2 * s1 * n1, maxCmpSeq}, "cmpSoftmax");
    Tensor fullK(kType, {maxBlockNum * blockSize, n2, dq}, "fullK");
    Tensor cmpK(DT_FP32, {b_v2, maxCmpSeq, n2, dq}, "cmpK");
    Tensor firstRope(qType, {maxCmpSeq, cmpBlockSize, n2, dr}, "firstRope");
    Tensor firstRopeInput(qType, {maxCmpSeq, cmpBlockSize, dr}, "firstRopeInput");
    Tensor topkRes(DT_INT32, {b, s1, 16}, "topkRes");
    int a = (((int((s2 - 32) / 16)) + 1) + 3) / 4;
    std::cout << "xxxxxxxxxxxxxxxxxx  s2:" << s2 << ", a: " << a << std::endl;
    Tensor topkInput(DT_FP32, {b, a}, "topkInput");

    MlaQuantInputs quantInputs;

    PostTensors postTensors{wUv, wo, wUvScale, smoothWUv, woScale, smoothWo};
    // 4. 计算接口
    DynamicNsa(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, quantInputs,
        prologConfig, eps, eps, cacheMode, topkIndices, /*kvNopeCache, kRopeCache,*/ kvCacheActSeq,
        blockTable, front, near, topk, slcBlockSize, blockSize, // genKvSlc
        /*qNope, qRope, slcActSeqs,*/ softmaxScale, saTileConfig,                // slcAttn
        /*x, */ gateW1, gateW2, gateSimW1, GateMode::standard,                   // gatedscore
        cmpAtten, winSize, winAttntileConfig,                                    // gen win
        postTensors, postConfig,                                  // post
        outputKvCache, outputKrCache, postOut, cmpKvCache_v2, cmpKrCache_v2, cmpBlockTable_v2, actSeqLen_v2,
        actCmpSeqLen_v2, mlpWk1_v2, mlpWk2_v2, mlpCos_v2, mlpSin_v2, cmpAttn, cmpSoftmax, fullK, cmpK, firstRope,
        firstRopeInput, topkRes, topkInput, cmpBlockSize, cmpStride, cmpTileConfig, debug);
}

TEST_F(NSAUtest, nsa_b_16_fp16) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();

    std::vector<int> inputParams = {16, 1, 8192, 128, 1, 0, 0};

    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256}; // (n1, d)

    WinAttenTileShapeConfig winAttnTileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    winAttnTileConfig.gTile = gTileSize;
    winAttnTileConfig.vNopeTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.vRopeTileShape = {NUM_128, NUM_64};
    winAttnTileConfig.outTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (winSize, dN+dR) -> (n1, s2Tile)
    winAttnTileConfig.v1TileShape = {NUM_16, NUM_256}; // (n1, s2Tile)
    winAttnTileConfig.c2TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, winSize) @ (winSize, dN) -> (n1, d)
    winAttnTileConfig.v2TileShape = {NUM_16, NUM_256}; // (n1, d)

    PostTileConfig postConfig = {16, 1};
    MlaTileConfig prologConfig = {16, 1};


    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {64, 64};           // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {1, 64, 64};      // (1, cmpBlockSize, dk)
    config.mlpRopeTile.fourDim = {1, 64, 1, 64};    // (1, cmpBlockSize, n2, dK) * (1, cmpBlockSize, 1, dK) & RotateHalf
    config.mlpRopeTile.fiveDim = {1, 64, 1, 64, 2}; // (1, cmpBlockSize, n2, dk / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {32, 1, 192};              // (cmpBlockSize, n2, d)
    config.mlpCmpTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {1, 128};                     // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (1, n2, d)
    // CmpAttn
    config.attnTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (g, effSeq)
    config.attnTile.v1TileShape = {16, 128};                    // (g, effSeq)
    config.attnTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // (g, dN)


    std::string cacheMode = "PA_BSND";
    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true>(params, prologConfig, winAttnTileConfig, saTileConfig, postConfig, config, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false>(params, prologConfig, winAttnTileConfig, saTileConfig, postConfig, config, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false>(params, prologConfig, winAttnTileConfig, saTileConfig, postConfig, config, cacheMode);
    }
}


TEST_F(NSAUtest, nsa_b_16_fp16_debug) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    std::vector<int> inputParams = {16, 1, 8192, 128, 1, 0, 0};

    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];

    SATileShapeConfig saTileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256}; // (n1, d)
    saTileConfig.kvSlcV0TileShape = {64, 256};

    WinAttenTileShapeConfig winAttnTileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    winAttnTileConfig.gTile = gTileSize;
    winAttnTileConfig.vNopeTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.vRopeTileShape = {NUM_128, NUM_64};
    winAttnTileConfig.outTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (winSize, dN+dR) -> (n1, s2Tile)
    winAttnTileConfig.v1TileShape = {NUM_16, NUM_256}; // (n1, s2Tile)
    winAttnTileConfig.c2TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, winSize) @ (winSize, dN) -> (n1, d)
    winAttnTileConfig.v2TileShape = {NUM_16, NUM_256}; // (n1, d)

    KvSlcTileShapeConfig kvSlcTileConfig;
    kvSlcTileConfig.v0TileShape = {32, 32};

    PostTileConfig postConfig = {16, 1};
    MlaTileConfig prologConfig = {16, 1};


    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {64, 64};           // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {1, 64, 64};      // (1, cmpBlockSize, dk)
    config.mlpRopeTile.fourDim = {1, 64, 1, 64};    // (1, cmpBlockSize, n2, dK) * (1, cmpBlockSize, 1, dK) & RotateHalf
    config.mlpRopeTile.fiveDim = {1, 64, 1, 64, 2}; // (1, cmpBlockSize, n2, dk / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {32, 1, 192};              // (cmpBlockSize, n2, d)
    config.mlpCmpTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {1, 128};                     // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (1, n2, d)
    // CmpAttn
    config.attnTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (g, effSeq)
    config.attnTile.v1TileShape = {16, 128};                    // (g, effSeq)
    config.attnTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // (g, dN)


    std::string cacheMode = "PA_BSND";
    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true, false, true>(params, prologConfig, winAttnTileConfig, saTileConfig, postConfig, config, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false, false, true>(params, prologConfig, winAttnTileConfig, saTileConfig, postConfig, config, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, true>(params, prologConfig, winAttnTileConfig, saTileConfig, postConfig, config, cacheMode);
    }
}
