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
 * \file test_nsa_kv_selected_attention.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/nsa_selected_attention.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class KvSlcAttnUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        // skip pass, ut only execute model op code
        config::SetPassDefaultConfig(KEY_DISABLE_PASS, true);
    }

    void TearDown() override {}
};

template <typename T = npu::tile_fwk::float16>
void TestKvSlcAttn(const NSAV1SimpleParams &params, SATileShapeConfig& saTileConfig) {
    int b = params.b;
    int s1 = params.s1;
    int s2 = params.s2;
    int n1 = params.n1;
    int n2 = params.n2;
    int v_dim = params.kv_lora_rank;
    int dn = v_dim;
    int dr = params.rope_dim;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int blockSize = params.blockSize;
    int cmpBlockSize = params.cmpBlockSize;
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;
    int smax = params.topk * params.slcBlockSize;

    std::vector<int> kvCacheActSeqVec(b, s2);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    // 1. 设置shape
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

    std::vector<int64_t> shape_selAtten = {b, s1, n1, v_dim};

    // 2. 构造tensor
    Tensor topkIndices(DT_INT32, topkIndicesShape, "topkTensor");
    Tensor topkTensorShape(DT_INT32, topkTensorShapeShape, "topkTensorShape");
    Tensor kvNopeCache(dType, kvNopeCacheShape, "kNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "vNopeCache");
    Tensor kvCacheActSeq(DT_INT32, kvCacheActSeqShape, "kvCacheActSeq");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor slcActSeqs(DT_INT32, slcActSeqsShape, "slcActSeqs");

    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");

    Tensor attenOut(DT_FP32, shape_selAtten, "attenOut");


    SelectedAttention(topkIndices, kvNopeCache, kRopeCache, kvCacheActSeq, blockTable,
        qNope, qRope, attenOut,
        n1, n2, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize,
        saTileConfig);
}

TEST_F(KvSlcAttnUtest, kv_slc_attn_ut) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();

    std::vector<int> inputParams = {16, 1, 8192, 128, 1, 0, 0};

    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128}; // (n1, d)

    TestKvSlcAttn<npu::tile_fwk::float16>(params, saTileConfig);
}
