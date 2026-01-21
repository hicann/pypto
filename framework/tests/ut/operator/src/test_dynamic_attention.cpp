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
 * \file test_dynamic_attention.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "operator/models/deepseek/attention.h"
#include "interface/tensor/float.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class DynamicAttentionUtTest : public testing::Test {
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

template <typename T = npu::tile_fwk::float16, bool splitReduceLastDim = false, bool splitK = false, bool nz= false,
    bool usePrefetch = false>
void TestDynamicAttention(std::vector<int64_t> &params, PaTileShapeConfig &paTileConfig,
     bool isQuant = false, std::string cacheMode = "BNSD") {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];

    int vHeadDim = params[9];
    int blockSize =params[10];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    std::vector<int> atcSeqs(b, s2);

    int blockNum = 0;
    for (auto seq : atcSeqs) {
        blockNum += CeilDiv(seq, blockSize);
    }

    float softmaxScale = static_cast<float>(1.0 / sqrtf((kvLoraRank + qkRopeHeadDim)));
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(atcSeqs.begin(), atcSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> w_qa_shape = {h, qLoraRank};
    std::vector<int64_t> w_qb_shape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> w_kv_b_k_shape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cos_shape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {kvLoraRank};
    std::vector<int64_t> kv_len_shape = {b, s};
    std::vector<int64_t> kv_cache_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_shape = {b, 1, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        kv_cache_shape = {blockNum, blockSize, 1, kvLoraRank};
        kr_cache_shape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }
    // pa
    std::vector<int64_t> blockTableShape = {b, 1, s2, qkRopeHeadDim};
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};
    std::vector<int64_t> fake_out_shape = {b, s, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> fake_out_shape1 = {n, b * s, qkNopeHeadDim};

    std::vector<int64_t> w_qb_scale_shape;
    if (isQuant) {
        w_qb_scale_shape = {1, n * q_head_dim};
    }

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    TileOpFormat paFormat = cacheMode == "PA_NZ" ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    //mla_prolog
    Tensor x(dType, x_shape, "x");
    Tensor wDq(dType, w_qa_shape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantIn, w_qb_shape, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dType, w_kv_a_shape, "wDkvKr", weightFormat);
    Tensor wUk(dType, w_kv_b_k_shape, "wUk", weightFormat);
    Tensor gamma_cq(dType, gamma_cq_shape, "gamma_cq");
    Tensor gamma_ckv(dType, gamma_ckv_shape, "gamma_ckv");
    Tensor cos(dType, cos_shape, "cos");
    Tensor sin(dType, cos_shape, "sin");
    Tensor kv_len(DT_INT64, kv_len_shape, "kv_len"); // int64
    Tensor kv_cache(dType, kv_cache_shape, "kv_cache", paFormat);
    Tensor kr_cache(dType, kr_cache_shape, "kr_cache", paFormat);

    Tensor output_q(dType, {b*s*n, kvLoraRank}, "output_q");
    Tensor output_q_rope(dType, {b*s*n, qkRopeHeadDim}, "output_q_rope");
    Tensor output_kv_cache(dType, {b * 1 * s2, kvLoraRank}, "output_kv_cache", paFormat);
    Tensor output_kr_cache(dType, {b * 1 * s2, qkRopeHeadDim}, "output_kr_cache", paFormat);

    Tensor fakeOut(dType, {b * s, n, qkNopeHeadDim}, "fakeOut");
    Tensor fakeOut1(dType, {n, b * s, qkNopeHeadDim}, "fakeOut1");
    // pa
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    //out mla
    Tensor paOut(DT_FP32, {b * n * s, kvLoraRank}, "paOut");
    //post
    Tensor weightUV(dType, {n, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {n * vHeadDim, h}, "weightO");
    Tensor weightOScaleW(DT_FP32, {1, h}, "weightOScaleW");
    // output
    Tensor postOut(dType, {b, s, h}, "postOut");

    int tileB = b;
    RoPETileShapeConfigNew ropeConfig {
        {tileB, 1, 64}, // (b,s,d)
        {tileB, 1, 1, 64}, // Q (b,s,n,d)
        {tileB, 1, 1, 64}, // K (b,s,1,d)
        {tileB, 1, 1, 32, 2} // (b,s,n,d//2,2)
    };

    MlaQuantInputs quantInputs;
    Attention(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                output_q, output_q_rope, output_kv_cache, output_kr_cache, quantInputs, ropeConfig, /*---*/
                blockTable, actSeqs, paOut, blockSize, softmaxScale, paTileConfig, /*---*/
                weightUV, weightO, weightOScaleW, postOut, 1e-5f, 1e-5f, cacheMode);
}

TEST_F(DynamicAttentionUtTest, dynamic_attention_low_nz) { // b_n_s_s2_h_q_lora_rank
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;   //7168
    int n = 32;
    int qLoraRank = 1536;   //1536
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 256;
    std::vector<int64_t> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = false;

    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {nTile, 64};

    TestDynamicAttention<npu::tile_fwk::float16, splitReduceLastDim, splitK, nz>(params, tileConfig, false, "PA_NZ");
}

TEST_F(DynamicAttentionUtTest, dynamic_attention_low) { // b_n_s_s2_h_q_lora_rank
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;   //7168
    int n = 32;
    int qLoraRank = 1536;   //1536
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 256;
    std::vector<int64_t> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = false;

    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {nTile, 64};

    TestDynamicAttention<npu::tile_fwk::float16, splitReduceLastDim, splitK, nz>(params, tileConfig, false);
}