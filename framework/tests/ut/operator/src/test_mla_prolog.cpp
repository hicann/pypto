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
 * \file test_mla_prolog.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/mla_prolog.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class MlaPrologUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};


template <typename T = npu::tile_fwk::float16, bool splitReduceLastDim = true, bool splitK = false, bool nz = false, bool usePrefetch = false>
void TestMlaPrologV2(std::vector<int> &params, bool isQuant = false, bool hasSmooth = false,
        int blockSize = 128, std::string cacheMode = "BNSD") {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank
    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DataType::DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DataType::DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DataType::DT_BF16;
    } else {
        dType = DataType::DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DataType::DT_INT8 : dType;

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
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    if (cacheMode == "PA_BSND") {
        int blockNum = b * (s2 / blockSize);
        kv_cache_shape = {blockNum, blockSize, 1, kvLoraRank};
        kr_cache_shape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }


    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        Tensor x = Tensor(dType, x_shape, "x"); //32_1_7168
        TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor wDq = Tensor(dType, w_qa_shape, "wDq", weightFormat);
        Tensor wUqQr = Tensor(dTypeQuantIn, w_qb_shape, "wUqQr", weightFormat);
        if constexpr (usePrefetch) {
            wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
            wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
        }
        Tensor wDkvKr = Tensor(dType, w_kv_a_shape, "wDkvKr", weightFormat);
        Tensor wUk = Tensor(dType, w_kv_b_k_shape, "wUk", weightFormat);
        Tensor gammaCq = Tensor(dType, gamma_cq_shape, "gammaCq");
        Tensor gammaCkv = Tensor(dType, gamma_ckv_shape, "gammaCkv");
        Tensor cos = Tensor(dType, cos_shape, "cos");
        Tensor sin = Tensor(dType, cos_shape, "sin");
        Tensor kv_len = Tensor(DT_INT64, kv_len_shape, "kv_len");  // int64
        Tensor kv_cache = Tensor(dType, kv_cache_shape, "kv_cache");
        Tensor kr_cache = Tensor(dType, kr_cache_shape, "kr_cache");
        // output
        Tensor output_q = Tensor(dType, q_out_shape, "output_q");
        Tensor output_q_rope = Tensor(dType, q_rope_out_shape, "output_q_rope");


        RoPETileShapeConfigNew ropeConfig{
            {b, 1, 64}, // (b,s,d)
            {b, 1, 1, 64}, // Q (b,s,n,d)
            {b, 1, 1, 64}, // K (b,s,1,d)
            {b, 1, 1, 32, 2} // (b,s,n,d//2,2)
        };

        MlaQuantInputs quantInputs;

        if (isQuant) {
            std::vector<int64_t> w_qb_scale_shape = {1, n * q_head_dim};
            std::vector<int64_t> smooth_cq_shape = {1, qLoraRank};
            Tensor w_qb_scale = Tensor(DataType::DT_FP32, w_qb_scale_shape, "w_qb_scale");
            quantInputs.dequantScaleWUqQr = w_qb_scale;
            Tensor smooth_cq = Tensor(DT_FP32, smooth_cq_shape, "smooth_cq");
            if (hasSmooth) {
                quantInputs.smoothScalesCq = smooth_cq;
                smooth_cq.SetCachePolicy(CachePolicy::PREFETCH, true);
            }
            config::SetBuildStatic(true);
            FUNCTION("MlaPrologUt",
                {x, wDq, wUqQr, w_qb_scale, smooth_cq, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos,
                 kv_len, kv_cache, kr_cache, output_q, output_q_rope}) {
                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f, cacheMode,
                    splitReduceLastDim,  splitK);
            };
        } else {
            config::SetBuildStatic(true);
            FUNCTION("MlaPrologUt",
                {x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos,
                 kv_len, kv_cache, kr_cache, output_q, output_q_rope}) {
                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f, cacheMode,
                    splitReduceLastDim,  splitK);
            };
        }
    }
}

TEST_F(MlaPrologUtest, mla_ut_bf16_high_quant_smooth_nz_pa_bsnd) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{{-1, 2}});
    int b = 32;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank};
    TestMlaPrologV2<npu::tile_fwk::bfloat16, splitReduceLastDim, splitK, nz, usePrefetch>(params, true, true,
                                                                                 blockSize, cacheMode);
}
