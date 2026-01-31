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
 * \file test_dynamic_mla.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/dynamic_mla.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DyMla : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

void pre() {

}

void performanceConfig() {
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 4}});
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{3, 4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2 * 1024 * 1024);
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = true,
    bool isSmooth = true, bool usePrefetch = true>
void TestMlaPrologV2(const SimpleParams &params) {
    SetInterpreterConfig();
    pre();

    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
    int h = params.h;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int kvLoraRank = params.kv_lora_rank;
    int q_head_dim = params.q_head_dim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> wDkvKrShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cos_shape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {kvLoraRank};
    std::vector<int64_t> kv_len_shape = {b, s};
    std::vector<int64_t> kv_cache_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_shape = {b, 1, s2, qkRopeHeadDim};
    if (params.cacheMode != "BNSD") {
        int blockNum = b * (s2 / params.blockSize);
        kv_cache_shape = {blockNum, params.blockSize, 1, kvLoraRank};
        kr_cache_shape = {blockNum, params.blockSize, 1, qkRopeHeadDim};
    }
    std::vector<int64_t> w_qb_scale_shape = {1, n * q_head_dim};
    std::vector<int64_t> smooth_cq_shape{1, qLoraRank};
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};

    Tensor x(dType, x_shape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", weightFormat);
    Tensor gamma_cq(dType, gamma_cq_shape, "gamma_cq");
    Tensor gamma_ckv(dType, gamma_ckv_shape, "gamma_ckv");
    Tensor cos(dType, cos_shape, "cos");
    Tensor sin(dType, cos_shape, "sin");
    Tensor kv_len(DT_INT64, kv_len_shape, "kv_len"); // int64
    Tensor kv_cache(dType, kv_cache_shape, "kv_cache");
    Tensor kr_cache(dType, kr_cache_shape, "kr_cache");
    Tensor w_qb_scale(DT_FP32, w_qb_scale_shape, "w_qb_scale");
    Tensor smooth_cq(DT_FP32, smooth_cq_shape, "smooth_cq");

    // output
    Tensor output_kv_cache(dType, kv_cache_shape, "output_kv_cache");
    Tensor output_kr_cache(dType, kr_cache_shape, "output_kr_cache");
    Tensor output_q(dType, q_out_shape, "output_q");
    Tensor output_q_rope(dType, q_rope_out_shape, "output_q_rope");

    RoPETileShapeConfigNew ropeConfig{
        {b, 1, 64}, // (b,s,d)
        {b, 1, 1, 64}, // Q (b,s,n,d)
        {b, 1, 1, 64}, // K (b,s,1,d)
        {b, 1, 1, 32, 2}  // (b,s,n,d//2,2)
    };

    MlaQuantInputs quantInputs;
    // // output
    std::vector<T> golden1 = getGoldenVec<T>(q_out_shape, "/q_golden.bin");
    std::vector<T> golden2 = getGoldenVec<T>(q_rope_out_shape, "/q_rope_golden.bin");
    std::vector<T> golden3 = getGoldenVec<T>(kv_cache_shape, "/kv_cache_golden.bin");
    std::vector<T> golden31 = getGoldenVec<T>(kv_cache_shape, "/kv_cache.bin");
    std::vector<T> golden4 = getGoldenVec<T>(kr_cache_shape, "/kr_cache_golden.bin");

    auto xData = CreateTensorData<T>(x, x_shape, "/x.bin");
    auto wDqData = CreateTensorData<T>(wDq, wDqShape, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtype>(wUqQr, wUqQrShape, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(wUk, wUkShape, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<T>(wDkvKr, wDkvKrShape, "/wDkvKr.bin");
    auto gammaCqData = CreateTensorData<T>(gamma_cq, gamma_cq_shape, "/gamma_cq.bin");
    auto gammaCkvData = CreateTensorData<T>(gamma_ckv, gamma_ckv_shape, "/gamma_ckv.bin");
    auto cosData = CreateTensorData<T>(cos, cos_shape, "/cos.bin");
    auto sinData = CreateTensorData<T>(sin, cos_shape, "/sin.bin");
    auto kvLenData = CreateTensorData<int64_t>(kv_len, kv_len_shape, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<T>(kv_cache, kv_cache_shape, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(kr_cache, kr_cache_shape, "/kr_cache.bin");
    auto wQbScaleData = CreateTensorData<float>(w_qb_scale, w_qb_scale_shape, "/w_qb_scale.bin");
    auto smoothCqData = CreateTensorData<float>(smooth_cq, smooth_cq_shape, "/smooth_cq.bin");
    auto outputQData = RawTensorData::CreateConstantTensor<T>(output_q, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(output_q_rope, 0.0);

    auto golden1Data = CreateTensorData<T>(output_q, q_out_shape, "/q_golden.bin");
    auto golden2Data = CreateTensorData<T>(output_q_rope, q_rope_out_shape, "/q_rope_golden.bin");
    auto golden3Data = CreateTensorData<T>(kv_cache, kv_cache_shape, "/kv_cache_golden.bin");
    auto golden4Data = CreateTensorData<T>(kr_cache, kr_cache_shape, "/kr_cache_golden.bin");

    ProgramData::GetInstance().PrepareData(
        {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData, sinData, cosData, kvLenData,
            kvCacheData, krCacheData, wQbScaleData, smoothCqData},
        {outputQData, outputQRopeData, kvCacheData, krCacheData}, {golden1Data, golden2Data, golden3Data, golden4Data});
    if (isQuant) {
        quantInputs.dequantScaleWUqQr = w_qb_scale;
        if (isSmooth) {
            quantInputs.smoothScalesCq = smooth_cq;
        }
    }
    config::SetPassConfig("PVC2_OOO", "InferMemoryConflict", KEY_DISABLE_PASS, true);
    MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache, quantInputs,
        ropeConfig, output_q, output_q_rope, output_kv_cache, output_kr_cache, 1e-5f, 1e-5f, params.cacheMode, splitK,
        isSmooth);
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
        {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData, sinData, cosData, kvLenData,
            kvCacheData, krCacheData, wQbScaleData, smoothCqData},
        {outputQData, outputQRopeData, kvCacheData, krCacheData});
    std::cout << "qNope ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)outputQData->data(), 0.008f, 16));
    std::cout << "qRope ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)outputQRopeData->data(), 0.005f, 16));
    std::cout << "kv ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden3, (T *)kvCacheData->data(), 0.003f, 16));
    std::cout << "kr ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden4, (T *)krCacheData->data(), 0.003f, 16));
#endif
}

TEST_F(DyMla, low) {
    // verifyConfig();
    performanceConfig();
    TestMlaPrologV2<npu::tile_fwk::float16>(SimpleParams::getLowParams());
}

TEST_F(DyMla, low_PA_BSND) {
    // verifyConfig();
    performanceConfig();
    npu::tile_fwk::SimpleParams params = SimpleParams::getLowParams();
    params.cacheMode = "PA_BSND";
    TestMlaPrologV2<npu::tile_fwk::float16>(params);
}

TEST_F(DyMla, low_PA_NZ) {
    // verifyConfig();
    performanceConfig();
    npu::tile_fwk::SimpleParams params = SimpleParams::getLowParams();
    params.cacheMode = "PA_NZ";
    TestMlaPrologV2<npu::tile_fwk::float16>(params);
}

TEST_F(DyMla, low_bf) {
    performanceConfig();
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(SimpleParams::getLowParams());
}

TEST_F(DyMla, high) {
    performanceConfig();
    TestMlaPrologV2<npu::tile_fwk::float16>(SimpleParams::getHighParams());
}

TEST_F(DyMla, high_PA_NZ) {
    performanceConfig();
    npu::tile_fwk::SimpleParams params = SimpleParams::getHighParams();
    params.cacheMode = "PA_NZ";
    TestMlaPrologV2<npu::tile_fwk::float16>(params);
}

} // namespace
