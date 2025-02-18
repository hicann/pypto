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
 * \file test_mla_prolog_quant_v32.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek_v3.2_exp/mla_prolog_quant_v32.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class MlaPrologQuantV32STest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct TestShapeParams {
    int b;
    int s;
    int s2;
    int n1;
    int h;
    int qLoraRank;
    int qkNopeHeadDim;
    int qkRopeHeadDim;
    int kvLoraRank;
    int blockSize;
};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = npu::tile_fwk::float16,  typename wDtype = int8_t, bool isQuantA = false, bool isQuantB = true, bool nz = true>
void TestMlaPrologQuantV32(
    const TestShapeParams &params, const MlaTileConfig &tileConfig, std::string layoutKey = "PA_NZ") {

    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n1 = params.n1;
    int n2 = 1;
    int h = params.h;
    int qLoraRank = params.qLoraRank;
    int qkNopeHeadDim = params.qkNopeHeadDim;
    int qkRopeHeadDim = params.qkRopeHeadDim;
    int kvLoraRank = params.kvLoraRank;
    int blockSize = params.blockSize;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    DataType dTypeQuantA = (std::is_same<wDtype, int8_t>::value && isQuantA) ? DT_INT8 : dType;
    DataType dTypeQuantB = (std::is_same<wDtype, int8_t>::value && isQuantB) ? DT_INT8 : dType;
    DataType dTypeKvQuant = dTypeQuantB;
    using wDtypeA = typename std::conditional<isQuantA, wDtype, T>::type;
    using wDtypeB = typename std::conditional<isQuantB, wDtype, T>::type;
    using kvDtype = typename std::conditional<isQuantB, int8_t, T>::type;

    std::vector<int64_t> tokenXShape = {b, s, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int64_t> dequantScaleWUqQrShape = {n1 * qHeadDim, 1};
    std::vector<int64_t> wDkvKrShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n1, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> ropeCosShape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> rmsnormGammaCqShape = {qLoraRank};
    std::vector<int64_t> rmsnormGammaCkvShape = {kvLoraRank};
    std::vector<int64_t> cacheIndexShape = {b, s};
    int blockNum = b * ((s2 + blockSize - 1) / blockSize);
    std::vector<int64_t> kvCacheShape = {blockNum, blockSize, n2, kvLoraRank};
    std::vector<int64_t> krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
    std::vector<int64_t> kScaleCacheShape = {blockNum, blockSize, n2, 4};
    // output
    std::vector<int64_t> kvCacheOutShape = {blockNum, blockSize, n2, kvLoraRank};
    std::vector<int64_t> krCacheOutShape = {blockNum, blockSize, n2, qkRopeHeadDim};
    std::vector<int64_t> kScaleCacheOutShape = {blockNum, blockSize, n2, 4};
    std::vector<int64_t> qNopeOutShape = {b * s, n1, kvLoraRank};
    std::vector<int64_t> qRopeOutShape = {b * s, n1, qkRopeHeadDim};
    std::vector<int64_t> qNormOutShape = {b * s, qLoraRank};
    std::vector<int64_t> qNormScaleOutShape = {b * s, 1};


    Tensor tokenX(dType, tokenXShape, "tokenX");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dTypeQuantA, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantB, wUqQrShape, "wUqQr", weightFormat);
    Tensor wDkvKr(dTypeQuantA, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", TileOpFormat::TILEOP_ND);
    Tensor rmsnormGammaCq(dType, rmsnormGammaCqShape, "rmsnormGammaCq");
    Tensor rmsnormGammaCkv(dType, rmsnormGammaCkvShape, "rmsnormGammaCkv");
    Tensor ropeCos(dType, ropeCosShape, "ropeCos");
    Tensor ropeSin(dType, ropeCosShape, "ropeSin");
    Tensor cacheIndex(DT_INT64, cacheIndexShape, "cacheIndex"); // int64
    Tensor kvCache(dTypeKvQuant, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor kScaleCache(DT_FP32, kScaleCacheShape, "kScaleCache");
    Tensor dequantScaleWUqQr(DT_FP32, dequantScaleWUqQrShape, "dequantScaleWUqQr");

    // output
    Tensor outputKvCache(dTypeKvQuant, kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheOutShape, "outputKrCache");
    Tensor outputKScaleCache(DT_FP32, kScaleCacheOutShape, "outputKScaleCache");
    Tensor outputQNope(dType, qNopeOutShape, "outputQNope");
    Tensor outputQRope(dType, qRopeOutShape, "outputQRope");
    Tensor outputQNorm(dTypeKvQuant, qNormOutShape, "outputQNorm");
    Tensor outputQNormScale(DT_FP32, qNormScaleOutShape, "outputQNormScale");

    // dynamic shape
    Tensor dynamicTokenX(dType, {-1, -1, h}, "dynamicX");
    Tensor dynamicRopeCos(dType, {-1, -1, qkRopeHeadDim}, "dynamicRopeCos");
    Tensor dynamicRopeSin(dType, {-1, -1, qkRopeHeadDim}, "dynamicRopeSin");
    Tensor dynamicCacheIndex(DT_INT64, {-1, -1}, "dynamicCacheIndex"); // int64
    Tensor dynamicOutputQNope(dType, {-1, n1, kvLoraRank}, "dynamicOutputQ");
    Tensor dynamicOutputQRope(dType, {-1, n1, qkRopeHeadDim}, "dynamicOutputQRope");
    Tensor dynamicOutputQNorm(dTypeKvQuant, {-1, qLoraRank}, "dynamicOutputQNorm");
    Tensor dynamicOutputQNormScale(DT_FP32, {-1, 1}, "dynamicOutputQNormScale");

    // output
    std::vector<T> golden1 = getGoldenVec<T>(qNopeOutShape, "/q_golden.bin");
    std::vector<T> golden2 = getGoldenVec<T>(qRopeOutShape, "/q_rope_golden.bin");
    std::vector<kvDtype> golden3 = getGoldenVec<kvDtype>(kvCacheOutShape, "/kv_cache_golden.bin");
    std::vector<T> golden4 = getGoldenVec<T>(krCacheOutShape, "/kr_cache_golden.bin");
    std::vector<float> golden5, golden7;
    if (isQuantB) {
        golden5 = getGoldenVec<float>(kScaleCacheOutShape, "/kv_quant_scale_cache_golden.bin");
    }
    auto golden6 = getGoldenVec<kvDtype>(qNormOutShape, "/rms_norm_golden.bin");
    if (isQuantB) {
        golden7 = getGoldenVec<float>(qNormScaleOutShape, "/rms_norm_scale_golden.bin");
    }

    auto tokenXData = CreateTensorData<T>(tokenX, tokenXShape, "/x.bin");
    auto wDqData = CreateTensorData<wDtypeA>(wDq, wDqShape, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtypeB>(wUqQr, wUqQrShape, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(wUk, wUkShape, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<wDtypeA>(wDkvKr, wDkvKrShape, "/wDkvKr.bin");
    auto rmsnormGammaCqData = CreateTensorData<T>(rmsnormGammaCq, rmsnormGammaCqShape, "/gamma_cq.bin");
    auto rmsnormGammaCkvData = CreateTensorData<T>(rmsnormGammaCkv, rmsnormGammaCkvShape, "/gamma_ckv.bin");
    auto ropeCosData = CreateTensorData<T>(ropeCos, ropeCosShape, "/cos.bin");
    auto ropeSinData = CreateTensorData<T>(ropeSin, ropeCosShape, "/sin.bin");
    auto cacheIndexData = CreateTensorData<int64_t>(cacheIndex, cacheIndexShape, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<kvDtype>(kvCache, kvCacheShape, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(krCache, krCacheShape, "/kr_cache.bin");
    auto outputQNopeData = RawTensorData::CreateConstantTensor<T>(outputQNope, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(outputQRope, 0.0);
    auto outputQNormData = RawTensorData::CreateConstantTensor<kvDtype>(outputQNorm, 0.0);
    auto outputQNormScaleData = RawTensorData::CreateConstantTensor<float>(outputQNormScale, 0.0);
    RawTensorDataPtr kScaleCacheData;
    if (isQuantB) {
        kScaleCacheData = CreateTensorData<float>(kScaleCache, kScaleCacheShape, "/kv_quant_scale_cache.bin");
    } else {
        kScaleCacheData = RawTensorData::CreateConstantTensor<float>(kScaleCache, 0.0);
    }
    RawTensorDataPtr dequantScaleWUqQrData;
    if (isQuantB) {
        dequantScaleWUqQrData = CreateTensorData<float>(dequantScaleWUqQr, dequantScaleWUqQrShape, "/w_qb_scale.bin");
    } else {
        dequantScaleWUqQrData = nullptr;
    }
    std::vector<RawTensorDataPtr> inputDataList = {tokenXData, wDqData, wUqQrData, dequantScaleWUqQrData, wUkData, wDkvKrData,
        rmsnormGammaCqData, rmsnormGammaCkvData, ropeCosData, ropeSinData, cacheIndexData, kvCacheData, krCacheData, kScaleCacheData};
    ProgramData::GetInstance().AppendInputs({inputDataList});

    std::vector<RawTensorDataPtr> outputDataList = {outputQNormData, outputQNormScaleData, outputQNopeData, outputQRopeData};
    ProgramData::GetInstance().AppendOutputs({outputDataList});

    std::vector<RawTensorDataPtr> goldenDataList = {
        RawTensorData::CreateTensor<kvDtype>(outputQNorm, golden6),
        RawTensorData::CreateTensor<float>(outputQNormScale, golden7),
        RawTensorData::CreateTensor<T>(outputQNope, golden1),
        RawTensorData::CreateTensor<T>(outputQRope, golden2)
    };
    ProgramData::GetInstance().AppendGoldens({goldenDataList});

    MlaPrologQuantV32(dynamicTokenX, wDq, wUqQr, dequantScaleWUqQr, wUk, wDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        dynamicRopeCos, dynamicRopeSin, dynamicCacheIndex, kvCache, krCache, kScaleCache,
        dynamicOutputQNorm, dynamicOutputQNormScale,
        dynamicOutputQNope, dynamicOutputQRope,
        outputKvCache, outputKrCache, outputKScaleCache,
        1e-5f, 1e-5f, layoutKey, tileConfig);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "qNope ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)outputQNopeData->data(), 0.008f, 1));
    std::cout << "qRope ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)outputQRopeData->data(), 0.005f, 1));
    std::cout << "kv ====== " << std::endl;
    if (isQuantB) {
        EXPECT_TRUE(resultCmpAbsDelta<kvDtype>(golden3, (kvDtype *)kvCacheData->data(), 1.0f, 1));
    } else {
        EXPECT_TRUE(resultCmp<kvDtype>(golden3, (kvDtype *)kvCacheData->data(), 0.003f, 1));
    }
    std::cout << "kr ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden4, (T *)krCacheData->data(), 0.003f, 1));
    if (isQuantB) {
        std::cout << "kScaleCache ====== " << std::endl;
        EXPECT_TRUE(resultCmp<float>(golden5, (float *)kScaleCacheData->data(), 0.003f, 1));
    }
    if (isQuantB) {
        std::cout << "qNorm ====== " << std::endl;
        EXPECT_TRUE(resultCmpAbsDelta<kvDtype>(golden6, (kvDtype *)outputQNormData->data(), 1.0, 1));
        std::cout << "qNormScale ====== " << std::endl;
        EXPECT_TRUE(resultCmp<float>(golden7, (float *)outputQNormScaleData->data(), 0.003f, 1));
    } else {
        std::cout << "qNorm ====== " << std::endl;
        EXPECT_TRUE(resultCmp<kvDtype>(golden6, (kvDtype *)outputQNormData->data(), 0.003f, 1));
    }
}

////// fp16, quant, weight nd, "PA_BSND"
TEST_F(MlaPrologQuantV32STest, b1_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {1, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 1;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b4_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {4, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 4;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b8_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {8, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 4;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b16_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b64_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b128_s64k2_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {128, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// allquant test
TEST_F(MlaPrologQuantV32STest, b32_s64k1_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s64k4_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// allquant test
TEST_F(MlaPrologQuantV32STest, b32_s1k4_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 1 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s4k4_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 4 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s16k4_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 16 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s128k4_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 128 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// small shape
TEST_F(MlaPrologQuantV32STest, b1_s11_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {1, 1, 1, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 1;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b1_s129_1_pa_nd_fp16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {1, 1, 129, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 1;

    TestMlaPrologQuantV32<npu::tile_fwk::float16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

////// bf16, quant, weight nd, "PA_BSND"
TEST_F(MlaPrologQuantV32STest, b1_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {1, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 1;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b4_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {4, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 4;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b8_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {8, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 4;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b16_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b64_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b128_s64k2_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {128, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// allquant test
TEST_F(MlaPrologQuantV32STest, b32_s64k1_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s64k4_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// allquant test
TEST_F(MlaPrologQuantV32STest, b32_s1k4_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 1 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s4k4_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 4 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s16k4_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 16 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b32_s128k4_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 4, 128 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 16;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// small shape
TEST_F(MlaPrologQuantV32STest, b1_s11_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {1, 1, 1, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 1;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

TEST_F(MlaPrologQuantV32STest, b1_s129_1_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {1, 1, 129, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 1;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// unaligned shape
TEST_F(MlaPrologQuantV32STest, b104_s8k1_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {104, 1, 8 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 13;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}

// special case from test
TEST_F(MlaPrologQuantV32STest, b32_s127104_3_pa_nd_bf16_quantB) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 3, 127104, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig;
    tileConfig.tileBS = 4;

    TestMlaPrologQuantV32<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}
} // namespace
