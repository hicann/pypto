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
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/mla_prolog_quant_v32.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class MlaPrologQuantV32UTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

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

template <typename T = npu::tile_fwk::float16,  typename wDtype = int8_t, bool isQuantA = false, bool isQuantB = true, bool nz = true>
void TestMlaPrologQuantV32Ut(const TestShapeParams &params, const MlaTileConfig &tileConfig, std::string layoutKey = "PA_NZ") {
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
    int blockNum = b * (s2 / blockSize);
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

    // dynamic shape
    Tensor dynamicTokenX(dType, {-1, -1, h}, "dynamicX");
    Tensor dynamicRopeCos(dType, {-1, -1, qkRopeHeadDim}, "dynamicRopeCos");
    Tensor dynamicRopeSin(dType, {-1, -1, qkRopeHeadDim}, "dynamicRopeSin");
    Tensor dynamicCacheIndex(DT_INT64, {-1, -1}, "dynamicCacheIndex"); // int64
    Tensor dynamicOutputQNope(dType, {-1, n1, kvLoraRank}, "dynamicOutputQ");
    Tensor dynamicOutputQRope(dType, {-1, n1, qkRopeHeadDim}, "dynamicOutputQRope");
    Tensor dynamicOutputQNorm(dTypeKvQuant, {-1, qLoraRank}, "dynamicOutputQNorm");
    Tensor dynamicOutputQNormScale(DT_FP32, {-1, 1}, "dynamicOutputQNormScale");

    MlaPrologQuantV32(dynamicTokenX, wDq, wUqQr, dequantScaleWUqQr, wUk, wDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        dynamicRopeCos, dynamicRopeSin, dynamicCacheIndex, kvCache, krCache, kScaleCache,
        dynamicOutputQNorm, dynamicOutputQNormScale,
        dynamicOutputQNope, dynamicOutputQRope,
        outputKvCache, outputKrCache, outputKScaleCache,
        1e-5f, 1e-5f, layoutKey, tileConfig);
}

TEST_F(MlaPrologQuantV32UTest, b8_s64k2_pa_nd_bf16_quantB) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {8, 2, 64 * 1024, 128, 7168, 1536, 128, 64, 512, 128};
    std::string layoutKey = "PA_BSND";
    MlaTileConfig tileConfig = {4, 1};

    TestMlaPrologQuantV32Ut<npu::tile_fwk::bfloat16, int8_t, false, true, false>(params, tileConfig, layoutKey);
}