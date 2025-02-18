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
 * \file mla_prolog_utest.cpp
 * \brief
 */
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/dynamic_mla.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class MlaPrologUTest : public testing::Test {
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
    int n;
    int h;
    int qLoraRank;
    int qkNopeHeadDim;
    int qkRopeHeadDim;
    int kvLoraRank;
    int blockSize;
};

template <typename T = npu::tile_fwk::float16,  typename wDtype = int8_t, bool isQuantA = false, bool isQuantB = true,
    bool isSmooth = true, bool nz = true, bool usePrefetch = true>
void TestMlaPrologUt(const TestShapeParams &params, const MlaTileConfig &tileConfig, std::string cacheMode = "PA_NZ") {
    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
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

    std::vector<int64_t> xShape = {b, s, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cosShape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {kvLoraRank};
    std::vector<int64_t> kvLenShape = {b, s};
    std::vector<int64_t> kvCacheShape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> krCacheShape = {b, 1, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        int blockNum = b * (s2 / blockSize);
        kvCacheShape = {blockNum, blockSize, 1, kvLoraRank};
        krCacheShape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }
    std::vector<int64_t> scaleWDqShape = {1, qLoraRank};
    std::vector<int64_t> scaleWUqQrShape = {1, n * qHeadDim};
    std::vector<int64_t> scaleWDkvKrShape = {1, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> smoothCqShape{1, qLoraRank};
    // output
    std::vector<int64_t> qOutShape = {b, s, n, kvLoraRank};
    std::vector<int64_t> qRopeOutShape = {b, s, n, qkRopeHeadDim};

    Tensor x(dType, xShape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dTypeQuantA, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantB, wUqQrShape, "wUqQr", weightFormat);
    Tensor wDkvKr(dTypeQuantA, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", weightFormat);
    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");
    Tensor cos(dType, cosShape, "cos");
    Tensor sin(dType, cosShape, "sin");
    Tensor cacheIndex(DT_INT64, kvLenShape, "cacheIndex"); // int64
    Tensor kvCache(dType, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor scaleWDq(DT_FP32, scaleWDqShape, "scaleWDq");
    Tensor scaleWUqQr(DT_FP32, scaleWUqQrShape, "scaleWUqQr");
    Tensor scaleWDkvKr(DT_FP32, scaleWDkvKrShape, "scaleWDkvKr");
    Tensor smoothCq(DT_FP32, smoothCqShape, "smoothCq");

    // output
    Tensor outputKvCache(dType, kvCacheShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheShape, "outputKrCache");
    Tensor outputQ(dType, qOutShape, "outputQ");
    Tensor outputQRope(dType, qRopeOutShape, "outputQRope");

    MlaQuantInputs quantInputs;
    if (isQuantA) {
        quantInputs.dequantScaleWDq = scaleWDq;
        quantInputs.dequantScaleWDkvKr = scaleWDkvKr;
    }
    if (isQuantB) {
        quantInputs.dequantScaleWUqQr = scaleWUqQr;
        if (isSmooth) {
            quantInputs.smoothScalesCq = smoothCq;
        }
    }

    MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, quantInputs,
        tileConfig, outputQ, outputQRope, outputKvCache, outputKrCache, 1e-5f, 1e-5f, cacheMode);
}

TEST_F(MlaPrologUTest, b16_s1_pa_nd_fp16_allquant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 1};

    TestMlaPrologUt<npu::tile_fwk::float16, int8_t, true, true, true, true, true>(params, tileConfig, cacheMode);
}