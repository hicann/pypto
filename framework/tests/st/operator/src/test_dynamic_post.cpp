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
 * \file test_dynamic_post.cpp
 * \brief
 */

#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"
#include "test_dev_func_runner.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "operator/models/nsa/attention_post.h"
#include "tilefwk/tensor.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class AttentionPostSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct TestPostParams {
    int b;
    int n;
    int s;
    int h;
    int kvLoraRank;
    int vHeadDim;
};

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = npu::tile_fwk::float16, bool nz = true, typename wUvDType = int8_t, bool isSmoothWUv = false,
    typename wODType = int8_t, bool isSmoothWo = false>
void TestAttentionPost(const TestPostParams &params, const PostTileConfig &tileConfig, float precision) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    int b = params.b;
    int n = params.n;
    int s = params.s;
    int h = params.h;
    int kvLoraRank = params.kvLoraRank;
    int vHeadDim = params.vHeadDim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuantWUv = std::is_same<wUvDType, int8_t>::value;
    bool isQuantWo = std::is_same<wODType, int8_t>::value;

    std::vector<int64_t> xShape = {b, s, n, kvLoraRank};
    std::vector<int64_t> wUvShape = {n, kvLoraRank, vHeadDim};
    std::vector<int64_t> wUvScaleShape = {n, 1, vHeadDim};
    std::vector<int64_t> smoothWUvShape = {1, kvLoraRank};
    std::vector<int64_t> woShape = {n * vHeadDim, h};
    std::vector<int64_t> woScaleShape = {1, h};
    std::vector<int64_t> smoothWoShape = {1, n * vHeadDim};
    std::vector<int64_t> outShape = {b, s, h};

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor x(dType, xShape, "x");
    Tensor wUv(isQuantWUv ? DT_INT8 : dType, wUvShape, "wUv");
    Tensor wo(isQuantWo ? DT_INT8 : dType, woShape, "wo", weightFormat);
    Tensor postOut(dType, outShape, "postOut");

    std::vector<T> goldenDate = getGoldenVec<T>(outShape, "/golden_output.bin");

    auto xData = CreateTensorData<T>(x, "/x.bin");
    auto wUvData = CreateTensorData<wUvDType>(wUv, "/w_uv.bin");
    auto woData = CreateTensorData<wODType>(wo, "/w_o.bin");
    auto outputData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

    std::vector<RawTensorDataPtr> outputDataList = {outputData};
    std::vector<RawTensorDataPtr> inputDataList = {xData, wUvData, woData};

    QuantTensorWithData wUvQuant{isQuantWUv, isSmoothWUv, wUvScaleShape, smoothWUvShape, "wUvScale", "smoothWUv",
        "/w_uv_scale.bin", "/smooth_w_uv.bin"};
    CreateQuantTensorAndData(wUvQuant);
    inputDataList.emplace_back(wUvQuant.scale.dataPtr);
    inputDataList.emplace_back(wUvQuant.smooth.dataPtr);

    QuantTensorWithData wOQuant{
        isQuantWo, isSmoothWo, woScaleShape, smoothWoShape, "woScale", "smoothWo", "/w_o_scale.bin", "/smooth_w_o.bin"};
    CreateQuantTensorAndData(wOQuant);
    inputDataList.emplace_back(wOQuant.scale.dataPtr);
    inputDataList.emplace_back(wOQuant.smooth.dataPtr);

    ProgramData::GetInstance().AppendInputs({inputDataList});

    ProgramData::GetInstance().AppendOutputs({outputDataList});

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(postOut, goldenDate),
    });

    PostTensors postTensors{
        wUv, wo, wUvQuant.scale.tensor, wUvQuant.smooth.tensor, wOQuant.scale.tensor, wOQuant.smooth.tensor};
    AttentionPostStandalone(x, postTensors, tileConfig, postOut);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);

    std::cout << "postOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(goldenDate, (T *)outputData->data(), precision));
    resultCmp<T>(goldenDate, (T *)outputData->data(), precision, int(b * s * h * precision), 1000, false, false, 16);
#endif
}

void PerformanceConfig() {
    const int pg_upper_bound = 500000;
    config::SetPassOption(SG_PG_UPPER_BOUND, pg_upper_bound);
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
}

////// fp16, nz, quant
TEST_F(AttentionPostSTest, b16_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {64, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// bf16, nz, quant
TEST_F(AttentionPostSTest, b16_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {64, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// fp16, nd, quant
TEST_F(AttentionPostSTest, b16_s1_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// fp16, nz, no quant
TEST_F(AttentionPostSTest, b32_s1_nz_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

////// fp16, nd, no quant
TEST_F(AttentionPostSTest, b16_s1_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b16_s2_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s1_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s2_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b16_s1_nz_fp16_quant_all) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_bf16_quant_all) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_fp16_quant_all) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true, int8_t, true>(params, tileConfig, 0.006f);
}

} // namespace
