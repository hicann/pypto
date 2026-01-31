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
 * \file test_quant_lightning_indexer_prolog.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
#include "interface/inner/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "operator/models/deepseek_v3.2_exp/quant_lightning_indexer_prolog.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class QuantLightningIndexerPrologSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct QuantIndexerPrologInputData {
    RawTensorDataPtr xData;
    RawTensorDataPtr qNormData;
    RawTensorDataPtr qNormScaleData;
    RawTensorDataPtr wQbData;
    RawTensorDataPtr wQbScaleData;
    RawTensorDataPtr wkData;
    RawTensorDataPtr wProjData;
    RawTensorDataPtr lnGammaKData;
    RawTensorDataPtr lnBetaKData;
    RawTensorDataPtr cosIdxRopeData;
    RawTensorDataPtr sinIdxRopeData;
    RawTensorDataPtr hadamardQData;
    RawTensorDataPtr hadamardKData;
    RawTensorDataPtr kCacheData;
    RawTensorDataPtr kCacheScaleData;
    RawTensorDataPtr kCacheIndexData;
};

struct QuantIndexerPrologOutputData {
    RawTensorDataPtr qInt8Data;
    RawTensorDataPtr qScaleData;
    RawTensorDataPtr kInt8Data;
    RawTensorDataPtr kScaleData;
    RawTensorDataPtr weightsData;
};

struct QuantIndexerPrologOutputGolden {
    std::vector<int8_t> qInt8Golden;
    std::vector<npu::tile_fwk::float16> qScaleGolden;
    std::vector<int8_t> kInt8Golden;
    std::vector<npu::tile_fwk::float16> kScaleGolden;
    std::vector<npu::tile_fwk::float16> weightsGolden;
};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
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

template <typename T, bool nz>
QuantIndexerPrologInputData PrepareQuantIndexerPrologInputsData(const QuantIndexerPrologInput &inputs) {
    auto xData = CreateTensorData<T>(inputs.x, "/token_x.bin");
    auto qNormData = CreateTensorData<int8_t>(inputs.qNorm, "/q_norm.bin");
    auto qNormScaleData = CreateTensorData<float>(inputs.qNormScale, "/q_norm_scale.bin");
    auto wQbData = CreateTensorData<int8_t>(inputs.wQb, nz ? "/w_idx_qb_nz.bin" : "/w_idx_qb.bin"); // nz
    auto wQbScaleData = CreateTensorData<float>(inputs.wQbScale, "/w_idx_qb_scale.bin");
    auto wkData = CreateTensorData<T>(inputs.wk, nz ? "/w_idx_k_nz.bin" : "/w_idx_k.bin");               // nz
    auto wProjData = CreateTensorData<T>(inputs.wProj, nz ? "/weights_proj_nz.bin" : "/w_idx_proj.bin"); // nz
    auto lnGammaKData = CreateTensorData<T>(inputs.lnGammaK, "/layer_norm_gamma.bin");
    auto lnBetaKData = CreateTensorData<T>(inputs.lnBetaK, "/layer_norm_beta.bin");
    auto cosIdxRopeData = CreateTensorData<T>(inputs.cosIdxRope, "/cos_idx_rope.bin");
    auto sinIdxRopeData = CreateTensorData<T>(inputs.sinIdxRope, "/sin_idx_rope.bin");
    auto hadamardQData = CreateTensorData<T>(inputs.hadamardQ, "/hadamard_q.bin");
    auto hadamardKData = CreateTensorData<T>(inputs.hadamardK, "/hadamard_k.bin");
    auto kCacheData = CreateTensorData<int8_t>(inputs.kCache, "/idx_k_cache.bin");
    auto kCacheScaleData = CreateTensorData<npu::tile_fwk::float16>(inputs.kCacheScale, "/idx_k_scale_cache.bin");
    auto kCacheIndexData = CreateTensorData<int64_t>(inputs.kCacheIndex, "/idx_k_cache_index.bin");

    return QuantIndexerPrologInputData{xData, qNormData, qNormScaleData, wQbData, wQbScaleData, wkData, wProjData,
        lnGammaKData, lnBetaKData, cosIdxRopeData, sinIdxRopeData, hadamardQData, hadamardKData, kCacheData,
        kCacheScaleData, kCacheIndexData};
}

QuantIndexerPrologOutputData PrepareQuantIndexerPrologOutputsData(const QuantIndexerPrologOutput &outputs) {
    auto qInt8Data = RawTensorData::CreateConstantTensor<int8_t>(outputs.qInt8, 0);
    auto qScaleData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(outputs.qScale, 0.0f);
    auto kInt8Data = RawTensorData::CreateConstantTensor<int8_t>(outputs.kInt8, 0);
    auto kScaleData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(outputs.kScale, 0.0f);
    auto weightsData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(outputs.weights, 0.0f);
    return QuantIndexerPrologOutputData{qInt8Data, qScaleData, kInt8Data, kScaleData, weightsData};
}

QuantIndexerPrologOutputGolden PrepareQuantIndexerPrologOutputsGolden(const QuantIndexerPrologOutput &outputs) {
    auto qInt8Golden = getGoldenVec<int8_t>(outputs.qInt8.GetShape(), "/query_golden.bin");
    auto qScaleGolden = getGoldenVec<npu::tile_fwk::float16>(outputs.qScale.GetShape(), "/query_scale_golden.bin");
    auto kInt8Golden = getGoldenVec<int8_t>(outputs.kInt8.GetShape(), "/idx_k_cache_out_golden.bin");
    auto kScaleGolden =
        getGoldenVec<npu::tile_fwk::float16>(outputs.kScale.GetShape(), "/idx_k_scale_cache_out_golden.bin");
    auto weightsGolden = getGoldenVec<npu::tile_fwk::float16>(outputs.weights.GetShape(), "/weights_golden.bin");
    return QuantIndexerPrologOutputGolden{qInt8Golden, qScaleGolden, kInt8Golden, kScaleGolden, weightsGolden};
}

template <typename T = npu::tile_fwk::bfloat16, bool nz = true>
void TestQuantLightningIndexerProlog(QuantIndexerConfigs &configs) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    
    constexpr int64_t nzFirstDim = 16;
    constexpr int64_t b16C0Dim = 16;
    constexpr int64_t b8C0Dim = 32;

    // Read Inputs
    int paramsSize = 11;
    std::vector<int32_t> input_param(paramsSize);
    readInput<int32_t>(GetGoldenDir() + "/input_param.bin", input_param);
    auto t = input_param[2];
    auto h = input_param[3];
    auto qLoraRank = input_param[4];
    auto headDim = input_param[5];
    auto headNum = input_param[6];
    auto ropeHeadDim = input_param[7];
    auto blockSize = input_param[8];
    auto blockNum = input_param[9];
    auto nKV = input_param[10];

    DataType dType = (std::is_same<T, npu::tile_fwk::bfloat16>::value) ? DT_BF16 : DT_FP16;
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor x(dType, {t, h}, "x");
    Tensor qNorm(DT_INT8, {t, qLoraRank}, "qNorm");
    Tensor qNormScale(DT_FP32, {t, 1}, "qNormScale");
    Tensor wQb(
        DT_INT8, {headNum * headDim / b8C0Dim, qLoraRank / nzFirstDim, nzFirstDim, b8C0Dim}, "wQb", weightFormat);
    Tensor wQbScale(DT_FP32, {headNum * headDim, 1}, "wQbScale");
    Tensor wk(dType, {headDim / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}, "wk", weightFormat);
    Tensor wProj(dType, {headNum / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}, "wProj", weightFormat);
    Tensor lnGammaK(dType, {headDim}, "lnGammaK");
    Tensor lnBetaK(dType, {headDim}, "lnBetaK");
    Tensor cosIdxRope(dType, {t, ropeHeadDim}, "cosIdxRope");
    Tensor sinIdxRope(dType, {t, ropeHeadDim}, "sinIdxRope");
    Tensor hadamardQ(dType, {headDim, headDim}, "hadamardQ");
    Tensor hadamardK(dType, {headDim, headDim}, "hadamardK");
    Tensor kCache(DT_INT8, {blockNum, blockSize, nKV, headDim}, "kCache");
    Tensor kCacheScale(DT_FP16, {blockNum, blockSize, nKV, 1}, "kCacheScale");
    Tensor kCacheIndex(DT_INT64, {t}, "kCacheIndex");

    Tensor dynamicX(dType, {-1, h}, "dynamicX");                                       // dynamic
    Tensor dynamicQNorm(DT_INT8, {-1, qLoraRank}, "dynamicQNorm");                     // dynamic
    Tensor dynamicQNormScale(DT_FP32, {-1, 1}, "dynamicQNormScale");                   // dynamic
    Tensor dynamicCosIdxRope(dType, {-1, ropeHeadDim}, "dynamicCosIdxRope");           // dynamic
    Tensor dynamicSinIdxRope(dType, {-1, ropeHeadDim}, "dynamicSinIdxRope");           // dynamic
    Tensor dynamicKCache(DT_INT8, {-1, blockSize, nKV, headDim}, "dynamicKCache");     // dynamic
    Tensor dynamicKCacheScale(DT_FP16, {-1, blockSize, nKV, 1}, "dynamicKCacheScale"); // dynamic
    Tensor dynamicKCacheIndex(DT_INT64, {-1}, "dynamicKCacheIndex");                   // dynamic

    auto symT = GetInputShape(dynamicX, 0);
    auto symBlockNum = GetInputShape(dynamicKCache, 0);

    QuantIndexerPrologInput staticInput{x, qNorm, qNormScale, wQb, wQbScale, wk, wProj, lnGammaK, lnBetaK, cosIdxRope,
        sinIdxRope, hadamardQ, hadamardK, kCache, kCacheScale, kCacheIndex};
    QuantIndexerPrologInput dynamicInput{dynamicX, dynamicQNorm, dynamicQNormScale, wQb, wQbScale, wk, wProj, lnGammaK,
        lnBetaK, dynamicCosIdxRope, dynamicSinIdxRope, hadamardQ, hadamardK, dynamicKCache, dynamicKCacheScale,
        dynamicKCacheIndex};
    QuantIndexerPrologInputData inputData = PrepareQuantIndexerPrologInputsData<T, nz>(staticInput);

    // outputs
    Tensor qInt8(DT_INT8, {t, headNum, headDim}, "qInt8");
    Tensor qScale(DT_FP16, {t, headNum, 1}, "qScale");
    Tensor kInt8(DT_INT8, {blockNum, blockSize, nKV, headDim}, "kInt8");
    Tensor kScale(DT_FP16, {blockNum, blockSize, nKV, 1}, "kScale");
    Tensor weights(DT_FP16, {t, headNum}, "weights");

    Tensor dynamicQInt8(DT_INT8, {symT, headNum, headDim}, "dynamicQInt8");
    Tensor dynamicQScale(DT_FP16, {symT, headNum, 1}, "dynamicQScale");
    Tensor dynamicKInt8(DT_INT8, {symBlockNum, blockSize, nKV, headDim}, "dynamicKInt8");
    Tensor dynamicKScale(DT_FP16, {symBlockNum, blockSize, nKV, 1}, "dynamicKScale");
    Tensor dynamicWeights(DT_FP16, {symT, headNum}, "dynamicWeights");

    QuantIndexerPrologOutput staticOutput{qInt8, qScale, kInt8, kScale, weights};
    QuantIndexerPrologOutput dynamicOutput{dynamicQInt8, dynamicQScale, dynamicKInt8, dynamicKScale, dynamicWeights};
    QuantIndexerPrologOutputData outputData = PrepareQuantIndexerPrologOutputsData(staticOutput);
    // output golden
    QuantIndexerPrologOutputGolden outputGolden = PrepareQuantIndexerPrologOutputsGolden(staticOutput);

    std::vector<RawTensorDataPtr> inputDataList = {inputData.xData, inputData.qNormData, inputData.qNormScaleData,
        inputData.wQbData, inputData.wQbScaleData, inputData.wkData, inputData.wProjData, inputData.lnGammaKData,
        inputData.lnBetaKData, inputData.cosIdxRopeData, inputData.sinIdxRopeData, inputData.hadamardQData,
        inputData.hadamardKData, inputData.kCacheData, inputData.kCacheScaleData, inputData.kCacheIndexData};

    std::vector<RawTensorDataPtr> outputDataList = {
        outputData.qInt8Data, outputData.qScaleData, outputData.weightsData};

    QuantIndexerPrologAttr attrs;
    attrs.eps = 1e-6f;
    attrs.layeroutKey = "PA_BSND";
    attrs.layeroutQuery = "TND";

    QuantLightningIndexerProlog(dynamicInput, dynamicOutput, attrs, configs);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "qInt8 ====== " << std::endl;
    const float error_threshold = 0.0001f;
    const size_t error_count_threshold = 1000;
    EXPECT_TRUE(resultCmp<int8_t>(
        outputGolden.qInt8Golden, (int8_t *)outputData.qInt8Data->data(), 1, 0, error_count_threshold, false, true, 0));
    std::cout << "qScale ====== " << std::endl;
    EXPECT_TRUE(resultCmp<npu::tile_fwk::float16>(outputGolden.qScaleGolden,
        (npu::tile_fwk::float16 *)outputData.qScaleData->data(), error_threshold, 0, error_count_threshold, false, true,
        0));

    std::cout << "kInt8 ====== " << std::endl;
    EXPECT_TRUE(resultCmp<int8_t>(
        outputGolden.kInt8Golden, (int8_t *)inputData.kCacheData->data(), 1, 0, error_count_threshold, false, true,
        0));

    std::cout << "kScale ====== " << std::endl;
    EXPECT_TRUE(resultCmp<npu::tile_fwk::float16>(outputGolden.kScaleGolden,
        (npu::tile_fwk::float16 *)inputData.kCacheScaleData->data(), error_threshold, 0, error_count_threshold,
        false, true, 0));

    std::cout << "weight ======" << std::endl;
    EXPECT_TRUE(resultCmp<npu::tile_fwk::float16>(outputGolden.weightsGolden,
        (npu::tile_fwk::float16 *)outputData.weightsData->data(), error_threshold, 0, error_count_threshold, false,
        true, 0));
}

TEST_F(QuantLightningIndexerPrologSTest, b4_s1_2_s2_64k) {
    QuantIndexerConfigs configs;
    configs.qLinear = {32, 32, 512, 512, 128, 128};
    configs.qHd = {64, 64, 128, 128, 128, 128};
    configs.kLinear = {16, 16, 512, 512, 64, 64};
    configs.wLinear = {16, 16, 1024, 1024, 32, 32};
    TestQuantLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(configs);
}

TEST_F(QuantLightningIndexerPrologSTest, b8_s1_2_s2_64k) {
    QuantIndexerConfigs configs;
    configs.qLinear = {32, 32, 512, 512, 128, 128};
    configs.qHd = {64, 64, 128, 128, 128, 128};
    configs.kLinear = {16, 16, 512, 512, 64, 64};
    configs.wLinear = {16, 16, 1024, 1024, 32, 32};
    TestQuantLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(configs);
}

TEST_F(QuantLightningIndexerPrologSTest, b1_s1_4k_s2_64k) {
    QuantIndexerConfigs configs;
    configs.qLinear = {32, 32, 512, 512, 128, 128};
    configs.qHd = {64, 64, 128, 128, 128, 128};
    configs.kLinear = {16, 16, 512, 512, 64, 64};
    configs.wLinear = {16, 16, 1024, 1024, 32, 32};
    TestQuantLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(configs);
}

TEST_F(QuantLightningIndexerPrologSTest, b2_s1_4k_s2_64k) {
    QuantIndexerConfigs configs;
    configs.qLinear = {32, 32, 512, 512, 128, 128};
    configs.qHd = {64, 64, 128, 128, 128, 128};
    configs.kLinear = {16, 16, 512, 512, 64, 64};
    configs.wLinear = {16, 16, 1024, 1024, 32, 32};
    TestQuantLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(configs);
}

TEST_F(QuantLightningIndexerPrologSTest, b128_s1_4_s2_8k) {
    QuantIndexerConfigs configs;
    configs.qLinear = {128, 128, 256, 256, 256, 256};
    configs.qHd = {128, 128, 64, 64, 128, 128};
    configs.kLinear = {64, 64, 256, 256, 128, 128};
    configs.wLinear = {32, 32, 512, 512, 64, 64};
    configs.tSubTile = 2;
    configs.chunkSize = 1;
    configs.l1ReuseParam = {
        {1, 4}, {3, 4}
    };

    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 512);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 512);
    TestQuantLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(configs);
}
} // namespace
