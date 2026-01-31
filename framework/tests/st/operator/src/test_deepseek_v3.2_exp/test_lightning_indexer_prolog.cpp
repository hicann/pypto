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
 * \file test_dynamic_mla_prolog.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "operator/models/deepseek_v3.2_exp/lightning_indexer_prolog.h"
#include "operator/models/deepseek_v3.2_exp/dsia_common.h"
#include "test_cost_macro.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class LightningIndexerPrologSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

void PerformanceConfig() {
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 4}});
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2 * 1024 * 1024);
}

static IndexerShapeParams ReadParams(const RopeTileShapeConfig &ropeTileConfigs, const IndexerTileShapeConfig &indexerConfigs, int tileBS) {
    int paramsSize = 11;
    std::vector<int32_t> input_param(paramsSize);
    readInput<int32_t>(GetGoldenDir() + "/input_param.bin", input_param);

    IndexerShapeParams params;
    params.s2 = input_param[0];
    params.b = input_param[1];
    params.seq = input_param[2];
    params.dim = input_param[3];
    params.qLoraRank = input_param[4];
    params.headDim = input_param[5];
    params.headNum = input_param[6];
    params.ropeHeadDim = input_param[7];
    params.blockSize = input_param[8]; // PA block size
    params.blockNum = input_param[9];
    params.nKV = input_param[10];
    params.indexerTileConfigs = indexerConfigs;
    params.ropeTileConfigs = ropeTileConfigs;
    params.tileBS = tileBS;

    std::cout << "Read params:" << std::endl;
    std::cout << "s2=" << params.s2 << ", b=" << params.b << ", seq=" << params.seq << ", dim=" << params.dim
              << ", qLoraRank=" << params.qLoraRank << ", headDim=" << params.headDim << ", headNum=" << params.headNum
              << ", ropeHeadDim=" << params.ropeHeadDim << ", blockSize=" << params.blockSize
              << ", blockNum=" << params.blockNum << ", nKV=" << params.nKV << ", tileBS=" << params.tileBS << std::endl;

    return params;
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
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
IndexerPrologInputData PrepareIndexerPrologInputsData(const IndexerPrologInput &inputs) {
    auto xData = CreateTensorData<T>(inputs.x, "/token_x.bin");
    auto qrData = CreateTensorData<T>(inputs.qr, "/qr.bin");
    auto qWData = CreateTensorData<T>(inputs.qW, nz ? "/wq_b_nz.bin" : "/wq_b.bin");
    auto kWData = CreateTensorData<T>(inputs.kW, nz ? "/wk_nz.bin" : "/wk.bin");
    auto projWData = CreateTensorData<T>(inputs.projW, nz ? "/weights_proj_nz.bin" : "/weights_proj.bin");
    auto lnWData = CreateTensorData<T>(inputs.lnW, "/weight_layer_norm.bin");
    auto lnBiasData = CreateTensorData<T>(inputs.lnBias, "/bias_layer_norm.bin");
    auto cosData = CreateTensorData<T>(inputs.cos, "/cos_idx_rope.bin");
    auto sinData = CreateTensorData<T>(inputs.sin, "/sin_idx_rope.bin");
    auto kCacheData = CreateTensorData<T>(inputs.kCache, "/idx_k_cache.bin");
    auto kCacheIndexData = CreateTensorData<int32_t>(inputs.kCacheIndex, "/idx_k_cache_index.bin");
    auto blockTableData = CreateTensorData<int32_t>(inputs.blockTable, "/idx_block_table.bin");
    return IndexerPrologInputData{xData, qrData, qWData, kWData, projWData, lnWData, lnBiasData, cosData, sinData,
        kCacheData, kCacheIndexData, blockTableData};
}

template <typename T>
IndexerPrologOutputData PrepareIndexerPrologOutputsData(const IndexerPrologOutput &outputs) {
    auto qOutData = RawTensorData::CreateConstantTensor<T>(outputs.query, 0.0f);
    auto weightData = RawTensorData::CreateConstantTensor<T>(outputs.weight, 0.0f);
    return IndexerPrologOutputData{qOutData, weightData};
}

template <typename T>
IndexerPrologOutputGolden<T> PrepareIndexerPrologOutputsGolden(const IndexerPrologOutput &outputs) {
    auto queryGolden = getGoldenVec<T>(outputs.query.GetShape(), "/query_golden.bin");
    auto weightGolden = getGoldenVec<T>(outputs.weight.GetShape(), "/weights_golden.bin");
    auto kCacheOutGolden = getGoldenVec<T>(outputs.kCacheOut.GetShape(), "/idx_k_cache_out_golden.bin");
    return IndexerPrologOutputGolden<T>{queryGolden, weightGolden, kCacheOutGolden};
}

template <typename T = npu::tile_fwk::bfloat16, bool nz = true>
void TesLightningIndexerProlog(const IndexerShapeParams &params) {

    // inputs
    DataType dType = (std::is_same<T, npu::tile_fwk::bfloat16>::value) ? DT_BF16 : DT_FP16;
    int b = params.b;
    int seq = params.seq;
    int dim = params.dim;
    int qLoraRank = params.qLoraRank;
    int headDim = params.headDim;
    int headNum = params.headNum;
    int ropeHeadDim = params.ropeHeadDim;
    int blockNum = params.blockNum;
    int blockSize = params.blockSize;
    int nKV = params.nKV;
    int s2 = params.s2;
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor x(dType, {b, seq, dim}, "x");
    Tensor qr(dType, {b, seq, qLoraRank}, "qr");
    Tensor qW(dType, {qLoraRank, headNum * headDim}, "qW", weightFormat);
    Tensor kW(dType, {dim, headDim}, "kW", weightFormat);
    Tensor projW(dType, {dim, headNum}, "projW", weightFormat);
    Tensor lnW(dType, {headDim}, "lnW");
    Tensor lnBias(dType, {headDim}, "lnBias");
    Tensor cos(dType, {b, seq, ropeHeadDim}, "cos");
    Tensor sin(dType, {b, seq, ropeHeadDim}, "sin");
    Tensor kCache(dType, {blockNum, blockSize, nKV, headDim}, "kCache");
    Tensor kCacheIndex(DT_INT32, {b, seq}, "kCacheIndex");
    Tensor blockTable(DT_INT32, {b, s2 / blockSize}, "actSeqLen");
    IndexerPrologInput input{x, qr, qW, kW, projW, lnW, lnBias, cos, sin, kCache, kCacheIndex, blockTable};
    IndexerPrologInputData inputData = PrepareIndexerPrologInputsData<T, nz>(input);

    // outputs
    Tensor query(dType, {b * seq, headNum, headDim}, "qOut");
    Tensor weight(dType, {b * seq, headNum}, "weightOut");
    Tensor kCacheOut(dType, {blockNum, blockSize, nKV, headDim}, "kCacheOut");
    IndexerPrologOutput output{query, weight, kCacheOut};
    IndexerPrologOutputData outputData = PrepareIndexerPrologOutputsData<T>(output);
    // output golden
    IndexerPrologOutputGolden<T> outputGolden = PrepareIndexerPrologOutputsGolden<T>(output);

    std::vector<RawTensorDataPtr> inputDataList = {inputData.xData, inputData.qrData, inputData.qWData,
        inputData.kWData, inputData.projWData, inputData.lnWData, inputData.lnBiasData, inputData.cosData,
        inputData.sinData, inputData.kCacheData, inputData.kCacheIndexData, inputData.blockTableData};
    std::vector<RawTensorDataPtr> outputDataList = {outputData.queryData, outputData.weightData};

    LightningIndexerProlog(input, output, params);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "query ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(outputGolden.queryGolden, (T *)outputData.queryData->data(), 0.003f));
    std::cout << "weight ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(outputGolden.weightGolden, (T *)outputData.weightData->data(), 0.003f));
    std::cout << "kCacheOut ======" << std::endl;
    // This is an inplace output
    EXPECT_TRUE(resultCmp<T>(outputGolden.kCacheOutGolden, (T *)inputData.kCacheData->data(), 0.003f, 0, 1000, false, true, 0));
}

TEST_F(LightningIndexerPrologSTest, bf16_indexer_prolog) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 1, 64, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 16, 16, 256, 256, 128, 128}, // c1TileShape
        {1, 256, 128, 128 }, // v1TileShape
        { 16, 16, 256, 256, 128, 128}, // c2TileShape
        {1, 128, 128, 128 } // v2TileShape
    };

    auto params = ReadParams(ropeTileConfigs, indexerConfigs, -1);
    PerformanceConfig();
    TesLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(params);
}

TEST_F_WITH_COST(LightningIndexerPrologSTest, b48_s1_1_s2_8k, 26) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 1, 64, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 16, 16, 256, 256, 128, 128}, // c1TileShape
        { 1, 256, 128, 128 }, // v1TileShape
        { 16, 16, 256, 256, 128, 128}, // c2TileShape
        { 1, 128, 128, 128 } // v2TileShape
    };
    auto params = ReadParams(ropeTileConfigs, indexerConfigs, -1);
    PerformanceConfig();
    TesLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(params);
}

TEST_F(LightningIndexerPrologSTest, b2_s1_2_s2_2k) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 16, 128, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 32, 32, 256, 256, 128, 128}, // c1TileShape
        {1, 256, 128, 128 }, // v1TileShape
        { 32, 32, 256, 256, 128, 128}, // c2TileShape
        {1, 128, 128, 128 } // v2TileShape
    };
    auto params = ReadParams(ropeTileConfigs, indexerConfigs, -1);

    PerformanceConfig();
    TesLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(params);

}

TEST_F(LightningIndexerPrologSTest, b35_s1_2_s2_8k) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 16, 128, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 32, 32, 256, 256, 128, 128}, // c1TileShape
        {128, 256, 128, 128 }, // v1TileShape
        { 32, 32, 256, 256, 128, 128}, // c2TileShape
        {128, 128, 128, 128 } // v2TileShape
    };
    auto params = ReadParams(ropeTileConfigs, indexerConfigs, -1);

    PerformanceConfig();
    TesLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(params);
}

TEST_F(LightningIndexerPrologSTest, b40_s1_4_s2_8k) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 16, 128, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 32, 32, 256, 256, 128, 128}, // c1TileShape
        {128, 256, 128, 128 }, // v1TileShape
        { 32, 32, 256, 256, 128, 128}, // c2TileShape
        {128, 128, 128, 128 } // v2TileShape
    };
    auto params = ReadParams(ropeTileConfigs, indexerConfigs, 32);

    PerformanceConfig();
    TesLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(params);
}

TEST_F_WITH_COST(LightningIndexerPrologSTest, b4_s1_1_s2_64k, 26) {
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 256},
        { 32, 128, 128},
        { 1, 64, 128, 128}
    };
    IndexerTileShapeConfig indexerConfigs{
        { 16, 16, 256, 256, 128, 128}, // c1TileShape
        {1, 256, 128, 128 }, // v1TileShape
        { 16, 16, 256, 256, 128, 128}, // c2TileShape
        {1, 128, 128, 128 } // v2TileShape
    };
    auto params = ReadParams(ropeTileConfigs, indexerConfigs, -1);
    PerformanceConfig();
    TesLightningIndexerProlog<npu::tile_fwk::bfloat16, true>(params);
}
} // namespace