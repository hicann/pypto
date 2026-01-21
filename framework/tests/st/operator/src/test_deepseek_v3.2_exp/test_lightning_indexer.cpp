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
 * \file test_lightning_indexer.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/lightning_indexer.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class LightningIndexerSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(
    Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

void TestLightningIndexer(LightningIndexerConfigs &tileConfig) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    int paramsSize = 9;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", input_param);

    const int64_t b = input_param[0];
    const int64_t s1 = input_param[1];
    const int64_t n1 = input_param[2];
    const int64_t d = input_param[3];
    const int64_t blockNum = input_param[4];
    const int64_t blockSize = input_param[5];
    const int64_t n2 = input_param[6];
    const int64_t maxBlockNum = input_param[7];
    const int64_t selectedCount = input_param[8];

    Tensor staticQuery(DT_INT8, {b * s1, n1, d}, "staticQuery");
    Tensor staticKey(DT_INT8, {blockNum, blockSize, n2, d}, "staticKey");
    Tensor staticQScale(DT_FP16, {b * s1, n1}, "staticQScale");
    Tensor staticKScale(DT_FP16, {blockNum, blockSize, n2}, "staticKScale");
    Tensor staticWeights(DT_FP16, {b * s1, n1}, "staticWeights");
    Tensor staticActSeq(DT_INT32, {b}, "staticActSeq");
    Tensor staticBlockTable(DT_INT32, {b, maxBlockNum}, "staticBlockTable");
    Tensor staticTopkRes(DT_INT32, {b * s1, n2, selectedCount}, "staticTopkRes");
    Tensor staticFirstMm(DT_FP16, {b * s1 * n1, maxBlockNum * blockSize}, "staticFirstMm");
    Tensor staticMmOut(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "staticMmOut");
    Tensor staticTopkValue(DT_FP32, {b * s1, n2, selectedCount}, "staticTopkValue");

    // dynamic input
    Tensor query(DT_INT8, {-1, n1, d}, "query");
    Tensor qScale(DT_FP16, {-1, n1}, "qScale");
    Tensor key(DT_INT8, {-1, blockSize, n2, d}, "key");
    Tensor kScale(DT_FP16, {-1, blockSize, n2}, "kScale");
    Tensor weights(DT_FP16, {-1, n1}, "weights");
    Tensor actSeq(DT_INT32, {-1}, "actSeq");
    Tensor blockTable(DT_INT32, {-1, -1}, "blockTable");
    // dynamic axis infer from input
    auto symT = GetInputShape(query, 0);
    auto symMaxBlock = GetInputShape(blockTable, 1);
    // dynamic output
    Tensor firstMm(DT_FP16, {symT * n1, symMaxBlock * blockSize}, "firstMm");
    Tensor mmOut(DT_FP32, {symT * n2, symMaxBlock * blockSize}, "MmOut");
    Tensor topkRes(DT_INT32, {symT, n2, selectedCount}, "topkRes");
    Tensor topkValue(DT_FP32, {symT, n2, selectedCount}, "topkValue");
    // static golden
    auto firstMmGolden = getGoldenVec<npu::tile_fwk::float16>({b * s1 * n1, maxBlockNum * blockSize}, "/first_mm.bin");
    auto mmGolden = getGoldenVec<float>({b * s1 * n2, maxBlockNum * blockSize}, "/mm_out.bin");
    auto topkResGolden = getGoldenVec<int32_t>({b * s1, n2, selectedCount}, "/topk_res.bin");
    auto topkValueGolden = getGoldenVec<float>({b * s1, n2, selectedCount}, "/topk_value.bin");
    // static input
    auto qData = CreateTensorData<int8_t>(staticQuery, {b * s1, n1, d}, "/query.bin");
    auto kData = CreateTensorData<int8_t>(staticKey, {blockNum, blockSize, n2, d}, "/key.bin");
    auto qsData = CreateTensorData<npu::tile_fwk::float16>(staticQScale, {b * s1, n1}, "/q_scale.bin");
    auto ksData = CreateTensorData<npu::tile_fwk::float16>(staticKScale, {blockNum, blockSize, n2}, "/k_scale.bin");
    auto wData = CreateTensorData<npu::tile_fwk::float16>(staticWeights, {b * s1, n1}, "/weights.bin");
    auto sData = CreateTensorData<int32_t>(staticActSeq, {b}, "/act_seq.bin");
    auto bData = CreateTensorData<int32_t>(staticBlockTable, {b, maxBlockNum}, "/block_table.bin");
    // static output
    auto firstMmData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(staticFirstMm, 0.0f);
    auto mmData = RawTensorData::CreateConstantTensor<float>(staticMmOut, 0.0f);
    auto topkResData = RawTensorData::CreateConstantTensor<int32_t>(staticTopkRes, 0);
    auto topkValueData = RawTensorData::CreateConstantTensor<float>(staticTopkValue, 0.0f);

    std::vector<RawTensorDataPtr> inputDataList = {qData, qsData, kData, ksData, wData, sData, bData};
    std::vector<RawTensorDataPtr> outputDataList = {topkResData, firstMmData, mmData, topkValueData};

    std::set<int> unrollList = {32, 16, 8, 4, 1};
    FUNCTION(
        "LightningIndexer", {query, qScale, key, kScale, weights, actSeq, blockTable}, {topkRes, firstMm, mmOut, topkValue}) {
        LightningIndexerImpl(query, qScale, key, kScale, weights, actSeq, blockTable, selectedCount, topkRes,
            tileConfig, unrollList, &firstMm, &mmOut, &topkValue);
    }

    DevFuncRunner::Run(
        Program::GetInstance().GetLastFunction(), inputDataList, outputDataList, DeviceLauncherConfig(0));
    constexpr float PRE_TAIL = 1e-4f;
    constexpr int TOPK_COUNT = 100;
    constexpr float ratio = 5e-3f;
    std::cout << "=======================firstMm===============================" << std::endl;
    EXPECT_TRUE(resultCmp(firstMmGolden, (npu::tile_fwk::float16 *)firstMmData->data(), PRE_TAIL, 0, TOPK_COUNT, false, false));
    std::cout << "=======================mmOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp(mmGolden, (float *)mmData->data(), PRE_TAIL, 0, TOPK_COUNT, false, false));
    std::cout << "=======================topkValue===============================" << std::endl;
    EXPECT_TRUE(resultCmp(topkValueGolden, (float *)topkValueData->data(), PRE_TAIL, 0, TOPK_COUNT, false, false));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp4TopK(topkResGolden, (int32_t *)topkResData->data(), selectedCount, ratio));
}

// LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2
TEST_F(LightningIndexerSTest, lightning_indexer_quant_4_b_2_s1_64k_s2) {
    LightningIndexerConfigs config;
    config.s1Tile = 2; // s1Tile = s1
    config.topkTile = 8192;
    config.c1Tile = {128, 128, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.c2Tile = {64, 64, 128, 128, 128, 128};   // (m, M), (k, K), (n, N)
    config.extendParam.reluType = npu::tile_fwk::Matrix::ReLuType::ReLu;
    float scale = 2048.0;
    config.extendParam.scaleValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scale));

    TestLightningIndexer(config);
}