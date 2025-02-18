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
 * \file test_dynamic_fused_compress_kv_select.cpp
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
#include "operator/models/deepseek_v3.2_exp/lightning_indexer_topk.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicIndexerTopk : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

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

void TestLightningIndexerTopkQuant(IndexerTile &tileConfig) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 9;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", input_param);

    const int b = input_param[0];
    const int s1 = input_param[1];
    const int n1 = input_param[2];
    const int d = input_param[3];
    const int blockNum = input_param[4];
    const int blockSize = input_param[5];
    const int n2 = input_param[6];
    const int maxBlockNum = input_param[7];
    const int selectedCount = input_param[8];

    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};

    Tensor staticQuery(DT_INT8, {b, s1, n1, d}, "staticQuery");
    Tensor staticKey(DT_INT8, {blockNum, blockSize, n2, d}, "staticKey");
    Tensor staticQScale(DT_FP16, {b, s1, n1, 1}, "staticQScale");
    Tensor staticKScale(DT_FP16, {blockNum, blockSize, n2, 1}, "staticKScale");
    Tensor staticWeights(DT_FP16, {b, s1, n1}, "staticWeights");
    Tensor staticActSeq(DT_INT32, {b}, "staticActSeq");
    Tensor staticBlockTable(DT_INT32, {b, maxBlockNum}, "staticBlockTable");
    Tensor staticTopkRes(DT_INT32, {b, s1, n2, selectedCount}, "staticTopkRes");
    Tensor staticTmpOut(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "staticTmpOut");
    Tensor staticTopkValue(DT_FP32, {b, s1, n2, selectedCount}, "staticTopkValue");

    Tensor query(DT_INT8, {-1, -1, n1, d}, "query");
    Tensor key(DT_INT8, {-1, blockSize, n2, d}, "key");
    Tensor qScale(DT_FP16, {-1, -1, n1, 1}, "qScale");
    Tensor kScale(DT_FP16, {-1, blockSize, n2, 1}, "kScale");
    Tensor weights(DT_FP16, {-1, -1, n1}, "weights");
    Tensor actSeq(DT_INT32, {-1}, "actSeq");
    Tensor blockTable(DT_INT32, {-1, -1}, "blockTable");

    auto symB = GetInputShape(query, 0);
    auto symS1 = GetInputShape(query, 1);
    auto symMaxBlock = GetInputShape(blockTable, 1);

    Tensor topkRes(DT_INT32, {symB, symS1, n2, selectedCount}, "topkRes");
    Tensor tmpOut(DT_FP32, {symB * symS1 * n2, symMaxBlock * blockSize}, "tmpOut");
    Tensor topkValue(DT_FP32, {symB, symS1, n2, selectedCount}, "topkValue");

    auto topkResGolden = getGoldenVec<int32_t>({b, s1, n2, selectedCount}, "/topk_res.bin");
    auto tmpGolden = getGoldenVec<float>({b * s1 * n2, maxBlockNum * blockSize}, "/tmp_out.bin");
    auto topkValueGolden = getGoldenVec<float>({b, s1, n2, selectedCount}, "/topk_value.bin");

    auto qData = CreateTensorData<int8_t>(staticQuery, {b, s1, n1, d}, "/query.bin");
    auto kData = CreateTensorData<int8_t>(staticKey, {blockNum, blockSize, n2, d}, "/key.bin");
    auto qsData = CreateTensorData<npu::tile_fwk::float16>(staticQScale, {b, s1, n1, 1}, "/q_scale.bin");
    auto ksData = CreateTensorData<npu::tile_fwk::float16>(staticKScale, {blockNum, blockSize, n2, 1}, "/k_scale.bin");
    auto wData = CreateTensorData<npu::tile_fwk::float16>(staticWeights, {b, s1, n1}, "/weights.bin");
    auto sData = CreateTensorData<int32_t>(staticActSeq, {b}, "/act_seq.bin");
    auto bData = CreateTensorData<int32_t>(staticBlockTable, {b, maxBlockNum}, "/block_table.bin");
    auto topkResData = RawTensorData::CreateConstantTensor<int32_t>(staticTopkRes, 0);
    auto tmpData = RawTensorData::CreateConstantTensor<float>(staticTmpOut, 0);
    auto topkValueData = RawTensorData::CreateConstantTensor<float>(staticTopkValue, 0);

    std::vector<RawTensorDataPtr> inputDataList = {qData, kData, qsData, ksData, wData, sData, bData};
    // std::vector<RawTensorDataPtr> outputDataList = {topkResData, tmpData};
    std::vector<RawTensorDataPtr> outputDataList = {topkResData, tmpData, topkValueData};

    FUNCTION("IndexerTopk", {query, key, qScale, kScale, weights, actSeq, blockTable}, {topkRes, tmpOut, topkValue}) {
        LightningIndexerTopkImpl(query, key, true, &qScale, &kScale,
            weights, actSeq, blockTable, topkRes, selectedCount, tileConfig, unrollList,
            &tmpOut, &topkValue);
    }

    DevFuncRunner::Run(
        Program::GetInstance().GetLastFunction(), inputDataList, outputDataList, DeviceLauncherConfig(0));
    constexpr float PRE_TAIL = 1e-5f;
    constexpr int TOPK_COUNT = 100;
    constexpr float ratio = 5e-3f;
    std::cout << "=======================topkValue===============================" << std::endl;
    EXPECT_TRUE(resultCmp(topkValueGolden, (float *)topkValueData->data(), PRE_TAIL, 0, TOPK_COUNT, false, true));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp4TopK(topkResGolden, (int32_t *)topkResData->data(), selectedCount, ratio));
}

// DynamicIndexerTopk.indexer_topk_quant_4_b_1_s1_64k_s2
TEST_F(DynamicIndexerTopk, indexer_topk_quant_4_b_1_s1_64k_s2) {
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 100 * 1024 * 1024); // mistake
    config::SetPassOption(SG_PG_LOWER_BOUND, 1024);
    config::SetPassOption(SG_PG_UPPER_BOUND, 1024 * 1024);
    config::SetPassOption(CUBE_L1_REUSE_MODE, 32);
    config::SetPassOption(SG_PARALLEL_NUM, 2);
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{
                                                                                    {-1, 16}
    });
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption<uint8_t>(
        DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
                            static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));

    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 128);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 128);
    IndexerTile config;

    config.weightTile = {64, 128};
    config.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.v1Tile = {64, 128};
    config.topkTile = {1, 4096};
    config.addsTile = {1, 1, 1, 4096};

    TestLightningIndexerTopkQuant(config);
}