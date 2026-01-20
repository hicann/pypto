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
 * \file test_onboard_genAtten.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/gen_Attention.h"
#include "test_dev_func_runner.h"
#include "test_data_loader.h"
#include "test_cost_macro.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class TestGenAtten : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_128 = 128;
constexpr int NUM_512 = 512;

void GenAttentionCompute(TestDataLoader& data, GenAttenTileShapeConfig &tileConfig) {
    int b = std::get<int>(data.Param("b"));
    int s1 = std::get<int>(data.Param("s1"));
    int n = std::get<int>(data.Param("n"));
    int d = std::get<int>(data.Param("d"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);

    GenAttentionCompute(
        data.InputTensorCheck("cmp_atten", dType, {b, s1, n, d}),       // 也可以使用 data.InputTensor("cmp_atten"), 不做二次校验
        data.InputTensorCheck("sel_atten", dType, {b, s1, n, d}),
        data.InputTensorCheck("win_atten", dType, {b, s1, n, d}),
        data.InputTensorCheck("gating_score", dType, {b, s1, n, NUM_3}),
        data.OutputTensorCheck("attention_out", dType, {b, s1, n, d}),  // 也可以使用 data.OutputTensor("attention_out"), 不做二次校验
        tileConfig
    );
}

template<typename T = npu::tile_fwk::float16>
void genAtten(TestDataLoader& data, GenAttenTileShapeConfig &tileConfig) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    int b = std::get<int>(data.Param("b"));
    int s1 = std::get<int>(data.Param("s1"));
    int n = std::get<int>(data.Param("n"));
    int d = std::get<int>(data.Param("d"));
    std::string dTypeStr = std::get<string>(data.Param("dtype"));
    DataType dType = CostModel::ToDataType(dTypeStr);
    data.Dump();    // 打印 tensor 信息

    FUNCTION("GenAtten", data.GetInputTensorList(), data.GetOutputTensorList()) {
        GenAttentionCompute(data, tileConfig);
    }

    auto goldenData = data.GoldenDataCheck("attention_out", dType, {b, s1, n, d});  // 也可以使用 data.GoldenData("attention_out")
    auto outputData = data.GetOutputDataList()[data.GetOutputNameToIdx("attention_out")];

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), data.GetInputDataList(), data.GetOutputDataList());
    EXPECT_TRUE(resultCmp<T>((T *)goldenData->data(), (T *)outputData->data(), goldenData->GetSize(), 0.001f));
#endif
}

TEST_F_WITH_COST(TestGenAtten, TestDynamicGenAtten_B_16_S1_1_FP16, 11) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    genAtten<npu::tile_fwk::float16>(data, tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAtten_B_16_S1_1_FP32) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    genAtten<float>(data, tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAtten_B_16_S1_1_BF16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    genAtten<npu::tile_fwk::bfloat16>(data, tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAtten_B_16_S1_2_FP16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    genAtten<npu::tile_fwk::float16>(data, tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAtten_B_16_S1_2_FP32) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    genAtten<float>(data, tileConfig);
}

TEST_F(TestGenAtten, TestDynamicGenAtten_B_16_S1_2_BF16) {
    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    std::string configPath = GetGoldenDir() + "/config.json";
    TestDataLoader data(configPath);
    genAtten<npu::tile_fwk::bfloat16>(data, tileConfig);
}