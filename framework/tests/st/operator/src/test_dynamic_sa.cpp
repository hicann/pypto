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
 * \file test_dynamic_sa.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/slc_attn.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicSATest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

struct SaConfig {
    bool manualUnroll{false};
    int maxUnrollTimes{1};
    bool onlyBatchLoop{false};
    bool isNzFormat{false};
};

template <typename T = npu::tile_fwk::float16>
void TestSa(SaTileShapeConfig& tileConfig, SaConfig config) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    std::vector<uint8_t> devProgBinary;

    int paramsSize = 8;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);

    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nkv = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int smax = input_param[6];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    std::cout << "====input param==== b sq nq nkv dn dr smax: " << b << " " << sq << " " << nq << " " << nkv << " " << dn << " " << dr << " " << smax << std::endl;

    TileOpFormat kvFormat = config.isNzFormat ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    std::vector<int64_t> qNopeShape = {b * sq * nq, dn};
    std::vector<int64_t> qRopeShape = {b * sq * nq, dr};
    std::vector<int64_t> kSlcShape = {b * sq * nkv * smax, dn + dr};
    std::vector<int64_t> vSlcShape = {b * sq * nkv * smax, dn};
    std::vector<int64_t> actSeqsShape = {b, sq};
    std::vector<int64_t> saOutShape = {b, sq, nq, dn};

    Tensor actSeqs(DT_INT32, actSeqsShape, "actSeqs");
    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kSlc(dType, kSlcShape, "kSlc", kvFormat);
    Tensor vSlc(dType, vSlcShape, "vSlc", kvFormat);
    Tensor saOut(DT_FP32, saOutShape, "saOut");

    // 读数据
    int qNopeSize = std::accumulate(qNopeShape.begin(), qNopeShape.end(), 1, std::multiplies<>());
    int qRopeSize = std::accumulate(qRopeShape.begin(), qRopeShape.end(), 1, std::multiplies<>());
    int kSlcSize = std::accumulate(kSlcShape.begin(), kSlcShape.end(), 1, std::multiplies<>());
    int vSlcSize = std::accumulate(vSlcShape.begin(), vSlcShape.end(), 1, std::multiplies<>());
    int actSeqsSize = std::accumulate(actSeqsShape.begin(), actSeqsShape.end(), 1, std::multiplies<>());
    int saOutSize = std::accumulate(saOutShape.begin(), saOutShape.end(), 1, std::multiplies<>());

    std::vector<int> seq(actSeqsSize, 0);
    std::vector<T> qNopeData(qNopeSize, 0);
    std::vector<T> qRopeData(qRopeSize, 0);
    std::vector<T> kSlcData(kSlcSize, 0);
    std::vector<T> vSlcData(vSlcSize, 0);

    readInput<int>(GetGoldenDir() + "/actual_seq.bin", seq);
    readInput<T>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<T>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    if (config.isNzFormat) {
    } else {
        readInput<T>(GetGoldenDir() + "/k_slc.bin", kSlcData);
        readInput<T>(GetGoldenDir() + "/v_slc.bin", vSlcData);
    }

    std::vector<float> golden(saOutSize, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(qNope, qNopeData),
        RawTensorData::CreateTensor<T>(qRope, qRopeData),
        RawTensorData::CreateTensor<T>(kSlc, kSlcData),
        RawTensorData::CreateTensor<T>(vSlc, vSlcData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(saOut, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(saOut, golden),
    });

    SlcAttn(qNope, qRope, kSlc, vSlc, actSeqs, nq, nkv, softmaxScale, saOut, tileConfig);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.0005f));
    // EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.0005f, 0, 1000, true));
}

TEST_F(DynamicSATest, slc_attn_fp16) { // 测试项：fp16, flash小块
    SaTileShapeConfig tileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 128; // for s2Loop split
    tileConfig.gTile = gTile;
    tileConfig.sKvTile = sTile;
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {16, 256}; // (n1, d)
    SaConfig config;
    TestSa<npu::tile_fwk::float16>(tileConfig, config);
}

TEST_F(DynamicSATest, slc_attn_mtp_s1_2_fp16) { // 测试项：fp16, s1=2, g切分, flash大块
    SaTileShapeConfig tileConfig;
    const int gTile = 64; // for gLoop split
    const int sTile = 512; // for s2Loop split
    tileConfig.gTile = gTile;
    tileConfig.sKvTile = sTile;
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {gTile, 128}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {gTile, 128}; // (n1, d)
    SaConfig config;
    TestSa<npu::tile_fwk::float16>(tileConfig, config);
}

TEST_F(DynamicSATest, slc_attn_bf16_b48_s1_perf) { // 测试项：性能用例，bf16, b=48, s1=1
    SaTileShapeConfig tileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    tileConfig.gTile = gTile;
    tileConfig.sKvTile = sTile;
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {64, 128}; // (n1, d)
    SaConfig config;
    TestSa<npu::tile_fwk::bfloat16>(tileConfig, config);
}
