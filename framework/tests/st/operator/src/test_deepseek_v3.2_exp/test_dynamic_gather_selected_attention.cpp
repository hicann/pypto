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
 * \file test_dynamic_gather_selected_attention.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>
#include "tilefwk/data_type.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/gather_selected_attention.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicGatherSlcFlashAttnDSASTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T = npu::tile_fwk::float16>
void TestSa(SaTileShapeConfig& tileConfig) {
    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }
    int paramsSize = 11;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);
    int b = input_param.at(0);
    int sq = input_param.at(1);
    int nq = input_param.at(2);
    int nkv = input_param.at(3);
    int maxKVSeq = input_param.at(4);
    int dn = input_param.at(5);
    int dr = input_param.at(6);
    int blockNum = input_param.at(7);
    int blockSize = input_param.at(8);
    int topk = input_param.at(9);
    int isKnQuant = input_param.at(10);
    
    int maxBlockNumPerBatch = CeilDiv(maxKVSeq, blockSize);
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    std::cout << "====input param==== b sq nq nkv dn dr blockNum blockSize topk is_kn_quant: " 
                << b << " " << sq << " " << nq << " " << nkv << " " << dn << " " << dr << " " << blockNum << " " 
                << blockSize << " " << topk << " " << isKnQuant << std::endl;
    std::vector<int64_t> qNopeShape = {b * sq * nq, dn};
    std::vector<int64_t> qRopeShape = {b * sq * nq, dr};
    std::vector<int64_t> knShape = {blockNum * blockSize, dn};
    std::vector<int64_t> krShape = {blockNum * blockSize, dr};
    std::vector<int64_t> knScalesShape = {blockNum * blockSize, 4};
    std::vector<int64_t> topKIndciesShape = {b * sq, nkv * topk};
    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> actSeqsShape = {b};
    std::vector<int64_t> saOutShape = {b, sq, nq, dn};

    auto qNope = CreateTensorAndData<T>(qNopeShape, dType, "qNope", "/q_nope.bin", {0});
    auto qRope = CreateTensorAndData<T>(qRopeShape, dType, "qRope", "/q_rope.bin", {0});
    auto kRope2D = CreateTensorAndData<T>(krShape, dType, "kr", "/k_rope.bin", {0});
    auto knScales = CreateTensorAndData<float>(knScalesShape, DT_FP32, "knScales", "/kn_scales.bin", {0});
    auto topKIndcies = CreateTensorAndData<int32_t>(topKIndciesShape, DT_INT32, "topKIndcies", "/topk_indcies.bin", {0});
    auto blockTable = CreateTensorAndData<int32_t>(blockTableShape, DT_INT32, "blockTable", "/block_table.bin", {0, 1});
    auto actSeqs = CreateTensorAndData<int32_t>(actSeqsShape, DT_INT32, "actSeqs", "/actual_seq.bin", {0});

    Tensor saOut(dType, {b, sq, nq, dn}, "saOut");
    RawTensorDataPtr saOutData = RawTensorData::CreateConstantTensorData<T>(saOutShape, dType, 0);

    if(isKnQuant == 0){
        auto kNope2D = CreateTensorAndData<T>(knShape, dType, "kn", "/k_nope.bin", {0});
        SelectedAttentionV2(
            qNope.tensor, qRope.tensor, kNope2D.tensor, kRope2D.tensor, knScales.tensor,
            topKIndcies.tensor, blockTable.tensor, actSeqs.tensor, nq, nkv, softmaxScale, topk, blockSize, maxBlockNumPerBatch, saOut, tileConfig
        );
        // 读数据
        int saOutSize = std::accumulate(saOutShape.begin(), saOutShape.end(), 1, std::multiplies<>());
        std::vector<T> golden(saOutSize, 0);
        readInput(GetGoldenDir() + "/atten_out.bin", golden);

        ProgramData::GetInstance().AppendInputs({
            qNope.dataPtr, qRope.dataPtr, kNope2D.dataPtr, kRope2D.dataPtr, 
            knScales.dataPtr, topKIndcies.dataPtr, blockTable.dataPtr, actSeqs.dataPtr
        });
        ProgramData::GetInstance().AppendOutputs({
            saOutData
        });
        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(golden, (T *)outs->data(), 0.0005f));
    } else {
        // kn int8
        auto kNope2D = CreateTensorAndData<int8_t>(knShape, DT_INT8, "kn", "/k_nope.bin", {0});
        SelectedAttentionV2(
            qNope.tensor, qRope.tensor, kNope2D.tensor, kRope2D.tensor, knScales.tensor,
            topKIndcies.tensor, blockTable.tensor, actSeqs.tensor, nq, nkv, softmaxScale, topk, blockSize, maxBlockNumPerBatch, saOut, tileConfig
        );
        int saOutSize = std::accumulate(saOutShape.begin(), saOutShape.end(), 1, std::multiplies<>());
        std::vector<T> golden(saOutSize, 0);
        readInput(GetGoldenDir() + "/atten_out.bin", golden);

        ProgramData::GetInstance().AppendInputs({
            qNope.dataPtr, qRope.dataPtr, kNope2D.dataPtr, kRope2D.dataPtr, 
            knScales.dataPtr, topKIndcies.dataPtr, blockTable.dataPtr, actSeqs.dataPtr
        });
        ProgramData::GetInstance().AppendOutputs({
            saOutData
        });
        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(golden, (T *)outs->data(), 0.0005f));
    }
}

SaTileShapeConfig GetDefaultSaTileShapeConfig(const int gTile, const int sTile) {
    SaTileShapeConfig tileConfig;
    tileConfig.gTile = gTile; // for gLoop split
    tileConfig.sKvTile = sTile; // for s2Loop split
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {64, 128}; // (n1, d)
    return tileConfig;
}

SaTileShapeConfig GetPerfSaTileShapeConfig(const int gTile, const int sTile) {
    SaTileShapeConfig tileConfig;
    tileConfig.gTile = gTile; // for gLoop split
    tileConfig.sKvTile = sTile; // for s2Loop split
    tileConfig.c1TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {8, 2048}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {64, 128}; // (n1, d)
    return tileConfig;
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, SFA_b4_s2_seq64K_int8_perf) {

    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{});
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 1 * 1024 * 1024);
    config::SetPassOption(SG_PG_UPPER_BOUND, 20000);
    config::SetPassOption(SG_PG_LOWER_BOUND, 512);

    // config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 8}});

    // config::SetRuntimeOption<uint8_t>(
    //     DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
    //                         static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));
    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 128);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 128);

    config::SetPassOption(SG_PARALLEL_NUM, 20);
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{{-1, 2}});

    SaTileShapeConfig tileConfig = GetPerfSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b4_s2_seqTest1_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b32_s1_seq511) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b32_s1_seq511_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b1_s1_seq2049) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b1_s3_seq2047) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b128_s1_seq8k) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b8_s1_seq128k) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b4_s1_seqTest1) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b8_s1_seqTest2) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b8_s4_seqTest2) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicGatherSlcFlashAttnDSASTest, dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}