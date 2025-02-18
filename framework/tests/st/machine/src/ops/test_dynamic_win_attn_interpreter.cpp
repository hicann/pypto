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
 * \file test_dynamic_win_atten.cpp
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
#include "operator/models/nsa/win_attention.h"
#include "test_dev_func_runner.h"


using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicWinAttenInterpreterTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};


constexpr int NUM_2 = 2;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;

template <typename T = npu::tile_fwk::float16>
void TestWinAttenInterpreter(WinAttenTileShapeConfig& tileConfig) {
    config::SetHostOption(ONLY_CODEGEN, true);

    SetInterpreterConfig();

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }


    int paramsSize = 9;
    std::vector<int> inputParam(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", inputParam);


    int b = inputParam[0];
    int sQ = inputParam[1];
    int nQ = inputParam[2];
    int nKV = inputParam[3];
    int sMax = inputParam[4];
    int dN = inputParam[5];
    int dR = inputParam[6];
    int blockSize = inputParam[7];
    int windowSize = inputParam[8];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dN + dR)));
    std::cout << "====input param==== " << std::endl;
    std::cout <<" b = " << b << " sQ = " << sQ << " nQ = " << nQ << " nKV = " << nKV << " sMax =" << sMax << " dN = " << dN
        << " dR = " << dR << " blockSize = " << blockSize << " windowSize = " << windowSize << std::endl;


    int maxBlock = (sMax + blockSize - 1) / blockSize;
    std::vector<int64_t> qNopeShape = {b * sQ * nQ, dN};
    std::vector<int64_t> qRopeShape = {b * sQ * nQ, dR};
    std::vector<int64_t> vNopeCacheShape = {b * maxBlock * blockSize , nKV * dN};
    std::vector<int64_t> kRopeCacheShape = {b * maxBlock * blockSize , nKV * dR};
    std::vector<int64_t> attentionOutShape = {b ,sQ ,nQ, dN};
    std::vector<int64_t> blockTableShape = {b, maxBlock};

    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor vNopeCache(dType, vNopeCacheShape, "vNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "kRopeCache");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor attentionOut(DT_FP32, attentionOutShape, "attentionOut");

    // 读数据
    int qNopeSize = std::accumulate(qNopeShape.begin(), qNopeShape.end(), 1, std::multiplies<>());
    int qRopeSize = std::accumulate(qRopeShape.begin(), qRopeShape.end(), 1, std::multiplies<>());
    int vNopeCacheSize = std::accumulate(vNopeCacheShape.begin(), vNopeCacheShape.end(), 1, std::multiplies<>());
    int kRopeCacheSize = std::accumulate(kRopeCacheShape.begin(), kRopeCacheShape.end(), 1, std::multiplies<>());
    int blockTableSize = std::accumulate(blockTableShape.begin(), blockTableShape.end(), 1, std::multiplies<>());
    int winAttenOutSize = std::accumulate(attentionOutShape.begin(), attentionOutShape.end(), 1, std::multiplies<>());

    std::vector<int> seq(b);
    std::vector<T> qNopeData(qNopeSize, 0);
    std::vector<T> qRopeData(qRopeSize, 0);
    std::vector<T> vNopeCacheData(vNopeCacheSize, 0);
    std::vector<T> kRopeCacheData(kRopeCacheSize, 0);
    std::vector<int> blockTableData(blockTableSize, 0);

    readInput<int>(GetGoldenDir() + "/actual_seq_list.bin", seq);
    readInput<T>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<T>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    readInput<T>(GetGoldenDir() + "/k_cache_nope.bin", vNopeCacheData);
    readInput<T>(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheData);
    readInput<int>(GetGoldenDir() + "/block_table.bin", blockTableData);

    std::vector<float> golden(winAttenOutSize, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(qNope, qNopeData),
        RawTensorData::CreateTensor<T>(vNopeCache, vNopeCacheData),
        RawTensorData::CreateTensor<T>(qRope, qRopeData),
        RawTensorData::CreateTensor<T>(kRopeCache, kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(attentionOut, 0),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(attentionOut, golden),
    });

    WinAttentionDebug(qNope, vNopeCache, qRope, kRopeCache, nQ, nKV, blockTable, actSeqs, windowSize,
        blockSize, softmaxScale, attentionOut, tileConfig);
}

TEST_F(DynamicWinAttenInterpreterTest, test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_fp16_inter) {
    WinAttenTileShapeConfig tileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    tileConfig.gTile = gTileSize;
    tileConfig.vNopeTileShape = {NUM_16, NUM_256};
    tileConfig.vRopeTileShape = {NUM_128, NUM_64};
    tileConfig.outTileShape = {NUM_16, NUM_256};
    tileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (s2Tile, dN+dR) -> (n1, s2Tile)
    tileConfig.v1TileShape = {NUM_32, NUM_128}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTileSize, gTileSize, NUM_128, NUM_128, NUM_128, NUM_128}; // (n1, s2Tile) @ (s2Tile, dN) -> (n1, d)
    tileConfig.v2TileShape = {NUM_16, NUM_256}; // (n1, d)
    // WinConfig config;
    TestWinAttenInterpreter<npu::tile_fwk::float16>(tileConfig);
}