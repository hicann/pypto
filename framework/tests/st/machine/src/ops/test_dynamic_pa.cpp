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
 * \file test_dynamic_pa.cpp
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
#include "operator/models/deepseek/page_attention.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicPATest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

static void readBlockTableFromFile(const std::string& filename, int rows, int cols, std::vector<std::vector<int>> & blockTable) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        inFile.read(reinterpret_cast<char*>(blockTable[i].data()), cols * sizeof(int));
    }

    inFile.close();
    return;
}

struct PaConfig {
    bool manualUnroll{false};
    int maxUnrollTimes{1};
    bool onlyBatchLoop{false};
    bool isNzFormat{false};
    bool isImmediateSymScalar{false};
};

void testPa(PaTileShapeConfig& tileConfig, PaConfig config) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);

    std::vector<uint8_t> devProgBinary;
    int paramsSize = 8;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);

    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    std::vector<int> seq(b);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    std::vector<std::vector<int>> blockTableVector(b, std::vector<int>(maxBlockNumPerBatch, 0));

    TileOpFormat kvFormat = config.isNzFormat ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache", kvFormat);
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache", kvFormat);
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope", kvFormat);
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> qNopeData(b * nq * sq * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> qRopeData(b * nq * sq * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> kNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> kRopeCacheData(blockNum * blockSize * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> vNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<int32_t> blockTableData(b * maxBlockNumPerBatch, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    if (config.isNzFormat) {
        readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope_nz.bin", kNopeCacheData);
        readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope_nz.bin", kRopeCacheData);
        readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache_nz.bin", vNopeCacheData);
    } else {
        readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheData);
        readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheData);
        readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache.bin", vNopeCacheData);
    }
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", blockTableData);

    readBlockTableFromFile(GetGoldenDir() + "/block_table.bin", b, maxBlockNumPerBatch, blockTableVector);
    std::vector<float> golden(b * sq * nq * dn, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);

    if (!config.isImmediateSymScalar) {
        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNopeCache, kNopeCacheData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(vNopeCache, vNopeCacheData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRopeCache, kRopeCacheData),

            RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
            RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
        });
    } else {
        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNopeCache, kNopeCacheData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(vNopeCache, vNopeCacheData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
            RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRopeCache, kRopeCacheData),
            });
    }

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(paOut, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(paOut, golden),
    });

    if (config.onlyBatchLoop) {
        PageAttentionHighThroughput(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
            tileConfig, config.maxUnrollTimes);
    } else {
         if (!config.manualUnroll) {
            if (!config.isImmediateSymScalar) {
                PageAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
                    tileConfig, config.maxUnrollTimes, config.isNzFormat);
            } else {
                PageAttentionWithImmScalar(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTableVector/*vector*/, seq/*vector*/, blockSize, softmaxScale, paOut,
                    tileConfig, config.maxUnrollTimes, config.isNzFormat);
            }
        } else {
            PageAttentionWithManualUnroll(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
                tileConfig, config.maxUnrollTimes);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.0005f));
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    PaConfig config;
    config.isNzFormat = true;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_imm_scalar) {
    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 256;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    PaConfig config;
    config.isNzFormat = true;
    config.isImmediateSymScalar = true;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_unroll) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    PaConfig config;
    config.maxUnrollTimes = 4;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_manual_unroll) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    PaConfig config;
    config.manualUnroll = true;
    config.maxUnrollTimes = 4;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_dyn_valid_shape) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    PaConfig config;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_high_throughput_dview_large) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    PaConfig config;
    config.isNzFormat = true;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_high_throughput_only_batch_loop) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    PaConfig config;
    config.maxUnrollTimes = 4;
    config.onlyBatchLoop = true;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_high_throughput_dview_large_dyn_valid_shape) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    PaConfig config;
    config.maxUnrollTimes = 4;
    config.onlyBatchLoop = true;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_noflash_unalign) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 128};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize}; // n, D, S2
    tileConfig.v1TileShape = {nTile, 128};
    tileConfig.c2TileShape = {nTile, nTile, 128, 128, blockSize, blockSize}; // n, S2, D
    tileConfig.v2TileShape = {nTile, 128};
    PaConfig config;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_noflash) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 128};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize}; // n, D, S2
    tileConfig.v1TileShape = {nTile, 128};
    tileConfig.c2TileShape = {nTile, nTile, 128, 128, blockSize, blockSize}; // n, S2, D
    tileConfig.v2TileShape = {nTile, 128};
    PaConfig config;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_low_lantency_dyn_unalign) {
    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    const int blockSize = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 128};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize}; // n, D, S2
    tileConfig.v1TileShape = {nTile, 128};
    tileConfig.c2TileShape = {nTile, nTile, 128, 128, blockSize, blockSize}; // n, S2, D
    tileConfig.v2TileShape = {nTile, 128};
    PaConfig config;
    testPa(tileConfig, config);
}

TEST_F(DynamicPATest, dynamic_pa_high_throughput_dview_large_dyn_unalign) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    PaConfig config;
    testPa(tileConfig, config);
}
