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
 * \file test_llama.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/llama/llama_def.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

#define LLMA_CYCLE_THRESHOLD 8192
#define LLMA_L1REUSE_THRESHOLD 4

class LLamaLayerTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        aclInit(nullptr);
        rtSetDevice(GetDeviceIdByEnvVar());
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, false);
        config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
        config::SetPassOption(VEC_NBUFFER_MODE, 1);
        config::SetPassOption(CUBE_L1_REUSE_MODE, LLMA_L1REUSE_THRESHOLD);
        config::SetPassOption(SG_PG_LOWER_BOUND, LLMA_CYCLE_THRESHOLD);
        rtSetDevice(GetDeviceIdByEnvVar());
    }
};

void RunLLamaLayer(const AttentionDims &dimsCfg, float threadhold = 0.001f) {
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;
    int size0 = b * n * s * d;
    int size1 = n * d * n * d;
    vector<float> golden(size0);
    string basepath = GetGoldenDir();
    void *h_ptr = readToDev(basepath + "/hiddenStates.bin", size0);
    void *aw_ptr = readToDev<uint16_t>(basepath + "/attnWeight.bin", size1 * 3);
    void *dw_ptr = readToDev<uint16_t>(basepath + "/denseWeight.bin", size1);
    void *fw_ptr = readToDev<uint16_t>(basepath + "/ffnWeight.bin", size1 * 3);
    readInput(basepath + "/llama_layer_golden_res.bin", golden);
    uint64_t outputSize = size0 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("LLAMALAYER") {
        Tensor H(DataType::DT_FP32, {b * s, n * d}, (uint8_t *)h_ptr, "H");
        Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, (uint8_t *)aw_ptr, "AW");
        Tensor DW(DataType::DT_FP16, {n * d, n * d}, (uint8_t *)dw_ptr, "DW");
        Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, (uint8_t *)fw_ptr, "FW");
        Tensor Res(DataType::DT_FP32, {b * s, n * d},  out_ptr, "Res");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("LLAMA", {H, AW, DW, FW, Res}) {
            Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout << std::hex << "addr----" << (uint64_t)out_ptr << std::endl;
    std::vector<float> res(size0);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    int ret = resultCmp(golden, res, threadhold);
    EXPECT_EQ(ret, true);
}

TEST_F(LLamaLayerTest, llama_1_1_128_128)
{
    AttentionDims dimsCfg = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayer(dimsCfg, 0.005f);
}

TEST_F(LLamaLayerTest, llama_1_1_256_128)
{
    AttentionDims dimsCfg = {1, 1, 256, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayer(dimsCfg, 0.005f);
}

TEST_F(LLamaLayerTest, llama_1_1_512_128)
{
    AttentionDims dimsCfg = {1, 1, 512, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayer(dimsCfg, 0.005f);
}

TEST_F(LLamaLayerTest, llama_1_1_1024_128)
{
    AttentionDims dimsCfg = {1, 1, 1024, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayer(dimsCfg, 0.005f);
}
TEST_F(LLamaLayerTest, llama_1_1_4096_128)
{
    AttentionDims dimsCfg = {1, 1, 4096, 128, DFT_SINGLE_M, 1024};
    RunLLamaLayer(dimsCfg, 0.005f);
}
