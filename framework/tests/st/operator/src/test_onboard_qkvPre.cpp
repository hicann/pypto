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
 * \file test_onboard_qkvPre.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class QkvPreOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T = npu::tile_fwk::float16>
void TestQkvPre(std::vector<int> &params, string dataPath) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    int b = params[0];
    int s = params[1];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DataType::DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DataType::DT_FP16;
    }
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DataType::DT_BF16;
    }

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> w_qa_shape = {h, qLoraRank};
    std::vector<int64_t> w_qb_shape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> q_shape = {b, s, n, q_head_dim};
    std::vector<int64_t> kv_shape = {b, s, kvLoraRank + qkRopeHeadDim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int capacity_w_qa = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int capacity_w_qb = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_a = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int capacity_q = std::accumulate(q_shape.begin(), q_shape.end(), 1, std::multiplies<>());
    int capacity_kv = std::accumulate(kv_shape.begin(), kv_shape.end(), 1, std::multiplies<>());

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q * sizeof(T);
    uint64_t outputSize1 = capacity_kv * sizeof(T);
    uint8_t* q_out_ptr = allocDevAddr(outputSize0);
    uint8_t* kv_out_ptr = allocDevAddr(outputSize1);


    ConfigManager::Instance();
    PROGRAM("QkvPre") {

        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
        void *w_qa_ptr = readToDev<T>(dataPath + "/w_qa.bin", capacity_w_qa);
        void *w_qb_ptr = readToDev<T>(dataPath + "/w_qb.bin", capacity_w_qb);
        void *w_kv_a_ptr = readToDev<T>(dataPath + "/w_kv_a.bin", capacity_w_kv_a);

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        Tensor w_qa(dType, w_qa_shape, (uint8_t *)w_qa_ptr, "w_qa");
        Tensor w_qb(dType, w_qb_shape, (uint8_t *)w_qb_ptr, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, (uint8_t *)w_kv_a_ptr, "w_kv_a");
        Tensor output_q(dType, q_shape, q_out_ptr, "output_q");
        Tensor output_kv(dType, kv_shape, kv_out_ptr, "output_kv");

        AttentionW aw;
        aw.qAProjW = w_qa;
        aw.qBProjW = w_qb;
        aw.kvAProjWithMqaW = w_kv_a;
        Tensor kvBProjWK;  // not used in qkvPre
        Tensor kvBProjWV;  // not used in qkvPre
        Tensor oProjW;       // not used in qkvPre
        aw.kvBProjWK = kvBProjWK;
        aw.kvBProjWV = kvBProjWV;
        aw.oProjW = oProjW;

        std::tuple<Tensor, Tensor> res;
        DeepseekAttention Attention(g_deepseekConfig, aw, 1);

        config::SetBuildStatic(true);
        FUNCTION("QkvPre_T", {x, w_qa, w_qb, w_kv_a, output_q, output_kv}) {
            auto q_kv = Attention.QkvPre2(x);
            output_q = q_kv[0];
            output_kv = q_kv[1];
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<T> q_golden(capacity_q);
    std::vector<T> q_npu(capacity_q);
    std::vector<T> kv_golden(capacity_kv);
    std::vector<T> kv_npu(capacity_kv);
    readInput<T>(dataPath + "/q_golden.bin", q_golden);
    readInput<T>(dataPath + "/kv_golden.bin", kv_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)kv_out_ptr, outputSize1);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<T>(q_golden, q_npu, 0.005f);
    EXPECT_EQ(ret0, true);
    std::cout << "\n====== resultCmp: output kv start" << std::endl;
    int ret1 = resultCmp<T>(kv_golden, kv_npu, 0.003f);
    EXPECT_EQ(ret1, true);
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_4_2_1_256_256_512) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;  //
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_32_2_1_256_256_512) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_bfloat16_32_2_1_256_256_512) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_32_32_1_256_256_512) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 32;
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_32_32_1_256_256_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_32_32_1_256_1024_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 1024;  //
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_32_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_bfloat16_32_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_4_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_bfloat16_4_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPre<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

template <typename T = npu::tile_fwk::float16>
void TestQkvPreFp32(std::vector<int> &params, string dataPath) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    int b = params[0];
    int s = params[1];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DataType::DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DataType::DT_FP16;
    }
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DataType::DT_BF16;
    }

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> w_qa_shape = {h, qLoraRank};
    std::vector<int64_t> w_qb_shape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> q_shape = {b, s, n, q_head_dim};
    std::vector<int64_t> kv_shape = {b, s, kvLoraRank + qkRopeHeadDim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int capacity_w_qa = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int capacity_w_qb = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_a = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int capacity_q = std::accumulate(q_shape.begin(), q_shape.end(), 1, std::multiplies<>());
    int capacity_kv = std::accumulate(kv_shape.begin(), kv_shape.end(), 1, std::multiplies<>());

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q * sizeof(float);
    uint64_t outputSize1 = capacity_kv * sizeof(float);
    uint8_t* q_out_ptr = allocDevAddr(outputSize0);
    uint8_t* kv_out_ptr = allocDevAddr(outputSize1);


    ConfigManager::Instance();
    PROGRAM("QkvPreFp32") {
        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
        void *w_qa_ptr = readToDev<T>(dataPath + "/w_qa.bin", capacity_w_qa);
        void *w_qb_ptr = readToDev<T>(dataPath + "/w_qb.bin", capacity_w_qb);
        void *w_kv_a_ptr = readToDev<T>(dataPath + "/w_kv_a.bin", capacity_w_kv_a);

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        Tensor w_qa(dType, w_qa_shape, (uint8_t *)w_qa_ptr, "w_qa");
        Tensor w_qb(dType, w_qb_shape, (uint8_t *)w_qb_ptr, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, (uint8_t *)w_kv_a_ptr, "w_kv_a");
        Tensor output_q(DataType::DT_FP32, q_shape, q_out_ptr, "output_q");
        Tensor output_kv(DataType::DT_FP32, kv_shape, kv_out_ptr, "output_kv");

        AttentionW aw;
        aw.qAProjW = w_qa;
        aw.qBProjW = w_qb;
        aw.kvAProjWithMqaW = w_kv_a;
        Tensor kvBProjWK;  // not used in qkvPre
        Tensor kvBProjWV;  // not used in qkvPre
        Tensor oProjW;       // not used in qkvPre
        aw.kvBProjWK = kvBProjWK;
        aw.kvBProjWV = kvBProjWV;
        aw.oProjW = oProjW;

        std::tuple<Tensor, Tensor> res;
        DeepseekAttention Attention(g_deepseekConfig, aw, 1);

        config::SetBuildStatic(true);
        FUNCTION("QkvPreFp32_T", {x, w_qa, w_qb, w_kv_a, output_q, output_kv}) {
            auto q_kv = Attention.QkvPreFp32(x);
            output_q = std::get<0>(q_kv);
            output_kv = std::get<1>(q_kv);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> q_golden(capacity_q);
    std::vector<float> q_npu(capacity_q);
    std::vector<float> kv_golden(capacity_kv);
    std::vector<float> kv_npu(capacity_kv);
    readInput<float>(dataPath + "/q_golden.bin", q_golden);
    readInput<float>(dataPath + "/kv_golden.bin", kv_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)kv_out_ptr, outputSize1);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<float>(q_golden, q_npu, 0.005f);
    EXPECT_EQ(ret0, true);
    std::cout << "\n====== resultCmp: output kv start" << std::endl;
    int ret1 = resultCmp<float>(kv_golden, kv_npu, 0.003f);
    EXPECT_EQ(ret1, true);
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_float16_32_2_1_256_256_512_fp32) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPreFp32<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_bfloat16_32_2_1_256_256_512_fp32) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPreFp32<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(QkvPreOnBoardTest, test_qkvPre_bfloat16_32_32_1_256_7168_1536_fp32) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestQkvPreFp32<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}
