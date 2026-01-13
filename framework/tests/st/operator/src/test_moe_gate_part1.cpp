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
 * \file test_moe_gate_part1.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class MoEGatePart1OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(MoEGatePart1OnBoardTest, test_moe_gate_part1) {
    aclInit(nullptr);
    constexpr int32_t H = 7168;
    constexpr int32_t S = 1;
    constexpr int32_t B = 16;
    constexpr int32_t nRoutedExperts = 256;

    rtSetDevice(GetDeviceIdByEnvVar());

    // set input data
    uint64_t input_e_score_bias_size = nRoutedExperts;
    uint64_t input_hidden_state_size = B * S * H;
    uint64_t input_weight_size = nRoutedExperts * H;
    void *input_e_score_bias_ptr = readToDev<float>(GetGoldenDir() + "/input_e_score_bias.bin", input_e_score_bias_size);
    void *input_hidden_state_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/input_hidden_state.bin", input_hidden_state_size);
    void *input_weight_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/input_weight.bin", input_weight_size);
    assert(input_e_score_bias_ptr!=nullptr && input_hidden_state_ptr!=nullptr && input_weight_ptr!=nullptr);
    // set output data
    uint64_t outputSize = B * S * nRoutedExperts;
    uint64_t outputCapacity = B * S * nRoutedExperts * sizeof(float);

    uint8_t* out_score_ptr = allocDevAddr(outputCapacity);
    uint8_t* out_score4choice_ptr = allocDevAddr(outputCapacity);
    assert(out_score_ptr!=nullptr && out_score4choice_ptr!=nullptr);

    // set prepare shape
    std::vector<int64_t> input_e_score_bias_shape = {1, nRoutedExperts};
    std::vector<int64_t> input_hidden_state_shape = {B, S, H};
    std::vector<int64_t> input_weight_shape = {nRoutedExperts, H};
    std::vector<int64_t> output_shape = {B*S, nRoutedExperts};

    // process
    PROGRAM("MOE_GATE_PART1") {
        config::Reset();
        TileShape::Current().SetCubeTile({16, 16}, {512, 512}, {64, 64});
        TileShape::Current().SetVecTile({16, 64});
        Tensor input_e_score_bias(DataType::DT_FP32, input_e_score_bias_shape, (uint8_t *)input_e_score_bias_ptr, "InputEScoresCorrectionBias");
        Tensor input_hidden_state(DataType::DT_FP16, input_hidden_state_shape, (uint8_t *)input_hidden_state_ptr, "InputHiddenState");
        Tensor input_weight(DataType::DT_FP16, input_weight_shape, (uint8_t *)input_weight_ptr, "InputWeight");
        Tensor output_score(DataType::DT_FP32, output_shape, out_score_ptr, "OutputScore");
        Tensor output_score4choice(DataType::DT_FP32, output_shape, out_score4choice_ptr, "OutputScore4Choice");

        config::SetBuildStatic(true);
        FUNCTION("MOE_GATE_Part1_1", {input_e_score_bias, input_hidden_state, input_weight, output_score, output_score4choice}) {
            Tensor input_hidden_state_reshape = Reshape(input_hidden_state, {B*S, H});
            Tensor logits = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP32, input_hidden_state_reshape, input_weight, false, true);
            output_score = Sigmoid(logits);
            output_score4choice = Add(output_score, input_e_score_bias);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    // test with golden data
    std::vector<float> res_scores(outputSize);
    std::vector<float> res_scores4choice(outputSize);
    std::vector<float> golden_scores(outputSize);
    std::vector<float> golden_scores4choice(outputSize);

    machine::GetRA()->CopyFromTensor((uint8_t *)res_scores.data(), out_score_ptr, outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res_scores4choice.data(), out_score4choice_ptr, outputCapacity);

    readInput(GetGoldenDir() + "/output_score.bin", golden_scores);
    readInput(GetGoldenDir() + "/output_score4choice.bin", golden_scores4choice);

    int ret_sroce = resultCmp(golden_scores, res_scores, 0.003f);
    int ret_score4choice = resultCmp(golden_scores4choice, res_scores4choice, 0.003f);
    EXPECT_EQ(ret_sroce, true);
    EXPECT_EQ(ret_score4choice, true);
};
