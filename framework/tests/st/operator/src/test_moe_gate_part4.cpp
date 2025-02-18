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
 * \file test_moe_gate_part4.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

constexpr double DF_1E_20 = 1e-20f;

class MoEPart4OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(MoEPart4OnBoardTest, test_operation_b_2) {
    aclInit(nullptr);
    constexpr int32_t nRoutedExperts = 256;
    constexpr int32_t numExpertsPerTopk = 8;
    constexpr int32_t S = 1;
    constexpr int32_t B = 2;

    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = B * S * numExpertsPerTopk * sizeof(float);//B * topk_group_align * sizeof(float) * 100;
    uint64_t inputSize = B * S * nRoutedExperts* sizeof(float);
    uint8_t* out_topk_weight = allocDevAddr(outputSize);
    // uint8_t* out_group_mask_ptr = allocDevAddr(outputSize);
    PROGRAM("MOE_GATE_PART4") {
        std::vector<int64_t> input_shape = {B * S, nRoutedExperts};
        std::vector<int64_t> output_shape = {B * S, numExpertsPerTopk};
        void *input_score = readToDev(GetGoldenDir() + "/input_score.bin", inputSize);
        void *input_tmp_score = readToDev(GetGoldenDir() + "/input_tmp_score.bin", inputSize);
        TileShape::Current().SetVecTile({16, 32});
        Tensor inputScores(DataType::DT_FP32, input_shape, (uint8_t *)input_score, "input_scores");
        Tensor inputTmpScores(DataType::DT_FP32, input_shape, (uint8_t *)input_tmp_score, "input_tmp_scores");
        Tensor outputTensor(DataType::DT_FP32, output_shape, out_topk_weight, "output_tensor");

        config::SetBuildStatic(true);
        FUNCTION("MOE_GATE_PART4_T", {inputScores, inputTmpScores, outputTensor}) {
            auto topk_idx = std::get<1>(TopK(inputScores, numExpertsPerTopk, -1)); // [b*s,256]->[b*s,8]
            auto topk_weight = GatherElements(inputTmpScores, topk_idx, 1); // [b*s,8]
            auto topk_weight_sum = Sum(topk_weight, 1, true);      // [b*s,8]->[b*s,1]
            auto denominator = Add(topk_weight_sum, Element(DataType::DT_FP32, DF_1E_20)); // [b*s,1]
            outputTensor = Div(topk_weight, denominator); // [b*s,numExpertsPerTok]
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden_val(B * S * numExpertsPerTopk);
    std::vector<float> dev_res(B * S * numExpertsPerTopk);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_topk_weight, outputSize);
    readInput(GetGoldenDir() + "/golden.bin", golden_val);

    resultCmp(golden_val, dev_res, 0.001f);
}
