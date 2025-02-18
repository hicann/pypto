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
 * \file test_moe_gate_part2.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class MoEGatePart2OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(MoEGatePart2OnBoardTest, test_operation_b_2) {
    aclInit(nullptr);
    constexpr int32_t nRoutedExperts = 256;
    constexpr int32_t nGroup = 8;
    constexpr int32_t topkGroup = 4;
    constexpr int32_t S = 1;
    constexpr int32_t B = 2;

    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t inputSize = B * nRoutedExperts * sizeof(float);
    uint64_t outputSize = B * topkGroup * sizeof(float);
    uint8_t* out_group_idx_ptr = allocDevAddr(outputSize);
    uint8_t* out_group_mask_ptr = allocDevAddr(outputSize);
    PROGRAM("MOE_GATE_PART2") {
        std::vector<int64_t> input_shape = {B, nRoutedExperts};
        std::vector<int64_t> input_reshape = {B * nGroup, 32};
        std::vector<int64_t> output_idx_shape = {B, topkGroup};
        std::vector<int64_t> output_mask_shape = {B, nGroup};

        void *scores_for_choice_ptr = readToDev(GetGoldenDir() + "/scores_for_choice.bin", inputSize);
        TileShape::Current().SetVecTile({16, 32});
        Tensor input_scores_for_choice(DataType::DT_FP32, input_shape,
                                    (uint8_t *)scores_for_choice_ptr, "scores_for_choice");
        Tensor output_group_idx(DataType::DT_FP32, output_idx_shape, (uint8_t *)out_group_idx_ptr, "group_idx");
        Tensor output_group_mask(DataType::DT_FP32, output_mask_shape, (uint8_t *)out_group_mask_ptr, "group_mask");
        config::SetBuildStatic(true);
        FUNCTION("MOE_GATE_PART2_T", {input_scores_for_choice, output_group_idx, output_group_mask}) {
            Tensor scores_for_choice_reshape;
            scores_for_choice_reshape = Reshape(input_scores_for_choice, input_reshape);
            auto output_topk2 = TopK(scores_for_choice_reshape, 2, -1);
            auto sum_result = Sum(std::get<0>(output_topk2), -1, true);
            sum_result = Reshape(sum_result, {B*S, nGroup});
            auto output_topk4 = TopK(sum_result, 4, -1);
            output_group_idx = std::get<1>(output_topk4);
            output_group_mask = Mul(std::get<0>(output_topk4), Element(DataType::DT_FP32, 0.0));
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<int32_t> golden_idx(B * topkGroup);
    std::vector<int32_t> golden_mask(B * topkGroup);
    std::vector<int32_t> dev_idx(B * topkGroup);
    std::vector<int32_t> dev_mask(B * topkGroup);
    printf("out_group_idx_ptr is %p\n", out_group_idx_ptr);
    printf("out_group_mask_ptr is %p\n", out_group_mask_ptr);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_idx.data(), (uint8_t *)out_group_idx_ptr, outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_mask.data(), (uint8_t *)out_group_mask_ptr, outputSize);

    // 真值比对
    readInput(GetGoldenDir() + "/group_idx.bin", golden_idx);
    readInput(GetGoldenDir() + "/group_mask.bin", golden_mask);

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < topkGroup; j++) {
            EXPECT_EQ(*((int32_t*)dev_idx.data() + i * topkGroup + j),
                    *((int32_t*)golden_idx.data() + i * topkGroup + j));
        }
    }
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < topkGroup; j++) {
            EXPECT_EQ(*((int32_t*)dev_mask.data() + i * topkGroup + j),
                    *((int32_t*)golden_mask.data() + i * topkGroup + j));
        }
    }
}

TEST_F(MoEGatePart2OnBoardTest, test_operation_b_1024) {
    aclInit(nullptr);
    constexpr int32_t nRoutedExperts = 256;
    constexpr int32_t nGroup = 8;
    constexpr int32_t topkGroup = 4;
    constexpr int32_t S = 1;
    constexpr int32_t B = 1024;

    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t inputSize = B * nRoutedExperts * sizeof(float);
    uint64_t outputSize = B * topkGroup * sizeof(float);
    uint8_t* out_group_idx_ptr = allocDevAddr(outputSize);
    uint8_t* out_group_mask_ptr = allocDevAddr(outputSize);
    PROGRAM("MOE_GATE_PART2") {
        std::vector<int64_t> input_shape = {B, nRoutedExperts};
        std::vector<int64_t> input_reshape = {B * nGroup, 32};
        std::vector<int64_t> output_idx_shape = {B, topkGroup};
        std::vector<int64_t> output_mask_shape = {B, nGroup};

        void *scores_for_choice_ptr = readToDev(GetGoldenDir() + "/scores_for_choice.bin", inputSize);
        TileShape::Current().SetVecTile({16, 32});
        Tensor input_scores_for_choice(DataType::DT_FP32, input_shape,
                                    (uint8_t *)scores_for_choice_ptr, "scores_for_choice");
        Tensor output_group_idx(DataType::DT_FP32, output_idx_shape, (uint8_t *)out_group_idx_ptr, "group_idx");
        Tensor output_group_mask(DataType::DT_FP32, output_mask_shape, (uint8_t *)out_group_mask_ptr, "group_mask");
        config::SetBuildStatic(true);
        FUNCTION("MOE_GATE_PART2_T", {input_scores_for_choice, output_group_idx, output_group_mask}) {
            Tensor scores_for_choice_reshape;
            scores_for_choice_reshape = Reshape(input_scores_for_choice, input_reshape);
            auto output_topk2 = TopK(scores_for_choice_reshape, 2, -1);
            auto sum_result = Sum(std::get<0>(output_topk2), -1, true);
            sum_result = Reshape(sum_result, {B*S, nGroup});
            auto output_topk4 = TopK(sum_result, 4, -1);
            output_group_idx = std::get<1>(output_topk4);
            output_group_mask = Mul(std::get<0>(output_topk4), Element(DataType::DT_FP32, 0.0));
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<int32_t> golden_idx(B * topkGroup);
    std::vector<int32_t> golden_mask(B * topkGroup);
    std::vector<int32_t> dev_idx(B * topkGroup);
    std::vector<int32_t> dev_mask(B * topkGroup);
    printf("out_group_idx_ptr is %p\n", out_group_idx_ptr);
    printf("out_group_mask_ptr is %p\n", out_group_mask_ptr);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_idx.data(), (uint8_t *)out_group_idx_ptr, outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_mask.data(), (uint8_t *)out_group_mask_ptr, outputSize);

    // 真值比对
    readInput(GetGoldenDir() + "/group_idx.bin", golden_idx);
    readInput(GetGoldenDir() + "/group_mask.bin", golden_mask);

    for (int i = 0; i < B; i++) {
        for (int j = 0; j < topkGroup; j++) {
            EXPECT_EQ(*((int32_t*)dev_idx.data() + i * topkGroup + j),
                    *((int32_t*)golden_idx.data() + i * topkGroup + j));
        }
    }
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < topkGroup; j++) {
            EXPECT_EQ(*((int32_t*)dev_mask.data() + i * topkGroup + j),
                    *((int32_t*)golden_mask.data() + i * topkGroup + j));
        }
    }
}
