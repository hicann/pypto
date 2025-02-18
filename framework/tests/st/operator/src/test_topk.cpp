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
 * \file test_topk.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
using namespace npu::tile_fwk;

class TopkOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

struct TopKParams {
    int32_t shape0;
    int32_t shape1;
    int32_t k;
    bool isLargest;
};

void TopKOnBoardFunc(TopKParams& params){
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int32_t shape0 = params.shape0;
    int32_t shape1 = params.shape1;
    int32_t k = params.k;
    bool isLargest = params.isLargest;

    uint64_t inputSize = shape0 * shape1 * sizeof(float);
    uint64_t outputSize = shape0 * k * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("TOPK") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputSize);
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        auto output = std::make_tuple(Tensor(DataType::DT_FP32, output_shape, out_ptr, "npu_val"),
                                      Tensor(DataType::DT_FP32, output_shape, out_ptr1, "resDics"));

        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)}) {
            output = TopK(input_a, k, -1, isLargest);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden_val(shape0 * k);
    std::vector<int32_t> golden_idx(shape0 * k);
    std::vector<float> dev_val(shape0 * k);
    std::vector<int32_t> dev_idx(shape0 * k);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_val.data(), (uint8_t *)out_ptr, outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_idx.data(), (uint8_t *)out_ptr1, outputSize);

    readInput(GetGoldenDir() + "/val.bin", golden_val);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);

    int ret_val = resultCmp(golden_val, dev_val, 0.001f);
    int ret_idx = resultCmp(golden_idx, dev_idx, 0);
    EXPECT_EQ(ret_val, true);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_128_32_32_topk) {
    TopKParams params;
    params.shape0 = 128;
    params.shape1 = 32;
    params.k = 32;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_128_32_16_topk) {
    TopKParams params;
    params.shape0 = 128;
    params.shape1 = 32;
    params.k = 16;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_4_32_8_topk) {
    TopKParams params;
    params.shape0 = 4;
    params.shape1 = 32;
    params.k = 8;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_2_16_8_topk) {
    TopKParams params;
    params.shape0 = 2;
    params.shape1 = 16;
    params.k = 8;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_2_8_4_topk) {
    TopKParams params;
    params.shape0 = 2;
    params.shape1 = 8;
    params.k = 4;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_1_8_4_topk) {
    TopKParams params;
    params.shape0 = 1;
    params.shape1 = 8;
    params.k = 4;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_2_288_15_topk) {
    TopKParams params;
    params.shape0 = 2;
    params.shape1 = 288;
    params.k = 15;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TopkOnBoardTest, test_operation_tensor_2_288_15_topk_reverse) {
    TopKParams params;
    params.shape0 = 2;
    params.shape1 = 288;
    params.k = 15;
    params.isLargest = false;
    TopKOnBoardFunc(params);
}
