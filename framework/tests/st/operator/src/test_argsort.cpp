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
 * \file test_argsort.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
using namespace npu::tile_fwk;

class ArgsortOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(ArgsortOnBoardTest, test_operation_tensor_128_32_argsort) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 128;
    constexpr int32_t shape1 = 32;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0 * shape1 * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, shape1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0 * shape1);
    std::vector<int32_t> res2(shape0 * shape1);
    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(ArgsortOnBoardTest, test_operation_tensor_4_32_argsort) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 4;
    constexpr int32_t shape1 = 32;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0 * shape1 * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, shape1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0 * shape1);
    std::vector<int32_t> res2(shape0 * shape1);
    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(ArgsortOnBoardTest, test_operation_tensor_2_16_argsort) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 2;
    constexpr int32_t shape1 = 16;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0 * shape1 * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, shape1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0 * shape1);
    std::vector<int32_t> res2(shape0 * shape1);

    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(ArgsortOnBoardTest, test_operation_tensor_32_argsort) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 32;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0  * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0};
        std::vector<int64_t> output_shape = {shape0};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0);
    std::vector<int32_t> res2(shape0);

    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(ArgsortOnBoardTest, test_operation_tensor_64_argsort) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 64;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0  * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0};
        std::vector<int64_t> output_shape = {shape0};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0);
    std::vector<int32_t> res2(shape0);

    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(ArgsortOnBoardTest, test_operation_tensor_4_32__argsort) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 4;
    constexpr int32_t shape1 = 32;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0 * shape1 * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, shape1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1, false);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0 * shape1);
    std::vector<int32_t> res2(shape0 * shape1);
    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}

TEST_F(ArgsortOnBoardTest, test_operation_tensor_64__argsort_moe) {
    aclInit(nullptr);
    constexpr int32_t shape0 = 64;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = shape0  * sizeof(float);
    uint8_t* out_ptr1 = allocDevAddr(outputSize);
    PROGRAM("ARGSORT") {
        std::vector<int64_t> input_shape = {shape0};
        std::vector<int64_t> output_shape = {shape0};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", outputSize);
        TileShape::Current().SetVecTile({shape0});
        Tensor input_a(DataType::DT_FP32, input_shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, output_shape, out_ptr1, "resDics");
        config::SetBuildStatic(true);
        FUNCTION("ARGSORT_T", {input_a, output}) {
            output = ArgSort(input_a, -1, false);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden_idx(shape0);
    std::vector<int32_t> res2(shape0);

    machine::GetRA()->CopyFromTensor((uint8_t *)res2.data(), (uint8_t *)out_ptr1, outputSize);
    readInput(GetGoldenDir() + "/idx.bin", golden_idx);
    int ret_idx = resultCmp(golden_idx, res2, 0);
    EXPECT_EQ(ret_idx, true);
}
