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
 * \file test_onboard_concat.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class ConcatOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(ConcatOnBoardTest, test_concat_dim4_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 64, 64};
    std::vector<int64_t> resShape = {2, 2, 64, 128};
    DataType dtype = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    int resCap = cap * 2;
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        void *x_ptr = readToDev(GetGoldenDir() + "/concat_2_2_64_64_operand1.bin", cap);
        void *y_ptr = readToDev(GetGoldenDir() + "/concat_2_2_64_64_operand2.bin", cap);
        TileShape::Current().SetVecTile({1, 1, 32, 32});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, shape, (uint8_t *)y_ptr, "y");
        Tensor output(dtype, resShape, out_ptr, "res");

        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", {input_x, input_y, output}) {
            output = Cat(std::vector<Tensor>{input_x, input_y}, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_2_2_64_64_res.bin", golden);
    std::cout << "====== output size:" << dev_res.size() << std::endl;
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ConcatOnBoardTest, test_concat_exp_dim4_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 32, 32};
    std::vector<int64_t> resShape = {2, 2, 32, 64};
    DataType dtype = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    int resCap = cap * 2;
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        void *x_ptr = readToDev(GetGoldenDir() + "/concat_exp_2_2_32_32_operand1.bin", cap);
        void *y_ptr = readToDev(GetGoldenDir() + "/concat_exp_2_2_32_32_operand2.bin", cap);
        TileShape::Current().SetVecTile({2, 2, 32, 32});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, shape, (uint8_t *)y_ptr, "y");
        //Tensor output1(dtype, resShape, "z");
        Tensor output2(dtype, resShape, out_ptr, "res");

        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", {input_x, input_y, output2}) {
            Tensor output1 = Cat(std::vector<Tensor>{input_x, input_y}, -1);
            output2 = Exp(output1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_exp_2_2_32_32_res.bin", golden);
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ConcatOnBoardTest, test_exp_concat_dim4_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 32, 64};
    std::vector<int64_t> resShape = {2, 2, 32, 128};
    DataType dtype = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    int resCap = cap * 2;
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        void *x_ptr = readToDev(GetGoldenDir() + "/concat_exp_2_2_32_32_operand1.bin", cap);
        void *y_ptr = readToDev(GetGoldenDir() + "/concat_exp_2_2_32_32_operand2.bin", cap);
        TileShape::Current().SetVecTile({2, 2, 32, 32});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, shape, (uint8_t *)y_ptr, "y");
        //Tensor output1(dtype, resShape, "z");
        Tensor output2(dtype, resShape, out_ptr, "res");

        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", {input_x, input_y, output2}) {
            Tensor input_x_1 = Exp(input_x);
            Tensor input_y_1 = Exp(input_y);
            TileShape::Current().SetVecTile({2, 2, 16, 32});
            output2 = Cat(std::vector<Tensor>{input_x_1, input_y_1}, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_exp_2_2_32_32_res.bin", golden);
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ConcatOnBoardTest, test_concat_sqrt_dim4_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape1 = {2, 2, 32, 64};
    std::vector<int64_t> shape2 = {2, 2, 64, 64};
    std::vector<int64_t> resShape = {2, 2, 96, 64};
    DataType dtype = DataType::DT_FP32;
    int cap1 = shape1[0] * shape1[1] * shape1[2] * shape1[3];
    int cap2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];
    int resCap = resShape[0] * resShape[1] * resShape[2] * resShape[3];
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        void *x_ptr = readToDev(GetGoldenDir() + "/concat_sqrt_fp32_operand1.bin", cap1);
        void *y_ptr = readToDev(GetGoldenDir() + "/concat_sqrt_fp32_operand2.bin", cap2);
        TileShape::Current().SetVecTile({2, 2, 32, 32});
        Tensor input_x(dtype, shape1, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, shape2, (uint8_t *)y_ptr, "y");
        //Tensor output1(dtype, resShape, "z");
        Tensor output2(dtype, resShape, out_ptr, "res");

        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", {input_x, input_y, output2}) {
            Tensor output1 = Cat(std::vector<Tensor>{input_x, input_y}, 2);
            output2 = Sqrt(output1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_sqrt_fp32_res.bin", golden);
    std::cout << "====== output size:" << dev_res.size() << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}


TEST_F(ConcatOnBoardTest, test_concat_100_inputs_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 4, 16};
    std::vector<int64_t> resShape = {2, 2, 4, 1600};
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    int resCap = resShape[0] * resShape[1] * resShape[2] * resShape[3];
    DataType dtype = DataType::DT_FP32;
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        std::vector<Tensor> inputs;
        void *x_ptr = nullptr;
        for (int i = 0; i < 100; i++) {
            x_ptr = readToDev(GetGoldenDir() + "/concat_100_inputs_" + std::to_string(i) + "_fp32.bin", cap);
            Tensor input(dtype, shape, (uint8_t *)x_ptr, "input" + std::to_string(i));
            inputs.emplace_back(input);
        }
        TileShape::Current().SetVecTile({2, 2, 4, 16});
        Tensor output(dtype, resShape, out_ptr, "res");
        std::vector<std::reference_wrapper<const Tensor>> paras(inputs.begin(), inputs.end());
        paras.emplace_back(output);
        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", paras) {
            output = Cat(inputs, -1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_100_inputs_res_fp32.bin", golden);
    std::cout << "====== output size:" << dev_res.size() << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ConcatOnBoardTest, test_concat_128_inputs_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 1, 8, 8};
    std::vector<int64_t> resShape = {2, 1, 1024, 8};
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    int resCap = resShape[0] * resShape[1] * resShape[2] * resShape[3];
    DataType dtype = DataType::DT_FP32;
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        std::vector<Tensor> inputs;
        void *x_ptr = nullptr;
        for (int i = 0; i < 128; i++) {
            x_ptr = readToDev(GetGoldenDir() + "/concat_128_inputs_" + std::to_string(i) + "_fp32.bin", cap);
            Tensor input(dtype, shape, (uint8_t *)x_ptr, "input" + std::to_string(i));
            inputs.emplace_back(input);
        }
        TileShape::Current().SetVecTile({2, 1, 8, 8});
        Tensor output(dtype, resShape, out_ptr, "res");
        std::vector<std::reference_wrapper<const Tensor>> paras(inputs.begin(), inputs.end());
        paras.emplace_back(output);
        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", paras) {
            output = Cat(inputs, -2);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_128_inputs_res_fp32.bin", golden);
    std::cout << "====== output size:" << dev_res.size() << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ConcatOnBoardTest, test_concat_dim2_float32_moe) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape0 = {3, 7168};
    std::vector<int64_t> shape1 = {64, 7168};
    std::vector<int64_t> resShape = {67, 7168};
    DataType dtype = DataType::DT_FP32;
    int cap0 = shape0[0] * shape0[1];
    int cap1 = shape1[0] * shape1[1];
    int resCap = resShape[0] * resShape[1];
    uint64_t outputSize = resCap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Concat") {
        void *x_ptr = readToDev(GetGoldenDir() + "/concat_3_7168_operand1.bin", cap0);
        void *y_ptr = readToDev(GetGoldenDir() + "/concat_64_7168_operand2.bin", cap1);
        TileShape::Current().SetVecTile({1, 7168});
        Tensor input_x(dtype, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, shape1, (uint8_t *)y_ptr, "y");
        Tensor output(dtype, resShape, out_ptr, "res");

        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T", {input_x, input_y, output}) {
            output = Cat(std::vector<Tensor>{input_x, input_y}, -2);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(resCap);
    std::vector<float> dev_res(resCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/concat_67_7168_res.bin", golden);
    std::cout << "====== output size:" << dev_res.size() << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
