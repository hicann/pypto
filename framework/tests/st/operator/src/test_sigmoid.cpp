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
 * \file test_sigmoid.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class SigmoidTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(SigmoidTest, test_32_32_tileop_sigmoid) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 32;
    int shape1 = 32;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, shape1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SIGMOID") {
        TileShape::Current().SetVecTile({16, 16});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SIGMOID", {input_a, output}) {
            output = Sigmoid(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SigmoidTest, test_32_256_tileop_sigmoid_realcase) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 32;
    int shape1 = 256;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, shape1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SIGMOID") {
        TileShape::Current().SetVecTile({16, 32});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SIGMOID", {input_a, output}) {
            output = Sigmoid(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SigmoidTest, test_2_32_32_tileop_sigmoid) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 2;
    int shape1 = 32;
     int shape2 = 32;
    std::vector<int64_t> shape = {shape0, shape1, shape2};
    std::vector<int64_t> outshape = {shape0, shape1, shape2};
    int inputCapacity = shape0 * shape1 * shape2;
    int outputCapacity = shape0 * shape1 * shape2;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SIGMOID") {
        TileShape::Current().SetVecTile({16, 16, 16});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SIGMOID", {input_a, output}) {
            output = Sigmoid(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SigmoidTest, test_2_2_32_32_tileop_sigmoid) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 2;
    int shape1 = 2;
    int shape2 = 32;
    int shape3 = 32;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0, shape1, shape2, shape3};
    int inputCapacity = shape0 * shape1 * shape2 * shape3;
    int outputCapacity = shape0 * shape1 * shape2 * shape3;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SIGMOID") {
        TileShape::Current().SetVecTile({16, 16, 16, 16});

        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SIGMOID", {input_a, output}) {
            output = Sigmoid(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
