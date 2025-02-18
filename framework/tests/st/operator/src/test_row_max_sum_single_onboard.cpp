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
 * \file test_row_max_sum_single_onboard.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class RowMaxSumSingleOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_max_single) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 257;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({128, 64});

        void *x_ptr = readToDev(GetGoldenDir() + "/257_128/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/257_128/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_sum_single) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 257;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        TileShape::Current().SetVecTile({128, 64});

        void *x_ptr = readToDev(GetGoldenDir() + "/257_128/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowSumSingle", {input_a, output}) {
            output = Sum(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/257_128/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_max_single_3dim) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 8;
    int shape1 = 4;
    int shape2 = 128;
    std::vector<int64_t> shape = {shape0, shape1, shape2};
    std::vector<int64_t> outshape = {shape0, shape1, 1};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({2, 1, 64});

        void *x_ptr = readToDev(GetGoldenDir() + "/8_4_128/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/8_4_128/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_sum_single_3dim_mla_rmsNorm) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    // rmsNorm:[B,S,qLoraRank]
    int shape0 = 16;
    int shape1 = 1;
    int shape2 = 1536;
    std::vector<int64_t> shape = {shape0, shape1, shape2};
    std::vector<int64_t> outshape = {shape0, shape1, 1};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        TileShape::Current().SetVecTile({8, 1, 128});

        void *x_ptr = readToDev(GetGoldenDir() + "/16_1_1536/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowSumSingle", {input_a, output}) {
            output = Sum(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/16_1_1536/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_max_single_4dim_softmax) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    // softmax rowmax: [B,N,1,S2]
    int shape0 = 2;
    int shape1 = 128;
    int shape2 = 1;
    int shape3 = 256;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0, shape1, shape2, 1};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *x_ptr = readToDev(GetGoldenDir() + "/2_128_1_256/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/2_128_1_256/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_max_single_4dim_softmax_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    // softmax rowmax: [B,N,1,S2]
    int shape0 = 1;
    int shape1 = 128;
    int shape2 = 1;
    int shape3 = 248;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0, shape1, shape2, 1};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({1, 64, 1, 64});

        void *x_ptr = readToDev(GetGoldenDir() + "/1_128_1_248/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/1_128_1_248/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_sum_single_4dim_softmax) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    // softmax rowsum: [B,N,1,S2]
    int shape0 = 32;
    int shape1 = 128;
    int shape2 = 1;
    int shape3 = 256;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0, shape1, shape2, 1};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        TileShape::Current().SetVecTile({1, 64, 1, 128});

        void *x_ptr = readToDev(GetGoldenDir() + "/32_128_1_256/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowSumSingle", {input_a, output}) {
            output = Sum(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/32_128_1_256/sum_res.bin", golden);


    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_sum_single_3dim_moe) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Reduce") {
        TileShape::Current().SetVecTile({2, 8, 512});

        void *x_ptr = readToDev(GetGoldenDir() + "/6_1_8_1024/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("Reduce3dimMoe", {input_a, output}) {
            output = Sum(input_a, 1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/6_1_8_1024/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_sum_single_3dim_big_moe) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 8;
    int S = 1;
    int numExpertsPerTok = 8;
    int H = 7168;
    std::vector<int64_t> shape = {B * S, numExpertsPerTok, H};
    std::vector<int64_t> outshape = {B * S, 1, H};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Reduce") {
        TileShape::Current().SetVecTile({1, 8, 512});

        void *x_ptr = readToDev(GetGoldenDir() + "/8_1_8_7168/x.bin", inputCapacity);
        // void *x_ptr = readToDev(GetGoldenDir() + "/6_1_8_1024/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("Reduce3dimMoeBig", {input_a, output}) {
            output = Sum(input_a, 1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/8_1_8_7168/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_operation_row_sum_single_2dim_moe) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int B = 8;
    int S = 1;
    int nRoutedExperts = 256;
    std::vector<int64_t> shape = {B * S, nRoutedExperts};
    std::vector<int64_t> outshape = {1, nRoutedExperts};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outshape.begin(), outshape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Reduce") {
        TileShape::Current().SetVecTile({8, 256});

        void *x_ptr = readToDev(GetGoldenDir() + "/8_1_1_256/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("Reduce2dimMoeBig", {input_a, output}) {
            output = Sum(input_a, 0, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/8_1_1_256/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_sum_single_4dim_axis0_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 6;
    int shape1 = 2;
    int shape2 = 8;
    int shape3 = 255;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outShape = {1, shape1, shape2, shape3};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outShape.begin(), outShape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Reduce") {
        TileShape::Current().SetVecTile({8, 2, 8, 256});

        void *x_ptr = readToDev(GetGoldenDir() + "/6_2_8_255/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("Reduce4dimMoe", {input_a, output}) {
            output = Sum(input_a, 0, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/6_2_8_255/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_sum_single_4dim_axis1_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 4;
    int shape1 = 2;
    int shape2 = 8;
    int shape3 = 255;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outShape = {shape0, 1, shape2, shape3};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outShape.begin(), outShape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Reduce") {
        TileShape::Current().SetVecTile({2, 8, 8, 256});

        void *x_ptr = readToDev(GetGoldenDir() + "/4_2_8_255/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("Reduce4dimMoe", {input_a, output}) {
            output = Sum(input_a, 1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/4_2_8_255/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_sum_single_4dim_axis2_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 3;
    int shape1 = 2;
    int shape2 = 8;
    int shape3 = 255;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outShape = {shape0, shape1, 1, shape3};
    int inputCapacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    int outputCapacity = std::accumulate(outShape.begin(), outShape.end(), 1, std::multiplies<>());
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Reduce") {
        TileShape::Current().SetVecTile({2, 8, 8, 256});

        void *x_ptr = readToDev(GetGoldenDir() + "/3_2_8_255/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("Reduce4dimMoe", {input_a, output}) {
            output = Sum(input_a, 2, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/3_2_8_255/sum_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_sum_single_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 4;
    int shape1 = 530;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        TileShape::Current().SetVecTile({4, 1024});

        void *x_ptr = readToDev(GetGoldenDir() + "/4_530/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowSumSingle", {input_a, output}) {
            output = Sum(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/4_530/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_sum_single_unalign_4_93) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 4;
    int shape1 = 93;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        TileShape::Current().SetVecTile({4, 1024});

        void *x_ptr = readToDev(GetGoldenDir() + "/4_93/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowSumSingle", {input_a, output}) {
            output = Sum(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/4_93/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_sum_single_unalign_4d) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 3;
    int shape1 = 3;
    int shape2 = 4;
    int shape3 = 530;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0, shape1, shape2, 1};
    int inputCapacity = shape0 * shape1 * shape2 * shape3;
    int outputCapacity = shape0 * shape1 * shape2 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowSumSingle") {
        TileShape::Current().SetVecTile({4, 4, 4, 1024});

        void *x_ptr = readToDev(GetGoldenDir() + "/3_3_4_530/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowSumSingle", {input_a, output}) {
            output = Sum(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/3_3_4_530/sum_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_max_single_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 4;
    int shape1 = 93;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({4, 1024});

        void *x_ptr = readToDev(GetGoldenDir() + "/4_93/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/4_93/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_max_single_unalign_4_93) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 4;
    int shape1 = 93;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};
    int inputCapacity = shape0 * shape1;
    int outputCapacity = shape0 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({4, 1024});

        void *x_ptr = readToDev(GetGoldenDir() + "/4_93/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/4_93/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RowMaxSumSingleOnBoardTest, test_row_max_single_unalign_4d) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int shape0 = 3;
    int shape1 = 3;
    int shape2 = 4;
    int shape3 = 530;
    std::vector<int64_t> shape = {shape0, shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0, shape1, shape2, 1};
    int inputCapacity = shape0 * shape1 * shape2 * shape3;
    int outputCapacity = shape0 * shape1 * shape2 * 1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("RowMaxSingle") {
        TileShape::Current().SetVecTile({4, 4, 4, 1024});

        void *x_ptr = readToDev(GetGoldenDir() + "/3_3_4_530/x.bin", inputCapacity);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, outshape, out_ptr, "C");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("RowMaxSingle", {input_a, output}) {
            output = Amax(input_a, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/3_3_4_530/max_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
