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
 * \file test_onboard.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/llama/llama_def.h"
#include "test_dev_func_runner.h"

namespace {
int in0 = 2;
int row = 64;
int col = 64;
const int capacity = row * col;
const int capacity_64 = 1 * col;
const int capacity_dim3 = 2 * row * col;
const int capacity_dim4 = 1 * 1 * 16 * 16;
const int capacity_8_8_8 = 8 * 8 * 8;
const int capacity_2_2_16_16 = 2 * 2 * 16 * 16;
const int capacity_2_2_8_8 = 2 * 2 * 8 * 8;
const int capacity_2_1_8_8 = 2 * 1 * 8 * 8;
const int capacity_4_4_16_16 = 4 * 4 * 16 * 16;
const int capacity_4_1_16_16 = 4 * 1 * 16 * 16;
const int capacity_16_16_64_64 = 16 * 16 * 64 * 64;
const int capacity_8_32_32 = 8 * 32 * 32;
const int capacity_8_8_32_32 = 8 * 8 * 32 * 32;
const int capacity_4_4_4_16_16 = 4 * 4 * 4 * 16 * 16;
const int capacity_8_80_80 = 8 * 80 * 80;
const int capacity_64_128 = 64 * 128;
const int capacity_32_32 = 32 * 32;
const int capacity_16_32_32 = 16 * 32 * 32;
const int capacity_1_1_32 = 1 * 1 * 32;
const int capacity_8_16_16 = 8 * 16 * 16;
const int capacity_1_16_16 = 1 * 16 * 16;
const int capacity_8_8_1 = 8 * 8;
const int capacity_8_8_7168 = 8 * 8 * 7168;
const int capacity_8_1_1 = 8 * 1 * 1;
const int capacity_8_16_1 = 8 * 16 * 1;
const int capacity_1_1_1 = 1 * 1 * 1;
const int capacity_8_1_16 = 8 * 1 * 16;
const int capacity_8_8_1_1 = 8 * 8 * 1 * 1;
const int capacity_8_8_1_256 = 8 * 8 * 1 * 256;
}

using namespace npu::tile_fwk;

class OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(OnBoardTest, test_sin_dim2_float32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {64, 64};
    DataType dtype = DataType::DT_FP32;
    int cap = shape[0] * shape[1];
    uint64_t outputSize = cap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Sin") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_dim2_fp32.bin", cap); // true means no cut
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor output(dtype, shape, out_ptr, "sin");

        config::SetBuildStatic(true);
        FUNCTION("SIN_T", {input_x, output}) {
            output = Sin(input_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> x(cap);
    std::vector<float> golden(cap);
    std::vector<float> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/sin_golden_fp32.bin", golden);
    readInput(GetGoldenDir() + "/x_dim2_fp32.bin", x);
    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_cos_dim4_float16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 64, 64};
    DataType dtype = DataType::DT_FP16;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    uint64_t outputSize = cap * sizeof(uint16_t);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("Cos") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_dim_4_fp16.bin", cap); // true means no cut
        TileShape::Current().SetVecTile({1, 1, 32, 64});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor output(dtype, shape, out_ptr, "sin");

        config::SetBuildStatic(true);
        FUNCTION("COS_T", {input_x, output}) {
            output = Cos(input_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<npu::tile_fwk::float16> x(cap);
    std::vector<npu::tile_fwk::float16> golden(cap);
    std::vector<npu::tile_fwk::float16> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/cos_golden_fp16.bin", golden);
    readInput(GetGoldenDir() + "/x_dim_4_fp16.bin", x);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_gather_float_case1) {
    int B = 1;
    int S = 32;
    int S2 = 32;
    int D = 64;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    char buffer[100];
    sprintf(buffer, "../tests/Gather/%d_%d_%d_%d/", B, S, S2, D);
    std::string inputDir(buffer);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GATHER") {
        void *x_ptr = readToDev(inputDir + "x.bin", capacity0);
        void *indices_ptr = readToDev(inputDir + "indices.bin", capacity1);
        // TileShape::Current().SetVecTile({1, 32, 64});
        // TileShape::Current().SetVecTile({1, 16, 64});
        TileShape::Current().SetVecTile({1, 16, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_T", {input_src0, input_src1, output}) {
            output = Gather(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(inputDir + "y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_gather_float_case2) {
    int B = 1;
    int S = 64;
    int S2 = 64;
    int D = 256;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    char buffer[100];
    sprintf(buffer, "../tests/Gather/%d_%d_%d_%d/", B, S, S2, D);
    std::string inputDir(buffer);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GATHER") {
        void *x_ptr = readToDev(inputDir + "x.bin", capacity0);
        void *indices_ptr = readToDev(inputDir + "indices.bin", capacity1);
        // TileShape::Current().SetVecTile({1, 64, 64});
        // TileShape::Current().SetVecTile({1, 32, 64});
        TileShape::Current().SetVecTile({1, 32, 128});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_T", {input_src0, input_src1, output}) {
            output = Gather(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(inputDir + "y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_gather_float_case3) {
    int B = 32;
    int S = 1;
    int S2 = 1;
    int D = 64;
    std::vector<int64_t> shape0 = {S, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    char buffer[100];
    sprintf(buffer, "../tests/Gather/%d_%d_%d_%d/", B, S, S2, D);
    std::string inputDir(buffer);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GATHER") {
        void *x_ptr = readToDev(inputDir + "x.bin", capacity0);
        void *indices_ptr = readToDev(inputDir + "indices.bin", capacity1);
        // TileShape::Current().SetVecTile({1, 1, 64});
        // TileShape::Current().SetVecTile({1, 1, 32});
        // TileShape::Current().SetVecTile({32, 1, 64});
        // TileShape::Current().SetVecTile({16, 1, 64});
        // TileShape::Current().SetVecTile({32, 1, 32});
        TileShape::Current().SetVecTile({16, 1, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_T", {input_src0, input_src1, output}) {
            output = Gather(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(inputDir + "y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_gather_float_case4) {
    int B = 16;
    int S = 64;
    int S2 = 64;
    int D = 512;
    std::vector<int64_t> shape0 = {S, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    char buffer[100];
    sprintf(buffer, "../tests/Gather/%d_%d_%d_%d/", B, S, S2, D);
    std::string inputDir(buffer);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GATHER") {
        void *x_ptr = readToDev(inputDir + "x.bin", capacity0);
        void *indices_ptr = readToDev(inputDir + "indices.bin", capacity1);
        TileShape::Current().SetVecTile({1, 32, 128});
        // TileShape::Current().SetVecTile({1, 64, 64});
        // TileShape::Current().SetVecTile({2, 32, 64});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_T", {input_src0, input_src1, output}) {
            output = Gather(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(inputDir + "y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_concat_all2all) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * 2 * sizeof(float);
    uint8_t* outputGmAddr = allocDevAddr(outputSize);
    PROGRAM("CONCAT") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev("../tests/AsmdTensor/concat_2dim_x.bin", capacity);
        void *y_ptr = readToDev("../tests/AsmdTensor/concat_2dim_y.bin", capacity);
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output;

        FUNCTION("CONCAT_T") {
            output = Cat(std::vector<Tensor>{input_a, input_b}, 1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(capacity * 2);
    std::vector<float> res(capacity * 2);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outputGmAddr, outputSize);
    readInput("../tests/AsmdTensor/concat_2dim_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    for (size_t i = 0; i < golden.size(); ++i) {
        std::cout << golden[i] << "," << res[i] << std::endl;
    }
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_concat_4) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 13 * 2 * 10 * 10 * sizeof(float);
    uint8_t* outputGmAddr = allocDevAddr(outputSize);
    PROGRAM("CONCAT") {
        std::vector<int64_t> shape1 = {2, 2, 10, 10};
        std::vector<int64_t> shape2 = {3, 2, 10, 10};
        std::vector<int64_t> shape3 = {8, 2, 10, 10};
        void *x_ptr = readToDev("../tests/AsmdTensor/concat_4dim_operand1.bin", 2 * 2 * 10 * 10);
        void *y_ptr = readToDev("../tests/AsmdTensor/concat_4dim_operand2.bin", 3 * 2 * 10 * 10);
        void *z_ptr = readToDev("../tests/AsmdTensor/concat_4dim_operand3.bin", 8 * 2 * 10 * 10);
        TileShape::Current().SetVecTile({6, 6, 6, 6});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor input_c(DataType::DT_FP32, shape3, (uint8_t *)z_ptr, "C");
        Tensor output;

        config::SetBuildStatic(true);
        FUNCTION("CONCAT_T") {
            output = Cat(std::vector<Tensor>{input_a, input_b, input_c}, 0);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(13 * 2 * 10 * 10);
    std::vector<float> res(13 * 2 * 10 * 10);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outputGmAddr, outputSize);
    readInput("../tests/AsmdTensor/concat_4dim_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    for (size_t i = 0; i < golden.size(); ++i) {
        std::cout << golden[i] << "," << res[i] << std::endl;
    }
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_16_64_64_tileop_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({8, 8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_16_64_65_tileop_add_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {16, 16, 64, 65};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);

    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shapeSize);
        TileShape::Current().SetVecTile({8, 8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_16_39_65_tileop_add_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {16, 16, 39, 65};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);

    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shapeSize);
        TileShape::Current().SetVecTile({8, 8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_32_1_tileop_add_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {32, 1};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);

    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shapeSize);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_32_1_tileop_sub_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {32, 1};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);

    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shapeSize);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("SUB_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_32_1_tileop_mul_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {32, 1};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);

    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shapeSize);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_16_64_64_tileop_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({8, 4, 16, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("SUB_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_16_64_64_tileop_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({4, 8, 16, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_16_64_64_tileop_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({8, 8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("DIV_T", {input_a, input_b, output}) {
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_80_80_tileop_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_80_80 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {8, 80, 80};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_80_80);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_80_80);
        TileShape::Current().SetVecTile({4, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_80_80);
    std::vector<float> res(capacity_8_80_80);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_80_80_tileop_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_80_80 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape = {8, 80, 80};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_80_80);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_80_80);
        TileShape::Current().SetVecTile({4, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("SUB_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_80_80);
    std::vector<float> res(capacity_8_80_80);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_80_80_tileop_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_80_80 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {8, 80, 80};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_80_80);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_80_80);
        TileShape::Current().SetVecTile({4, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_80_80);
    std::vector<float> res(capacity_8_80_80);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_80_80_tileop_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_80_80 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {8, 80, 80};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_80_80);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_80_80);
        TileShape::Current().SetVecTile({4, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("DIV_T", {input_a, input_b, output}) {
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_80_80);
    std::vector<float> res(capacity_8_80_80);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_64_128_tileop_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_64_128 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {64, 128};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_64_128);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_64_128);
        TileShape::Current().SetVecTile({32, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_64_128);
    std::vector<float> res(capacity_64_128);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_64_128_tileop_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_64_128 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape = {64, 128};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_64_128);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_64_128);
        TileShape::Current().SetVecTile({10, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("SUB_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_64_128);
    std::vector<float> res(capacity_64_128);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_64_128_tileop_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_64_128 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {64, 128};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_64_128);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_64_128);
        TileShape::Current().SetVecTile({32, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_64_128);
    std::vector<float> res(capacity_64_128);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_64_128_tileop_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_64_128 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {64, 128};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_64_128);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_64_128);
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("DIV_T", {input_a, input_b, output}) {
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_64_128);
    std::vector<float> res(capacity_64_128);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_dim4_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_dim4 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {1, 1, 16, 16};
        void *x_ptr = readToDev(GetGoldenDir() + "/add_dim4_x.bin", capacity_dim4);
        void *y_ptr = readToDev(GetGoldenDir() + "/add_dim4_y.bin", capacity_dim4);
        TileShape::Current().SetVecTile({1, 1, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_dim4);
    std::vector<float> res(capacity_dim4);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/add_dim4_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_dim2_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/add_x.bin", capacity);
        void *y_ptr = readToDev(GetGoldenDir() + "/add_y.bin", capacity);
        TileShape::Current().SetVecTile({64, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/add_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// ----------------------------------expand
TEST_F(OnBoardTest, test_operation_tensor_2_2_8_8_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_2_2_8_8 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {2, 2, 8, 8};
        std::vector<int64_t> shape1 = {2, 1, 8, 8};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_2_2_8_8);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_2_1_8_8);
        TileShape::Current().SetVecTile({2, 2, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_2_2_8_8);
    std::vector<float> res(capacity_2_2_8_8);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    std::vector<float> input01(capacity_2_2_8_8);
    readInput(GetGoldenDir() + "/x.bin", input01);
    std::vector<float> input02(capacity_2_1_8_8);
    readInput(GetGoldenDir() + "/y.bin", input02);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_1_n_to_m_n_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 64 * 32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape1 = {64, 32};
        std::vector<int64_t> shape2 = {1, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", 64 * 32);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", 1 * 32);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape1, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape2, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape1, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(64 * 32);
    std::vector<float> res(64 * 32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_4_4_16_16_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_4_4_16_16 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {4, 4, 16, 16};
        std::vector<int64_t> shape1 = {4, 1, 16, 16};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_4_4_16_16);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_4_1_16_16);
        TileShape::Current().SetVecTile({2, 2, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_4_4_16_16);
    std::vector<float> res(capacity_4_4_16_16);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    std::vector<float> input01(capacity_4_4_16_16);
    readInput(GetGoldenDir() + "/x.bin", input01);
    std::vector<float> input02(capacity_4_1_16_16);
    readInput(GetGoldenDir() + "/y.bin", input02);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_1_1_32_to_16_32_32_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {16, 32, 32};
        std::vector<int64_t> shape1 = {1, 1, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_32_32);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_1_1_32);
        TileShape::Current().SetVecTile({8, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_32_32);
    std::vector<float> res(capacity_16_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    std::vector<float> input01(capacity_16_32_32);
    readInput(GetGoldenDir() + "/x.bin", input01);
    std::vector<float> input02(capacity_1_1_32);
    readInput(GetGoldenDir() + "/y.bin", input02);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_16_1_to_8_16_16_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_16_16 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 16, 16};
        std::vector<int64_t> shape1 = {8, 16, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_16_16);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_16_1);
        TileShape::Current().SetVecTile({4, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_16_16);
    std::vector<float> res(capacity_8_16_16);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_8_1_to_8_8_7168_expand_Mul_moe) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_8_7168 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape0 = {8, 8, 7168};
        std::vector<int64_t> shape1 = {8, 8, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_8_7168);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_8_1);
        TileShape::Current().SetVecTile({8, 8, 128});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_b, input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_8_7168);
    std::vector<float> res(capacity_8_8_7168);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_8_1_to_8_8_7168_expand_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_8_7168 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape0 = {8, 8, 1};
        std::vector<int64_t> shape1 = {8, 8, 7168};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_8_1);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_8_7168);
        TileShape::Current().SetVecTile({8, 8, 128});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("SUB_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_8_7168);
    std::vector<float> res(capacity_8_8_7168);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_1_16_to_8_16_16_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_16_16 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 16, 16};
        std::vector<int64_t> shape1 = {8, 1, 16};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_16_16);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_1_16);
        TileShape::Current().SetVecTile({4, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_16_16);
    std::vector<float> res(capacity_8_16_16);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_1_16_16_to_8_16_16_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_16_16 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 16, 16};
        std::vector<int64_t> shape1 = {1, 16, 16};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_16_16);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_1_16_16);
        TileShape::Current().SetVecTile({4, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_16_16);
    std::vector<float> res(capacity_8_16_16);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_8_1_1_to_8_16_16_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_16_16 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 16, 16};
        std::vector<int64_t> shape1 = {8, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_16_16);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_1_1);
        TileShape::Current().SetVecTile({4, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_16_16);
    std::vector<float> res(capacity_8_16_16);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_1_1_1_to_8_16_16_expand_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_8_16_16 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 16, 16};
        std::vector<int64_t> shape1 = {1, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_16_16);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_1_1_1);
        TileShape::Current().SetVecTile({4, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_16_16);
    std::vector<float> res(capacity_8_16_16);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int size0 = 32*32*256;
    int size1 = 32*32;
    uint64_t outputSize = size0 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {32, 32, 1, 256};
        std::vector<int64_t> shape1 = {32, 32, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", size0);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", size1);
        TileShape::Current().SetVecTile({16, 16, 1, 16});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Add(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(size0);
    std::vector<float> res(size0);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int size0 = 32*32*256;
    int size1 = 32*32;
    uint64_t outputSize = size0 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape0 = {32, 32, 1, 256};
        std::vector<int64_t> shape1 = {32, 32, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", size0);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", size1);
        TileShape::Current().SetVecTile({16, 16, 1, 32});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("SUB_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(size0);
    std::vector<float> res(size0);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int size0 = 32*32*256;
    int size1 = 32*32;
    uint64_t outputSize = size0 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape0 = {32, 32, 1, 256};
        std::vector<int64_t> shape1 = {32, 32, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", size0);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", size1);
        TileShape::Current().SetVecTile({16, 16, 1, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(size0);
    std::vector<float> res(size0);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int size0 = 32*32*256;
    int size1 = 32*32;
    uint64_t outputSize = size0 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape0 = {32, 32, 1, 256};
        std::vector<int64_t> shape1 = {32, 32, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", size0);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", size1);
        TileShape::Current().SetVecTile({16, 16, 1, 32});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(size0);
    std::vector<float> res(size0);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_8_1_1_to_8_8_1_256_tileop_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    uint64_t outputSize = capacity_8_8_1_256 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 8, 1, 256};
        std::vector<int64_t> shape1 = {8, 8, 1, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_8_1_256);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capacity_8_8_1_1);
        TileShape::Current().SetVecTile({8, 8, 1, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Sub(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_8_1_256);
    std::vector<float> res(capacity_8_8_1_256);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
TEST_F(OnBoardTest, test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul01) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int capShape1 = 64;
    int capShape2 = 128*64;
    uint64_t outputSize = capShape2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape0 = {1, 1, 1, 64};
        std::vector<int64_t> shape1 = {1, 128, 1, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capShape1);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capShape2);
        TileShape::Current().SetVecTile({1, 16, 1, 32});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capShape2);
    std::vector<float> res(capShape2);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul02) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int capShape1 = 128*64;
    int capShape2 = 64;
    uint64_t outputSize = capShape2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape0 = {1, 128, 1, 64};
        std::vector<int64_t> shape1 = {1, 1, 1, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capShape1);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capShape2);
        TileShape::Current().SetVecTile({1, 1, 1, 64});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capShape1);
    std::vector<float> res(capShape1);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
// Fail
TEST_F(OnBoardTest, test_operation_tensor_1_1_64_to_32_1_64_tileop_mul03) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int capShape1 = 32*64;
    int capShape2 = 64;
    uint64_t outputSize = capShape2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape0 = {32, 1, 64};
        std::vector<int64_t> shape1 = {1, 1, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capShape1);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capShape2);
        TileShape::Current().SetVecTile({16, 1, 32});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capShape1);
    std::vector<float> res(capShape1);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_8_8_1_to_8_8_7168_expand_mul) {
    aclInit(nullptr);
    int ccc = 7168;
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 8*8*ccc * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {8, 8, ccc};
        std::vector<int64_t> shape1 = {8, 8, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", 8*8*ccc);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", 8*8);
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            TileShape::Current().SetVecTile({8, 8, 128});
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(8*8*ccc);
    std::vector<float> res(8*8*ccc);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

// ----------------------------------expand end

TEST_F(OnBoardTest, test_unary_operation_32_32_tileop_exp) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("EXP") {
        std::vector<int64_t> shape = {32, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_32_32);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXP_T", {input_a, output}) {
            output = Exp(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_32_32);
    std::vector<float> res(capacity_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_32_32_tileop_exp) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("EXP") {
        std::vector<int64_t> shape = {16, 32, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_32_32);
        TileShape::Current().SetVecTile({8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXP_T", {input_a, output}) {
            output = Exp(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_32_32);
    std::vector<float> res(capacity_16_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_16_64_64_tileop_exp) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("EXP") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({8, 8, 16, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXP_T", {input_a, output}) {
            output = Exp(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_32_32_tileop_sqrt) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SQRT") {
        std::vector<int64_t> shape = {32, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_32_32);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SQRT_T", {input_a, output}) {
            output = Sqrt(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_32_32);
    std::vector<float> res(capacity_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_32_32_tileop_sqrt) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SQRT") {
        std::vector<int64_t> shape = {16, 32, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_32_32);
        TileShape::Current().SetVecTile({8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SQRT_T", {input_a, output}) {
            output = Sqrt(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_32_32);
    std::vector<float> res(capacity_16_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_16_64_64_tileop_sqrt) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SQRT") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({8, 8, 16, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SQRT_T", {input_a, output}) {
            output = Sqrt(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_16_64_70_tileop_sqrt) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    const int capacity_16_16_64_70 = 16 * 16 * 64 * 70;
    uint64_t outputSize = capacity_16_16_64_70 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SQRT") {
        std::vector<int64_t> shape = {16, 16, 64, 70};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_70);
        TileShape::Current().SetVecTile({8, 8, 16, 32});
        Tensor input_a(DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("SQRT_T", {input_a, output}) {
            output = Sqrt(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_70);
    std::vector<float> res(capacity_16_16_64_70);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_32_32_tileop_reciprocal) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RECIPROCAL") {
        std::vector<int64_t> shape = {32, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_32_32);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("RECIPROCAL_T", {input_a, output}) {
            output = Reciprocal(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_32_32);
    std::vector<float> res(capacity_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_32_32_tileop_reciprocal) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_32_32 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RECIPROCAL") {
        std::vector<int64_t> shape = {16, 32, 32};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_32_32);
        TileShape::Current().SetVecTile({8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("RECIPROCAL_T", {input_a, output}) {
            output = Reciprocal(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_32_32);
    std::vector<float> res(capacity_16_32_32);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_unary_operation_16_16_64_64_tileop_reciprocal) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_16_16_64_64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("RECIPROCAL") {
        std::vector<int64_t> shape = {16, 16, 64, 64};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_16_16_64_64);
        TileShape::Current().SetVecTile({2, 8, 16, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("RECIPROCAL_T", {input_a, output}) {
            output = Reciprocal(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_16_16_64_64);
    std::vector<float> res(capacity_16_16_64_64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim2_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/adds_2d_x.bin", capacity);
        TileShape::Current().SetVecTile({64, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("ADD_S", {input_a, output}) {
            output = Add(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/adds_2d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_add_vs_dim2_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {79, 85};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        TileShape::Current().SetVecTile({16, 128});
        Tensor input_a(DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("ADD_S", {input_a, output}) {
            output = Add(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_mul_vs_dim3_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 79, 85};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        TileShape::Current().SetVecTile({1, 16, 128});
        Tensor input_a(DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("MUL_S", {input_a, output}) {
            output = Mul(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_sub_vs_dim4_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 67, 125};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        TileShape::Current().SetVecTile({2, 1, 16, 128});
        Tensor input_a(DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("SUB_S", {input_a, output}) {
            output = Sub(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_div_vs_dim1_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {125};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        TileShape::Current().SetVecTile({128});
        Tensor input_a(DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("DIV_S", {input_a, output}) {
            output = Div(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim2_add_FP16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * sizeof(uint16_t);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/adds_2d_x.bin", capacity);
        TileShape::Current().SetVecTile({64, 64});
        Tensor input_a(DataType::DT_FP16, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP16, shape, out_ptr, "C");
        config::SetBuildStatic(true);
        FUNCTION("ADD_S", {input_a, output}) {
            output = Add(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<npu::tile_fwk::float16> golden(capacity);
    std::vector<npu::tile_fwk::float16> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/adds_2d_res.bin", golden);

    std::vector<npu::tile_fwk::float16> x(outputSize);
    readInput(GetGoldenDir() + "/adds_2d_x.bin", x);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim2_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/subs_2d_x.bin", capacity);
        TileShape::Current().SetVecTile({64, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("SUB_S", {input_a, output}) {
            output = Sub(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/subs_2d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim2_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/muls_2d_x.bin", capacity);
        TileShape::Current().SetVecTile({64, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("MUL_S", {input_a, output}) {
            output = Mul(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/muls_2d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim2_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/divs_2d_x.bin", capacity);
        TileShape::Current().SetVecTile({64, 64});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("DIV_S", {input_a, output}) {
            output = Div(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/divs_2d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim1_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 64 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {col};
        void *x_ptr = readToDev(GetGoldenDir() + "/divs_1d_x.bin", 1 * 64);
        TileShape::Current().SetVecTile(32);
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("DIV_S", {input_a, output}) {
            // auto reshapeInput = Reshape(input_a, {1 * col});
            output = Div(input_a, value);
            // output = Reshape(reshapeOutput, shape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(64);
    std::vector<float> res(64);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/divs_1d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim3_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_dim3 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {in0, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/adds_3d_x.bin", capacity_dim3);
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("ADD_S", {input_a, output}) {
            output = Add(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_dim3);
    std::vector<float> res(capacity_dim3);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/adds_3d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim4_add) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 2 * 2 * capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape = {2, 2, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/adds_4d_x.bin", 2 * 2 * capacity);
        TileShape::Current().SetVecTile({1, 1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("ADD_S", {input_a, output}) {
            output = Add(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(2 * 2 * capacity);
    std::vector<float> res(2 * 2 * capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/adds_4d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim3_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_dim3 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape = {in0, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/subs_3d_x.bin", capacity_dim3);
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("SUB_S", {input_a, output}) {
            output = Sub(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_dim3);
    std::vector<float> res(capacity_dim3);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/subs_3d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim4_sub) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 2 * 2 * capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("SUB") {
        std::vector<int64_t> shape = {2, 2, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/subs_4d_x.bin", 2 * 2 * capacity);
        TileShape::Current().SetVecTile({1, 1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("SUB_S", {input_a, output}) {
            output = Sub(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(2 * 2 * capacity);
    std::vector<float> res(2 * 2 * capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/subs_4d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim3_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_dim3 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {in0, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/muls_3d_x.bin", capacity_dim3);
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("MUL_S", {input_a, output}) {
            output = Mul(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_dim3);
    std::vector<float> res(capacity_dim3);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/muls_3d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim4_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 2 * 2 * capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {2, 2, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/muls_4d_x.bin", 2 * 2 * capacity);
        TileShape::Current().SetVecTile({1, 1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("MUL_S", {input_a, output}) {
            output = Mul(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(2 * 2 * capacity);
    std::vector<float> res(2 * 2 * capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/muls_4d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_32_32_1_256_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int cap = 32 * 32 * 256;
    uint64_t outputSize = cap * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        std::vector<int64_t> shape = {32, 32, 1, 256};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", cap);
        TileShape::Current().SetVecTile({16, 16, 1, 16});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 0.07256f);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("MUL_S", {input_a, output}) {
            output = Mul(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(cap);
    std::vector<float> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim3_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_dim3 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {in0, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/divs_3d_x.bin", capacity_dim3);
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("DIV_S", {input_a, output}) {
            output = Div(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_dim3);
    std::vector<float> res(capacity_dim3);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/divs_3d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_scalar_dim4_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = 2 * 2 * capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {2, 2, row, col};
        void *x_ptr = readToDev(GetGoldenDir() + "/divs_4d_x.bin", 2 * 2 * capacity);
        TileShape::Current().SetVecTile({1, 1, 32, 32});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Element value(DataType::DT_FP32, 1.5);
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("DIV_S", {input_a, output}) {
            output = Div(input_a, value);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(2 * 2 * capacity);
    std::vector<float> res(2 * 2 * capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/divs_4d_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_tensor_16_32_32_to_16_32_1_tileop_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int capShape1 = 16 * 32 * 32;
    int capShape2 = 16 * 32 * 1;
    uint64_t outputSize = capShape1 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("ADD") {
        std::vector<int64_t> shape0 = {16, 32, 32};
        std::vector<int64_t> shape1 = {16, 32, 1};
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capShape1);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", capShape2);
        TileShape::Current().SetVecTile({8, 8, 8});
        Tensor input_a(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape1, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape0, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("ADD_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capShape1);
    std::vector<float> res(capShape1);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_scatterupdate_case1) {
    int B = 2;
    int S = 1;
    int S2 = 512;
    int kvLoraRank = 512;
    int qkRopeHeadDim = 64;

    std::vector<int64_t> shape0 = {B, 1, S2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> shape1 = {S};
    std::vector<int64_t> shape2 = {B, 1, S, kvLoraRank + qkRopeHeadDim};

    int capacity0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];

    char buffer[100];
    sprintf(buffer, "../tests/ScatterUpdate/%d_%d_%d_%d_%d/", B, S, S2, kvLoraRank, qkRopeHeadDim);
    std::string inputDir(buffer);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity0 * sizeof(float);
    uint8_t* outputGmAddr = allocDevAddr(outputSize);

    PROGRAM("SCATTERUPDATE") {
        readToDev(inputDir + "x.bin", capacity0);
 // false:用tensor graph展开
        TileShape::Current().SetVecTile(1, 1, 256, 128);

        Tensor kv_len(DataType::DT_INT32, {1, 1}, "kv_len");
        Tensor past_key_states(DataType::DT_FP32, {B, 1, S2, kvLoraRank + qkRopeHeadDim}, "past_key_states");
        // Tensor compressed_kv(DataType::DT_FP32, {B, S, kvLoraRank}, "past_key_states");
        // Tensor k_pe_rope(DataType::DT_FP32, {B, 1, S, qkRopeHeadDim}, "k_pe_rope"); // (b,1,s,qkRopeHeadDim)
        Tensor res;
        Tensor key_states(DataType::DT_FP32, {B, 1, S, kvLoraRank + qkRopeHeadDim}, "past_key_states");
        Tensor past_key_states_new(DataType::DT_FP32, {B, 1, S2, kvLoraRank + qkRopeHeadDim}, "past_key_states_new");

        config::SetBuildStatic(true);
        FUNCTION("SCATTERUPDATE_T") {
            past_key_states_new = ScatterUpdate(past_key_states, kv_len, key_states, -2);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity0);
    std::vector<float> dev_res(capacity0);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)outputGmAddr, outputSize);
    readInput(inputDir + "z_golden.bin", golden);
    std::cout << "====== output size:" << capacity0 << std::endl;
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_mul_large_row) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {1, 16384};
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    uint64_t outputSize = shapeSize * sizeof(float);

    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("MUL") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", shapeSize);
        void *y_ptr = readToDev(GetGoldenDir() + "/y.bin", shapeSize);
        TileShape::Current().SetVecTile({1, 16384});
        Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");
        ConfigManager::Instance();

        config::SetBuildStatic(true);
        FUNCTION("MUL_T", {input_a, input_b, output}) {
            output = Mul(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(shapeSize);
    std::vector<float> res(shapeSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_matmul_add_dynamic) {
    TileShape::Current().SetVecTile({128, 128});
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    const int m = 128;
    const int k = 256;
    const int n = 512;
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);

    Tensor tensor_a = Tensor(DataType::DT_FP16, {m, k}, "tensor_a", TileOpFormat::TILEOP_ND);
    Tensor tensor_b = Tensor(DataType::DT_FP16, {k, n}, "tensor_b", TileOpFormat::TILEOP_ND);
    Tensor tensor_c = Tensor(DataType::DT_FP16, {m, n}, "tensor_c", TileOpFormat::TILEOP_ND);
    Tensor tensor_d = Tensor(DataType::DT_FP16, {m, n}, "tensor_d", TileOpFormat::TILEOP_ND);
    Tensor tensor_o = Tensor(DataType::DT_FP16, {m, n}, "tensor_o", TileOpFormat::TILEOP_ND);

    std::vector<float16> aData(m * k, 0);
    std::vector<float16> bData(k * n, 0);
    std::vector<float16> cData(m * n, 0);
    std::vector<float16> dData(m * n, 0);
    std::vector<float16> golden(m * n, 0);

    readInput<float16>(GetGoldenDir() + "/matmulx.bin", aData);
    readInput<float16>(GetGoldenDir() + "/matmuly.bin", bData);
    readInput<float16>(GetGoldenDir() + "/add1.bin", cData);
    readInput<float16>(GetGoldenDir() + "/add2.bin", dData);
    readInput<float16>(GetGoldenDir() + "/res.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float16>(tensor_a, aData),
        RawTensorData::CreateTensor<float16>(tensor_b, bData),
        RawTensorData::CreateTensor<float16>(tensor_c, cData),
        RawTensorData::CreateTensor<float16>(tensor_d, dData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float16>(tensor_o, static_cast<float16>(0.0)),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float16>(tensor_o, golden),
    });

    FUNCTION("matmul_add", {tensor_a, tensor_b, tensor_c, tensor_d}, {tensor_o}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            (void)mIdx;
            Tensor tmp1 = npu::tile_fwk::Matrix::Matmul(DataType::DT_FP16, tensor_a, tensor_b, false, false, false);
            Tensor tmp2 = Add(tmp1, tensor_c);
            tensor_o = Add(tmp2, tensor_d);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float16 *)outs->data(), 0.001f));
}