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
 * \file test_onboard_tensorindex.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class TensorIndexOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(TensorIndexOnBoardTest, test_tensorindex_float_32_64_1_32) {
    int S2 = 32;
    int D = 64;
    int B = 1;
    int S = 32;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("TensorIndex") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        // TileShape::Current().SetVecTile({1, 32, 64});
        // TileShape::Current().SetVecTile({1, 16, 64});
        TileShape::Current().SetVecTile({1, 16, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("TensorIndex_T", {input_src0, input_src1, output}) {
            output = TensorIndex(input_src0, input_src1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(TensorIndexOnBoardTest, test_tensorindex_float_64_256_1_64) {
    int S2 = 64;
    int D = 256;
    int B = 1;
    int S = 64;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("TensorIndex") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        // TileShape::Current().SetVecTile({1, 64, 64});
        // TileShape::Current().SetVecTile({1, 32, 64});
        TileShape::Current().SetVecTile({1, 32, 128});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("TensorIndex_T", {input_src0, input_src1, output}) {
            output = TensorIndex(input_src0, input_src1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(TensorIndexOnBoardTest, test_tensorindex_float_1_64_32_1) {
    int D = 64;
    int B = 32;
    int S = 1;
    std::vector<int64_t> shape0 = {S, D};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("TensorIndex") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        // TileShape::Current().SetVecTile({1, 1, 64});
        // TileShape::Current().SetVecTile({1, 1, 32});
        // TileShape::Current().SetVecTile({32, 1, 64});
        // TileShape::Current().SetVecTile({16, 1, 64});
        // TileShape::Current().SetVecTile({32, 1, 32});
        TileShape::Current().SetVecTile({16, 1, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("TensorIndex_T", {input_src0, input_src1, output}) {
            output = TensorIndex(input_src0, input_src1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(TensorIndexOnBoardTest, test_tensorindex_float_64_512_16_64) {
    int D = 512;
    int B = 16;
    int S = 64;
    std::vector<int64_t> shape0 = {S, D};
    std::vector<int64_t> shape1 = {B, S};
    std::vector<int64_t> shape2 = {B, S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1] * shape2[2];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("TensorIndex") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        TileShape::Current().SetVecTile({1, 32, 128});
        // TileShape::Current().SetVecTile({1, 64, 64});
        // TileShape::Current().SetVecTile({2, 32, 64});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("TensorIndex_T", {input_src0, input_src1, output}) {
            output = TensorIndex(input_src0, input_src1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(TensorIndexOnBoardTest, test_tensorindex_float_8_7168_64_moe) {
    int S2 = 8;
    int D = 7168;
    int S = 64;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {S};
    std::vector<int64_t> shape2 = {S, D};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("TensorIndex") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        TileShape::Current().SetVecTile({32, 128});
        // TileShape::Current().SetVecTile({8, 512});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)x_ptr, "x");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        ConfigManager::Instance();
        config::SetBuildStatic(true);
        FUNCTION("TensorIndex_T", {input_src0, input_src1, output}) {
            output = TensorIndex(input_src0, input_src1);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/y_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}
