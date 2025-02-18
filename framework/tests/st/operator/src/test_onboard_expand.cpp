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
 * \file test_onboard_expand.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class ExpandOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(ExpandOnBoardTest, test_expand_32_1_to_32_32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 32;
    int S1 = 1;
    int D0 = 32;
    int D1 = 32;
    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("EXPAND") {
        void *x_ptr = readToDev(GetGoldenDir() + "/expand_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DataType::DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXPAND_T", {input_a, output}) {
            output = Expand(input_a, dstShape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/expand_res.bin", golden);
    for (size_t i = 0; i < res.size(); i++) {
        cout<<"res["<<i<<"]="<<res[i]<<" golden["<<i<<"]="<<golden[i]<<endl;
    }

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ExpandOnBoardTest, test_expand_32_8_1_to_32_8_32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 32;
    int S1 = 8;
    int S2 = 1;
    int D0 = 32;
    int D1 = 8;
    int D2 = 32;
    std::vector<int64_t> srcShape = {S0, S1, S2};
    std::vector<int64_t> dstShape = {D0, D1, D2};

    int srcCapacity = srcShape[0] * srcShape[1] * srcShape[2];
    int dstCapacity = dstShape[0] * dstShape[1] * dstShape[2];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("EXPAND") {
        void *x_ptr = readToDev(GetGoldenDir() + "/expand_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({8, 8, 16});
        Tensor input_a(DataType::DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXPAND_T", {input_a, output}) {
            output = Expand(input_a, dstShape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/expand_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ExpandOnBoardTest, test_expand_32_1_to_32_23) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 32;
    int S1 = 1;
    int D0 = 32;
    int D1 = 23;
    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("EXPAND") {
        void *x_ptr = readToDev(GetGoldenDir() + "/expand_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({16, 16});
        Tensor input_a(DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXPAND_T", {input_a, output}) {
            output = Expand(input_a, dstShape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/expand_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ExpandOnBoardTest, test_expand_32_8_1_to_32_8_23) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 32;
    int S1 = 8;
    int S2 = 1;
    int D0 = 32;
    int D1 = 8;
    int D2 = 23;
    std::vector<int64_t> srcShape = {S0, S1, S2};
    std::vector<int64_t> dstShape = {D0, D1, D2};

    int srcCapacity = srcShape[0] * srcShape[1] * srcShape[2];
    int dstCapacity = dstShape[0] * dstShape[1] * dstShape[2];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("EXPAND") {
        void *x_ptr = readToDev(GetGoldenDir() + "/expand_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({8, 8, 16});
        Tensor input_a(DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXPAND_T", {input_a, output}) {
            output = Expand(input_a, dstShape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/expand_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ExpandOnBoardTest, test_expand_for_4_dim) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> srcShape = {1, 32, 400, 23};
    std::vector<int64_t> dstShape = {8, 32, 400, 23};

    int srcCapacity = srcShape[0] * srcShape[1] * srcShape[2] * srcShape[3];
    int dstCapacity = dstShape[0] * dstShape[1] * dstShape[2] * dstShape[3];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("EXPAND") {
        void *x_ptr = readToDev(GetGoldenDir() + "/expand_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({2, 2, 128, 16});
        Tensor input_a(DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXPAND_T", {input_a, output}) {
            output = Expand(input_a, dstShape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/expand_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(ExpandOnBoardTest, test_expand_1_1_to_1_16384) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 1;
    int S1 = 1;
    int D0 = 1;
    int D1 = 16384;
    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("EXPAND") {
        void *x_ptr = readToDev(GetGoldenDir() + "/expand_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({1, 16384});
        Tensor input_a(DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("EXPAND_T", {input_a, output}) {
            output = Expand(input_a, dstShape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/expand_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
