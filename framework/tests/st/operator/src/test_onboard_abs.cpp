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
 * \file test_onboard_abs.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class AbsOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(AbsOnBoardTest, test_abs_8_4608) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 8;
    int S1 = 4608;
    int D0 = 8;
    int D1 = 4608;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(uint16_t);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("ABS") {
        void *x_ptr = readToDev(GetGoldenDir() + "/abs_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({8, 128});
        Tensor input_a(DataType::DT_FP16, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP16, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("ABS_T", {input_a, output}) {
            output = Abs(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<npu::tile_fwk::float16> x(dstCapacity);
    std::vector<npu::tile_fwk::float16> golden(dstCapacity);
    std::vector<npu::tile_fwk::float16> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/abs_golden.bin", golden);
    readInput(GetGoldenDir() + "/abs_x.bin", x);

    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(AbsOnBoardTest, test_abs_8_4609) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 8;
    int S1 = 4609;
    int D0 = 8;
    int D1 = 4609;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(uint16_t);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("ABS") {
        void *x_ptr = readToDev(GetGoldenDir() + "/abs_x_not_align.bin", srcCapacity);
        TileShape::Current().SetVecTile({8, 128});
        Tensor input_a(DT_FP16, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP16, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("ABS_T", {input_a, output}) {
            output = Abs(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<npu::tile_fwk::float16> x(dstCapacity);
    std::vector<npu::tile_fwk::float16> golden(dstCapacity);
    std::vector<npu::tile_fwk::float16> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/abs_golden_not_align.bin", golden);
    readInput(GetGoldenDir() + "/abs_x_not_align.bin", x);

    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(AbsOnBoardTest, test_abs_1_16384) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 1;
    int S1 = 16384;
    int D0 = 1;
    int D1 = 16384;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("ABS") {
        void *x_ptr = readToDev(GetGoldenDir() + "/abs_x_not_align.bin", srcCapacity);
        TileShape::Current().SetVecTile({1, 16384});
        Tensor input_a(DT_FP16, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DT_FP16, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("ABS_T", {input_a, output}) {
            output = Abs(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<npu::tile_fwk::float16> x(dstCapacity);
    std::vector<npu::tile_fwk::float16> golden(dstCapacity);
    std::vector<npu::tile_fwk::float16> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/abs_golden_not_align.bin", golden);
    readInput(GetGoldenDir() + "/abs_x_not_align.bin", x);

    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
