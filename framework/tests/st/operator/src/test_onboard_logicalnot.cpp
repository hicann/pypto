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
 * \file test_onboard_logicalnot.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class LogicalNotOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(LogicalNotOnBoardTest, test_logicalnot_16_32_fp32) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 16;
    int S1 = 32;
    int D0 = 16;
    int D1 = 32;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    int srcCapacity = srcShape[0] * srcShape[1];
    int dstCapacity = dstShape[0] * dstShape[1];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("LOGICALNOT") {
        void *x_ptr = readToDev(GetGoldenDir() + "/logicalnotdim2_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({8, 16});
        Tensor input_a(DataType::DT_FP32, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP32, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("LOGICALNOT_T", {input_a, output}) {
            output = LogicalNot(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/logicalnotdim2_golden.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(LogicalNotOnBoardTest, test_logicalnot_16_32_32_fp16) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int S0 = 16;
    int S1 = 32;
    int S2 = 32;
    int D0 = 16;
    int D1 = 32;
    int D2 = 32;

    std::vector<int64_t> srcShape = {S0, S1, S2};
    std::vector<int64_t> dstShape = {D0, D1, D2};

    int srcCapacity = srcShape[0] * srcShape[1] * srcShape[2];
    int dstCapacity = dstShape[0] * dstShape[1] * dstShape[2];

    uint64_t outputSize = dstCapacity * sizeof(uint16_t);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    PROGRAM("LOGICALNOT") {
        void *x_ptr = readToDev(GetGoldenDir() + "/logicalnotdim3_x.bin", srcCapacity);
        TileShape::Current().SetVecTile({8, 16, 16});
        Tensor input_a(DataType::DT_FP16, srcShape, (uint8_t *)x_ptr, "A");
        Tensor output(DataType::DT_FP16, dstShape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("LOGICALNOT_T", {input_a, output}) {
            output = LogicalNot(input_a);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(dstCapacity);
    std::vector<npu::tile_fwk::float16> golden(dstCapacity);
    std::vector<npu::tile_fwk::float16> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/logicalnotdim3_golden.bin", golden);
    readInput(GetGoldenDir() + "/logicalnotdim3_x.bin", x);
    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
