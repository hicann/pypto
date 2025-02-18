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
 * \file test_vecdup.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class VecdupTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(VecdupTest, TestVecDup) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_FP32, 2.0);
    int outputCapacity = 32 * 32;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("VECDUP") {
        TileShape::Current().SetVecTile({16, 1, 16});

        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("VECDUP_T", {output}) {
                output = npu::tile_fwk::Full(src, DT_FP32, shape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}

TEST_F(VecdupTest, TestVecDupUnaligned) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    std::vector<int64_t> shape{2, 2, 256, 7};
    Element src(DataType::DT_FP32, 2.0);
    int outputCapacity = 2 * 2 * 256 * 7;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("VECDUP") {
        TileShape::Current().SetVecTile({1, 1, 256, 16});

        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("VECDUP_T", {output}) {
                output = npu::tile_fwk::Full(src, DT_FP32, shape);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/res.bin", golden);

    int ret = resultCmp(golden, res, 0.003f);
    EXPECT_EQ(ret, true);
}
