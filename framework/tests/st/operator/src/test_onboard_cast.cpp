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
 * \file test_onboard_cast.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class CastOnBoard : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {
    // Test configuration structure to reduce function parameters
    struct CastTestConfig {
        std::vector<int64_t> shape;
        DataType inputType;
        DataType outputType;
        std::vector<int64_t> tileShapes;
        std::string inputFile;
        std::string goldenFile;
        CastMode castMode;
    };

    // Helper function to eliminate duplicate cast test code
    template<typename InputType, typename OutputType>
    void RunCastTest(const CastTestConfig& config) {
        aclInit(nullptr);
        rtSetDevice(GetDeviceIdByEnvVar());

        int dstCapacity = config.shape[0] * config.shape[1];
        int srcCapacity = config.shape[0] * config.shape[1];
        uint64_t outputSize = dstCapacity * sizeof(OutputType);
        uint8_t* out_ptr = allocDevAddr(outputSize);

        PROGRAM("Cast") {
            void *x_ptr = readToDev(GetGoldenDir() + "/" + config.inputFile, dstCapacity);
            TileShape::Current().SetVecTile(config.tileShapes);
            Tensor i_x(config.inputType, config.shape, (uint8_t *)x_ptr, "x");
            Tensor o_x(config.outputType, config.shape, out_ptr, "cast");

            config::SetBuildStatic(true);
            FUNCTION("CAST_test", {i_x, o_x}) {
                o_x = Cast(i_x, config.outputType, config.castMode);
            }
        }
        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

        std::vector<InputType> x(srcCapacity);
        std::vector<OutputType> golden(dstCapacity);
        std::vector<OutputType> res(dstCapacity);
        machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
        readInput(GetGoldenDir() + "/" + config.inputFile, x);
        readInput(GetGoldenDir() + "/" + config.goldenFile, golden);
        int ret = resultCmpCast<InputType, OutputType>(x, golden, res, 0.001f);
        EXPECT_EQ(ret, true);
    }
}

TEST_F(CastOnBoard, test_cast_fp32toint32rint_1_4608) {
    CastTestConfig config = {
        {1, 4608},
        DataType::DT_FP32,
        DataType::DT_INT32,
        {1, 32},
        "fp32toint32rint_x.bin",
        "fp32toint32rint_golden.bin",
        CAST_RINT
    };
    RunCastTest<float, int32_t>(config);
}

TEST_F(CastOnBoard, test_cast_int32tofp16none_1_4608) {
    CastTestConfig config = {
        {1, 4608},
        DataType::DT_INT32,
        DataType::DT_FP16,
        {1, 16},
        "int32tofp16none_x.bin",
        "int32tofp16none_golden.bin",
        CAST_NONE
    };
    RunCastTest<int32_t, npu::tile_fwk::float16>(config);
}

TEST_F(CastOnBoard, test_cast_fp16toint8trunc_1_4608) {
    CastTestConfig config = {
        {1, 4608},
        DataType::DT_FP16,
        DataType::DT_INT8,
        {1, 32},
        "fp16toint8trunc_x.bin",
        "fp16toint8trunc_golden.bin",
        CAST_TRUNC
    };
    RunCastTest<npu::tile_fwk::float16, int8_t>(config);
}

TEST_F(CastOnBoard, test_cast_fp16tofp32_unalign) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {4, 130};
    DataType iType = DataType::DT_FP16;
    DataType oType = DataType::DT_FP32;
    int dstCapacity = shape[0] * shape[1];
    int srcCapacity = shape[0] * shape[1];

    uint64_t outputSize = dstCapacity * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);

    PROGRAM("Cast") {
        void *x_ptr1 = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/fp16tofp32_unalign_x1.bin", srcCapacity);
        Tensor input1(iType, shape, (uint8_t *)x_ptr1, "x_ptr1");
        Tensor output(oType, shape, (uint8_t *)out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("CAST_test", {input1, output}) {
            TileShape::Current().SetVecTile({4, 256});
            output = Cast(input1, oType, CAST_NONE);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<npu::tile_fwk::float16> x1(srcCapacity);
    std::vector<float> golden(dstCapacity);
    std::vector<float> res(dstCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/fp16tofp32_unalign_golden.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(CastOnBoard, test_cast_fp32toint32rint_1_16384) {
    CastTestConfig config = {
        {1, 16384},
        DataType::DT_FP32,
        DataType::DT_INT32,
        {1, 16384},
        "fp32toint32rint_x.bin",
        "fp32toint32rint_golden.bin",
        CAST_RINT
    };
    RunCastTest<float, int32_t>(config);
}
