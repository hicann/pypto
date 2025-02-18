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
 * \file test_onboard_gather_element.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class GatherElementOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(GatherElementOnBoardTest, test_gather_element_float_16_70_8_40_1) {
    int S0 = 16;
    int S1 = 70;
    int D0 = 8;
    int D1 = 40;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {D0, D1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GatherElement") {
        void *params_ptr = readToDev(GetGoldenDir() + "/params.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        TileShape::Current().SetVecTile({4, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)params_ptr, "params");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_ELEMET_T", {input_src0, input_src1, output}) {
            output = GatherElements(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(GatherElementOnBoardTest, test_gather_element_float_16_64_8_32_1) {
    int S0 = 16;
    int S1 = 64;
    int D0 = 8;
    int D1 = 32;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {D0, D1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GatherElement") {
        void *params_ptr = readToDev(GetGoldenDir() + "/params.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        TileShape::Current().SetVecTile({4, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)params_ptr, "params");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_ELEMET_T", {input_src0, input_src1, output}) {
            output = GatherElements(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(GatherElementOnBoardTest, test_gather_element_float_16_64_7_32_1) {
    int S0 = 16;
    int S1 = 64;
    int D0 = 7;
    int D1 = 32;

    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 1;
    std::vector<int64_t> shape2 = {D0, D1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GatherElement") {
        void *params_ptr = readToDev(GetGoldenDir() + "/params.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        TileShape::Current().SetVecTile({4, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)params_ptr, "params");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_ELEMET_T", {input_src0, input_src1, output}) {
            output = GatherElements(input_src0, input_src1, axis);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity2);
    std::vector<float> dev_res(capacity2);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/res_golden.bin", golden);
    std::cout << "====== output size:" << capacity2 << std::endl;

    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(GatherElementOnBoardTest, test_gather_element_float_16_64_7_32_0) {
    int S0 = 16;
    int S1 = 64;
    int D0 = 7;
    int D1 = 32;
    std::vector<int64_t> shape0 = {S0, S1};
    std::vector<int64_t> shape1 = {D0, D1};
    int axis = 0;
    std::vector<int64_t> shape2 = {D0, D1};

    int capacity0 = shape0[0] * shape0[1];
    int capacity1 = shape1[0] * shape1[1];
    int capacity2 = shape2[0] * shape2[1];

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity2 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("GatherElement") {
        void *params_ptr = readToDev(GetGoldenDir() + "/params.bin", capacity0);
        void *indices_ptr = readToDev(GetGoldenDir() + "/indices.bin", capacity1);
        TileShape::Current().SetVecTile({4, 32});

        Tensor input_src0(DataType::DT_FP32, shape0, (uint8_t *)params_ptr, "params");
        Tensor input_src1(DataType::DT_INT32, shape1, (uint8_t *)indices_ptr, "indices");
        Tensor output(DataType::DT_FP32, shape2, out_ptr, "output");

        config::SetBuildStatic(true);
        FUNCTION("GATHER_ELEMET_T", {input_src0, input_src1, output}) {
            output = GatherElements(input_src0, input_src1, axis);
        }
        DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

        std::vector<float> golden(capacity2);
        std::vector<float> dev_res(capacity2);
        machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), (uint8_t *)out_ptr, outputSize);
        readInput(GetGoldenDir() + "/res_golden.bin", golden);
        std::cout << "====== output size:" << capacity2 << std::endl;

        int ret = resultCmp(golden, dev_res, 0.001f);
        EXPECT_EQ(ret, true);
    }
}
