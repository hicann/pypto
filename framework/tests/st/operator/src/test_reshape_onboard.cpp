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
 * \file test_reshape_onboard.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/llama/llama_def.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(OnBoardTest, test_operation_gm_reshape) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    const int capacity_8_8_8 = 8 * 8 * 8;
    uint64_t outputSize = capacity_8_8_8 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {8, 8, 8};
        void *x_r_ptr = readToDev(GetGoldenDir() + "/reshapegm_x_r.bin", capacity_8_8_8);
        // void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_8_8);
        void *y_ptr = readToDev(GetGoldenDir() + "/reshapegm_y.bin", capacity_8_8_8);
        TileShape::Current().SetVecTile({4, 16, 40});
        Tensor input_a_r(DataType::DT_FP32, {64, 8}, (uint8_t *)x_r_ptr, "A_R");
        // Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");

        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("DIV_T", {input_a_r, input_b, output}) {
            Tensor input_a;
            input_a = Reshape(input_a_r, shape);
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_8_8);
    std::vector<float> res(capacity_8_8_8);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/reshapegm_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_ub_reshape) {
    //1 to 1
    aclInit(nullptr);

    rtSetDevice(GetDeviceIdByEnvVar());
    const int capacity_8_8_8 = 8 * 8 * 16;
    uint64_t outputSize = capacity_8_8_8 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {8, 8, 16};
        void *x_r_ptr = readToDev(GetGoldenDir() + "/reshapeub_x_r.bin", capacity_8_8_8);
        // void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_8_8);
        void *y_ptr = readToDev(GetGoldenDir() + "/reshapeub_y.bin", capacity_8_8_8);
        TileShape::Current().SetVecTile({64, 8, 8});
        Tensor input_a_r(DataType::DT_FP32, {64, 16}, (uint8_t *)x_r_ptr, "A_R");
        // Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");

        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("DIV_T",  {input_a_r, input_b, output}) {
            Tensor input_a_r_exp;
            input_a_r_exp= Exp(input_a_r);
            Tensor input_a;
            input_a = Reshape(input_a_r_exp, shape);
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_8_8);
    std::vector<float> res(capacity_8_8_8);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/reshapeub_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_gm_reshape_2dimto3dim) {
    // multi to 1
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    const int capacity = 16 * 8 * 8;
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {16, 8 ,8};
        void *x_r_ptr = readToDev(GetGoldenDir() + "/reshapeub_x_r.bin", capacity);
        // void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity);
        void *y_ptr = readToDev(GetGoldenDir() + "/reshapeub_y.bin", capacity);
        TileShape::Current().SetVecTile({8, 32, 32});
        Tensor input_a_r(DataType::DT_FP32, {16, 64}, (uint8_t *)x_r_ptr, "A_R");
        // Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");

        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("DIV_T",  {input_a_r, input_b, output}) {
            Tensor input_a_r_exp;
            input_a_r_exp= Exp(input_a_r);
            Tensor input_a;
            input_a = Reshape(input_a_r_exp, shape);
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/reshapeub_res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_ub_reshape_3dimto2dim) {
    //1 to multi
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    const int capacity = 32 * 8 * 8;
    uint64_t outputSize = capacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {32, 64};
        void *x_r_ptr = readToDev(GetGoldenDir() + "/reshapeub_x_r.bin", capacity);
        // void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity);
        void *y_ptr = readToDev(GetGoldenDir() + "/reshapeub_y.bin", capacity);
        TileShape::Current().SetVecTile({16, 32, 8, 8});
        Tensor input_a_r(DataType::DT_FP32, {32, 8 ,8}, (uint8_t *)x_r_ptr, "A_R");
        // Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("DIV_T", {input_a_r, input_b, output}) {
            Tensor input_a_r_exp;
            input_a_r_exp= Exp(input_a_r);
            Tensor input_a;
            input_a = Reshape(input_a_r_exp, shape);
            output = Div(input_a, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);

    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/reshapeub_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_operation_ub_withoutreshape_3dimto2dim) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    const int capacity_8_8_8 = 16 * 8 * 8;
    uint64_t outputSize = capacity_8_8_8 * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);
    PROGRAM("DIV") {
        std::vector<int64_t> shape = {16, 64};
        void *x_r_ptr = readToDev(GetGoldenDir() + "/reshapeub_x_r.bin", capacity_8_8_8);
        // void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", capacity_8_8_8);
        void *y_ptr = readToDev(GetGoldenDir() + "/reshapeub_y.bin", capacity_8_8_8);
        TileShape::Current().SetVecTile({16, 32, 8, 8});
        Tensor input_a_r(DataType::DT_FP32, {16, 64}, (uint8_t *)x_r_ptr, "A_R");
        // Tensor input_a(DataType::DT_FP32, shape, (uint8_t *)x_ptr, "A");
        Tensor input_b(DataType::DT_FP32, shape, (uint8_t *)y_ptr, "B");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "C");

        config::SetBuildStatic(true);
        FUNCTION("DIV_T", {input_a_r, input_b, output}) {
            Tensor input_a_r_exp;
            input_a_r_exp= Exp(input_a_r);
            // Tensor input_a;
            // input_a = Reshape(input_a_r_exp, shape);
            output = Div(input_a_r_exp, input_b);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(capacity_8_8_8);
    std::vector<float> res(capacity_8_8_8);

    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/reshapeub_res.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_reshape_matmul_mul) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int m = 64;
    int k = 64;
    int n = 64;
    int a_size = m * k;
    int c_size = m * n;
    std::vector<int64_t> a_shape = {m, k};
    std::vector<int64_t> b_shape = {k, n};
    std::vector<int64_t> c_shape = {m, n};
    std::vector<float> golden(c_size);
    void *a_r_ptr = readToDev(GetGoldenDir() + "/a_r.bin", a_size);
    void *b_ptr = readToDev(GetGoldenDir() + "/b.bin", a_size);
    void *e_ptr = readToDev(GetGoldenDir() + "/e.bin", a_size);
    readInput(GetGoldenDir() + "/res.bin", golden);
    uint64_t outputSize = c_size * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("RESHAPE_WITH_MM") {
        config::Reset();
        constexpr AttentionCubeTileConfig TEMP_DFS_CUBE_CFG = {32, 32, 32, 32, 32, 32, 32, 32};
        TileShape::Current().SetCubeTile({TEMP_DFS_CUBE_CFG.c1L0, TEMP_DFS_CUBE_CFG.c1L1M},
            {TEMP_DFS_CUBE_CFG.c1L0, TEMP_DFS_CUBE_CFG.c1L1K}, {TEMP_DFS_CUBE_CFG.c1L0, TEMP_DFS_CUBE_CFG.c1L1N});
        TileShape::Current().SetVecTile({32, 32, 32});

        Tensor A_R(DataType::DT_FP16, {8, 8, 64}, (uint8_t *)a_r_ptr, "A_R");
        Tensor B(DataType::DT_FP16, b_shape, (uint8_t *)b_ptr, "B");
        Tensor E(DataType::DT_FP32, {8, 8, 64}, (uint8_t *)e_ptr, "E");
        Tensor RES(DataType::DT_FP32, {8, 8, 64}, out_ptr, "RES");

        config::SetBuildStatic(true);
        FUNCTION("RESHAPE", {A_R, B, E, RES}) {
            Tensor A = Reshape(A_R, a_shape);
            Tensor C = Matrix::Matmul<false, false>(DataType::DT_FP32, A, B);
            Tensor D = Reshape(C, {8, 8, 64});
            RES = Mul(D, E);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    assert(outputSize == c_size * sizeof(float));
    std::vector<float> res(c_size);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
