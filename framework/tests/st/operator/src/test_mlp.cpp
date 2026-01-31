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
 * \file test_mlp.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class MlpTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

constexpr float F_1 = 1.0;
constexpr float F_NEGA_1 = -1.0;

TEST_F(MlpTest, test_16_7168_tileop)
{
    // 初始化
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    // 创建输入输出shape
    int b = 64;
    int s = 1;
    int h = 7168;
    int shape0 = b * s;
    int shape1 = h;
    int shape2 = 2048;

    std::vector<int64_t> hiddenStatesShape = {shape0, shape1};
    std::vector<int64_t> ffnwegiht = {shape1, shape2};
    std::vector<int64_t> outshape = {shape0, shape1};

    // 分配输入输出空间
    int inputCapacity = shape0 * shape1;
    int input1Capacity = shape1 * shape2;
    int outputCapacity = shape0 * shape1;
    uint64_t outputSize = outputCapacity * sizeof(float);
    uint8_t* out_ptr = allocDevAddr(outputSize);

    // 创建PROGRAM
    PROGRAM("MLP") {
        TileShape::Current().SetVecTile(32, 256);
        TileShape::Current().SetCubeTile({32, 32}, {128, 256}, {128, 128});

        // 构建输入矩阵
        void *x_ptr = readToDev<float>(GetGoldenDir() + "/hidden_states.bin", inputCapacity);
        Tensor hiddenStates(DataType::DT_FP32, hiddenStatesShape, (uint8_t *)x_ptr, "A");

        void *x1_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight1.bin", input1Capacity);
        Tensor ffnweigth1(DataType::DT_FP16, ffnwegiht, (uint8_t *)x1_ptr, "B");

        void *x2_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight2.bin", input1Capacity);
        Tensor ffnweigth2(DataType::DT_FP16, ffnwegiht, (uint8_t *)x2_ptr, "C");

        void *x3_ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight3.bin", input1Capacity);
        Tensor ffnweigth3(DataType::DT_FP16, ffnwegiht, (uint8_t *)x3_ptr, "D");

        Tensor output(DataType::DT_FP32, outshape, out_ptr, "E");

        config::SetBuildStatic(true);
        FUNCTION("MLP_T", {hiddenStates, ffnweigth1, ffnweigth2, ffnweigth3, output})     {
            auto castRes = Cast(hiddenStates, DataType::DT_FP16);
            auto gate = Matrix::Matmul(DataType::DT_FP32, castRes, ffnweigth1, false, false, true);

            // swish: x / (1 + e^(-x))
            auto swish = Mul(gate, Element(DataType::DT_FP32, F_NEGA_1));
            swish = Exp(swish);
            swish = Add(swish, Element(DataType::DT_FP32, F_1));
            swish = Div(gate, swish);

            // up_proj
            // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]
            auto up = Matrix::Matmul(DataType::DT_FP32, castRes, Cast(ffnweigth2, DataType::DT_FP16), false, false, true);
            swish = Mul(swish, up);
            auto swish_fp16 = Cast(swish, DataType::DT_FP16);

            // down_proj
            // [b*s, n*d*3] [n*d, n*d*3]^T => [b*s, n*d]
            output = Matrix::Matmul(DataType::DT_FP32, swish_fp16, Cast(ffnweigth3, DataType::DT_FP16), false, true, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(outputCapacity);
    std::vector<float> res(outputCapacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);

    readInput(GetGoldenDir() + "/final_out.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}
