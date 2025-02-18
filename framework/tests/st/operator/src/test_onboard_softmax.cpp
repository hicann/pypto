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
 * \file test_onboard_softmax.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class SoftmaxOnBoard : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> GetGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

TEST_F(SoftmaxOnBoard, test_softmax_cast_in) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 1, 128};
    DataType iType = DataType::DT_FP16;
    DataType oType = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];

    uint64_t outputSize = cap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Cast_In") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_softmax_cast_in.bin", cap);
        TileShape::Current().SetVecTile({1, 2, 1, 16});
        Tensor i_x(iType, shape, (uint8_t *)x_ptr, "x");
        Tensor o_x(oType, shape, out_ptr, "cast_out");

        config::SetBuildStatic(true);
        FUNCTION("Softmax_Cast_In", {i_x, o_x}) {
            o_x = Cast(i_x, oType);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(cap);
    std::vector<float> golden(cap);
    std::vector<float> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_softmax_cast_in.bin", x);
    readInput(GetGoldenDir() + "/softmax_cast_in.bin", golden);
    int ret = resultCmpCast<npu::tile_fwk::float16, float>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_cast_out) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 1, 128};
    DataType iType = DataType::DT_FP32;
    DataType oType = DataType::DT_FP16;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];

    uint64_t outputSize = cap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Cast_Out") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_softmax_cast_out.bin", cap);
        TileShape::Current().SetVecTile({1, 2, 1, 16});
        Tensor i_x(iType, shape, (uint8_t *)x_ptr, "x");
        Tensor o_x(oType, shape, out_ptr, "cast_out");

        config::SetBuildStatic(true);
        FUNCTION("Softmax_Cast_Out", {i_x, o_x}) {
            o_x = Cast(i_x, oType);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(cap);
    std::vector<npu::tile_fwk::float16> golden(cap);
    std::vector<npu::tile_fwk::float16> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_softmax_cast_out.bin", x);
    readInput(GetGoldenDir() + "/softmax_cast_out.bin", golden);
    int ret = resultCmpCast<float, npu::tile_fwk::float16>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_sum_single) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> ishape = {2, 2, 1, 128};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], 1};
    DataType dtype = DataType::DT_FP32;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_SumSingle") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_sum.bin", icap);
        TileShape::Current().SetVecTile({1, 2, 1, 8});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_sum");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_SUM_T", {i_x, o_x}) {
            o_x = Sum(i_x, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(icap);
    std::vector<float> golden(oCap);
    std::vector<float> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_sum.bin", x);
    readInput(GetGoldenDir() + "/softmax_sum.bin", golden);
    int ret = resultCmpUnary<float>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_max_single) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> ishape = {2, 2, 1, 128};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], 1};
    DataType dtype = DataType::DT_FP32;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_MaxSingle") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_max.bin", icap);
        TileShape::Current().SetVecTile({1, 2, 1, 8});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_max");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_MAX_T", {i_x, o_x}) {
            o_x = Amax(i_x, -1, true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(icap);
    std::vector<float> golden(oCap);
    std::vector<float> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_max.bin", x);
    readInput(GetGoldenDir() + "/softmax_max.bin", golden);
    int ret = resultCmpUnary<float>(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_exp) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> shape = {2, 2, 1, 128};
    DataType dtype = DataType::DT_FP32;
    int cap = shape[0] * shape[1] * shape[2] * shape[3];
    uint64_t outputSize = cap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Exp") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_exp.bin", cap);
        TileShape::Current().SetVecTile({1, 2, 1, 8});
        Tensor input_x(dtype, shape, (uint8_t *)x_ptr, "x");
        Tensor output(dtype, shape, out_ptr, "Softmax_Exp");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_EXP_T", {input_x, output}) {
            output = Exp(input_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(cap);
    std::vector<float> golden(cap);
    std::vector<float> res(cap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_exp.bin", x);
    readInput(GetGoldenDir() + "/softmax_exp.bin", golden);
    int ret = resultCmpUnary(x, golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_div) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> lshape = {2, 2, 1, 128};
    std::vector<int64_t> rshape = {2, 2, 1, 1};
    std::vector<int64_t> oshape = {2, 2, 1, 128};
    DataType dtype = DataType::DT_FP32;
    int lcap = lshape[0] * lshape[1] * lshape[2] * lshape[3];
    int rcap = rshape[0] * rshape[1] * rshape[2] * rshape[3];
    int ocap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = ocap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_Div") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_div.bin", lcap);
        void *y_ptr = readToDev(GetGoldenDir() + "/y_div.bin", rcap);
        TileShape::Current().SetVecTile({1, 2, 1, 8});
        Tensor input_x(dtype, lshape, (uint8_t *)x_ptr, "x");
        Tensor input_y(dtype, rshape, (uint8_t *)y_ptr, "y");
        Tensor output(dtype, oshape, out_ptr, "Softmax_Div");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_SUB_T", {input_x, input_y, output}) {
            output = Div(input_x, input_y);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(ocap);
    std::vector<float> res(ocap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/softmax_div.bin", golden);
    int ret = resultCmp<float>(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_sum_all) {
    // 初始化和设置deviceId
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    // 设置shape
    std::vector<int64_t> ishape = {2, 2, 1, 128};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    // 计算输入和输出元素数量
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    // 计算输出大小，分配输出在Dev上的内存
    uint64_t outputSize = oCap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    // 指定算子计算的数据类型
    DataType dtype = DataType::DT_FP16;
    PROGRAM("Softmax_Sum_All") {
        // 读取数据数据
        void *x_ptr = readToDev(GetGoldenDir() + "/x_sum_all.bin", icap);
        // 设置切分shape
        TileShape::Current().SetVecTile({1, 2, 1, 64});
        // 声明输入输出tensor
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_sum_all");
        // 指定当前Function为静态实现
        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_SUM_T", {i_x, o_x}) {
            // 调用Softmax计算函数
            o_x = SoftmaxNew(i_x);
        }
    } // Program结束的时候会自动触发编译
    // 上板执行
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(icap);
    std::vector<npu::tile_fwk::float16> golden(oCap);
    std::vector<npu::tile_fwk::float16> res(oCap);
    // 把输出数据从Device内存拷贝到Host内存
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_sum_all.bin", x);
    readInput(GetGoldenDir() + "/softmax_sum_all.bin", golden);
    // 比对计算结果
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_full_inference) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> ishape = {2, 2, 32, 256};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP16;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("Softmax_full_inference") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_full.bin", icap);
        TileShape::Current().SetVecTile({1, 1, 32, 256});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "Softmax_full_inference");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_FULL_INFERENCE_T", {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(icap);
    std::vector<npu::tile_fwk::float16> golden(oCap);
    std::vector<npu::tile_fwk::float16> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_full.bin", x);
    readInput(GetGoldenDir() + "/softmax_full_inference.bin", golden);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_deepseek) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> ishape = {4, 8, 1, 512};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP16;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(uint16_t);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("softmax_deepseek") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x_deepseek.bin", icap);
        TileShape::Current().SetVecTile({1, 2, 1, 256});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_deepseek");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_DEEPSEEK", {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<npu::tile_fwk::float16> x(icap);
    std::vector<npu::tile_fwk::float16> golden(oCap);
    std::vector<npu::tile_fwk::float16> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x_deepseek.bin", x);
    readInput(GetGoldenDir() + "/softmax_deepseek.bin", golden);
    int ret = resultCmpUnary<npu::tile_fwk::float16>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_flash_attention) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    std::vector<int64_t> ishape = {32, 32, 1, 256};
    std::vector<int64_t> oshape = {ishape[0], ishape[1], ishape[2], ishape[3]};
    DataType dtype = DataType::DT_FP32;
    int icap = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    int oCap = oshape[0] * oshape[1] * oshape[2] * oshape[3];
    uint64_t outputSize = oCap * sizeof(float);
    uint8_t *out_ptr = allocDevAddr(outputSize);
    PROGRAM("softmax_fa") {
        void *x_ptr = readToDev(GetGoldenDir() + "/x.bin", icap);
        TileShape::Current().SetVecTile({4, 4, 1, 64});
        Tensor i_x(dtype, ishape, (uint8_t *)x_ptr, "x");
        Tensor o_x(dtype, oshape, out_ptr, "softmax_deepseek");

        config::SetBuildStatic(true);
        FUNCTION("SOFTMAX_FA", {i_x, o_x}) {
            o_x = SoftmaxNew(i_x);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> x(icap);
    std::vector<float> golden(oCap);
    std::vector<float> res(oCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), out_ptr, outputSize);
    readInput(GetGoldenDir() + "/x.bin", x);
    readInput(GetGoldenDir() + "/softmax.bin", golden);
    int ret = resultCmpUnary<float>(x, golden, res, 0.001f, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(SoftmaxOnBoard, test_softmax_dyn) {
    config::SetHostOption(ONLY_CODEGEN, true);
    // 设置输入输出shape
    std::vector<int64_t> shape = {32, 32, 1, 256};
    // 指定计算数据类型
    DataType dtype = DataType::DT_FP32;
    // 声明输入输出Tensor
    Tensor input(dtype, shape, "input");
    Tensor output(dtype, shape, "output");

    // 准备输入和输出数据
    std::vector<float> goldenData = GetGoldenVec<float>(shape, "/softmax.bin");
    auto inputData = CreateTensorData<float>(input, "/x.bin");
    auto outputData = RawTensorData::CreateConstantTensor<float>(output, 0.0);
    std::vector<RawTensorDataPtr> inputDataList = {inputData};
    std::vector<RawTensorDataPtr> outputDataList = {outputData};

    // 调用Softmax动态实现函数
    SoftmaxDynamic(input, output);
#ifdef BUILD_WITH_CANN
    // 上板执行
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    // 比对计算结果
    EXPECT_TRUE(resultCmp<float>(goldenData, (float *)outputData->data(), 0.001f));
#endif
}
