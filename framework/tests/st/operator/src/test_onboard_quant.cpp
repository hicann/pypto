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
 * \file test_onboard_quant.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
#include "test_cost_macro.h"

using namespace npu::tile_fwk;
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr int VALUE8 = 8;
constexpr int VALUE32 = 32;
constexpr int VALUE64 = 64;
constexpr int VALUE128 = 128;

class QuantOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

void TestQuant(std::vector<int64_t>& inputShape) {
    int shapeDim = inputShape.size();
    std::vector<int64_t> scaleShape(shapeDim,0);
    for (int i = 0; i < shapeDim; i++) {
        scaleShape[i] = (i == shapeDim - 1) ? 1 : inputShape[i];
    }

    uint64_t capacity = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<>());
    uint64_t capacityScale = std::accumulate(scaleShape.begin(), scaleShape.end(), 1, std::multiplies<>());

    std::vector<int64_t> vecTileShape  = {VALUE128, VALUE128};

    // depend on shapeDim
    switch (shapeDim) {
        case DIM2: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]); break;
        case DIM3: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[0], vecTileShape[1]); break;
        case DIM4: TileShape::Current().SetVecTile(1, 1, vecTileShape[0], vecTileShape[1]); break;
        default:
            ASSERT(true) << "unsupport dim "<< shapeDim<<" \n";
            break;
    }

    typedef int8_t dtype;
    // typedef float dtype;

    uint64_t outSize = capacity * sizeof(dtype);
    uint64_t scaleSize = capacityScale * sizeof(float);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    uint8_t* out_ptr = allocDevAddr(outSize);
    uint8_t* scale_ptr = allocDevAddr(scaleSize);

    PROGRAM("Quant") {
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP16, inputShape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_INT8, inputShape, out_ptr, "output");
        Tensor scaleDeQuant(DataType::DT_FP32, scaleShape, scale_ptr, "scaleDeQuant");

        config::SetBuildStatic(true);
        FUNCTION("Quant", {input, output, scaleDeQuant}) {
            auto res = Quant(input);
            output = std::get<0>(res);
            scaleDeQuant = std::get<1>(res);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<dtype> output_golden(capacity);
    std::vector<dtype> output_npu(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)output_npu.data(), (uint8_t *)out_ptr, outSize);
    readInput(GetGoldenDir() + "/output_golden.bin", output_golden);
    int ret0 = resultCmp<dtype>(output_golden, output_npu, 0.0001f);

    std::vector<float> scale_golden(capacityScale);
    std::vector<float> scale_npu(capacityScale);
    machine::GetRA()->CopyFromTensor((uint8_t *)scale_npu.data(), (uint8_t *)scale_ptr, scaleSize);
    readInput(GetGoldenDir() + "/scale_dequant_golden.bin", scale_golden);
    int ret1 = resultCmp<float>(scale_golden, scale_npu, 0.0001f);

    EXPECT_EQ(ret0 && ret1, true);
}

void TestQuant3D(std::vector<int64_t>& inputShape) {
    int shapeDim = inputShape.size();
    std::vector<int64_t> scaleShape(shapeDim,0);
    for (int i = 0; i < shapeDim; i++) {
        scaleShape[i] = (i == shapeDim - 1) ? 1 : inputShape[i];
    }

    uint64_t capacity = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<>());
    uint64_t capacityScale = std::accumulate(scaleShape.begin(), scaleShape.end(), 1, std::multiplies<>());

    std::vector<int64_t> vecTileShape  = {VALUE8, VALUE8, VALUE32};

    // depend on shapeDim
    switch (shapeDim) {
        case DIM2: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]); break;
        case DIM3: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1], vecTileShape[2]); break;
        case DIM4: TileShape::Current().SetVecTile(1, 1, vecTileShape[0], vecTileShape[1]); break;
        default:
            ASSERT(true) << "unsupport dim "<< shapeDim<<" \n";
            break;
    }

    typedef int8_t dtype;
    // typedef float dtype;

    uint64_t outSize = capacity * sizeof(dtype);
    uint64_t scaleSize = capacityScale * sizeof(float);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    uint8_t* out_ptr = allocDevAddr(outSize);
    uint8_t* scale_ptr = allocDevAddr(scaleSize);

    PROGRAM("Quant") {
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_BF16, inputShape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_INT8, inputShape, out_ptr, "output");
        Tensor scaleDeQuant(DataType::DT_FP32, scaleShape, scale_ptr, "scaleDeQuant");

        config::SetBuildStatic(true);
        FUNCTION("Quant", {input, output, scaleDeQuant}) {
            auto res = Quant(input);
            output = std::get<0>(res);
            scaleDeQuant = std::get<1>(res);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<dtype> output_golden(capacity);
    std::vector<dtype> output_npu(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)output_npu.data(), (uint8_t *)out_ptr, outSize);
    readInput(GetGoldenDir() + "/output_golden.bin", output_golden);
    int ret0 = resultCmp<dtype>(output_golden, output_npu, 0.0001f);

    std::vector<float> scale_golden(capacityScale);
    std::vector<float> scale_npu(capacityScale);
    machine::GetRA()->CopyFromTensor((uint8_t *)scale_npu.data(), (uint8_t *)scale_ptr, scaleSize);
    readInput(GetGoldenDir() + "/scale_dequant_golden.bin", scale_golden);
    int ret1 = resultCmp<float>(scale_golden, scale_npu, 0.0001f);

    EXPECT_EQ(ret0 && ret1, true);
}

void TestQuantWithSmoothFactor(std::vector<int64_t>& inputShape) {
    int shapeDim = inputShape.size();
    std::vector<int64_t> scaleShape(shapeDim,0);
    for (int i = 0; i < shapeDim; i++) {
        scaleShape[i] = (i == shapeDim - 1) ? 1 : inputShape[i];
    }
    std::vector<int64_t> smoothFactorShape = {1, inputShape[shapeDim - 1]};
    uint64_t capacity = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<>());
    uint64_t capacityScale = std::accumulate(scaleShape.begin(), scaleShape.end(), 1, std::multiplies<>());
    uint64_t capacitySmoothFactor = std::accumulate(smoothFactorShape.begin(), smoothFactorShape.end(), 1, std::multiplies<>());

    std::vector<int64_t> vecTileShape  = {VALUE32, VALUE128};

    // depend on shapeDim
    switch (shapeDim) {
        case DIM2: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]); break;
        case DIM3: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[0], vecTileShape[1]); break;
        case DIM4: TileShape::Current().SetVecTile(1, 1, vecTileShape[0], vecTileShape[1]); break;
        default:
            ASSERT(true) << "unsupport dim "<< shapeDim<<" \n";
            break;
    }

    typedef int8_t dstType;

    uint64_t outSize = capacity * sizeof(dstType);
    uint64_t scaleSize = capacityScale * sizeof(float);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    uint8_t* out_ptr = allocDevAddr(outSize);
    uint8_t* scale_ptr = allocDevAddr(scaleSize);
    PROGRAM("Quant") {
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        void *smooth_factor_ptr_ = readToDev(GetGoldenDir() + "/smooth_factor.bin", capacitySmoothFactor);
        Tensor input(DataType::DT_FP16, inputShape, (uint8_t *)input_ptr, "input");
        Tensor smoothFactor(DataType::DT_FP32, smoothFactorShape, (uint8_t *)smooth_factor_ptr_, "smoothFactor");
        Tensor output(DataType::DT_INT8, inputShape, out_ptr, "output");
        Tensor scaleDeQuant(DataType::DT_FP32, scaleShape, scale_ptr, "scaleDeQuant");

        config::SetBuildStatic(true);
        FUNCTION("Quant", {input, output, smoothFactor, scaleDeQuant}) {
            auto res = Quant(input, true, true, smoothFactor);
            output = std::get<0>(res);
            scaleDeQuant = std::get<1>(res);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<dstType> output_golden(capacity);
    std::vector<dstType> output_npu(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)output_npu.data(), (uint8_t *)out_ptr, outSize);
    readInput(GetGoldenDir() + "/output_golden.bin", output_golden);
    int ret0 = resultCmp<dstType>(output_golden, output_npu, 0.0001f);

    std::vector<float> scale_golden(capacityScale);
    std::vector<float> scale_npu(capacityScale);
    machine::GetRA()->CopyFromTensor((uint8_t *)scale_npu.data(), (uint8_t *)scale_ptr, scaleSize);
    readInput(GetGoldenDir() + "/scale_dequant_golden.bin", scale_golden);
    int ret1 = resultCmp<float>(scale_golden, scale_npu, 0.0001f);

    EXPECT_EQ(ret0 && ret1, true);
}

TEST_F_WITH_COST(QuantOnBoardTest, test_Quant_32_1_7168, 17) {
    std::vector<int64_t> inputShape = {32, 1, 7168};
    TestQuant(inputShape);
}

TEST_F(QuantOnBoardTest, test_Quant_32_7168) {
   std::vector<int64_t> inputShape = {32, 7168};
    TestQuant(inputShape);
}

TEST_F(QuantOnBoardTest, test_Quant_Smooth_32_4_128) {
    std::vector<int64_t> inputShape = {32, 4, 128};
    TestQuantWithSmoothFactor(inputShape);
}

TEST_F(QuantOnBoardTest, test_Quant_Smooth_32_7168) {
   std::vector<int64_t> inputShape = {32, 7168};
    TestQuantWithSmoothFactor(inputShape);
}

class QuantMMOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

void TestQuantMM(std::vector<int64_t>& shapeA, std::vector<int64_t>& shapeW) {
    assert (shapeA.size() == DIM2);
    int m = shapeA[0];
    int k = shapeA[1];
    int n = shapeW[1];
    std::vector<int64_t> shapeScaleW = {1, n};
    std::vector<int64_t> shapeRes = {m, k};
    int capacityA = m * k;
    int capacityW = k * n;
    int capacityScaleW = 1 * n;
    int capacityRes = m * n;
    typedef npu::tile_fwk::bfloat16 srcAType;
    typedef int8_t srcWType;
    typedef float srcScaleWType;
    typedef npu::tile_fwk::bfloat16 dstType;
    void *matA_ptr = readToDev<srcAType>(GetGoldenDir() + "/quant_mm_a.bin", capacityA);
    void *matW_ptr = readToDev<srcWType>(GetGoldenDir() + "/quant_mm_w.bin", capacityW);
    void *matScaleW_ptr = readToDev<srcScaleWType>(GetGoldenDir() + "/quant_mm_scale_w.bin", capacityScaleW);
    assert(matA_ptr != nullptr && matW_ptr != nullptr && matScaleW_ptr != nullptr);
    uint32_t outputSize = capacityRes * sizeof(dstType);
    uint8_t* matRes_ptr = allocDevAddr(outputSize);

    PROGRAM("QUANTMM") {
        config::Reset();
        std::vector<int64_t> vecTileShape  = {VALUE32, VALUE64};
        TileShape::Current().SetCubeTile({VALUE32, VALUE32}, {VALUE128, VALUE128}, {VALUE128, VALUE128});
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        Tensor matA(DataType::DT_BF16, shapeA, (uint8_t *)matA_ptr, "MatA");
        Tensor matW(DataType::DT_INT8, shapeW, (uint8_t *)matW_ptr, "MatW");
        Tensor matScaleW(DataType::DT_FP32, shapeScaleW, (uint8_t *)matScaleW_ptr, "MatScaleW");
        Tensor matRes(DataType::DT_BF16, shapeRes, matRes_ptr, "MatRes");
        config::SetBuildStatic(true);
        FUNCTION("QUANTMM", {matA, matW, matScaleW, matRes}) {
            matRes = npu::tile_fwk::Matrix::QuantMM(matA, matW, matScaleW);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<dstType> res(capacityRes);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), matRes_ptr, outputSize);
    std::vector<dstType> golden(capacityRes);
    readInput(GetGoldenDir() + "/quant_mm_golden.bin", golden);
    std::vector<srcAType> a(capacityA);
    readInput(GetGoldenDir() + "/quant_mm_a.bin", a);
    std::vector<srcWType> w(capacityW);
    readInput(GetGoldenDir() + "/quant_mm_w.bin", w);
    std::vector<srcScaleWType> scaleW(capacityScaleW);
    readInput(GetGoldenDir() + "/quant_mm_scale_w.bin", scaleW);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

void TestQuantMM3D(std::vector<int64_t>& shapeA, std::vector<int64_t>& shapeW) {
    assert (shapeA.size() == DIM3);
    int m = shapeA[1];
    int k = shapeA[2];
    int n = shapeW[2];
    int b = shapeA[0];
    std::vector<int64_t> shapeScaleW = {b, 1, n};
    std::vector<int64_t> shapeRes = {b, m, n};

    int capacityA = b * m * k;
    int capacityW = b * k * n;
    int capacityScaleW = b * 1 * n;
    int capacityRes = b * m * n;
    typedef npu::tile_fwk::bfloat16 srcAType;
    typedef int8_t srcWType;
    typedef float srcScaleWType;
    typedef npu::tile_fwk::bfloat16 dstType;
    void *matA_ptr = readToDev<srcAType>(GetGoldenDir() + "/quant_mm_a.bin", capacityA);
    void *matW_ptr = readToDev<srcWType>(GetGoldenDir() + "/quant_mm_w.bin", capacityW);
    void *matScaleW_ptr = readToDev<srcScaleWType>(GetGoldenDir() + "/quant_mm_scale_w.bin", capacityScaleW);
    assert(matA_ptr != nullptr && matW_ptr != nullptr && matScaleW_ptr != nullptr);
    uint32_t outputSize = capacityRes * sizeof(dstType);
    uint8_t* matRes_ptr = allocDevAddr(outputSize);

    PROGRAM("QUANTMM") {
        config::Reset();
        TileShape::Current().SetCubeTile({VALUE32, VALUE32}, {VALUE128, VALUE128}, {VALUE128, VALUE128});
        TileShape::Current().SetVecTile(VALUE8, VALUE8, VALUE32);
        Tensor matA(DataType::DT_BF16, shapeA, (uint8_t *)matA_ptr, "MatA");
        Tensor matW(DataType::DT_INT8, shapeW, (uint8_t *)matW_ptr, "MatW");
        Tensor matScaleW(DataType::DT_FP32, shapeScaleW, (uint8_t *)matScaleW_ptr, "MatScaleW");
        Tensor matRes(DataType::DT_BF16, shapeRes, matRes_ptr, "MatRes");
        config::SetBuildStatic(true);
        FUNCTION("QUANTMM", {matA, matW, matScaleW, matRes}) {
            matRes = npu::tile_fwk::Matrix::QuantMM(matA, matW, matScaleW);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<dstType> res(capacityRes);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), matRes_ptr, outputSize);
    std::vector<dstType> golden(capacityRes);
    readInput(GetGoldenDir() + "/quant_mm_golden.bin", golden);
    std::vector<srcAType> a(capacityA);
    readInput(GetGoldenDir() + "/quant_mm_a.bin", a);
    std::vector<srcWType> w(capacityW);
    readInput(GetGoldenDir() + "/quant_mm_w.bin", w);
    std::vector<srcScaleWType> scaleW(capacityScaleW);
    readInput(GetGoldenDir() + "/quant_mm_scale_w.bin", scaleW);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F_WITH_COST(QuantMMOnBoardTest, test_QuantMM_32_16384_times_16384_7168_np, 18) {
    std::vector<int64_t> shapeA = {32, 16384};
    std::vector<int64_t> shapeW = {16384, 7168};
    TestQuantMM(shapeA, shapeW);
}

TEST_F(QuantMMOnBoardTest, test_QuantMM_128_32_512_times_128_512_128_torch) {
    std::vector<int64_t> shapeA = {128, 32, 512};
    std::vector<int64_t> shapeW = {128, 512, 128};
    TestQuantMM3D(shapeA, shapeW);
}
