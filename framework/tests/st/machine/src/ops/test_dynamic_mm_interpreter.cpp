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
 * \file test_dynamic_mm.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
const size_t MM_SHAPE_SIZE = 3;
const size_t MM_VIEW_SHAPE_SIZE = 2;
const size_t MM_SHAPE_N_IDX = 2;

class DynamicMatmulInterpreterTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

inline SymbolicScalar CeilDivSymbolicScalar(SymbolicScalar a, int b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename dtype>
Tensor constructMatmulTensor(const std::vector<int64_t> &shape, const string &name, bool isNz) {
    auto dataType = GetAstDtype<dtype>();
    if (isNz) {
        return Tensor(dataType, shape, name, TileOpFormat::TILEOP_NZ);
    }
    return Tensor(dataType, shape, name);
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NonSplitFunc(const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};

    FUNCTION("testNoSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = View(tensor_a, aShape, aValidShape, {mIdx, 0});
            Tensor dyn_b = View(tensor_b, bShape, bValidShape, {0, 0});
            tensor_c = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void MSplitFunc(
    const std::vector<int64_t> &viewShape, const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};

    FUNCTION("testMSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
            LoopRange(0, CeilDivSymbolicScalar(transA ? aShape[1] : aShape[0], viewShape[0]), 1)) {
            Tensor dyn_a;
            if (transA) {
                dyn_a = View(tensor_a, {aShape[0], viewShape[0]},
                    {aShape[0], std::min(aShape[1] - viewShape[0] * mIdx, viewShape[0])}, {0, mIdx * viewShape[0]});
            } else {
                dyn_a = View(tensor_a, {viewShape[0], aShape[1]},
                    {std::min(aShape[0] - viewShape[0] * mIdx, viewShape[0]), aShape[1]}, {mIdx * viewShape[0], 0});
            }
            Tensor dyn_b = View(tensor_b, bShape, bValidShape, {0, 0});
            Tensor res = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
            Assemble(res, {mIdx * viewShape[0], 0}, tensor_c);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NSplitFunc(
    const std::vector<int64_t> &viewShape, const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};

    FUNCTION("testNSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
            LoopRange(0, CeilDivSymbolicScalar(transB ? bShape[0] : bShape[1], viewShape[1]), 1)) {
            Tensor dyn_a = View(tensor_a, aShape, aValidShape, {0, 0});
            Tensor dyn_b;
            if (!transB) {
                dyn_b = View(tensor_b, {bShape[0], viewShape[1]},
                    {bShape[0], std::min(bShape[1] - viewShape[1] * nIdx, viewShape[1])}, {0, nIdx * viewShape[1]});
            } else {
                dyn_b = View(tensor_b, {viewShape[1], bShape[1]},
                    {std::min(bShape[0] - viewShape[1] * nIdx, viewShape[1]), bShape[1]}, {nIdx * viewShape[1], 0});
            }
            Tensor res = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
            Assemble(res, {0, nIdx * viewShape[1]}, tensor_c);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void MNSplitFunc(
    const std::vector<int64_t> &viewShape, const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};

    FUNCTION("testNSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
            LoopRange(0, CeilDivSymbolicScalar(transA ? aShape[1] : aShape[0], viewShape[0]), 1)) {
            LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
                LoopRange(0, CeilDivSymbolicScalar(transB ? bShape[0] : bShape[1], viewShape[1]), 1)) {
                Tensor dyn_a;
                if (transA) {
                    dyn_a = View(tensor_a, {aShape[0], viewShape[0]},
                        {aShape[0], std::min(aShape[1] - viewShape[0] * mIdx, viewShape[0])}, {0, mIdx * viewShape[0]});
                } else {
                    dyn_a = View(tensor_a, {viewShape[0], aShape[1]},
                        {std::min(aShape[0] - viewShape[0] * mIdx, viewShape[0]), aShape[1]}, {mIdx * viewShape[0], 0});
                }
                Tensor dyn_b;
                if (!transB) {
                    dyn_b = View(tensor_b, {bShape[0], viewShape[1]},
                        {bShape[0], std::min(bShape[1] - viewShape[1] * nIdx, viewShape[1])}, {0, nIdx * viewShape[1]});
                } else {
                    dyn_b = View(tensor_b, {viewShape[1], bShape[1]},
                        {std::min(bShape[0] - viewShape[1] * nIdx, viewShape[1]), bShape[1]}, {nIdx * viewShape[1], 0});
                }
                Tensor res = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
                Assemble(res, {mIdx * viewShape[0], nIdx * viewShape[1]}, tensor_c);
            }
        }
    }
}

template <typename inputDtype, typename outputDtype, bool transA, bool transB, bool isCNz>
void TestDynMatmul(
    const std::vector<int64_t>& mmShape, bool isANz, bool isBNz, const std::vector<int64_t> &viewShape, string dataPath) {

    SetInterpreterConfig();

    if (mmShape.size() != MM_SHAPE_SIZE || viewShape.size() != MM_VIEW_SHAPE_SIZE) {
        return;
    }

    int m = mmShape[0];
    int k = mmShape[1];
    int n = mmShape[MM_SHAPE_N_IDX];
    Tensor tensor_a = transA ? constructMatmulTensor<inputDtype>({k, m}, "tensor_a", isANz) :
                               constructMatmulTensor<inputDtype>({m, k}, "tensor_a", isANz);
    Tensor tensor_b = transB ? constructMatmulTensor<inputDtype>({n, k}, "tensor_b", isBNz) :
                               constructMatmulTensor<inputDtype>({k, n}, "tensor_b", isBNz);
    Tensor tensor_c = constructMatmulTensor<outputDtype>({m, n}, "tensor_c", isCNz);

    int viewM = viewShape[0];
    int viewN = viewShape[1];

    std::vector<inputDtype> aData(m * k, 0);
    std::vector<inputDtype> bData(k * n, 0);
    std::vector<outputDtype> golden(m * n, 0);

    readInput<inputDtype>(dataPath + "/mat_a.bin", aData);
    readInput<inputDtype>(dataPath + "/mat_b.bin", bData);
    readInput<outputDtype>(dataPath + "/mat_c.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<inputDtype>(tensor_a, aData),
        RawTensorData::CreateTensor<inputDtype>(tensor_b, bData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<outputDtype>(tensor_c, 0.0f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<outputDtype>(tensor_c, golden),
    });

    if (viewM > 0 && viewN > 0) {
        MNSplitFunc<outputDtype, transA, transB, isCNz>(viewShape, tensor_a, tensor_b, tensor_c);
    } else if (viewM > 0) {
        MSplitFunc<outputDtype, transA, transB, isCNz>(viewShape, tensor_a, tensor_b, tensor_c);
    } else if (viewN > 0) {
        NSplitFunc<outputDtype, transA, transB, isCNz>(viewShape, tensor_a, tensor_b, tensor_c);
    } else {
        NonSplitFunc<outputDtype, transA, transB, isCNz>(tensor_a, tensor_b, tensor_c);
    }

    // // excute
    // DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    // auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    // EXPECT_TRUE(resultCmp(golden, (outputDtype *)outs->data(), 0.001f));
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_B_ND_bf16) {
    TileShape::Current().SetCubeTile({64, 64}, {128, 128}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {96, -1};
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false, false>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_B_NZ_bf16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 32;
    int n = 512;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false, false>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_Bt_ND_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 128;
    int k = 257;
    int n = 511;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, 256};
    TestDynMatmul<npu::tile_fwk::float16, float, false, true, false>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_Bt_NZ_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 1;
    int k = 512;
    int n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<npu::tile_fwk::float16, float, false, true, false>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_B_NZ_int8) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 32;
    int n = 512;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<int8_t, int32_t, false, false, false>({m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_Bt_NZ_int8) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 1;
    int k = 512;
    int n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<int8_t, int32_t, false, true, false>({m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_B_ND_bf16_tile1) {
    TileShape::Current().SetCubeTile({64, 64}, {256, 256}, {128, 128});
    int m = 128;
    int k = 256;
    int n = 512;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {90, 256};
    TestDynMatmul<npu::tile_fwk::bfloat16, float, false, false, false>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_Bt_ND_fp16_tile2) {
    TileShape::Current().SetCubeTile({32, 32}, {512, 512}, {32, 32});
    int m = 16;
    int k = 512;
    int n = 512;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<npu::tile_fwk::float16, float, false, true, false>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_B_NZ_int8_tile3) {
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {512, 512});
    int m = 16;
    int k = 32;
    int n = 512;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<int8_t, int32_t, false, false, false>({m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_Bt_NZ_int8_tile4) {
    TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {32, 32});
    int m = 1;
    int k = 512;
    int n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<int8_t, int32_t, false, true, false>({m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicMatmulInterpreterTest, mm_A_ND_B_ND_C_NZ) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 16;
    int k = 192;
    int n = 128;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    TestDynMatmul<npu::tile_fwk::float16, float, false, false, true>(
        {m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}
} // namespace