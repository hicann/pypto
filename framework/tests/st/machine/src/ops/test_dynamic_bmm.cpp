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
 * \file test_dynamic_bmm.cpp
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
const size_t BMM_SHAPE_SIZE = 4;
const size_t BMM_VIEW_SHAPE_SIZE = 3;
const size_t BMM_SHAPE_N_DIM = 2;
const size_t BMM_SHAPE_N_IDX = 3;
class DynamicBatchMatmulTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename dtype>
Tensor constructMatmulTensor(const std::vector<int64_t> &shape, const string &name, bool isNz) {
    auto dataType = GetAstDtype<dtype>();
    return isNz ? Tensor(dataType, shape, name, TileOpFormat::TILEOP_NZ) : Tensor(dataType, shape, name);
}

inline SymbolicScalar CeilDivSymbolicScalar(SymbolicScalar a, int64_t b) {
    return b == 0 ? a : (a + b - 1) / b;
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NonSplitFunc(const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1], aShape[2]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1], bShape[2]};

    int64_t m = transA ? aShape[2] : aShape[1];
    int64_t k = transA ? aShape[1] : aShape[2];
    int64_t n = transB ? bShape[1] : bShape[2];

    FUNCTION("testNoSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = View(tensor_a, aShape, aValidShape, {0, mIdx, 0});
            Tensor dyn_b = View(tensor_b, bShape, bValidShape, {0, 0, 0});
            TileShape::Current().SetMatrixSize({m, k, n});
            tensor_c = Matrix::BatchMatmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void MSplitFunc(
    const std::vector<int64_t> &viewShape, const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1], aShape[2]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1], bShape[2]};

    int64_t n = transB ? bShape[1] : bShape[2];
    int64_t m = transA ? aShape[2] : aShape[1];
    int64_t k = transA ? aShape[1] : aShape[2];

    FUNCTION("testMSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
            LoopRange(0, CeilDivSymbolicScalar(transA ? aShape[BMM_SHAPE_N_DIM] : aShape[1], viewShape[1]), 1)) {
            Tensor dyn_a;
            if (transA) {
                dyn_a = View(tensor_a, {aShape[0], aShape[1], viewShape[1]},
                    {aShape[0], aShape[1], std::min(aShape[2] - viewShape[1] * mIdx, viewShape[1])}, {0, 0, mIdx * viewShape[1]});
            } else {
                dyn_a = View(tensor_a, {aShape[0], viewShape[1], aShape[2]},
                    {aShape[0], std::min(aShape[1] - viewShape[1] * mIdx, viewShape[1]), aShape[2]}, {0, mIdx * viewShape[1], 0});
            }
            Tensor dyn_b = View(tensor_b, bShape, bValidShape, {0, 0, 0});
            TileShape::Current().SetMatrixSize({m, k, n});
            Tensor res = Matrix::BatchMatmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
            Assemble(res, {0, mIdx * viewShape[0], 0}, tensor_c);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NSplitFunc(
    const std::vector<int64_t> &viewShape, const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1], aShape[2]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1], bShape[2]};

    int64_t m = transA ? aShape[2] : aShape[1];
    int64_t k = transA ? aShape[1] : aShape[2];
    int64_t n = transB ? bShape[1] : bShape[2];

    FUNCTION("testNSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
            LoopRange(0, CeilDivSymbolicScalar(transB ? bShape[1] : bShape[BMM_SHAPE_N_DIM], viewShape[BMM_SHAPE_N_DIM]), 1)) {
            Tensor dyn_a = View(tensor_a, aShape, aValidShape, {0, 0, 0});
            Tensor dyn_b;
            if (!transB) {
                dyn_b = View(tensor_b, {bShape[0], bShape[1], viewShape[2]},
                    {bShape[0], bShape[1], std::min(bShape[2] - viewShape[2] * nIdx, viewShape[2])}, {0, 0, nIdx * viewShape[2]});
            } else {
                dyn_b = View(tensor_b, {bShape[0], viewShape[2], bShape[2]},
                    {bShape[0], std::min(bShape[1] - viewShape[2] * nIdx, viewShape[2]), bShape[2]}, {0, nIdx * viewShape[2], 0});
            }
            TileShape::Current().SetMatrixSize({m, k, n});
            Tensor res = Matrix::BatchMatmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
            Assemble(res, {0, 0, nIdx * viewShape[2]}, tensor_c);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void MNSplitFunc(
    const std::vector<int64_t> &viewShape, const Tensor &tensor_a, const Tensor &tensor_b, Tensor &tensor_c) {
    const auto &aShape = tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1], aShape[2]};
    const auto &bShape = tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1], bShape[2]};

    int64_t m = transA ? aShape[2] : aShape[1];
    int64_t n = transB ? bShape[1] : bShape[2];
    int64_t k = transA ? aShape[1] : aShape[2];

    FUNCTION("testNSplit", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
            LoopRange(0, CeilDivSymbolicScalar(transA ? aShape[BMM_SHAPE_N_DIM] : aShape[1], viewShape[1]), 1)) {
            LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
                LoopRange(0, CeilDivSymbolicScalar(transB ? bShape[1] : bShape[BMM_SHAPE_N_DIM], viewShape[BMM_SHAPE_N_DIM]), 1)) {
                Tensor dyn_a;
                if (transA) {
                    dyn_a = View(tensor_a, {aShape[0], aShape[1], viewShape[1]},
                        {aShape[0], aShape[1], std::min(aShape[2] - viewShape[1] * mIdx, viewShape[1])},
                        {0, 0, mIdx * viewShape[1]});
                } else {
                    dyn_a = View(tensor_a, {aShape[0], viewShape[1], aShape[2]},
                        {aShape[0], std::min(aShape[1] - viewShape[1] * mIdx, viewShape[1]), aShape[2]},
                        {0, mIdx * viewShape[1], 0});
                }
                Tensor dyn_b;
                if (!transB) {
                    dyn_b = View(tensor_b, {bShape[0], bShape[1], viewShape[2]},
                        {bShape[0], bShape[1], std::min(bShape[2] - viewShape[2] * nIdx, viewShape[2])},
                        {0, 0, nIdx * viewShape[2]});
                } else {
                    dyn_b = View(tensor_b, {bShape[0], viewShape[2], bShape[2]},
                        {bShape[0], std::min(bShape[1] - viewShape[2] * nIdx, viewShape[2]), bShape[2]},
                        {0, nIdx * viewShape[2], 0});
                }
                TileShape::Current().SetMatrixSize({m, k, n});
                Tensor res = Matrix::BatchMatmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
                Assemble(res, {0, mIdx * viewShape[0], nIdx * viewShape[1]}, tensor_c);
            }
        }
    }
}

template <typename inputDtype, typename outputDtype, bool transA, bool transB, bool isCNz>
void TestDynBatchMatmul(
    const std::vector<int64_t>& mmShape, bool isANz, bool isBNz, const std::vector<int64_t> &viewShape, string dataPath) {
    SetInterpreterConfig();

    if (mmShape.size() != BMM_SHAPE_SIZE || viewShape.size() != BMM_VIEW_SHAPE_SIZE) {
        return;
    }
    int64_t b = mmShape[0];
    int64_t m = mmShape[1];
    int64_t k = mmShape[2];
    int64_t n = mmShape[BMM_SHAPE_N_IDX];
    Tensor tensor_a = transA ? constructMatmulTensor<inputDtype>({b, k, m}, "tensor_a", isANz) :
                               constructMatmulTensor<inputDtype>({b, m, k}, "tensor_a", isANz);
    Tensor tensor_b = transB ? constructMatmulTensor<inputDtype>({b, n, k}, "tensor_b", isBNz) :
                               constructMatmulTensor<inputDtype>({b, k, n}, "tensor_b", isBNz);
    Tensor tensor_c = constructMatmulTensor<outputDtype>({b, m, n}, "tensor_c", isCNz);

    std::vector<inputDtype> aData(b * m * k, 0);
    std::vector<inputDtype> bData(b * k * n, 0);
    std::vector<outputDtype> golden(b * m * n, 0);

    readInput<outputDtype>(dataPath + "/mat_c.bin", golden);
    readInput<inputDtype>(dataPath + "/mat_a.bin", aData);
    readInput<inputDtype>(dataPath + "/mat_b.bin", bData);

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<outputDtype>(tensor_c, 0.0f),
    });

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<inputDtype>(tensor_a, aData),
        RawTensorData::CreateTensor<inputDtype>(tensor_b, bData),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<outputDtype>(tensor_c, golden),
    });

    int64_t viewM = viewShape[1];
    int64_t viewN = viewShape[2];
    if (viewM > 0 && viewN > 0) {
        MNSplitFunc<outputDtype, transA, transB, isCNz>(viewShape, tensor_a, tensor_b, tensor_c);
    } else if (viewM > 0) {
        MSplitFunc<outputDtype, transA, transB, isCNz>(viewShape, tensor_a, tensor_b, tensor_c);
    } else if (viewN > 0) {
        NSplitFunc<outputDtype, transA, transB, isCNz>(viewShape, tensor_a, tensor_b, tensor_c);
    } else {
        NonSplitFunc<outputDtype, transA, transB, isCNz>(tensor_a, tensor_b, tensor_c);
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (outputDtype *)outs->data(), 0.001f));
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_Bt_ND_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t b = 3;
    int64_t m = 2;
    int64_t k = 576;
    int64_t n = 4096;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1, -1};
    TestDynBatchMatmul<npu::tile_fwk::float16, float, false, true, false>(
        {b, m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_Bt_NZ_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t b = 4;
    int64_t m = 96;
    int64_t k = 128;
    int64_t n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1, -1};
    TestDynBatchMatmul<npu::tile_fwk::float16, float, false, true, false>(
        {b, m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_A_B_ND_bf16_tile1) {
    TileShape::Current().SetCubeTile({128, 128}, {256, 256}, {128, 128});
    int64_t b = 6;
    int64_t m = 1;
    int64_t k = 576;
    int64_t n = 4096;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1, -1};
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, false, false>(
        {b, m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_At_Bt_ND_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t b = 3;
    int64_t m = 2;
    int64_t k = 576;
    int64_t n = 4096;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1, -1};
    TestDynBatchMatmul<npu::tile_fwk::float16, float, true, true, false>(
        {b, m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulTest, test_bmm_At_Bt_ANZ_BND_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t b = 3;
    int64_t m = 128;
    int64_t k = 256;
    int64_t n = 512;
    bool isANz = true;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1, -1};
    TestDynBatchMatmul<npu::tile_fwk::float16, float, true, true, false>(
        {b, m, k, n}, isANz, isBNz, viewShape, GetGoldenDir());
}
}// namespace