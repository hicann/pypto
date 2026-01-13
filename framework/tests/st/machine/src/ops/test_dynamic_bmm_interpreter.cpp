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

class DynamicBatchMatmulInterpreterTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename InputT, typename OutputT, bool IsBtrans = false, bool IsBNZ = false>
void TestDynBatchMatmul(int b, int m, int k, int n, string dataPath) {
    config::SetHostOption(ONLY_CODEGEN, true);

    SetInterpreterConfig();

    int ka = k;
    int kb = k;
    int nb = n;
    if constexpr (IsBtrans) {
        std::swap(kb, nb);
    }
    std::vector<int64_t> shape_a = {b, m, ka};
    std::vector<int64_t> shape_b = {b, kb, nb};
    std::vector<int64_t> shape_c = {b, m, n};

    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OutputT>();

    Tensor tensor_a(InputAstDtype, shape_a, "tensor_a");
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_b(InputAstDtype, shape_b, "tensor_b", bfmt);
    Tensor tensor_c(OutputAstDtype, shape_c, "tensor_c");

    std::vector<InputT> aData(b * m * k, 0);
    std::vector<InputT> bData(b * k * n, 0);
    std::vector<OutputT> golden(b * m * n, 0);

    readInput<InputT>(dataPath + "/mat_a.bin", aData);
    readInput<InputT>(dataPath + "/mat_b.bin", bData);
    readInput<OutputT>(dataPath + "/mat_c.bin", golden);

     ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<InputT>(tensor_a, aData),
        RawTensorData::CreateTensor<InputT>(tensor_b, bData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<OutputT>(tensor_c, 0.0f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<OutputT>(tensor_c, golden),
    });

    FUNCTION("test_dyn_bmm", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = View(tensor_a, {b, m, ka}, {b, m, ka}, {0, mIdx, 0});
            Tensor dyn_b = View(tensor_b, {b, kb, nb}, {b, kb, nb}, {0, 0, 0});
            if constexpr (IsBNZ) {
                TileShape::Current().SetMatrixSize({m, k, n});
            }
            tensor_c = Matrix::BatchMatmul(OutputAstDtype, dyn_a, dyn_b, false, IsBtrans);
        }
    }
}

template <typename InputT, typename OutputT, bool IsBtrans = false, bool IsBNZ = false>
void TestDynBatchMatmul4D(vector<int> b1, vector<int> b2, int m, int k, int n, string dataPath) {
    config::SetHostOption(ONLY_CODEGEN, true);

    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);

    int ka = k;
    int kb = k;
    int nb = n;
    if constexpr (IsBtrans) {
        std::swap(kb, nb);
    }
    std::vector<int64_t> shape_a = {b1[0], b1[1], m, ka};
    std::vector<int64_t> shape_b = {b2[0], b2[1], kb, nb};
    std::vector<int64_t> shape_c = {b1[0], b1[1], m, n};

    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OutputT>();
    Tensor tensor_a(InputAstDtype, shape_a, "tensor_a");
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor tensor_b(InputAstDtype, shape_b, "tensor_b", bfmt);
    Tensor tensor_c(OutputAstDtype, shape_c, "tensor_c");

    std::vector<InputT> aData(b1[0] * b1[1] * m * k, 0);
    std::vector<InputT> bData(b2[0] * b2[1] * k * n, 0);
    std::vector<OutputT> golden(b1[0] * b1[1] * m * n, 0);
    // read
    readInput<InputT>(dataPath + "/mat_a.bin", aData);
    readInput<InputT>(dataPath + "/mat_b.bin", bData);
    readInput<OutputT>(dataPath + "/mat_c.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<InputT>(tensor_a, aData),
        RawTensorData::CreateTensor<InputT>(tensor_b, bData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<OutputT>(tensor_c, 0.0f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<OutputT>(tensor_c, golden),
    });

    FUNCTION("main", {tensor_a, tensor_b}, {tensor_c}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = View(tensor_a, {b1[0], b1[1], m, ka}, {b1[0], b1[1], m, ka}, {0, 0, mIdx, 0});
            Tensor dyn_b = View(tensor_b, {b2[0], b2[1], kb, nb}, {b2[0], b2[1], kb, nb}, {0, 0, 0, 0});
            if constexpr (IsBNZ) {
                TileShape::Current().SetMatrixSize({m, k, n});
            }
            tensor_c = Matrix::BatchMatmul(OutputAstDtype, dyn_a, dyn_b, false, IsBtrans);
        }
    }
}

TEST_F(DynamicBatchMatmulInterpreterTest, test_bmm_A_B_ND_bf16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int b = 2;
    int m = 64;
    int k = 128;
    int n = 384;
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, false> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulInterpreterTest, test_bmm_A_Bt_ND_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int b = 2;
    int m = 2;
    int k = 320;
    int n = 512;
    TestDynBatchMatmul<npu::tile_fwk::float16, float, true, false> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulInterpreterTest, test_bmm_A_B_NZ_bf16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int b = 2;
    int m = 16;
    int k = 512;
    int n = 128;
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, true> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulInterpreterTest, test_bmm_A_Bt_NZ_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int b = 2;
    int m = 96;
    int k = 128;
    int n = 256;
    TestDynBatchMatmul<npu::tile_fwk::float16, float, true, true> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulInterpreterTest, test_bmm_A_B_ND_bf16_tile1) {
    TileShape::Current().SetCubeTile({128, 128}, {256, 256}, {128, 128});
    int b = 3;
    int m = 1;
    int k = 576;
    int n = 512;
    TestDynBatchMatmul<npu::tile_fwk::bfloat16, float, false, false> (b, m, k, n, GetGoldenDir());
}

TEST_F(DynamicBatchMatmulInterpreterTest, bmm4D_A_B_NZ) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int m = 16, k = 64, n = 32;
    vector<int> b1 = {4, 5};
    vector<int> b2 = {4, 5};
    string indtype = "fp16";
    string outdtype = "fp16";
    //ReadCSV(b, m, n, k, indtype, outdtype);

    if (indtype == "fp16") {
        TestDynBatchMatmul4D<npu::tile_fwk::float16, float, false, true>(b1, b2, m, k, n, GetGoldenDir());
    }
    else if (indtype == "bf16") {
        TestDynBatchMatmul4D<npu::tile_fwk::bfloat16, float, false, true>(b1, b2, m, k, n, GetGoldenDir());
    }
    else if (indtype == "int8") {
        TestDynBatchMatmul4D<int8_t, int32_t, false, true>(b1, b2, m, k, n, GetGoldenDir());
    }
}
}// namespace
