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
 * \file test_dynamic_attention_post.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicAttentionPostTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {
constexpr int NUM_32 = 32;
constexpr int NUM_500000 = 500000;

// ============================ Cast+R
void PaPostDebugCastFirstR1(Tensor &postIn, Tensor &r1Out) {
    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;

    FUNCTION("main",
        {postIn}, {r1Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)
            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});

            std::vector<SymbolicScalar> dynOffset = {bIdx*bTile*S, 0, 0};
            Assemble(r1Res, dynOffset, r1Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_r1) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");

    Tensor r1Out(DT_BF16, {B * S, N, kvLoraRank}, "r1Out");

    PaPostDebugCastFirstR1(postIn, r1Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> golden(B*S*N*kvLoraRank, 0);
    readInput(GetGoldenDir() + "/r1.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),

    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(r1Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================ Cast+R+T
void PaPostDebugCastFirstT1(Tensor &postIn, Tensor &t1Out) {
    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;

    FUNCTION("main",
        {postIn}, {t1Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)
            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            std::vector<SymbolicScalar> dynOffset = {0, bIdx*bTile*S, 0};
            Assemble(t1Res, dynOffset, t1Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_t1) {
    config::SetHostOption(ONLY_CODEGEN, true);
    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");

    Tensor t1Out(DT_BF16, {N, B * S, kvLoraRank}, "t1Out");

    PaPostDebugCastFirstT1(postIn, t1Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> golden(B*S*N*kvLoraRank, 0);
    readInput(GetGoldenDir() + "/t1.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),

    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(t1Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================Cast+R+T+Bmm4
void PaPostDebugCastFirstBmm4(Tensor &postIn, Tensor &weightUV, Tensor &bmm4Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV}, {bmm4Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int64_t bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8L, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32L, bTile * S), std::min(32L, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {0, bIdx*bTile*S, 0};
            Assemble(bmmRes, dynOffset, bmm4Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_bmm4) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor bmm4Out(DT_BF16, {N, B*S, vHeadDim}, "bmm4Out");

    PaPostDebugCastFirstBmm4(postIn, weightUV, bmm4Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<npu::tile_fwk::bfloat16> golden(N*B*S*vHeadDim, 0);
    readInput(GetGoldenDir() + "/bmm4.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(bmm4Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================ Cast+R+T+Bmm4+T3R2
void PaPostDebugCastFirstCrtb4tr(Tensor &postIn, Tensor &weightUV, Tensor &r2Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV}, {r2Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile*S, 0};
            Assemble(r2Res, dynOffset, r2Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_crtb4tr) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor r2Out(DT_BF16, {B* S, N*vHeadDim}, "r2Out");

    PaPostDebugCastFirstCrtb4tr(postIn, weightUV, r2Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);

    std::vector<npu::tile_fwk::bfloat16> golden(B* S* N*vHeadDim, 0);
    readInput(GetGoldenDir() + "/r2.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(r2Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================ T1
void PaPostDebugCastFirstOnlyT1(Tensor &r1Res, Tensor &t1Out) {
    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;

    FUNCTION("main",
        {r1Res}, {t1Out}) {
        SymbolicScalar B = r1Res.GetShape()[0]; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(r1Res, {bTile * S, N, kvLoraRank}, {bIdx * bTile * S, 0, 0});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(postInUnit, {0, 1}); // (N, bTile * S, kvLoraRank)

            std::vector<SymbolicScalar> dynOffset = {0, bIdx*bTile*S, 0};
            Assemble(t1Res, dynOffset, t1Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlyt1) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];

    std::vector<uint8_t> devProgBinary;

    Tensor r1Res(DT_BF16, {B * S, N, kvLoraRank}, "r1Res");

    Tensor t1Out(DT_BF16, {N, B * S, kvLoraRank}, "t1Out");

    PaPostDebugCastFirstOnlyT1(r1Res, t1Out);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> r1ResData(B * S * N * kvLoraRank, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/r1.bin", r1ResData);
    std::vector<npu::tile_fwk::bfloat16> golden(B*S*N*kvLoraRank, 0);
    readInput(GetGoldenDir() + "/t1.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(r1Res, r1ResData),

    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(t1Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================= OnlyBmm4
void PaPostNewOnlyBmm4(Tensor &bmm4In, Tensor &weightUV, Tensor &bmm4Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {bmm4In, weightUV}, {bmm4Out}) {
        SymbolicScalar B = bmm4In.GetShape()[1] / S; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto bmm4InUnit = View(bmm4In, {N, bTile * S, kvLoraRank}, {0, bIdx * bTile * S, 0});
            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)},
                {std::min(128L, vHeadDim), std::min(128L, vHeadDim)}); // raw 8*1  512   128
            // 需要保证B*M*N*sizeof(bf16)可以放得下UB  （128*4*128*2=131072
            auto bmmRes = Matrix::BatchMatmul(dtype, bmm4InUnit, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {0, bIdx * bTile * S, 0};
            Assemble(bmmRes, dynOffset, bmm4Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlybmm4) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor bmm4In(DT_BF16, {N, B * S, kvLoraRank}, "bmm4In");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");

    Tensor bmm4Out(DT_BF16, {N, B*S, vHeadDim}, "bmm4Out");

    PaPostNewOnlyBmm4(bmm4In, weightUV, bmm4Out);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> bmm4InData(B * S * N * kvLoraRank, 0);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/t1.bin", bmm4InData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);

    std::vector<npu::tile_fwk::bfloat16> golden(N*B*S*vHeadDim, 0);
    readInput(GetGoldenDir() + "/bmm4.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(bmm4In, bmm4InData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(bmm4Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================= OnlyBmm4
void PaPostNewOnlyBmm4Fail(Tensor &bmm4In, Tensor &weightUV, Tensor &bmm4Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {bmm4In, weightUV}, {bmm4Out}) {
        SymbolicScalar B = bmm4In.GetShape()[1] / S; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto bmm4InUnit = View(bmm4In, {N, bTile * S, kvLoraRank}, {0, bIdx * bTile * S, 0});
            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)},
                {std::min(128L, vHeadDim), std::min(128L, vHeadDim)});      // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, bmm4InUnit, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {0, bIdx * bTile * S, 0};
            Assemble(bmmRes, dynOffset, bmm4Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlybmm4_fail) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    // int dtypeNum = params[6];

    std::vector<uint8_t> devProgBinary;

    Tensor bmm4In(DT_BF16, {N, B * S, kvLoraRank}, "bmm4In");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");

    Tensor bmm4Out(DT_BF16, {N, B*S, vHeadDim}, "bmm4Out");

    PaPostNewOnlyBmm4Fail(bmm4In, weightUV, bmm4Out);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> bmm4InData(B * S * N * kvLoraRank, 0);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/t1.bin", bmm4InData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);

    std::vector<npu::tile_fwk::bfloat16> golden(N*B*S*vHeadDim, 0);
    readInput(GetGoldenDir() + "/bmm4.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(bmm4In, bmm4InData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(bmm4Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================= MM5_ND
void PaPostNewOnlyMm5Nd(Tensor &quant0In, Tensor &weightO, Tensor &mm5Out) {
    auto N = 128;
    auto vHeadDim = 128;
    auto H = 7168;
    int S = 1;

    FUNCTION("main",
        {quant0In, weightO}, {mm5Out}) {
        SymbolicScalar B = quant0In.GetShape()[0] / S; // S=1
        std::cout<<"B: "<<B<<std::endl;
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto quant0InUnit = View(quant0In, {bTile * S, N * vHeadDim}, {bIdx * bTile * S, 0});
            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128, N * vHeadDim), std::min(128, N * vHeadDim)}, {std::min(512, H), std::min(512, H)});
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quant0InUnit, weightO);

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(res, dynOffset, mm5Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlymm5_nd) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int vHeadDim = params[5];
    // int dtypeNum = params[6];

    std::vector<uint8_t> devProgBinary;

    Tensor quant0In(DT_INT8, {B * S, N * vHeadDim}, "quant0In"); // a
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor mm5Out(DT_INT32, {B * S, H}, "mm5Out");

    PaPostNewOnlyMm5Nd(quant0In, weightO, mm5Out);

    // 读数据
    std::vector<int8_t> quant0InData(B * S * N * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);

    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);
    readInput<int8_t>(GetGoldenDir() + "/quant0_int8.bin", quant0InData);

    std::vector<int32_t> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/mm5_int32.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int8_t>(quant0In, quant0InData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(mm5Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.005f));
}

// ============================= MM5_ND_K
void PaPostNewOnlyMm5NdK(Tensor &quant0In, Tensor &weightO, Tensor &mm5Out) {
    auto N = 128;
    auto vHeadDim = 128;
    auto H = 7168;
    int S = 1;

    FUNCTION("main",
        {quant0In, weightO}, {mm5Out}) {
        SymbolicScalar B = quant0In.GetShape()[0] / S; // S=1
        std::cout<<"B: "<<B<<std::endl;
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto quant0InUnit = View(quant0In, {bTile * S, N * vHeadDim}, {bIdx * bTile * S, 0});

            // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128, N * vHeadDim), std::min(128, N * vHeadDim)}, {std::min(512, H), std::min(512, H)}, 
                false, true);                                                                            // raw  16  2048  128
            Tensor res = npu::tile_fwk::Matrix::Matmul(DT_INT32, quant0InUnit, weightO); // (bTile*S, H)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(res, dynOffset, mm5Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlymm5_ndk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor quant0In(DT_INT8, {B * S, N * vHeadDim}, "quant0In"); // a
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor mm5Out(DT_INT32, {B * S, H}, "mm5Out");

    // 读数据
    std::vector<int8_t> quant0InData(B * S * N * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);

    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);
    readInput<int8_t>(GetGoldenDir() + "/quant0_int8.bin", quant0InData);

    std::vector<int32_t> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/mm5_int32.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int8_t>(quant0In, quant0InData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(mm5Out, 0),
    });

    PaPostNewOnlyMm5NdK(quant0In, weightO, mm5Out);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.005f));
}

// =============================  MM5_ND_K+UnquantR3
void PaPostNewMm5NdkUnquantR3(Tensor &quant0In, Tensor &weightO, Tensor &weightOScaleW, Tensor &quantOutFp32,
                              Tensor &postOut) {
    auto N = 128;
    auto vHeadDim = 128;
    auto H = 7168;
    int S = 1;

    FUNCTION("main",
        {quant0In, weightO, weightOScaleW, quantOutFp32}, {postOut}) {
        SymbolicScalar B = quant0In.GetShape()[0] / S; // S=1
        std::cout<<"B: "<<B<<std::endl;
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto quant0InUnit = View(quant0In, {bTile * S, N * vHeadDim}, {bIdx * bTile * S, 0});
            auto quantOutFp32Unit = View(quantOutFp32, {bTile * S, 1}, {bIdx * bTile * S, 0});

            // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128, N * vHeadDim), std::min(128, N * vHeadDim)},
                {std::min(512, H), std::min(512, H)}, false, true); // raw  16  2048  128
            Tensor res = npu::tile_fwk::Matrix::Matmul(DT_INT32, quant0InUnit, weightO);  // (bTile*S, H)

            TileShape::Current().SetVecTile(std::min(8, bTile * S), std::min(1024, H)); // raw (8, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, quantOutFp32Unit);//(B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1,H)
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});
            TileShape::Current().SetVecTile(std::min(8, bTile * S), S, std::min(1024, H));
            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_mm5ndk_unquant_r3) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor quant0In(DT_INT8, {B * S, N * vHeadDim}, "quant0In"); // a
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor quantOutFp32(DT_FP32, {B*S, 1}, "quantOutFp32");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostNewMm5NdkUnquantR3(quant0In, weightO, weightOScaleW, quantOutFp32, postOut);

    // 读数据
    std::vector<int8_t> quant0InData(B * S * N * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    std::vector<float> weightOScaleWData(H, 0);
    std::vector<float> quantOutFp32Data(B*S*1, 0);

    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);
    readInput<int8_t>(GetGoldenDir() + "/quant0_int8.bin", quant0InData);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);
    readInput<float>(GetGoldenDir() + "/quant0_fp32.bin", quantOutFp32Data);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int8_t>(quant0In, quant0InData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
        RawTensorData::CreateTensor<float>(quantOutFp32, quantOutFp32Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================= MM5_NZ
void PaPostNewOnlyMm5Nz(Tensor &quant0In, Tensor &weightO, Tensor &mm5Out) {
    auto N = 128;
    auto vHeadDim = 128;
    auto H = 7168;
    int S = 1;

    FUNCTION("main",
        {quant0In, weightO}, {mm5Out}) {
        SymbolicScalar B = quant0In.GetShape()[0] / S; // S=1
        std::cout<<"B: "<<B<<std::endl;
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto quant0InUnit = View(quant0In, {bTile * S, N * vHeadDim}, {bIdx * bTile * S, 0});

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128, N * vHeadDim), std::min(128, N * vHeadDim)}, {std::min(512, H), std::min(512, H)});
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quant0InUnit, weightO);

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(res, dynOffset, mm5Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlymm5_nz) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor quant0In(DT_INT8, {B * S, N * vHeadDim}, "quant0In"); // a
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor mm5Out(DT_INT32, {B * S, H}, "mm5Out");

    PaPostNewOnlyMm5Nz(quant0In, weightO, mm5Out);

    // 读数据
    std::vector<int8_t> quant0InData(B * S * N * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);

    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);
    readInput<int8_t>(GetGoldenDir() + "/quant0_int8.bin", quant0InData);

    std::vector<int32_t> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/mm5_int32.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int8_t>(quant0In, quant0InData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(mm5Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.005f));
}

// ============================= MM5_NZ_K
void PaPostNewOnlyMm5NzK(Tensor &quant0In, Tensor &weightO, Tensor &mm5Out) {
    auto N = 128;
    auto vHeadDim = 128;
    auto H = 7168;
    int S = 1;

    FUNCTION("main",
        {quant0In, weightO}, {mm5Out}) {
        SymbolicScalar B = quant0In.GetShape()[0] / S; // S=1
        std::cout<<"B: "<<B<<std::endl;
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto quant0InUnit = View(quant0In, {bTile * S, N * vHeadDim}, {bIdx * bTile * S, 0});

            // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128, N * vHeadDim), std::min(128, N * vHeadDim)},
                {std::min(512, H), std::min(512, H)}, false, true); // raw  16  2048  128
            Tensor res = npu::tile_fwk::Matrix::Matmul(DT_INT32, quant0InUnit, weightO);  // (bTile*S, H)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(res, dynOffset, mm5Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_onlymm5_nzk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor quant0In(DT_INT8, {B * S, N * vHeadDim}, "quant0In"); // a
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor mm5Out(DT_INT32, {B * S, H}, "mm5Out");

    PaPostNewOnlyMm5NzK(quant0In, weightO, mm5Out);

    // 读数据
    std::vector<int8_t> quant0InData(B * S * N * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);

    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);
    readInput<int8_t>(GetGoldenDir() + "/quant0_int8.bin", quant0InData);

    std::vector<int32_t> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/mm5_int32.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int8_t>(quant0In, quant0InData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(mm5Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.005f));
}

// ============================Cast
void PaPostDebugCastFirst(Tensor &postIn, Tensor &cast1Out) {
    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;

    FUNCTION("main",
        {postIn}, {cast1Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)
            auto cast1 = Cast(postInUnit, DT_BF16);

            std::vector<SymbolicScalar> dynOffset = {bIdx*bTile*S*N, 0};
            Assemble(cast1, dynOffset, cast1Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");

    Tensor cast1Out(DT_BF16, {B * S * N, kvLoraRank}, "cast1Out");

    PaPostDebugCastFirst(postIn, cast1Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> golden(B*S*N*kvLoraRank, 0);
    readInput(GetGoldenDir() + "/cast1.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),

    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(cast1Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// =============================quant
void PaPostCastFirstQuant(Tensor &postIn, Tensor &r2In, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                          Tensor &quantInt8Out, Tensor &quantFp32Out) {
    auto N = weightUV.GetShape()[0];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {postIn,r2In, weightUV, weightO, weightOScaleW}, {quantInt8Out, quantFp32Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(B / bTile)) {
            auto r2InUnit = View(r2In, {bTile * S, N * vHeadDim}, {bIdx * bTile * S, 0});

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (8, 128*128)
            auto quantA = Quant(r2InUnit);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(quantizedA, dynOffset, quantInt8Out);
            Assemble(dequantScaleA, dynOffset, quantFp32Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_quant) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor r2In(DT_BF16, {B * S, N * vHeadDim}, "r2In");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor quantInt8Out(DT_INT8, {B * S, N * vHeadDim}, "quantInt8Out");
    Tensor quantFp32Out(DT_FP32, {B * S, 1}, "quantFp32Out");

    PaPostCastFirstQuant(postIn, r2In, weightUV, weightO, weightOScaleW, quantInt8Out, quantFp32Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    std::vector<npu::tile_fwk::bfloat16> r2InData(B * S * N * vHeadDim, 0);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    std::vector<float> weightOScaleWData(H, 0);

    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/r2.bin", r2InData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<int8_t> golden0(B * S * N * vHeadDim, 0);
    readInput(GetGoldenDir() + "/quant0_int8.bin", golden0);
    std::vector<float> golden1(B * S * 1, 0);
    readInput(GetGoldenDir() + "/quant0_fp32.bin", golden1);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(r2In, r2InData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int8_t>(quantInt8Out, 0),
        RawTensorData::CreateConstantTensor<float>(quantFp32Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout<<"=======================QuantInt8Out: "<<std::endl;
    auto outs0 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden0, (int8_t *)outs0->data(), 0.005f));
    std::cout<<"=======================QuantFp32Out: "<<std::endl;
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(golden1, (float *)outs1->data(), 0.005f));
}

// =============================t3r2
void PaPostCastFirstT3r2(Tensor &bmm4In, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW, Tensor &r2Out) {
    auto N = weightUV.GetShape()[0];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {bmm4In,weightUV, weightO, weightOScaleW}, {r2Out}) {
        SymbolicScalar B = bmm4In.GetShape()[1] / S; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(B / bTile)) {
            auto bmm4InUnit = View(bmm4In, {N, bTile * S, vHeadDim}, {0, bIdx * bTile * S, 0});

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmm4InUnit, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile*S, 0};
            Assemble(r2Res, dynOffset, r2Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_t3r2) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    // int dtypeNum = params[6];

    std::vector<uint8_t> devProgBinary;

    Tensor bmm4In(DT_BF16, {N, B * S, vHeadDim}, "bmm4In");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor r2Out(DT_BF16, {B* S, N*vHeadDim}, "r2Out");

    PaPostCastFirstT3r2(bmm4In, weightUV, weightO, weightOScaleW, r2Out);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> bmm4InData(N* B * S* vHeadDim, 0);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    std::vector<float> weightOScaleWData(H, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/bmm4.bin", bmm4InData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B* S* N*vHeadDim, 0);
    readInput(GetGoldenDir() + "/r2.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(bmm4In, bmm4InData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(r2Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// =============================t3
void PaPostCastFirstT3(Tensor &bmm4In, Tensor &t3Out) {
    auto N = 128;
    auto vHeadDim = 128;
    int S = 1;

    FUNCTION("main",
        {bmm4In}, {t3Out}) {
        SymbolicScalar B = bmm4In.GetShape()[1] / S; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(B / bTile)) {
            auto bmm4InUnit = View(bmm4In, {N, bTile * S, vHeadDim}, {0, bIdx * bTile * S, 0});

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmm4InUnit, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile*S, 0, 0};
            Assemble(t3Res, dynOffset, t3Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_t3) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor bmm4In(DT_BF16, {N, B * S, vHeadDim}, "bmm4In");
    Tensor t3Out(DT_BF16, {B* S, N, vHeadDim}, "t3Out");

    PaPostCastFirstT3(bmm4In, t3Out);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> bmm4InData(N* B * S* vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/bmm4.bin", bmm4InData);

    std::vector<npu::tile_fwk::bfloat16> golden(B* S* N*vHeadDim, 0);
    readInput(GetGoldenDir() + "/t3.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(bmm4In, bmm4InData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(t3Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// =============================r2
void PaPostCastFirstR2(Tensor &t3In, Tensor &r2Out) {
    auto N = 128;
    auto vHeadDim = 128;
    int S = 1;

    FUNCTION("main",
        {t3In}, {r2Out}) {
        SymbolicScalar B = t3In.GetShape()[0] / S; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(B / bTile)) {
            auto t3InUnit = View(t3In, {bTile * S, N, vHeadDim}, {bIdx * bTile * S, 0, 0});

            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3InUnit, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile*S, 0};
            Assemble(r2Res, dynOffset, r2Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_r2) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int vHeadDim = params[5];
    // int dtypeNum = params[6];

    std::vector<uint8_t> devProgBinary;

    Tensor t3In(DT_BF16, {B * S, N, vHeadDim}, "t3In");
    Tensor r2Out(DT_BF16, {B* S, N*vHeadDim}, "r2Out");

    PaPostCastFirstR2(t3In, r2Out);

    // 读数据
    std::vector<npu::tile_fwk::bfloat16> t3InData(N* B * S* vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/t3.bin", t3InData);

    std::vector<npu::tile_fwk::bfloat16> golden(B* S* N*vHeadDim, 0);
    readInput(GetGoldenDir() + "/r2.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(t3In, t3InData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(r2Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// =============================unQuantR3
void PaPostCastFirstUnquantR3(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                              Tensor &quantOutFp32, Tensor &postOut) {
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn,weightUV, weightO, weightOScaleW, quantOutFp32}, {postOut}) {
        SymbolicScalar B = postIn.GetShape()[0] / S; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(B / bTile)) {
            auto postInUnit = View(postIn, {bTile * S, H}, {bIdx * bTile * S, 0});
            auto quantOutFp32Unit = View(quantOutFp32, {bTile * S, 1}, {bIdx * bTile * S, 0});

            TileShape::Current().SetVecTile(std::min(8, bTile * S), std::min(1024L, H)); // raw (8, 7168)
            auto res = Cast(postInUnit, DataType::DT_FP32);
            res = Mul(res, quantOutFp32Unit);//(B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);  // (1,H)
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});
            TileShape::Current().SetVecTile(std::min(8, bTile * S), S, std::min(1024L, H));
            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_unquant_r3) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_INT32, {B * S, H}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor quantOutFp32(DT_FP32, {B*S, 1}, "quantOutFp32");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostCastFirstUnquantR3(postIn, weightUV, weightO, weightOScaleW, quantOutFp32, postOut);

    // 读数据
    std::vector<int32_t> postInData(B * S* H, 0);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    std::vector<float> weightOScaleWData(H, 0);
    std::vector<float> quantOutFp32Data(B*S*1, 0);

    readInput<int32_t>(GetGoldenDir() + "/mm5_int32.bin", postInData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);
    readInput<float>(GetGoldenDir() + "/quant0_fp32.bin", quantOutFp32Data);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
        RawTensorData::CreateTensor<float>(quantOutFp32, quantOutFp32Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================ Cast+R+T+Bmm4+T3R2+Quant
void PaPostDebugCastFirstCrtb4trQuant(Tensor &postIn, Tensor &weightUV, Tensor &quantInt8Out, Tensor &quantFp32Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV}, {quantInt8Out, quantFp32Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (8, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(quantizedA, dynOffset, quantInt8Out);
            Assemble(dequantScaleA, dynOffset, quantFp32Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_crtb4tr_quant) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor quantInt8Out(DT_INT8, {B * S, N * vHeadDim}, "quantInt8Out");
    Tensor quantFp32Out(DT_FP32, {B * S, 1}, "quantFp32Out");

    PaPostDebugCastFirstCrtb4trQuant(postIn, weightUV, quantInt8Out, quantFp32Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);

    std::vector<int8_t> golden0(B * S * N * vHeadDim, 0);
    readInput(GetGoldenDir() + "/quant0_int8.bin", golden0);
    std::vector<float> golden1(B * S * 1, 0);
    readInput(GetGoldenDir() + "/quant0_fp32.bin", golden1);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int8_t>(quantInt8Out, 0),
        RawTensorData::CreateConstantTensor<float>(quantFp32Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout<<"=======================QuantInt8Out: "<<std::endl;
    auto outs0 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden0, (int8_t *)outs0->data(), 0.005f));
    std::cout<<"=======================QuantFp32Out: "<<std::endl;
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(golden1, (float *)outs1->data(), 0.005f));
}

// ============================ Cast+R+T+Bmm4+T3R2+QuantFail
void PaPostDebugCastFirstCrtb4trQuantFail(Tensor &postIn, Tensor &weightUV, Tensor &quantInt8Out,
                                          Tensor &quantFp32Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV}, {quantInt8Out, quantFp32Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (8, 128*128) 此处Fail
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(quantizedA, dynOffset, quantInt8Out);
            Assemble(dequantScaleA, dynOffset, quantFp32Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_crtb4tr_quant_fail) {
    config::SetHostOption(ONLY_CODEGEN, true);
    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor quantInt8Out(DT_INT8, {B * S, N * vHeadDim}, "quantInt8Out");
    Tensor quantFp32Out(DT_FP32, {B * S, 1}, "quantFp32Out");

    PaPostDebugCastFirstCrtb4trQuantFail(postIn, weightUV, quantInt8Out, quantFp32Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);

    std::vector<int8_t> golden0(B * S * N * vHeadDim, 0);
    readInput(GetGoldenDir() + "/quant0_int8.bin", golden0);
    std::vector<float> golden1(B * S * 1, 0);
    readInput(GetGoldenDir() + "/quant0_fp32.bin", golden1);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int8_t>(quantInt8Out, 0),
        RawTensorData::CreateConstantTensor<float>(quantFp32Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::cout<<"=======================QuantInt8Out: "<<std::endl;
    auto outs0 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden0, (int8_t *)outs0->data(), 0.005f));
    std::cout<<"=======================QuantFp32Out: "<<std::endl;
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(golden1, (float *)outs1->data(), 0.005f));
}

// ============================ Cast+R+T+Bmm4+T3R2+Quant+mm5ND
void PaPostDebugCastFirstCrtb4trQMM5ND(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &mm5Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO}, {mm5Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (8, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)}, {std::min(512L, H), std::min(512L, H)});
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, weightO);

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(res, dynOffset, mm5Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_crtb4trq_mm5nd) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor mm5Out(DT_INT32, {B * S, H}, "mm5Out");

    PaPostDebugCastFirstCrtb4trQMM5ND(postIn, weightUV, weightO, mm5Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);

    std::vector<int32_t> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/mm5_int32.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(mm5Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.005f));
}

// ============================ Cast+R+T+Bmm4+T3R2+Quant+mm5NDk
void PaPostDebugCastFirstCrtb4trQMM5NDk(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &mm5Out) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO}, {mm5Out}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)
            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (8, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)},
                {std::min(512L, H), std::min(512L, H)}, false, true); // raw  16  2048  128
            Tensor res = npu::tile_fwk::Matrix::Matmul(DT_INT32, quantizedA, weightO);

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile * S, 0};
            Assemble(res, dynOffset, mm5Out);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_cast_first_crtb4trq_mm5ndk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor mm5Out(DT_INT32, {B * S, H}, "mm5Out");

    PaPostDebugCastFirstCrtb4trQMM5NDk(postIn, weightUV, weightO, mm5Out);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);

    std::vector<int32_t> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/mm5_int32.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(mm5Out, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.005f));
}

// ============================ All +mm5+unsplitK+low
void PaPostDebugCastFirstMm5UnsplitKLow(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                                        Tensor &postOut) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO, weightOScaleW}, {postOut}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 2;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (2*1*32, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 32, kvLoraRank}); // raw (2*1, 32, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({32, 2, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(32, std::min(8, bTile * S), vHeadDim); // raw (32, 2, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 32, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (2, 32*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)}, {std::min(512L, H), std::min(512L, H)});
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, weightO);

            TileShape::Current().SetVecTile(std::min(8, bTile * S), std::min(1024L, H)); // raw (2, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);//(B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1,H)
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            TileShape::Current().SetVecTile(std::min(8, bTile * S), S, std::min(1024L, H));
            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nd_unsplitk_low) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5UnsplitKLow(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nz_unsplitk_low) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5UnsplitKLow(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================ All +mm5+unsplitK
void PaPostDebugCastFirstMm5UnsplitK(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                                     Tensor &postOut) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO, weightOScaleW}, {postOut}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 8;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 8, kvLoraRank}); // raw (8*1, 128, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({64, 8, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(64, std::min(8, bTile * S), vHeadDim); // raw (128, 8, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 64, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (8, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)}, {std::min(512L, H), std::min(512L, H)});
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, weightO);

            TileShape::Current().SetVecTile(std::min(8, bTile * S), std::min(1024L, H)); // raw (8, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);//(B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1,H)
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            TileShape::Current().SetVecTile(std::min(8, bTile * S), S, std::min(1024L, H));
            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nd_unsplitk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5UnsplitK(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData);
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nz_unsplitk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5UnsplitK(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.005f));
}

// ============================ All +mm5+splitK
void PaPostDebugCastFirstMm5SplitK(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                                   Tensor &postOut) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO, weightOScaleW}, {postOut}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 32;
        config::SetPassOption(SG_PG_UPPER_BOUND, 500000);  // 300000(1024/167us)   700000(512/174us)   500000(512/171us)
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            auto r1Res = Reshape(postInUnit, {bTile*S, N, kvLoraRank}); // 128个
            config::SetSemanticLabel("CAST+TRANSPOSE1");
            TileShape::Current().SetVecTile({std::min(32, bTile * S), 2, kvLoraRank}); // raw (bTile*1, 128, 512)
            auto cast1 = Cast(r1Res, DT_BF16);
            auto t1Res = Transpose(cast1, {0, 1}); // (N, bTile * S, kvLoraRank)    // 128个

            config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
            config::SetSemanticLabel("BMM4");
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(512L, kvLoraRank)}, {vHeadDim, vHeadDim},
                true);                                                 // raw bTile*1  512   128   // 128/4个
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            config::SetSemanticLabel("TRANSPOSE3");
            TileShape::Current().SetVecTile(4, std::min(32, bTile * S), vHeadDim); // raw (128, bTile*1, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim) // 128个
            config::SetSemanticLabel("RESHAPE2");
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            config::SetSemanticLabel("QUANT");
            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (bTile*1, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)},
                {std::min(512L, H), std::min(512L, H)}, false, true); // 14个
            Tensor res = npu::tile_fwk::Matrix::Matmul(DT_INT32, quantizedA, weightO);

            config::SetSemanticLabel("CMMC");
            TileShape::Current().SetVecTile(std::min(32, bTile * S), std::min(32L, H)); // raw (bTile*1, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);   // (B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1, H)  // 224个
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            config::SetSemanticLabel("RESHAPE3");
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
// mm5 normal unsplitK
void PaPostDebugCastFirstMm5NormalUnSplitK(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                                           Tensor &postOut) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO, weightOScaleW}, {postOut}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 32;
        config::SetPassOption(SG_PG_UPPER_BOUND, 500000);  // 300000(1024/167us)   700000(512/174us)   500000(512/171us)
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            auto r1Res = Reshape(postInUnit, {bTile*S, N, kvLoraRank}); // 128个
            config::SetSemanticLabel("CAST+TRANSPOSE1");
            TileShape::Current().SetVecTile({std::min(32, bTile * S), 2, kvLoraRank}); // raw (bTile*1, 128, 512)
            auto cast1 = Cast(r1Res, DT_BF16);
            auto t1Res = Transpose(cast1, {0, 1}); // (N, bTile * S, kvLoraRank)    // 128个

            config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
            config::SetSemanticLabel("BMM4");
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(512L, kvLoraRank)}, {vHeadDim, vHeadDim},
                true);                                                 // raw bTile*1  512   128   // 128/4个
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            config::SetSemanticLabel("TRANSPOSE3");
            TileShape::Current().SetVecTile(4, std::min(32, bTile * S), vHeadDim); // raw (128, bTile*1, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim) // 128个
            config::SetSemanticLabel("RESHAPE2");
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            config::SetSemanticLabel("QUANT");
            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (bTile*1, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(512L, N * vHeadDim), std::min(512L, N * vHeadDim)}, {std::min(64L, H), std::min(64L, H)},
                true); // raw  bTile*1  16k  7168
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, weightO);

            config::SetSemanticLabel("CMMC");
            TileShape::Current().SetVecTile(std::min(32, bTile * S), std::min(32L, H)); // raw (bTile*1, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);   // (B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1, H)  // 224个
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            config::SetSemanticLabel("RESHAPE3");
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nz_splitk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5SplitK(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.004f));
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nz_normal_unsplitk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5NormalUnSplitK(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.004f));
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nd_splitk) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5SplitK(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData); // ND
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.004f));
}

// ============================ All +mm5+splitK+low
void PaPostDebugCastFirstMm5SplitKLow(Tensor &postIn, Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW,
                                      Tensor &postOut) {
    auto dtype = weightUV.GetStorage()->Datatype();  // bf16
    auto N = weightUV.GetShape()[0];
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {postIn, weightUV, weightO, weightOScaleW}, {postOut}) {
        SymbolicScalar B = postIn.GetShape()[0] / N; // S=1
        const int bTile = 2;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / bTile, 1)) {
            auto postInUnit = View(postIn, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(64L, bTile * S * N), kvLoraRank}); // raw (2*1*32, 512)

            auto cast1 = Cast(postInUnit, DT_BF16);
            auto r1Res = Reshape(cast1, {bTile*S, N, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(8, bTile * S), 32, kvLoraRank}); // raw (2*1, 32, 512)
            auto t1Res = Transpose(r1Res, {0, 1}); // (N, bTile * S, kvLoraRank)

            TileShape::Current().SetVecTile({32, 2, 128});
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(256L, kvLoraRank)}, {vHeadDim, vHeadDim}); // raw 8*1  512   128
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(32, std::min(8, bTile * S), vHeadDim); // raw (32, 2, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim)
            TileShape::Current().SetVecTile(std::min(8, bTile * S), 32, vHeadDim);
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (2, 32*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(32, bTile * S), std::min(32, bTile * S)},
                {std::min(128L, N * vHeadDim), std::min(128L, N * vHeadDim)},
                {std::min(512L, H), std::min(512L, H)}, false, true); // raw  16  2048  128
            Tensor res = npu::tile_fwk::Matrix::Matmul(DT_INT32, quantizedA, weightO);  // (bTile*S, H)

            TileShape::Current().SetVecTile(std::min(8, bTile * S), std::min(1024L, H)); // raw (2, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);//(B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1,H)
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            TileShape::Current().SetVecTile(std::min(8, bTile * S), S, std::min(1024L, H));
            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}

TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nz_splitk_low) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5SplitKLow(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.0001f));
}
TEST_F(DynamicAttentionPostTest, dynamic_pa_post_new_mm5nd_splitk_low) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<uint8_t> devProgBinary;

    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO");  // ND
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PaPostDebugCastFirstMm5SplitKLow(postIn, weightUV, weightO, weightOScaleW, postOut);

    // 读数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o_nd.bin", weightOData); // ND
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);

    std::vector<npu::tile_fwk::bfloat16> golden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::bfloat16 *)outs->data(), 0.0001f));
}

// ================Pa+PaPost bf16 b48
void PageAttentionPostBf16(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &postIn, Tensor &weightUV, Tensor &weightO,
    Tensor &weightOScaleW, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes, int bTile) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    int batchSize = blockTable.GetShape()[0];
    int nQ = qNope.GetShape()[0] / batchSize; // B*1*N

    auto N = weightUV.GetShape()[0];;
    auto kvLoraRank = weightUV.GetShape()[1];
    auto vHeadDim = weightUV.GetShape()[2];
    auto H = weightO.GetShape()[1];
    int S = 1;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, postIn, weightUV, weightO, weightOScaleW},
        {attentionOut, postOut}) {
        SymbolicScalar nLoop = nQ / nTile;
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1),
                     PowersOf2(maxUnrollTimes)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = View(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = View(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    Assemble(qn, {0, 0}, qi);
                    Assemble(qr, {0, dN}, qi);

                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = View(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = View(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kj, false, true); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                    auto sijScale = Mul(sij, Element(DataType::DT_FP32, softmaxScale)); // (curNTile, curS2Tile)

                    auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF (IsLoopBegin(bn, 0)) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto oiTmp = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        auto miNew = Maximum(mi, tildaMij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t1 = Sub(mi, miNew);           // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t6 = Mul(t2, li);       // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto liNew = Add(t6, t5);    // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)

                        auto q3 = Mul(oi, t2); // (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }

        SymbolicScalar B = attentionOut.GetShape()[0] / N; // S=1
        config::SetPassOption(SG_PG_UPPER_BOUND, NUM_500000);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, B / (bTile <= 0 ? 1 : bTile), 1), PowersOf2(maxUnrollTimes), true) {
            auto postInUnit = View(attentionOut, {bTile * S * N, kvLoraRank}, {bIdx * bTile * S * N, 0});
            TileShape::Current().SetVecTile({std::min(32L, bTile * S * N), kvLoraRank});

            auto r1Res = Reshape(postInUnit, {bTile*S, N, kvLoraRank}); // 128个
            TileShape::Current().SetVecTile({std::min(NUM_32, bTile * S), 2, kvLoraRank}); // raw (bTile*1, 128, 512)
            auto cast1 = Cast(r1Res, DT_BF16);
            auto t1Res = Transpose(cast1, {0, 1}); // (N, bTile * S, kvLoraRank)    // 128个

            TileShape::Current().SetCubeTile({std::min(NUM_32, bTile * S), std::min(NUM_32, bTile * S)},
                {std::min(256L, kvLoraRank), std::min(5L, kvLoraRank)}, {vHeadDim, vHeadDim},
                true);                                                 // raw bTile*1  512   128   // 128/4个
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (N, bTile, kvLoraRank) * (N, kvLoraRank, vHeadDim) -> (N, bTile, vHeadDim)

            TileShape::Current().SetVecTile(1, std::min(NUM_32, bTile * S), vHeadDim); // raw (128, bTile*1, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (N, bTile, vHeadDim) -> (bTile, N, vHeadDim) // 128个
            auto r2Res = Reshape(t3Res, {bTile * S, N * vHeadDim}); // (bTile * S, N, vHeadDim) -> (bTile * S, N*vHeadDim)

            TileShape::Current().SetVecTile(1, N * vHeadDim); // raw (bTile*1, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(bTile * S, N*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(bTile * S, 1)

            // // (bTile*S, N*vHeadDim) @ (N*vHeadDim, H) = (bTile*S, H)
            // // int8 @ int8 = int32
            TileShape::Current().SetCubeTile({std::min(NUM_32, bTile * S), std::min(NUM_32, bTile * S)},
                {std::min(512L, N * vHeadDim), std::min(512L, N * vHeadDim)}, {std::min(64L, H), std::min(64L, H)},
                true); // raw  bTile*1  16k  7168
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, weightO);

            TileShape::Current().SetVecTile(std::min(NUM_32, bTile * S), std::min(32L, H)); // raw (bTile*1, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);   // (B*S, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, H});
            res = Mul(res, weightOScaleW2Dim);   // (1, H)  // 224个
            Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
            auto postOutTmp = Reshape(bmm5Res, {bTile, S, H});

            std::vector<SymbolicScalar> dynOffset = {bIdx * bTile, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}
void testPaPostBf16(PaTileShapeConfig& tileConfig, int maxUnrollTimes, int bTile) {
    config::SetHostOption(ONLY_CODEGEN, true);

    std::vector<uint8_t> devProgBinary;

    int paramsSize = 8;
    int postParamsSize = 7;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);

    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    std::vector<int64_t> params(postParamsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);

    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];

    std::vector<int> seq(b);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");
    Tensor postIn(DT_FP32, {B * S * N, kvLoraRank}, "postIn");
    Tensor weightUV(DT_BF16, {N, kvLoraRank, vHeadDim}, "weightUV");
    Tensor weightO(DT_INT8, {N * vHeadDim, H}, "weightO", TileOpFormat::TILEOP_NZ); // NZ
    Tensor weightOScaleW(DT_FP32, {H}, "weightOScaleW");
    Tensor postOut(DT_BF16, {B, S, H}, "postOut");

    PageAttentionPostBf16(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize,
                        softmaxScale, postIn, weightUV, weightO, weightOScaleW, paOut, postOut, tileConfig, maxUnrollTimes, bTile);

    // 读数据
    // PA数据
    std::vector<npu::tile_fwk::bfloat16> qNopeData(b * nq * sq * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> qRopeData(b * nq * sq * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> kNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<npu::tile_fwk::bfloat16> kRopeCacheData(blockNum * blockSize * dr, 0);
    std::vector<npu::tile_fwk::bfloat16> vNopeCacheData(blockNum * blockSize * dn, 0);
    std::vector<int32_t> blockTableData(b * maxBlockNumPerBatch, 0);

    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", qNopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", qRopeData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheData);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache.bin", vNopeCacheData);
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", blockTableData);

    // POST数据
    std::vector<float> postInData(B * S * N * kvLoraRank, 0);
    readInput<float>(GetGoldenDir() + "/input.bin", postInData);
    std::vector<npu::tile_fwk::bfloat16> weightUVData(N * kvLoraRank * vHeadDim, 0);
    readInput<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/w_uv.bin", weightUVData);
    std::vector<int8_t> weightOData(N * vHeadDim * H, 0);
    readInput<int8_t>(GetGoldenDir() + "/w_o.bin", weightOData);// NZ
    std::vector<float> weightOScaleWData(H, 0);
    readInput<float>(GetGoldenDir() + "/w_o_scale_w.bin", weightOScaleWData);
    std::vector<npu::tile_fwk::bfloat16> paPostgolden(B * S * H, 0);
    readInput(GetGoldenDir() + "/attn_output.bin", paPostgolden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qNope, qNopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kNopeCache, kNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(vNopeCache, vNopeCacheData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(qRope, qRopeData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(kRopeCache, kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, seq),
        RawTensorData::CreateTensor<float>(postIn, postInData),
        RawTensorData::CreateTensor<npu::tile_fwk::bfloat16>(weightUV, weightUVData),
        RawTensorData::CreateTensor<int8_t>(weightO, weightOData),
        RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(paOut, 0),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(postOut, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(postInData, (float *)outs->data(), 0.005f));
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(paPostgolden, (npu::tile_fwk::bfloat16 *)outs1->data(), 0.04f));
}

TEST_F(DynamicAttentionPostTest, dynamic_pa_papost_bf16_b48) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {16, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {16, 256};
    /*
                     powersOf  tileB
                         2      24   ok
                         2      16   ok
                         2      12   ok
                         4      12   ok
                         8      6    ok
                         4      1    nok
                         4      2    ok
    */
    testPaPostBf16(tileConfig, 2, 16);
}

}