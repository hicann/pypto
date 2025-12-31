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
 * \file test_codegen_gather.cpp
 * \brief Unit test for codegen.
 */
#include <vector>
#include <string>
using std::string;
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenGather : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

constexpr const int GATHER_SHAPE0 = 16;
constexpr const int GATHER_SHAPE1 = 32;


Function &testGatherEle(bool isSupportTileTensor, string funcName) {
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    } else {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }
    constexpr const int32_t nRoutedExperts = 32;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({GATHER_SHAPE0, GATHER_SHAPE1});
    Tensor inputScores(DT_FP32, inputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor}) {
        auto topkIdx = std::get<1>(TopK(inputScores, numExpertsPerTopk, -1));       // [b*s,256]->[b*s,8]
        auto topkWeight = GatherElements(inputTmpScores, topkIdx, 1);                // [b*s,8]
        auto topkWeightSum = Sum(topkWeight, 1, true);                           // [b*s,8]->[b*s,1]
        auto denominator = Add(topkWeightSum, Element(DataType::DT_FP32, 1e-20f)); // [b*s,1]
        outputTensor = Div(topkWeight, denominator);                                // [b*s,numExpertsPerTok]
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    return *function;
}
TEST_F(TestCodegenGather, TestGatherEle) {
    testGatherEle(false, "GATHER_ELEMET_T");
}

TEST_F(TestCodegenGather, TestGatherEleTileTensor) {
    InsertTileTensorOp(Opcode::OP_GATHER_ELEMENT, "TgatherElement");
    Function &func = testGatherEle(true, "GATHER_ELEMET_TILETENSOR");
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 3770440571518270053

extern "C" [aicore] void TENSOR_GATHER_ELEMET_TILETENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E256 = (float __ubuf__ *)get_imm(0x0); // size: 0x100
float *UB_S0_E256_T = (float *)get_imm(0x0); // size: 0x100
float __ubuf__ *UB_S256_E1280 = (float __ubuf__ *)get_imm(0x100); // size: 0x400
float *UB_S256_E1280_T = (float *)get_imm(0x100); // size: 0x400
float __ubuf__ *UB_S1472_E1728 = (float __ubuf__ *)get_imm(0x5c0); // size: 0x100
float *UB_S1472_E1728_T = (float *)get_imm(0x5c0); // size: 0x100
float __ubuf__ *UB_S1280_E1408 = (float __ubuf__ *)get_imm(0x500); // size: 0x80
float *UB_S1280_E1408_T = (float *)get_imm(0x500); // size: 0x80
int32_t __ubuf__ *UB_S1408_E1472 = (int32_t __ubuf__ *)get_imm(0x580); // size: 0x40
int32_t *UB_S1408_E1472_T = (int32_t *)get_imm(0x580); // size: 0x40
float __ubuf__ *UB_S1728_E1792 = (float __ubuf__ *)get_imm(0x6c0); // size: 0x40
float *UB_S1728_E1792_T = (float *)get_imm(0x6c0); // size: 0x40
int32_t __ubuf__ *UB_S1792_E1824 = (int32_t __ubuf__ *)get_imm(0x700); // size: 0x20
int32_t *UB_S1792_E1824_T = (int32_t *)get_imm(0x700); // size: 0x20
float __ubuf__ *UB_S1824_E1888 = (float __ubuf__ *)get_imm(0x720); // size: 0x40
float *UB_S1824_E1888_T = (float *)get_imm(0x720); // size: 0x40
float __ubuf__ *UB_S1888_E1920 = (float __ubuf__ *)get_imm(0x760); // size: 0x20
float *UB_S1888_E1920_T = (float *)get_imm(0x760); // size: 0x20
using UBTileTensorFP32Dim2_9 = TileTensor<float, StaticLayout2Dim<1, 8, 1, 8>, Hardware::UB>;
using UBTileTensorFP32Dim2_6 = TileTensor<float, StaticLayout2Dim<2, 8, 2, 8>, Hardware::UB>;
using UBTileTensorINT32Dim1_7 = TileTensor<int32_t, StaticLayout1Dim<8, 8>, Hardware::UB>;
using UBTileTensorINT32Dim2_5 = TileTensor<int32_t, StaticLayout2Dim<2, 8, 2, 8>, Hardware::UB>;
using UBTileTensorFP32Dim2_3 = TileTensor<float, StaticLayout2Dim<2, 128, 2, 128>, Hardware::UB>;
using UBTileTensorFP32Dim2_8 = TileTensor<float, StaticLayout2Dim<2, 1, 2, 8>, Hardware::UB>;
using UBTileTensorFP32Dim2_4 = TileTensor<float, StaticLayout2Dim<2, 16, 2, 16>, Hardware::UB>;
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using GMTileTensorFP32Dim2_10 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, StaticLayout2Dim<2, 32, 2, 32>, Hardware::UB>;
GMTileTensorFP32Dim2_10 gmTensor_25((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 2)->Addr, DynLayout2Dim(Shape2Dim(2, 8), Stride2Dim(8, 1)));
UBTileTensorFP32Dim2_9 ubTensor_16((uint64_t)UB_S1888_E1920_T);
UBTileTensorFP32Dim2_8 ubTensor_15((uint64_t)UB_S1824_E1888_T);
UBTileTensorFP32Dim2_6 ubTensor_11((uint64_t)UB_S1728_E1792_T);
UBTileTensorINT32Dim2_5 ubTensor_9((uint64_t)UB_S1408_E1472_T);
GMTileTensorFP32Dim2_2 gmTensor_6((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 1)->Addr, DynLayout2Dim(Shape2Dim(2, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_1 ubTensor_5((uint64_t)UB_S1472_E1728_T);
UBTileTensorINT32Dim1_7 ubTensor_12((uint64_t)UB_S1792_E1824_T);
UBTileTensorFP32Dim2_3 ubTensor_3((uint64_t)UB_S256_E1280_T);
UBTileTensorFP32Dim2_6 ubTensor_20((uint64_t)UB_S1824_E1888_T);
UBTileTensorFP32Dim2_4 ubTensor_7((uint64_t)UB_S1280_E1408_T);
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout2Dim(Shape2Dim(2, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E256_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim(0, 0));
set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
TBitSort<1, 0, 1>(ubTensor_3, ubTensor_1);
TLoad(ubTensor_5, gmTensor_6, Coord2Dim(0, 0));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
pipe_barrier(PIPE_V);
TMrgSort<1, 8, 1>(ubTensor_7, ubTensor_3);
pipe_barrier(PIPE_V);
TExtract<8, 1, 1>(ubTensor_9, ubTensor_7);
pipe_barrier(PIPE_V);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TgatherElement<4>(ubTensor_11, ubTensor_5, ubTensor_9, ubTensor_12);
pipe_barrier(PIPE_V);
TRowSumSingle(ubTensor_15, ubTensor_11, ubTensor_16);
pipe_barrier(PIPE_V);
TAddS<float>(ubTensor_15, ubTensor_15, 9.99999968e-21);
pipe_barrier(PIPE_V);
TExpand<3>(ubTensor_20, ubTensor_15);
pipe_barrier(PIPE_V);
TDiv(ubTensor_20, ubTensor_11, ubTensor_20);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_25, ubTensor_20, Coord2Dim(0, 0));
}
)!!!";

    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk