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
 * \file test_codegen_sort.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenSort : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

struct TopKParams {
    int32_t shape0;
    int32_t shape1;
    int32_t k;
    bool isLargest;
};
void TopKOnBoardFunc(TopKParams &params) {
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 1}});

    int32_t shape0 = params.shape0;
    int32_t shape1 = params.shape1;
    int32_t k = params.k;
    bool isLargest = params.isLargest;

    std::vector<int64_t> input_shape = {shape0, shape1};
    std::vector<int64_t> output_shape = {shape0, k};
    TileShape::Current().SetVecTile({shape0, shape1});
    Tensor input_a(DataType::DT_FP32, input_shape, "A");
    auto output = std::make_tuple(
        Tensor(DataType::DT_FP32, output_shape, "npu_val"), Tensor(DataType::DT_FP32, output_shape, "resDics"));

    FUNCTION("TOPK_T_TILETENSOR", {input_a, std::get<0>(output), std::get<1>(output)}) {
        output = TopK(input_a, k, -1, isLargest);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TOPK_T_TILETENSOR");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 14959618075485442411

extern "C" [aicore] void TENSOR_TOPK_T_TILETENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E16384 = (float __ubuf__ *)get_imm(0x0); // size: 0x4000
float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float __ubuf__ *UB_S16384_E49152 = (float __ubuf__ *)get_imm(0x4000); // size: 0x8000
float *UB_S16384_E49152_T = (float *)get_imm(0x4000); // size: 0x8000
float __ubuf__ *UB_S49152_E49408 = (float __ubuf__ *)get_imm(0xc000); // size: 0x100
float *UB_S49152_E49408_T = (float *)get_imm(0xc000); // size: 0x100
float __ubuf__ *UB_S49408_E49664 = (float __ubuf__ *)get_imm(0xc100); // size: 0x100
float *UB_S49408_E49664_T = (float *)get_imm(0xc100); // size: 0x100
float __ubuf__ *UB_S49664_E66048 = (float __ubuf__ *)get_imm(0xc200); // size: 0x4000
float *UB_S49664_E66048_T = (float *)get_imm(0xc200); // size: 0x4000
int32_t __ubuf__ *UB_S66048_E82432 = (int32_t __ubuf__ *)get_imm(0x10200); // size: 0x4000
int32_t *UB_S66048_E82432_T = (int32_t *)get_imm(0x10200); // size: 0x4000
using UBTileTensorINT32Dim2_5 = TileTensor<int32_t, StaticLayout2Dim<128, 32, 128, 32>, Hardware::UB>;
using GMTileTensorINT32Dim2_6 = TileTensor<__gm__ int32_t, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_3 = TileTensor<float, StaticLayout2Dim<128, 64, 128, 64>, Hardware::UB>;
using UBTileTensorFP32Dim2_4 = TileTensor<float, StaticLayout2Dim<1, 64, 1, 64>, Hardware::UB>;
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, StaticLayout2Dim<128, 32, 128, 32>, Hardware::UB>;
GMTileTensorINT32Dim2_6 gmTensor_15((__gm__ int32_t*)((__gm__ GMTensorInfo*)(param) + 1)->Addr, DynLayout2Dim(Shape2Dim(128, 32), Stride2Dim(32, 1)));
GMTileTensorFP32Dim2_2 gmTensor_13((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 2)->Addr, DynLayout2Dim(Shape2Dim(128, 32), Stride2Dim(32, 1)));
UBTileTensorINT32Dim2_5 ubTensor_11((uint64_t)UB_S66048_E82432_T);
UBTileTensorFP32Dim2_1 ubTensor_9((uint64_t)UB_S49664_E66048_T);
UBTileTensorFP32Dim2_4 ubTensor_7((uint64_t)UB_S49408_E49664_T);
UBTileTensorFP32Dim2_4 ubTensor_4((uint64_t)UB_S49152_E49408_T);
UBTileTensorFP32Dim2_3 ubTensor_3((uint64_t)UB_S16384_E49152_T);
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout2Dim(Shape2Dim(128, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E16384_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim(0, 0));
set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
TBitSort<1, 0, 1>(ubTensor_3, ubTensor_1, ubTensor_4);
pipe_barrier(PIPE_V);
SUBKERNEL_PHASE2
TMrgSort<1, 32, 32>(ubTensor_3, ubTensor_3, ubTensor_7);
pipe_barrier(PIPE_V);
TExtract<32, 0, 1>(ubTensor_9, ubTensor_3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TExtract<32, 1, 1>(ubTensor_11, ubTensor_3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_13, ubTensor_9, Coord2Dim(0, 0));
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
TStore(gmTensor_15, ubTensor_11, Coord2Dim(0, 0));
}
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenSort, TestTopKTileTensor) {
    TopKParams params;
    params.shape0 = 128;
    params.shape1 = 32;
    params.k = 32;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

} // namespace npu::tile_fwk
