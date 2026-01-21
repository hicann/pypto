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
 * \file test_codegen_row_sum_line.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenRowSumLine : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenRowSumLine, TestOperationRowSumLineTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "C");

    std::string funcName = "Reduce3dimMoe_TILERENSOR";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {input_a, output}) {
        output = Sum(input_a, 1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar(), false, ""};
        subFunc.second->InsertDynParam("sym_23_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_23_dim_1", fakeParam);
        subFunc.second->InsertDynParam("sym_23_dim_2", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 12633186027675011673

extern "C" [aicore] void TENSOR_Reduce3dimMoe_TILERENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E32768 = (float __ubuf__ *)get_imm(0x0); // size: 0x8000
float *UB_S0_E32768_T = (float *)get_imm(0x0); // size: 0x8000
float __ubuf__ *UB_S32768_E36864 = (float __ubuf__ *)get_imm(0x8000); // size: 0x1000
float *UB_S32768_E36864_T = (float *)get_imm(0x8000); // size: 0x1000
float __ubuf__ *UB_S36864_E45056 = (float __ubuf__ *)get_imm(0x9000); // size: 0x2000
float *UB_S36864_E45056_T = (float *)get_imm(0x9000); // size: 0x2000
using GMTileTensorFP32Dim3_5 = TileTensor<__gm__ float, DynLayout3Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_4 = TileTensor<float, StaticLayout2Dim<4, 512, 4, 512>, Hardware::UB>;
using UBTileTensorFP32Dim3_3 = TileTensor<float, StaticLayout3Dim<2, 1, 512, 2, 1, 512>, Hardware::UB>;
using GMTileTensorFP32Dim3_2 = TileTensor<__gm__ float, DynLayout3Dim, Hardware::GM>;
using UBTileTensorFP32Dim3_1 = TileTensor<float, StaticLayout3Dim<2, 8, 512, 2, 8, 512>, Hardware::UB>;
GMTileTensorFP32Dim3_5 gmTensor_6((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), DynLayout3Dim(Shape3Dim(6, 1, 1024), Stride3Dim(1024, 1024, 1)));
UBTileTensorFP32Dim2_4 ubTensor_4((uint64_t)UB_S36864_E45056_T);
UBTileTensorFP32Dim3_3 ubTensor_3((uint64_t)UB_S32768_E36864_T);
GMTileTensorFP32Dim3_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), DynLayout3Dim(Shape3Dim(6, 8, 1024), Stride3Dim(8192, 1024, 1)));
UBTileTensorFP32Dim3_1 ubTensor_1((uint64_t)UB_S0_E32768_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord3Dim(0, 0, 0));
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
SUBKERNEL_PHASE2
TRowSumLine<3>(ubTensor_3, ubTensor_1, ubTensor_4);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_6, ubTensor_3, Coord3Dim(0, 0, 0));
}
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk