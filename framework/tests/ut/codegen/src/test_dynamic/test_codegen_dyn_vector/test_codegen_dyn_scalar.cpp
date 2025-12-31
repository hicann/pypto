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
 * \file test_codegen_dyn_scalar.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynScalar : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

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

TEST_F(TestCodegenDynScalar, TestScalarAdds) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarAddS";
    FUNCTION(funcName, {input, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ScalarAddS(input, Element(DataType::DT_FP32, 127.0), true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);

    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
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
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynScalar, TestScalarDivs) {
    std::vector<int64_t> vecTileShape = {128, 128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{s, b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1], vecTileShape[2]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarDivS";
    FUNCTION(funcName, {input, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ScalarDivS(input, Element(DataType::DT_FP32, 127.0), true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
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
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_2", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynScalar, TestAddsTileTensor) {
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor out(DT_FP32, {-1, s}, "out");
    TileShape::Current().SetVecTile({128, 64});

    auto funcName = "ADDS_TILETENSOR";
    FUNCTION(funcName, {t0}, {out}) {
        auto shape0 = GetInputShape(t0, 0);
        auto loop1 = (shape0 + s - 1) / s;
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop1)) {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            auto t = Add(t0s, Element(DT_FP32, 3.0));
            Assemble(t, {idx * s, 0}, out);
        }
    }

#if ENABLE_HIDDENLOOP
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX);
#endif
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
#if ENABLE_HIDDENLOOP
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 8862770922887829658

extern "C" [aicore] void TENSOR_ADDS_TILETENSOR_Unroll1_PATH0_hiddenfunc0_8_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E4096 = (float __ubuf__ *)get_imm(0x0); // size: 0x1000
float *UB_S0_E4096_T = (float *)get_imm(0x0); // size: 0x1000
uint64_t sym_2_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_2_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
uint64_t sym_7_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_7_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, LocalLayout2Dim<32, 32>, Hardware::UB>;
GMTileTensorFP32Dim2_2 gmTensor_5((__gm__ float*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 1, 10)), Stride2Dim(GET_PARAM_STRIDE_2(param, 1, 10))));
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1))));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E4096_T, (Shape2Dim(sym_2_dim_0, sym_2_dim_1)));
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
SUBKERNEL_PHASE2
TAddS<float>(ubTensor_1, ubTensor_1, 3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_5, ubTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
}
)!!!";
#else
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 8862770922887829658

extern "C" [aicore] void TENSOR_ADDS_TILETENSOR_Unroll1_PATH0_4_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E4096 = (float __ubuf__ *)get_imm(0x0); // size: 0x1000
float *UB_S0_E4096_T = (float *)get_imm(0x0); // size: 0x1000
uint64_t sym_2_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_2_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
uint64_t sym_7_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_7_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, LocalLayout2Dim<32, 32>, Hardware::UB>;
GMTileTensorFP32Dim2_2 gmTensor_5((__gm__ float*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 1, 10)), Stride2Dim(GET_PARAM_STRIDE_2(param, 1, 10))));
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1))));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E4096_T, (Shape2Dim(sym_2_dim_0, sym_2_dim_1)));
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
SUBKERNEL_PHASE2
TAddS<float>(ubTensor_1, ubTensor_1, 3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_5, ubTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
}
)!!!";
#endif

    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk