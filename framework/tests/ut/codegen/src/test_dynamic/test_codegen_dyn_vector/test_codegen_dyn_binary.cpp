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
 * \file test_codegen_dyn_binary.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynBinary : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

void TestAddDynBody(const std::vector<int64_t> &shape, const std::vector<int64_t> &tile_shape, const std::string &name,
    bool isNeedCalcMinForBinaryOperands = false) {
    TileShape::Current().SetVecTile(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION(name, {input_a, input_b, output}) {
        LOOP(name, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(input_a, input_b);
        }
    }

    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
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
            if (op.GetOpcode() == Opcode::OP_ADD && isNeedCalcMinForBinaryOperands) {
                op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
            }
        }
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar(), false, ""};
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBinary, TestCodegenAddDim2) {
    TestAddDynBody({64, 64}, {64, 64}, "ADD");
}

TEST_F(TestCodegenDynBinary, TestCodegenAddDim2SrcNotSameShape) {
    TestAddDynBody({64, 64}, {64, 64}, "ADD", true);
}

TEST_F(TestCodegenDynBinary, TestAddsDynamic) {
    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, "A");
    Element value(DataType::DT_FP32, 1.5);
    Tensor output(DataType::DT_FP32, shape, "C");
    ConfigManager::Instance();

    std::string funcName = "ADD_S";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(input_a, value);
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
        subFunc.second->dynParamTable_.emplace("sym_4_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_1", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBinary, TestGatherEle) {
    constexpr const int32_t nRoutedExperts = 256;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({16, 32});
    Tensor inputScores(DT_INT32, outputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    std::string funcName = "GATHER_ELEMET_T";
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            outputTensor = GatherElements(inputTmpScores, inputScores, 1); // [b*s,8]
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
        DynParamInfo fakeParam = {2, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar(), false, ""};
        subFunc.second->InsertDynParam("sym_2_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_2_dim_1", fakeParam);
        subFunc.second->InsertDynParam("sym_4_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_4_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
TEST_F(TestCodegenDynBinary, TestGatherEleTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);

    constexpr const int32_t nRoutedExperts = 256;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({16, 32});
    Tensor inputScores(DT_INT32, outputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    std::string funcName = "GATHER_ELEMET_TILETENSOR";
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            outputTensor = GatherElements(inputTmpScores, inputScores, 1); // [b*s,8]
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
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 5182520263222270487

extern "C" [aicore] void TENSOR_GATHER_ELEMET_TILETENSOR_Unroll1_PATH0_hiddenfunc0_7_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E2048 = (float __ubuf__ *)get_imm(0x0); // size: 0x800
float *UB_S0_E2048_T = (float *)get_imm(0x0); // size: 0x800
int32_t __ubuf__ *UB_S2048_E2112 = (int32_t __ubuf__ *)get_imm(0x800); // size: 0x40
int32_t *UB_S2048_E2112_T = (int32_t *)get_imm(0x800); // size: 0x40
float __ubuf__ *UB_S2112_E2176 = (float __ubuf__ *)get_imm(0x840); // size: 0x40
float *UB_S2112_E2176_T = (float *)get_imm(0x840); // size: 0x40
int32_t __ubuf__ *UB_S2176_E2208 = (int32_t __ubuf__ *)get_imm(0x880); // size: 0x20
int32_t *UB_S2176_E2208_T = (int32_t *)get_imm(0x880); // size: 0x20
uint64_t sym_10_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 0);
uint64_t sym_10_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 1);
uint64_t sym_6_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_6_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
uint64_t sym_9_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_9_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
using GMTileTensorFP32Dim2_7 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_5 = TileTensor<float, LocalLayout2Dim<2, 8>, Hardware::UB>;
using UBTileTensorINT32Dim1_6 = TileTensor<int32_t, LocalLayout1Dim<8>, Hardware::UB>;
using GMTileTensorINT32Dim2_4 = TileTensor<__gm__ int32_t, DynLayout2Dim, Hardware::GM>;
using UBTileTensorINT32Dim2_3 = TileTensor<int32_t, LocalLayout2Dim<2, 8>, Hardware::UB>;
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, LocalLayout2Dim<2, 256>, Hardware::UB>;
UBTileTensorINT32Dim1_6 ubTensor_6((uint64_t)UB_S2176_E2208_T, (Shape2Dim(sym_9_dim_0, sym_9_dim_1)));
UBTileTensorFP32Dim2_5 ubTensor_5((uint64_t)UB_S2112_E2176_T, (Shape2Dim(sym_9_dim_0, sym_9_dim_1)));
GMTileTensorINT32Dim2_4 gmTensor_4((__gm__ int32_t*)GET_PARAM_ADDR(param, 0, 0), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 0)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 0))));
UBTileTensorINT32Dim2_3 ubTensor_3((uint64_t)UB_S2048_E2112_T, (Shape2Dim(sym_9_dim_0, sym_9_dim_1)));
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 0)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 0))));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E2048_T, (Shape2Dim(sym_6_dim_0, sym_6_dim_1)));
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
TLoad(ubTensor_3, gmTensor_4, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TgatherElement<4>(ubTensor_5, ubTensor_1, ubTensor_3, ubTensor_6);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_4, ubTensor_5, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 1))));
}
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynBinary, AddUnalignTileTensor) {
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(64, 64);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    // NEXTNEXT: delete after tileop adapted layout mode
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> inputShape = {b * sq, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor input1(DT_FP32, inputShape, "intput1");
    Tensor input2(DT_FP32, inputShape, "intput2");
    Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    std::string loopName = "L0_TILETENSOR";
    FUNCTION("main", {input1, input2, curSeq}, {out}) {
        LOOP(loopName, FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            auto seq = GetTensorData(curSeq, {batchId, 0});
            Tensor intput11 = View(input1, {sq, d}, {seq, d}, {batchId, 0});
            Tensor intput22 = View(input2, {sq, d}, {seq, d}, {batchId, 0});
            auto tmp = Add(intput11, intput22);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 100);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input1, 1.0),
        RawTensorData::CreateConstantTensor<float>(input2, 1.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

#if ENABLE_HIDDENLOOP
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + loopName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + loopName + SUB_FUNC_SUFFIX);
#endif

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
#if ENABLE_HIDDENLOOP
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 13526864639772037405

extern "C" [aicore] void TENSOR_L0_TILETENSOR_Unroll1_PATH0_hiddenfunc0_7_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E16384 = (float __ubuf__ *)get_imm(0x0); // size: 0x4000
float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float __ubuf__ *UB_S16384_E32768 = (float __ubuf__ *)get_imm(0x4000); // size: 0x4000
float *UB_S16384_E32768_T = (float *)get_imm(0x4000); // size: 0x4000
uint64_t sym_13_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 0);
uint64_t sym_13_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 1);
uint64_t sym_76_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_76_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
uint64_t sym_77_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_77_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, LocalLayout2Dim<64, 64>, Hardware::UB>;
GMTileTensorFP32Dim2_2 gmTensor_8((__gm__ float*)GET_PARAM_ADDR(param, 2, 19), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 2, 19)), Stride2Dim(GET_PARAM_STRIDE_2(param, 2, 19))));
GMTileTensorFP32Dim2_2 gmTensor_4((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1))));
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 1, 10)), Stride2Dim(GET_PARAM_STRIDE_2(param, 1, 10))));
UBTileTensorFP32Dim2_1 ubTensor_3((uint64_t)UB_S16384_E32768_T, (Shape2Dim(sym_77_dim_0, sym_77_dim_1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E16384_T, (Shape2Dim(sym_76_dim_0, sym_76_dim_1)));
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
TLoad(ubTensor_3, gmTensor_4, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TAdd(ubTensor_1, ubTensor_1, ubTensor_3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_8, ubTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 1))));
}
)!!!";
#else
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 13526864639772037405

extern "C" [aicore] void TENSOR_L0_TILETENSOR_Unroll1_PATH0_3_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E16384 = (float __ubuf__ *)get_imm(0x0); // size: 0x4000
float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float __ubuf__ *UB_S16384_E32768 = (float __ubuf__ *)get_imm(0x4000); // size: 0x4000
float *UB_S16384_E32768_T = (float *)get_imm(0x4000); // size: 0x4000
uint64_t sym_13_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 0);
uint64_t sym_13_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 1);
uint64_t sym_43_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_43_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
uint64_t sym_44_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_44_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, LocalLayout2Dim<64, 64>, Hardware::UB>;
GMTileTensorFP32Dim2_2 gmTensor_4((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1))));
UBTileTensorFP32Dim2_1 ubTensor_3((uint64_t)UB_S16384_E32768_T, (Shape2Dim(sym_44_dim_0, sym_44_dim_1)));
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 1, 10)), Stride2Dim(GET_PARAM_STRIDE_2(param, 1, 10))));
GMTileTensorFP32Dim2_2 gmTensor_8((__gm__ float*)GET_PARAM_ADDR(param, 2, 19), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 2, 19)), Stride2Dim(GET_PARAM_STRIDE_2(param, 2, 19))));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E16384_T, (Shape2Dim(sym_43_dim_0, sym_43_dim_1)));
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
TLoad(ubTensor_3, gmTensor_4, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TAdd(ubTensor_1, ubTensor_1, ubTensor_3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_8, ubTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 1))));
}
)!!!";
#endif

    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk
