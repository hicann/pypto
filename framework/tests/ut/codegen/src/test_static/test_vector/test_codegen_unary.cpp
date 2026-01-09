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
 * \file test_codegen_unary.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenUnary : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

void TestRowMaxSingleBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Amax(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowMaxSingleDim2) {
    TestRowMaxSingleBody({8, 128}, {8, 1}, {2, 64}, "ROWMAXSINGLE_DIM2");
}

TEST_F(TestCodegenUnary, RowMaxSingleDim3) {
    TestRowMaxSingleBody({8, 4, 128}, {8, 4, 1}, {2, 1, 64}, "ROWMAXSINGLE_DIM3");
}

TEST_F(TestCodegenUnary, RowMaxSingleDim4) {
    TestRowMaxSingleBody({8, 4, 4, 128}, {8, 4, 4, 1}, {2, 1, 1, 64}, "ROWMAXSINGLE_DIM4");
}

void TestRowSumSingleBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Sum(input_a, -1, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowSumSingleDim2) {
    TestRowSumSingleBody({8, 128}, {8, 1}, {2, 64}, "ROWSUMSINGLE_DIM2");
}

TEST_F(TestCodegenUnary, RowSumSingleDim3) {
    TestRowSumSingleBody({8, 4, 128}, {8, 4, 1}, {2, 1, 64}, "ROWSUMSINGLE_DIM3");
}

TEST_F(TestCodegenUnary, RowSumSingleDim4) {
    TestRowSumSingleBody({8, 4, 4, 128}, {8, 4, 4, 1}, {2, 1, 1, 64}, "ROWSUMSINGLE_DIM4");
}

void TestTransposeVnchwconvBody(std::vector<int64_t> shape, std::vector<int64_t> outShape,
    std::vector<int> transposeShape, std::vector<int64_t> tileShape, std::string name,
    bool isSupportTileTensor = false) {
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, outShape, "output");
    config::SetBuildStatic(true);
    FUNCTION(name, {input, output}) {
        output = Transpose(input, transposeShape);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim2) {
    TestTransposeVnchwconvBody({16, 32}, {32, 16}, {0, 1}, {16, 16}, "TRANSPOSE_VNCHWCONV_DIM2");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim4) {
    TestTransposeVnchwconvBody({1, 2, 32, 16}, {1, 2, 16, 32}, {3, 2}, {1, 1, 16, 16}, "TRANSPOSE_VNCHWCONV_DIM4");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim5) {
    TestTransposeVnchwconvBody(
        {1, 1, 2, 32, 16}, {1, 1, 2, 16, 32}, {3, 4}, {1, 1, 1, 16, 16}, "TRANSPOSE_VNCHWCONV_DIM5");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim2TileTensor) {
    TestTransposeVnchwconvBody({16, 32}, {32, 16}, {0, 1}, {16, 16}, "TransposeVnchwconvDim2TileTensor", true);
}

void TestRowMaxExpandBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = RowMaxExpand(input_a);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowMaxExpandDim2) {
    TestRowMaxExpandBody({128, 64}, {128, 64}, {16, 16}, "ROWMAXEXPAND_DIM2");
}

Function &TestFullBody(std::vector<int64_t> shape, std::vector<int64_t> tileShape, std::string name,
    bool isSupportTileTensor = false) {
    if(isSupportTileTensor){
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, shape, "C");
    Element value(DataType::DT_FP32, 2.0);
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Full(value, DT_FP32, shape, {});
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    return *function;
}

TEST_F(TestCodegenUnary, FullDim2TileTensor) {
    Function& func = TestFullBody({32, 32}, {16, 16}, "FULL_DIM2_TILETENSOR", true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 16465914044878388469

extern "C" [aicore] void TENSOR_FULL_DIM2_TILETENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E1024 = (float __ubuf__ *)get_imm(0x0); // size: 0x400
float *UB_S0_E1024_T = (float *)get_imm(0x0); // size: 0x400
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, StaticLayout2Dim<16, 16, 16, 16>, Hardware::UB>;
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout2Dim(Shape2Dim(32, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E1024_T);
TVecDup<float>(ubTensor_1, 2.000000);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_2, ubTensor_1, Coord2Dim(0, 0));
}
)!!!";

    EXPECT_EQ(res, expect);
}

Function &TestCastBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
    std::string name, bool isSupportTileTensor = false) {
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_INT32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Cast(input_a, DT_FP32);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    return *function;
}

TEST_F(TestCodegenUnary, CastDim1) {
    TestCastBody({128}, {128}, {64}, "CAST_DIM2");
}

#if 0
TEST_F(TestCodegenUnary, CastDim1TileTensor) {
    Function& func = TestCastBody({128}, {128}, {64}, "CAST_DIM2_TILETENSOR", true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 7393518754888662784

extern "C" [aicore] void TENSOR_CAST_DIM2_TILETENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
int32_t __ubuf__ *UB_S0_E256 = (int32_t __ubuf__ *)get_imm(0x0); // size: 0x100
int32_t *UB_S0_E256_T = (int32_t *)get_imm(0x0); // size: 0x100
float __ubuf__ *UB_S256_E512 = (float __ubuf__ *)get_imm(0x100); // size: 0x100
float *UB_S256_E512_T = (float *)get_imm(0x100); // size: 0x100
using GMTileTensorFP32Dim1_4 = TileTensor<__gm__ float, DynLayout1Dim, Hardware::GM>;
using UBTileTensorFP32Dim1_3 = TileTensor<float, StaticLayout1Dim<64, 64>, Hardware::UB>;
using GMTileTensorINT32Dim1_2 = TileTensor<__gm__ int32_t, DynLayout1Dim, Hardware::GM>;
using UBTileTensorINT32Dim1_1 = TileTensor<int32_t, StaticLayout1Dim<64, 64>, Hardware::UB>;
GMTileTensorFP32Dim1_4 gmTensor_5((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 1)->Addr, DynLayout1Dim(Shape1Dim(128), Stride1Dim(1)));
UBTileTensorFP32Dim1_3 ubTensor_3((uint64_t)UB_S256_E512_T);
GMTileTensorINT32Dim1_2 gmTensor_2((__gm__ int32_t*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout1Dim(Shape1Dim(128), Stride1Dim(1)));
UBTileTensorINT32Dim1_1 ubTensor_1((uint64_t)UB_S0_E256_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord1Dim(0));
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
SUBKERNEL_PHASE2
TCast<0>(ubTensor_3, ubTensor_1);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_5, ubTensor_3, Coord1Dim(0));
}
)!!!";

    EXPECT_EQ(res, expect);
}
#endif // if 0

Function &TestExpandBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
    std::string name, bool isSupportTileTensor = false) {
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    } else {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");

    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Expand(input_a, outShape);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    return *function;
}

TEST_F(TestCodegenUnary, ExpandDim2Axis0TileTensor) {
    Function &func = TestExpandBody({1, 22}, {22, 22}, {2, 2}, "EXPAND_TILETENSOR", true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 16742946980865972364

extern "C" [aicore] void TENSOR_EXPAND_TILETENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E32 = (float __ubuf__ *)get_imm(0x0); // size: 0x20
float *UB_S0_E32_T = (float *)get_imm(0x0); // size: 0x20
float __ubuf__ *UB_S32_E96 = (float __ubuf__ *)get_imm(0x20); // size: 0x40
float *UB_S32_E96_T = (float *)get_imm(0x20); // size: 0x40
using GMTileTensorFP32Dim2_4 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_3 = TileTensor<float, StaticLayout2Dim<2, 2, 2, 8>, Hardware::UB>;
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, StaticLayout2Dim<1, 2, 1, 8>, Hardware::UB>;
GMTileTensorFP32Dim2_4 gmTensor_5((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 1)->Addr, DynLayout2Dim(Shape2Dim(22, 22), Stride2Dim(22, 1)));
UBTileTensorFP32Dim2_3 ubTensor_3((uint64_t)UB_S32_E96_T);
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout2Dim(Shape2Dim(1, 22), Stride2Dim(22, 1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E32_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim(0, 0));
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
SUBKERNEL_PHASE2
TExpand<2>(ubTensor_3, ubTensor_1);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_5, ubTensor_3, Coord2Dim(0, 0));
}
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenUnary, ExpandDim2Axis0) {
    TestExpandBody({1, 22}, {22, 22}, {2, 2}, "EXPAND_T");
}

TEST_F(TestCodegenUnary, ExpandDim4Axis0) {
    TestExpandBody({1, 22, 8, 17}, {4, 22, 8, 17}, {2, 16, 4, 8}, "EXPAND_T");
}

TEST_F(TestCodegenUnary, ExpandDim4Axis1) {
    TestExpandBody({4, 1, 8, 17}, {4, 22, 8, 17}, {2, 16, 4, 8}, "EXPAND_T");
}

void TestRowSumBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape,
    std::string name, unsigned axis) {
    TileShape::Current().SetVecTile(tileShape);

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outShape, "C");

    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Sum(input_a, axis, true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowSumDim4Axis2) {
    TestRowSumBody({3, 2, 8, 255}, {3, 2, 1, 255}, {2, 8, 8, 255}, "ROWSUMAXIS2", 2);
}

TEST_F(TestCodegenUnary, RowSumDim3Axis1) {
    TestRowSumBody({2, 8, 255}, {2, 1, 255}, {8, 8, 255}, "ROWSUMAXIS1", 1);
}

TEST_F(TestCodegenUnary, RowSumDim2Axis0) {
    TestRowSumBody({8, 255}, {1, 255}, {8, 255}, "ROWSUMAXIS0", 0);
}

TEST_F(TestCodegenUnary, TestVecDup) {
    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_INT32, static_cast<int64_t>(2));
    std::string funcName = "VECDUP";
    TileShape::Current().SetVecTile({16, 1, 16});

    Tensor output(DataType::DT_INT32, shape, "C");
    config::SetBuildStatic(true);
    FUNCTION(funcName, {output}) {
        output = npu::tile_fwk::Full(src, DT_INT32, shape);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, TestVecDupUnaligned) {
    std::vector<int64_t> shape{2, 2, 256, 7};
    Element src(DataType::DT_FP32, 2.0);
    TileShape::Current().SetVecTile({1, 1, 256, 16});

    Tensor output(DataType::DT_FP32, shape, "C");

    std::string funcName = "VECDUP_T";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {output}) {
        output = npu::tile_fwk::Full(src, DT_FP32, shape);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, TestRowMaxLine) {
    config::SetBuildStatic(true);

    std::vector<int64_t> shape = {2, 2, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestRowMaxLine";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto &op = function->AddOperation(Opcode::OP_ROWMAXLINE, {localTensorSrc}, {localTensorDst});
    op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, function->GetFunctionType());
    function->GetTensorMap().inverseMap_[localTensorSrc->GetMagic()] = localTensorSrc;
    function->GetTensorMap().inverseMap_[localTensorDst->GetMagic()] = localTensorDst;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::Trowmaxline_<float, 1, 2, 2, 64, 2, 2, 64, 2, 2, 64, 2>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0);
)!!!";
    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk