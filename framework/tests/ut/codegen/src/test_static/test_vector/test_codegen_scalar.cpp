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
 * \file test_codegen_scalar.cpp
 * \brief Unit test for codegen.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr int VALUE128 = 128;
constexpr float F_127 = 127.0;

class TestCodegenScalar : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

void TestQuant(std::vector<int64_t> &inputShape) {
    int shapeDim = inputShape.size();
    std::vector<int64_t> scaleShape(shapeDim, 0);
    for (int i = 0; i < shapeDim; i++) {
        scaleShape[i] = (i == shapeDim - 1) ? 1 : inputShape[i];
    }

    std::vector<int64_t> vecTileShape = {VALUE128, VALUE128};

    // depend on shapeDim
    switch (shapeDim) {
        case DIM2: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]); break;
        case DIM3: TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[0], vecTileShape[1]); break;
        case DIM4: TileShape::Current().SetVecTile(1, 1, vecTileShape[0], vecTileShape[1]); break;
        default: ASSERT(true) << "unsupport dim " << shapeDim << " \n"; break;
    }

    Tensor input(DataType::DT_FP16, inputShape, "input");
    Tensor output(DataType::DT_INT8, inputShape, "output");
    Tensor scaleDeQuant(DataType::DT_FP32, scaleShape, "scaleDeQuant");

    std::string funcName = "Quant";
    FUNCTION(funcName, {input, output, scaleDeQuant}) {
        auto res = Quant(input);
        output = std::get<0>(res);
        scaleDeQuant = std::get<1>(res);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenScalar, TestQuant_32_1_7168) {
    std::vector<int64_t> inputShape = {32, 1, 7168};
    TestQuant(inputShape);
}

TEST_F(TestCodegenScalar, TestScalarOp) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarAddS";
    FUNCTION(funcName, {input, output}) {
        auto output_a = ScalarAddS(input, Element(DataType::DT_FP32, F_127), true);
        auto output_b = ScalarSubS(output_a, Element(DataType::DT_FP32, F_127), true);
        auto output_c = ScalarMulS(output_b, Element(DataType::DT_FP32, F_127), true);
        auto output_d = ScalarDivS(output_c, Element(DataType::DT_FP32, F_127), true);
        output = ScalarMaxS(output_d, Element(DataType::DT_FP32, F_127), true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenScalar, TestPipeAll) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "UBSpillOut", SYMBOL_STACK_BASE);
    const std::vector<int64_t> offset = {0, 0};
    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    Operation &syncOp = function->AddOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {ddrTensor}, {ubTensor});
    syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1};

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    CodeGenOpCloudNPU cop({symbolManager, *function, *(function->rootFunc_->programs_[0]), syncOp});
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    std::string res = cop.GenOpCode();
    std::string expect = R"!!!(pipe_barrier(PIPE_ALL);
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenScalar, TestAicpuCallOp) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestAicpuCallOp";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    Operation &op = function->AddOperation(npu::tile_fwk::Opcode::OP_AICPU_CALL_AIV, {ubTensor}, {});
    op.SetAttribute(OpAttributeKey::aicpuCall, 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    CodeGenOpCloudNPU cop({symbolManager, *function, *(function->rootFunc_->programs_[0]), op});
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    std::string res = cop.GenOpCode();
    std::string expect = R"!!!(TileOp::AicpuCall<0,0>(GET_CURRENT_TASKID());
)!!!";

    EXPECT_EQ(res, expect);
}

void TestCVSyncBody(Opcode syncOpcode) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";

    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto &op = function->AddOperation(syncOpcode, {localTensor}, {localOutTensor});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, function->GetFunctionType());
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    std::string res = cop.GenOpCode();
    std::string expect;
    if (syncOpcode == Opcode::OP_CV_SYNC_SRC) {
        expect = R"!!!(set_intra_block(PIPE_S, 0);
)!!!";
    } else {
        expect = R"!!!(wait_intra_block(PIPE_S, 0);
)!!!";
    }
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenScalar, InjectSyncSet) {
    TestCVSyncBody(Opcode::OP_CV_SYNC_SRC);
}

TEST_F(TestCodegenScalar, InjectSyncWait) {
    TestCVSyncBody(Opcode::OP_CV_SYNC_DST);
}
} // namespace npu::tile_fwk
