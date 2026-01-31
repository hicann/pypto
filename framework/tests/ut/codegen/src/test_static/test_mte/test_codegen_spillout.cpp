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
 * \file test_codegen_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen_op.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenSpillOut : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    }

    void TearDown() override {}
};

TEST_F(TestCodegenSpillOut, UBSpillOut) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "UBSpillOut";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, shape,
        TileOpFormat::TILEOP_ND, "UBSpillOut", SYMBOL_STACK_BASE); // trawmagic must use SYMBOL_STACK_BASE
    const std::vector<int64_t> offset = {0, 0};
    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, function->GetFunctionType());
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::UBCopyOut<float, 1, 1, 1, 64, 64, /*dst stride*/ 1, 1, 64, 64,/*src stride*/ 1, 1, 64, 64 >((__gm__ float*)GMStackBase, (__ubuf__ float*)UB_S0_E0);
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenSpillOut, UBSpillOutTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "UBSpillOutTileTensor";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, shape,
        TileOpFormat::TILEOP_ND, "UBSpillOutTileTensor", SYMBOL_STACK_BASE); // trawmagic must use SYMBOL_STACK_BASE
    const std::vector<int64_t> offset = {0, 0};
    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    op.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({16, 16}), shapeImme, shapeImme));

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, function->GetFunctionType());
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;
    cop.UpdateTileTensorInfo();

    std::string res = symbolManager->GenTileTensorDefList();
    std::string expect = R"!!!(UBTileTensorFP32Dim2_2 ubTensor_2((uint64_t)UB_S0_E0_T);
GMTileTensorFP32Dim2_1 gmTensor_1((__gm__ float*)((__gm__ uint8_t*)GMStackBase + 16), DynLayout2Dim(Shape2Dim(64, 64), Stride2Dim(64, 1)));
)!!!";
    EXPECT_EQ(res, expect);

    res = cop.GenOpCode();
    expect = R"!!!(TStore(gmTensor_1, ubTensor_2, Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenSpillOut, L1SpillOut) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "L1SpillOut";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, shape,
        TileOpFormat::TILEOP_ND, "L1SpillOut", SYMBOL_STACK_BASE); // trawmagic must use SYMBOL_STACK_BASE
    const std::vector<int64_t> offset = {0, 0};
    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    const std::vector<SymbolicScalar> dynValidShape = {SymbolicScalar("S0"), SymbolicScalar("S1")};
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_L1_COPY_OUT, {l1Tensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, function->GetFunctionType());
    function->GetTensorMap().inverseMap_[l1Tensor->GetMagic()] = l1Tensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    cop.GenOpCode();
}
} // namespace npu::tile_fwk
