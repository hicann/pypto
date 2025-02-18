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
 * \file test_codegen_dyn_spillout.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenDynSpillOut : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynSpillOut, UBSpillOut) {

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
    function->SetUnderDynamicFunction(true);
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "UBSpillOut", SYMBOL_STACK_BASE);
    const std::vector<int64_t> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto ubTensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    ubTensor->UpdateSubgraphID(0);
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    ubTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    ubTensor->SetMagic(3);
    ubTensor->SetAttr(OpAttributeKey::needAlloc, true);
    ubTensor->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    ubTensor->memoryrange = TileRange{0, 0, 0};

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[ubTensor->GetMagic()] = ubTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    cop.GenOpCode();
}

TEST_F(TestCodegenDynSpillOut, L1SpillOut) {

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
    function->SetUnderDynamicFunction(true);
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(DataType::DT_FP32, shape, TileOpFormat::TILEOP_ND, "L1SpillOut", SYMBOL_STACK_BASE);
    const std::vector<int64_t> offset = {0, 0};

    auto ddrTensor = std::make_shared<LogicalTensor>(*function, ddrRawTensor, offset, shape);
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);

    auto l1Tensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    l1Tensor->UpdateSubgraphID(0);
    l1Tensor->SetMemoryTypeOriginal(MemoryType::MEM_L1);
    l1Tensor->SetMemoryTypeToBe(MemoryType::MEM_L1);
    l1Tensor->SetMagic(3);
    l1Tensor->SetAttr(OpAttributeKey::needAlloc, true);
    l1Tensor->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    l1Tensor->memoryrange = TileRange{0, 0, 0};

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {l1Tensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[l1Tensor->GetMagic()] = l1Tensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    cop.GenOpCode();
}
} // namespace npu::tile_fwk
