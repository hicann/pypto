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
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynSpillOut : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynSpillOut, UBSpillOut) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

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

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "UBSpillOut",
        SYMBOL_STACK_BASE, dynValidShape});
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

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
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

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
    function->SetUnderDynamicFunction(true);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1SpillOut",
        SYMBOL_STACK_BASE, dynValidShape});
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {l1Tensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

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

TEST_F(TestCodegenDynSpillOut, L1SpillTileTensor) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    InsertTileTensorOp(Opcode::OP_L1_COPY_IN, "TLoad");
    InsertTileTensorOp(Opcode::OP_L1_COPY_OUT, "TStore");

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "L1SpillTileTensor";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1SpillOut",
        SYMBOL_STACK_BASE, dynValidShape});
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto &op = function->rootFunc_->programs_[0]->AddOperation(Opcode::OP_COPY_OUT, {l1Tensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    auto &op2 = function->rootFunc_->programs_[0]->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {l1Tensor});
    op2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L1, shapeImme, shapeImme));
    op2.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    CodeGenCtx ctx;
    CodeGenCloudNPU codegen(ctx);
    codegen.GenCode(*function, {});
    const std::string res = GetResultFromCpp(*function);
    std::string expect =
        R"!!!(TStore<TStoreConfig<CopyOutMode::ND2ND, 0, 0>>(gmTensor_18, l1Tensor_19, Coord2Dim(0, 0));)!!!";
    CheckStringExist(expect, res);

    expect =
       R"!!!(TLoad<CopyInMode::ND2ND, PaddingMode::NO_PADDING>(l1Tensor_19, gmTensor_18, Coord2Dim(0, 0), 64, 64);)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
