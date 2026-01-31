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
 * \file test_codegen_dyn_cmp.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynCmp : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynCmp, TestDynOpCmp) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestDynOpCmp";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorInput1 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput2 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(
        Opcode::OP_CMP, {localTensorInput1, localTensorInput2}, {localTensorRes, localTensorTmp});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorInput1->GetMagic()] = localTensorInput1;
    function->GetTensorMap().inverseMap_[localTensorInput2->GetMagic()] = localTensorInput2;
    function->GetTensorMap().inverseMap_[localTensorRes->GetMagic()] = localTensorRes;
    function->GetTensorMap().inverseMap_[localTensorTmp->GetMagic()] = localTensorTmp;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynCompare<float, 1, 64, 64, 1, 64, 64, 1, 64, 64, 0, 0>((__ubuf__ uint8_t*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, (__ubuf__ uint8_t*)UB_S0_E0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCmp, TestDynOpCmpS) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestDynOpCmpS";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorInput1 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_CMPS, {localTensorInput1}, {localTensorRes, localTensorTmp});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", 0);
    op.SetAttribute(OpAttributeKey::scalar, scalaVal);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorInput1->GetMagic()] = localTensorInput1;
    function->GetTensorMap().inverseMap_[localTensorRes->GetMagic()] = localTensorRes;
    function->GetTensorMap().inverseMap_[localTensorTmp->GetMagic()] = localTensorTmp;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynCmps<float, 1, 64, 64, 1, 64, 64, 0, 0>((__ubuf__ uint8_t*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, (__ubuf__ uint8_t*)UB_S0_E0, 1.000000);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCmp, CmpTileTensor) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    InsertTileTensorOp(Opcode::OP_CMP, "TCompare");
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string cmpFuncName = "CmpTileTensor";
    config::SetBuildStatic(true);
    FUNCTION(cmpFuncName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + cmpFuncName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    localTensor->UpdateDynValidShape(dynValidShape);
    localOutTensor->UpdateDynValidShape(dynValidShape);
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto &op = function->AddOperation(Opcode::OP_CMP, {localTensor, localTensor}, {localOutTensor, localOutTensor});
    int64_t cmpParam = 0;
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", cmpParam);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", cmpParam);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    cop.UpdateTileTensorInfo();
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TCompare<0, 0>(ubTensor_0, ubTensor_0, ubTensor_0, ubTensor_0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk