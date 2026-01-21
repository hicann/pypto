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
 * \file test_codegen_dyn_scatter.cpp
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
class TestCodegenDynScatter : public ::testing::Test {
public:
    static void SetUpTestCase() {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    static void TearDownTestCase() {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynScatter, TestDynOpScatterElement) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestDynOpScatterElement";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorIdx = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, {32}});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShapeIdx = {32};
    localTensorSrc->UpdateDynValidShape(dynValidShape);
    localTensorIdx->UpdateDynValidShape(dynValidShapeIdx);
    localTensorDst->UpdateDynValidShape(dynValidShape);

    auto &op = function->AddOperation(Opcode::OP_SCATTER_ELEMENT, {localTensorSrc, localTensorIdx}, {localTensorDst});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
    op.SetAttribute(OpAttributeKey::scalar, scalaVal);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorSrc->GetMagic()] = localTensorSrc;
    function->GetTensorMap().inverseMap_[localTensorIdx->GetMagic()] = localTensorIdx;
    function->GetTensorMap().inverseMap_[localTensorDst->GetMagic()] = localTensorDst;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynTscatterElementS<float, float, float, 1, 1, 32, 1, 64, 64, 3, 0>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (float)1, 1, 1, 1, 32);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynScatter, TestOpDynScatter) {

    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestOpDynScatter";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    auto localTensorSelf = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorSrc = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorIdx = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, {32}});
    auto localTensorDst = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_INT32, MemoryType::MEM_UB, {32}});

    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShapeIdx = {32};
    localTensorSelf->UpdateDynValidShape(dynValidShape);
    localTensorSrc->UpdateDynValidShape(dynValidShape);
    localTensorIdx->UpdateDynValidShape(dynValidShapeIdx);
    localTensorDst->UpdateDynValidShape(dynValidShape);

    auto &op =
        function->AddOperation(Opcode::OP_SCATTER, {localTensorSelf, localTensorIdx, localTensorSrc},
        {localTensorDst, localTensorTmp});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorSelf->GetMagic()] = localTensorSelf;
    function->GetTensorMap().inverseMap_[localTensorSrc->GetMagic()] = localTensorSrc;
    function->GetTensorMap().inverseMap_[localTensorIdx->GetMagic()] = localTensorIdx;
    function->GetTensorMap().inverseMap_[localTensorDst->GetMagic()] = localTensorDst;
    function->GetTensorMap().inverseMap_[localTensorTmp->GetMagic()] = localTensorTmp;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynTscatter<float, float, 1, 1, 32, 1, 64, 64, 1, 64, 64, 3, 0>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 32);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk