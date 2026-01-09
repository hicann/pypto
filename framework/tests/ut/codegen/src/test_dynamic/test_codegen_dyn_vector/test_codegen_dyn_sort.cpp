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
 * \file test_codegen_dyn_sort.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenDynSort : public ::testing::Test {
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

struct TestContext {
    Function *function;
    std::shared_ptr<LogicalTensor> localTensor;
    std::shared_ptr<LogicalTensor> localOutTensor;
    Operation *op;
};

std::string generateCodeForOp(Operation *op) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(*op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    cop.Init(*op);
    return cop.GenOpCode();
}

TestContext prepareSortParamForUT(Opcode opcode) {
    std::vector<int64_t> shape = {64, 64};

    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC3, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC4, dynValidShape});

    auto &op = function->AddOperation(opcode, {localTensor}, {localOutTensor});

    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    TestContext param;
    param.function = function;
    param.localTensor = localTensor;
    param.localOutTensor = localOutTensor;
    param.op = &op;
    return param;
}

TEST_F(TestCodegenDynSort, TestDynBitSort) {
    auto param = prepareSortParamForUT(Opcode::OP_BITSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynBitSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynMrgSort) {
    auto param = prepareSortParamForUT(Opcode::OP_MRGSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynExtract) {
    auto param = prepareSortParamForUT(Opcode::OP_EXTRACT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "mode", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynExtract<float, float, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynTiledMgrSort) {
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestDynTiledMgrSort";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    auto localTensorInput1 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput2 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput3 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_TILEDMRGSORT,
        {localTensorInput1, localTensorInput2, localTensorInput3, localTensorInput3}, {localTensorRes, localTensorTmp});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
    op.SetAttribute(OpAttributeKey::scalar, scalaVal);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorInput1->GetMagic()] = localTensorInput1;
    function->GetTensorMap().inverseMap_[localTensorInput2->GetMagic()] = localTensorInput2;
    function->GetTensorMap().inverseMap_[localTensorInput3->GetMagic()] = localTensorInput3;
    function->GetTensorMap().inverseMap_[localTensorRes->GetMagic()] = localTensorRes;
    function->GetTensorMap().inverseMap_[localTensorTmp->GetMagic()] = localTensorTmp;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynTiledMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 64, 0>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, 64, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk