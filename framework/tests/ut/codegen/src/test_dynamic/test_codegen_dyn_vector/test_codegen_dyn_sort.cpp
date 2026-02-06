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
constexpr const unsigned TOPK_OP_X_IDX = 0;
constexpr const unsigned TOPK_OP_Y_IDX = 1;
constexpr const unsigned TOPK_OP_TMP_IDX = 2;

class TestCodegenDynSort : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
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

Operation &GetTopkOp(Function *function, Opcode opCode, const LogicalTensors &tensors) {
    if (opCode == Opcode::OP_TOPK_SORT) {
        auto &op = function->AddOperation(
            opCode, {tensors[TOPK_OP_Y_IDX]}, {tensors[TOPK_OP_TMP_IDX], tensors[TOPK_OP_X_IDX]});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", 0);
        SymbolicScalar startIdx(1);
        op.SetAttribute(OpAttributeKey::dynScalar, startIdx);
        return op;
    }
    
    auto &op = function->AddOperation(opCode, {tensors[TOPK_OP_X_IDX]}, {tensors[TOPK_OP_Y_IDX]});
    return op;
}

void TestTopkBody(Opcode opCode, const std::string &expect) {
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    Element scalaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestDynTopkSort";
    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    auto yVar = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto tmpVar = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto xVar = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = GetTopkOp(function, opCode, {xVar, yVar, tmpVar});

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[yVar->GetMagic()] = yVar;
    function->GetTensorMap().inverseMap_[tmpVar->GetMagic()] = tmpVar;
    function->GetTensorMap().inverseMap_[xVar->GetMagic()] = xVar;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopkSort) {
    std::string expect =
        R"!!!(TileOp::DynTopKSort<float, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1);
)!!!";
    TestTopkBody(Opcode::OP_TOPK_SORT, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopkMerge) {
    std::string expect =
        R"!!!(TileOp::DynTopKMerge<float, 64, 32>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0);
)!!!";
    TestTopkBody(Opcode::OP_TOPK_MERGE, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopkExract) {
    std::string expect =
        R"!!!(TileOp::DynTopKExtract<float, float, 64, 64, 64, 32>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0);
)!!!";
    TestTopkBody(Opcode::OP_TOPK_EXTRACT, expect);
}

TEST_F(TestCodegenDynSort, TestDynTwoTileMrgSort) {
    auto param = prepareSortParamForUT(Opcode::OP_TWOTILEMRGSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "firstshape", 32);

    std::string res = generateCodeForOp(param.op);
    std::string expect = 
        R"!!!(TileOp::DynTwoTileMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 32>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynExtractSingle) {
    auto param = prepareSortParamForUT(Opcode::OP_EXTRACT_SINGLE);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "maskmode", 0);
    std::string res = generateCodeForOp(param.op);
    std::string expect = 
        R"!!!(TileOp::DynExtractSingle<float, float, 1, 1, 64, 64, 1, 1, 64, 64, 0, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk