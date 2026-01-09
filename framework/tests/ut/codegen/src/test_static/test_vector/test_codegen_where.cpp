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
 * \file test_codegen_where.cpp
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
#include "codegen/cloudnpu/op_print_param_def.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {
class TestCodegenWhere : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    void TearDown() override {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
};

Operation &GetWhereOp(Function *function, Opcode opCode, const LogicalTensors &inputs) {
    Element scalaVal1(DataType::DT_FP32, 1.0);
    Element scalaVal2(DataType::DT_FP32, 2.0);

    if (opCode == Opcode::OP_WHERE_SS) {
        auto &op = function->AddOperation(opCode, {inputs[ToUnderlying(WhereOpIdx::condIdx)]},
            {inputs[ToUnderlying(WhereOpIdx::resIdx)], inputs[ToUnderlying(WhereOpIdx::tempIdx)]});
        std::vector<Element> scalars = {scalaVal1, scalaVal2};
        op.SetAttribute(OpAttributeKey::vectorScalar, scalars);
        op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", 0);
        return op;
    } else if (opCode == Opcode::OP_WHERE_ST || opCode == Opcode::OP_WHERE_TS) {
        auto &op = function->AddOperation(opCode,
            {inputs[ToUnderlying(WhereOpIdx::condIdx)], inputs[ToUnderlying(WhereOpIdx::src1Idx)]},
            {inputs[ToUnderlying(WhereOpIdx::resIdx)], inputs[ToUnderlying(WhereOpIdx::tempIdx)]});
        op.SetAttribute(OpAttributeKey::scalar, scalaVal1);
        op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", 0);
        return op;
    }

    EXPECT_EQ(opCode, Opcode::OP_WHERE_TT) << "Unsupported opcode: " << OpcodeManager::Inst().GetOpcodeStr(opCode);
    auto &op = function->AddOperation(opCode,
        {inputs[ToUnderlying(WhereOpIdx::condIdx)], inputs[ToUnderlying(WhereOpIdx::src0Idx)],
            inputs[ToUnderlying(WhereOpIdx::src1Idx)]},
        {inputs[ToUnderlying(WhereOpIdx::resIdx)], inputs[ToUnderlying(WhereOpIdx::tempIdx)]});
    op.SetAttribute(OP_ATTR_PREFIX + "whereBitMode", 0);
    return op;
}

void TestWhereBody(const Opcode opCode, const std::string &caseName, const std::string &expect) {
    config::SetBuildStatic(true);

    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = caseName;
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    auto localTensorCond = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorInput = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorOther = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorResult = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});

    auto &op = GetWhereOp(
        function, opCode, {localTensorResult, localTensorCond, localTensorTmp, localTensorInput, localTensorOther});

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::STATIC, {}, true);
    function->GetTensorMap().inverseMap_[localTensorCond->GetMagic()] = localTensorCond;
    function->GetTensorMap().inverseMap_[localTensorInput->GetMagic()] = localTensorInput;
    function->GetTensorMap().inverseMap_[localTensorOther->GetMagic()] = localTensorOther;
    function->GetTensorMap().inverseMap_[localTensorResult->GetMagic()] = localTensorResult;
    function->GetTensorMap().inverseMap_[localTensorTmp->GetMagic()] = localTensorTmp;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenWhere, TestOpWhereSS) {
    std::string expect =
        R"!!!(TileOp::Where_SS<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, float(1), float(2), 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_SS, "TestOpWhereSS", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereST) {
    std::string expect =
        R"!!!(TileOp::Where_ST<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, float(1), (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_ST, "TestOpWhereST", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereTS) {
    std::string expect =
        R"!!!(TileOp::Where_TS<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, float(1), 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_TS, "TestOpWhereTS", expect);
}

TEST_F(TestCodegenWhere, TestOpWhereTT) {
    std::string expect =
        R"!!!(TileOp::Where_TT<float, float, /*DstRawShape*/ 1, 64, 64, /*ConditionRawShape*/ 1, 64, 64, /*Src0RawShape*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 1, 1);
)!!!";
    TestWhereBody(Opcode::OP_WHERE_TT, "TestOpWhereTT", expect);
}

} // namespace npu::tile_fwk