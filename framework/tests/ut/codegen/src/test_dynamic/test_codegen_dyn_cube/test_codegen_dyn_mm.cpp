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
 * \file test_codegen_dyn_mm.cpp
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
class TestCodegenDynMM : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynMM, TestDynMatmulTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);

    std::vector<int64_t> shape = {64, 64};
    std::vector<int64_t> tileShape = {64, 64};
    std::vector<int64_t> shapeBias = {1, 64};
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    Tensor inputA(DT_FP16, shape, "A");
    Tensor inputB(DT_FP16, shape, "B");
    Tensor output(DT_FP16, shape, "C");

    std::string funcName = "TestDynMatmulTileTensor";
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
    auto localTensorA = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0A, shape, dynValidShape});
    auto localTensorB = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0B, shape, dynValidShape});
    auto localTensorBias =
        CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_BT, shapeBias, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});

    auto &op =
        function->AddOperation(Opcode::OP_A_MUL_B, {localTensorA, localTensorB, localTensorBias}, {localOutTensor});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "has_bias", true);

    function->GetTensorMap().inverseMap_[localTensorA->GetMagic()] = localTensorA;
    function->GetTensorMap().inverseMap_[localTensorB->GetMagic()] = localTensorB;
    function->GetTensorMap().inverseMap_[localTensorBias->GetMagic()] = localTensorBias;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop({symbolManager, *function, *function->rootFunc_->programs_[0], op, {}});

    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TMatmul(l0cTensor_1, l0aTensor_2, l0bTensor_3, btTensor_4);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynMM, TestMatmulMXTileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    std::vector<int64_t> mxShape = {64, 64};
    std::vector<int64_t> shapeBias = {1, 64};
    TileShape::Current().SetVecTile(mxShape);
    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    Tensor inputA(DT_FP16, mxShape, "A");
    Tensor inputB(DT_FP16, mxShape, "B");
    Tensor output(DT_FP16, mxShape, "C");

    std::string funcNameMx = "TestMatmulMXTileTensor";
    FUNCTION(funcNameMx, {inputA, inputB, output}) {
        LOOP(funcNameMx, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcNameMx + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorAMX = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0AMX, mxShape, dynValidShape});
    auto localTensorBMX = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0BMX, mxShape, dynValidShape});
    auto localTensorBias =
        CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_BT, shapeBias, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, mxShape, dynValidShape});

    auto &op =
        function->AddOperation(Opcode::OP_A_MUL_B, {localTensorAMX, localTensorBMX, localTensorBias}, {localOutTensor});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "has_bias", true);

    function->GetTensorMap().inverseMap_[localTensorAMX->GetMagic()] = localTensorAMX;
    function->GetTensorMap().inverseMap_[localTensorBMX->GetMagic()] = localTensorBMX;
    function->GetTensorMap().inverseMap_[localTensorBias->GetMagic()] = localTensorBias;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    std::shared_ptr<SymbolManager> symbolManagerMX = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManagerMX);
    CodeGenOpCloudNPU cop({symbolManagerMX, *function, *function->rootFunc_->programs_[0], op, {}});

    std::string res = cop.GenOpCode();
    std::string expect = R"!!!(MatmulMX(l0cTensor_1, l0a_mxTensor_2, l0b_mxTensor_3, btTensor_4);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk