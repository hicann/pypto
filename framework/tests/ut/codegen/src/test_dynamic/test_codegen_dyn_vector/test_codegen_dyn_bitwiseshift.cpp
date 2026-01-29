/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_bitwiseshift.cpp
 * \brief Unit test for codegen.
 */

#include <iostream>

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "passes/pass_mgr/pass_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
class TestCodegenDynBitwiseShift : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynBitwiseShift, TestCodegenBitwiseRightShiftTileTensor) {
    std::vector<int64_t> shape{32, 32};
    TileShape::Current().SetVecTile(shape);
    std::string funcName = "BitwiseRightShift_TileTensor";
    Tensor input_a(DT_INT16, shape, "A");
    Tensor input_b(DT_INT16, shape, "B");
    Tensor output(DT_INT16, shape, "C");

    FUNCTION(funcName, {input_a, input_b, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = BitwiseRightShift(input_a, input_b);
        }
    }

    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBitwiseShift, TestCodegenBitwiseLeftShiftTileTensor) {
    std::vector<int64_t> shape{32, 32};
    TileShape::Current().SetVecTile(shape);
    std::string funcName = "BitwiseLeftShift_TileTensor";
    Tensor input_a(DT_INT16, shape, "A");
    Tensor input_b(DT_INT16, shape, "B");
    Tensor output(DT_INT16, shape, "C");

    FUNCTION(funcName, {input_a, input_b, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = BitwiseLeftShift(input_a, input_b);
        }
    }

    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

} // namespace npu::tile_fwk
