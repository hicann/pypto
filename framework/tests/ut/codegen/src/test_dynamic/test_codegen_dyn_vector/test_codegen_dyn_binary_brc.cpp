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
 * \file test_codegen_dyn_binary_brc.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynBinaryBrc : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

// mul (32, 512), (32, 1)
TEST_F(TestCodegenDynBinaryBrc, TestMulDynamic) {
    config::SetOperationConfig(KEY_FORCE_COMBINE_AXIS, true);
    std::vector<int64_t> shape1 = {32, 512};
    std::vector<int64_t> shape2 = {32, 1};
    TileShape::Current().SetVecTile({32, 256});
    Tensor input_a(DataType::DT_FP32, shape1, "A");
    Tensor input_b(DataType::DT_FP32, shape1, "B");
    Tensor output(DataType::DT_FP32, shape1, "C");
    ConfigManager::Instance();

    std::string funcName = "MUL_T";
    FUNCTION(funcName, {input_a, input_b, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            // add RowSumSingle to test brc case
            auto input_c = Sum(input_b, -1, true);
            output = Mul(input_a, input_c);
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
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar(), false, ""};
        subFunc.second->dynParamTable_.emplace("sym_18_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_18_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_19_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_19_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_32_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_32_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_42_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_42_dim_1", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk