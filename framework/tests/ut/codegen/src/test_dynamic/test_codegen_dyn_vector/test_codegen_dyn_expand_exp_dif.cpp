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
 * \file test_codegen_dyn_expand_exp_dif.cpp
 * \brief Unit test for expand_exp_dif codegen.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynExpandExpDif : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }

    void TearDown() override {}
};

void TestExpandExpDif(const Shape &shape_x, const Shape &shape_y, const std::string &expect) {
    TileShape::Current().SetVecTile(shape_x);
    Tensor input_x(DT_FP32, shape_x, "input_x");
    Tensor input_y(DT_FP32, shape_y, "input_y");
    Tensor output(DT_FP32, shape_x, "output");

    ConfigManager::Instance();
    std::string funcName = "TestExpandExpDif";
    FUNCTION(funcName, {input_x, input_y, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = ExpandExpDif(input_x, input_y);
        }
    }

    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    CheckStringExist(expect, GetResultFromCpp(*function));
}

TEST_F(TestCodegenDynExpandExpDif, TestExpandExpDifLastAxis) {
    const Shape shape_x = {16, 128};
    const Shape shape_y = {16, 1};
    std::string expect =
        R"!!!(TExpandExpDif<TileOp::BroadcastOperand::RIGHT_OPERAND>(ubTensor_15, ubTensor_15, ubTensor_13);)!!!";
    TestExpandExpDif(shape_x, shape_y, expect);
}

TEST_F(TestCodegenDynExpandExpDif, TestExpandExpDifSecondaryLastAxis) {
    const Shape shape_x = {16, 128};
    const Shape shape_y = {1, 128};
    std::string expect = R"!!!(TExpandExpDif(ubTensor_9, ubTensor_9, ubTensor_11);)!!!";
    TestExpandExpDif(shape_x, shape_y, expect);
}
} // namespace npu::tile_fwk
