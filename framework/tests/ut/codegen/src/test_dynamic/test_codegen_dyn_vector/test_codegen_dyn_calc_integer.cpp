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
 * \file test_codegen_dyn_calc_integer.cpp
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

class TestCodegenDynCalcInteger : public ::testing::Test {
public:
    static void SetUpTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynCalcInteger, TestDynOpCeil) {
    MockFuncDynUnaryConf config;
    auto function =
        GenMockFuncDynUnary("TestDynOpCeil", config, [](Tensor &input, Tensor &output) { output = Ceil(input); });

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect =
        R"!!!(TCeil(ubTensor_1, ubTensor_1);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynCalcInteger, TestDynOpFloor) {
    MockFuncDynUnaryConf config;
    auto function =
        GenMockFuncDynUnary("TestDynOpFloor", config, [](Tensor &input, Tensor &output) { output = Floor(input); });

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect =
        R"!!!(TFloor(ubTensor_1, ubTensor_1);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynCalcInteger, TestDynOpTrunc) {
    MockFuncDynUnaryConf config;
    auto function =
        GenMockFuncDynUnary("TestDynOpTrunc", config, [](Tensor &input, Tensor &output) { output = Trunc(input); });

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    std::string res = GetResultFromCpp(*function);
    std::string expect =
        R"!!!(TTrunc(ubTensor_1, ubTensor_1);
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
