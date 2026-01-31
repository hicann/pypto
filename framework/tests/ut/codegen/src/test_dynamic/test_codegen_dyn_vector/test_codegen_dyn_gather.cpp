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
 * \file test_codegen_dyn_gather.cpp
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
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynGather : public ::testing::Test {
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

constexpr const int GATHER_SHAPE0 = 16;
constexpr const int GATHER_SHAPE1 = 32;

TEST_F(TestCodegenDynGather, TestGather) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    constexpr const int S2 = 32;
    constexpr const int D = 64;
    constexpr const int B = 1;
    constexpr const int S = 32;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    TileShape::Current().SetVecTile({1, GATHER_SHAPE0, GATHER_SHAPE1});

    Tensor inputSrc0(DT_FP32, shape0, "x");
    Tensor inputSrc1(DT_INT32, shape1, "indices");
    Tensor output(DT_FP32, shape2, "output");

    ConfigManager::Instance();
    std::string funcName = "GATHER_T";
    FUNCTION(funcName, {inputSrc0, inputSrc1, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Gather(inputSrc0, inputSrc1, axis);
        }
    }
#if ENABLE_HIDDENLOOP
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
#endif
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
TEST_F(TestCodegenDynGather, TestGatherLayout) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S2 = 32;
    constexpr const int D = 64;
    constexpr const int B = 1;
    constexpr const int S = 32;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    TileShape::Current().SetVecTile({1, GATHER_SHAPE0, GATHER_SHAPE1});

    Tensor inputSrc0(DT_FP32, shape0, "x");
    Tensor inputSrc1(DT_INT32, shape1, "indices");
    Tensor output(DT_FP32, shape2, "output");

    ConfigManager::Instance();
    std::string funcName = "GATHER_T";
    FUNCTION(funcName, {inputSrc0, inputSrc1, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Gather(inputSrc0, inputSrc1, axis);
        }
    }
#if ENABLE_HIDDENLOOP
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
#endif
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk
