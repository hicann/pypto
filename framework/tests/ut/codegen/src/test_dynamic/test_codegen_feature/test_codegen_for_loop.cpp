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
 * \file test_codegen_dyn_binary.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenForLoop : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenForLoop, TestForLoop) {
    std::vector<int64_t> shape = {2, 2, 2, 8};
    std::vector<int64_t> tile_shape = {2, 2, 2, 8};

    TileShape::Current().SetVecTile(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "Output");

    std::string name = "TestForLoop";
    FUNCTION(name, {input_a, input_b, output}) {
        LOOP(name, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto res1 = Add(input_a, input_b);
            auto res2 = Sub(input_a, input_b);
            output = Mul(res1, res2);
        }
    }

    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);

    // 定义第一个待检查的目标代码片段
    const std::string expect1 = R"(for (uint16_t idx0 = 0; idx0 < 1; ++idx0) {
  for (uint16_t idx1 = 0; idx1 < sym_84_dim_0; ++idx1) {
    for (uint16_t idx2 = 0; idx2 < sym_84_dim_1; ++idx2) {
        auto tileOffsets = TileOffset(idx0, idx1, idx2);
        ubTensor_11_low2DimInLoop.SetAddr(ubTensor_11.GetLinearAddr(tileOffsets));
        ubTensor_3_low2DimInLoop.SetAddr(ubTensor_3.GetLinearAddr(tileOffsets));
        ubTensor_1_low2DimInLoop.SetAddr(ubTensor_1.GetLinearAddr(tileOffsets));
        ubTensor_5_low2DimInLoop.SetAddr(ubTensor_5.GetLinearAddr(tileOffsets));
        TAdd(ubTensor_5_low2DimInLoop, ubTensor_1_low2DimInLoop, ubTensor_3_low2DimInLoop);
        TSub(ubTensor_11_low2DimInLoop, ubTensor_1_low2DimInLoop, ubTensor_3_low2DimInLoop);
    }
  }
})";
    CheckStringExist(expect1, res);

    const std::string expect2 = R"(for (uint16_t idx0 = 0; idx0 < 1; ++idx0) {
  for (uint16_t idx1 = 0; idx1 < sym_84_dim_0; ++idx1) {
    for (uint16_t idx2 = 0; idx2 < sym_84_dim_1; ++idx2) {
        auto tileOffsets = TileOffset(idx0, idx1, idx2);
        ubTensor_11_low2DimInLoop.SetAddr(ubTensor_11.GetLinearAddr(tileOffsets));
        ubTensor_5_low2DimInLoop.SetAddr(ubTensor_5.GetLinearAddr(tileOffsets));
        TMul(ubTensor_5_low2DimInLoop, ubTensor_5_low2DimInLoop, ubTensor_11_low2DimInLoop);
    }
  }
})";
    CheckStringExist(expect2, res);
}

} // namespace npu::tile_fwk
