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
 * \file test_codegen_dyn_row_sum_line.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynRowSumLine : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynRowSumLine, TestOperationRowSumLine) {
    int shape0 = 6;
    int shape1 = 1;
    int shape2 = 8;
    int shape3 = 1024;
    std::vector<int64_t> shape = {shape0 * shape1, shape2, shape3};
    std::vector<int64_t> outshape = {shape0 * shape1, 1, shape3};
    TileShape::Current().SetVecTile({2, 8, 512});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "C");

    std::string funcName = "Reduce3dimMoe";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Sum(input_a, 1, true);
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
        subFunc.second->InsertDynParam("sym_23_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_23_dim_1", fakeParam);
        subFunc.second->InsertDynParam("sym_23_dim_2", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynRowSumLine, TestOperationRowSumSingleTileTensor) {
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);

    int shape0 = 257;
    int shape1 = 128;
    std::vector<int64_t> shape = {shape0, shape1};
    std::vector<int64_t> outshape = {shape0, 1};

    TileShape::Current().SetVecTile({128, 64});

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outshape, "C");

    std::string funcName = "RowSumSingle_TILETENSOR";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Sum(input_a, -1, true);
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.001f),
    });
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

} // namespace npu::tile_fwk