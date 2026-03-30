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
 * \file test_codegen_dyn_una.cpp
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
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenDynUna : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    void TearDown() override { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynUna, TestAbsDynamic) {
    int S0 = 8;
    int S1 = 4608;
    int D0 = 8;
    int D1 = 4608;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({8, 128});
    Tensor input_a(DataType::DT_FP16, srcShape, "A");
    Tensor output(DataType::DT_FP16, dstShape, "C");

    std::string funcName = "TestAbsDynamic";
    FUNCTION(funcName, {input_a, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Abs(input_a);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    ctx.isMainBlock = true;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynUna, TestDynExpand) {
    std::vector<int64_t> shape = {64, 64};
    std::vector<int64_t> shape1 = {1, 64};
    auto function = GenMockFuncDyn("TestDynExpand");
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape1 = {1, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape1, dynValidShape1});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_EXPAND, {localTensor}, {localOutTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPUCtx opCtx(symbolManager, *function, *function->rootFunc_->programs_[0], op, {});
    CodeGenOpCloudNPU cop(opCtx);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynTexpand_<float, /*DS*/ 1, 64, 64, /*SS*/ 1, 1, 64, 2>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, 1, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynUna, TestPadDynamic) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int S0 = 6;
    int S1 = 12;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP32, srcShape, "input");
    Tensor output(DataType::DT_FP32, dstShape, "output");

    std::string funcName = "TestPadDynamic";
    FUNCTION(funcName, {input, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Pad(input, {0, 6, 0, 8}, "constant", 2.0);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    const std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TPad<pto::PadValueCustom((float)2.0)>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestPadDynamicFP16) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int S0 = 6;
    int S1 = 12;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP16, srcShape, "input");
    Tensor output(DataType::DT_FP16, dstShape, "output");

    std::string funcName = "TestPadDynamicFP16";
    FUNCTION(funcName, {input, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Pad(input, {0, 6, 0, 8}, "constant", std::numeric_limits<float>::infinity());
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    const std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TPad<pto::PadValue::Max>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestFillPadDynamicBF16) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int S0 = 12;
    int S1 = 20;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_BF16, srcShape, "input");
    Tensor output(DataType::DT_BF16, dstShape, "output");

    std::string funcName = "TestFillPadDynamicBF16";
    FUNCTION(funcName, {input, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = FillPad(input, "constant", 0.0f);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    const std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TFillPad<pto::PadValueCustom((bfloat16_t)0.0)>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestFillPadDynamic) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    int S0 = 12;
    int S1 = 20;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP32, srcShape, "input");
    Tensor output(DataType::DT_FP32, dstShape, "output");

    std::string funcName = "TestFillPadDynamic";
    FUNCTION(funcName, {input, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = FillPad(input, "constant", 0.0f);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    const std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(TFillPad<pto::PadValueCustom((float)0.0)>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk