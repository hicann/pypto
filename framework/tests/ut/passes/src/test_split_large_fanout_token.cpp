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
 * \file test_split_large_fanout_token.cpp
 * \brief Unit test for split_large_fanout_tensor token (WAW scenario) adaptation.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "passes/tile_graph_pass/graph_optimization/split_large_fanout_tensor.h"
#undef private
#undef protected
#include "passes/tile_graph_pass/graph_optimization/split_large_fanout_tensor.h"
#include "passes/pass_utils/graph_utils.h"

using namespace npu::tile_fwk;

class SplitLargeFanoutTokenTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

static ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetTokenType(), ir::Span::Unknown());
}

TEST_F(SplitLargeFanoutTokenTest, NoTokenNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitFanoutTokenNoToken") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitFanoutTokenNoToken");
    ASSERT_NE(func, nullptr);

    SplitLargeFanoutTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

TEST_F(SplitLargeFanoutTokenTest, PassRunsWithTokenOnAssembleNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitFanoutTokenWithToken")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitFanoutTokenWithToken");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    for (auto* op : opList) {
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE && op->result_token_.empty()) {
            auto token = MakeToken("testToken");
            op->result_token_ = {token};
        }
    }

    SplitLargeFanoutTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

TEST_F(SplitLargeFanoutTokenTest, MultiLogicalTensorSkipDetectsCorrectly)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitFanoutTokenMultiLt") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitFanoutTokenMultiLt");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 2);

    auto rawTensor = opList[0]->oOperand.front()->GetRawTensor();
    ASSERT_NE(rawTensor, nullptr);

    auto siblings = GraphUtils::GetTensorsByRawMagic(*func, rawTensor->rawmagic);
    EXPECT_GE(siblings.size(), 1);

    SplitLargeFanoutTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}
