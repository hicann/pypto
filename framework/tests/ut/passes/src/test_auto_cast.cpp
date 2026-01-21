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
 * \file test_auto_cast.cpp
 * \brief Unit test for Auto Cast.
 */

#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tensor_graph_pass/auto_cast.h"

namespace npu {
namespace tile_fwk {

class AutoCastTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
    }
    void TearDown() override {}
};

TEST_F(AutoCastTest, AddBF16) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t3"}), true);
    Function *function = G.GetFunction();
    EXPECT_EQ(function->Operations().size(), 1);
    AutoCast autoCast;
    autoCast.RunOnFunction(*function);
    const int opNum3 = 3;
    EXPECT_EQ(function->Operations().size(), opNum3);
}

TEST_F(AutoCastTest, AddCascadeBF16) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t3"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_BF16, {16, 16}, "t4"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ADD1", "ADD2"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    const int opNum2 = 2;
    EXPECT_EQ(function->Operations().size(), opNum2);
    AutoCast autoCast;
    autoCast.RunOnFunction(*function);
    const int opNum5 = 5;
    EXPECT_EQ(function->Operations().size(), opNum5);
}
} // namespace tile_fwk
} // namespace npu