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
 * \file test_pass_dependency.cpp
 * \brief Unit test for PassDependency.
 */

#include <gtest/gtest.h>
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_dependency.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {

class TestPassDependency : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

TEST_F(TestPassDependency, TestCheckStrategyDependency) {
    PassDependency &passDependency = PassDependency::Instance();

    std::vector<std::string> normalPasses = {
        PassNameStr(PassName::GRAPH_PARTITION), PassNameStr(PassName::PRE_GRAPH_PROCESS),
        PassNameStr(PassName::INPLACE_PROCESS), PassNameStr(PassName::INFER_DYN_SHAPE), PassNameStr(PassName::SUBGRAPH_TO_FUNCTION)};
    std::vector<std::string> passesLessDependency = {PassNameStr(PassName::GRAPH_PARTITION),
        PassNameStr(PassName::INFER_DYN_SHAPE), PassNameStr(PassName::SUBGRAPH_TO_FUNCTION)};
    std::vector<std::string> passesConsecutiveDup = {PassNameStr(PassName::GRAPH_PARTITION),
        PassNameStr(PassName::GRAPH_PARTITION), PassNameStr(PassName::PRE_GRAPH_PROCESS),
        PassNameStr(PassName::INPLACE_PROCESS), PassNameStr(PassName::INFER_DYN_SHAPE),
        PassNameStr(PassName::SUBGRAPH_TO_FUNCTION), PassNameStr(PassName::SUBGRAPH_TO_FUNCTION),
        PassNameStr(PassName::SUBGRAPH_TO_FUNCTION), PassNameStr(PassName::SUBGRAPH_TO_FUNCTION)};
    
    EXPECT_EQ(passDependency.CheckStrategyDependency("normalPasses", normalPasses), SUCCESS);
    EXPECT_EQ(passDependency.CheckStrategyDependency("passesLessDependency", passesLessDependency), WARNING);
    EXPECT_EQ(passDependency.CheckStrategyDependency("passesConsecutiveDup", passesConsecutiveDup), WARNING);
}
} // namespace tile_fwk
} // namespace npu