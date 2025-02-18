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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "ut_json/ut_json_tool.h"

namespace npu::tile_fwk {
class TestExpandFunction : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(TestExpandFunction, ExpandFunctionTest) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
    });

    std::vector<int64_t> shape{64, 64};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    constexpr int TILE_SHAPE = 32;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);

    FUNCTION("A") {
        c = Div(a, b);
    }

    std::string jsonFilePath = "./config/pass/json/expand_function.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    Function* currentFunction = Program::GetInstance().GetCurrentFunction();

    auto opListBefore = currentFunction->Operations().DuplicatedOpList();
    int divNumBefore = 0;
    int divNumAfter = 0;
    for (auto &op : opListBefore) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumBefore++;
        }
    }
    Program testProgram;
    ExpandFunction expandFunction;
    expandFunction.RunOnFunction(*currentFunction);
    auto opListAfter = currentFunction->Operations().DuplicatedOpList();
    for (auto &op : opListAfter) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumAfter++;
        }
    }
    constexpr int TEST_RES1 = 1;
    constexpr int TEST_RES2 = 4;
    EXPECT_EQ(divNumBefore, TEST_RES1);
    EXPECT_EQ(divNumAfter, TEST_RES2);
}
} // namespace npu::tile_fwk
