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
 * \file test_pass_manager.cpp
 * \brief Unit test for pass manager.
 */
#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "passes/pass_mgr/pass_manager.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_mgr/pass_registry.h"
#include "ut_json/ut_json_tool.h"
#include "computational_graph_builder.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {
class PassTestCast : public Pass {
public:
    PassTestCast() : Pass("PassTestCast") {}

    Status PreCheck(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status PostCheck(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status RunOnFunction(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status CreateLogFolder(const std::string &topFolder, size_t i) const override {
        (void) topFolder;
        (void) i;
        return FAILED;
    }
    Status PrintFunction(Function& function, const std::string &logFolder, bool beforeFunction) override {
        (void) function;
        (void) logFolder;
        (void) beforeFunction;
        return FAILED;
    }
    Status DumpFunctionJson(Function& function, const std::string &logFolder, bool beforeFunction) override {
        (void) function;
        (void) logFolder;
        (void) beforeFunction;
        return FAILED;
    }
    Status PreRun(Function &function) override {
        (void) function;
        return FAILED;
    }
    Status PostRun(Function &function) override {
        (void) function;
        return FAILED;
    }
};

class PassManagerTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }
    void TearDown() override {}
};

TEST_F(PassManagerTest, TestPassManager) {
    REG_PASS(PassTestCast);
    PassManager::Instance().RegisterStrategy("PM_TEST", {
                        {   "PassTestCast1",   "PassTestCast1"}});
    PassManager::Instance().RegisterStrategy("PM_TEST2", {
                        {   "PassTestCast1",   "PassTestCast1"}});
    auto errPasses = PassManager::Instance().GetStrategyPasses("PM_TEST1");
    EXPECT_TRUE(errPasses.empty());
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPassManager", "TestPassManager", nullptr);
    EXPECT_TRUE(PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "PM_TEST") == SUCCESS);
    EXPECT_TRUE(PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "PM_TEST2") == SUCCESS);
}

TEST_F(PassManagerTest, TestPassBase) {
    PassTestCast passTestCase;
    auto logFolder = passTestCase.LogFolder("output", 0);
    EXPECT_TRUE(logFolder.empty() == false);
    auto currFunctionPtr1 = std::make_shared<Function>(Program::GetInstance(), "TestPassManager1", "TestPassManager1", nullptr);
    PassConfigs configs;
    configs.printGraph = true;
    passTestCase.SetPassConfigs(configs);
    auto res = passTestCase.Run(*currFunctionPtr1, "TestPassManager1", "TestPassManager1");
    EXPECT_TRUE(res == FAILED);
    configs.printGraph = false;
    configs.dumpGraph = true;
    passTestCase.SetPassConfigs(configs);
    res = passTestCase.Run(*currFunctionPtr1, "TestPassManager1", "TestPassManager1");
    EXPECT_TRUE(res == FAILED);
    configs.printGraph = false;
    configs.dumpGraph = false;
    configs.preCheck = true;
    passTestCase.SetPassConfigs(configs);
    res = passTestCase.Run(*currFunctionPtr1, "TestPassManager1", "TestPassManager1");
    EXPECT_TRUE(res == FAILED);
}

TEST_F(PassManagerTest, TestPassStrategy) {
    PassManager::Instance().RegisterStrategy("StrategyTest", {
                        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
                        {      "InferMemoryConflict",      "InferMemoryConflict"},
                        {           "ExpandFunction",           "ExpandFunction"}});
    // user define
    auto strategyPasses = PassManager::Instance().GetStrategyPasses("StrategyTest");
    EXPECT_TRUE(!strategyPasses.empty());
    // default strategy
    auto pvcPasses = PassManager::Instance().GetStrategyPasses("PVC2_OOO");
    EXPECT_TRUE(!pvcPasses.empty());
    // empty strategy
    auto strategyPasses1 = PassManager::Instance().GetStrategyPasses("StrategyTest1");
    EXPECT_TRUE(strategyPasses1.empty());
}

TEST_F(PassManagerTest, TestPassReg) {
    PassManager::Instance().RegisterStrategy("TestPassReg", {
                        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
                        {   "RemoveRedundantReshape",           "ExpandFunction"}});
    // user define
    auto strategyPasses = PassManager::Instance().GetStrategyPasses("TestPassReg");
    EXPECT_TRUE(strategyPasses.size() == 1);
}

void GetGraph(ComputationalGraphBuilder &G) {
    std::vector<int64_t> tileShape{16,16};
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_MULS, Opcode::OP_ADDS, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"COPY_IN", "MULS", "ADDS", "COPY_OUT"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
}

TEST_F(PassManagerTest, TestPassDFX) {
    PassManager::Instance().RegisterStrategy("TestPassDFX", {
                        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"}});
    ComputationalGraphBuilder G;
    GetGraph(G);
    Function *function = G.GetFunction();
    auto rootPath = config::LogTopFolder();
    PassManager::Instance().RunPass(Program::GetInstance(), *function, "TestPassDFX");
    auto afterJsonPath = rootPath + "/Pass_00_RemoveRedundantReshape/After_000_RemoveRedundantReshape_PROGRAM_ENTRY.json";
    auto beforeJsonPath = rootPath + "/Pass_00_RemoveRedundantReshape/Before_000_RemoveRedundantReshape_PROGRAM_ENTRY.json";
    auto beforeIRPath = rootPath + "/Pass_00_RemoveRedundantReshape/Before_000_RemoveRedundantReshape_PROGRAM_ENTRY.tifwkgr";
    auto afterIRPath = rootPath + "/Pass_00_RemoveRedundantReshape/After_000_RemoveRedundantReshape_PROGRAM_ENTRY.tifwkgr";
    EXPECT_FALSE(IsPathExist(afterJsonPath));
    EXPECT_FALSE(IsPathExist(beforeJsonPath));
    EXPECT_FALSE(IsPathExist(beforeJsonPath));
    EXPECT_FALSE(IsPathExist(afterJsonPath));
    config::SetPassConfig("TestPassDFX", "RemoveRedundantReshape", "print_graph", true);
    config::SetPassConfig("TestPassDFX", "RemoveRedundantReshape", "dump_graph", true);
    config::SetPassConfig("TestPassDFX", "RemoveRedundantReshape", "dump_graph", true);
    PassManager::Instance().RunPass(Program::GetInstance(), *function, "TestPassDFX");
    EXPECT_TRUE(IsPathExist(afterJsonPath));
    EXPECT_TRUE(IsPathExist(beforeJsonPath));
    EXPECT_TRUE(IsPathExist(beforeJsonPath));
    EXPECT_TRUE(IsPathExist(afterJsonPath));
    config::SetPassConfig("TestPassDFX", "RemoveRedundantReshape", "DISABLE_PASS", true);
    PassManager::Instance().RunPass(Program::GetInstance(), *function, "TestPassDFX");
}
}
