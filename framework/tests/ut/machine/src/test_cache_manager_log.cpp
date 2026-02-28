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
 * \file test_cache_manager_log.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "tilefwk/pypto_fwk_log.h"

#define private public
#include "machine/cache_manager/cache_manager.h"
#undef private

using namespace npu::tile_fwk;

namespace {
const std::string CM_TEST_TMP_DIR = "/tmp/test_cache_manager_log";
}

class TestCacheManagerLog : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(CM_TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + CM_TEST_TMP_DIR;
        [[maybe_unused]] int ret = system(cmd.c_str());
    }

    void SetUp() override {
        const char *env = std::getenv("HOME");
        origHome_ = env ? std::string(env) : "";
        hasHome_ = (env != nullptr);
    }

    void TearDown() override {
        if (hasHome_) {
            setenv("HOME", origHome_.c_str(), 1);
        } else {
            unsetenv("HOME");
        }
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
    }

private:
    std::string origHome_;
    bool hasHome_ = false;
};

TEST_F(TestCacheManagerLog, Initialize_HomeEnvNotSet) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    unsetenv("HOME");

    CacheManager cm;
    bool result = cm.Initialize();
    EXPECT_FALSE(result);
}

TEST_F(TestCacheManagerLog, Initialize_CreateCacheDirFails) {
    config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
    setenv("HOME", "/proc", 1);

    CacheManager cm;
    bool result = cm.Initialize();
    EXPECT_FALSE(result);
}

TEST_F(TestCacheManagerLog, SaveTaskFile_FilesAlreadyExist) {
    CacheManager cm;
    cm.cacheMode_ = CacheMode::Enable;
    cm.isInit_ = true;
    cm.cacheDirPath_ = CM_TEST_TMP_DIR + "/cache_exist";
    CreateMultiLevelDir(cm.cacheDirPath_);
    std::string opFuncName = "ut_log_test_func";
    OpInfoManager::GetInstance().GetOpFuncName() = opFuncName;

    std::string cacheKey = "ut_log_test_key";
    std::string binFile = cm.cacheDirPath_ + "/ast_op_" + cacheKey + ".o";
    std::string soFile = cm.cacheDirPath_ + "/lib" + opFuncName + "_control.so";
    std::string jsonFile = cm.cacheDirPath_ + "/lib" + opFuncName + "_control.json";

    std::ofstream(binFile).close();
    std::ofstream(soFile).close();
    std::ofstream(jsonFile).close();

    auto machineTask = std::make_shared<MachineTask>(1, reinterpret_cast<Function *>(0x1234));
    machineTask->SetCacheKey(cacheKey);
    auto deviceTask = std::make_unique<DeviceAgentTask>(machineTask);

    cm.SaveTaskFile(deviceTask.get());

    SUCCEED();
}
