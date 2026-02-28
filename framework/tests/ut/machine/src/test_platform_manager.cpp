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
 * \file test_platform_manager.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <fstream>
#include <cstdlib>
#include "interface/utils/file_utils.h"
#include "tilefwk/tilefwk_log.h"

#define private public
#include "machine/platform/platform_manager.h"
#undef private

using namespace npu::tile_fwk;

namespace {
const std::string PM_TEST_TMP_DIR = "/tmp/test_platform_manager_log";
}

class TestPlatformManagerLog : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(PM_TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + PM_TEST_TMP_DIR;
        [[maybe_unused]] int ret = system(cmd.c_str());
    }

    void SetUp() override {
        const char *env = std::getenv("ASCEND_HOME_PATH");
        origAscendHomePath_ = env ? std::string(env) : "";
        hasAscendHomePath_ = (env != nullptr);
    }

    void TearDown() override {
        if (hasAscendHomePath_) {
            setenv("ASCEND_HOME_PATH", origAscendHomePath_.c_str(), 1);
        } else {
            unsetenv("ASCEND_HOME_PATH");
        }
    }

private:
    std::string origAscendHomePath_;
    bool hasAscendHomePath_ = false;
};

TEST_F(TestPlatformManagerLog, Initialize_EmptySocVersion) {
    PlatformManager pm;
    bool result = pm.Initialize("");
    EXPECT_FALSE(result);
}

TEST_F(TestPlatformManagerLog, Initialize_NoAscendHomePath) {
    unsetenv("ASCEND_HOME_PATH");

    PlatformManager pm;
    bool result = pm.Initialize("Ascend910B1");
    EXPECT_FALSE(result);
}

