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
#include <string>
#include "interface/utils/file_utils.h"
#include "tilefwk/pypto_fwk_log.h"

#define private public
#include "machine/platform/platform_manager.h"
#undef private

using namespace npu::tile_fwk;

namespace {
const std::string PM_TEST_TMP_DIR = "/tmp/test_platform_manager_log";

std::string PlatformConfigDir()
{
    return PM_TEST_TMP_DIR + "/data/platform_config";
}

std::string PlatformIniPath(const std::string &socVersion)
{
    return PlatformConfigDir() + "/" + socVersion + ".ini";
}

void WriteMinimalPlatformIni(const std::string &path)
{
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(ofs.is_open());
    ofs << "[version]\n";
    ofs << "SoC_version=UtMinimalSoc\n";
    ofs << "Short_SoC_version=UtMin\n";
    ofs << "AIC_version=AIC-UT\n";
    ofs << "\n[SoCInfo]\n";
    ofs << "ai_core_cnt=7\n";
    ofs.close();
}
} // namespace

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

TEST_F(TestPlatformManagerLog, Initialize_MissingPlatformFile_LogsAndReturnsFalse) {
    ASSERT_TRUE(CreateMultiLevelDir(PlatformConfigDir()));
    setenv("ASCEND_HOME_PATH", PM_TEST_TMP_DIR.c_str(), 1);

    PlatformManager pm;
    EXPECT_FALSE(pm.Initialize("NonexistentSocName_xyz"));
}

TEST_F(TestPlatformManagerLog, Initialize_UnreadablePlatformFile_LogsAndReturnsFalse) {
    ASSERT_TRUE(CreateMultiLevelDir(PlatformConfigDir()));
    const std::string iniPath = PlatformIniPath("UtUnreadableSoc");
    WriteMinimalPlatformIni(iniPath);
    if (::chmod(iniPath.c_str(), 0) != 0) {
        GTEST_SKIP() << "chmod(0) not supported in this environment";
    }

    setenv("ASCEND_HOME_PATH", PM_TEST_TMP_DIR.c_str(), 1);
    PlatformManager pm;
    EXPECT_FALSE(pm.Initialize("UtUnreadableSoc"));

    (void)::chmod(iniPath.c_str(), S_IRUSR | S_IWUSR);
}