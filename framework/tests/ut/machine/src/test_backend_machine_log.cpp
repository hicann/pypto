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
 * \file test_backend_machine_log.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>

#include "machine/host/backend.h"
#include "interface/utils/file_utils.h"
#include "interface/configs/config_manager.h"
#include "interface/machine/host/machine_task.h"
#include "interface/cache/function_cache.h"
#include "machine/host/device_agent_task.h"
#include "machine/cache_manager/cache_manager.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "interface/program/program.h"
#include "tilefwk/pypto_fwk_log.h"

using namespace npu::tile_fwk;

extern "C" std::string GetPlatformFile(const std::string &socVersion);
extern "C" std::string GetPlatformInfo();
extern "C" int32_t Execute(MachineTask *task, FunctionCache &cache);

namespace {
const std::string TEST_TMP_DIR = "/tmp/test_backend_ml";
}

class TestBackendMachineLog : public testing::Test {
public:
    static void SetUpTestCase() {
        CreateMultiLevelDir(TEST_TMP_DIR);
    }

    static void TearDownTestCase() {
        std::string cmd = "rm -rf " + TEST_TMP_DIR;
        [[maybe_unused]]int ret = system(cmd.c_str());
    }

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestBackendMachineLog, GetPlatformFile_NoAscendHomePath) {
    std::string savedEnv;
    const char *env = std::getenv("ASCEND_HOME_PATH");
    if (env != nullptr) {
        savedEnv = env;
    }
    unsetenv("ASCEND_HOME_PATH");

    std::string result = GetPlatformFile("Ascend910B1");
    EXPECT_EQ(result, "");

    if (!savedEnv.empty()) {
        setenv("ASCEND_HOME_PATH", savedEnv.c_str(), 1);
    }
}

TEST_F(TestBackendMachineLog, GetPlatformFile_EmptySocVersion) {
    std::string result = GetPlatformFile("");
    EXPECT_EQ(result, "");
}

TEST_F(TestBackendMachineLog, GetPlatformFile_NonExistentPlatformFile) {
    std::string savedEnv;
    const char *env = std::getenv("ASCEND_HOME_PATH");
    if (env != nullptr) {
        savedEnv = env;
    }
    setenv("ASCEND_HOME_PATH", TEST_TMP_DIR.c_str(), 1);

    std::string result = GetPlatformFile("NonExistentSocVersion");
    EXPECT_EQ(result, "");

    if (!savedEnv.empty()) {
        setenv("ASCEND_HOME_PATH", savedEnv.c_str(), 1);
    } else {
        unsetenv("ASCEND_HOME_PATH");
    }
}

TEST_F(TestBackendMachineLog, Execute_CacheRecoverFails) {
    Program::GetInstance().Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "exec_t0");
    Tensor out(DT_FP32, {s, s}, "exec_out");
    FUNCTION("test_execute_cache", {t0}, {out}) {
        auto temp = Add(t0, t0);
        Assemble(temp, {0, 0}, out);
    }

    Function *func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    MachineTask task(1, func);
    task.SetCacheReuseType(CacheReuseType::Bin);
    task.SetCacheKey("nonexistent_cache_key_12345");

    FunctionCache cache;
    int32_t ret = Execute(&task, cache);
    EXPECT_EQ(ret, 0);
}