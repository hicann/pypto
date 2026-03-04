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
* \file test_cann_host_runtime.cpp	 
* \brief	 
*/	 

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>

#include "tilefwk/cann_host_runtime.h" 
#include "interface/utils/file_utils.h"

using namespace npu::tile_fwk;
namespace {	 
const std::string TEST_TMP_DIR = "/tmp/test_cann_host_runtime";	 
}

class TestCannHostRuntime : public testing::Test {	 
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

TEST_F(TestCannHostRuntime, GetPlatformFile_NoAscendHomePath) { 
    std::string savedEnv; 
    const char *env = std::getenv("ASCEND_HOME_PATH"); 
    if (env != nullptr) { 
        savedEnv = env; 
    } 
    unsetenv("ASCEND_HOME_PATH"); 
    
    std::string result = CannHostRuntime::Instance().GetPlatformFile("Ascend910B1"); 
    EXPECT_EQ(result, ""); 

    if (!savedEnv.empty()) { 
        setenv("ASCEND_HOME_PATH", savedEnv.c_str(), 1); 
    } 
} 

TEST_F(TestCannHostRuntime, GetPlatformFile_EmptySocVersion) { 
    std::string result = CannHostRuntime::Instance().GetPlatformFile(""); 
    EXPECT_EQ(result, ""); 
} 

TEST_F(TestCannHostRuntime, GetPlatformFile_NonExistentPlatformFile) { 
    std::string savedEnv; 
    const char *env = std::getenv("ASCEND_HOME_PATH"); 
    if (env != nullptr) { 
        savedEnv = env; 
    } 
    setenv("ASCEND_HOME_PATH", TEST_TMP_DIR.c_str(), 1); 
    
    std::string result = CannHostRuntime::Instance().GetPlatformFile("NonExistentSocVersion"); 
    EXPECT_EQ(result, "");

    if (!savedEnv.empty()) { 
        setenv("ASCEND_HOME_PATH", savedEnv.c_str(), 1); 
    } else { 
        unsetenv("ASCEND_HOME_PATH"); 
    } 
}
