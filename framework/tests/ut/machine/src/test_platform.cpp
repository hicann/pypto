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
 * \file test_platform.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/utils/file_utils.h"
#include "machine/platform/platform_manager.h"

using namespace npu::tile_fwk;

class PlatformTest : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "PlatformTest SetUpTestCase" << std::endl;
        std::cout << "current path = " << RealPath("./") << std::endl;
        ASSERT_FALSE(RealPath("./stubs").empty());
        setenv("ASCEND_HOME_PATH", "./stubs", 1);
    }

    static void TearDownTestCase() {
        std::cout << "PlatformTest TearDownTestCase" << std::endl;
        unsetenv("ASCEND_AICPU_PATH");
    }

    void SetUp() override {
        std::cout << "PlatformTest SetUp" << std::endl;
    }

    void TearDown() override {
        std::cout << "PlatformTest TearDown" << std::endl;
    }
};

TEST_F(PlatformTest, TestPlatfromCase1) {
    EXPECT_EQ(PlatformManager::Instance().Initialize("Ascend910F1"), false);
    EXPECT_EQ(PlatformManager::Instance().Initialize("Ascend910B1"), true);
    EXPECT_EQ(PlatformManager::Instance().Initialize("Ascend910B1"), true);

    EXPECT_EQ(PlatformManager::Instance().GetSocVersion(), "Ascend910B1");
    EXPECT_EQ(PlatformManager::Instance().GetShortSocVersion(), "Ascend910B");
    EXPECT_EQ(PlatformManager::Instance().GetAicVersion(), "AIC-C-220");

    EXPECT_EQ(PlatformManager::Instance().GetAiCoreCnt(), 24);
    EXPECT_EQ(PlatformManager::Instance().GetVecCoreCnt(), 48);
    EXPECT_EQ(PlatformManager::Instance().GetMemorySize(), 68719476736);
    EXPECT_EQ(PlatformManager::Instance().GetL2Type(), 0);
    EXPECT_EQ(PlatformManager::Instance().GetL2Size(), 201326592);
    EXPECT_EQ(PlatformManager::Instance().GetL2PageNum(), 64);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreCubeFreq(), 1850);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL0ASize(), 65536);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL0BSize(), 65536);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL0CSize(), 131072);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL1Size(), 524288);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreUbSize(), 196608);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreUbBlockSize(), 32);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreUbBankSize(), 4096);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreUbBankNum(), 64);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreUbBankGroupNum(), 16);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreDdrRate(), 32);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreDdrReadRate(), 32);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreDdrWriteRate(), 32);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL2Rate(), 110);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL2ReadRate(), 110);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreL2WriteRate(), 86);

    for (const auto &pair : PlatformManager::Instance().GetAiCoreIntrinsicDtypeMap()) {
        std::cout << pair.first << " : ";
        for (const std::string &item : pair.second) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }

    for (const auto &pair : PlatformManager::Instance().GetVectorCoreIntrinsicDtypeMap()) {
        std::cout << pair.first << " : ";
        for (const std::string &item : pair.second) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::string> aicoreDtypeVec;
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreIntrinsicDtype("", aicoreDtypeVec), false);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreIntrinsicDtype("vhwconv", aicoreDtypeVec), false);
    EXPECT_EQ(PlatformManager::Instance().GetAiCoreIntrinsicDtype("vnchwconv", aicoreDtypeVec), true);
    std::vector<std::string> vectorcoreDtypeVec;
    EXPECT_EQ(PlatformManager::Instance().GetVectorCoreIntrinsicDtype("", vectorcoreDtypeVec), false);
    EXPECT_EQ(PlatformManager::Instance().GetVectorCoreIntrinsicDtype("vvad", vectorcoreDtypeVec), false);
    EXPECT_EQ(PlatformManager::Instance().GetVectorCoreIntrinsicDtype("mmad", vectorcoreDtypeVec), true);

    EXPECT_EQ(aicoreDtypeVec.size(), 8);

    PlatformManager::Instance().Finalize();
}