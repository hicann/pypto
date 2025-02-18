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
 * \file test_runtime.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include "machine/runtime/runtime.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include <iostream>
#include "interface/configs/config_manager.h"
using namespace npu::tile_fwk;

class RuntimeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        npu::tile_fwk::Program::GetInstance().Reset();
        npu::tile_fwk::config::SetPlatformConfig(npu::tile_fwk::KEY_ONLY_HOST_COMPILE, true);
    }

    void TearDown() override {}
};

TEST(RuntimeTest, Runtime01) {
    std::cout << "start to test runtime" << std::endl;
    machine::GetRA()->MapAiCoreReg();
}

