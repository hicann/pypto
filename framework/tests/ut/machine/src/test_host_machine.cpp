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
 * \file test_host_machine.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/pypto_fwk_log.h"

#define private public
#include "interface/machine/host/host_machine.h"
#undef private

using namespace npu::tile_fwk;

extern "C" {
struct Backend {
    void *runPass;
    void *getResumePath;
    void *execute;
    void *simuExecute;
    void *platform;
    void *matchCache;

    static Backend &GetBackend();
};
}

class TestHostMachineLog : public testing::Test {
public:
    void SetUp() override {
        auto &hm = HostMachine::GetInstance();
        if (!hm.initialized_.load()) {
            hm.Init(HostMachineMode::API);
        }
        hm.curTask = nullptr;
    }

    void TearDown() override {
        auto &hm = HostMachine::GetInstance();
        if (hm.curTask != nullptr) {
            delete hm.curTask;
            hm.curTask = nullptr;
        }
        hm.curTaskId_ = 0;
    }
};

TEST_F(TestHostMachineLog, GetPlatformInfo_BackendNotFound) {
    auto &backend = Backend::GetBackend();
    void *savedPlatform = backend.platform;

    backend.platform = nullptr;

    auto &hm = HostMachine::GetInstance();
    std::string result = hm.GetPlatformInfo();
    EXPECT_TRUE(result.empty());

    backend.platform = savedPlatform;
}

TEST_F(TestHostMachineLog, SubTask_CurTaskAlreadyRunning) {
    auto &hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;

    hm.SubTask(nullptr);
    EXPECT_NE(hm.curTask, nullptr);
    MachineTask *firstTask = hm.curTask;

    hm.SubTask(nullptr);
    EXPECT_NE(hm.curTask, nullptr);
    EXPECT_NE(hm.curTask, firstTask);

    delete firstTask;
}

TEST_F(TestHostMachineLog, Compile_NullTaskWhenCurTaskNull) {
    auto &hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;
    hm.curTask = nullptr;

    MachineTask *result = hm.Compile(nullptr);
    EXPECT_EQ(result, nullptr);
}