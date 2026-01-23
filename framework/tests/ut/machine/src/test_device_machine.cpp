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
 * \file test_device_machine.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/device_machine.h"

using namespace npu::tile_fwk;

class DeviceMachineTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

extern "C" int StaticTileFwkBackendKernelServer(void *targ);
extern "C" int PyptoKernelCtrlServerInit(void *targ);

TEST(DeviceMachineTest, test_get_task_time) {
    DeviceArgs args = {};
    args.nrAicpu = 1;
    std::uint64_t tastWastTime = 0;
    args.taskWastTime = (uint64_t)&tastWastTime;

    std::thread aicpus[1];
    std::atomic<int> idx{0};
    for (int i = 0; i < 1; i++) {
        aicpus[i] = std::thread([&]() {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            sprintf(name, "aicput%d", tidx);
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            StaticTileFwkBackendKernelServer(&args);
        });
    }

    for (int i = 0; i < 1; i++) {
        aicpus[i].join();
    }
}

TEST(DeviceMachineTest, test_ctrl_server) {
    DeviceKernelArgs args;
    args.inputs = nullptr;
    auto ret = PyptoKernelCtrlServerInit((void*)&args);
    EXPECT_EQ(ret, -1);
}