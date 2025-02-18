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
 * \file test_device_runner.cpp
 * \brief
 */

#include <regex>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "machine/runtime/device_runner.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/platform/platform_manager.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/aicore_manager.h"
#include "machine/device/tilefwk/aicpu_common.h"
class TestDeviceRunner : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestDeviceRunner, test_device_runner_get_task_time) {
    // auto runner = npu::tile_fwk::DeviceRunner::Get();
    npu::tile_fwk::DeviceRunner runner;
    std::uint64_t tastWastTime = 0;
    runner.args_.taskWastTime = (uint64_t)&tastWastTime;
    runner.GetTasksTime();
}

TEST_F(TestDeviceRunner, test_set_pmu_event) {
    // auto runner = npu::tile_fwk::DeviceRunner::Get();
    for (int i = 0; i < 9; i++) {
        setenv("PROF_PMU_EVENT_TYPE", std::to_string(i).c_str(), 1);
        npu::tile_fwk::DeviceRunner runner;
        runner.GetPmuEventType();
    }
}

TEST_F(TestDeviceRunner, test_ini_device_runner) {
    npu::tile_fwk::DeviceRunner runner;
    runner.Init();
}

TEST_F(TestDeviceRunner, test_ini_device_args_arch32) {
    DeviceArgs args_;
    args_.archInfo = ArchInfo::DAV_2201;
    npu::tile_fwk::DeviceRunner runner;
    runner.InitDeviceArgs(args_);
}

TEST_F(TestDeviceRunner, test_ini_device_args_arch35) {
    DeviceArgs args_;
    args_.archInfo = ArchInfo::DAV_3510;
    npu::tile_fwk::DeviceRunner runner;
    runner.InitDeviceArgs(args_);
}

TEST_F(TestDeviceRunner, test_ini_proflevel) {
    std::unique_ptr<npu::tile_fwk::dynamic::AicpuTaskManager> aicpuTaskPtr = std::make_unique<npu::tile_fwk::dynamic::AicpuTaskManager>();
    std::unique_ptr<npu::tile_fwk::dynamic::AiCoreManager> AiCoreManagerPtr = std::make_unique<npu::tile_fwk::dynamic::AiCoreManager>(*aicpuTaskPtr);
    AiCoreManagerPtr->aicNum_ = 0;
    AiCoreManagerPtr->aivNum_ = 1;
    AiCoreManagerPtr->aivStart_ = 0;
    AiCoreManagerPtr->aivEnd_ = 1;
    AiCoreManagerPtr->aicStart_ = 0;
    AiCoreManagerPtr->aicEnd_ = 0;
    AiCoreManagerPtr->aicpuIdx_ = 0;
    npu::tile_fwk::dynamic::AiCoreProf prof(*AiCoreManagerPtr);
    int64_t *oriRegAddrs_ = (int64_t *)malloc(sizeof(int64_t) * 1024 * 2);
    int64_t *regAddrs_ = oriRegAddrs_ + 1024;
    regAddrs_[0] = (int64_t)&regAddrs_[0];
    ToSubMachineConfig toSubMachineConfig;
    AdprofReportAdditionalInfo(0,0,0);
    toSubMachineConfig.profConfig.Add(ProfConfig::OFF);
    prof.ProfInit(regAddrs_, regAddrs_, toSubMachineConfig.profConfig);
    toSubMachineConfig.profConfig.Remove(ProfConfig::OFF);
    toSubMachineConfig.profConfig.Add(ProfConfig::AICPU_FUNC);
    prof.ProfInit(regAddrs_, regAddrs_, toSubMachineConfig.profConfig);
    toSubMachineConfig.profConfig.Remove(ProfConfig::AICPU_FUNC);
    toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_TIME);
    prof.ProfInit(regAddrs_, regAddrs_, toSubMachineConfig.profConfig);
    toSubMachineConfig.profConfig.Remove(ProfConfig::AICORE_TIME);
    toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_PMU);
    prof.ProfInit(regAddrs_, regAddrs_, toSubMachineConfig.profConfig);
    prof.ProfStart();
    int32_t aicoreId = 0;
    int32_t subgraphId = 0;
    int32_t taskId = 0;
    TaskStat *taskStat = new TaskStat{1,0,0,0,1,1};
    prof.ProInitHandShake();
    prof.ProInitAiCpuTaskStat();
    int threadIdx = 0;
    npu::tile_fwk::dynamic::AiCpuTaskStat *aiCpuStat = new npu::tile_fwk::dynamic::AiCpuTaskStat{0,0,0,0,1};
    npu::tile_fwk::dynamic::AiCpuHandShakeSta handShakeSta;
    prof.ProfGet(aicoreId, subgraphId, taskId, taskStat);
    prof.ProfGetAiCpuTaskStat(threadIdx, aiCpuStat);
    prof.ProGetHandShake(threadIdx, &handShakeSta);
    prof.ProfStopHandShake();
    prof.ProfStopAiCpuTaskStat();
    prof.ProfStop();
    prof.GetAiCpuTaskStat(taskId);
    delete aiCpuStat;
    delete taskStat;
    free(oriRegAddrs_);
}

TEST_F(TestDeviceRunner, test_create_proflevel) {
    std::unique_ptr<npu::tile_fwk::dynamic::AicpuTaskManager> aicpuTaskPtr = std::make_unique<npu::tile_fwk::dynamic::AicpuTaskManager>();
    std::unique_ptr<npu::tile_fwk::dynamic::AiCoreManager> AiCoreManagerPtr = std::make_unique<npu::tile_fwk::dynamic::AiCoreManager>(*aicpuTaskPtr);
    npu::tile_fwk::dynamic::AiCoreProf prof(*AiCoreManagerPtr);

    ProfConfig config0;
    EXPECT_TRUE(config0.Empty());
    EXPECT_EQ(prof.AiCoreProf::CreateProfLevel(config0), npu::tile_fwk::dynamic::PROF_LEVEL_OFF);
    ProfConfig config1;
    config1.Add(ProfConfig::AICORE_PMU);
    EXPECT_EQ(prof.AiCoreProf::CreateProfLevel(config1), npu::tile_fwk::dynamic::PROF_LEVEL_FUNC_LOG_PMU);
    EXPECT_FALSE(config1.Empty());
    ProfConfig config2;
    config2.Add(ProfConfig::AICORE_TIME);
    EXPECT_EQ(prof.AiCoreProf::CreateProfLevel(config2), npu::tile_fwk::dynamic::PROF_LEVEL_FUNC_LOG);
    ProfConfig config3;
    config3.Add(ProfConfig::AICPU_FUNC);
    EXPECT_EQ(prof.AiCoreProf::CreateProfLevel(config3), npu::tile_fwk::dynamic::PROF_LEVEL_FUNC);
    EXPECT_TRUE(config3.Contains(ProfConfig::AICPU_FUNC));

    ProfConfig config4 = config2 | config3;
    EXPECT_EQ(config4.value, 0x3);
    ProfConfig config5 = ProfConfig::AICPU_FUNC | ProfConfig::AICORE_TIME;
    EXPECT_TRUE(config5.Overlaps(ProfConfig::AICPU_FUNC));
    EXPECT_TRUE(config5.Overlaps(ProfConfig::AICORE_TIME));
    EXPECT_FALSE(config5.Overlaps(ProfConfig::AICORE_PMU));

    ProfConfig config6 = ProfConfig::AICPU_FUNC | ProfConfig::AICORE_TIME;
    config6.Remove(ProfConfig::AICPU_FUNC);
    EXPECT_EQ(config6.value, ProfConfig::AICORE_TIME);
    
    config6.Remove(ProfConfig::AICORE_TIME);
    EXPECT_EQ(config6.value, 0);

    ToSubMachineConfig config7;
    EXPECT_EQ(config7.profConfig.value, ProfConfig::OFF);
}