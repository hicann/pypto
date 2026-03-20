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
 * \file test_compiler_monitor.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"

namespace npu::tile_fwk {
class CompilerMonitor : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
    }

    void TearDown() override {}
};

TEST_F(CompilerMonitor, CompilerMonitorInitial) {
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetStageTimeoutFlag("Pass");
    MonitorManager::Instance().GetStageTimeoutFlag("Pass");
    MonitorManager::Instance().GetStageStartTime();
    MonitorManager::Instance().GetStageElapsedTotals();
}

TEST_F(CompilerMonitor, CompilerMonitorImpl) {
    MonitorImpl* impl_ = new MonitorImpl(&(MonitorManager::Instance()));
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(5);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);
    impl_->Start();
    impl_->StartMonitoring();
    sleep(10);
    impl_->StopMonitoring();
    impl_->Stop();

    delete impl_;
}

TEST_F(CompilerMonitor, CompilerMonitorTestPrint) {
    MonitorImpl* impl_ = new MonitorImpl(&(MonitorManager::Instance()));
    MonitorManager::Instance().Initialize(true, 2, 4, 5);
    MonitorManager::Instance().SetTotalFunctionCount(5);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);
    MonitorManager::Instance().StartStage("Pass");
    sleep(1);
    MonitorManager::Instance().EndStage("Pass");
    impl_->Start();
    impl_->StartMonitoring();
    sleep(1);
    MonitorManager::Instance().NotifyCompilationFinished();
    impl_->StopMonitoring();
    impl_->Stop();

    delete impl_;
}

} // namespace npu::tile_fwk
