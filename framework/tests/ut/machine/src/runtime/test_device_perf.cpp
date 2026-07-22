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
 * \file test_device_perf.cpp
 * \brief UT for machine/runtime/runner/device_perf.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/device_perf.h"
#undef private
#include "interface/configs/config_manager.h"
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;

TEST(DevicePerfTest, AllPerfOps_NoCrash)
{
    {
        DevicePerf perf;
    }

    DevicePerf perf;
    DeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 4;
    perf.args_ = args;
    EXPECT_EQ(perf.GetPerfDataSize(), 2u + 4u + 1u);

    perf.SyncProfData(false);
    perf.ResetMetrics(0);
    perf.ResetMetrics(999);
    perf.ReleasePerfData();
    EXPECT_TRUE(perf.perfData_.empty());

    perf.StopMachinePerfTraceDumpThread();
    perf.StartMachinePerfTraceDumpThread();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}
