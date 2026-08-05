/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <vector>

#include "machine/runtime/runner/dump_device_perf.h"
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DumpDevicePerfTest : public testing::Test {};

TEST_F(DumpDevicePerfTest, DumpDevTaskPerfData_EmptyPerfData)
{
    DeviceArgs args{};
    std::vector<void*> perfData;
    DumpDevTaskPerfData(args, perfData, false);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpDevTaskPerfData_EnvNotSet)
{
    DeviceArgs args{};
    std::vector<void*> perfData(1, nullptr);
    DumpDevTaskPerfData(args, perfData, false);
    SUCCEED();
}
