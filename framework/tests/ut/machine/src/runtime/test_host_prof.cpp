/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_host_prof.cpp
 * \brief UT for machine/runtime/runner/host_prof.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"
#undef private
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(HostProfTest, AllProfOps_NoCrash)
{
    EXPECT_EQ(HostProf::GetProfSwitch(), 0ULL);
    EXPECT_EQ(HostProf::GetProfType(), 0u);

    HostProf::RegHostProf();

    HostProf prof;
    prof.SetProfFunction(nullptr);
    prof.HostProfReportApi(100, 200);
    prof.HostProfReportNodeInfo(200, 24, 1);
    prof.HostProfReportContextInfo(200);
    prof.HostProfReportCacheTaskInfo(nullptr, 1, 1);
}

TEST(HostProfTest, HostProfInit_NullData) { EXPECT_EQ(HostProf::HostProfInit(0, nullptr, 0), -1); }

TEST(HostProfTest, HostProfInit_ZeroLen)
{
    int dummy = 0;
    EXPECT_EQ(HostProf::HostProfInit(0, &dummy, 0), -1);
}

TEST(HostProfTest, HostProfInit_InvalidType)
{
    MspfCommandHandle handle{};
    EXPECT_EQ(HostProf::HostProfInit(999, &handle, sizeof(handle)), -1);
}

TEST(HostProfTest, HostProfInit_TooSmallLen)
{
    MspfCommandHandle handle{};
    EXPECT_EQ(HostProf::HostProfInit(static_cast<uint32_t>(RtProfCtrlType::SWITCH), &handle, 1), -1);
}

TEST(HostProfTest, HostProfInit_ValidInput)
{
    MspfCommandHandle handle{};
    handle.profSwitch = 0xABCD;
    handle.type = 42;
    EXPECT_EQ(HostProf::HostProfInit(static_cast<uint32_t>(RtProfCtrlType::SWITCH), &handle, sizeof(handle)), 0);
    EXPECT_EQ(HostProf::GetProfSwitch(), 0xABCDULL);
    EXPECT_EQ(HostProf::GetProfType(), 42u);
    HostProf::profSwitch_ = 0;
    HostProf::profType_ = 0;
}

TEST(HostProfTest, BuildTensor_NullTensorInfo)
{
    HostProf prof;
    MspfTensorData tensorData{};
    prof.BuildTensor(MSPF_GE_TENSOR_TYPE_INPUT, static_cast<RawTensorDataPtr>(nullptr), tensorData);
    EXPECT_EQ(tensorData.tensorType, MSPF_GE_TENSOR_TYPE_INPUT);
    EXPECT_EQ(tensorData.format, 2u);
    EXPECT_EQ(tensorData.dataType, 0u);
    EXPECT_EQ(tensorData.shape[0], 0u);
}

TEST(HostProfTest, BuildCacheTensorInfo_NullTaskInfo)
{
    HostProf prof;
    prof.BuildCacheTensorInfo(nullptr);
}

TEST(HostProfTest, HostProfReportCacheTaskInfo_NullStream)
{
    HostProf prof;
    prof.opName_ = "test_op";
    prof.HostProfReportCacheTaskInfo(nullptr, 1, 1);
}

TEST(HostProfTest, IsCacheOpInfoEnable_NullStream)
{
    HostProf prof;
    EXPECT_FALSE(prof.IsCacheOpInfoEnable(nullptr));
}
