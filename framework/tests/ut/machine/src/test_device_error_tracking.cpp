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
 * \file test_device_error_tracking.cpp
 * \brief UT for filtering the process-global runtime exception callback to PyPTO tasks only.
 */

#include <gtest/gtest.h>

#include "adapter/api/runtime_define.h"
#include "machine/runtime/runner/device_error_tracking.h"

using namespace npu::tile_fwk;

class DeviceErrorTrackingTest : public testing::Test {};

TEST_F(DeviceErrorTrackingTest, NullExceptionIsNotPyPTO) { EXPECT_FALSE(IsPyPTOAicoreException(nullptr)); }

TEST_F(DeviceErrorTrackingTest, NonPyPTOAicoreNameIsIgnored)
{
    RtExceptionInfo exceptionInfo = {};
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin = reinterpret_cast<void*>(0x2000);
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = "OtherKernel";
    EXPECT_FALSE(IsPyPTOAicoreException(&exceptionInfo));
}

TEST_F(DeviceErrorTrackingTest, NullAicoreKernelNameIsNotPyPTO)
{
    RtExceptionInfo exceptionInfo = {};
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    EXPECT_FALSE(IsPyPTOAicoreException(&exceptionInfo));
}

TEST_F(DeviceErrorTrackingTest, PyPTOAicoreNameIsAccepted)
{
    RtExceptionInfo exceptionInfo = {};
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICORE;
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.bin = reinterpret_cast<void*>(0x1000);
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = "PyPTO_test_kernel";
    EXPECT_TRUE(IsPyPTOAicoreException(&exceptionInfo));
}

TEST_F(DeviceErrorTrackingTest, AicpuExceptionIsIgnored)
{
    RtExceptionInfo exceptionInfo = {};
    exceptionInfo.expandInfo.type = RtExceptionExpandType::AICPU;
    exceptionInfo.expandInfo.u.aicpuInfo.functionName = "DynTileFwkKernelServer";
    EXPECT_FALSE(IsPyPTOAicoreException(&exceptionInfo));
}

TEST_F(DeviceErrorTrackingTest, UnsupportedExceptionTypeIsIgnored)
{
    RtExceptionInfo exceptionInfo = {};
    exceptionInfo.expandInfo.type = RtExceptionExpandType::FFTS_PLUS;
    EXPECT_FALSE(IsPyPTOAicoreException(&exceptionInfo));
}
