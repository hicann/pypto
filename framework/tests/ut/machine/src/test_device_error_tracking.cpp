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
 * \file test_device_error_tracking.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <functional>

#ifdef BUILD_WITH_CANN
#include "acl/acl_rt.h"
#include "runtime/base.h"
std::string CaptureStdout(std::function<void()> func) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return "";
    }

    int old_stdout = dup(STDOUT_FILENO);
    if (old_stdout == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }

    if (dup2(pipefd[1], STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        close(old_stdout);
        return "";
    }

    close(pipefd[1]);
    func();
    fflush(stdout);

    if (dup2(old_stdout, STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(old_stdout);
        return "";
    }

    char buffer[4096] = {0};
    ssize_t len = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);
    close(old_stdout);

    return std::string(buffer, len);
}
#endif

#include "machine/runtime/device_error_tracking.h"

#ifdef BUILD_WITH_CANN
using namespace npu::tile_fwk;

TEST(DeviceErrorTrackingTest, GetExceptionTypeNameCoversAllCases) {
    EXPECT_STREQ(getExceptionTypeName(RT_EXCEPTION_INVALID), "exception invalid error");
    EXPECT_STREQ(getExceptionTypeName(RT_EXCEPTION_FFTS_PLUS), "exception ffts_plus error");
    EXPECT_STREQ(getExceptionTypeName(RT_EXCEPTION_AICORE), "exception aicore error");
    EXPECT_STREQ(getExceptionTypeName(RT_EXCEPTION_UB), "exception ub error");
    EXPECT_STREQ(getExceptionTypeName(RT_EXCEPTION_CCU), "exception ccu error");
    EXPECT_STREQ(getExceptionTypeName(RT_EXCEPTION_FUSION), "exception fusion error");
    EXPECT_STREQ(getExceptionTypeName(static_cast<rtExceptionExpandType_t>(100)), "unknown error type");
}

TEST(DeviceErrorTrackingTest, AicpuErrorCallBackOutputsCorrectInfo) {
    aclrtExceptionInfo exceptionInfo{};
    memset(&exceptionInfo, 0, sizeof(aclrtExceptionInfo));
    exceptionInfo.taskid = 123;
    exceptionInfo.streamid = 456;
    exceptionInfo.tid = 789;
    exceptionInfo.deviceid = 0;
    exceptionInfo.retcode = 1;
    exceptionInfo.expandInfo.type = RT_EXCEPTION_AICORE;

    char kernelName[] = "test_kernel";
    exceptionInfo.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = kernelName;

    std::string output = CaptureStdout([&]() {
        AicpuErrorCallBack(&exceptionInfo);
    });

    EXPECT_NE(output.find("task_id = 123, stream_id = 456"), std::string::npos);
    EXPECT_NE(output.find("Exception Type: exception aicore error"), std::string::npos);
    EXPECT_NE(output.find("taskid: 123, streamid: 456, tid: 789, deviceid: 0, retcode: 1"), std::string::npos);
    EXPECT_NE(output.find("kernelName = test_kernel"), std::string::npos);
}

TEST(DeviceErrorTrackingTest, InitializeErrorCallbackExecutesNormally) {
    std::string output = CaptureStdout([&]() {
        InitializeErrorCallback();
    });
    SUCCEED() << "InitializeErrorCallback executed normally";
}
#endif
