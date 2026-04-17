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
 * \file device_error_tracking.cpp
 * \brief
 */

#include <iostream>
#ifdef BUILD_WITH_CANN
#include "adapter/api/acl_api.h"

namespace npu::tile_fwk {
const char* getExceptionTypeName(RtExceptionExpandType type)
{
    switch (type) {
        case RtExceptionExpandType::INVALID:
            return "exception invalid error";
        case RtExceptionExpandType::FFTS_PLUS:
            return "exception ffts_plus error";
        case RtExceptionExpandType::AICORE:
            return "exception aicore error";
        case RtExceptionExpandType::UB:
            return "exception ub error";
        case RtExceptionExpandType::CCU:
            return "exception ccu error";
        case RtExceptionExpandType::FUSION:
            return "exception fusion error";
        default:
            return "unknown error type";
    }
}

void AicpuErrorCallBack(AclRtExceptionInfo* exceptionInfo)
{
    printf(
        "ErrorTracking callback in, task_id = %u, stream_id = %u.\n", exceptionInfo->taskid, exceptionInfo->streamid);
    const char* typeName = getExceptionTypeName(exceptionInfo->expandInfo.type);
    printf("[ERROR] Exception Type: %s\n", typeName);
    printf(
        "taskid: %u, streamid: %u, tid: %u, deviceid: %u, retcode: %u\n", exceptionInfo->taskid,
        exceptionInfo->streamid, exceptionInfo->tid, exceptionInfo->deviceid, exceptionInfo->retcode);
    printf("kernelName = %s\n", exceptionInfo->expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName);
}

void InitializeErrorCallback()
{
    AclError ret = AclRtSetExceptionInfoCallback(&AicpuErrorCallBack);
    if (ret != ACLRT_SUCCESS) {
        printf("Failed to set exception callback: %d\n", ret);
    }
}
} // namespace npu::tile_fwk
#endif
