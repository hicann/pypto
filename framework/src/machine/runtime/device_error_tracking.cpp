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
#include "acl/acl_rt.h"
#include "runtime/base.h"

namespace npu::tile_fwk {
const char* getExceptionTypeName(rtExceptionExpandType_t type)
{
    switch (type) {
        case RT_EXCEPTION_INVALID:      return "exception invalid error";
        case RT_EXCEPTION_FFTS_PLUS:    return "exception ffts_plus error";
        case RT_EXCEPTION_AICORE:       return "exception aicore error";
        case RT_EXCEPTION_UB:           return "exception ub error";
        case RT_EXCEPTION_CCU:          return "exception ccu error";
        case RT_EXCEPTION_FUSION:       return "exception fusion error";
        default:                        return "unknown error type";
    }
}

void AicpuErrorCallBack(aclrtExceptionInfo *exceptionInfo)
{
    printf("ErrorTracking callback in, task_id = %u, stream_id = %u.\n", exceptionInfo->taskid, exceptionInfo->streamid);
    const char* typeName = getExceptionTypeName(exceptionInfo->expandInfo.type);
    printf("[ERROR] Exception Type: %s\n", typeName);
    printf("taskid: %u, streamid: %u, tid: %u, deviceid: %u, retcode: %u\n",
        exceptionInfo->taskid,
        exceptionInfo->streamid,
        exceptionInfo->tid,
        exceptionInfo->deviceid,
        exceptionInfo->retcode);
    printf("kernelName = %s\n",
        exceptionInfo->expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName);
}

void InitializeErrorCallback()
{
    aclError ret = aclrtSetExceptionInfoCallback(&AicpuErrorCallBack);
    if (ret != ACL_SUCCESS) {
        printf("Failed to set exception callback: %d\n", ret);
    }
}
}
#endif