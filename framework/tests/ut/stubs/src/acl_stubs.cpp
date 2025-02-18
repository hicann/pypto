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
 * \file acl_stubs.cpp
 * \brief
 */

#include <acl/acl_base.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

extern "C" {

aclError aclFinalize()
{
    return 0;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    (void)stream;
    (void)event;
    return 0;
}

aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    (void)event;
    (void)flag;
    return 0;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    (void)event;
    (void)stream;
    return 0;
}

aclError aclrtCreateEvent(aclrtEvent *event)
{
    (void)event;
    return 0;
}

aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag)
{
    (void)event;
    (void)flag;
    return 0;
}

aclError aclInit(const char *configPath)
{
    (void)configPath;
    return 0;
}



aclError aclrtSetDevice(int32_t deviceId)
{
    (void)deviceId;
    return 0;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    (void)devPtr;
    (void)size;
    (void)policy;
    return 0;
}

aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status,
                                aclmdlRI *modelRI)
{
    (void)stream;
    (void)status;
    (void)modelRI;
    return 0;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode)
{
    (void)mode;
    return 0;
}

aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback)
{
    (void)callback;
    return 0;
}
}
