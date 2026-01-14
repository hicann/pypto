/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_stubs.cpp
 * \brief
 */

#include "hcom.h"

extern "C" {
HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle)
{
    (void)group;
    (void)commHandle;
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *Mc2Tiling, void **commContext)
{
    (void)comm;
    (void)stream;
    (void)Mc2Tiling;
    (void)commContext;
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    (void)comm;
    (void)commName;
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms)
{
    (void)ndev;
    (void)devices;
    (void)comms;
    return HCCL_SUCCESS;
}

}
