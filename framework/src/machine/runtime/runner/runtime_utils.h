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
 * \file runtime_utils.h
 * \brief
 */

#pragma once

#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"
#include "adapter/api/runtime_api.h"
#include "adapter/api/acl_api.h"
#include "machine/runtime/context/stream_context.h"

namespace npu::tile_fwk {
inline void CheckDeviceId()
{
    int32_t devId = 0;
    int32_t getDeviceResult = RuntimeGetDevice(&devId);
    if (getDeviceResult != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_DEVICE_FAILED, "fail get device id, check if set device id");
        return;
    }
    MACHINE_LOGI("Current device is %d.", devId);
}

inline int32_t GetUserDeviceId()
{
    int32_t userDeviceId = 0;
    RuntimeGetDevice(&userDeviceId);
    return userDeviceId;
}

inline int32_t GetLogDeviceId()
{
    int32_t logicDeviceId = 0;
    int32_t userDeviceId = GetUserDeviceId();
    ASSERT(RtErr::RT_DEVICE_FAILED, RuntimeGetLogicDevIdByUserDevId(userDeviceId, &logicDeviceId) == RT_SUCCESS)
        << "Trans usrDeviceId: " << userDeviceId << " to logDevId not success";
    MACHINE_LOGD("Current userDeviceId=%d, logicDeviceId=%d.", userDeviceId, logicDeviceId);
    return logicDeviceId;
}

inline uint64_t GetRuntimeL2Offset()
{
    uint64_t offset = 0;
    int32_t userDeviceId = GetUserDeviceId();
    RuntimeGetL2CacheOffset(userDeviceId, &offset);
    MACHINE_LOGD("L2 cache offset of device[%d] is %lu", userDeviceId, offset);
    return offset;
}

inline uint64_t AlignSize(const uint64_t bytes, const uint32_t aligns = 512U)
{
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline void* DevMallocWithAlignSize(const uint64_t size, const RtMemType memType)
{
    uint8_t* devPtr = nullptr;
    if (RuntimeMalloc(reinterpret_cast<void**>(&devPtr), AlignSize(size), memType, 0) != 0) {
        MACHINE_LOGW("Fail to malloc dev memory with size[%lu] and mem type[%u].", size, memType);
        return nullptr;
    }
    return devPtr;
}

inline void CopyFromTensor(uint8_t* hostDstAddr, const uint8_t* devSrcAddr, const uint64_t size)
{
#ifdef RUN_WITH_ASCEND_CAMODEL
    RuntimeMemcpy(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST);
#else
    RuntimeMemcpyAsync(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST,
        GetStreamContext().GetAiCoreStream());
    RuntimeStreamSynchronize(GetStreamContext().GetAiCoreStream());
#endif
}

inline void ExchangeCaptureMode(const bool& isCapture)
{
    if (isCapture) {
        AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
        AclMdlRICaptureThreadExchangeMode(&mode);
        MACHINE_LOGI("captureMode is: %d", static_cast<int>(mode));
    }
}

inline void* RegisterKernelBinary(const std::vector<uint8_t>& kernelBinary)
{
    void* hdl = nullptr;
    if (kernelBinary.empty()) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Kernel binary is empty.");
        return hdl;
    }
    RtDevBinary binary = {
        .magic = RT_DEV_BINARY_MAGIC_ELF,
        .version = 0,
        .data = kernelBinary.data(),
        .length = kernelBinary.size(),
    };
    if (RuntimeRegisterAllKernel(&binary, &hdl) != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Failed to register kernel bin");
    }
    return hdl;
}

int GetCfgBlockdim();

uint32_t GetProcessId();
}
