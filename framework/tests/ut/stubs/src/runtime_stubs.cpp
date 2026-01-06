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
 * \file runtime_stubs.cpp
 * \brief
 */

#include <securec.h>
#include <runtime/rt.h>
#include <runtime/base.h>

#define EVENT_LENTH 10
#define DEVICE_VALUE 2000
#define TIME 12345

extern "C" {
rtError_t rtCtxSetCurrent(rtContext_t ctx)
{
    (void)ctx;
    return RT_ERROR_NONE;
}

rtError_t rtGetStreamId(rtStream_t stream, int32_t *streamId)
{
    (void)stream;
    *streamId = 0;
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx)
{
    int x = 1;
    *ctx = reinterpret_cast<void*>(x);
    return RT_ERROR_NONE;
}

rtError_t rtCtxSetDryRun (rtContext_t ctx, rtDryRunFlag_t enable, uint32_t flag)
{
    (void)ctx;
    (void)enable;
    (void)flag;
    return RT_ERROR_NONE;
}

rtError_t rtEventGetTimeStamp(uint64_t *time, rtEvent_t event)
{
    (void)event;
    *time = TIME;
    return RT_ERROR_NONE;
}

rtError_t rtEventCreate(rtEvent_t *event)
{
    *event = new int[EVENT_LENTH];
    return RT_ERROR_NONE;
}
rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream)
{
    (void)event;
    (void)stream;
    return RT_ERROR_NONE;
}

rtError_t rtEventSynchronize(rtEvent_t event)
{
    (void)event;
    return RT_ERROR_NONE;
}

rtError_t rtEventDestroy(rtEvent_t event)
{
    delete [] (int*)event;
    return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId)
{
    (void)type;
    (void)moduleId;
    (void)size;
    *devPtr =(void*)0x12345678;
    return RT_ERROR_NONE;

}

rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count)
{
    (void)devPtr;
    (void)destMax;
    (void)value;
    (void)count;
    return RT_ERROR_NONE;
}

rtError_t rtFree(void *devPtr)
{
    (void)devPtr;
    return RT_ERROR_NONE;
}

rtError_t rtMallocHost(void **hostPtr,  uint64_t  size, uint16_t moduleId)
{
    (void)moduleId;
    *hostPtr = new uint8_t[size];
    return RT_ERROR_NONE;
}

rtError_t rtFreeHost(void *hostPtr)
{
    delete[] (uint8_t*)hostPtr;
    return RT_ERROR_NONE;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority)
{
    (void)priority;
    *stream = new uint32_t;
    return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream)
{
    delete (uint32_t*)stream;
    return RT_ERROR_NONE;
}

rtError_t rtSetDevice(int32_t device)
{
    (void)device;
    return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronize(rtStream_t stream)
{
    (void)stream;
    return RT_ERROR_NONE;
}

rtError_t rtGetSocVersion(char* version, const uint32_t maxlen)
{
    (void)version;
    (void)maxlen;
    return 1;
}

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind)
{
    (void)destMax;
    (void)kind;
    (void)count;
    (void)src;
    (void)dst;
    return RT_ERROR_NONE;
}

rtError_t rtMemcpyEx(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind)
{
  return rtMemcpy(dst, destMax, src, count, kind);
}

rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count,
                        rtMemcpyKind_t kind, rtStream_t stream)
{
    (void)dst;
    (void)destMax;
    (void)src;
    (void)count;
    (void)kind;
    (void)stream;
    return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsyncWithoutCheckKind(void *dst, uint64_t destMax, const void *src, uint64_t count,
                                        rtMemcpyKind_t kind, rtStream_t stream)
{
  return rtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event)
{
    (void)stream;
    (void)event;
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceCount(int32_t* count)
{
    *count = 1;
    return RT_ERROR_NONE;
}

rtError_t rtDeviceReset(int32_t device)
{
    (void)device;
    return RT_ERROR_NONE;
}

rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end)
{
    (void)start;
    (void)end;
    *time = 10.0f;
    return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle)
{
    (void)bin;
    (void)handle;
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunch(const void *stubFunc,
    uint32_t blockDim,
    void *args,
    uint32_t argsSize,
    rtSmDesc_t *smDesc,
    rtStream_t stream)
{
    (void)stubFunc;
    (void)blockDim;
    (void)argsSize;
    (void)smDesc;
    (void)stream;
    (void)args;
    return RT_ERROR_NONE;
}

rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize)
{
    (void)aiCoreMemorySize;
    return RT_ERROR_NONE;
}

rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) {
    (void)bin;
    *handle = (void*)0x12345678;
    return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device)
{
    *device = 0;
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    (void)deviceId;
    (void)moduleType;
    (void)infoType;
    *value = DEVICE_VALUE;
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim,
                                   rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                   rtStream_t stm, uint32_t flags,
                                   const rtTaskCfgInfo_t *cfgInfo)
{
    (void)stubFunc;
    (void)blockDim;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)flags;
    (void)cfgInfo;
    return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t * const opName,
                                                const uint32_t blockDim, const rtAicpuArgsEx_t *argsInfo,
                                                rtSmDesc_t * const smDesc, const rtStream_t stm,
                                                const uint32_t flags)
{
    (void)kernelType;
    (void)opName;
    (void)blockDim;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)flags;
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    (void)hdl;
    (void)tilingKey;
    (void)blockDim;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)cfgInfo;
    return RT_ERROR_NONE;
}

rtError_t halGetDeviceInfoByBuff(uint32_t deviceId, int32_t moduleType, int32_t infoType, void *buf, int32_t *size_n) {
    (void) deviceId;
    (void) moduleType;
    (void) infoType;
    (void) buf;
    (void) size_n;
    return RT_ERROR_NONE;
}

rtError_t halMemCtl(int type, void *paramValue, size_t paramValueSize, void *outValue, size_t *outSizeRet) {
    (void) type;
    (void) paramValue;
    (void) paramValueSize;
    (void) outValue;
    (void) outSizeRet;
    return RT_ERROR_NONE;
}

rtError_t rtGetL2CacheOffset(uint32_t deviceId, uint64_t *offset) {
    (void)deviceId;
    (void)offset;
    return RT_ERROR_NONE;
}

rtError_t rtGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId) {
    (void)userDevId;
    (void)logicDevId;
    return RT_ERROR_NONE;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    (void)stm;
    (void)captureMdl;
    return RT_ERROR_NONE;
}

struct rtLoadBinaryConfig_t;
rtError_t rtsBinaryLoadFromFile([[maybe_unused]] const char * const binPath, [[maybe_unused]] const rtLoadBinaryConfig_t *const optionalCfg,
    [[maybe_unused]] rtBinHandle *binHandle)
{
    return RT_ERROR_NONE;
}

rtError_t rtsFuncGetByName([[maybe_unused]] const rtBinHandle binHandle, [[maybe_unused]] const char *kernelName,
    [[maybe_unused]] rtFuncHandle *funcHandle)
{
    return RT_ERROR_NONE;
}

struct rtKernelLaunchCfg_t;
rtError_t rtsLaunchCpuKernel([[maybe_unused]] const rtFuncHandle funcHandle, [[maybe_unused]]  const uint32_t blockDim,
    [[maybe_unused]] rtStream_t st, [[maybe_unused]] const rtKernelLaunchCfg_t *cfg, [[maybe_unused]] rtCpuKernelArgs_t *argsInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t blockDim,
                                    const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream,
                                    uint32_t flags)
{
    (void)soName;
    (void)kernelName;
    (void)blockDim;
    (void)argsInfo;
    (void)smDesc;
    (void)stream;
    (void)flags;
    return RT_ERROR_NONE;
}
}
