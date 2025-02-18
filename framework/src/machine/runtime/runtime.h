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
 * \file runtime.h
 * \brief
 */

#pragma once

#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <dlfcn.h>
#include <map>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>
#include <cassert>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <execinfo.h>
#include "interface/utils/log.h"
#include "interface/utils/common.h"
#include "interface/inner/config.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"

#ifdef BUILD_WITH_CANN
#include "acl/acl.h"
#include "runtime/rt.h"
#include "runtime/rt_preload_task.h"
#endif

constexpr int ADDR_MAP_TYPE_REG_AIC_CTRL = 2;
constexpr int ADDR_MAP_TYPE_REG_AIC_PMU_CTRL = 3;

struct AddrMapInPara {
    unsigned int addr_type;
    unsigned int devid;
};

struct AddrMapOutPara {
    unsigned long long ptr;
    unsigned long long len;
};

typedef enum tagProcType {
    PROCESS_CP1 = 0,
    PROCESS_CP2,
    PROCESS_DEV_ONLY,
    PROCESS_QS,
    PROCESS_HCCP,
    PROCESS_USER,
    PROCESS_CPTYPE_MAX
} processType_t;

enum res_map_type {
    RES_AICORE = 0,
    RES_HSCB_AICORE,
    RES_L2BUFF,
    RES_C2C,
    RES_MAP_TYPE_MAX
};

struct res_map_info {
    processType_t target_proc_type;
    enum res_map_type res_type;
    unsigned int res_id;
    unsigned int flag;
    unsigned int rsv[1];
};

namespace npu::tile_fwk {

#ifdef BUILD_WITH_CANN

inline void CheckDeviceId() {
    int32_t devId = 0;
    int32_t getDeviceResult = rtGetDevice(&devId);
    if (getDeviceResult != RT_ERROR_NONE) {
        ALOG_ERROR_F("fail get device id, check if set device id");
        return;
    }
 }

struct HugePageDesc {
    uint8_t *baseAddr;
    size_t allSize;
    size_t current;
    HugePageDesc(uint8_t *addr, size_t size) : baseAddr(addr), allSize(size), current(0) {}
};

inline size_t MemSizeAlign(const size_t bytes, const uint32_t aligns = 512U) {
    const size_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline int32_t GetUserDeviceId() {
    int32_t userDeviceId = 0;
    rtGetDevice(&userDeviceId);
    return userDeviceId;
}

inline int32_t GetLogDeviceId() {
    int32_t logicDeviceId = 0;
    int32_t userDeviceId = GetUserDeviceId();
    ASSERT(rtGetLogicDevIdByUserDevId(userDeviceId, &logicDeviceId) == RT_ERROR_NONE) << "Trans usrDeviceId: " <<
           userDeviceId << " to logDevId not success";
    ALOG_DEBUG_F("Current userDeviceId is %d, logic Deviceid is %d", userDeviceId, logicDeviceId);
    return logicDeviceId;
}

inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

class RuntimeAgentMemory {
public:
    void AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        auto alignSize = MemSizeAlign(size);
        ALOG_INFO_F("RuntimeAgent::Alloc size[%u] with align size[%lu].", size, alignSize);
        if (TryGetHugePageMem(devAddr, alignSize)) {
            return;
        }
        size_t allocSize = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        int res = rtMalloc((void **)devAddr, allocSize, ONG_GB_HUGE_PAGE_FLAGS, 0);
        if (res != 0) {
            ALOG_WARN_F("1G page mem alloc failed, turn to 2M page.\n");
            res = rtMalloc((void **)devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
            if (res != 0) {
                ALOG_ERROR_F("RuntimeAgent::AllocDevAddr failed for size %lu", size);
                return;
            }
            allocatedDevAddr.emplace_back(*devAddr);
            ALOG_INFO_F("AllocDevAddr %p size is %lu", *devAddr, size);
            return;
        }
        allocatedDevAddr.emplace_back(*devAddr);
        hugePageVec.emplace_back(HugePageDesc(*devAddr, allocSize));
        if (!TryGetHugePageMem(devAddr, alignSize)) {
            ALOG_ERROR_F("RuntimeAgent::AllocDevAddr failed for size %lu", size);
            return;
        }
        ALOG_INFO_F("Alloc 1G page mem %p size is %lu", *devAddr, allocSize);
        return;
    }

    bool IsHugePageMemory(uint8_t *devAddr) const {
        for (auto &hugepage : hugePageVec) {
            if (devAddr >= hugepage.baseAddr && devAddr < hugepage.baseAddr + hugepage.allSize)
                return true;
        }
        return false;
    }

    static void CopyToDev(uint8_t *devDstAddr, uint8_t *hostSrcAddr, uint64_t size) {
        rtMemcpy(devDstAddr, size, hostSrcAddr, size, RT_MEMCPY_HOST_TO_DEVICE);
        ALOG_DEBUG_F("RuntimeAgent::CopyToDev for src %lx to dst %lx with size %u", reinterpret_cast<uint64_t>(hostSrcAddr),
            reinterpret_cast<uint64_t>(devDstAddr), size);
    }

    static void CopyFromDev(uint8_t *hostDstAddr, uint8_t *devSrcAddr, uint64_t size) {
        rtMemcpy(hostDstAddr, size, devSrcAddr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    int GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType);
    int GetAicoreRegInfoForDAV3510(std::vector<int64_t> &regs, std::vector<int64_t> &regsPmu);

    // Only used in test case.
    void *MapAiCoreReg();
    
    bool GetValidGetPgMask() const {
        return validGetPgMask;
    }
protected:
    void DestroyMemory() {
        for (uint8_t *addr : allocatedDevAddr) {
            rtFree(addr);
        }
    }
private:
    bool TryGetHugePageMem(uint8_t **devAddr, uint64_t alignSize) {
        for (size_t i = 0; i < hugePageVec.size(); ++i) {
            if (hugePageVec[i].current + alignSize <= hugePageVec[i].allSize) {
                *devAddr = hugePageVec[i].baseAddr + hugePageVec[i].current;
                hugePageVec[i].current += alignSize;
                ALOG_INFO_F("HugePage Mem get with size:%u addr:%p.", alignSize, *devAddr);
                return true;
            }
        }
        return false;
    }
private:
    bool validGetPgMask = true;
    std::vector<HugePageDesc> hugePageVec;
    std::vector<uint8_t *> allocatedDevAddr;
};

class RuntimeAgentStream {
public:
    rtStream_t &GetStream() { return raStreamInstance; }

    aclrtStream &GetScheStream() { return raStreamInstanceSche; }

    rtStream_t &GetCtrlStream() { return raStreamInstanceCtrl; }

    void CreateStream() {
        rtStreamCreate(&raStreamInstance, RT_STREAM_PRIORITY_DEFAULT);
        rtStreamCreate(&raStreamInstanceSche, RT_STREAM_PRIORITY_DEFAULT);
        rtStreamCreate(&raStreamInstanceCtrl, RT_STREAM_PRIORITY_DEFAULT);
    }
    void DestroyStream() {
        rtStreamDestroy(raStreamInstance);
        rtStreamDestroy(raStreamInstanceSche);
        rtStreamDestroy(raStreamInstanceCtrl);
    }
private:
    rtStream_t raStreamInstance{0};
    rtStream_t raStreamInstanceCtrl{0};
    aclrtStream raStreamInstanceSche{0};
};

class RuntimeAgent : public RuntimeAgentMemory, public RuntimeAgentStream {
public:
    RuntimeAgent(RuntimeAgent &other) = delete;

    void operator=(const RuntimeAgent &other) = delete;

    static RuntimeAgent *GetAgent() {
        static RuntimeAgent inst;
        return &inst;
    }

protected:
    RuntimeAgent() {
#ifdef RUN_WITH_ASCEND_CAMODEL
        // don't call aclInit, it will cause camodel running fail
#else
        aclInited = aclInit(nullptr) == 0;
#endif
        Init();
    }

public:
    ~RuntimeAgent() { Finalize(); }

public:
    static uint64_t GetL2Offset () {
        uint64_t offset = 0;
        int32_t userDeviceId = GetUserDeviceId();
        rtGetL2CacheOffset(userDeviceId, &offset);
        ALOG_DEBUG_F("rtGetL2CacheOffset %lu", offset);
        return offset;
    }

    void CopyFromTensor(uint8_t *hostDstAddr, uint8_t *devSrcAddr, uint64_t size) {
#ifdef RUN_WITH_ASCEND_CAMODEL
        rtMemcpy(hostDstAddr, size, devSrcAddr, size, RT_MEMCPY_DEVICE_TO_HOST);
#else
        rtMemcpyAsync(hostDstAddr, size, devSrcAddr, size, RT_MEMCPY_DEVICE_TO_HOST, GetStream());
        rtStreamSynchronize(GetStream());
#endif
    }

    void FreeTensor(uint8_t *devAddr) const {
        ALOG_DEBUG_F("RuntimeAgent::FreeTensor");
        if (IsHugePageMemory(devAddr))
            return;
        rtFree(devAddr);
    }

    void Finalize() {
        if (aclInited) {
            DestroyMemory();
            DestroyStream();
#ifndef RUN_WITH_ASCEND_CAMODEL
            aclFinalize();
#endif
        }

        ALOG_DEBUG_F("RuntimeAgent: runtime quit");
    }

private:
    void Init() {
        ALOG_INFO_F("RuntimeAgent: Init acl runtime!");
        CheckDeviceId();
        ALOG_DEBUG_F("RuntimeAgent: Create a default stream!");
        CreateStream();
    }

private:
    bool aclInited{false};
};

class RuntimeHostAgentMemory {
public:
    void backtracePrint(int count = 1000) {
        std::vector<void*> backtraceStack(count);
        int backtraceStackCount = backtrace(backtraceStack.data(), static_cast<int>(backtraceStack.size()));
        char **backtraceSymbolList = backtrace_symbols(backtraceStack.data(), backtraceStackCount);
        free(backtraceSymbolList);
    }
#define DEVICE_ALLOC_ALIGN 512
    uint8_t* AllocHostAddr(uint64_t size) {
        if (size == 0) {
            ALOG_ERROR_F("Malloc size is 0!");
            return nullptr;
        }
        // Device allocate always 512 aligned.
        auto hostPtr = (uint8_t *)malloc(size + DEVICE_ALLOC_ALIGN);
        allocatedHostAddr.emplace_back(hostPtr);
        auto resultPtr = (uint8_t *)((((uint64_t)hostPtr) + DEVICE_ALLOC_ALIGN - 1) / DEVICE_ALLOC_ALIGN * DEVICE_ALLOC_ALIGN);
        return resultPtr;
    }
protected:
    void DestroyMemory() {
        for (uint8_t *addr : allocatedHostAddr) {
            free(addr);
        }
    }
private:
    std::vector<uint8_t *> allocatedHostAddr;
};

class RuntimeHostAgent : public RuntimeHostAgentMemory {
public:
    RuntimeHostAgent(RuntimeHostAgent &other) = delete;

    void operator=(const RuntimeHostAgent &other) = delete;

    static RuntimeHostAgent *GetAgent() {
        static RuntimeHostAgent inst;
        return &inst;
    }

protected:
    RuntimeHostAgent() {
        Init();
    }

public:
    ~RuntimeHostAgent() { Finalize(); }

public:
    void Finalize() {
        if (hostInited) {
            DestroyMemory();
        }
    }

private:
    void Init() {
        hostInited = true;
    }

private:
    bool hostInited{false};
};

namespace machine {

inline npu::tile_fwk::RuntimeAgent *GetRA() {
    return npu::tile_fwk::RuntimeAgent::GetAgent();
}

inline npu::tile_fwk::RuntimeHostAgent *GetRuntimeHostAgent() {
    return npu::tile_fwk::RuntimeHostAgent::GetAgent();
}

} // namespace machine
#else
class RuntimeHostAgentMemory {
public:
    void backtracePrint(int count = 1000) {
        std::vector<void*> backtraceStack(count);
        int backtraceStackCount = backtrace(backtraceStack.data(), static_cast<int>(backtraceStack.size()));
        char **backtraceSymbolList = backtrace_symbols(backtraceStack.data(), backtraceStackCount);
        free(backtraceSymbolList);
    }
#define DEVICE_ALLOC_ALIGN 512
    uint8_t* AllocHostAddr(uint64_t size) {
        if (size == 0) {
            ALOG_ERROR_F("Malloc size is 0!");
            return nullptr;
        }
        // Device allocate always 512 aligned.
        auto hostPtr = (uint8_t *)malloc(size + DEVICE_ALLOC_ALIGN);
        allocatedHostAddr.emplace_back(hostPtr);
        auto resultPtr = (uint8_t *)((((uint64_t)hostPtr) + DEVICE_ALLOC_ALIGN - 1) / DEVICE_ALLOC_ALIGN * DEVICE_ALLOC_ALIGN);
        return resultPtr;
    }
protected:
    void DestroyMemory() {
        for (uint8_t *addr : allocatedHostAddr) {
            free(addr);
        }
    }
private:
    std::vector<uint8_t *> allocatedHostAddr;
};

class RuntimeHostAgent : public RuntimeHostAgentMemory {
public:
    RuntimeHostAgent(RuntimeHostAgent &other) = delete;

    void operator=(const RuntimeHostAgent &other) = delete;

    static RuntimeHostAgent *GetAgent() {
        static RuntimeHostAgent inst;
        return &inst;
    }

protected:
    RuntimeHostAgent() {
        Init();
    }

public:
    ~RuntimeHostAgent() { Finalize(); }

public:
    void Finalize() {
        if (hostInited) {
            DestroyMemory();
        }
    }

private:
    void Init() {
        hostInited = true;
    }

private:
    bool hostInited{false};
};

namespace machine {
inline npu::tile_fwk::RuntimeHostAgent *GetRuntimeHostAgent() {
    return npu::tile_fwk::RuntimeHostAgent::GetAgent();
}
}
#endif
} // namespace npu::tile_fwk
