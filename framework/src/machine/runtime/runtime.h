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
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk_log.h"

#ifdef BUILD_WITH_CANN
#include "acl/acl.h"
#include "runtime/rt.h"
#include "runtime/rt_preload_task.h"
#endif

constexpr int ADDR_MAP_TYPE_REG_AIC_CTRL = 2;
constexpr int ADDR_MAP_TYPE_REG_AIC_PMU_CTRL = 3;

static constexpr uint64_t SENTINEL_VALUE = 0xDEADBEEFDEADBEEF;
static constexpr uint32_t SENTINEL_NUM = 64;
static constexpr uint32_t SENTINEL_MEM_SIZE = 512;

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

class RuntimeHostAgentMemory {
public:
#define DEVICE_ALLOC_ALIGN 512
    uint8_t* AllocHostAddr(uint64_t size, bool cached = true, bool simuDevAlign = true) {
        if (size == 0) {
            MACHINE_LOGE("Malloc size is 0!");
            return nullptr;
        }
        // Device allocate always 512 aligned.
        auto hostPtr = (uint8_t *)malloc(size + DEVICE_ALLOC_ALIGN);
        if (hostPtr == nullptr) {
            MACHINE_LOGE("Malloc failed size !", size);
            return nullptr;
        }
        memset_s(hostPtr, size + DEVICE_ALLOC_ALIGN, 0, size + DEVICE_ALLOC_ALIGN);
        if (cached) {
            allocatedHostAddr.emplace_back(hostPtr);
        }
        return simuDevAlign ?
            (uint8_t *)((((uint64_t)hostPtr) + DEVICE_ALLOC_ALIGN - 1) / DEVICE_ALLOC_ALIGN * DEVICE_ALLOC_ALIGN) :
            hostPtr;
    }

    void Free(uint8_t* ptr) {
        if(ptr) {
            free(ptr);
        }
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

#ifdef BUILD_WITH_CANN

inline void CheckDeviceId() {
    int32_t devId = 0;
    int32_t getDeviceResult = rtGetDevice(&devId);
    if (getDeviceResult != RT_ERROR_NONE) {
        MACHINE_LOGE("fail get device id, check if set device id");
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
    MACHINE_LOGD("Current userDeviceId is %d, logic Deviceid is %d", userDeviceId, logicDeviceId);
    return logicDeviceId;
}

inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

class RuntimeAgentMemory {
public:
    RuntimeAgentMemory() {
        needMemCheck_ = (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL);
        sentinelVec_ = std::vector<uint64_t>(SENTINEL_NUM, SENTINEL_VALUE);
    }
    void PutSentinelAddr(uint8_t *baseAddr, uint64_t baseSize) {
        if (needMemCheck_) {
            uint8_t *sentinelAddr = baseAddr + baseSize;
            if (rtMemcpy(sentinelAddr, SENTINEL_MEM_SIZE, sentinelVec_.data(), SENTINEL_MEM_SIZE, RT_MEMCPY_HOST_TO_DEVICE) != 0) {
                ALOG_WARN_F("Memory copy sentinel value failed! Do not check memory.");
                return;
            }
            ALOG_INFO_F("Base addr add %p with sentinelAddr %p.", baseAddr, sentinelAddr);
            sentinelValMap_[baseAddr].push_back(sentinelAddr);
        }
    }
    void AllocDevAddr(uint8_t **devAddr, uint64_t size, bool tmpAddr = false) {
        auto alignSize = MemSizeAlign(size);
        if (needMemCheck_) {
            alignSize += SENTINEL_MEM_SIZE;
        }
        MACHINE_LOGI("RuntimeAgent::Alloc size[%lu] with align size[%lu].", size, alignSize);
        if (TryGetHugePageMem(devAddr, size, alignSize, tmpAddr)) {
            return;
        }
        size_t allocSize = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        int res = rtMalloc((void **)devAddr, allocSize, ONG_GB_HUGE_PAGE_FLAGS, 0);
        if (res != 0) {
            MACHINE_LOGW("1G page mem alloc failed, turn to 2M page.\n");
            res = rtMalloc((void **)devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
            if (res != 0) {
                MACHINE_LOGE("RuntimeAgent::AllocDevAddr failed for size %lu", size);
                return;
            }
            if (tmpAddr) {
                allocatedTmpDevAddr.emplace_back(*devAddr);
            } else {
                allocatedDevAddr.emplace_back(*devAddr);
            }
            MACHINE_LOGI("AllocDevAddr %p size is %lu", *devAddr, size);
            PutSentinelAddr(*devAddr, size);
            return;
        }
        if (tmpAddr) {
            allocatedTmpDevAddr.emplace_back(*devAddr);
            tmpHugePageVec.emplace_back(HugePageDesc(*devAddr, allocSize));
        } else {
            allocatedDevAddr.emplace_back(*devAddr);
            hugePageVec.emplace_back(HugePageDesc(*devAddr, allocSize));
        }
        if (!TryGetHugePageMem(devAddr, size, alignSize, tmpAddr)) {
            MACHINE_LOGE("RuntimeAgent::AllocDevAddr failed for size %lu", size);
            return;
        }
        MACHINE_LOGI("Alloc 1G page mem %p size is %lu.", *devAddr, allocSize);
        return;
    }

    // Check sentinel values for memory corruption
    bool CheckAllSentinels() {
        if (!needMemCheck_) {
            return true;
        }
        bool allGood = true;
        for (auto &iter : sentinelValMap_) {
            if (!CheckSentinel(iter.first, false)) {
                allGood = false;
            }
        }
        if (!allGood) {
            ALOG_ERROR_F("CheckAllSentinels failed.");
        }
        sentinelValMap_.clear();
        return allGood;
    }
    void PrintSentinelVal(std::vector<uint64_t> &sentinelVal, uint8_t *sentinelAddr) {
        std::ostringstream oss;
        uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(sentinelVal.data());
        oss << "Print Sentinel val in hex with ori val[" << std::hex << "0x" << SENTINEL_VALUE << "]" << std::endl;
        ALOG_ERROR_F("%s", oss.str().c_str());
        oss.str("");
        for (uint32_t i = 0; i < SENTINEL_MEM_SIZE; ++i) {
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte_ptr[i];
            if ((i + 1) % 16 == 0) {
                oss << std::endl;
            } else {
                oss << " ";
            }
            if ((i + 1) % 64 == 0) {
                ALOG_ERROR_F("Sentinel Addr:%p Val:[\n%s]", sentinelAddr + i, oss.str().c_str());
                oss.str("");
            }
        }
    }
    // Check sentinel values for memory corruption
    bool CheckSentinel(uint8_t *baseAddr, bool remove = true) {
        if (!needMemCheck_ || sentinelValMap_.empty()) {
            return true;
        }
        // UT no need check sentinel
        if (baseAddr == reinterpret_cast<uint8_t*>(0x12345678)) {
            return true;
        }
        auto iter = sentinelValMap_.find(baseAddr);
        if (iter == sentinelValMap_.end()) {
            ALOG_ERROR_F("Base addr %p not found in map, need check code.", baseAddr);
            return false;
        }
        std::vector<uint64_t> sentinelVal(SENTINEL_NUM, 0);
        bool allGood = true;
        auto &sentinelVec = iter->second;
        for (auto sentinelAddr : sentinelVec) {
            ALOG_INFO_F("Check base:%p sentinelAddr:%p.", baseAddr, sentinelAddr);
            if (rtMemcpy(sentinelVal.data(), SENTINEL_MEM_SIZE, sentinelAddr, SENTINEL_MEM_SIZE, RT_MEMCPY_DEVICE_TO_HOST) != 0) {
                ALOG_WARN_F("Memory copy D2H failed! Do not check memory.");
                break;
            }
            if (memcmp(sentinelVal.data(), sentinelVec_.data(), SENTINEL_MEM_SIZE) != 0) {
                PrintSentinelVal(sentinelVal, sentinelAddr);
                allGood = false;
            }
        }
        if (!allGood) {
            ALOG_ERROR_F("BaseAddr:%p check sentinel failed.", baseAddr);
        } else {
            ALOG_INFO_F("BaseAddr:%p check sentinel Ok.", baseAddr);
        }
        if (remove) {
            sentinelValMap_.erase(baseAddr);
        }
        return allGood;
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
        MACHINE_LOGD("RuntimeAgent::CopyToDev for src %lx to dst %lx with size %u", reinterpret_cast<uint64_t>(hostSrcAddr),
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

    void FreeTmpMemory() {
        for (uint8_t *addr : allocatedTmpDevAddr) {
            ASSERT(CheckSentinel(addr));
            rtFree(addr);
        }
        allocatedTmpDevAddr.clear();
        tmpHugePageVec.clear();
    }

protected:
    void DestroyMemory() {
        for (uint8_t *addr : allocatedDevAddr) {
            ASSERT(CheckSentinel(addr));
            rtFree(addr);
        }
        allocatedDevAddr.clear();
        hugePageVec.clear();
        FreeTmpMemory();
    }
private:
    bool TryGetHugePageMem(uint8_t **devAddr, uint64_t oriSize, uint64_t alignSize, bool tmpAddr) {
        std::vector<HugePageDesc> &pageVec = tmpAddr ? tmpHugePageVec : hugePageVec;
        for (size_t i = 0; i < pageVec.size(); ++i) {
            if (pageVec[i].current + alignSize <= pageVec[i].allSize) {
                *devAddr = pageVec[i].baseAddr + pageVec[i].current;
                PutSentinelAddr(pageVec[i].baseAddr, pageVec[i].current + oriSize);
                pageVec[i].current += alignSize;
                MACHINE_LOGI("HugePage Mem get with size:%u addr:%p.", alignSize, *devAddr);
                return true;
            }
        }
        return false;
    }
    void BacktracePrint(int count = 1000) {
        std::vector<void*> backtraceStack(count);
        int backtraceStackCount = backtrace(backtraceStack.data(), static_cast<int>(backtraceStack.size()));
        char **backtraceSymbolList = backtrace_symbols(backtraceStack.data(), backtraceStackCount);
        for (int i = 0; i < backtraceStackCount; i++) {
            ALOG_INFO_F("backtrace frame[%d]: %s", i, backtraceSymbolList[i]);
        }
        free(backtraceSymbolList);
    }
private:
    bool validGetPgMask = true;
    bool needMemCheck_{false};
    std::vector<uint64_t> sentinelVec_;
    std::unordered_map<uint8_t *, std::vector<uint8_t *>> sentinelValMap_;
    std::vector<HugePageDesc> hugePageVec;
    std::vector<HugePageDesc> tmpHugePageVec;
    std::vector<uint8_t *> allocatedDevAddr;
    std::vector<uint8_t *> allocatedTmpDevAddr;
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
        MACHINE_LOGD("rtGetL2CacheOffset %lu", offset);
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
        MACHINE_LOGD("RuntimeAgent::FreeTensor");
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

        MACHINE_LOGD("RuntimeAgent: runtime quit");
    }

private:
    void Init() {
        MACHINE_LOGI("RuntimeAgent: Init acl runtime!");
        CheckDeviceId();
        MACHINE_LOGD("RuntimeAgent: Create a default stream!");
        CreateStream();
    }

private:
    bool aclInited{false};
};
namespace machine {
inline npu::tile_fwk::RuntimeAgent *GetRA() {
    return npu::tile_fwk::RuntimeAgent::GetAgent();
}
} // namespace machine
#else

#endif
} // namespace npu::tile_fwk
