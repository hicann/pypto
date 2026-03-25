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
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "machine/utils/machine_error.h"
#include "memory_pool.h"

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

#ifdef BUILD_WITH_CANN

inline void CheckDeviceId() {
    int32_t devId = 0;
    int32_t getDeviceResult = rtGetDevice(&devId);
    if (getDeviceResult != RT_ERROR_NONE) {
        MACHINE_LOGE(RtErr::RT_DEVICE_FAILED, "fail get device id, check if set device id");
        return;
    }
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
    MACHINE_LOGD("Current userDeviceId=%d, logicDeviceId=%d.", userDeviceId, logicDeviceId);
    return logicDeviceId;
}

class RuntimeAgentMemory {
public:
    void AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        bool success = memPool_.AllocDevAddrInPool(devAddr, size);
        if (!success) {
            MACHINE_LOGE(DevCommonErr::ALLOC_FAILED,
                           "RuntimeAgent::AllocDevAddrInPool failed for size %lu", size);
            devAddr = nullptr;
        } else {
            MACHINE_LOGI("RuntimeAgentMemory: Alloc success %p", *devAddr);
        }
    }

    void FreeDevAddr(uint8_t *devAddr) {
        if (!devAddr) return; 
        memPool_.FreeDevAddr(devAddr);
    }

    void DynamicRecycle() {
        memPool_.DynamicRecycle();
    }

    void PrintPoolStatus() {
        memPool_.PrintPoolStatus();
    }

    bool CheckAllSentinels() {
        return memPool_.CheckAllSentinels();
    }

    static void CopyToDev(uint8_t *devDstAddr, uint8_t *hostSrcAddr, uint64_t size) {
        rtMemcpy(devDstAddr, size, hostSrcAddr, size, RT_MEMCPY_HOST_TO_DEVICE);
        MACHINE_LOGD("RuntimeAgent::CopyToDev src=%#lx, dst=%#lx, size=%lu", reinterpret_cast<uint64_t>(hostSrcAddr),
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
        memPool_.DestroyPool();
    }
private:
    bool validGetPgMask = true;
    DevMemoryPool memPool_;
};

class RuntimeAgentStream {
public:
    rtStream_t &GetStream() { return raStreamInstance; }

    aclrtStream &GetScheStream() { return raStreamInstanceSche; }

    rtStream_t &GetCtrlStream() { return raStreamInstanceCtrl; }

    rtStream_t &GetCurrentStream() { return currentStream; }

    void SetCurrentStream(aclrtStream &stream) { currentStream = stream; }

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
    aclrtStream currentStream{0};
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
        MACHINE_LOGD("rtGetL2CacheOffset=%lu", offset);
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

    void FreeTensor(uint8_t *devAddr) {
        MACHINE_LOGD("RuntimeAgent::FreeTensor");
        this->FreeDevAddr(devAddr);
    }

    void Finalize() {
        if (aclInited) {
            DestroyMemory();
            DestroyStream();
#ifndef RUN_WITH_ASCEND_CAMODEL
            aclFinalize();
#endif
        }

        MACHINE_LOGD("RuntimeAgent: Runtime quit!");
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
