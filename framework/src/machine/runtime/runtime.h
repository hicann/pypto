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

#include <vector>

#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"
#include "machine/utils/machine_error.h"
#include "machine/runtime/memory_pool.h"

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

enum res_map_type { RES_AICORE = 0, RES_HSCB_AICORE, RES_L2BUFF, RES_C2C, RES_MAP_TYPE_MAX };

struct res_map_info {
    processType_t target_proc_type;
    enum res_map_type res_type;
    unsigned int res_id;
    unsigned int flag;
    unsigned int rsv[1];
};

namespace npu::tile_fwk {
inline void CheckDeviceId()
{
    int32_t devId = 0;
    int32_t getDeviceResult = RuntimeGetDevice(&devId);
    if (getDeviceResult != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_DEVICE_FAILED, "fail get device id, check if set device id");
        return;
    }
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

class RuntimeAgentMemory {
public:
    void AllocDevAddr(uint8_t** devAddr, uint64_t size)
    {
        bool success = memPool_.AllocDevAddrInPool(devAddr, size);
        if (!success) {
            MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "RuntimeAgent::AllocDevAddrInPool failed for size %lu", size);
            devAddr = nullptr;
        } else {
            MACHINE_LOGI("RuntimeAgentMemory: Alloc success %p", *devAddr);
        }
    }

    void FreeDevAddr(uint8_t* devAddr)
    {
        if (!devAddr)
            return;
        memPool_.FreeDevAddr(devAddr);
    }

    void DynamicRecycle() { memPool_.DynamicRecycle(); }

    void PrintPoolStatus() { memPool_.PrintPoolStatus(); }

    bool CheckAllSentinels() { return memPool_.CheckAllSentinels(); }

    static void CopyToDev(uint8_t* devDstAddr, uint8_t* hostSrcAddr, uint64_t size)
    {
        RuntimeMemcpy(devDstAddr, size, hostSrcAddr, size, RtMemcpyKind::HOST_TO_DEVICE);
        MACHINE_LOGD(
            "RuntimeAgent::CopyToDev src=%#lx, dst=%#lx, size=%lu", reinterpret_cast<uint64_t>(hostSrcAddr),
            reinterpret_cast<uint64_t>(devDstAddr), size);
    }

    static void CopyFromDev(uint8_t* hostDstAddr, uint8_t* devSrcAddr, uint64_t size)
    {
        RuntimeMemcpy(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST);
    }

    int GetAicoreRegInfo(std::vector<int64_t>& aic, std::vector<int64_t>& aiv, const int& addrType);
    int GetAicoreRegInfoForDAV3510(std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu);

    // Only used in test case.
    void* MapAiCoreReg();

    bool GetValidGetPgMask() const { return validGetPgMask; }

protected:
    void DestroyMemory() { memPool_.DestroyPool(); }

private:
    bool validGetPgMask = true;
    DevMemoryPool memPool_;
};

class RuntimeAgentStream {
public:
    RtStream& GetStream() { return raStreamInstance; }

    AclRtStream& GetScheStream() { return raStreamInstanceSche; }

    RtStream& GetCtrlStream() { return raStreamInstanceCtrl; }

    RtStream& GetCurrentStream() { return currentStream; }

    void SetCurrentStream(AclRtStream& stream) { currentStream = stream; }

    void CreateStream()
    {
        RuntimeStreamCreate(&raStreamInstance, RT_STREAM_PRIORITY_DEFAULT);
        RuntimeStreamCreate(&raStreamInstanceSche, RT_STREAM_PRIORITY_DEFAULT);
        RuntimeStreamCreate(&raStreamInstanceCtrl, RT_STREAM_PRIORITY_DEFAULT);
    }
    void DestroyStream()
    {
        RuntimeStreamDestroy(raStreamInstance);
        RuntimeStreamDestroy(raStreamInstanceSche);
        RuntimeStreamDestroy(raStreamInstanceCtrl);
    }

private:
    RtStream raStreamInstance{0};
    RtStream raStreamInstanceCtrl{0};
    AclRtStream raStreamInstanceSche{0};
    AclRtStream currentStream{0};
};

class RuntimeAgent : public RuntimeAgentMemory, public RuntimeAgentStream {
public:
    RuntimeAgent(RuntimeAgent& other) = delete;

    void operator=(const RuntimeAgent& other) = delete;

    static RuntimeAgent* GetAgent()
    {
        static RuntimeAgent inst;
        return &inst;
    }

protected:
    RuntimeAgent()
    {
#ifdef RUN_WITH_ASCEND_CAMODEL
        // don't call AclInit, it will cause camodel running fail
#else
        AclInited = AclInit(nullptr) == 0;
#endif
        Init();
    }

public:
    ~RuntimeAgent() { Finalize(); }

public:
    static uint64_t GetL2Offset()
    {
        uint64_t offset = 0;
        int32_t userDeviceId = GetUserDeviceId();
        RuntimeGetL2CacheOffset(userDeviceId, &offset);
        MACHINE_LOGD("RuntimeGetL2CacheOffset=%lu", offset);
        return offset;
    }

    void CopyFromTensor(uint8_t* hostDstAddr, uint8_t* devSrcAddr, uint64_t size)
    {
#ifdef RUN_WITH_ASCEND_CAMODEL
        RuntimeMemcpy(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST);
#else
        RuntimeMemcpyAsync(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST, GetStream());
        RuntimeStreamSynchronize(GetStream());
#endif
    }

    void FreeTensor(uint8_t* devAddr)
    {
        MACHINE_LOGD("RuntimeAgent::FreeTensor");
        this->FreeDevAddr(devAddr);
    }

    void Finalize()
    {
        if (AclInited) {
            DestroyMemory();
            DestroyStream();
#ifndef RUN_WITH_ASCEND_CAMODEL
            AclFinalize();
#endif
        }

        MACHINE_LOGD("RuntimeAgent: Runtime quit!");
    }

private:
    void Init()
    {
        MACHINE_LOGI("RuntimeAgent: Init acl runtime!");
        CheckDeviceId();
        MACHINE_LOGD("RuntimeAgent: Create a default stream!");
        CreateStream();
    }

private:
    bool AclInited{false};
};
namespace machine {
inline npu::tile_fwk::RuntimeAgent* GetRA() { return npu::tile_fwk::RuntimeAgent::GetAgent(); }
} // namespace machine
} // namespace npu::tile_fwk
