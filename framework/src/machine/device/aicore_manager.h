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
 * \file aicore_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <sys/ioctl.h>
#include <functional>
#include <vector>
#include <atomic>
#include <array>
#include <semaphore.h>
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "aicpu_task_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"
#include "aicore_prof.h"
#include "dynamic/device_utils.h"
#include "aicore_dump.h"
#include "interface/utils/common.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk {
const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;
const uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0;
const uint32_t REG_SPR_COND = 0x4C8;
const uint32_t REG_SPR_MAGIC = 0x78;
const int INVALID_CORE_IDX = 0xFF;

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;
const uint32_t CORE_NUM_PER_AI_CORE = 3;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const uint32_t READY_ID_FIX_CACHE_NUM = 800;
const uint32_t AICORE_TYPE_NUM = 2;

constexpr uint32_t MAX_AICORE_NUM = 108;
constexpr uint32_t NAX_AIV_TOTAL_NUM = 72;
constexpr uint32_t MAX_MANAGER_AIV_NUM = NAX_AIV_TOTAL_NUM;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS                     // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31            // 低32位存储的task的状态
#define REG_HIGH_TASK_ID(regVal) ((regVal) >> 32) & REG_31_BITS            // 高32位存储的taskid
#define REG_HIGH_TASK_STATE(regVal) (((regVal) >> 32) & REG_32_BITS) >> 31 // 高32位存储的task状态
constexpr uint32_t TASK_FIN_STATE = 1;                                     // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0;                                     // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2;                                       // 一次寄存器task个数

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_THIRTY_TWO = 32;

constexpr uint32_t DEFAULT_QUEUE_SIZE = 64;

constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    TaskInfo(int idx, uint64_t id) : coreIdx(idx), taskId(id) {}
};

struct sdma_l2_cmo_desc {
    unsigned long   src_addr;
    size_t          size;
    char            cmo_opcode;
};

#define SDMA_FILE "/dev/sdma"

#define IOCTL_SDMA_L2_CMO  _IOW('s', 3, struct sdma_l2_cmo_desc)

struct DeviceTaskCtrl {
    int taskType{DEVICE_TASK_TYPE_INVALID};
    uint64_t taskId{0};
    void *devTask{nullptr};
    uint64_t finishedAicFunctionCnt{0}; // 所有aicpu处理完成的aic function个数，多线程增加修改
    uint64_t finishedAivFunctionCnt{0}; // 所有aicpu处理完成的aiv function个数，多线程增加修改
    uint64_t finishedAicpuFunctionCnt{0}; // 所有aicpu处理完成的aicpu function个数，多线程增加修改
    std::atomic<uint64_t> finishedFunctionCnt{0};
    std::atomic<int> refcnt{-1};
    void (*finishFunc)(void *devTask){nullptr};
    int retCode{0};
    std::array<std::array<std::atomic<bool>, npu::tile_fwk::dynamic::MAX_SCHEDULE_AICPU_NUM>, AICORE_TYPE_NUM>  isAicpuIdle;

    bool IsFree() { return refcnt == -1; }

    void PutTask(int ret) {
        if (ret != 0)
            retCode = ret;

        auto cnt = refcnt--;
        while (refcnt.load(std::memory_order_relaxed) != 0)
            ;

        if (cnt == 1) {
            if (finishFunc) {
                finishFunc(devTask);
            }
            refcnt = -1;
        }
    }
};

inline void ReadyQueueLock(StaticReadyCoreFunctionQueue *rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void ReadyQueueUnLock(StaticReadyCoreFunctionQueue *rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

void SdmaPrefetch(DeviceTask *devTask);

class AiCoreManager {
public:
    AiCoreManager(AicpuTaskManager &aicpuTaskManager) : aicpuTaskManager_(aicpuTaskManager), prof_(*this){};
    ~AiCoreManager(){};

    inline void InitTaskData(DeviceTaskCtrl *taskCtrl) {
        curTaskCtrl_ = taskCtrl;
        curDevTask_ = static_cast<DeviceTask *>(taskCtrl->devTask);
        ForEachManageAicore([this](int coreIdx) {
            volatile int64_t *funcData = &args_[coreIdx]->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX];
            *funcData = reinterpret_cast<int64_t>(&curDevTask_->coreFuncData);
        });
        readyAicCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);
    }

    inline void CountSendTask(uint64_t& sentAic, uint64_t& sentAiv) {
        sentAic += sendCnt_[static_cast<int>(CoreType::AIC)];
        sentAiv += sendCnt_[static_cast<int>(CoreType::AIV)];
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] += sentAic;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] += sentAiv;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    template <bool enableAicpuTask = false>
    inline void RunCoreTask(DeviceTaskCtrl *taskCtrl) {
        uint64_t sentAic = 0;
        uint64_t sentAiv = 0;
        DispatchAiCoreTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
        CountSendTask(sentAic, sentAiv);
        DispatchAiCoreTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);
        CountSendTask(sentAic, sentAiv);
        uint64_t sentAicpu = 0UL;
        if constexpr (enableAicpuTask) {
            if (aicpuIdx_ == 1) {
                sentAicpu = ResolveDepForAicpuTask();
            }
        }

        taskCtrl->finishedFunctionCnt.fetch_add(sentAic + sentAiv + reSolveHubCnt_ + sentAicpu,
            std::memory_order_relaxed);

        DEV_IF_VERBOSE_DEBUG {
            __sync_fetch_and_add(&(taskCtrl->finishedAicFunctionCnt), sentAic);
            __sync_fetch_and_add(&(taskCtrl->finishedAivFunctionCnt), sentAiv);
            __sync_fetch_and_add(&(taskCtrl->finishedAicpuFunctionCnt), sentAicpu);
            procAicCoreFunctionCnt_ += sentAic;
            procAivCoreFunctionCnt_ += sentAiv;
            procAicpuFunctionCnt_ += sentAicpu;
            DEV_VERBOSE_DEBUG("finish send  aic task cnt: %lu,  aiv task cnt: %lu, hub task cnt:%lu, aicpu task cnt:%lu, target totalcnt: %lu \n",
                taskCtrl->finishedAicFunctionCnt, taskCtrl->finishedAivFunctionCnt,
                reSolveHubCnt_, taskCtrl->finishedAicpuFunctionCnt, curDevTask_->coreFunctionCnt);
        }
        reSolveHubCnt_ = 0;
    }

    int RunTask(DeviceTaskCtrl *taskCtrl);

    int Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl = nullptr);

    void PushTask(DeviceTaskCtrl *taskCtrl) { taskQueue_.Enqueue(taskCtrl); }

private:
    void DumpTaskProf();

    void ProfStop();

    void DumpAiCoreStatus() const;

    void DumpTaskTensor(int &coreIdx, volatile TaskStat *stat);

    bool CheckTaskFinished(int coreIdx);

    int WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd);

    uint64_t TryBatchSendTask(CoreType type, StaticReadyCoreFunctionQueue* readyQue, int coreIdxStart, int coreIdxEnd);

    uint32_t BatchSendTask(CoreType type, uint64_t *newTask, uint32_t taskCount, int coreIdxStart, int coreIdxEnd, bool isLifo);

    uint64_t DispatchAiCoreTask(CoreType type, StaticReadyCoreFunctionQueue* readyQue, int coreIdxStart, int coreIdxEnd);

    void SendTaskToAiCore(CoreType type, int coreIdx, uint64_t newTask);

    void SetAiCpuStat(int coreIdx, uint64_t taskId);

    void AddTask(int coreIdx, uint64_t taskId);

    void PushReadyQue(StaticReadyCoreFunctionQueue *readyQue, void *idList, uint32_t idCnt) const;

    void ResolveDepForAllAiCore(CoreType type, StaticReadyCoreFunctionQueue *readyQue, int coreIdxStart, int coreIdxEnd);

    void BatchPushReadyQueue();

    uint64_t ResolveDepForAicpuTask();

    inline bool IsNoTaskDispatch(int coreIdx) {
        return (runningIds_[coreIdx] == AICORE_TASK_INIT && pendingIds_[coreIdx] == AICORE_TASK_INIT);
    }

    bool SendTaskDirectlyWhenCoreRunReady(CoreType type, int coreIdx);

    void ResolveWhenSyncMode(CoreType type, uint32_t finTaskId, uint32_t finTaskState, int coreIdx);

    void ResolveByRegVal(CoreType type, int coreIdx, uint64_t finTaskRegVal);

    inline void PushAicpuTaskQueue(uint64_t taskId) {
        aicpuTaskManager_.TaskEnqueue(taskId);
    }

    int GetNextSendCoreIdx(int coreType);

    bool SendTaskDirectlyWhenTaskReady(int coreType, int64_t taskId);

    void PushReadyTask(int coreType, int64_t taskId);

    void ResolveVirtualPure(uint64_t dep, CoreFunctionReadyState* readyState);

    void ResolveVirtualMix(uint64_t dep, CoreFunctionReadyState* readyState);

    void ResolveByCoreType(int coretype, uint64_t depTaskId, CoreFunctionReadyState *readyState);

    void ResolveDep(uint64_t finishId);

    void ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId);

    bool IsExistOtherAicpuIdle(CoreType type);

    inline void AicpuIsBusy(CoreType type) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_] != false) {
            curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_].store(false, std::memory_order_relaxed);
        }
    }

    inline void AicpuIsIdle(CoreType type) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_] != true) {
            curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_].store(true, std::memory_order_relaxed);
        }
    }

    inline void BatchGetFinishedTask(uint64_t finTask[], int coreIdxStart, int coreIdxEnd) {
        uint64_t finTaskGet;
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            finTaskGet = GetFinishedTask(i);
            finTask[i - coreIdxStart] = finTaskGet;
        }
    }

    inline uint64_t GetFinishedTask(int coreIdx) { return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]); }

    inline void ResetCnt() {
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] = 0;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] = 0;
        readyCount[static_cast<int>(CoreType::AIC)] = 0;
        readyCount[static_cast<int>(CoreType::AIV)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    void Init(int threadIdx, DeviceArgs *deviceArgs);

    int HandkShake();

    /* assign aic and aiv core index section for this aicpu */
    void UpdateAiCoreBlockIndexSection();

    inline int GetPhyIdByBlockId(int coreIdx) { return blockIdToPhyCoreId_[coreIdx]; }

    void MapRegistersForAllCores();

    inline void ForEachManageAicore(std::function<void(int coreIdx)> func) const {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            func(i);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            func(i);
        }
    }

    inline void ForEachManageAicoreReverse(std::function<void(int coreIdx)> func) const {
        for (int i = aicEnd_ - 1; i >= aicStart_; i--) {
            func(i);
        }
        for (int i = aivEnd_ -1; i >= aivStart_ ; i--) {
            func(i);
        }
    }

    inline int ForEachManageAicoreWithRet(std::function<int(int coreIdx)> func) const {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        for (int i = aicStart_; i < aicEnd_; ++i) {
            ret = func(i);
            if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
                DEV_ERROR("proc aicore aic %d failed.\n", i);
                return ret;
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            ret = func(i);
            if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
                DEV_ERROR("proc aicore aiv %d failed.\n", i);
                return ret;
            }
        }
        return ret;
    }

    inline uint32_t ReadReg32(int coreIdx, int offset) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
          return *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset));
        }
        return 0;
    }

    inline void WriteReg32(int coreIdx, int offset, uint32_t val) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
          *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset)) = val;
        }
        return;
    }

    inline void SetReadyQueue(int coreIdx, uint64_t value) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx == -1) {
            return;
        }
        volatile uint64_t *readyQ = readyRegQueues_[idx];
        if (readyQ != nullptr) {
            *readyQ = value;
        }
    }

    inline uint64_t GetReadyQueueRegValue(int coreIdx) {
        volatile uint64_t *readyQ = readyRegQueues_[GetPhyIdByBlockId(coreIdx)];
        if (readyQ != nullptr) {
            return *readyQ;
        }
        return 0;
    }

    inline void WriteReg32ALl(int offset, uint32_t val) {
        for (int i = 0; i < aicNum_ + aivNum_; i++) {
            if (regAddrs_[i] != 0) {
                *(reinterpret_cast<volatile uint32_t *>(regAddrs_[i] + offset)) = val;
            }
        }
    }

    void AbnormalStop();

    void NormalStop();

    inline int GetAllAiCoreNum() { return aicNum_ + aivNum_; }
    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }
    inline void SetNextDfxPos(int coreIdx) {
            taskDfxStatPos_[coreIdx] =
                taskDfxStatPos_[coreIdx] == REG_LOW_TASK_PING ? REG_LOW_TASK_PONG : REG_LOW_TASK_PING;
    }
    inline int GetDfxPos(int coreIdx) { return taskDfxStatPos_[coreIdx]; }

    // DFX
    void DfxProcAfterFinishTask(int coreIdx, uint64_t taskId);
public:
    uint64_t GetTaskStartTime() {
        return task_start_time_;
    }
    uint64_t GetTaskEndTime() {
        return task_end_time_;
    }
private:
    bool isFirstTaskSend_{true};
    bool firstLock[AICORE_TYPE_NUM]{true,true};
    uint64_t task_start_time_{UINT64_MAX};
    uint64_t task_end_time_{0};
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int aicStart_{0};
    int aicpuNum_{npu::tile_fwk::dynamic::MAX_SCHEDULE_AICPU_NUM};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    uint64_t procAicCoreFunctionCnt_{0};
    uint64_t procAivCoreFunctionCnt_{0};
    uint64_t procAicpuFunctionCnt_{0};
    int64_t *regAddrs_{nullptr};
    int64_t sharedBuffer_{0};
    DeviceTask *curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};
    AicpuTaskManager &aicpuTaskManager_;

    std::array<uint64_t, MAX_AICORE_NUM> runningIds_;
    std::array<uint64_t, MAX_AICORE_NUM> pendingIds_;

    /* 低32位任务 存储的dfx状态信息， 乒乓存储 0 or 1,
       高32位任务 存储的dfx状态信息， 乒乓存储 2 or 3
    */
    std::array<int, MAX_AICORE_NUM> taskDfxStatPos_;

    std::array<int, MAX_AICORE_NUM> blockIdToPhyCoreId_;
    std::array<volatile uint64_t *, MAX_AICORE_NUM> readyRegQueues_;
    std::array<volatile uint64_t *, MAX_AICORE_NUM> finishRegQueues_;
    std::array<KernelArgs *, MAX_AICORE_NUM> args_;

    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> taskQueue_;

    /* prepare aicore ready task list */
    StaticReadyCoreFunctionQueue *readyAicCoreFunctionQue_{nullptr};
    StaticReadyCoreFunctionQueue *readyAivCoreFunctionQue_{nullptr};

    AiCoreProf prof_;
    AicoreDump aicoreDump_;
    int64_t dotStatus_{0};
    uint64_t waitTaskCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t corePendReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t coreRunReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t runReadyCoreIdx_[AICORE_TYPE_NUM][MAX_MANAGER_AIV_NUM];
    uint32_t lastPendReadyCoreIdx_[AICORE_TYPE_NUM]{0,0};
    uint64_t reSolveHubCnt_{0};

    uint64_t readyIds[AICORE_TYPE_NUM][READY_ID_FIX_CACHE_NUM];
    uint64_t readyCount[AICORE_TYPE_NUM]{0,0};
    std::vector<uint64_t> readyIdsExtend[AICORE_TYPE_NUM];

    uint32_t sendCnt_[AICORE_TYPE_NUM]{0,0};

    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];

    bool isNeedWriteRegForFastPath_{true};
    uint32_t regSprDataMainBase_{REG_SPR_DATA_MAIN_BASE};
    uint32_t regSprCond_{REG_SPR_COND};

    friend class AiCoreProf;
};
} // namespace npu::tile_fwk
