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
 * \file wrap_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/tilefwk/core_func_data.h"
namespace npu::tile_fwk::dynamic {

using SendTaskToAiCoreFunc = std::function<void(CoreType type, int coreIdx, uint64_t newTask)>;

enum class MixResourceType {
    MIX_UNKNOWN = 0,
    MIX_1C1V = 1,
    MIX_1C2V = 2
};

enum class DieId {
    DIE_0 = 0,
    DIE_1 = 1,
    DIE_MIX = 2,
    DIE_UNKNOW
};

inline void WrapInfoQueueLock(WrapInfoQueue* rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void WrapInfoQueueUnLock(WrapInfoQueue* rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

#define RETURN_NULL_IF_NOT(val) \
    if (!val) {  \
        return;  \
    }

#define RETURN_RET_IF_NOT(val, ret) \
    if (!val) {  \
        return ret;  \
    }

class WrapManager {
public:
    ~WrapManager(){};
    WrapManager(){};

    DeviceTask* curDevTask_;
    uint32_t* coreRunReadyCnt_;
    uint32_t* runReadyCoreIdx_[AICORE_TYPE_NUM];
    uint32_t* corePendReadyCnt_;
    uint32_t* pendingIds_;
    uint32_t* runningIds_;

    int aicValidNum_{0};
    int curDie0MaxCpuId_{0};
    int curDie1StartCpuId_{0};
    DieId dieId_{DieId::DIE_MIX};

    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    // Queue managed by each thread, elem is wrapInfo's addr
    StaticReadyCoreFunctionQueue wrapQueueForThread_{0, 0, nullptr, 0};
    uint32_t* wrapTasklist_{nullptr};
    uint32_t wrapCoreStatus_[MAX_AICORE_NUM]{0};
    SendTaskToAiCoreFunc SendTaskToAiCore;
    bool isOpenMixSche {false};
    ArchInfo archInfo;

    // for die-to-die shchedule
    ReadyCoreFunctionQueue* readyDieAicFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* readyDieAivFunctionQue_[DIE_NUM] = {nullptr};

    inline void InitDeviceInfo(DeviceArgs *deviceArgs, int schedIdx) {
        archInfo = deviceArgs->archInfo;
        InitDieMaxCpuId(static_cast<int>(deviceArgs->scheCpuNum));
        InitDieId(schedIdx);
    }

    inline void InitDieMaxCpuId(int scheCpuNum) {
        curDie0MaxCpuId_ = scheCpuNum >> 1;
        // In odd scenes, scheCpuIdx = curDie0MaxCpuId_ is DIE_MIX, else is DIE_1
        curDie1StartCpuId_ = (scheCpuNum & 1) ? curDie0MaxCpuId_ + 1 : curDie0MaxCpuId_;
    }

    inline void InitDieId(int schedIdx) {
        if (schedIdx < curDie0MaxCpuId_) {
            dieId_ = DieId::DIE_0;
        } else if (schedIdx >= curDie1StartCpuId_) {
            dieId_ = DieId::DIE_1;
        } else {
            dieId_ = DieId::DIE_MIX;
        }
    }

    inline void GetDieSchedIdRange(int &schedStart, int &schedEnd, int scheCpuNum) {
        if (dieId_ == DieId::DIE_0) {
            schedStart = 0;
            schedEnd = curDie0MaxCpuId_;
        } else if (dieId_ == DieId::DIE_1) {
            schedStart = curDie1StartCpuId_;
            schedEnd = scheCpuNum;
        }
    }

    inline DieId GetDieId() {
        return dieId_;
    }

    inline void Init(DeviceTask* curDevTask, uint32_t* coreRunReadyCnt, uint32_t* runReadyCoreIdxZero,
        uint32_t* runReadyCoreIdxOne, uint32_t* corePendReadyCnt, uint32_t* pendingIds, uint32_t* runningIds,
        int aicValidNum, SendTaskToAiCoreFunc func) {

        if (archInfo != ArchInfo::DAV_3510) {
            return;
        }
        isOpenMixSche = curDevTask->mixTaskData.wrapIdNum > 0;
        curDevTask_ = curDevTask;
        coreRunReadyCnt_ = coreRunReadyCnt;
        runReadyCoreIdx_[CORE_IDX_AIV] = runReadyCoreIdxZero;
        runReadyCoreIdx_[CORE_IDX_AIC] = runReadyCoreIdxOne;

        corePendReadyCnt_ = corePendReadyCnt;
        pendingIds_ = pendingIds;
        runningIds_ = runningIds;

        aicValidNum_ = aicValidNum;
        SendTaskToAiCore = func;
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue *>(curDevTask_->mixTaskData.readyWrapCoreFunctionQue);
        wrapTasklist_ = reinterpret_cast<uint32_t *>(curDevTask_->mixTaskData.wrapTasklist);

        wrapQueueForThread_.head = 0;
        wrapQueueForThread_.tail = 0;
        wrapQueueForThread_.elem = curDevTask_->mixTaskData.wrapIdNum == 0 ? nullptr :
            static_cast<uint64_t *>(malloc(curDevTask_->mixTaskData.wrapIdNum * sizeof(uint64_t)));
        SetDieReadyQueue(curDevTask->dieReadyFunctionQue);
    }

    inline void Deinit() {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        if (wrapQueueForThread_.elem != nullptr) {
            free(wrapQueueForThread_.elem);
            wrapQueueForThread_.elem = nullptr;
        }
        memset_s(wrapCoreStatus_, sizeof(wrapCoreStatus_), 0, sizeof(wrapCoreStatus_));
    }

    inline bool GetIsMixarch() {
        return archInfo == ArchInfo::DAV_3510;
    }

    inline bool GetWrapCoreAvailable(int coreIdx) {
        // for not DAV_3510 or wrapIdNum == 0 situation, MixSche is closed, so wrapCoreAvailable always return true
        RETURN_RET_IF_NOT(isOpenMixSche, true);
        return wrapCoreStatus_[coreIdx] == 0;
    }

    inline uint32_t GetAvailableCoreIdx(MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        if (mixType == MixResourceType::MIX_1C1V) {
            for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
                for (uint32_t j = 0; j < coreRunReadyCnt_[CORE_IDX_AIV]; j++) {
                    uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][i];
                    uint32_t aivIdx = runReadyCoreIdx_[CORE_IDX_AIV][j];
                    if (aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ == aivIdx) {
                        return aicIdx;
                    }
                }
            }
            return INVALID_CORE_IDX;
        }

        for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
            for (uint32_t j = 0; j < coreRunReadyCnt_[CORE_IDX_AIV]; j++) {
                for (uint32_t k = 0; k < coreRunReadyCnt_[CORE_IDX_AIV]; k++) {
                    uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][i];
                    uint32_t aivIdx0 = runReadyCoreIdx_[CORE_IDX_AIV][j];
                    uint32_t aivIdx1 = runReadyCoreIdx_[CORE_IDX_AIV][k];
                    if (aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ == aivIdx0 && aivIdx0 + 1 == aivIdx1) {
                        return aicIdx;
                    }
                }
            }
        }
        return INVALID_CORE_IDX;
    }

    inline void RemoveRunReadyCoreIdxForWrap(uint32_t coreIdx, MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        coreRunReadyCnt_[CORE_IDX_AIC]--;
        corePendReadyCnt_[CORE_IDX_AIC]--;
        // if coreIdx is at the tail of runReadyCoreIdx_, no processing is need, simply cnt--
        if (runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]] != coreIdx) {
            // if coreIdx isnt at the tail of runReadyCoreIdx_, replace it by tail data
            for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
                if (runReadyCoreIdx_[CORE_IDX_AIC][i] == coreIdx) {
                    // swap tail data with coreIdx
                    runReadyCoreIdx_[CORE_IDX_AIC][i] = runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]];
                    runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]] = coreIdx;
                }
            }
        }

        coreRunReadyCnt_[CORE_IDX_AIV]--;
        corePendReadyCnt_[CORE_IDX_AIV]--;
        uint32_t aivIdx0 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        if (runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] != aivIdx0) {
            for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIV]; i++) {
                if (runReadyCoreIdx_[CORE_IDX_AIV][i] == aivIdx0) {
                    runReadyCoreIdx_[CORE_IDX_AIV][i] = runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]];
                    runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] = aivIdx0;
                }
            }
        }

        CheckCoreIdxInitStatus(coreIdx);
        CheckCoreIdxInitStatus(aivIdx0);

        if (mixType != MixResourceType::MIX_1C1V) {
            coreRunReadyCnt_[CORE_IDX_AIV]--;
            corePendReadyCnt_[CORE_IDX_AIV]--;
            uint32_t aivIdx1 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1;
            if (runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] != aivIdx1) {
                for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIV]; i++) {
                    if (runReadyCoreIdx_[CORE_IDX_AIV][i] == aivIdx1) {
                        runReadyCoreIdx_[CORE_IDX_AIV][i] = runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]];
                        runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] = aivIdx1;
                    }
                }
            }
            CheckCoreIdxInitStatus(aivIdx1);
            DEV_VERBOSE_DEBUG("remove coreIdx %u  %u  %u", coreIdx, aivIdx0, aivIdx1);
        } else {
            DEV_VERBOSE_DEBUG("remove coreIdx %u  %u", coreIdx, aivIdx0);
        }
    }

    inline void CheckCoreIdxInitStatus(uint32_t coreIdx) {
        DEV_IF_VERBOSE_DEBUG {
            if (pendingIds_[coreIdx] != AICORE_TASK_INIT || runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DEV_ERROR(CtrlErr::TASK_STATS_ABNORMAL, "#sche.task.run.wrap.stats: core[%u]: pendingId=%x, runningId=%x, is illegal!", coreIdx, pendingIds_[coreIdx], runningIds_[coreIdx]);
            }
        }
    }

    inline void AddRunReadyCoreIdxForWrap(uint32_t coreIdx, MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        uint32_t aivIdx0 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]++] = coreIdx;
        runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]++] = aivIdx0;
        corePendReadyCnt_[CORE_IDX_AIC]++;
        corePendReadyCnt_[CORE_IDX_AIV]++;
        CheckCoreIdxInitStatus(coreIdx);
        CheckCoreIdxInitStatus(aivIdx0);
        if (mixType != MixResourceType::MIX_1C1V) {
            uint32_t aivIdx1 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1;
            CheckCoreIdxInitStatus(aivIdx1);
            runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]++] = aivIdx1;
            DEV_VERBOSE_DEBUG("add coreIdx %u  %u  %u", coreIdx, coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_,
                coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1);
            corePendReadyCnt_[CORE_IDX_AIV]++;
        } else {
            DEV_VERBOSE_DEBUG("add coreIdx %u  %u", coreIdx, coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_);
        }
    }

    inline void UpdateWrapQueueForThread() {
        // when readyWrapCoreFunctionQueue has valid value and has available wrapCore
        // move wrapId from readyWrapCoreFunctionQueue to wrapQueueForThread, and occpy wrapCore
        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = tail - head;
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return;
        }

        while (taskCount-- > 0) {
            WrapInfo *wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->head];
            uint32_t wrapId = wrapInfo->wrapId;
            MixResourceType mixType = static_cast<MixResourceType>(wrapInfo->mixResourceType);

            uint32_t avaiCoreIdx = GetAvailableCoreIdx(mixType);
            if (avaiCoreIdx == INVALID_CORE_IDX) {
                DEV_VERBOSE_DEBUG("no available wrap core.");
                WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
                return;
            }

            DEV_VERBOSE_DEBUG("move wrapId[%u] to wrapQueueForThread. occupy coreIdx[%u]", wrapId, avaiCoreIdx);
            wrapQueueForThread_.elem[wrapQueueForThread_.tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->head, 1, std::memory_order_release);
            RemoveRunReadyCoreIdxForWrap(avaiCoreIdx, mixType);

            wrapInfo->aicCoreIdx = avaiCoreIdx;
            wrapInfo->aivCoreIdxZero = avaiCoreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
            wrapInfo->aivCoreIdxOne = wrapInfo->aivCoreIdxZero + (mixType != MixResourceType::MIX_1C1V ? 1 : 0);
            wrapCoreStatus_[wrapInfo->aicCoreIdx] = 1;
            wrapCoreStatus_[wrapInfo->aivCoreIdxZero] = 1;
            wrapCoreStatus_[wrapInfo->aivCoreIdxOne] = 1;
            DEV_VERBOSE_DEBUG("add wrapInfo, aicCoreIdx = %u, aivCoreIdxZero = %u, aivCoreIdxOne = %u, taskCnt = %u, mixResourceType = %u",
                wrapInfo->aicCoreIdx, wrapInfo->aivCoreIdxZero, wrapInfo->aivCoreIdxOne, wrapInfo->taskCnt, static_cast<uint32_t>(wrapInfo->mixResourceType));
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
    }

    inline void DispatchMixCoreTask() {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        UpdateWrapQueueForThread();
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            WrapInfo *wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
            std::vector<uint32_t> sendTaskIdx;
            ReadyQueueLock(&wrapInfo->tasklist);

            for (uint32_t taskIdx = wrapInfo->tasklist.head; taskIdx < wrapInfo->tasklist.tail; taskIdx++) {
                uint32_t taskId = wrapInfo->tasklist.elem[taskIdx];
                CoreType coreType = GetCoreType(taskId);
                DEV_VERBOSE_DEBUG("try to send wrapId[%u]'s taskIdx[%u] taskId[%u]", wrapInfo->wrapId, taskIdx, taskId);
                if (coreType == CoreType::AIC) {
                    SendTaskToAiCore(coreType, wrapInfo->aicCoreIdx, taskId);
                    sendTaskIdx.push_back(taskIdx);
                } else if (coreType == CoreType::AIV) {
                    int32_t wrapVecId = GetWrapVecId(taskId);
                    if (wrapVecId == 0 || wrapVecId == -1) {
                        SendTaskToAiCore(coreType, wrapInfo->aivCoreIdxZero, taskId);
                        sendTaskIdx.push_back(taskIdx);
                    } else if (wrapVecId == 1) {
                        SendTaskToAiCore(coreType, wrapInfo->aivCoreIdxOne, taskId);
                        sendTaskIdx.push_back(taskIdx);
                    }
                }
            }
            for (int32_t i = static_cast<int32_t>(sendTaskIdx.size()) - 1; i >= 0; i--) {
                std::swap(wrapInfo->tasklist.elem[sendTaskIdx[i]], wrapInfo->tasklist.elem[--wrapInfo->tasklist.tail]);
            }
            ReadyQueueUnLock(&wrapInfo->tasklist);
        }
    }

    int32_t GetWrapId(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcId]);
        if (opWrapList[opIndex] != -1) {
            return MakeMixWrapID(funcId, opWrapList[opIndex]);
        } else {
            return -1;
        }
    }

    uint32_t GetWrapTaskNum(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapTaskNumList = reinterpret_cast<uint32_t*>(dyntask->devTask.mixTaskData.opWrapTaskNumList[funcId]);
        return opWrapTaskNumList[opIndex];
    }

    int32_t GetWrapVecId(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].wrapVecId;
    }

    CoreType GetCoreType(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return static_cast<CoreType>(cceBinary[callList[opIndex]].coreType);
    }

    uint32_t GetMixResourceType(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].mixResourceType;
    }

    bool IsBindedWrapId(uint32_t taskId) {
        RETURN_RET_IF_NOT(isOpenMixSche, false);
        if (GetWrapId(taskId) == -1) {
            return false;
        }
        return true;
    }

    inline void PushTaskToTasklist(uint32_t wrapId, uint32_t taskId) {
        WrapInfo *wrapInfo = nullptr;
        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        for (uint32_t idx = 0; idx < readyWrapCoreFunctionQue_->tail; idx++) {
            if (readyWrapCoreFunctionQue_->elem[idx].wrapId == wrapId) {
                wrapInfo = &readyWrapCoreFunctionQue_->elem[idx];
                break;
            }
        }

        if (wrapInfo == nullptr) {
            // add a new wrapinfo
            wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail];
            wrapInfo->wrapId = wrapId;
            wrapInfo->aicCoreIdx = 0;
            wrapInfo->aivCoreIdxZero = 0;
            wrapInfo->aivCoreIdxOne = 0;
            wrapInfo->taskCnt = GetWrapTaskNum(taskId);
            wrapInfo->mixResourceType = GetMixResourceType(taskId);
            wrapInfo->tasklist.head = 0;
            wrapInfo->tasklist.tail = 0;
            wrapInfo->tasklist.lock = 0;
            wrapInfo->tasklist.capacity = wrapInfo->taskCnt;
            if (readyWrapCoreFunctionQue_->tail == 0) {
                wrapInfo->tasklist.elem = wrapTasklist_;
            } else {
                auto preQueue = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail - 1];
                wrapInfo->tasklist.elem = preQueue->tasklist.elem + preQueue->tasklist.capacity;
            }
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->tail, 1, std::memory_order_release);
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
        ReadyQueueLock(&wrapInfo->tasklist);
        wrapInfo->tasklist.elem[wrapInfo->tasklist.tail++] = taskId;
        ReadyQueueUnLock(&wrapInfo->tasklist);
    }

    inline void ResolveDepForMixCore(uint32_t taskId) {
        // resolve dep, if has available core, send task directly, else call PushTaskToTasklist, try to send task in next loop
        uint32_t wrapId = GetWrapId(taskId);
        DEV_VERBOSE_DEBUG("taskId = %u, wrapId = %u", taskId, wrapId);

        WrapInfo *wrapInfo = nullptr;
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            if (reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx])->wrapId == wrapId) {
                wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
                break;
            }
        }

        if (wrapInfo == nullptr) { // the wrap is not in this thread
            DEV_VERBOSE_DEBUG("the wrapId %u is not in this thread, push taskId %u to tasklist", wrapId, taskId);
            PushTaskToTasklist(wrapId, taskId);
            return;
        }

        // if the wrap is in this thread, try to send task directly
        if (GetCoreType(taskId) == CoreType::AIC) {
            DEV_VERBOSE_DEBUG("directly send taskId %u to cubecore", taskId);
            SendTaskToAiCore(CoreType::AIC, wrapInfo->aicCoreIdx, taskId);
            return;
        }

        if (GetCoreType(taskId) == CoreType::AIV) {
            int32_t wrapVecId = GetWrapVecId(taskId);
            if (wrapVecId == 0 || wrapVecId == -1) {
                DEV_VERBOSE_DEBUG("directly send taskId %u to veccore0", taskId);
                SendTaskToAiCore(CoreType::AIV, wrapInfo->aivCoreIdxZero, taskId);
                return;
            } else if (wrapVecId == 1) {
                DEV_VERBOSE_DEBUG("directly send taskId %u to veccore1", taskId);
                SendTaskToAiCore(CoreType::AIV, wrapInfo->aivCoreIdxOne, taskId);
                return;
            }
        }
        DEV_VERBOSE_DEBUG("there is no available core, push taskId %u to tasklist", taskId);
        PushTaskToTasklist(wrapId, taskId);
    }

    inline void UpdateFinishIdForMixCore(uint32_t finishId) {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        if (GetWrapId(finishId) == -1) {
            return;
        }
        uint32_t wrapId = GetWrapId(finishId);
        WrapInfo *wrapInfo = nullptr;
        uint32_t wrapIdx = 0;
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            if (reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx])->wrapId == wrapId) {
                wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
                wrapIdx = idx;
                break;
            }
        }

        if (wrapInfo == nullptr) {
            DEV_ERROR(DevCommonErr::NULLPTR, "#sche.task.run.wrap.dep.resolve: cant find wrapInfo in wrapQueueForThread!");
            return;
        }
        wrapInfo->taskCnt--;
        if (wrapInfo->taskCnt == 0) { // all tasks for this wrap finish
            DEV_VERBOSE_DEBUG("wrapId %u 's all tasks finish, release wrapcore", wrapId);
            AddRunReadyCoreIdxForWrap(wrapInfo->aicCoreIdx, static_cast<MixResourceType>(wrapInfo->mixResourceType)); // free wrap core
            wrapCoreStatus_[wrapInfo->aicCoreIdx] = 0;
            wrapCoreStatus_[wrapInfo->aivCoreIdxZero] = 0;
            wrapCoreStatus_[wrapInfo->aivCoreIdxOne] = 0;
            std::swap(wrapQueueForThread_.elem[wrapIdx], wrapQueueForThread_.elem[--wrapQueueForThread_.tail]);
        }
    }

    // for die-to-die schedule
    inline void SetDieReadyQueue(const struct DieReadyQueueData dieReadyFunctionQue) {
        for (size_t i = 0 ; i < DIE_NUM ; i++) {
           readyDieAivFunctionQue_[i] =  reinterpret_cast<ReadyCoreFunctionQueue *>(dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
           readyDieAicFunctionQue_[i] =  reinterpret_cast<ReadyCoreFunctionQueue *>(dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        }
    }

    inline ReadyCoreFunctionQueue* GetDieReadyQueue(CoreType type, ReadyCoreFunctionQueue* defaultReadyQue) {
        if (!GetIsMixarch() || dieId_ == DieId::DIE_MIX || dieId_ == DieId::DIE_UNKNOW) {
            return defaultReadyQue;
        }
        size_t dieIndex = static_cast<size_t>(dieId_);
        ReadyCoreFunctionQueue* dieReadyQueue = nullptr;
        switch(type) {
            case CoreType::AIC:
                dieReadyQueue = readyDieAicFunctionQue_[dieIndex];
                break;
            case CoreType::AIV:
                dieReadyQueue = readyDieAivFunctionQue_[dieIndex];
                break;
            default:
                break;
        }
        return (dieReadyQueue != nullptr) ? dieReadyQueue : defaultReadyQue;
    }
};
}