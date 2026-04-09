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
#include "aicore_constants.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/tilefwk/core_func_data.h"
#include "aicore_constants.h"

namespace npu::tile_fwk::dynamic {

struct SchDeviceTaskContext;
using SendTaskToAiCoreFunc = std::function<void(struct SchDeviceTaskContext* devCtx, CoreType type, int coreIdx, uint64_t newTask)>;
using AddReadyCoreIdxFunc = std::function<void(int coreIdx, int type)>;

enum class MixResourceType { MIX_UNKNOWN = 0, MIX_1C1V = 1, MIX_1C2V = 2 };

enum class DieId { DIE_0 = 0, DIE_1 = 1, DIE_MIX = 2, DIE_UNKNOW };

inline void WrapInfoQueueLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void WrapInfoQueueUnLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

inline uint32_t GetTaskNumByMixResType(uint8_t mixType)
{
    switch (mixType) {
        case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
            return 2;
        case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
            return 3;
        default:
            return 0;
    }
}

inline bool IsMixTaskFinish(WrapInfo* wrapInfo)
{
    switch (wrapInfo->mixResourceType) {
        case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
            return wrapInfo->tasklist[0] == AICORE_TASK_STOP && wrapInfo->tasklist[1] == AICORE_TASK_STOP;
        case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
            return wrapInfo->tasklist[0] == AICORE_TASK_STOP && wrapInfo->tasklist[1] == AICORE_TASK_STOP &&
                   wrapInfo->tasklist[2] == AICORE_TASK_STOP;
        default:
            DEV_ERROR(
                DevCommonErr::PARAM_INVALID, "#sche.wrap.invalid_mode: illegal mixType: %hhu\n",
                wrapInfo->mixResourceType);
            return false;
    }
}

#define RETURN_NULL_IF_NOT(val) \
    if (!val) {                 \
        return;                 \
    }

#define RETURN_RET_IF_NOT(val, ret) \
    if (!val) {                     \
        return ret;                 \
    }

class WrapManager {
public:
    ~WrapManager() {};
    WrapManager() {};

    SchDeviceTaskContext* schDevTaskCtx{nullptr};
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

    uint8_t* coreIdxPosition_{nullptr};
    bool* wrapCoreAvail_{nullptr};
    AddReadyCoreIdxFunc AddReadyCoreIdx{nullptr};

    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    // Queue managed by each thread, elem is wrapInfo's addr
    StaticReadyCoreFunctionQueue wrapQueueForThread_{0, 0, nullptr, 0};
    uint32_t* wrapTasklist_{nullptr};
    SendTaskToAiCoreFunc SendTaskToAiCore;
    bool isOpenMixSche{false};
    ArchInfo archInfo;

    // for die-to-die shchedule
    ReadyCoreFunctionQueue* readyDieAicFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* readyDieAivFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAicFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAivFunctionQue_{nullptr};

    inline void InitDeviceInfo(DeviceArgs* deviceArgs, int schedIdx)
    {
        archInfo = deviceArgs->archInfo;
        InitDieMaxCpuId(static_cast<int>(deviceArgs->scheCpuNum));
        InitDieId(schedIdx);
    }

    inline void InitDieMaxCpuId(int scheCpuNum)
    {
        curDie0MaxCpuId_ = scheCpuNum >> 1;
        // In odd scenes, scheCpuIdx = curDie0MaxCpuId_ is DIE_MIX, else is DIE_1
        curDie1StartCpuId_ = (scheCpuNum & 1) ? curDie0MaxCpuId_ + 1 : curDie0MaxCpuId_;
    }

    inline void InitDieId(int schedIdx)
    {
        if (schedIdx < curDie0MaxCpuId_) {
            dieId_ = DieId::DIE_0;
        } else if (schedIdx >= curDie1StartCpuId_) {
            dieId_ = DieId::DIE_1;
        } else {
            dieId_ = DieId::DIE_MIX;
        }
    }

    inline void GetDieSchedIdRange(int& schedStart, int& schedEnd, int scheCpuNum)
    {
        if (dieId_ == DieId::DIE_0) {
            schedStart = 0;
            schedEnd = curDie0MaxCpuId_;
        } else if (dieId_ == DieId::DIE_1) {
            schedStart = curDie1StartCpuId_;
            schedEnd = scheCpuNum;
        }
    }

    inline DieId GetDieId() { return dieId_; }

    inline void RemoveMixReadyCoreIdx(int coreIdx, int type)
    {
        uint32_t tail = --coreRunReadyCnt_[type];
        uint8_t pos = coreIdxPosition_[coreIdx];
        if (pos != tail) {
            runReadyCoreIdx_[type][pos] = runReadyCoreIdx_[type][tail];
            coreIdxPosition_[runReadyCoreIdx_[type][pos]] = pos;
        }
        coreIdxPosition_[coreIdx] = INVALID_COREIDX_POSITION;
        corePendReadyCnt_[type]--;
    }

    inline void Init(
        SchDeviceTaskContext* devTaskctx, DeviceTask* curDevTask, uint32_t* coreRunReadyCnt,
        uint32_t* runReadyCoreIdxZero, uint32_t* runReadyCoreIdxOne,
        uint32_t* corePendReadyCnt, uint32_t* pendingIds, uint32_t* runningIds, int aicValidNum,
        uint8_t* coreIdxPosition, bool* wrapCoreAvail, SendTaskToAiCoreFunc func,
        AddReadyCoreIdxFunc addReadyCoreIdxFunc)
    {
        if (archInfo != ArchInfo::DAV_3510) {
            return;
        }
        schDevTaskCtx = devTaskctx;
        isOpenMixSche = curDevTask->mixTaskData.wrapIdNum > 0;
        curDevTask_ = curDevTask;
        coreRunReadyCnt_ = coreRunReadyCnt;
        runReadyCoreIdx_[CORE_IDX_AIV] = runReadyCoreIdxZero;
        runReadyCoreIdx_[CORE_IDX_AIC] = runReadyCoreIdxOne;
        corePendReadyCnt_ = corePendReadyCnt;
        pendingIds_ = pendingIds;
        runningIds_ = runningIds;

        aicValidNum_ = aicValidNum;
        coreIdxPosition_ = coreIdxPosition;
        wrapCoreAvail_ = wrapCoreAvail;
        SendTaskToAiCore = func;
        AddReadyCoreIdx = addReadyCoreIdxFunc;
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue*>(curDevTask_->mixTaskData.readyWrapCoreFunctionQue);

        wrapQueueForThread_.head = 0;
        wrapQueueForThread_.tail = 0;
        wrapQueueForThread_.elem =
            curDevTask_->mixTaskData.wrapIdNum == 0 ?
                nullptr :
                static_cast<uint64_t*>(malloc(curDevTask_->mixTaskData.wrapIdNum * sizeof(uint64_t)));
        SetDieReadyQueue(curDevTask->dieReadyFunctionQue);

        selectReadyDieAicFunctionQue_ =
            GetDieReadyQueue(
                CoreType::AIC, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAicCoreFunctionQue));
        selectReadyDieAivFunctionQue_ =
            GetDieReadyQueue(
                CoreType::AIV, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAivCoreFunctionQue));
    }

    inline void Deinit()
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        if (wrapQueueForThread_.elem != nullptr) {
            free(wrapQueueForThread_.elem);
            wrapQueueForThread_.elem = nullptr;
        }
    }

    inline bool GetIsMixarch() { return archInfo == ArchInfo::DAV_3510; }

    inline uint32_t GetAvailableWrapCoreNum(WrapInfo* wrapTasks[], uint32_t maxTaskCnt)
    {
        uint32_t validReadyCnt = 0, idx = 0;
        uint32_t aicReadyCnt = coreRunReadyCnt_[CORE_IDX_AIC];
        for (uint32_t taskIdx = 0; taskIdx < maxTaskCnt && idx < aicReadyCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrapTasks[taskIdx];
            uint32_t* aicoreIdxList = wrapInfo->aicoreIdxList;
            uint8_t mixType = wrapInfo->mixResourceType;
            switch (wrapInfo->mixResourceType) {
                case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
                    while (idx < aicReadyCnt) {
                        uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
                        uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
                        idx++;
                        if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION) {
                            CheckCoreIdxInitStatus(aicIdx);
                            CheckCoreIdxInitStatus(aivIdx0);
                            aicoreIdxList[WRAP_IDX_AIC] = aicIdx;
                            aicoreIdxList[WRAP_IDX_AIV0] = aivIdx0;
                            aicoreIdxList[WRAP_IDX_AIV1] = aivIdx0;
                            validReadyCnt++;
                            break;
                        }
                    }
                    break;
                case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
                    while (idx < aicReadyCnt) {
                        uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
                        uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
                        uint32_t aivIdx1 = aivIdx0 + 1;
                        idx++;
                        if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION &&
                            coreIdxPosition_[aivIdx1] != INVALID_COREIDX_POSITION) {
                            CheckCoreIdxInitStatus(aicIdx);
                            CheckCoreIdxInitStatus(aivIdx0);
                            CheckCoreIdxInitStatus(aivIdx1);
                            aicoreIdxList[WRAP_IDX_AIC] = aicIdx;
                            aicoreIdxList[WRAP_IDX_AIV0] = aivIdx0;
                            aicoreIdxList[WRAP_IDX_AIV1] = aivIdx1;
                            validReadyCnt++;
                            break;
                        }
                    }
                    break;
                default:
                    DEV_ERROR(DevCommonErr::PARAM_INVALID, "#sche.wrap.invalid_mode: illegal mixType: %u\n", mixType);
                    break;
            }
        }
        return validReadyCnt;
    }

    inline void UpdateWrapQueueAndRmvCoreIdx(WrapInfo* wrapTasks[], uint32_t taskCount)
    {
        for (uint32_t taskIdx = 0; taskIdx < taskCount; taskIdx++) {
            WrapInfo* wrapInfo = wrapTasks[taskIdx];
            wrapQueueForThread_.elem[wrapQueueForThread_.tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            uint32_t* aicoreIdxList = wrapInfo->aicoreIdxList;
            uint8_t mixType = wrapInfo->mixResourceType;
            switch (mixType) {
                case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
                    RemoveMixReadyCoreIdx(aicoreIdxList[WRAP_IDX_AIC], static_cast<int>(CoreType::AIC));
                    RemoveMixReadyCoreIdx(aicoreIdxList[WRAP_IDX_AIV0], static_cast<int>(CoreType::AIV));
                    wrapCoreAvail_[aicoreIdxList[WRAP_IDX_AIC]] = false;
                    wrapCoreAvail_[aicoreIdxList[WRAP_IDX_AIV0]] = false;
                    break;
                case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
                    RemoveMixReadyCoreIdx(aicoreIdxList[WRAP_IDX_AIC], static_cast<int>(CoreType::AIC));
                    RemoveMixReadyCoreIdx(aicoreIdxList[WRAP_IDX_AIV0], static_cast<int>(CoreType::AIV));
                    RemoveMixReadyCoreIdx(aicoreIdxList[WRAP_IDX_AIV1], static_cast<int>(CoreType::AIV));
                    wrapCoreAvail_[aicoreIdxList[WRAP_IDX_AIC]] = false;
                    wrapCoreAvail_[aicoreIdxList[WRAP_IDX_AIV0]] = false;
                    wrapCoreAvail_[aicoreIdxList[WRAP_IDX_AIV1]] = false;
                    break;
                default:
                    DEV_ERROR(DevCommonErr::PARAM_INVALID, "#sche.wrap.invalid_mode: illegal mixType: %u\n", mixType);
                    break;
            }
            DEV_VERBOSE_DEBUG(
                "add wrapInfo, aicCoreIdx = %u, aivCoreIdxZero = %u, aivCoreIdxOne = %u, mixResourceType = %hhu",
                aicoreIdxList[WRAP_IDX_AIC], aicoreIdxList[WRAP_IDX_AIV0], aicoreIdxList[WRAP_IDX_AIV1], mixType);
        }
    }

    inline void CheckCoreIdxInitStatus(uint32_t coreIdx)
    {
        DEV_IF_VERBOSE_DEBUG
        {
            if (pendingIds_[coreIdx] != AICORE_TASK_INIT || runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DEV_ERROR(
                    CtrlErr::TASK_STATS_ABNORMAL,
                    "#sche.task.run.wrap.stats: core[%u]: pendingId=%x, runningId=%x, is illegal!", coreIdx,
                    pendingIds_[coreIdx], runningIds_[coreIdx]);
            }
        }
    }

    inline void AddRunReadyCoreIdxForWrap(uint32_t coreIdx, CoreType coreType)
    {
        CheckCoreIdxInitStatus(coreIdx);
        AddReadyCoreIdx(coreIdx, static_cast<int>(coreType));
        uint32_t idx = (coreType == CoreType::AIC) ? CORE_IDX_AIC : CORE_IDX_AIV;
        corePendReadyCnt_[idx]++;
    }

    inline void UpdateWrapQueueForThread()
    {
        // when readyWrapCoreFunctionQueue has valid value and has available wrapCore
        // move wrapId from readyWrapCoreFunctionQueue to wrapQueueForThread, and occpy wrapCore

        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        if (tail - head == 0 || coreRunReadyCnt_[CORE_IDX_AIC] == 0) {
            return;
        }

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);

#ifdef NO_EARLY_SEND_TASK
        uint32_t taskCount = 0;
        for (uint32_t i = readyWrapCoreFunctionQue_->head; i < readyWrapCoreFunctionQue_->tail; i++) {
            WrapInfo* info = &readyWrapCoreFunctionQue_->elem[i];
            bool isC1V1Ready =
                (info->mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C1V) &&
                 info->tasklist[0] != AICORE_TASK_INIT && info->tasklist[1] != AICORE_TASK_INIT);
            bool isC1V2Ready =
                (info->mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V) &&
                 info->tasklist[0] != AICORE_TASK_INIT && info->tasklist[1] != AICORE_TASK_INIT &&
                 info->tasklist[2] != AICORE_TASK_INIT); // 2:v1 index
            if (isC1V1Ready || isC1V2Ready) {
                std::swap(readyWrapCoreFunctionQue_->elem[i], readyWrapCoreFunctionQue_->elem[taskCount]);
                taskCount++;
            }
        }
#else
        head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = tail - head;
#endif
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return;
        }

        constexpr uint32_t maxTransTaskCnt = 5u;
        WrapInfo* localTasks[maxTransTaskCnt];
        uint32_t maxTaskCnt = taskCount > maxTransTaskCnt ? maxTransTaskCnt : taskCount;
        for (uint32_t i = 0; i < maxTaskCnt; i++) {
            localTasks[i] = &readyWrapCoreFunctionQue_->elem[head++];
        }
        uint32_t validTaskCnt = GetAvailableWrapCoreNum(localTasks, maxTaskCnt);
        __atomic_fetch_add(&readyWrapCoreFunctionQue_->head, validTaskCnt, std::memory_order_release);
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        UpdateWrapQueueAndRmvCoreIdx(localTasks, validTaskCnt);
    }

    inline void DispatchMixCoreTask()
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        UpdateWrapQueueForThread();
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            WrapInfo* wrapInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_.elem[idx]);
            uint32_t taskNum = GetTaskNumByMixResType(wrapInfo->mixResourceType);
            for (uint32_t taskIdx = 0; taskIdx < taskNum; taskIdx++) {
                uint32_t taskId = wrapInfo->tasklist[taskIdx];
                // 此处可能一个Task准备下发，另一个还没初始化。另一个准备下发时，前面一个已经结束
                if (taskId == AICORE_TASK_DISTRIBUTED || taskId == AICORE_TASK_INIT || taskId == AICORE_TASK_STOP) {
                    continue;
                }
                CoreType coreType = taskIdx == WRAP_IDX_AIC ? CoreType::AIC : CoreType::AIV;
                DEV_VERBOSE_DEBUG("try to send wrapId[%u]'s taskIdx[%u] taskId[%u]", wrapInfo->wrapId, taskIdx, taskId);
                SendTaskToAiCore(schDevTaskCtx, coreType, wrapInfo->aicoreIdxList[taskIdx], taskId);
                wrapInfo->tasklist[taskIdx] = AICORE_TASK_DISTRIBUTED;
            }
        }
    }

    int32_t GetWrapId(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcId]);
        if (opWrapList[opIndex] != -1) {
            return MakeMixWrapID(funcId, opWrapList[opIndex]);
        } else {
            return -1;
        }
    }

    int32_t GetWrapVecId(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].wrapVecId;
    }

    CoreType GetCoreType(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return static_cast<CoreType>(cceBinary[callList[opIndex]].coreType);
    }

    uint8_t GetMixResourceType(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].mixResourceType;
    }

    inline int32_t GetMixTaskIdx(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        auto coreType = static_cast<CoreType>(cceBinary[callList[opIndex]].coreType);
        auto wrapVecId = cceBinary[callList[opIndex]].wrapVecId;
        if (coreType == CoreType::AIC) {
            return WRAP_IDX_AIC;
        } else {
            return wrapVecId == 1 ? WRAP_IDX_AIV1 : WRAP_IDX_AIV0;
        }
    }

    bool IsBindedWrapId(uint32_t taskId)
    {
        RETURN_RET_IF_NOT(isOpenMixSche, false);
        if (GetWrapId(taskId) == -1) {
            return false;
        }
        return true;
    }

    inline void PushTaskToTasklist(uint32_t wrapId, uint32_t taskId, uint32_t taskIdx)
    {
        WrapInfo* wrapInfo = nullptr;
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
            wrapInfo->mixResourceType = GetMixResourceType(taskId);
            for (uint32_t i = 0; i < MAX_WRAP_TASK_NUM; i++) {
                wrapInfo->tasklist[i] = AICORE_TASK_INIT;
                wrapInfo->aicoreIdxList[i] = 0;
            }
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->tail, 1, std::memory_order_release);
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        wrapInfo->tasklist[taskIdx] = taskId;
    }

    inline void ResolveDepForMixCore(uint32_t taskId)
    {
        // resolve dep, if has available core, send task directly, else call PushTaskToTasklist, try to send task in
        // next loop
        uint32_t wrapId = GetWrapId(taskId);
        DEV_VERBOSE_DEBUG("taskId = %u, wrapId = %u", taskId, wrapId);

#ifdef NO_EARLY_SEND_TASK
        PushTaskToTasklist(wrapId, taskId, GetMixTaskIdx(taskId));
        return;
#endif
        WrapInfo* wrapInfo = nullptr;
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            if (reinterpret_cast<WrapInfo*>(wrapQueueForThread_.elem[idx])->wrapId == wrapId) {
                wrapInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_.elem[idx]);
                break;
            }
        }

        int32_t taskIdx = GetMixTaskIdx(taskId);

        if (wrapInfo == nullptr) { // the wrap is not in this thread
            DEV_VERBOSE_DEBUG("the wrapId %u is not in this thread, push taskId %u to tasklist", wrapId, taskId);
            PushTaskToTasklist(wrapId, taskId, taskIdx);
            return;
        }

        CoreType coreType = taskIdx == WRAP_IDX_AIC ? CoreType::AIC : CoreType::AIV;
        DEV_VERBOSE_DEBUG("directly send taskId %u to core, core type idx: %d", taskId, taskIdx);
        SendTaskToAiCore(schDevTaskCtx, coreType, wrapInfo->aicoreIdxList[taskIdx], taskId);
    }

    inline void UpdateFinishIdForMixCore(uint32_t finishId)
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        int32_t id = GetWrapId(finishId);
        if (id == -1) {
            return;
        }
        uint32_t wrapId = id;
        WrapInfo* wrapInfo = nullptr;
        uint32_t wrapIdx = 0;
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            if (reinterpret_cast<WrapInfo*>(wrapQueueForThread_.elem[idx])->wrapId == wrapId) {
                wrapInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_.elem[idx]);
                wrapIdx = idx;
                break;
            }
        }

        if (wrapInfo == nullptr) {
            DEV_ERROR(
                DevCommonErr::NULLPTR, "#sche.task.run.wrap.dep.resolve: cant find wrapInfo in wrapQueueForThread!");
            return;
        }

        int32_t taskIdx = GetMixTaskIdx(finishId);
        wrapInfo->tasklist[taskIdx] = AICORE_TASK_STOP;

        CoreType coreType = (taskIdx == WRAP_IDX_AIC) ? CoreType::AIC : CoreType::AIV;
        AddRunReadyCoreIdxForWrap(wrapInfo->aicoreIdxList[taskIdx], coreType); // free wrap core
        wrapCoreAvail_[wrapInfo->aicoreIdxList[taskIdx]] = true;

        if (IsMixTaskFinish(wrapInfo)) { // all tasks for this wrap finish
            DEV_VERBOSE_DEBUG("wrapId %u 's all tasks finish, release wrapcore", wrapId);
            std::swap(wrapQueueForThread_.elem[wrapIdx], wrapQueueForThread_.elem[--wrapQueueForThread_.tail]);
        }
    }

    // for die-to-die schedule
    inline void SetDieReadyQueue(const struct DieReadyQueueData dieReadyFunctionQue)
    {
        for (size_t i = 0; i < DIE_NUM; i++) {
            readyDieAivFunctionQue_[i] =
                reinterpret_cast<ReadyCoreFunctionQueue*>(dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
            readyDieAicFunctionQue_[i] =
                reinterpret_cast<ReadyCoreFunctionQueue*>(dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        }
    }

    inline ReadyCoreFunctionQueue* GetDieReadyQueue(CoreType type, ReadyCoreFunctionQueue* defaultReadyQue)
    {
        if (!GetIsMixarch() || dieId_ == DieId::DIE_MIX || dieId_ == DieId::DIE_UNKNOW) {
            return defaultReadyQue;
        }
        size_t dieIndex = static_cast<size_t>(dieId_);
        ReadyCoreFunctionQueue* dieReadyQueue = nullptr;
        switch (type) {
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

    ReadyCoreFunctionQueue* GetDieReadyAicQue() { return selectReadyDieAicFunctionQue_; }
    ReadyCoreFunctionQueue* GetDieReadyAivQue() { return selectReadyDieAivFunctionQue_; }
};
} // namespace npu::tile_fwk::dynamic
