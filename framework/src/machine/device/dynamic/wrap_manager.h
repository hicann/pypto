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

namespace npu::tile_fwk::dynamic {

struct SchDeviceTaskContext;
using SendTaskToAiCoreFunc =
    std::function<void(struct SchDeviceTaskContext* devCtx, CoreType type, int coreIdx, uint64_t newTask)>;

enum class MixResourceType { MIX_UNKNOWN = 0, MIX_1C1V = 1, MIX_1C2V = 2 };
enum class DieId { DIE_0 = 0, DIE_1 = 1, DIE_MIX = 2, DIE_UNKNOWN };

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

#define RETURN_NULL_IF_NOT(val) do { if (!(val)) return; } while (0)
#define RETURN_RET_IF_NOT(val, ret) do { if (!(val)) return (ret); } while (0)

class WrapManager {
public:
    ~WrapManager() {}
    WrapManager() {}

    SchDeviceTaskContext* schDevTaskCtx{nullptr};
    DeviceTask* curDevTask_;
    uint8_t* coreRunReadyCnt_;
    uint8_t* runReadyCoreIdx_[AICORE_TYPE_NUM];
    uint8_t* corePendReadyCnt_;
    uint32_t* pendingIds_;
    uint32_t* runningIds_;
    uint32_t aicStart_;
    uint32_t aicEnd_;

    int aicValidNum_{0};
    int curDie0MaxCpuId_{0};
    int curDie1StartCpuId_{0};
    DieId dieId_{DieId::DIE_MIX};

    uint8_t* coreIdxPosition_{nullptr};
    bool* wrapCoreAvail_{nullptr};

    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    // Queue managed by each thread, elem is wrapInfo's addr
    StaticReadyCoreFunctionQueue* wrapQueueForThread_{nullptr};
    SendTaskToAiCoreFunc SendTaskToAiCore;
    bool isOpenMixSche{false};
    bool isMixPending_{false};
    ArchInfo archInfo;
    int schedIdx_;

    // for die-to-die shchedule
    ReadyCoreFunctionQueue* readyDieAicFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* readyDieAivFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAicFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAivFunctionQue_{nullptr};

    inline void InitDeviceInfo(DeviceArgs* deviceArgs, int schedIdx)
    {
        archInfo = deviceArgs->archInfo;
        isMixPending_ = deviceArgs->all1c2vMixTask;
        InitDieMaxCpuId(static_cast<int>(deviceArgs->scheCpuNum));
        InitDieId(schedIdx);
        schedIdx_ = schedIdx;
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
        SchDeviceTaskContext* devTaskctx, DeviceTask* curDevTask, uint8_t* coreRunReadyCnt,
        uint8_t* runReadyCoreIdxZero, uint8_t* runReadyCoreIdxOne, uint8_t* corePendReadyCnt, uint32_t* pendingIds,
        uint32_t* runningIds, int aicValidNum, uint8_t* coreIdxPosition, bool* wrapCoreAvail, uint32_t aicStart, uint32_t aicEnd, SendTaskToAiCoreFunc func)
    {
        if (archInfo != ArchInfo::DAV_3510) return;
        schDevTaskCtx = devTaskctx;
        isOpenMixSche = curDevTask->mixTaskData.wrapIdNum > 0;
        curDevTask_ = curDevTask;
        coreRunReadyCnt_ = coreRunReadyCnt;
        runReadyCoreIdx_[CORE_IDX_AIV] = runReadyCoreIdxZero;
        runReadyCoreIdx_[CORE_IDX_AIC] = runReadyCoreIdxOne;
        corePendReadyCnt_ = corePendReadyCnt;
        pendingIds_ = pendingIds;
        runningIds_ = runningIds;
        aicEnd_ = aicEnd;
        aicStart_ = aicStart;

        aicValidNum_ = aicValidNum;
        coreIdxPosition_ = coreIdxPosition;
        wrapCoreAvail_ = wrapCoreAvail;
        SendTaskToAiCore = func;
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue*>(curDevTask_->mixTaskData.readyWrapCoreFunctionQue);
        wrapQueueForThread_ =
            reinterpret_cast<StaticReadyCoreFunctionQueue*>(curDevTask_->mixTaskData.wrapQueueForThread[schedIdx_]);

        SetDieReadyQueue(curDevTask->dieReadyFunctionQue);

        selectReadyDieAicFunctionQue_ = GetDieReadyQueue(
            CoreType::AIC, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAicCoreFunctionQue));
        selectReadyDieAivFunctionQue_ = GetDieReadyQueue(
            CoreType::AIV, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAivCoreFunctionQue));
    }

    inline bool GetIsMixarch() { return archInfo == ArchInfo::DAV_3510; }

    inline uint32_t GetAvailableWrapCoreCnt(uint32_t& core1c1vCnt, uint32_t& core1c2vCnt, uint32_t maxCoreCnt)
    {
        uint32_t aicReadyCnt = coreRunReadyCnt_[CORE_IDX_AIC];
        for (uint32_t idx = 0; idx < aicReadyCnt && core1c2vCnt < maxCoreCnt; idx++) {
            uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
            uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);
            uint32_t aivIdx1 = GetWrapAiv1CoreIdx(aivIdx0);
            if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION) {
                CheckCoreIdxInitStatus(aicIdx);
                CheckCoreIdxInitStatus(aivIdx0);
                if (coreIdxPosition_[aivIdx1] != INVALID_COREIDX_POSITION) {
                    CheckCoreIdxInitStatus(aivIdx1);
                    core1c2vCnt++;
                } else {
                    core1c1vCnt++;
                }
            }
        }
        return core1c2vCnt + core1c1vCnt;
    }

    inline uint32_t GetWrapCorePendingCnt(uint32_t& core1c1vCnt, uint32_t& core1c2vCnt) {
        for (uint32_t idx = aicStart_; idx < aicEnd_; idx++) {
            uint32_t aivIdx0 = GetWrapAiv0CoreIdx(idx);
            uint32_t aivIdx1 = GetWrapAiv1CoreIdx(aivIdx0);
            if (pendingIds_[idx] == AICORE_TASK_INIT && pendingIds_[aivIdx0] == AICORE_TASK_INIT) {
                if (pendingIds_[aivIdx1] == AICORE_TASK_INIT) {
                    core1c2vCnt++;
                } else {
                    core1c1vCnt++;
                }
            }
        }
        return core1c1vCnt + core1c2vCnt;
    }

    inline void RemoveCoreIdx(uint32_t coreIdx, CoreType coreType)
    {
        CheckCoreIdxInitStatus(coreIdx);
        RemoveMixReadyCoreIdx(coreIdx, static_cast<int>(coreType));
        wrapCoreAvail_[coreIdx] = false;
    }

    inline uint16_t GetWrapAiv0CoreIdx(uint16_t aicIdx) { return aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_; }
    inline uint16_t GetWrapAiv1CoreIdx(uint16_t aiv0Idx) { return aiv0Idx + 1; }

    inline void UpdateWrapQueueAndRmvCoreIdx(
        WrapInfo* wrap1c2vTasks[], uint32_t task1c2vCnt, WrapInfo* wrap1c1vTasks[], uint32_t task1c1vCnt)
    {
        uint32_t idx = 0;
        for (uint32_t taskIdx = 0; taskIdx < task1c2vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c2vTasks[taskIdx];
            wrapQueueForThread_->elem[wrapQueueForThread_->tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            while (idx < coreRunReadyCnt_[CORE_IDX_AIC]) {
                uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
                uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);
                uint32_t aivIdx1 = GetWrapAiv1CoreIdx(aivIdx0);

                if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION &&
                    coreIdxPosition_[aivIdx1] != INVALID_COREIDX_POSITION) {
                    wrapInfo->aicCoreIdx = aicIdx;
                    RemoveCoreIdx(aicIdx, CoreType::AIC);
                    RemoveCoreIdx(aivIdx0, CoreType::AIV);
                    RemoveCoreIdx(aivIdx1, CoreType::AIV);
                    break;
                } else {
                    idx++;
                }
            }
        }

        idx = 0;
        for (uint32_t taskIdx = 0; taskIdx < task1c1vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c1vTasks[taskIdx];
            wrapQueueForThread_->elem[wrapQueueForThread_->tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            while (idx < coreRunReadyCnt_[CORE_IDX_AIC]) {
                uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
                uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);

                if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION) {
                    wrapInfo->aicCoreIdx = aicIdx;
                    RemoveCoreIdx(aicIdx, CoreType::AIC);
                    RemoveCoreIdx(aivIdx0, CoreType::AIV);
                    break;
                } else {
                    idx++;
                }
            }
        }
    }

    inline void UpdateWrapQueueForThreadByPending(
        WrapInfo* wrap1c2vTasks[], uint32_t task1c2vCnt, WrapInfo* wrap1c1vTasks[], uint32_t task1c1vCnt)
    {
        (void)wrap1c1vTasks;
        (void)task1c1vCnt;
        uint32_t idx = aicStart_;
        for (uint32_t taskIdx = 0; taskIdx < task1c2vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c2vTasks[taskIdx];
            wrapQueueForThread_->elem[wrapQueueForThread_->tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            for(; idx < aicEnd_; idx++) {
                uint32_t aicIdx = idx;
                uint32_t aivIdx0 = GetWrapAiv0CoreIdx(aicIdx);
                uint32_t aivIdx1 = GetWrapAiv1CoreIdx(aivIdx0);

                if (pendingIds_[aicIdx] == AICORE_TASK_INIT && pendingIds_[aivIdx0] == AICORE_TASK_INIT &&
                    pendingIds_[aivIdx1] == AICORE_TASK_INIT) {
                    wrapInfo->aicCoreIdx = aicIdx;
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIC, CoreType::AIC, aicIdx);
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIV0, CoreType::AIV, aivIdx0);
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIV1, CoreType::AIV, aivIdx1);
                    corePendReadyCnt_[CORE_IDX_AIC]--;
                    corePendReadyCnt_[CORE_IDX_AIV] -= 2;
                    wrapCoreAvail_[aicIdx] = false;
                    wrapCoreAvail_[aivIdx0] = false;
                    wrapCoreAvail_[aivIdx1] = false;
                    idx++;
                    break;
                }
            }
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


    inline uint32_t CalculateTaskCountInSync()
    {
        uint32_t taskCount = 0;
        uint32_t head = readyWrapCoreFunctionQue_->head;
        for (uint32_t i = head; i < readyWrapCoreFunctionQue_->tail; i++) {
            WrapInfo* info = &readyWrapCoreFunctionQue_->elem[i];
            bool isC1V1Ready =
                (info->mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C1V) &&
                 info->tasklist[0] != AICORE_TASK_INIT && info->tasklist[1] != AICORE_TASK_INIT);
            bool isC1V2Ready =
                (info->mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V) &&
                 info->tasklist[0] != AICORE_TASK_INIT && info->tasklist[1] != AICORE_TASK_INIT &&
                 info->tasklist[2] != AICORE_TASK_INIT); // 2:v1 index
            if (isC1V1Ready || isC1V2Ready) {
                std::swap(readyWrapCoreFunctionQue_->elem[i], readyWrapCoreFunctionQue_->elem[head + taskCount]);
                taskCount++;
            }
        }
        return taskCount;
    }

    template <typename GetCoreCntFunc, typename PostProcessFunc>
    inline bool DispatchReadyTasksImpl(GetCoreCntFunc getCoreCnt, PostProcessFunc postProcess)
    {
        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        if (tail - head == 0) {
            return false;
        }
        uint32_t core1c1vCnt = 0;
        uint32_t core1c2vCnt = 0;
        constexpr uint32_t maxTaskCnt = 8u;
        uint32_t wrapCoreCnt = getCoreCnt(core1c1vCnt, core1c2vCnt);
        if (wrapCoreCnt == 0) return true;

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        head = readyWrapCoreFunctionQue_->head;
#ifdef NO_EARLY_SEND_TASK
        uint32_t taskCount = CalculateTaskCountInSync();
#else
        uint32_t taskCount = readyWrapCoreFunctionQue_->tail - head;
#endif
        if (unlikely(taskCount == 0)) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return false;
        }
        WrapInfo* localTasks[maxTaskCnt];
        uint32_t maxReadyCnt = taskCount > maxTaskCnt ? maxTaskCnt : taskCount;
        maxReadyCnt = maxReadyCnt > wrapCoreCnt ? wrapCoreCnt : maxReadyCnt;
        uint32_t taskHead = 0, taskTail = maxReadyCnt;
        while (taskHead < taskTail) {
            WrapInfo* info = &readyWrapCoreFunctionQue_->elem[head++];
            if (info->mixResourceType == static_cast<uint32_t>(MixResourceType::MIX_1C2V)) {
                if (core1c2vCnt == 0) break;
                localTasks[taskHead++] = info;
                core1c2vCnt--;
            } else {
                localTasks[--taskTail] = info;
            }
        }
        uint32_t valid1c1vCnt = maxReadyCnt - taskTail;
        uint32_t validReadyCnt = taskHead + valid1c1vCnt;
        readyWrapCoreFunctionQue_->head += validReadyCnt;
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        WrapInfo** task1c1vPtr = localTasks + taskTail;
        postProcess(localTasks, taskHead, task1c1vPtr, valid1c1vCnt);
        return taskCount > maxReadyCnt;
    }

    inline bool UpdateWrapQueueForThread()
    {
        return DispatchReadyTasksImpl(
            [this](uint32_t& c1c1v, uint32_t& c1c2v) {
                return GetAvailableWrapCoreCnt(c1c1v, c1c2v, 8u);
            },
            [this](WrapInfo* a[], uint32_t b, WrapInfo* c[], uint32_t d) {
                UpdateWrapQueueAndRmvCoreIdx(a, b, c, d);
            });
    }

    inline void CheckAndSendTask(WrapInfo* wrapInfo, uint32_t wrapAicoreIdx, CoreType coreType, uint16_t coreIdx)
    {
        uint32_t taskId = wrapInfo->tasklist[wrapAicoreIdx];
        // 此处可能一个Task准备下发，另一个还没初始化。另一个准备下发时，前面一个已经结束
        // AICORE_TASK_SUBMITTED、AICORE_TASK_INIT、taskId == AICORE_TASK_STOP最高位是1，正常taskId的最高位是0
        if (taskId & AICORE_FIN_MASK) return;
        DEV_VERBOSE_DEBUG(
            "try to send wrapId[%u]'s wrapAicoreIdx[%u] taskId[%u]", wrapInfo->wrapId, wrapAicoreIdx, taskId);
        SendTaskToAiCore(schDevTaskCtx, coreType, coreIdx, taskId);
        wrapInfo->tasklist[wrapAicoreIdx] = AICORE_TASK_SUBMITTED;
    }

    inline void TryAlloPendingCoreAndSend() {
        DispatchReadyTasksImpl(
            [this](uint32_t& c1c1v, uint32_t& c1c2v) {
                return GetWrapCorePendingCnt(c1c1v, c1c2v);
            },
            [this](WrapInfo* a[], uint32_t b, WrapInfo* c[], uint32_t d) {
                UpdateWrapQueueForThreadByPending(a, b, c, d);
            });
    }

    inline void SendMixCoreTasksInRange(uint32_t head, uint32_t tail)
    {
        for (uint32_t idx = head; idx < tail; idx++) {
            WrapInfo* wrapInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_->elem[idx]);
            switch (wrapInfo->mixResourceType) {
                case static_cast<uint8_t>(MixResourceType::MIX_1C1V): {
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIC, CoreType::AIC, wrapInfo->aicCoreIdx);
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIV0, CoreType::AIV, GetWrapAiv0CoreIdx(wrapInfo->aicCoreIdx));
                    break;
                }
                case static_cast<uint8_t>(MixResourceType::MIX_1C2V): {
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIC, CoreType::AIC, wrapInfo->aicCoreIdx);
                    uint16_t aiv0CoreIdx = GetWrapAiv0CoreIdx(wrapInfo->aicCoreIdx);
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIV0, CoreType::AIV, aiv0CoreIdx);
                    CheckAndSendTask(wrapInfo, WRAP_IDX_AIV1, CoreType::AIV, GetWrapAiv1CoreIdx(aiv0CoreIdx));
                    break;
                }
                default:
                    DEV_ERROR(
                        DevCommonErr::PARAM_INVALID, "#sche.wrap.invalid_mode: illegal mixType: %hhu\n",
                        wrapInfo->mixResourceType);
                    break;
            }
        }
    }

    inline void DispatchMixCoreTask()
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        bool hasAvailTask = UpdateWrapQueueForThread();
        SendMixCoreTasksInRange(wrapQueueForThread_->head, wrapQueueForThread_->tail);
        if (isMixPending_ && hasAvailTask) {
            TryAlloPendingCoreAndSend();
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

    inline int32_t GetWrapAicoreIdx(uint32_t coreType, int32_t wrapVecId)
    {
        if (coreType == static_cast<uint32_t>(CoreType::AIC)) {
            return WRAP_IDX_AIC;
        } else {
            return wrapVecId == 1 ? WRAP_IDX_AIV1 : WRAP_IDX_AIV0;
        }
    }

    inline int32_t GetWrapAicoreIdx(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        auto coreType = cceBinary[callList[opIndex]].coreType;
        auto wrapVecId = cceBinary[callList[opIndex]].wrapVecId;
        return GetWrapAicoreIdx(coreType, wrapVecId);
    }

    bool IsBindedWrapId(uint32_t taskId, uint32_t& wrapId)
    {
        RETURN_RET_IF_NOT(isOpenMixSche, false);
        int id = GetWrapId(taskId);
        if (id == -1) return false;
        wrapId = id;
        return true;
    }

    inline void PushTaskToTasklist(uint32_t wrapId, uint32_t taskId, uint32_t wrapAicoreIdx, uint8_t mixResourceType)
    {
        auto funcId = FuncID(taskId);
        auto opWrapId = GetOpWrapID(wrapId);
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto opWrapOffsetList = dyntask->devTask.mixTaskData.opWrapOffsetList[funcId];
        if (unlikely(opWrapOffsetList == nullptr)) {
            DEV_ERROR(
                DevCommonErr::NULLPTR, "#sche.resolve.wrap: the funcIndex:%u have wrapId but not found: %u!", funcId,
                wrapId);
            return;
        }
        uint16_t offset = opWrapOffsetList[opWrapId];
        if (offset != INVALID_UINT16_IDX) {
            WrapInfo* wrapInfo = &readyWrapCoreFunctionQue_->elem[offset];
            wrapInfo->tasklist[wrapAicoreIdx] = taskId;
            return;
        }

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        offset = opWrapOffsetList[opWrapId];
        if (unlikely(offset != INVALID_UINT16_IDX)) {
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            WrapInfo* wrapInfo = &readyWrapCoreFunctionQue_->elem[offset];
            wrapInfo->tasklist[wrapAicoreIdx] = taskId;
            return;
        }

        // add a new wrapinfo
        opWrapOffsetList[opWrapId] = readyWrapCoreFunctionQue_->tail;
        WrapInfo* wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail++];
        wrapInfo->mixResourceType = mixResourceType; // 需在锁释放前赋值，锁释放其它线程就能立即绑核、下发
        wrapInfo->tasklist[WRAP_IDX_AIC] = AICORE_TASK_INIT;
        wrapInfo->tasklist[WRAP_IDX_AIV0] = AICORE_TASK_INIT;
        wrapInfo->tasklist[WRAP_IDX_AIV1] = AICORE_TASK_INIT;
        wrapInfo->tasklist[wrapAicoreIdx] = taskId;
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        wrapInfo->wrapId = wrapId;
    }

    inline void ResolveDepForMixCore(uint32_t taskId, uint32_t wrapId, const DevCceBinary* cceBinary)
    {
        // resolve dep, if has available core, send task directly, else call PushTaskToTasklist, try to send task in
        // next loop
        DEV_VERBOSE_DEBUG("taskId = %u, wrapId = %u", taskId, wrapId);
        int32_t wrapAicoreIdx = GetWrapAicoreIdx(cceBinary->coreType, cceBinary->wrapVecId);
        PushTaskToTasklist(wrapId, taskId, wrapAicoreIdx, cceBinary->mixResourceType);
    }

    inline void UpdateFinishIdForMixCore(uint32_t finishId, CoreType coreType, uint32_t coreIdx)
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        (void)coreType;
        int32_t id = GetWrapId(finishId);
        if (id == -1) return;
        uint32_t wrapId = id;
        auto funcId = FuncID(finishId);
        auto opWrapId = GetOpWrapID(wrapId);
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto opWrapOffsetList = dyntask->devTask.mixTaskData.opWrapOffsetList[funcId];
        uint16_t offset = opWrapOffsetList[opWrapId];
        WrapInfo* wrapInfo = &readyWrapCoreFunctionQue_->elem[offset];

        int32_t wrapAicoreIdx = GetWrapAicoreIdx(finishId);
        wrapInfo->tasklist[wrapAicoreIdx] = AICORE_TASK_STOP;

        wrapCoreAvail_[coreIdx] = true;

        if (IsMixTaskFinish(wrapInfo)) { // all tasks for this wrap finish
            for (uint32_t idx = wrapQueueForThread_->head; idx < wrapQueueForThread_->tail; idx++) {
                auto tmpInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_->elem[idx]);
                if (tmpInfo->wrapId == wrapId) {
                    DEV_VERBOSE_DEBUG("wrapId %u 's all tasks finish, release wrapcore", wrapId);
                    std::swap(wrapQueueForThread_->elem[idx], wrapQueueForThread_->elem[--wrapQueueForThread_->tail]);
                    return;
                }
            }
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
        if (!GetIsMixarch() || dieId_ == DieId::DIE_MIX || dieId_ == DieId::DIE_UNKNOWN) {
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
