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
#include "securec.h"
#include "device_common.h"
#include "tilefwk/config.h"
#include "tilefwk/aicore_print.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/schema/schema.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/wrap_manager.h"

namespace npu::tile_fwk::dynamic {

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1; // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0; // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2; // 一次寄存器task个数

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    TaskInfo(int idx, uint64_t id) : coreIdx(idx), taskId(id) {}
};

class AiCoreManager {
public:
    explicit AiCoreManager(AicpuTaskManager &aicpuTaskManager) : aicpuTaskManager_(aicpuTaskManager), aicoreProf_(*this){};
    ~AiCoreManager(){};

    void InitLogger(AicoreLogger *logger) {
        logger_ = logger;
    }

    void SendDevTaskModel(DeviceTask *devTask) {
        int64_t funcdata;
        auto dyntask = (DynDeviceTask *)devTask;
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        ForEachManageAicore([&](int coreIdx) {
            auto logbuf = logger_ ? logger_[coreIdx].GetBuffer() : nullptr;
            aicoreHal_.InitTaskData(coreIdx, funcdata, (uint64_t)logbuf);
        });
    }
    inline void SetSchduleContext(SchduleContext * context) {
        this->context_ = context;
    }

    inline bool CheckAndResetReg() {
        if (!validGetPgMask_) {
            return true;
        }
        bool isValid = true;
        DEV_IF_DEVICE {
            if (aicoreHal_.GetRegSprDataMainBase() == DAV_3510::REG_SPR_DATA_MAIN_BASE) {
                return true;
            }
            auto regAddrs = aicoreHal_.GetRegAddrs();
            uint32_t regNum = aicoreHal_.GetregNum();
            for (uint32_t coreIdx = 0; coreIdx < regNum; ++coreIdx) {
                if (regAddrs[coreIdx] == 0) {
                    continue;
                }
                uint32_t currentStatus = *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE));
                if (currentStatus != REG_SPR_FAST_PATH_CLOSE) {
                    isValid = false;
                    *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE)) = REG_SPR_FAST_PATH_CLOSE;
                }
            }
        }
        return isValid;
    }
    inline void InitDevTask(DeviceTaskCtrl *taskCtrl) {
        isFirstTaskSend_ = true;
        curTaskCtrl_ = taskCtrl;
        curDevTask_ = taskCtrl->devTask;
        curTaskType_ = taskCtrl->taskType;
        curTaskId_ = taskCtrl->taskId;
        aicoreHal_.SetModel(taskCtrl->devTask->aicoreModel);

        if (!preFetchSuccess_) {
            SendDevTaskModel(curDevTask_);
        }

        readyAicCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);
        readyAicpuFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAicpuFunctionQue);
        wrapManager_.Init(curDevTask_, context_->coreRunReadyCnt_, context_->runReadyCoreIdx_[CORE_IDX_AIV],
            context_->runReadyCoreIdx_[CORE_IDX_AIC], context_->corePendReadyCnt_, pendingIds_.data(),
            runningIds_.data(), aicValidNum_, [&](CoreType coreType, int arg1, uint64_t arg2) {SendTaskToAiCore(coreType, arg1, arg2);});
    }

    void CountSendTask(uint64_t &sentAic, uint64_t &sentAiv) {
        sentAic = context_->sendCnt_[static_cast<int>(CoreType::AIC)];
        sentAiv = context_->sendCnt_[static_cast<int>(CoreType::AIV)];
        context_->waitTaskCnt_[static_cast<int>(CoreType::AIC)] += sentAic;
        context_->waitTaskCnt_[static_cast<int>(CoreType::AIV)] += sentAiv;
        context_->sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        context_->sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    template <bool enableAicpuTask = false>
    inline int32_t RunCoreTask(DeviceTaskCtrl *taskCtrl, uint64_t& sent) {
        int32_t ret = DEVICE_MACHINE_OK;
        (void)taskCtrl;
        wrapManager_.DispatchMixCoreTask();
        ret = DispatchAiCoreTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        ret = DispatchAiCoreTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }

        uint64_t sentAic = 0;
        uint64_t sentAiv = 0;
        CountSendTask(sentAic, sentAiv);

        sent = 0UL;
        if constexpr (enableAicpuTask) {
            if (IsNeedProcAicpuTask()) {
                ret = ResolveDepForAicpuTask(sent);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        }

        DEV_IF_VERBOSE_DEBUG {
            __sync_fetch_and_add(&(taskCtrl->finishedAicFunctionCnt), sentAic);
            __sync_fetch_and_add(&(taskCtrl->finishedAivFunctionCnt), sentAiv);
            __sync_fetch_and_add(&(taskCtrl->finishedAicpuFunctionCnt), sent);
            __sync_fetch_and_add(&(taskCtrl->finishedHubFunctionCnt), context_->resolveHubCnt_);
            procAicCoreFunctionCnt_ += sentAic;
            procAivCoreFunctionCnt_ += sentAiv;
            procAicpuFunctionCnt_ += sent;
            DEV_VERBOSE_DEBUG("finish send  aic task cnt: %lu,  aiv task cnt: %lu, hub task cnt:%lu,"
                "aicpu task cnt:%lu, target totalcnt: %lu.",
                taskCtrl->finishedAicFunctionCnt, taskCtrl->finishedAivFunctionCnt,
                taskCtrl->finishedHubFunctionCnt, taskCtrl->finishedAicpuFunctionCnt, curDevTask_->coreFunctionCnt);
        }
        sent += (sentAic + sentAiv);
        return ret;
    }

    void DumpAicoreLog(int coreIdx) {
        const int bufSize = 512;
        char buf[bufSize];
        while (logger_[coreIdx].Read(buf, bufSize)) {
            DEV_INFO("core-%d %s", coreIdx, buf);
        }
    }

    inline int RunTask(DeviceTaskCtrl *taskCtrl) {
        auto ret = ExecuteTask(taskCtrl);
        wrapManager_.Deinit();
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            DEV_ERROR("Aicpu %d proc finish %lu %lu %lu, but timeout !.", aicpuIdx_,
                taskCtrl->finishedFunctionCnt.load(), curDevTask_->coreFunctionCnt, taskCtrl->taskId);
            DumpAiCoreStatus();
        }
        return ret;
    }

    inline int32_t ExecuteTask(DeviceTaskCtrl *taskCtrl) {
        int32_t ret = ProcessTask(taskCtrl);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        ret = ProcessTaskLoop(taskCtrl);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        PerfMtTrace(PERF_TRACE_DEV_TASK_SCHED_EXEC, aicpuIdx_);
        PerfMtBegin(PERF_EVT_SYNC_AICORE, aicpuIdx_);
        int32_t rc = SyncTaskFinish();
        PerfMtTrace(PERF_TRACE_DEV_TASK_SYNC_CORE_STOP, aicpuIdx_);
        if (rc != DEVICE_MACHINE_OK) {
            ret = rc;
        }
        PerfMtEnd(PERF_EVT_SYNC_AICORE, aicpuIdx_);
        DEV_DEBUG("aicpu %d proc finish send all task,aic: %lu, aiv: %lu, aicpu: %lu, sync finish ret: %d.",
            aicpuIdx_, procAicCoreFunctionCnt_, procAivCoreFunctionCnt_, procAicpuFunctionCnt_, ret);
        return ret;
    }

    inline int32_t ProcessTask(DeviceTaskCtrl *taskCtrl) {
        int32_t ret = DEVICE_MACHINE_OK;
        seq = taskCtrl->taskId;
        DEV_INFO("receive new task %lu.", taskCtrl->taskId);
        InitDevTask(taskCtrl);

        uint64_t curSent = 0UL;
        if (!taskCtrl->isFirstDevTask) {
            ret = RunCoreTask(taskCtrl, curSent);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            taskCtrl->finishedFunctionCnt.fetch_add(curSent, std::memory_order_relaxed);
        }

        if (IsNeedProcAicpuTask()) {
            const bool profSwitch = aicoreProf_.ProfIsEnable();
            ret = aicpuTaskManager_.Init(reinterpret_cast<DynDeviceTask *>(curDevTask_), profSwitch);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    inline int ProcessTaskLoop(DeviceTaskCtrl *taskCtrl) {
        uint32_t lastSent = 0;
        uint64_t start = GetCycles();
        uint32_t allSentCnt = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed);
        while (allSentCnt < curDevTask_->coreFunctionCnt) {
            uint64_t curSent = 0;
            int32_t ret = RunCoreTask<true>(taskCtrl, curSent);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            if (likely(curSent == 0)) {
                if (lastSent > 0) {
                    taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
                    lastSent = 0;
                }
            } else {
                lastSent += curSent;
            }

            DEV_IF_DEVICE {
                if (GetCycles() - start > TIMEOUT_CYCLES) {
                    return DEVICE_MACHINE_TIMEOUT_CORETASK;
                }
            }
            // To prevent an unnecessary execution of RunCoreTask after the final batch of tasks is sent.
            allSentCnt = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) + lastSent;
        }
        if (lastSent > 0) {
            // Other SCH-AICPU are still waiting for the taskCtrl->finishedFunctionCnt actual value.
            taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
        }

        return DEVICE_MACHINE_OK;
    }

    inline void DumpLastWord(int coreIdx) {
        uint64_t status = aicoreHal_.GetAicoreStatus(coreIdx);
        if (pendingIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR("status %lu,pending taskid: %s,funcdata:  %s", status, std::to_string(pendingIds_[coreIdx]).c_str(),
            ((DynDeviceTask *)curDevTask_)->DumpTaskData(pendingIds_[coreIdx]).c_str());
        }
        if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR("status %lu,running taskid:%s,funcdata:  %s", status, std::to_string(runningIds_[coreIdx]).c_str(),
                ((DynDeviceTask *)curDevTask_)->DumpTaskData(runningIds_[coreIdx]).c_str());
        }
    }

    void ResetRegAll() {
        ForEachManageAicore([this](int coreIdx) {
            if (aicoreHal_.ReadPathReg(coreIdx) == REG_SPR_FAST_PATH_OPEN) {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            } else {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
            }
        });
    }

    inline void PostRun(int ret, DeviceTaskCtrl *taskCtrl) {
        if (ret) {
            DEV_ERROR("task %lu execute error %d, skip rest tasks", taskCtrl->taskId, ret);
            if constexpr (IsDeviceMode()) {
                ForEachManageAicore([&](int coreIdx) { DumpLastWord(coreIdx); });
            }
            do {
                taskCtrl->PutTask(ret);
            } while ((taskCtrl = taskQueue_->Dequeue()));

            if constexpr (IsDeviceMode()) {
                NormalStop(); // some core maybe timeout
            }
        }

        if constexpr (IsDeviceMode()) {
            PerfMtTrace(PERF_TRACE_WAIT_CORE_EXIT, aicpuIdx_);
            ProfStop();
        }
        DEV_INFO("Aicpu %d stop ret = %d, proc aic task cnt: %lu, aiv task cnt: %lu.",
                 aicpuIdx_, ret, procAicCoreFunctionCnt_, procAivCoreFunctionCnt_);
    }

    inline int Run(int threadIdx, DeviceArgs *deviceArgs, int schedIdx) {
        int ret = DEVICE_MACHINE_OK;
        DEV_DEBUG("schedule run threadIdx:%d", threadIdx);
        Init(threadIdx, deviceArgs, schedIdx);
        PerfMtTrace(PERF_TRACE_INIT, threadIdx);
        DEV_DEBUG("Schedule run init succ");
        DeviceTaskCtrl *taskCtrl = nullptr;
        taskQueue_ = &(reinterpret_cast<SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE>*>(deviceArgs->taskQueue)[schedIdx_]);
        if constexpr (IsDeviceMode()) {
            ret = HandShake();
            PerfMtTrace(PERF_TRACE_CORE_HAND_SHAKE, threadIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR("hand shake timeout.");
                AbnormalStop();
                while ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->PutTask(ret);
                }
                return ret;
            }
            reinterpret_cast<DevStartArgs *>(deviceArgs->startArgsAddr)->syncFlag = 1;
            aicoreProf_.ProfStart();
        }
        DEV_DEBUG("Schedule run start succ");
        uint64_t lastDevTaskFinCycle = 0;
        while (ret == 0) {
            DEV_DEBUG("Schedule task wait");
            taskCtrl = preFetchSuccess_ ? preFetchNextDevTaskCtrl_ : taskQueue_->Dequeue();
            DEV_DEBUG("Schedule task recv");
            if (taskCtrl == nullptr) {
                PerfMtTrace(PERF_TRACE_WAIT_ALL_DEV_TASK_FINISH, aicpuIdx_, lastDevTaskFinCycle);
                if (!isSendStop) {
                    SyncTaskFinish(true);
                }
                break;
            }
            PerfMtTrace(PERF_TRACE_DEV_TASK_RCV, aicpuIdx_);
            PROF_STAGE_BEGIN_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.before\n");
            PerfMtBegin(PERF_EVT_RUN_TASK, threadIdx);
            ret = RunTask(taskCtrl);
            lastDevTaskFinCycle = GetCycles();
            PerfMtEnd(PERF_EVT_RUN_TASK, threadIdx);
            DEV_DEBUG("Run task finish taskid=%d ret %d.", curTaskId_, ret);
            if (ret != 0)
                break;
            taskCtrl->PutTask(ret);
            PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, threadIdx);
            PROF_STAGE_END_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.after\n");
        }
        PostRun(ret, taskCtrl);
        return ret;
    }

    int32_t ProcessCompletedAicpuTask(uint64_t taskId) {
        int32_t ret = ResolveDepDyn(taskId);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return BatchPushReadyQueue();
    }

    inline void DumpAicorePerfTrace(std::ostringstream& oss) {
        (void)oss;
#if ENABLE_PERF_TRACE
        for (int i = aicStart_; i < aicEnd_; ++i) {
            int ret = aicoreHal_.DumpAicorePerfTrace(aicpuIdx_, i, CoreType::AIC, oss);
            if (ret == DEVICE_MACHINE_OK) {
                oss << ",";
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            int ret = aicoreHal_.DumpAicorePerfTrace(aicpuIdx_, i, CoreType::AIV, oss);
            if (ret == DEVICE_MACHINE_OK) {
                oss << ((i == aivEnd_ - 1) ? "" : ",");
            }
        }
#endif
    }

private:
    inline void DumpTaskProf() {
        ForEachManageAicoreWithRet([this] (int coreIdx) -> int { return aicoreHal_.DumpTaskProf(coreIdx);});
    }

    inline void ProfStop() {
        if (aicoreProf_.ProfIsEnable()) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
            DumpTaskProf();
#endif
        }

        aicoreProf_.ProfStop();
    }

    inline void DumpAiCoreStatus() const {
        DEV_IF_VERBOSE_DEBUG {
            ForEachManageAicore([this](int coreIdx) {
                if constexpr (IsDeviceMode()) {
                    aicoreHal_.DumpAicoreStatus(coreIdx);
                }
                DEV_VERBOSE_DEBUG("reg low task: runningid(%u) pendingid(%u) dfxpos(%d)", runningIds_[coreIdx],
                    pendingIds_[coreIdx], taskDfxStatPos_[coreIdx]);

                DEV_VERBOSE_DEBUG("send task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    sendTask_[coreIdx].size());
                for (size_t i = 0; i < sendTask_[coreIdx].size(); i++) {
                    DEV_VERBOSE_DEBUG("send task: seqno %d, taskId %lx", (int)i, sendTask_[coreIdx][i].taskId);
                }

                DEV_VERBOSE_DEBUG("recv finish task info ~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    recvFinTask_[coreIdx].size());
                for (size_t i = 0; i < recvFinTask_[coreIdx].size(); i++) {
                    DEV_VERBOSE_DEBUG("recv task: seqno %d, taskId %lx", (int)i, recvFinTask_[coreIdx][i].taskId);
                }

                DEV_VERBOSE_DEBUG("recv ack task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    recvAckTask_[coreIdx].size());
                for (size_t i = 0; i < recvAckTask_[coreIdx].size(); i++) {
                    DEV_VERBOSE_DEBUG("recv ack task: seqno %d, taskId %lx", static_cast<int>(i), recvAckTask_[coreIdx][i].taskId);
                }
            });
        }
    }

    inline bool CheckStopTaskCanBeSent(int coreIdx) {
        if (pendingIds_[coreIdx] == AICORE_TASK_INIT && runningIds_[coreIdx] == AICORE_TASK_INIT) {
            return true;
        }

        uint64_t finTaskVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
        bool bMatch = false;

        int type =static_cast<int>(AicoreType(coreIdx));
        if (likely(regLFinTaskState == TASK_FIN_STATE)) {
            if (pendingIds_[coreIdx] == regLFinTaskId) {
                bMatch = true;
                context_->runReadyCoreIdx_[type][context_->coreRunReadyCnt_[type]++] = coreIdx;
                context_->corePendReadyCnt_[type]++;
                if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
                    DfxProcAfterFinishTask(coreIdx, runningIds_[coreIdx]);
                }
                DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
                DEV_VERBOSE_DEBUG("rcv final pending task finish, pendtask: %u", regLFinTaskId);
            } else if (runningIds_[coreIdx] == regLFinTaskId && pendingIds_[coreIdx] == AICORE_TASK_INIT) {
                bMatch = true;
                context_->runReadyCoreIdx_[type][context_->coreRunReadyCnt_[type]++] = coreIdx;
                DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
                DEV_VERBOSE_DEBUG("rcv final running task finish, runningtask: %u", regLFinTaskId);
            }
        } else if (regLFinTaskState == TASK_ACK_STATE && pendingIds_[coreIdx] == regLFinTaskId) {
           // The core stop task can be sent once the last task ACK is received, without waiting for finish rsp.
           // The execution of the final task and the sending of the final core stop task can be parallelized.
            bMatch = true;
            context_->runReadyCoreIdx_[type][context_->coreRunReadyCnt_[type]++] = coreIdx;
            context_->corePendReadyCnt_[type]++;
            DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
            if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DfxProcAfterFinishTask(coreIdx, runningIds_[coreIdx]);
            }
            DEV_VERBOSE_DEBUG("rcv final pending task ack, pendtask: %u", regLFinTaskId);
        }

        if (bMatch) {
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            runningIds_[coreIdx] = AICORE_TASK_INIT;
            runningResolveIndexList_[coreIdx] = 0;
            return true;
        }

        return false;
    }

    inline bool PreFetchNextDevTask() {
        preFetchNextDevTaskCtrl_ = nullptr;
        preFetchSuccess_ = taskQueue_->TryDequeue(preFetchNextDevTaskCtrl_);

        DEV_INFO("Prefetch next dev task : success:%d, devtaskid:%lu.",
            preFetchSuccess_, preFetchNextDevTaskCtrl_ != nullptr ? preFetchNextDevTaskCtrl_->taskId : INVALID_DEV_TASK_ID);
        return preFetchSuccess_;
    }

    inline void SendPreFetchNextDevTaskDataToCore(int coreIdx) {
        if (preFetchNextDevTaskCtrl_ == nullptr) {
            return;
        }

        int64_t funcdata;
        auto dyntask = reinterpret_cast<DynDeviceTask *>(preFetchNextDevTaskCtrl_->devTask);
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        auto logbuf = logger_ ? logger_[coreIdx].GetBuffer() : nullptr;
        aicoreHal_.InitTaskData(coreIdx, funcdata, (uint64_t)logbuf);
        return;
    }

    enum class AicoreStatus {
        CORE_TASK_WAIT_FINISH = 0,
        CORE_SEND_STOP,
        CORE_FINISH_STOP,
    };

    void SendStopToCore(int coreIdx, bool isLastDevTask, AicoreStatus *coreStatus, int &finishStopNum) {
        DEV_IF_DEVICE {
            if (isLastDevTask) {
                NormalStopSingleCore(coreIdx);
                coreStatus[coreIdx] = AicoreStatus::CORE_FINISH_STOP;
                finishStopNum++;
                DEV_VERBOSE_DEBUG("Last devtask ,core %d send AICORE_TASK_STOP.", coreIdx);
            } else {
                uint64_t stopFlag =
                    (static_cast<uint64_t>(curTaskId_) << REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP + 1);
                aicoreHal_.SetReadyQueue(coreIdx, stopFlag);
                coreStatus[coreIdx] = AicoreStatus::CORE_SEND_STOP;
                DEV_VERBOSE_DEBUG("core %d send AICORE_FUNC_STOP %lx.", coreIdx, stopFlag);
            }
        } else {
            coreStatus[coreIdx] = AicoreStatus::CORE_FINISH_STOP;
            finishStopNum++;
        }
    }

    inline void PreSendStopToIdleCore(bool isLastDevTask, AicoreStatus *coreStatus, int &finishStopNum) {
        uint32_t aicIdleNum = context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIC)];
        uint32_t aivIdleNum = context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIV)];

        for (uint32_t i = 0; i < aicIdleNum; i++) {
            SendStopToCore(context_->runReadyCoreIdx_[static_cast<int>(CoreType::AIC)][i],
                isLastDevTask, coreStatus, finishStopNum);
        }

        for (uint32_t i = 0; i < aivIdleNum; i++) {
            SendStopToCore(context_->runReadyCoreIdx_[static_cast<int>(CoreType::AIV)][i],
                isLastDevTask, coreStatus, finishStopNum);
        }
    }

    inline void AicoreDevTaskFinishProc(int coreIdx,  bool isLastDevTask,
                AicoreStatus *coreStatus, int &finishStopNum) {
        if ((coreStatus[coreIdx] == AicoreStatus::CORE_TASK_WAIT_FINISH) && CheckStopTaskCanBeSent(coreIdx)) {
            SendStopToCore(coreIdx, isLastDevTask, coreStatus, finishStopNum);
        }

        if (!isLastDevTask) {
            DEV_IF_DEVICE {
                if ((coreStatus[coreIdx] == AicoreStatus::CORE_SEND_STOP) &&
                    (aicoreHal_.GetFinishedTask(coreIdx) == ((static_cast<uint64_t>(curTaskId_) <<
                        REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP | AICORE_FIN_MASK)))) {
                    SendPreFetchNextDevTaskDataToCore(coreIdx);
                    coreStatus[coreIdx] = AicoreStatus::CORE_FINISH_STOP;
                    finishStopNum++;
                    DEV_VERBOSE_DEBUG("core %d rsp AICORE_FUNC_STOP ack.", coreIdx);
                }
            }
        }

        return;
    }

    inline void DumpDfxWhenCoreNotStop(AicoreStatus *coreStatus) {
        for (int i = aicStart_; i < aicEnd_; i++) {
            if (coreStatus[i] != AicoreStatus::CORE_FINISH_STOP) {
                DEV_ERROR("left aic core %d not stop, status %d, pending:%u, rungning:%u, regfinishid: %lu,"
                "core last status:%lu", i, ToUnderlying(coreStatus[i]), pendingIds_[i], runningIds_[i],
                aicoreHal_.GetFinishedTask(i), aicoreHal_.GetAicoreStatus(i));
            }
        }

        for (int i = aivStart_; i < aivEnd_; i++) {
            if (coreStatus[i] != AicoreStatus::CORE_FINISH_STOP) {
                DEV_ERROR("left aiv core %d not stop, status %d, pending:%u, rungning:%u, regfinishid: %lu,"
                "core last status:%lu", i, ToUnderlying(coreStatus[i]), pendingIds_[i], runningIds_[i],
                aicoreHal_.GetFinishedTask(i), aicoreHal_.GetAicoreStatus(i));
            }
        }
    }

    inline int SyncTaskFinish(bool forceStop = false) {
        int finishStopNum = 0;
        int aicNum = aicEnd_ - aicStart_;
        int aivNum = aivEnd_ - aivStart_;
        int mngCoreNum = aicNum + aivNum;
        AicoreStatus coreStatus[MAX_AICORE_NUM] = {AicoreStatus::CORE_TASK_WAIT_FINISH};
        bool aicAllStop = false;
        bool aivAllStop = false;
        bool isLastDevTask = false;
        if (!forceStop) {
            isLastDevTask = reinterpret_cast<DynDeviceTask *>(curDevTask_)->IsLastTask();
            if (!isLastDevTask) {
                if (PreFetchNextDevTask() && preFetchNextDevTaskCtrl_ == nullptr) {
                    isLastDevTask = true;
                }
            } else {
                preFetchNextDevTaskCtrl_ = nullptr;
                preFetchSuccess_ = false;
            }
        } else {
            isLastDevTask = true;
        }

        if (isLastDevTask) {
            aicAllStop = (context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIC)] == static_cast<uint32_t>(aicNum));
            aivAllStop = (context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIV)] == static_cast<uint32_t>(aivNum));
            isSendStop = true;
        }

        PreSendStopToIdleCore(isLastDevTask, coreStatus, finishStopNum);

        uint64_t start_cycles = GetCycles();
        while (finishStopNum < mngCoreNum) {
            bool curIterAicAllStop = true;
            bool curIterAivAllStop = true;
            for (int i = aicStart_; (!aicAllStop) && i < aicEnd_; i++) {
                if (coreStatus[i] == AicoreStatus::CORE_FINISH_STOP) {
                    continue;
                }

                AicoreDevTaskFinishProc(i, isLastDevTask, coreStatus, finishStopNum);
                if (coreStatus[i] != AicoreStatus::CORE_FINISH_STOP) {
                    curIterAicAllStop = false;
                }
            }
            aicAllStop = curIterAicAllStop;

            for (int i = aivStart_; (!aivAllStop) && i < aivEnd_; i++) {
                if (coreStatus[i] == AicoreStatus::CORE_FINISH_STOP) {
                    continue;
                }

                AicoreDevTaskFinishProc(i, isLastDevTask, coreStatus, finishStopNum);
                if (coreStatus[i] != AicoreStatus::CORE_FINISH_STOP) {
                    curIterAivAllStop = false;
                }
            }
            aivAllStop = curIterAivAllStop;
            DEV_IF_DEVICE {
                if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
                    DumpDfxWhenCoreNotStop(coreStatus);
                    DEV_ERROR("SyncAicoreDevTaskFinish timeout notstopNum=%d.", mngCoreNum - finishStopNum);
                    return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
                }
            }
        }
        return SyncAicpuTaskFinish();
    }

    inline int32_t SyncAicpuTaskFinish() {
        if (IsNeedProcAicpuTask()) {
            auto ret = aicpuTaskManager_.SyncAicpuTaskFinish(*this);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    inline uint32_t GetReadyCoreNum(CoreType type) {
        if (enableFairSch_ && IsExistOtherAicpuIdle(type))  {
            return context_->coreRunReadyCnt_[static_cast<int>(type)];
        }
        return context_->corePendReadyCnt_[static_cast<int>(type)];
    }

    inline uint64_t TryBatchSendTask(CoreType type, ReadyCoreFunctionQueue* readyQue,
                int coreIdxStart, int coreIdxEnd) {
        if (__atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED) == __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED)) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Task: 0", aicpuIdx_);
            return 0;
        }

        uint32_t ready = GetReadyCoreNum(type);
        if (ready == 0 ) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Core: %u.", aicpuIdx_, ready);
            return 0;
        }
        PerfMtBegin(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        uint32_t readyId[MAX_MANAGER_AIV_NUM];
        ReadyQueueLock(readyQue);
        uint32_t head = __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = std::min(ready, tail - head);
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("AiCpud:%u, taskCount is zero", head);
            ReadyQueueUnLock(readyQue);
            PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
            return 0;
        }
        bool isRealLifo = (enableL2CacheSch_ && !firstLock[static_cast<int>(type)]);
        if (isRealLifo) {
            memcpy_s(readyId, taskCount * sizeof(uint64_t),
                reinterpret_cast<uint8_t *>(&readyQue->elem[tail - taskCount]), taskCount * sizeof(uint32_t));
            __atomic_fetch_sub(&readyQue->tail, taskCount, std::memory_order_release);
        } else {
            __atomic_fetch_add(&readyQue->head, taskCount, std::memory_order_release);
        }
        ReadyQueueUnLock((readyQue));
        DEV_VERBOSE_DEBUG("AiCpud:%d, pop all new task count: %u", aicpuIdx_, taskCount);
        BatchSendTask(type, isRealLifo ? &readyId[taskCount - 1] : &readyQue->elem[head],
            taskCount, coreIdxStart, coreIdxEnd, isRealLifo);
        DEV_VERBOSE_DEBUG("core ready cnt: %u", context_->corePendReadyCnt_[static_cast<int>(type)]);
        firstLock[static_cast<int>(type)] = false;
        PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        return taskCount;
    }

    inline uint32_t BatchSendTask(CoreType type, uint32_t *newTask, uint32_t taskCount,
        int coreIdxStart, int coreIdxEnd, bool isLifo) {
        uint32_t sendCnt = 0;
        uint32_t coreRunReadyCnt = context_->coreRunReadyCnt_[static_cast<int>(type)];
        DEV_VERBOSE_DEBUG("Begin Batch send %s task: corerunreadycnt:%u, pendreadyCnt:%u, taskCount:%u.",
            type == CoreType::AIC ? "AIC": "AIV", coreRunReadyCnt,
            context_->corePendReadyCnt_[static_cast<int>(type)], taskCount);
        while (sendCnt < static_cast<uint64_t>(coreRunReadyCnt) && sendCnt < taskCount) {
            DEV_VERBOSE_DEBUG("  ## send task use runready core %u.",
                context_->runReadyCoreIdx_[static_cast<int>(type)][context_->coreRunReadyCnt_[static_cast<int>(type)] - 1]);
            SendTaskToAiCore(type,
                context_->runReadyCoreIdx_[static_cast<int>(type)][--context_->coreRunReadyCnt_[static_cast<int>(type)]],
                isLifo ? *newTask-- : *newTask++);
            sendCnt++;
        }
        context_->corePendReadyCnt_[static_cast<int>(type)] -= sendCnt;

        uint32_t idx = context_->lastPendReadyCoreIdx_[static_cast<int>(type)];
        uint32_t coreNum = coreIdxEnd - coreIdxStart;
        uint32_t lastProcCore = idx;
        DEV_VERBOSE_DEBUG("  ## send task left pend ready cnt %u , last core index:%u.",
            context_->corePendReadyCnt_[static_cast<int>(type)], idx);
        while (context_->corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount) {
            if (pendingIds_[idx] == AICORE_TASK_INIT) {
                DEV_VERBOSE_DEBUG("  ## send task use pendready core %u.", idx);
                SendTaskToAiCore(type, idx, isLifo ? *newTask-- : *newTask++);
                sendCnt++;
                context_->corePendReadyCnt_[static_cast<int>(type)]--;
                lastProcCore = idx;
            }
            idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
        }

        if (lastProcCore != context_->lastPendReadyCoreIdx_[static_cast<int>(type)]) {
            context_->lastPendReadyCoreIdx_[static_cast<int>(type)] = coreIdxStart + (lastProcCore - coreIdxStart + 1) % coreNum;
        }
        DEV_VERBOSE_DEBUG("  ## finish send task left runreadycnt:%u pendreadycnt %u, last coreindex:%u.",
            context_->coreRunReadyCnt_[static_cast<int>(type)], context_->corePendReadyCnt_[static_cast<int>(type)], idx);
        return sendCnt;
    }

    inline int32_t DispatchAiCoreTask(CoreType type, ReadyCoreFunctionQueue* readyQue,
                                       int coreIdxStart, int coreIdxEnd) {
        int32_t ret = DEVICE_MACHINE_OK;
        if (context_->waitTaskCnt_[static_cast<int>(type)] > 0) {
            ret = ResolveDepForAllAiCore(type, coreIdxStart, coreIdxEnd);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            wrapManager_.DispatchMixCoreTask();
        }
        TryBatchSendTask(type, readyQue, coreIdxStart, coreIdxEnd);
        if (enableFairSch_) {
            if (context_->coreRunReadyCnt_[static_cast<int>(type)] > 0)  {
                AicpuIsIdle(type);
            } else {
                AicpuIsBusy(type);
            }
        }
        return ret;
    }

    inline void SendTaskToAiCore(CoreType type, int coreIdx, uint64_t newTask) {
        DEV_TRACE_DEBUG(LEvent(
            LUid(curTaskCtrl_->taskId, FuncID(newTask), GetRootIndex(newTask), TaskID(newTask), GetLeafIndex(newTask)),
            LActStart(coreIdx)));
        aicoreHal_.SetReadyQueue(coreIdx, (newTask + 1) & 0xFFFFFFFF);
        pendingIds_[coreIdx] = newTask;
        pendingResolveIndexList_[coreIdx] = 0;
        context_->sendCnt_[static_cast<int>(type)]++;

        if (isFirstTaskSend_) {
            PerfMtTrace(PERF_TRACE_DEV_TASK_SEND_FIRST_CALLOP_TASK, aicpuIdx_);
            isFirstTaskSend_ = false;
        }

        DEV_IF_VERBOSE_DEBUG {
            sendTask_[coreIdx].push_back(TaskInfo(coreIdx, newTask));
        }
        DEV_VERBOSE_DEBUG("Send task %lu, at core %d ,type:%d.", newTask, coreIdx, static_cast<int>(type));
    }

    inline void SetAiCpuStat(int coreIdx, uint64_t taskId) {
        struct AiCpuTaskStat aiCpuTaskStat;
        aiCpuTaskStat.taskId = taskId;
        aiCpuTaskStat.coreId = aicoreHal_.GetPhyIdByBlockId(coreIdx);
        aicoreProf_.AsmCntvc(aiCpuTaskStat.taskGetStart);
        aicoreProf_.SetAiCpuTaskStat(taskId, aiCpuTaskStat);
    };

    inline int32_t PushReadyQue(ReadyCoreFunctionQueue *readyQue, void *idList, uint32_t idCnt) const {
        ReadyQueueLock(readyQue);
        memcpy_s(
            &readyQue->elem[readyQue->tail], idCnt * sizeof(uint32_t), (uint8_t *)idList, idCnt * sizeof(uint32_t));
         __atomic_fetch_add(&readyQue->tail, idCnt, std::memory_order_release);
        DEV_IF_NONDEVICE {
            if (readyQue->tail > readyQue->capacity){
                DEV_ERROR("readyQue tail=%u > readyQue capacity=%u", readyQue->tail, readyQue->capacity);
                return DEVICE_MACHINE_ERROR;
            }
            DEV_ASSERT(readyQue->tail <= readyQue->capacity);
        }
        ReadyQueueUnLock(readyQue);
        return DEVICE_MACHINE_OK;
    }
    inline int32_t ResolveDepForAllAiCore(CoreType type, int coreIdxStart, int coreIdxEnd) {
        int32_t ret = DEVICE_MACHINE_OK;
        PerfMtBegin(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            if ((runningIds_[i] != AICORE_TASK_INIT || pendingIds_[i] != AICORE_TASK_INIT)) {
                ret = ResolveByRegVal(type, i);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
                if (enableFairSch_) {
                    if (readyAicCoreFunctionQue_->tail - readyAicCoreFunctionQue_->head == 0 ||
                        readyAivCoreFunctionQue_->tail - readyAivCoreFunctionQue_->head == 0) {
                        ret = BatchPushReadyQueue();
                        if (unlikely(ret != DEVICE_MACHINE_OK)) {
                            return ret;
                        }
                    }
                }
            }
        }

        ret = BatchPushReadyQueue();
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        PerfMtEnd(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        return ret;
    }

    inline int32_t BatchPushReadyQueue() {
        int32_t ret = DEVICE_MACHINE_OK;
        uint32_t aicIndex = static_cast<uint32_t>(CoreType::AIC);
        uint32_t aivIndex = static_cast<uint32_t>(CoreType::AIV);
        if (context_->readyCount[aicIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIC), context_->readyCount[aicIndex]);
            if (needSendCnt > 0) {
                context_->readyCount[aicIndex] -= BatchSendTask(CoreType::AIC, &context_->readyIds[aicIndex][context_->readyCount[aicIndex] - 1],
                    needSendCnt, aicStart_, aicEnd_, true);
            }
            DEV_VERBOSE_DEBUG("resolved new task, aic ready count: %u coretype:%u.", context_->readyCount[aicIndex], aicIndex);
            if (context_->readyCount[aicIndex] > 0) {
                ret = PushReadyQue(readyAicCoreFunctionQue_, context_->readyIds[aicIndex], context_->readyCount[aicIndex]);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            context_->readyCount[aicIndex] = 0;
        }

        if (context_->readyCount[aivIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIV), context_->readyCount[aivIndex]);
            if (needSendCnt > 0) {
                context_->readyCount[aivIndex] -= BatchSendTask(CoreType::AIV, &context_->readyIds[aivIndex][context_->readyCount[aivIndex] - 1],
                    needSendCnt, aivStart_, aivEnd_, true);
            }
            DEV_VERBOSE_DEBUG("resolved new task, aiv ready count: %u coretype: %u.", context_->readyCount[aivIndex], aivIndex);
            if (context_->readyCount[aivIndex] > 0) {
                ret = PushReadyQue(readyAivCoreFunctionQue_, context_->readyIds[aivIndex], context_->readyCount[aivIndex]);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            context_->readyCount[aivIndex] = 0;
        }
        return ret;
    }

    inline int32_t ResolveDepForAicpuTask(uint64_t& taskCount) {
        int32_t ret = aicpuTaskManager_.TaskProcess(taskCount);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return aicpuTaskManager_.TaskPoll(*this);
    }

    inline int32_t ResolveWhenSyncMode(CoreType type, uint32_t finTaskId, uint32_t finTaskState, int coreIdx)  {
        int32_t ret = DEVICE_MACHINE_OK;
        if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_VERBOSE_DEBUG("core index: %d, PendingTask Finished."
                " pending: %x.", coreIdx, pendingIds_[coreIdx]);
            ret = ResolveDepWithDfx(type, coreIdx, finTaskId);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            context_->corePendReadyCnt_[static_cast<int>(type)]++;
            context_->runReadyCoreIdx_[static_cast<int>(type)][context_->coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
        }
        return ret;
    }

    static uint64_t RuntimeCopyOutResolveCounterDecode(uint64_t aicpuCallCode) {
        return aicpuCallCode & 0xffff;
    }

    inline int32_t ResolveByRegVal(CoreType type, int coreIdx) {
        int32_t ret = DEVICE_MACHINE_OK;
        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreIdx);
        [[maybe_unused]] uint32_t aicpuCallCode = finTaskRegVal >> 32;
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);
        DEV_VERBOSE_DEBUG("reslove task core index: %d, finishtaskid:%x, finishstate: %u.", coreIdx, finTaskId, finTaskState);
#if SCHEDULE_USE_PENDING_AND_RUNING_SWITCH
        auto &pendingIdRef = pendingIds_[coreIdx];
        auto &pendingResolveIndexBaseRef = pendingResolveIndexList_[coreIdx];
        auto &runningIdRef = runningIds_[coreIdx];
        auto &runningResolveIndexBaseRef = runningResolveIndexList_[coreIdx];
        bool isWrapCoreAvailable = wrapManager_.GetWrapCoreAvailable(coreIdx);
        if (likely(finTaskId == pendingIdRef && finTaskState == TASK_FIN_STATE)) {
            // pending task is finished, resolve both running and pending task.
            DEV_VERBOSE_DEBUG("Pending Finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            if (isWrapCoreAvailable) { // wrapcore doesnt support pending & running yet
                context_->runReadyCoreIdx_[static_cast<int>(type)][context_->coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (runningIdValue != AICORE_TASK_INIT) {
                ret = ResolveDepWithDfx(type, coreIdx, runningIdValue, runningResolveIndexBaseValue);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            ret = ResolveDepWithDfx(type, coreIdx, pendingIdValue, pendingResolveIndexBaseValue);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else if (unlikely(finTaskId == pendingIdRef && aicpuCallCode != 0)) {
            // pending task is copyout, reolve both running and pending task.
            DEV_VERBOSE_DEBUG("Pending Copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValueCopyout = runningIdRef;
            int runningResolveIndexBaseValueCopyout = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = pendingIdRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            if (isWrapCoreAvailable) {
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (runningIdValueCopyout != AICORE_TASK_INIT) {
                ret = ResolveDepWithDfx(type, coreIdx, runningIdValueCopyout, runningResolveIndexBaseValueCopyout);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            ret = ResolveCopyOutDepDyn(copyOutResolveCounter, pendingIdValue, pendingResolveIndexBaseValue);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else if (finTaskId == pendingIdRef && finTaskState == TASK_ACK_STATE) {
            // pending task is acknowledged, resolve running task. And move pending to running
            DEV_VERBOSE_DEBUG("Pending Acknowledged: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            DEV_IF_VERBOSE_DEBUG {
                recvAckTask_[coreIdx].push_back(TaskInfo(coreIdx, finTaskId));
            }
            uint32_t runningIdValueAck = runningIdRef;
            int runningResolveIndexBaseValueAck = runningResolveIndexBaseRef;
            runningIdRef = finTaskId;
            runningResolveIndexBaseRef = pendingResolveIndexBaseRef;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            if (isWrapCoreAvailable) {
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (runningIdValueAck != AICORE_TASK_INIT) {
                ret = ResolveDepWithDfx(type, coreIdx, runningIdValueAck, runningResolveIndexBaseValueAck);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        } else if (finTaskId == runningIdRef && finTaskState == TASK_FIN_STATE) {
            // running task is finished, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG("Running finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            if (isWrapCoreAvailable && pendingIdRef == AICORE_TASK_INIT) {
                context_->runReadyCoreIdx_[static_cast<int>(type)][context_->coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            }
            ret = ResolveDepWithDfx(type, coreIdx, runningIdValue, runningResolveIndexBaseValue);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else if (unlikely(finTaskId == runningIdRef && aicpuCallCode != 0)) {
            // running task is copyout, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG("Running copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            ret = ResolveCopyOutDepDyn(copyOutResolveCounter, runningIdValue, runningResolveIndexBaseValue);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else {
            DEV_VERBOSE_DEBUG("Warning, maybe inconsistent state. coreidx: %d,finTask: %lx,pending: %x,running: %x.", coreIdx, finTaskRegVal, pendingIdRef, runningIdRef);
        }
#else
        ret = ResolveWhenSyncMode(type, finTaskId, finTaskState, coreIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
#endif
        return ret;
    }

    inline void PushAicpuTaskQueue(uint64_t taskId) {
        PushReadyQue(readyAicpuFunctionQue_, &taskId, 1);
    }

    inline bool TrySendTaskDirectly(int coreType, uint32_t taskId) {
        if (context_->coreRunReadyCnt_[coreType] > 0) {
            context_->corePendReadyCnt_[coreType]--;
            DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
            SendTaskToAiCore(static_cast<CoreType>(coreType),
                context_->runReadyCoreIdx_[coreType][--context_->coreRunReadyCnt_[coreType]], taskId);
            return true;
        }

        if (context_->corePendReadyCnt_[coreType] == 0) {
            return false;
        }

        if (enableFairSch_ && IsExistOtherAicpuIdle(static_cast<CoreType>(coreType))) {
            return false;
        }

        int startIdx;
        int coreNum;
        int idx = static_cast<int>(context_->lastPendReadyCoreIdx_[coreType]);
        if (coreType == static_cast<int>(CoreType::AIC)) {
            startIdx = aicStart_;
            coreNum = aicEnd_ - aicStart_;
        } else {
            startIdx = aivStart_;
            coreNum = aivEnd_ - aivStart_;
        }
        while (pendingIds_[idx] != AICORE_TASK_INIT) {
            idx = startIdx + (idx - startIdx + 1) % (coreNum);
        }
        context_->lastPendReadyCoreIdx_[coreType] = static_cast<uint32_t>(startIdx + (idx - startIdx + 1) % (coreNum));
        context_->corePendReadyCnt_[coreType]--;
        DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
        SendTaskToAiCore(static_cast<CoreType>(coreType), idx, taskId);
        return true;
    }

    inline int32_t PushReadyTask(int coreType, uint64_t taskId) {
        int32_t ret = DEVICE_MACHINE_OK;
        if (enableL2CacheSch_ && TrySendTaskDirectly(coreType, taskId)) {
            return DEVICE_MACHINE_OK;
        }

        if (unlikely(context_->readyCount[coreType] == READY_ID_FIX_CACHE_NUM)) {
            ReadyCoreFunctionQueue* readyQue =
                coreType == static_cast<int>(CoreType::AIC) ?  readyAicCoreFunctionQue_ : readyAivCoreFunctionQue_;
            ret = PushReadyQue(readyQue, context_->readyIds[coreType], context_->readyCount[coreType]);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            context_->readyCount[coreType] = 0;
        }
        context_->readyIds[coreType][context_->readyCount[coreType]++] = taskId;
        return ret;
    }

    inline uint64_t GetCostModelTaskTime(uint64_t coreIdx, uint64_t taskId, uint64_t currentTime) {
        auto funcId = FuncID(taskId);
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto costModelData = reinterpret_cast<CostModel::ModelData*>(curDevTask_->costModelData);
        if (costModelData == nullptr) return 0;
        auto source = dyntask->GetDynFuncDataCacheList()[funcId].devFunc;
        auto opIndex = TaskID(taskId);
        auto leafFunctionIdx = source->GetOperationAttrCalleeIndex(opIndex);
        auto timeCost = costModelData->functionTime[leafFunctionIdx];
        auto header = dyntask->GetDynFuncDataList();
        auto dyndata = reinterpret_cast<DynFuncData *>(&header->At(0));
        auto opAttrs = &dyndata->opAttrs[dyndata->opAtrrOffsets[TaskID(taskId)]];
        auto psgId = opAttrs[0];
        // devTaskId - funcId - leaf function Id - psgId
        std::string name = std::to_string(curTaskId_) + '-' + std::to_string(funcId) + '-' +
                           std::to_string(opIndex) + '-' + std::to_string(psgId);
        PerfMtEvent(PERF_EVT_TASK, coreIdx + PERF_AICORE_THREAD_START, currentTime, currentTime + timeCost, name);
        return timeCost;
    }

    inline int32_t ResolveDynStitched(DynDeviceTask *dyntask, int origfunc, int origop) {
        int32_t ret = DEVICE_MACHINE_OK;
        auto &duppedData = dyntask->GetDynFuncDataCacheList()[origfunc].duppedData;
        auto &stitchList = duppedData->GetOperationStitch(origop);
        auto cceBinary = dyntask->cceBinary;

        for (auto *node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            for (uint32_t i = 0; i < listSize; i++) {
                uint32_t id = node->At(i);
                auto funcId = FuncID(id);
                auto opIndex = TaskID(id);
                auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
                if (predCounts[opIndex] == 1 ||
                    __atomic_sub_fetch(&predCounts[opIndex], 1, __ATOMIC_RELAXED) == 0) {
                    auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
                    auto coreType = cceBinary[callList[opIndex]].coreType;
                    if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                        ret = ResolveDepDyn(id);
                        if (unlikely(ret != DEVICE_MACHINE_OK)) {
                            return ret;
                        }
                        context_->resolveHubCnt_++;
                    } else if (coreType == static_cast<int>(MachineType::AICPU)){
                        PushAicpuTaskQueue(id);
                    } else {
                        ret = PushReadyTask(static_cast<int>(coreType), id);
                        if (unlikely(ret != DEVICE_MACHINE_OK)) {
                            return ret;
                        }
                    }
                }
            }
        }
        return ret;
    }

    inline int GetRootIndex(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        return func->GetRootIndex();
    }

    inline int GetLeafIndex(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return callList[opIndex];
    }

    inline DevAscendFunctionDuppedData *GetDuppedData(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        return dyntask->dynFuncDataCacheList[funcId].duppedData;
    }

    inline int32_t ResolveDepDyn(uint64_t finishId, size_t resolveIndexBase = 0, int coreIdx = 0) {
        int32_t ret = DEVICE_MACHINE_OK;
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(finishId);
        auto opIndex = TaskID(finishId);

        wrapManager_.UpdateFinishIdForMixCore(finishId, coreIdx);
        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts =  dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int *succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        for (size_t i = succIndexList[resolveIndexBase]; i < succSize; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 ||
                __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(id, resolveIndexBase, coreIdx);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    context_->resolveHubCnt_++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))){
                    PushAicpuTaskQueue(id);
                } else if (wrapManager_.IsBindedWrapId(id)) {
                    wrapManager_.ResolveDepForMixCore(id);
                } else {
                    ret = PushReadyTask(static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }

        ret = ResolveDynStitched(dyntask, funcId, opIndex);
        return ret;
    }

    inline int32_t ResolveCopyOutDepDyn(uint32_t currResolveIndex, uint64_t taskId, uint32_t resolveIndexBase) {
        int32_t ret = DEVICE_MACHINE_OK;
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts =  dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int *succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        const int *succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        // here we don't use resolveIndexBase + 1, because at the beginning, resolveIndexBase is 0. And we resolve from 0.
        for (int i = succIndexList[resolveIndexBase]; i < succIndexList[currResolveIndex + 1]; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 ||
                __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    context_->resolveHubCnt_++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))){
                    PushAicpuTaskQueue(id);
                } else {
                    ret = PushReadyTask(static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }

    inline int32_t ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId, size_t resolveIndexBase = 0) {
        int32_t ret = DEVICE_MACHINE_OK;
        ret = ResolveDepDyn(finishId, resolveIndexBase, coreIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        DEV_VERBOSE_DEBUG("[Call]: Core %d Dispatch Task: %lu, %u, %u", coreIdx, seq,
                  FuncID(finishId), TaskID(finishId));
        DfxProcAfterFinishTask(coreIdx, finishId);
        context_->waitTaskCnt_[static_cast<int>(type)]--;
        return ret;
    }

    inline bool IsExistOtherAicpuIdle(CoreType type) {
        int idx = (schedIdx_ + 1) % aicpuNum_;
        while (idx != schedIdx_) {
            if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][idx].load(std::memory_order_relaxed) == true){
                return true;
            }
            idx = (idx + 1) % aicpuNum_;
        }
        return false;
    }

    inline void AicpuIsBusy(CoreType type) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][schedIdx_] != false) {
            curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][schedIdx_].store(false, std::memory_order_relaxed);
        }
    }

    inline void AicpuIsIdle(CoreType type) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][schedIdx_] != true) {
            curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][schedIdx_].store(true, std::memory_order_relaxed);
        }
    }

    inline void Init(int threadIdx, DeviceArgs *deviceArgs, int schedIdx) {
        aicNum_ = static_cast<int32_t>(deviceArgs->nrAic);
        aivNum_ = static_cast<int32_t>(deviceArgs->nrAiv);
        aicpuNum_ = deviceArgs->scheCpuNum;
        aicpuIdx_ = threadIdx;
        schedIdx_ = schedIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        aicoreHal_.Init(deviceArgs, &aicoreProf_);
        validGetPgMask_ = deviceArgs->validGetPgMask;
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
        runningResolveIndexList_.fill(0);
        pendingResolveIndexList_.fill(0);
        taskDfxStatPos_.fill(REG_LOW_TASK_PING);
        isSendStop = false;
        if (IsNeedProcAicpuTask()) {
            aicpuTaskManager_.InitDeviceArgs(deviceArgs);
        }

        wrapManager_.InitArchInfo(deviceArgs->archInfo);
        if (deviceArgs->machineConfig != static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH)) {
            if (aicpuNum_ > 1) {
                enableFairSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                    static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH);
            }
            enableL2CacheSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH);
        }
        UpdateAiCoreBlockIndexSection();
        if constexpr (IsDeviceMode()) {
            aicoreHal_.MapRegistersForAllCores(aicNum_);
            aicoreProf_.ProfInit(reinterpret_cast<int64_t *>(deviceArgs->corePmuRegAddr),
               reinterpret_cast<int64_t *>(deviceArgs->pmuEventAddr),
               deviceArgs->toSubMachineConfig.profConfig, deviceArgs->archInfo);
        } else {
            aicoreHal_.SetTaskTimeCost([this](uint64_t coreIdx, uint64_t taskId, uint64_t time)
                {return GetCostModelTaskTime(coreIdx, taskId, time); });
        }
        firstLock[static_cast<int>(CoreType::AIC)] = true;
        firstLock[static_cast<int>(CoreType::AIV)] = true;
        preFetchSuccess_ = false;
        preFetchNextDevTaskCtrl_ = nullptr;
        DEV_INFO("Init aicore manager aicNum_ %d aivNum_  %d sch_aicpuNum_ %d  aicpuIdx_ %d "
                  "aicValidNum_ %d aicoreHal_.regAddrs_ %p sharedBuffer_ %p machineConfig: %u.",
            aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, aicoreHal_.GetRegAddrs(),
            (void *)aicoreHal_.GetSharedBuffer(), static_cast<uint8_t>(deviceArgs->machineConfig));
    }

    inline int HandShakeByGm() {
        int rc = ForEachManageAicoreWithRet([this](int coreIdx) -> int {
            int ret = aicoreHal_.HandShakeByGm(coreIdx, dotStatus_);
            DEV_VERBOSE_DEBUG("coreidx %d handshake by gm phycorid %d.",
                coreIdx, aicoreHal_.GetPhyIdByBlockId(coreIdx));
            return ret;
        });

        return rc;
    }

    inline void HandShakeTryPreFetchDevTask(bool &needSendAic, bool &needSendAiv) {
        if (!preFetchSuccess_ && PreFetchNextDevTask()) {
            preFetchNextDevTaskCtrl_->isFirstDevTask = true;
            readyAicCoreFunctionQue_ =  reinterpret_cast<ReadyCoreFunctionQueue*>(preFetchNextDevTaskCtrl_->devTask->readyAicCoreFunctionQue);
            readyAivCoreFunctionQue_ =  reinterpret_cast<ReadyCoreFunctionQueue*>(preFetchNextDevTaskCtrl_->devTask->readyAivCoreFunctionQue);
            needSendAic = (readyAicCoreFunctionQue_->tail != readyAicCoreFunctionQue_->head);
            needSendAiv = (readyAivCoreFunctionQue_->tail != readyAivCoreFunctionQue_->head);
            SendDevTaskModel(preFetchNextDevTaskCtrl_->devTask);
            DEV_DEBUG("hand shake prefetch dev task success :needsendaic %d needsend aiv %d", needSendAic, needSendAiv);
        }
    }

    inline void HandShakePostProc(bool needSendAic, bool needSendAiv) {
        // send task by left ready core
        if (needSendAic) {
            __sync_synchronize();
            TryBatchSendTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
        }
        if (needSendAiv) {
            __sync_synchronize();
            TryBatchSendTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);
        }

        if (preFetchSuccess_) {
            uint64_t sentAic = 0;
            uint64_t sentAiv = 0;
            CountSendTask(sentAic, sentAiv);
            if (sentAic + sentAiv > 0) {
                preFetchNextDevTaskCtrl_->finishedFunctionCnt.fetch_add(sentAic + sentAiv, std::memory_order_relaxed);
            }
            DEV_DEBUG("hand shake presend task cnt : aic %lu, aiv %lu", sentAic, sentAiv);
        }
    }

    inline void DumpAicoreStatusWhenTimeout(bool *handFlag) {
        for (int i = aicStart_; i < aicEnd_; i++) {
            if (handFlag[i]) {
                DEV_ERROR("Aic core %d hand shake success.phyid %d ", i, aicoreHal_.GetPhyIdByBlockId(i));
            } else {
                DEV_ERROR("Aic core %d hand shake timeout status=%lu",  i, aicoreHal_.GetAicoreStatus(i));
            }
        }

        for (int i = aivStart_; i < aivEnd_; i++) {
            if (handFlag[i]) {
                DEV_ERROR("Aiv core %d hand shake success.phyid %d ", i,  aicoreHal_.GetPhyIdByBlockId(i));
            } else {
                DEV_ERROR("Aiv core %d hand shake timeout status=%lu", i, aicoreHal_.GetAicoreStatus(i));
            }
        }
    }

    inline int HandShakeByGmWithPreSendTask() {
        int handShakeNum = 0;
        int mngAicoreNum = aicEnd_ - aicStart_ + aivEnd_ - aivStart_;
        bool handFlag[MAX_AICORE_NUM] = {false};
        uint64_t start_cycles = GetCycles();
        bool needSendAic = false;
        bool needSendAiv = false;
        bool aicAllSuccess = false;
        bool aivAllSuccess = false;
        int aicSucessCnt = 0;
        int aivSucessCnt = 0;
        int aicTreshold = 4;
        int aivThreshold = 4;
        while (handShakeNum < mngAicoreNum) {
            HandShakeTryPreFetchDevTask(needSendAic, needSendAiv);

            bool curIterAllAicSuccess = true;
            bool curIterAllAivSuccess = true;
            for (int i = aicEnd_ - 1; (!aicAllSuccess) && i >= aicStart_; i--) {
                if (handFlag[i]) {
                    continue;
                }
                if (aicoreHal_.TryHandShakeByGm(i, dotStatus_)) {
                    handShakeNum++;
                    aicSucessCnt++;
                    handFlag[i] = true;
                    context_->corePendReadyCnt_[static_cast<int>(CoreType::AIC)]++;
                    context_->runReadyCoreIdx_[static_cast<int>(CoreType::AIC)][context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIC)]++] = i;
                } else {
                    curIterAllAicSuccess = false;
                }
            }
            aicAllSuccess = curIterAllAicSuccess;

            if (needSendAic && aicSucessCnt >= aicTreshold) {
                __sync_synchronize(); // sync  REG_SPR_FAST_PATH_ENABLE
                TryBatchSendTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
                aicSucessCnt = 0;
            }

            for (int i = aivEnd_ - 1; (!aivAllSuccess) && i >= aivStart_; i--) {
                if (handFlag[i]) {
                    continue;
                }
                if (aicoreHal_.TryHandShakeByGm(i, dotStatus_)) {
                    handShakeNum++;
                    aivSucessCnt++;
                    handFlag[i] = true;
                    context_->corePendReadyCnt_[static_cast<int>(CoreType::AIV)]++;
                    context_->runReadyCoreIdx_[static_cast<int>(CoreType::AIV)][context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIV)]++] = i;
                } else {
                    curIterAllAivSuccess = false;
                }
            }
            aivAllSuccess = curIterAllAivSuccess;

            if (needSendAiv && aivSucessCnt >= aivThreshold) {
                __sync_synchronize();
                TryBatchSendTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);
                aivSucessCnt = 0;
            }

            if (GetCycles() - start_cycles > HAND_SHAKE_TIMEOUT) {
                DumpAicoreStatusWhenTimeout(handFlag);
                DEV_ERROR("HandShakeByGmWithPreSendTask timeout notHandshakeNum=%d.", mngAicoreNum - handShakeNum);
                return DEVICE_MACHINE_ERROR;
            }
        }

        HandShakePostProc(needSendAic, needSendAiv);
        return DEVICE_MACHINE_OK;
    }

    inline int HandShake() {
        DEV_INFO("aicpu %d handshake start.", aicpuIdx_);
        int rc = HandShakeByGmWithPreSendTask();
        if (rc != DEVICE_MACHINE_OK) {
            DEV_ERROR("Aicpu %d handshake failed end.", aicpuIdx_);
            return rc;
        }

        DEV_INFO("Aicpu %d handshake sucess end.", aicpuIdx_);
        return 0;
    }

    /* assign aic and aiv core index section for this aicpu */
    inline void UpdateAiCoreBlockIndexSection() {
        auto f = [](int total, int idx, int part, int &start, int &end) {
            int perCpu = total / part;
            int remain = total % part;
            start = idx * perCpu + ((idx < remain) ? idx : remain);
            end = start + perCpu + ((idx < remain) ? 1 : 0);
        };

        f(aicValidNum_, schedIdx_, aicpuNum_, aicStart_, aicEnd_);
        f(AIV_NUM_PER_AI_CORE * aicValidNum_, schedIdx_, aicpuNum_, aivStart_, aivEnd_);
        aivStart_ += aicValidNum_;
        aivEnd_ += aicValidNum_;

        DEV_IF_NONDEVICE {
            context_->corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = aicEnd_ - aicStart_;
            context_->corePendReadyCnt_[static_cast<int>(CoreType::AIV)] = aivEnd_ - aivStart_;
            ForEachManageAicoreReverse(
                [this](int coreIdx) {
                int coreType = static_cast<int>(AicoreType(coreIdx));
                context_->runReadyCoreIdx_[coreType][context_->coreRunReadyCnt_[coreType]++] = coreIdx;
                });
        }

        context_->lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIV)] = static_cast<uint32_t>(aivStart_);
        context_->lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIC)] = static_cast<uint32_t>(aicStart_);
        aicoreHal_.SetMngCoreBlockId(aicStart_, aicEnd_, aivStart_, aivEnd_);
        DEV_DEBUG("assign core aic coreindex section: start %d end %d .", aicStart_, aicEnd_);
        DEV_DEBUG("assign core aiv coreindex section: start %d end %d .", aivStart_, aivEnd_);
    }

    inline int GetPhyIdByBlockId(int coreIdx) {
        return aicoreHal_.GetPhyIdByBlockId(coreIdx);
    }

    inline void ForEachManageAicore(std::function<void(int coreIdx)> func) const {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            func(i);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            func(i);
        }
    }

    inline void ForEachManageAicoreReverse(std::function<void(int coreIdx)> func) const {
        for (int i = aicEnd_ - 1; i >= aicStart_; --i) {
            func(i);
        }
        for (int i = aivEnd_ -1; i >= aivStart_ ; --i) {
            func(i);
        }
    }

    inline int ForEachManageAicoreWithRet(std::function<int(int coreIdx)> func) const {
        int ret = DEVICE_MACHINE_OK;
        for (int i = aicStart_; i < aicEnd_; ++i) {
            ret = func(i);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR("proc aicore aic %d failed.", i);
                return ret;
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            ret = func(i);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR("proc aicore aiv %d failed.", i);
                return ret;
            }
        }
        return ret;
    }

    inline void AbnormalStop() {
        ResetRegAll();
        CheckAndResetReg();
        DEV_INFO("aicore manager %d abnormal stopped.", aicpuIdx_);
    }

    inline void NormalStop() {
        DEV_INFO("aicore manager %d try normal stop .", aicpuIdx_);
        ForEachManageAicore([this](auto coreIdx) { aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1) ; });
        /* write to MAINBASE reg must be done before close 0x18 */
        __sync_synchronize();
        ForEachManageAicore([this](auto coreIdx) {
            aicoreHal_.ResetShakeBuf(coreIdx);
        });
        DEV_INFO("aicore manager %d normal stopped .", aicpuIdx_);
    }

    inline void NormalStopSingleCore(int coreIdx) {
        aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
        __sync_synchronize();
        aicoreHal_.ResetShakeBuf(coreIdx);
    }

    inline int GetAllAiCoreNum() { return aicNum_ + aivNum_; }
    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }
    inline void SetNextDfxPos(int coreIdx) {
            taskDfxStatPos_[coreIdx] =
                taskDfxStatPos_[coreIdx] == REG_LOW_TASK_PING ? REG_LOW_TASK_PONG : REG_LOW_TASK_PING;
    }
    inline int GetDfxPos(int coreIdx) { return taskDfxStatPos_[coreIdx]; }

    // DFX
    inline void DfxProcAfterFinishTask(int coreIdx, uint64_t taskId) {
        DEV_TRACE_DEBUG(LEvent(
            LUid(curTaskCtrl_->taskId, FuncID(taskId), GetRootIndex(taskId), TaskID(taskId), GetLeafIndex(taskId)),
            LActFinish(coreIdx)));
        if constexpr (!IsDeviceMode())
            return;

#if ENABLE_AICORE_PRINT
        DumpAicoreLog(coreIdx);
#endif

        volatile TaskStat *stat = aicoreHal_.GetTaskStat(coreIdx, 0);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        aicoreProf_.ProfGet(coreIdx, stat->subGraphId, stat->taskId, const_cast<TaskStat*>(stat));
#endif

        DEV_IF_VERBOSE_DEBUG {
            recvFinTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId));
        }

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        SetNextDfxPos(coreIdx); // pingpong 存储
#endif
    (void)stat;
    }

    inline bool IsNeedProcAicpuTask() {
        return aicpuIdx_ == 2;
    }

private:
    uint64_t seq;
    AicoreHAL aicoreHal_;
    bool isFirstTaskSend_{true};
    bool firstLock[AICORE_TYPE_NUM]{true,true};
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int schedIdx_{0};
    int aicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    int aicStart_{0};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    uint64_t procAicCoreFunctionCnt_{0};
    uint64_t procAivCoreFunctionCnt_{0};
    uint64_t procAicpuFunctionCnt_{0};
    bool enableL2CacheSch_{false};
    bool enableFairSch_{false};
    bool validGetPgMask_{true};

    DeviceTask* curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};
    int curTaskType_{0};
    int curTaskId_{0};

    std::array<uint32_t, MAX_AICORE_NUM> runningIds_;
    std::array<uint32_t, MAX_AICORE_NUM> pendingIds_;
    std::array<int, MAX_AICORE_NUM> runningResolveIndexList_;
    std::array<int, MAX_AICORE_NUM> pendingResolveIndexList_;

    /* prepare aicore ready task list */
    ReadyCoreFunctionQueue* readyAicCoreFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* readyAivCoreFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* readyAicpuFunctionQue_{nullptr};
    WrapManager wrapManager_;
    SchduleContext * context_{nullptr};

    bool preFetchSuccess_{false};
    DeviceTaskCtrl* preFetchNextDevTaskCtrl_{nullptr};

    std::array<int, MAX_AICORE_NUM> taskDfxStatPos_;

    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> *taskQueue_{nullptr};
    AicpuTaskManager &aicpuTaskManager_;
    AiCoreProf aicoreProf_;
    int64_t dotStatus_{0};
    bool isSendStop{false};

    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];

    AicoreLogger *logger_{nullptr};
    friend class AiCoreProf;
};
}
