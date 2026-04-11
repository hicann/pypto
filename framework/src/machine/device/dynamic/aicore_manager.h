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
#include <map>
#include <atomic>
#include <array>
#include <semaphore.h>
#include "machine/utils/dynamic/dev_start_args.h"
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
#include "machine/device/dump/aicore_dump.h"
#include "device_sche_context.h"

namespace npu::tile_fwk::dynamic {

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS            // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal) & REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1;                            // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0;                            // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2;                              // 一次寄存器task个数

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    uint32_t devTaskId;
    TaskInfo(int idx, uint64_t id, uint32_t devtaskid) : coreIdx(idx), taskId(id), devTaskId(devtaskid) {}
};
struct ResolveTaskContext {
    uint32_t finishIds{0};
    uint32_t resolveIndexBase{0};
    int finishCoreIdx{0};
};
class AiCoreManager {
public:
    explicit AiCoreManager(SchThreadStatus& schThreadStatus, AicpuTaskManager& aicpuTaskManager)
        : threadStatus(schThreadStatus), aicpuTaskManager_(aicpuTaskManager), aicoreProf_(*this) {};
    ~AiCoreManager() {};

    void InitLogger(AicoreLogger* logger) { logger_ = logger; }

    void InitCostModelFuncData(SchDeviceTaskContext* devTaskCtx)
    {
        int64_t funcdata;
        auto dyntask = (DynDeviceTask *)devTaskCtx->GetDeviceTask();
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        ForEachManageAicore([&](int coreIdx) {
            aicoreHal_.InitCostModelDevTaskData(coreIdx, funcdata);
        });
    }

    void InitCostModelFuncDataForOneCore(SchDeviceTaskContext* devTaskCtx, int coreIdx)
    {
        auto dyntask = (DynDeviceTask*)devTaskCtx->GetDeviceTask();
        aicoreHal_.InitCostModelDevTaskData(coreIdx, static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList())));
    }

    void InitAicoreParallelDevTask(ParallelSchDeviceTaskContext* parallelCtx)
    {
        DEV_IF_DEVICE {
            ForEachManageAicore([&](int coreIdx) {
                auto logbuf = logger_ ? logger_[coreIdx].GetBuffer() : nullptr;
                aicoreHal_.InitKernelArgs(coreIdx,  reinterpret_cast<int64_t>(logbuf));
                FillKernelArgsParallexDevTask(parallelCtx, coreIdx);
            });
        }
    }

    void FillKernelArgsParallexDevTask(ParallelSchDeviceTaskContext* parallelCtx, int coreIdx)
    {
        volatile ParallelDevTask* kernelParallDevTask = aicoreHal_.GetParallelDevTask(coreIdx);
        for (uint32_t idx = parallelCtx->front; idx < parallelCtx->rear; idx++) {
            auto dyntask = (DynDeviceTask *)(parallelCtx->Element(idx)->GetDeviceTask());
            aicoreHal_.SetParallelDevTask(
                kernelParallDevTask, idx, static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList())));
        }
        aicoreHal_.SetParallelDevTaskSize(kernelParallDevTask, parallelCtx->front, parallelCtx->rear);
        aicoreHal_.SetParallelDevTaskCtxVersion(coreIdx, parallelCtx->Version());
        DEV_VERBOSE_DEBUG("Fill prallel dev task for core %d, ver:%u", coreIdx, parallelCtx->Version());
    }

    inline void SetSchduleContext(SchduleContext* context)
    {
        this->context_ = context;
    }

    inline bool CheckAndResetReg()
    {
        if (!validGetPgMask_) {
            return true;
        }
        bool isValid = true;
        DEV_IF_DEVICE
        {
            if (aicoreHal_.GetRegSprDataMainBase() == DAV_3510::REG_SPR_DATA_MAIN_BASE) {
                return true;
            }
            auto regAddrs = aicoreHal_.GetRegAddrs();
            uint32_t regNum = aicoreHal_.GetregNum();
            for (uint32_t coreIdx = 0; coreIdx < regNum; ++coreIdx) {
                if (regAddrs[coreIdx] == 0) {
                    continue;
                }
                uint32_t currentStatus =
                    *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE));
                if (currentStatus != REG_SPR_FAST_PATH_CLOSE) {
                    isValid = false;
                    *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE)) =
                        REG_SPR_FAST_PATH_CLOSE;
                }
            }
        }
        return isValid;
    }

    inline void InitDevTask(SchDeviceTaskContext* deviceTaskCtx) {
        auto devTask = deviceTaskCtx->GetDeviceTask();
        aicoreHal_.SetModel(devTask->aicoreModel);
        deviceTaskCtx->wrapManager.Init(
            deviceTaskCtx, deviceTaskCtx->GetDeviceTask(), context_->coreRunReadyCnt_,
            context_->runReadyCoreIdx_[CORE_IDX_AIV], context_->runReadyCoreIdx_[CORE_IDX_AIC],
            context_->corePendReadyCnt_, pendingIds_.data(), runningIds_.data(), aicValidNum_,
            context_->coreIdxPosition_, context_->wrapCoreAvail_,
            [&](SchDeviceTaskContext* devTaskCtx, CoreType coreType, int arg1, uint64_t arg2)
            { SendTaskToAiCore(devTaskCtx, coreType, arg1, arg2); },
            [&](int coreIdx, int type) { AddReadyCoreIdx(coreIdx, type); });

        InitCostModelFuncData(deviceTaskCtx);
    }

    template <bool enableAicpuTask = false>
    inline int32_t RunCoreTask(SchDeviceTaskContext* devTaskCtx) {
        int32_t ret = DEVICE_MACHINE_OK;
        devTaskCtx->GetWrapManager().DispatchMixCoreTask();
        ret = DispatchAiCoreTask(devTaskCtx, CoreType::AIC, devTaskCtx->readyAicCoreFunctionQue, aicStart_, aicEnd_);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        ret = DispatchAiCoreTask(devTaskCtx, CoreType::AIV, devTaskCtx->readyAivCoreFunctionQue, aivStart_, aivEnd_);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }

        uint64_t aicpuTaskSent = 0UL;
        if constexpr (enableAicpuTask) {
            if (IsNeedProcAicpuTask()) {
                ret = ResolveDepForAicpuTask(aicpuTaskSent);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        }

        DEV_IF_VERBOSE_DEBUG
        {
            procAicCoreFunctionCnt_ += devTaskCtx->CurCoreTaskSent(CoreType::AIC);
            procAivCoreFunctionCnt_ += devTaskCtx->CurCoreTaskSent(CoreType::AIV);
            procAicpuFunctionCnt_ += aicpuTaskSent;
        }

        devTaskCtx->SetAicpuTaskSent(static_cast<uint32_t>(aicpuTaskSent));
        devTaskCtx->CountCoreTaskSent();
        return ret;
    }

    void DumpAicoreLog(int coreIdx)
    {
        const int bufSize = 512;
        char buf[bufSize];
        while (logger_[coreIdx].Read(buf, bufSize)) {
            DEV_INFO("core-%d %s", coreIdx, buf);
        }
    }

    inline int RunTask(SchDeviceTaskContext* deviceTaskCtx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        DEV_VERBOSE_DEBUG("Run device task entry stage : %d", ToUnderlying(deviceTaskCtx->CurStage()));
        while (true) {
            bool isStageFinish = false;
            switch (deviceTaskCtx->CurStage()) {
                case DevTaskExecStage::INIT: {
                    ret = PreProcessTask(deviceTaskCtx, isStageFinish);
                    if (isStageFinish) {
                        deviceTaskCtx->EntryStage(DevTaskExecStage::SEND_CORE_TASK);
                    }
                    break;
                }
                case DevTaskExecStage::SEND_CORE_TASK: {
                    ret = ProcessTaskLoop(deviceTaskCtx, isStageFinish);
                    if (isStageFinish) {
                        deviceTaskCtx->EntryStage(DevTaskExecStage::WAIT_TAIL_TASK_FINISH);
                        PerfMtTrace(PERF_TRACE_DEV_TASK_SCHED_EXEC, aicpuIdx_);
                    }
                    break;
                }
                case DevTaskExecStage::WAIT_TAIL_TASK_FINISH: {
                    PerfMtBegin(PERF_EVT_SYNC_AICORE, aicpuIdx_);
                    ret = SyncTaskFinish(deviceTaskCtx, isStageFinish);
                    PerfMtEnd(PERF_EVT_SYNC_AICORE, aicpuIdx_);
                    if (isStageFinish) {
                        deviceTaskCtx->GetWrapManager().Deinit();
                        if (deviceTaskCtx->GetDeviceTaskCtrl()->Finish(!deviceTaskCtx->IsParallel())) {
                            PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, aicpuIdx_);
                            deviceTaskCtx->EntryStage(DevTaskExecStage::FINISH);
                        } else {
                            deviceTaskCtx->EntryStage(DevTaskExecStage::WAIT_ALL_SCH_FINISH);
                        }
                        PerfMtTrace(PERF_TRACE_DEV_TASK_SYNC_CORE_STOP, aicpuIdx_);
                    }
                    break;
                }
                case DevTaskExecStage::WAIT_ALL_SCH_FINISH: {
                    isStageFinish = deviceTaskCtx->GetDeviceTaskCtrl()->TryWaitAllSchFinish();
                    if (isStageFinish) {
                        deviceTaskCtx->EntryStage(DevTaskExecStage::FINISH);
                        PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, aicpuIdx_);
                    }
                    break;
                }
                case DevTaskExecStage::FINISH: {
                    return DEVICE_MACHINE_OK;
                }
                default:
                    DEV_ERROR(SchedErr::FSM_STATUS_ERROR, "Invalid stage %d.", ToUnderlying(deviceTaskCtx->CurStage()));
                    ret = ToUnderlying(SchedErr::FSM_STATUS_ERROR);
                    break;
            }

            if (ret != DEVICE_MACHINE_OK) {
                break;
            }

            if (deviceTaskCtx->IsRunFinish()) {
                break;
            }

            if (!isStageFinish && deviceTaskCtx->IsParallel()) {
                DEV_VERBOSE_DEBUG("Run device task leave stage : %d", ToUnderlying(deviceTaskCtx->CurStage()));
                return ret; // wait parallel scheduled next time
            }
        }

        DEV_DEBUG("aicpu %d proc finish devtask(%lu),aic: %lu, aiv: %lu, aicpu: %lu, stage:%d, ret: %d.",
            aicpuIdx_, deviceTaskCtx->TaskId(), procAicCoreFunctionCnt_,
            procAivCoreFunctionCnt_, procAicpuFunctionCnt_, ToUnderlying(deviceTaskCtx->CurStage()), ret);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            DEV_ERROR(
                SchedErr::CORE_TASK_PROCESS_FAILED,
                "#sche.dtask.leave: Aicpu[%d] proc finish: finishedFunctionCnt=%lu, "
                "coreFunctionCnt=%lu, taskId=%lu, but timeout !.",
                aicpuIdx_, deviceTaskCtx->GetDeviceTaskCtrl()->finishedFunctionCnt.load(),
                deviceTaskCtx->GetDeviceTask()->coreFunctionCnt, deviceTaskCtx->TaskId());
        }
        return ret;
    }

    inline int32_t PreProcessTask(SchDeviceTaskContext* deviceTaskCtx, bool& isExecFinish)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        DEV_INFO("receive new task %lu, firstTaskSend=%d.", deviceTaskCtx->TaskId(), deviceTaskCtx->isFirstTaskSend);

        // The initialization of aicpu tasks takes time, so to reduce headroom overhead, a batch of tasks is sent first.
        if (!deviceTaskCtx->isFirstTaskSend) {
            InitDevTask(deviceTaskCtx);
            ret = RunCoreTask(deviceTaskCtx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }

        if (IsNeedProcAicpuTask()) {
            const bool profSwitch = aicoreProf_.ProfIsEnable();
            ret = aicpuTaskManager_.Init(reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask()), profSwitch);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }

        isExecFinish = true;
        return DEVICE_MACHINE_OK;
    }

    inline int ProcessTaskLoop(SchDeviceTaskContext* deviceTaskCtx, bool& isFinish)
    {
        uint64_t start = GetCycles();
        while (!deviceTaskCtx->IsCoreTaskSendFinish()) {
            int32_t ret = RunCoreTask<true>(deviceTaskCtx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }

            if (deviceTaskCtx->IsParallel()) { // wait parallel scheduled next time
                deviceTaskCtx->SyncAllSchCoreTaskSent();
                isFinish = deviceTaskCtx->IsCoreTaskSendFinish();
                return ret;
            }

            DEV_IF_DEVICE
            {
                if (GetCycles() - start > TIMEOUT_CYCLES) {
                    return DEVICE_MACHINE_TIMEOUT_CORETASK;
                }
            }
        }
        deviceTaskCtx->SyncAllSchCoreTaskSent();
        isFinish = true;
 
        return DEVICE_MACHINE_OK;
    }

    inline void DumpLastWord(int coreIdx)
    {
        uint64_t status = aicoreHal_.GetAicoreStatus(coreIdx);
        if (pendingIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR(
                SchedErr::ABNOMAL_LAST_WORD, "coreid=%d status=%lu, pending taskid=%x, parallelIdx=%u, coreVer=%u",
                coreIdx, status, pendingIds_[coreIdx],
                npu::tile_fwk::ParallelIndex(pendingIds_[coreIdx]), aicoreHal_.ParallelDevTaskCtxVersion(coreIdx));
        }
        if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR(
                SchedErr::ABNOMAL_LAST_WORD, "coreid=%d, status=%lu, running taskid=%x, parallelIdx=%u, coreVer=%u",
                coreIdx, status, runningIds_[coreIdx],
                npu::tile_fwk::ParallelIndex(runningIds_[coreIdx]), aicoreHal_.ParallelDevTaskCtxVersion(coreIdx));
        }
    }

    void ResetRegAll()
    {
        ForEachManageAicore([this](int coreIdx) {
            if (aicoreHal_.ReadPathReg(coreIdx) == REG_SPR_FAST_PATH_OPEN) {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            } else {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
            }
        });
    }

    inline void PostRun(int ret)
    {
        if (ret) {
            DEV_ERROR(
                ret, "#sche.dtask.leave.post: execute error=%d, skip rest tasks, "
                "runreadyAiv=%u runreadyaic=%u, %u, %u",
                ret, context_->coreRunReadyCnt_[0], context_->coreRunReadyCnt_[1],
                context_->corePendReadyCnt_[0], context_->corePendReadyCnt_[1]);
            if constexpr (IsDeviceMode()) {
                ForEachManageAicore([&](int coreIdx) { DumpLastWord(coreIdx); });
            }

            // skip device task of current parallel ctx
            auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
            for (uint32_t i = parallelDevTaskCtx.front; i != parallelDevTaskCtx.rear; ++i) {
                auto &taskCtx = parallelDevTaskCtx.elements[i % SCH_DEVTASK_MAX_PARALLELISM];
                taskCtx.GetDeviceTaskCtrl()->Finish(true);
                DEV_ERROR(
                    SchedErr::ABNOMAL_LAST_WORD, "Force finish parallel ctx  parallelidx:%u.",
                    i % SCH_DEVTASK_MAX_PARALLELISM);
                taskCtx.Dump();
            }

            DumpAiCoreStatus();
    
            // skip device tash of ctrl quene
            DeviceTaskCtrl* taskCtrl = nullptr;
            while (!taskQueue_->IsEmpty()) {
                if ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->Finish(true);
                }
            } ;

            if constexpr (IsDeviceMode()) {
                NormalStop(); // some core maybe timeout
            }
        }

        if constexpr (IsDeviceMode()) {
            PerfMtTrace(PERF_TRACE_WAIT_CORE_EXIT, aicpuIdx_);
            ProfStop();
        }
        DEV_INFO(
            "Aicpu[%d] stop: ret=%d, procAicTaskCnt=%lu, procAivTaskCnt=%lu.", aicpuIdx_, ret, procAicCoreFunctionCnt_,
            procAivCoreFunctionCnt_);
    }

    inline int RunManager(int threadIdx, DevStartArgs* devStartArgs, DeviceArgs* deviceArgs, int schedIdx)
    {
        int ret = DEVICE_MACHINE_OK;
        DEV_DEBUG("schedule run threadIdx=%d", threadIdx);
        Init(threadIdx, devStartArgs, deviceArgs, schedIdx);
        PerfMtTrace(PERF_TRACE_INIT, threadIdx);
        DEV_DEBUG("Schedule run init succ");
        DeviceTaskCtrl* taskCtrl = nullptr;
        taskQueue_ = &(devStartArgs->deviceRuntimeDataDesc.taskQueueList[schedIdx_]);
        if constexpr (IsDeviceMode()) {
            ret = HandShake(devStartArgs);
            PerfMtTrace(PERF_TRACE_CORE_HAND_SHAKE, threadIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT, "#sche.handshake.error: hand shake timeout.");
                AbnormalStop();
                while ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->Finish(true);
                }
                return ret;
            }
            aicoreProf_.ProfStart();
        }
        DEV_DEBUG("Schedule run start succ");
        uint64_t lastDevTaskFinCycle = 0;
        PROF_STAGE_BEGIN_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.before\n");
        uint64_t start_cycles = GetCycles();
        while (ret == 0) {
            // try fetch devtask from ctrl queue
            FillParallelDevtaskCtx();

            // run devtask
            ret = ProcessParallelDevTasks();
            if (ret != DEVICE_MACHINE_OK)
                break;
                        
            lastDevTaskFinCycle = GetCycles();
            if (context_->DevTaskEmpty() && taskCtrlDequeFinish) {
                PerfMtTrace(PERF_TRACE_WAIT_ALL_DEV_TASK_FINISH, aicpuIdx_, lastDevTaskFinCycle);
                if (!isSendStop) {
                    DEV_INFO("Send all core stop.");
                    SendAllCoreStop();
                }
                break;
            }

            DEV_IF_DEVICE {
                if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
                    ret =  ToUnderlying(SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT);
                    DEV_ERROR(ret,
                        "Schedule prallel devtask timeout, dequeueFinish=%d.", taskCtrlDequeFinish);
                    break;
                }
            }
        }
        PROF_STAGE_END_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.after\n");

        PostRun(ret);
        return ret;
    }

    int32_t ProcessCompletedAicpuTask(uint64_t taskId)
    {
        int32_t ret = ResolveDepDyn(context_->curSchDevTaskCtx, taskId);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return BatchPushReadyQueue(context_->curSchDevTaskCtx);
    }

    inline void DumpAicorePerfTrace(std::ostringstream& oss)
    {
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
    inline void DumpTaskProf()
    {
        ForEachManageAicoreWithRet([this](int coreIdx) -> int { return aicoreHal_.DumpTaskProf(coreIdx); });
    }

    inline void ProfStop()
    {
        if (aicoreProf_.ProfIsEnable()) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
            DumpTaskProf();
#endif
        }

        aicoreProf_.ProfStop();
    }

    inline void DumpAiCoreStatus() const
    {
        DEV_IF_VERBOSE_DEBUG
        {
            ForEachManageAicore([this](int coreIdx) {
                if constexpr (IsDeviceMode()) {
                    aicoreHal_.DumpAicoreStatus(coreIdx);
                }
                DEV_ERROR(
                    SchedErr::ABNOMAL_LAST_WORD, "reg low task: runningid(%u) pendingid(%u)", runningIds_[coreIdx],
                    pendingIds_[coreIdx]);

                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                    "send task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    sendTask_[coreIdx].size());
                for (size_t i = 0; i < sendTask_[coreIdx].size(); i++) {
                    DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                        "send task: seqno %d, taskId %lx, refreshDevTask %x prallelModifyflag:%x, deviceTaskId %u",
                        (int)i, sendTask_[coreIdx][i].taskId & 0xFFFFFFFF,
                        DevTaskId(sendTask_[coreIdx][i].taskId >> REG_HIGH_DTASKID_SHIFT),
                        ParallelDevTaskModifyFlag(sendTask_[coreIdx][i].taskId >> REG_HIGH_DTASKID_SHIFT),
                        sendTask_[coreIdx][i].devTaskId);
                }

                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                    "recv finish task info ~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    recvFinTask_[coreIdx].size());
                for (size_t i = 0; i < recvFinTask_[coreIdx].size(); i++) {
                    DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                        "recv task: seqno %d, taskId %lx, deviceTaskId %u", (int)i,
                        recvFinTask_[coreIdx][i].taskId, recvFinTask_[coreIdx][i].devTaskId);
                }

                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                    "recv ack task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    recvAckTask_[coreIdx].size());
                for (size_t i = 0; i < recvAckTask_[coreIdx].size(); i++) {
                    DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                        "recv ack task: seqno %d, taskId %lx", static_cast<int>(i), recvAckTask_[coreIdx][i].taskId);
                }
            });
        }
    }

    bool NeedProcCoreTaskRsp(SchDeviceTaskContext* devTaskCtx, int coreIdx)
    {
        if ((pendingIds_[coreIdx] != AICORE_TASK_INIT &&
                ParallelIndex(pendingIds_[coreIdx]) == devTaskCtx->parallelIdx) ||
            (runningIds_[coreIdx] != AICORE_TASK_INIT &&
                ParallelIndex(runningIds_[coreIdx]) == devTaskCtx->parallelIdx)) {
            DEV_VERBOSE_DEBUG("core %d have tail un-finished task %x  %x.", coreIdx, pendingIds_[coreIdx], runningIds_[coreIdx]);
            return true;
        }

        return false;
    }

    inline bool CheckStopTaskCanBeSent(
        SchDeviceTaskContext* devTaskCtx, int coreIdx, bool isLastDevTask, uint32_t& resloveParallelIdx)
    {
        if (!NeedProcCoreTaskRsp(devTaskCtx, coreIdx)) {
            return true;
        }

        uint64_t finTaskVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
        bool bMatch = false;

        auto &pendingIdRef = pendingIds_[coreIdx];
        auto &runningIdRef = runningIds_[coreIdx];
        int type = static_cast<int>(AicoreType(coreIdx));
        if (likely(regLFinTaskState == TASK_FIN_STATE)) {
            if (pendingIdRef == regLFinTaskId) {
                bMatch = true;
                AddReadyCoreIdx(coreIdx, type);
                context_->corePendReadyCnt_[type]++;

                if (runningIdRef != AICORE_TASK_INIT) {
                    if (ParallelIndex(runningIdRef) == devTaskCtx->parallelIdx) {
                        DfxProcAfterFinishTask(devTaskCtx, coreIdx, runningIdRef);
                    } else {
                        // attention: if finished task belong to other devicetask, need to do the dependency resolution
                        (void)ResolveDepWithDfx(
                            static_cast<CoreType>(type), coreIdx, runningIdRef,
                            runningResolveIndexList_[coreIdx], resloveParallelIdx);
                    }
                }

                if (ParallelIndex(regLFinTaskId) == devTaskCtx->parallelIdx) {
                    DfxProcAfterFinishTask(devTaskCtx, coreIdx, regLFinTaskId);
                } else {
                    // attention: if finished task belong to other devicetask, need to do the dependency resolution
                    (void)ResolveDepWithDfx(
                        static_cast<CoreType>(type), coreIdx, regLFinTaskId,
                        pendingResolveIndexList_[coreIdx], resloveParallelIdx);
                }

                DEV_VERBOSE_DEBUG("rcv final pending task finish, pendtask: %u", regLFinTaskId);
            } else if (runningIdRef == regLFinTaskId && pendingIdRef == AICORE_TASK_INIT) {
                bMatch = true;
                AddReadyCoreIdx(coreIdx, type);
                DfxProcAfterFinishTask(devTaskCtx, coreIdx, regLFinTaskId);
                DEV_VERBOSE_DEBUG("rcv final running task finish, runningtask: %u", regLFinTaskId);
            }
        } else if (isLastDevTask && regLFinTaskState == TASK_ACK_STATE && pendingIdRef == regLFinTaskId) {
           // The core stop task can be sent once the last task ACK is received, without waiting for finish rsp.
           // The execution of the final task and the sending of the final core stop task can be parallelized.
            bMatch = true;
            AddReadyCoreIdx(coreIdx, type);
            context_->corePendReadyCnt_[type]++;
            DfxProcAfterFinishTask(devTaskCtx, coreIdx, regLFinTaskId);
            if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DfxProcAfterFinishTask(devTaskCtx, coreIdx, runningIds_[coreIdx]);
            }
            DEV_VERBOSE_DEBUG("rcv final pending task ack, pendtask: %u", regLFinTaskId);
        }

        if (bMatch) {
            pendingIdRef = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexList_[coreIdx] = 0;
            context_->wrapCoreAvail_[coreIdx] = true;
            DEV_VERBOSE_DEBUG("core %d tail task finished.", coreIdx);
            return true;
        }

        return false;
    }

    void SendStopToCore(SchDeviceTaskContext* devTaskCtx, int coreIdx, bool isLastDevTask)
    {
        if (isLastDevTask) {
            DEV_IF_DEVICE {
                NormalStopSingleCore(coreIdx);
            } else {
                if (enableEslModel_) {
                    NormalStopSingleCore(coreIdx);
                } 
            }
            DEV_VERBOSE_DEBUG("Last devtask ,core %d send AICORE_TASK_STOP.", coreIdx);
        }

        devTaskCtx->coreTaskFinished[coreIdx] = 1;
        devTaskCtx->coreFinishedNum++;
        DEV_VERBOSE_DEBUG("Core %d finished, finishnum = %u. ", coreIdx, devTaskCtx->coreFinishedNum);
    }

    inline void SendStopToIdleCore(SchDeviceTaskContext* devTaskCtx, bool isLastDevTask) {
        uint32_t aicIdleNum = context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIC)];
        uint32_t aivIdleNum = context_->coreRunReadyCnt_[static_cast<int>(CoreType::AIV)];

        for (uint32_t i = 0; i < aicIdleNum; i++) {
            SendStopToCore(devTaskCtx, context_->runReadyCoreIdx_[static_cast<int>(CoreType::AIC)][i], isLastDevTask);
        }

        for (uint32_t i = 0; i < aivIdleNum; i++) {
            SendStopToCore(devTaskCtx, context_->runReadyCoreIdx_[static_cast<int>(CoreType::AIV)][i], isLastDevTask);
        }
    }

    inline void AicoreDevTaskFinishProc(
        SchDeviceTaskContext* devTaskCtx, int coreIdx, bool isLastDevTask, uint32_t& resloveParallelIdx) {
        if (CheckStopTaskCanBeSent(devTaskCtx, coreIdx, isLastDevTask, resloveParallelIdx)) {
            SendStopToCore(devTaskCtx, coreIdx, isLastDevTask);
        }
        return;
    }

    inline void DumpDfxWhenCoreNotStop(SchDeviceTaskContext* devTaskCtx)
    {
        for (int i = aicStart_; i < aicEnd_; i++) {
            if (!devTaskCtx->coreTaskFinished[i]) {
                DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT,
                    "#sche.task.end.sync.timeout: left aic core %d not stop, pending:%x, rungning:%x, regfinishid: %lx,"
                    "core last status:%lu backup status:%lu", i, pendingIds_[i], runningIds_[i],
                    aicoreHal_.GetFinishedTask(i), aicoreHal_.GetAicoreStatus(i), aicoreHal_.GetAicoreStatusBackup(i));
            }
        }

        for (int i = aivStart_; i < aivEnd_; i++) {
            if (!devTaskCtx->coreTaskFinished[i]) {
                DEV_ERROR(
                    SchedErr::TASK_WAIT_TIMEOUT,
                    "#sche.task.end.sync.timeout: left aiv core %d not stop, pending:%x, rungning:%x, regfinishid: %lx,"
                    "core last status:%lu backup status:%lu", i, pendingIds_[i], runningIds_[i],
                    aicoreHal_.GetFinishedTask(i), aicoreHal_.GetAicoreStatus(i), aicoreHal_.GetAicoreStatusBackup(i));
            }
        }
    }

    void SendAllCoreStop()
    {
        ForEachManageAicore([this](int coreIdx) {
            DEV_IF_DEVICE {
                NormalStopSingleCore(coreIdx);
            }
            DEV_VERBOSE_DEBUG("core %d send AICORE_TASK_STOP.", coreIdx);
        });
    }

    inline int SyncTaskFinish(SchDeviceTaskContext* devTaskCtx, bool& isFinish, bool forceStop = false)
    {
        int aicNum = aicEnd_ - aicStart_;
        int aivNum = aivEnd_ - aivStart_;
        uint32_t mngCoreNum = static_cast<uint32_t>(aicNum + aivNum);
        bool aicAllStop = false;
        bool aivAllStop = false;
        bool isLastDevTask = false;
        auto curDevTask = devTaskCtx->GetDeviceTask();
        if (!forceStop) {
            isLastDevTask = reinterpret_cast<DynDeviceTask*>(curDevTask)->IsLastTask();
        } else {
            isLastDevTask = true;
        }

        if (isLastDevTask && (context_->DeviceTaskCtxNum() == 1)) {
            isSendStop = true;
        }

        uint64_t start_cycles = GetCycles();
        uint32_t resloveParallelIdx = 0;
        while (devTaskCtx->coreFinishedNum < mngCoreNum) {
            bool curIterAicAllStop = true;
            bool curIterAivAllStop = true;
            for (int i = aicStart_; (!aicAllStop) && i < aicEnd_; i++) {
                if (devTaskCtx->coreTaskFinished[i]) {
                    continue;
                }

                AicoreDevTaskFinishProc(devTaskCtx, i, isSendStop, resloveParallelIdx);
                if (!devTaskCtx->coreTaskFinished[i]) {
                    curIterAicAllStop = false;
                }
            }
            aicAllStop = curIterAicAllStop;

            for (int i = aivStart_; (!aivAllStop) && i < aivEnd_; i++) {
                if (devTaskCtx->coreTaskFinished[i]) {
                    continue;
                }
                AicoreDevTaskFinishProc(devTaskCtx, i, isSendStop, resloveParallelIdx);
                if (!devTaskCtx->coreTaskFinished[i]) {
                    curIterAivAllStop = false;
                }
            }
            aivAllStop = curIterAivAllStop;

            // In a parallel scenario, wait parallel scheduled next time
            if (devTaskCtx->IsParallel()) {
                (void)BatchPushReadyQueForParallel(resloveParallelIdx);
                break;
            }
            
            DEV_IF_DEVICE
            {
                if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
                    DumpDfxWhenCoreNotStop(devTaskCtx);
                    DEV_ERROR(
                        SchedErr::TASK_WAIT_TIMEOUT,
                        "#sche.task.end.sync.timeout: SyncAicoreDevTaskFinish timeout notstopNum=%u.",
                        mngCoreNum - devTaskCtx->coreFinishedNum);
                    return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
                }
            }
        }

        if (devTaskCtx->coreFinishedNum == mngCoreNum) {
            isFinish = true;
            return SyncAicpuTaskFinish();
        }
        return DEVICE_MACHINE_OK;
    }

    inline int32_t SyncAicpuTaskFinish()
    {
        if (IsNeedProcAicpuTask()) {
            auto ret = aicpuTaskManager_.SyncAicpuTaskFinish(this);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    // 检查是否进入了尾批，当剩余任务数小于等于管理核心数时，认为进入了尾批
    inline bool CheckIsTailBatch(SchDeviceTaskContext* devTaskCtx, CoreType type, uint64_t& remaining)
    {
        if (devTaskCtx->IsParallel() || aicpuNum_ <= 1) {
            return false;
        }

        remaining =
            devTaskCtx->GetDeviceTask()->coreFunctionCnt -
            devTaskCtx->GetDeviceTaskCtrl()->finishedFunctionCnt.load(std::memory_order_relaxed);
        uint32_t totalCoreNum =
            (type == CoreType::AIC) ? static_cast<uint32_t>(aicNum_) : static_cast<uint32_t>(aivNum_);
        return (remaining > 0 && remaining <= static_cast<uint64_t>(totalCoreNum));
    }

    // 当进入尾批时，也选择保守策略，只分配完全空闲的核心
    inline uint32_t GetReadyCoreNum(CoreType type, bool isTail = false)
    {
        if ((enableFairSch_ || isTail) && IsExistOtherAicpuIdle(type)) {
            return context_->coreRunReadyCnt_[static_cast<int>(type)];
        }
        return context_->corePendReadyCnt_[static_cast<int>(type)];
    }

    inline uint64_t TryBatchSendTask(SchDeviceTaskContext* devTaskCtx, CoreType type, ReadyCoreFunctionQueue* readyQue,
                int coreIdxStart, int coreIdxEnd)
    {
        if (__atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED) == __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED)) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Task: 0", aicpuIdx_);
            return 0;
        }
        uint64_t remaining = 0;
        bool isTail = CheckIsTailBatch(devTaskCtx, type, remaining);
        uint32_t ready = GetReadyCoreNum(type, isTail);
        if (ready == 0) {
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
            memcpy_s(
                readyId, taskCount * sizeof(uint64_t), reinterpret_cast<uint8_t*>(&readyQue->elem[tail - taskCount]),
                taskCount * sizeof(uint32_t));
            __atomic_fetch_sub(&readyQue->tail, taskCount, std::memory_order_release);
        } else {
            __atomic_fetch_add(&readyQue->head, taskCount, std::memory_order_release);
        }
        ReadyQueueUnLock((readyQue));
        DEV_VERBOSE_DEBUG("AiCpud:%d, pop all new task count: %u", aicpuIdx_, taskCount);
        BatchSendTask(
            devTaskCtx, type, isRealLifo ? &readyId[taskCount - 1] : &readyQue->elem[head],
            taskCount, coreIdxStart, coreIdxEnd, isRealLifo);
        DEV_VERBOSE_DEBUG("core ready cnt: %u", context_->corePendReadyCnt_[static_cast<int>(type)]);
        firstLock[static_cast<int>(type)] = false;
        PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        return taskCount;
    }

    inline uint32_t BatchSendTask(
        SchDeviceTaskContext* devTaskCtx, CoreType type, uint32_t *newTask, uint32_t taskCount,
        int coreIdxStart, int coreIdxEnd, bool isLifo)
    {
        uint32_t sendCnt = 0;
        uint32_t coreRunReadyCnt = context_->coreRunReadyCnt_[static_cast<int>(type)];
        DEV_VERBOSE_DEBUG(
            "Begin Batch send %s task: corerunreadycnt:%u, pendreadyCnt:%u, taskCount:%u.",
            type == CoreType::AIC ? "AIC" : "AIV", coreRunReadyCnt, context_->corePendReadyCnt_[static_cast<int>(type)],
            taskCount);
        while (sendCnt < static_cast<uint64_t>(coreRunReadyCnt) && sendCnt < taskCount) {
            uint32_t coreIdx =
                context_
                    ->runReadyCoreIdx_[static_cast<int>(type)][context_->coreRunReadyCnt_[static_cast<int>(type)] - 1];
            RemoveReadyCoreIdx(coreIdx, static_cast<int>(type));
            SendTaskToAiCore(devTaskCtx, type, coreIdx, isLifo ? *newTask-- : *newTask++);
            sendCnt++;
        }
        context_->corePendReadyCnt_[static_cast<int>(type)] -= sendCnt;

        uint32_t idx = context_->lastPendReadyCoreIdx_[static_cast<int>(type)];
        uint32_t coreNum = coreIdxEnd - coreIdxStart;
        uint32_t lastProcCore = idx;
        DEV_VERBOSE_DEBUG(
            "  ## send task left pend ready cnt %u , last core index:%u.",
            context_->corePendReadyCnt_[static_cast<int>(type)], idx);
        while (context_->corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount) {
            if (pendingIds_[idx] == AICORE_TASK_INIT && context_->wrapCoreAvail_[idx]) {
                DEV_VERBOSE_DEBUG("  ## send task use pendready core %u.", idx);
                SendTaskToAiCore(devTaskCtx, type, idx, isLifo ? *newTask-- : *newTask++);
                sendCnt++;
                context_->corePendReadyCnt_[static_cast<int>(type)]--;
                lastProcCore = idx;
            }
            idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
        }

        if (lastProcCore != context_->lastPendReadyCoreIdx_[static_cast<int>(type)]) {
            context_->lastPendReadyCoreIdx_[static_cast<int>(type)] =
                coreIdxStart + (lastProcCore - coreIdxStart + 1) % coreNum;
        }
        DEV_VERBOSE_DEBUG(
            "  ## finish send task left runreadycnt:%u pendreadycnt %u, last coreindex:%u.",
            context_->coreRunReadyCnt_[static_cast<int>(type)], context_->corePendReadyCnt_[static_cast<int>(type)],
            idx);
        return sendCnt;
    }

    inline int32_t DispatchAiCoreTask(
        SchDeviceTaskContext* devTaskCtx, CoreType type, ReadyCoreFunctionQueue* readyQue,
        int coreIdxStart, int coreIdxEnd)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& wrapManager = devTaskCtx->GetWrapManager();
        if (devTaskCtx->waitTaskCnt[static_cast<int>(type)] > 0) {
            ret = ResolveDepForAllAiCore(devTaskCtx, type, coreIdxStart, coreIdxEnd);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            wrapManager.DispatchMixCoreTask();
        }
        if (wrapManager.GetIsMixarch()) {
            ReadyCoreFunctionQueue* dieReadyQue =
                (type == CoreType::AIC) ? wrapManager.GetDieReadyAicQue() : wrapManager.GetDieReadyAivQue();
            if (dieReadyQue != readyQue) {
                TryBatchSendTask(devTaskCtx, type, dieReadyQue, coreIdxStart, coreIdxEnd);
            }
        }
        TryBatchSendTask(devTaskCtx, type, readyQue, coreIdxStart, coreIdxEnd);
        if (enableFairSch_) {
            if (context_->coreRunReadyCnt_[static_cast<int>(type)] > 0) {
                AicpuIsIdle(type);
            } else {
                AicpuIsBusy(type);
            }
        }
        return ret;
    }

#define RAW_TENSOR_ADDR_MASK ((1UL << 63) - 1)
    inline DynFuncData* GetDynFuncData(DeviceTask* deviceTask, uint64_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTask);
        DynFuncHeader* head = (DynFuncHeader*)dyntask->GetDynFuncDataList();
        auto funcDataList = (DynFuncData*)(head + 1);
        auto funcData = &funcDataList[FuncID(taskId)];
        return funcData;
    }

    inline uint64_t GetTensorAddr(DynFuncData* dynFuncData, uint64_t rawTensorIndex)
    {
        auto desc = &dynFuncData->rawTensorDesc[rawTensorIndex];
        if (desc->location == npu::tile_fwk::RAW_TENSOR_LOCATION_LOCAL) {
            return dynFuncData->workspaceAddr + desc->offsetOrIndex;
        } else {
            return dynFuncData->rawTensorAddr[desc->offsetOrIndex] & RAW_TENSOR_ADDR_MASK;
        }
    }

    inline uint64_t GetCoa(DynFuncData* dynFuncData, const SymInt* attrs, int idx)
    {
        return attrs[idx].IsExpression() ? dynFuncData->exprTbl[attrs[idx].Value()] : attrs[idx].Value();
    }

    inline schema::shape SchemaGetShape(
        DynFuncData* dynFuncData, const SymInt* attrs, const DevAscendOperationOperandInfo& info)
    {
        auto attrOffset = info.staticOffsetAttrBeginIndex;
        std::vector<schema::Int64Type> shapeList;
        for (int d = 0; d < info.GetDim(); d++) {
            auto shapeIdx = attrOffset + d + info.GetDim() * 3;
            auto actualShape = GetCoa(dynFuncData, attrs, shapeIdx);
            shapeList.push_back(actualShape);
        }
        return schema::shape(schema::shapeList(shapeList));
    }

    inline schema::offset SchemaGetOffset(
        DynFuncData* dynFuncData, const SymInt* attrs, const DevAscendOperationOperandInfo& info)
    {
        auto attrOffset = info.staticOffsetAttrBeginIndex;
        std::vector<schema::Int64Type> offsetList;
        for (int d = 0; d < info.GetDim(); d++) {
            auto offsetIdx = attrOffset + d;
            auto actualOffset = GetCoa(dynFuncData, attrs, offsetIdx);
            offsetList.push_back(actualOffset);
        }
        return schema::offset(schema::offsetList(offsetList));
    }

    inline std::map<uint64_t, uint64_t> CalTensorAddrAndSize(SchDeviceTaskContext* devTaskCtx, uint64_t taskId)
    {
        std::map<uint64_t, uint64_t> tensorAddr2SizeMap;
        uint32_t opIdx = TaskID(taskId);
        auto duppedData = GetDuppedData(devTaskCtx->GetDeviceTask(), taskId);
        auto dynFuncData = GetDynFuncData(devTaskCtx->GetDeviceTask(), taskId);
        auto iOperandSize = duppedData->GetSource()->GetOperationIOperandSize(opIdx);
        for (size_t i = 0; i < iOperandSize; i++) {
            auto iOperand = duppedData->GetSource()->GetOperationIOperand(opIdx, i);
            auto base = GetTensorAddr(dynFuncData, iOperand->rawIndex);
            auto size = duppedData->GetRawTensorDataSize(iOperand->rawIndex);
            tensorAddr2SizeMap[base] = size;
        }

        auto oOperandSize = duppedData->GetSource()->GetOperationOOperandSize(opIdx);
        for (size_t i = 0; i < oOperandSize; i++) {
            auto oOperand = duppedData->GetSource()->GetOperationOOperand(opIdx, i);
            auto base = GetTensorAddr(dynFuncData, oOperand->rawIndex);
            auto size = duppedData->GetRawTensorDataSize(oOperand->rawIndex);
            tensorAddr2SizeMap[base] = size;
        }
        return tensorAddr2SizeMap;
    }

    inline void DumpSchemaOperationInfo(SchDeviceTaskContext* devTaskCtx, int coreIdx, uint64_t taskId)
    {
        DeviceTaskCtrl* curTaskCtrl = devTaskCtx->GetDeviceTaskCtrl();
        uint64_t deviceTaskId = curTaskCtrl->taskId;
        uint32_t funcId = FuncID(taskId);
        int rootIndex = GetRootIndex(devTaskCtx, taskId);
        int leafIndex = GetLeafIndex(devTaskCtx, taskId);
        uint32_t opIdx = TaskID(taskId);
        auto duppedData = GetDuppedData(devTaskCtx->GetDeviceTask(), taskId);
        auto dynFuncData = GetDynFuncData(devTaskCtx->GetDeviceTask(), taskId);
        auto attrBase = &duppedData->GetSource()->GetOperationAttr(opIdx, 0);

        DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex), LActStart(coreIdx)));
        DEV_TRACE_DEBUG_SPLIT(LEvent(
            LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex), duppedData->GetSource()->SchemaGetCoa(opIdx)));

        auto iOperandSize = duppedData->GetSource()->GetOperationIOperandSize(opIdx);
        DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex), LActIncastCount(iOperandSize)));
        for (size_t i = 0; i < iOperandSize; i++) {
            auto iOperand = duppedData->GetSource()->GetOperationIOperand(opIdx, i);
            auto base = GetTensorAddr(dynFuncData, iOperand->rawIndex);
            auto size = duppedData->GetRawTensorDataSize(iOperand->rawIndex);
            auto opInfo = duppedData->GetSource()->GetOperationIOperandInfo(opIdx, i);
            DEV_TRACE_DEBUG(LEvent(
                LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),
                LActIncast(
                    SchemaGetShape(dynFuncData, attrBase, opInfo), SchemaGetOffset(dynFuncData, attrBase, opInfo),
                    Range(base, base + size))));
        }

        auto oOperandSize = duppedData->GetSource()->GetOperationOOperandSize(opIdx);
        DEV_TRACE_DEBUG(
            LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex), LActOutcastCount(oOperandSize)));
        for (size_t i = 0; i < oOperandSize; i++) {
            auto oOperand = duppedData->GetSource()->GetOperationOOperand(opIdx, i);
            auto base = GetTensorAddr(dynFuncData, oOperand->rawIndex);
            auto size = duppedData->GetRawTensorDataSize(oOperand->rawIndex);
            auto opInfo = duppedData->GetSource()->GetOperationOOperandInfo(opIdx, i);
            DEV_TRACE_DEBUG(LEvent(
                LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),
                LActOutcast(
                    SchemaGetShape(dynFuncData, attrBase, opInfo), SchemaGetOffset(dynFuncData, attrBase, opInfo),
                    Range(base, base + size))));
        }
    }

    uint64_t UpdateParallelCtxAndCalcModifyFlag(int coreIdx, uint32_t coreParallelVersion)
    {
        uint64_t modifyFlag = 0;
        auto& parallelCtx = context_->schParallelDevTaskCtx;
        volatile ParallelDevTask* coreParallelDevTask = aicoreHal_.GetParallelDevTask(coreIdx);

        for (uint32_t i = parallelCtx.front; i < parallelCtx.rear; ++i) {
            uint32_t idx = i % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;

            SchDeviceTaskContext* devTaskCtx = parallelCtx.Element(i);
            if (devTaskCtx->IsFree()) {
                continue;
            }

            auto* dyntask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
            int64_t funcData = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
            if (devTaskCtx->bindParallelCtxVersion > coreParallelVersion) {
                modifyFlag |= (1ULL << idx);
                aicoreHal_.SetParallelDevTask(coreParallelDevTask, idx, funcData);
            }
        }
        return modifyFlag;
    }

    /*
     |--------16bit-------------|----16bit----|----1bit----|-----1bit------|------1bit-----|-----3bit--------|---10bit---|---16bit--|
     |-parallel ctx modifyflag--|--devtaskid--|----rspflag-|--pingpongflag-|---dcci flag---|--prallel index--|--func id--|--opindex-|
    */
    uint64_t EncodeTaskId(SchDeviceTaskContext* devTaskCtx, int coreIdx, uint64_t newTask) {
        uint32_t shift = TASKID_TASK_BITS + TASKID_FUNC_BITS;

        // encode parallel index
        uint64_t encodeTaskId = newTask | (devTaskCtx->parallelIdx << shift);
        uint32_t coreParallelVersion = aicoreHal_.ParallelDevTaskCtxVersion(coreIdx);

        // devicetask context not compatible， need notify aicore dcci
        if (devTaskCtx->bindParallelCtxVersion > coreParallelVersion) {
            DEV_INFO("Notify aicore(%d) refresh prallel devtask, devtaskVer:%u > coreVer:%u, newestVer: %u.",
                coreIdx, devTaskCtx->bindParallelCtxVersion, aicoreHal_.ParallelDevTaskCtxVersion(coreIdx),
                context_->PrallelVersion());

            // encode dcci flag
            shift += TASKID_PARALLEL_INDEX_BITS;
            encodeTaskId |= 1 << shift;

            // encode pingpong flag make sure encoded taskid is different with last time
            shift += TASKID_DEVTASK_DCCI_BITS;
            encodeTaskId |= pingPongFlag_[coreIdx] << shift;
            pingPongFlag_[coreIdx] ^= 1;

            // encode device taskid, aicore will validate it from taskid
            shift += 2;
            encodeTaskId |= devTaskCtx->TaskId() << shift;

            DEV_IF_DEVICE
            {
                // encode parallel ctx modify flag
                shift += REG_VAL_DEVTASK_ID_BITS;
                encodeTaskId |= UpdateParallelCtxAndCalcModifyFlag(coreIdx, coreParallelVersion) << shift;
            } else {
                if (enableEslModel_) {
                    // encode parallel ctx modify flag
                    shift += REG_VAL_DEVTASK_ID_BITS;
                    encodeTaskId |= UpdateParallelCtxAndCalcModifyFlag(coreIdx, coreParallelVersion) << shift;
                } else {
                    // costmodel don't support parallel devtask
                    InitCostModelFuncDataForOneCore(devTaskCtx, coreIdx);
                }
            }

            aicoreHal_.SetParallelDevTaskCtxVersion(coreIdx, context_->PrallelVersion());
        }
        return encodeTaskId;
    }

    inline void SendTaskToAiCore(SchDeviceTaskContext* devTaskCtx, CoreType type, int coreIdx, uint64_t newTask)
    {
        DEV_IF_VERBOSE_DEBUG { DumpSchemaOperationInfo(devTaskCtx, coreIdx, newTask); }

#if ENABLE_TENSOR_DUMP
        // dump input tensor
        if (unlikely(isEnableDump)) {
            aicoreDump_.DoDump(devTaskCtx->GetDeviceTask(), "input", newTask, GetPhyIdByBlockId(coreIdx));
        }
#endif
        uint64_t encodeTaskId = EncodeTaskId(devTaskCtx, coreIdx, newTask);
        aicoreHal_.SetReadyQueue(coreIdx, (encodeTaskId + 1));
        pendingIds_[coreIdx] = static_cast<uint32_t>(encodeTaskId & 0xFFFFFFFF);
        pendingResolveIndexList_[coreIdx] = 0;
        devTaskCtx->sendCnt[static_cast<int>(type)]++;

        if (!devTaskCtx->isFirstTaskSend) {
            PerfMtTrace(PERF_TRACE_DEV_TASK_SEND_FIRST_CALLOP_TASK, aicpuIdx_);
            devTaskCtx->isFirstTaskSend = true;
        }

        DEV_IF_VERBOSE_DEBUG
        {
            sendTask_[coreIdx].push_back(TaskInfo(coreIdx, encodeTaskId, devTaskCtx->TaskId()));
            if (devTaskCtx->GetWrapManager().IsBindedWrapId(newTask) && context_->wrapCoreAvail_[coreIdx]) {
                DEV_WARN("newTask[%lu][%lx] is mix task, but core[%d] is available!", newTask, newTask, coreIdx);
            }
            if (!devTaskCtx->GetWrapManager().IsBindedWrapId(newTask) && !context_->wrapCoreAvail_[coreIdx]) {
                DEV_WARN(
                    "newTask[%lu][%lx] is not mix task, but core[%d] is not available!", newTask, newTask, coreIdx);
            }
        }
        DEV_VERBOSE_DEBUG("Send task %lx, origin taskid %lx, at core %d ,type:%d.",
            encodeTaskId, newTask, coreIdx, static_cast<int>(type));
    }

    inline void SetAiCpuStat(int coreIdx, uint64_t taskId)
    {
        struct AiCpuTaskStat aiCpuTaskStat;
        aiCpuTaskStat.taskId = taskId;
        aiCpuTaskStat.coreId = aicoreHal_.GetPhyIdByBlockId(coreIdx);
        aicoreProf_.AsmCntvc(aiCpuTaskStat.taskGetStart);
        aicoreProf_.SetAiCpuTaskStat(taskId, aiCpuTaskStat);
    };

    inline void AddReadyCoreIdx(int coreIdx, int type)
    {
        context_->coreIdxPosition_[coreIdx] = context_->coreRunReadyCnt_[type];
        context_->runReadyCoreIdx_[type][context_->coreRunReadyCnt_[type]++] = coreIdx;
    }

    inline void RemoveReadyCoreIdx(int coreIdx, int type)
    {
        context_->coreRunReadyCnt_[type]--;
        context_->coreIdxPosition_[coreIdx] = INVALID_COREIDX_POSITION;
    }

    inline int32_t PushReadyQue(ReadyCoreFunctionQueue* readyQue, void* idList, uint32_t idCnt) const
    {
        ReadyQueueLock(readyQue);
        memcpy_s(&readyQue->elem[readyQue->tail], idCnt * sizeof(uint32_t), (uint8_t*)idList, idCnt * sizeof(uint32_t));
        __atomic_fetch_add(&readyQue->tail, idCnt, std::memory_order_release);
        DEV_IF_NONDEVICE
        {
            if (readyQue->tail > readyQue->capacity) {
                DEV_ERROR(
                    SchedErr::READY_QUEUE_OVERFLOW, "#sche.resolve.enqueue: readyQue tail=%u > readyQue capacity=%u",
                    readyQue->tail, readyQue->capacity);
                return DEVICE_MACHINE_ERROR;
            }
            DEV_ASSERT(SchedErr::READY_QUEUE_OVERFLOW, readyQue->tail <= readyQue->capacity);
        }
        ReadyQueueUnLock(readyQue);
        return DEVICE_MACHINE_OK;
    }

    inline int32_t ResolveDepForAllAiCore(SchDeviceTaskContext* devTaskCtx, CoreType type, int coreIdxStart, int coreIdxEnd)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        uint32_t resloveParallelIdx = 0;
        PerfMtBegin(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        ResolveTaskContext resolveCtx[MAX_MANAGER_AIV_NUM];
        uint32_t finishCnt = 0;
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            if (NeedProcCoreTaskRsp(devTaskCtx, i)) {
                // release finish core
                ret = ReleaseCoreByRegVal(type, i, resolveCtx, finishCnt, resloveParallelIdx);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        }

        if (!enableL2CacheSch_) {
            // send task to available core
            ReadyCoreFunctionQueue* readyQue =
                (type == CoreType::AIC) ? devTaskCtx->readyAicCoreFunctionQue : devTaskCtx->readyAivCoreFunctionQue;
            if (devTaskCtx->GetWrapManager().GetIsMixarch()) {
                ReadyCoreFunctionQueue* dieReadyQue =
                    (type == CoreType::AIC) ?
                    devTaskCtx->GetWrapManager().GetDieReadyAicQue() : devTaskCtx->GetWrapManager().GetDieReadyAivQue();
                if (dieReadyQue != readyQue) {
                    TryBatchSendTask(devTaskCtx, type, dieReadyQue, coreIdxStart, coreIdxEnd);
                }
            }
            TryBatchSendTask(devTaskCtx, type, readyQue, coreIdxStart, coreIdxEnd);
        }

        // resolve resolveCtx
        for (uint32_t i = 0; i < finishCnt; i++) {
            ret = ResolveDepWithDfx(
                type, resolveCtx[i].finishCoreIdx, resolveCtx[i].finishIds,
                resolveCtx[i].resolveIndexBase, resloveParallelIdx);
            if (enableFairSch_ && (resloveParallelIdx & (1U << devTaskCtx->parallelIdx))) {
                if (devTaskCtx->readyAicCoreFunctionQue->tail - devTaskCtx->readyAicCoreFunctionQue->head == 0 ||
                    devTaskCtx->readyAivCoreFunctionQue->tail - devTaskCtx->readyAivCoreFunctionQue->head == 0) {
                    ret = BatchPushReadyQueue(devTaskCtx);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }

        ret = BatchPushReadyQueForParallel(resloveParallelIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        PerfMtEnd(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        return ret;
    }

    int32_t BatchPushReadyQueForParallel(uint32_t resloveParallelIdx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        uint32_t mask = resloveParallelIdx;
        while (mask) {
            int idx = __builtin_ffs(mask) - 1;
            ret = BatchPushReadyQueue(context_->ParallelDeviceTaskCtx(idx));
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            mask &= (mask - 1);
        }
        return ret;
    }

    inline int32_t BatchPushReadyQueue(SchDeviceTaskContext* devTaskCtx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        uint32_t aicIndex = static_cast<uint32_t>(CoreType::AIC);
        uint32_t aivIndex = static_cast<uint32_t>(CoreType::AIV);
        if (devTaskCtx->readyCount[aicIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIC), devTaskCtx->readyCount[aicIndex]);
            if (needSendCnt > 0) {
                devTaskCtx->readyCount[aicIndex] -= BatchSendTask(devTaskCtx,
                    CoreType::AIC, &devTaskCtx->readyIds[aicIndex][devTaskCtx->readyCount[aicIndex] - 1], needSendCnt,
                    aicStart_, aicEnd_, true);
            }
            DEV_VERBOSE_DEBUG(
                "resolved new task, aic ready count: %u coretype:%u.", devTaskCtx->readyCount[aicIndex], aicIndex);
            if (devTaskCtx->readyCount[aicIndex] > 0) {
                ReadyCoreFunctionQueue* targetReadyQue = devTaskCtx->readyAicCoreFunctionQue;
                if (devTaskCtx->GetWrapManager().GetIsMixarch() &&
                    EnableDieScheduling(devTaskCtx, CoreType::AIC, devTaskCtx->readyIds[aicIndex][0])) {
                    targetReadyQue = devTaskCtx->GetWrapManager().GetDieReadyAicQue();
                }
                ret = PushReadyQue(targetReadyQue, devTaskCtx->readyIds[aicIndex], devTaskCtx->readyCount[aicIndex]);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            devTaskCtx->readyCount[aicIndex] = 0;
        }

        if (devTaskCtx->readyCount[aivIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIV), devTaskCtx->readyCount[aivIndex]);
            if (needSendCnt > 0) {
                devTaskCtx->readyCount[aivIndex] -= BatchSendTask(devTaskCtx,
                    CoreType::AIV, &devTaskCtx->readyIds[aivIndex][devTaskCtx->readyCount[aivIndex] - 1], needSendCnt,
                    aivStart_, aivEnd_, true);
            }
            DEV_VERBOSE_DEBUG(
                "resolved new task, aiv ready count: %u coretype: %u.", devTaskCtx->readyCount[aivIndex], aivIndex);
            if (devTaskCtx->readyCount[aivIndex] > 0) {
                ReadyCoreFunctionQueue* targetReadyQue = devTaskCtx->readyAivCoreFunctionQue;
                if (devTaskCtx->GetWrapManager().GetIsMixarch() &&
                    EnableDieScheduling(devTaskCtx, CoreType::AIV, devTaskCtx->readyIds[aivIndex][0])) {
                    targetReadyQue = devTaskCtx->GetWrapManager().GetDieReadyAivQue();
                }
                ret = PushReadyQue(targetReadyQue, devTaskCtx->readyIds[aivIndex], devTaskCtx->readyCount[aivIndex]);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            devTaskCtx->readyCount[aivIndex] = 0;
        }
        return ret;
    }

    inline int32_t ResolveDepForAicpuTask(uint64_t& taskCount)
    {
        int32_t ret = aicpuTaskManager_.TaskProcess(taskCount);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return aicpuTaskManager_.TaskPoll(this);
    }

    inline int32_t ResolveWhenSyncMode(
        CoreType type, uint32_t finTaskId, uint32_t finTaskState, int coreIdx, uint32_t& resloveParallelIdx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_VERBOSE_DEBUG(
                "core index: %d, PendingTask Finished."
                " pending: %x.",
                coreIdx, pendingIds_[coreIdx]);
            ret = ResolveDepWithDfx(type, coreIdx, finTaskId, 0, resloveParallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            if (context_->wrapCoreAvail_[coreIdx]) {
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
                AddReadyCoreIdx(coreIdx, static_cast<int>(type));
            }

            SchDeviceTaskContext* deviceTaskCtx = context_->ParallelDeviceTaskCtx(ParallelIndex(finTaskId));
            deviceTaskCtx->GetWrapManager().UpdateFinishIdForMixCore(finTaskId);
        }
        return ret;
    }

    static uint64_t RuntimeCopyOutResolveCounterDecode(uint64_t aicpuCallCode) { return aicpuCallCode & 0xffff; }

    inline void RecordResolveTask(
        ResolveTaskContext* ctx, uint32_t& finishCnt, int coreIdx, uint32_t taskId, int indexBase)
    {
        ctx[finishCnt].finishIds = taskId;
        ctx[finishCnt].resolveIndexBase = indexBase;
        ctx[finishCnt].finishCoreIdx = coreIdx;
        finishCnt++;
    }

    inline int32_t ReleaseCoreByRegVal(
        CoreType type, int coreIdx, ResolveTaskContext* ctx, uint32_t& finishCnt, uint32_t& resloveParallelIdx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreIdx);
        [[maybe_unused]] uint32_t aicpuCallCode = finTaskRegVal >> 32;
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);
        DEV_VERBOSE_DEBUG(
            "reslove task core index: %d, finishtaskid:%x, finishstate: %u.", coreIdx, finTaskId, finTaskState);

#if SCHEDULE_USE_PENDING_AND_RUNING_SWITCH
        auto& pendingIdRef = pendingIds_[coreIdx];
        auto& pendingResolveIndexBaseRef = pendingResolveIndexList_[coreIdx];
        auto& runningIdRef = runningIds_[coreIdx];
        auto& runningResolveIndexBaseRef = runningResolveIndexList_[coreIdx];
        if (likely(finTaskId == pendingIdRef && finTaskState == TASK_FIN_STATE)) {
            // pending task is finished, resolve both running and pending task.
            DEV_VERBOSE_DEBUG(
                "Pending Finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            if (context_->wrapCoreAvail_[coreIdx]) { // wrapcore doesnt support pending & running yet
                AddReadyCoreIdx(coreIdx, static_cast<int>(type));
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (runningIdValue != AICORE_TASK_INIT) {
                RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValue, runningResolveIndexBaseValue);
            }
            RecordResolveTask(ctx, finishCnt, coreIdx, pendingIdValue, pendingResolveIndexBaseValue);
            SchDeviceTaskContext* deviceTaskCtx = context_->ParallelDeviceTaskCtx(ParallelIndex(finTaskId));
            deviceTaskCtx->GetWrapManager().UpdateFinishIdForMixCore(finTaskId);
        } else if (unlikely(finTaskId == pendingIdRef && aicpuCallCode != 0)) {
            // pending task is copyout, reolve both running and pending task.
            DEV_VERBOSE_DEBUG(
                "Pending Copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValueCopyout = runningIdRef;
            int runningResolveIndexBaseValueCopyout = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = pendingIdRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            if (context_->wrapCoreAvail_[coreIdx]) {
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (runningIdValueCopyout != AICORE_TASK_INIT) {
                RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValueCopyout, runningResolveIndexBaseValueCopyout);
            }
            ret = ResolveCopyOutDepDyn(copyOutResolveCounter, pendingIdValue, pendingResolveIndexBaseValue, resloveParallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else if (finTaskId == pendingIdRef && finTaskState == TASK_ACK_STATE) {
            // pending task is acknowledged, resolve running task. And move pending to running
            DEV_VERBOSE_DEBUG(
                "Pending Acknowledged: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            DEV_IF_VERBOSE_DEBUG { recvAckTask_[coreIdx].push_back(TaskInfo(coreIdx, finTaskId, 0xFFFFFFFF)); }
            uint32_t runningIdValueAck = runningIdRef;
            int runningResolveIndexBaseValueAck = runningResolveIndexBaseRef;
            if (context_->wrapCoreAvail_[coreIdx]) {
                runningIdRef = finTaskId;
                runningResolveIndexBaseRef = pendingResolveIndexBaseRef;
                pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
                pendingResolveIndexBaseRef = 0;
                context_->corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (runningIdValueAck != AICORE_TASK_INIT) {
                RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValueAck, runningResolveIndexBaseValueAck);
            }
        } else if (finTaskId == runningIdRef && finTaskState == TASK_FIN_STATE) {
            // running task is finished, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG(
                "Running finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            if (pendingIdRef == AICORE_TASK_INIT) {
                AddReadyCoreIdx(coreIdx, static_cast<int>(type));
            }
            RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValue, runningResolveIndexBaseValue);
        } else if (unlikely(finTaskId == runningIdRef && aicpuCallCode != 0)) {
            // running task is copyout, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG(
                "Running copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            ret = ResolveCopyOutDepDyn(copyOutResolveCounter, runningIdValue, runningResolveIndexBaseValue, resloveParallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else {
            DEV_VERBOSE_DEBUG(
                "Warning, maybe inconsistent state. coreidx: %d,finTask: %lx,pending: %x,running: %x.", coreIdx,
                finTaskRegVal, pendingIdRef, runningIdRef);
        }
#else
        ret = ResolveWhenSyncMode(type, finTaskId, finTaskState, coreIdx, resloveParallelIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
#endif
        return ret;
    }

    inline void PushAicpuTaskQueue(SchDeviceTaskContext* devTaskCtx, uint64_t taskId)
    {
        PushReadyQue(devTaskCtx->readyAicpuFunctionQue, &taskId, 1);
    }

    inline bool TrySendTaskDirectly(SchDeviceTaskContext* devTaskCtx, int coreType, uint32_t taskId)
    {
        if (context_->coreRunReadyCnt_[coreType] > 0) {
            context_->corePendReadyCnt_[coreType]--;
            uint32_t coreIdx = context_->runReadyCoreIdx_[coreType][context_->coreRunReadyCnt_[coreType] - 1];
            RemoveReadyCoreIdx(coreIdx, coreType);
            DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
            SendTaskToAiCore(devTaskCtx, static_cast<CoreType>(coreType), coreIdx, taskId);
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
        while (pendingIds_[idx] != AICORE_TASK_INIT || !context_->wrapCoreAvail_[idx]) {
            idx = startIdx + (idx - startIdx + 1) % (coreNum);
        }
        context_->lastPendReadyCoreIdx_[coreType] = static_cast<uint32_t>(startIdx + (idx - startIdx + 1) % (coreNum));
        context_->corePendReadyCnt_[coreType]--;
        DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
        SendTaskToAiCore(devTaskCtx, static_cast<CoreType>(coreType), idx, taskId);
        return true;
    }

    inline int32_t PushReadyTask(SchDeviceTaskContext* devTaskCtx, int coreType, uint64_t taskId)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        if (enableL2CacheSch_ && (!devTaskCtx->IsParallel()) && TrySendTaskDirectly(devTaskCtx, coreType, taskId)) {
            return DEVICE_MACHINE_OK;
        }

        if (unlikely(devTaskCtx->readyCount[coreType] == READY_ID_FIX_CACHE_NUM)) {
            ReadyCoreFunctionQueue* readyQue =
                coreType == static_cast<int>(CoreType::AIC) ?
                devTaskCtx->readyAicCoreFunctionQue : devTaskCtx->readyAivCoreFunctionQue;
            auto& wrapManager = devTaskCtx->GetWrapManager();
            if (wrapManager.GetIsMixarch() &&
                EnableDieScheduling(devTaskCtx, static_cast<CoreType>(coreType), devTaskCtx->readyIds[coreType][0])) {
                readyQue =
                    coreType == static_cast<int>(CoreType::AIC) ?
                    wrapManager.GetDieReadyAicQue() : wrapManager.GetDieReadyAivQue();
            }
            ret = PushReadyQue(readyQue, devTaskCtx->readyIds[coreType], devTaskCtx->readyCount[coreType]);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            devTaskCtx->readyCount[coreType] = 0;
        }
        devTaskCtx->readyIds[coreType][devTaskCtx->readyCount[coreType]++] = taskId;
        return ret;
    }

    inline uint64_t GetCostModelTaskTime(uint64_t coreIdx, uint64_t taskId, uint64_t currentTime)
    {
        DeviceTask* curDevTask = context_->curSchDevTaskCtx->GetDeviceTask();
        auto funcId = FuncID(taskId);
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask);
        auto costModelData = reinterpret_cast<CostModel::ModelData*>(curDevTask->costModelData);
        if (costModelData == nullptr)
            return 0;
        auto source = dyntask->GetDynFuncDataCacheList()[funcId].devFunc;
        auto opIndex = TaskID(taskId);
        auto leafFunctionIdx = source->GetOperationAttrCalleeIndex(opIndex);
        auto timeCost = costModelData->functionTime[leafFunctionIdx];
        auto header = dyntask->GetDynFuncDataList();
        auto dyndata = reinterpret_cast<DynFuncData*>(&header->At(0));
        auto opAttrs = &dyndata->opAttrs[dyndata->opAtrrOffsets[TaskID(taskId)]];
        auto psgId = opAttrs[0];
        // devTaskId - funcId - leaf function Id - psgId
        std::string name = std::to_string(context_->curSchDevTaskCtx->TaskId()) + '-' + std::to_string(funcId) +
                           '-' + std::to_string(opIndex) + '-' + std::to_string(psgId);
        PerfMtEvent(PERF_EVT_TASK, coreIdx + PERF_AICORE_THREAD_START, currentTime, currentTime + timeCost, name);
        return timeCost;
    }

    inline int32_t ResolveDynStitched(
        SchDeviceTaskContext* deviceTaskCtx, DynDeviceTask* dyntask, int origfunc, int origop, int coreIdx = 0)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& duppedData = dyntask->GetDynFuncDataCacheList()[origfunc].duppedData;
        auto& stitchList = duppedData->GetOperationStitch(origop);
        auto cceBinary = dyntask->cceBinary;
        auto& wrapManager = deviceTaskCtx->GetWrapManager();

        for (auto* node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            for (uint32_t i = 0; i < listSize; i++) {
                uint32_t id = node->At(i);
                auto funcId = FuncID(id);
                auto opIndex = TaskID(id);
                auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
                bool needProcess =
                    predCounts[opIndex] == 1 || __atomic_sub_fetch(&predCounts[opIndex], 1, __ATOMIC_RELAXED) == 0;
                if (!needProcess) {
                    continue;
                }

                auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
                auto coreType = cceBinary[callList[opIndex]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(deviceTaskCtx, id, 0, coreIdx);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    deviceTaskCtx->resolveHubCnt++;
                } else if (coreType == static_cast<int>(MachineType::AICPU)) {
                    PushAicpuTaskQueue(deviceTaskCtx, id);
                } else if (wrapManager.IsBindedWrapId(id)) {
                    wrapManager.ResolveDepForMixCore(id);
                } else {
                    ret = PushReadyTask(deviceTaskCtx, static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }

    inline int GetRootIndex(SchDeviceTaskContext* deviceTaskCtx, uint32_t taskId) const
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(taskId);
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        return func->GetRootIndex();
    }

    inline int GetLeafIndex(SchDeviceTaskContext* deviceTaskCtx, uint32_t taskId) const
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return callList[opIndex];
    }

    inline DevAscendFunctionDuppedData* GetDuppedData(DeviceTask* deviceTask, uint32_t taskId) const
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTask);
        auto funcId = FuncID(taskId);
        return dyntask->dynFuncDataCacheList[funcId].duppedData;
    }

    inline int32_t ResolveDepDyn(
        SchDeviceTaskContext* deviceTaskCtx, uint64_t finishId, size_t resolveIndexBase = 0, int coreIdx = 0)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(finishId);
        auto opIndex = TaskID(finishId);
        auto& wrapManager = deviceTaskCtx->GetWrapManager();

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts =  dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int* succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        for (size_t i = succIndexList[resolveIndexBase]; i < succSize; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 || __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(deviceTaskCtx, id, resolveIndexBase, coreIdx);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    deviceTaskCtx->resolveHubCnt++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))) {
                    PushAicpuTaskQueue(deviceTaskCtx, id);
                } else if (wrapManager.IsBindedWrapId(id)) {
                    wrapManager.ResolveDepForMixCore(id);
                } else {
                    ret = PushReadyTask(deviceTaskCtx, static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }

        ret = ResolveDynStitched(deviceTaskCtx, dyntask, funcId, opIndex, coreIdx);
        return ret;
    }

    inline int32_t ResolveCopyOutDepDyn(
        uint32_t currResolveIndex, uint64_t taskId, uint32_t resolveIndexBase, uint32_t& resloveParallelIdx)
    {
        uint32_t taskParallelIndex = ParallelIndex(taskId);
        resloveParallelIdx |= (1U << taskParallelIndex);
        SchDeviceTaskContext* deviceTaskCtx = context_->ParallelDeviceTaskCtx(taskParallelIndex);

        int32_t ret = DEVICE_MACHINE_OK;
        auto dyntask = reinterpret_cast<DynDeviceTask *>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto& wrapManager = deviceTaskCtx->GetWrapManager();

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int* succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        const int* succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        // here we don't use resolveIndexBase + 1, because at the beginning, resolveIndexBase is 0. And we resolve from
        // 0.
        for (int i = succIndexList[resolveIndexBase]; i < succIndexList[currResolveIndex + 1]; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 || __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(deviceTaskCtx, id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    deviceTaskCtx->resolveHubCnt++;
                } else if (wrapManager.IsBindedWrapId(id)) {
                    wrapManager.ResolveDepForMixCore(id);
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))) {
                    PushAicpuTaskQueue(deviceTaskCtx, id);
                } else {
                    ret = PushReadyTask(deviceTaskCtx, static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }

    inline int32_t ResolveDepWithDfx(
        CoreType type, int coreIdx, uint64_t finishId, size_t resolveIndexBase, uint32_t& resloveParallelIdx)
    {
        uint32_t taskParallelIndex = ParallelIndex(finishId);
        resloveParallelIdx |= (1U << taskParallelIndex);
        SchDeviceTaskContext* deviceTaskCtx = context_->ParallelDeviceTaskCtx(taskParallelIndex);
        int32_t ret = DEVICE_MACHINE_OK;
        ret = ResolveDepDyn(deviceTaskCtx, finishId, resolveIndexBase, coreIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        DEV_VERBOSE_DEBUG(
            "[Call]: Core %d Dispatch Task: %lu, %u, %u, %u", coreIdx, deviceTaskCtx->TaskId(),
            FuncID(finishId), TaskID(finishId), DevTaskDcciFlag(finishId));
        DfxProcAfterFinishTask(deviceTaskCtx, coreIdx, finishId);
        deviceTaskCtx->waitTaskCnt[static_cast<int>(type)]--;
        return ret;
    }

    inline bool IsExistOtherAicpuIdle(CoreType type)
    {
        int idx = (schedIdx_ + 1) % aicpuNum_;
        while (idx != schedIdx_) {
            if (threadStatus.isAicpuIdle[static_cast<int>(type)][idx].load(std::memory_order_relaxed) == true) {
                return true;
            }
            idx = (idx + 1) % aicpuNum_;
        }
        return false;
    }

    inline bool EnableDieScheduling(SchDeviceTaskContext* deviceTaskCtx, CoreType type, uint32_t taskId)
    {
        auto duppedData = GetDuppedData(deviceTaskCtx->GetDeviceTask(), taskId);
        auto loopDieId = duppedData->loopDieId_;
        if (loopDieId < 0 || (loopDieId != static_cast<int8_t>(deviceTaskCtx->GetWrapManager().GetDieId()))) { // prevent parallel_loop incorrectly, task depends on other die
            return false;
        }
        if (!enableFairSch_) {
            return true;
        }
        int schedStart = 0;
        int schedEnd = 0;
        deviceTaskCtx->GetWrapManager().GetDieSchedIdRange(schedStart, schedEnd, aicpuNum_);
        const auto& idleMap = threadStatus.isAicpuIdle[static_cast<int>(type)];
        for (int idx = schedStart; idx < schedEnd; idx++) {
            if (idleMap[idx].load(std::memory_order_relaxed) == true) {
                return true;
            }
        }
        return false;
    }

    inline void AicpuIsBusy(CoreType type)
    {
        if (threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_] != false) {
            threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_].store(false, std::memory_order_relaxed);
        }
    }

    inline void AicpuIsIdle(CoreType type)
    {
        if (threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_] != true) {
            threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_].store(true, std::memory_order_relaxed);
        }
    }

    inline void Init(int threadIdx, DevStartArgs* startArgs, DeviceArgs* deviceArgs, int schedIdx)
    {
        (void)startArgs;
        aicNum_ = static_cast<int32_t>(deviceArgs->nrAic);
        aivNum_ = static_cast<int32_t>(deviceArgs->nrAiv);
        aicpuNum_ = deviceArgs->scheCpuNum;
        aicpuIdx_ = threadIdx;
        schedIdx_ = schedIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        enableEslModel_ = deviceArgs->enableEslModel;
        aicoreHal_.Init(deviceArgs, &aicoreProf_);
        validGetPgMask_ = deviceArgs->validGetPgMask;
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
        runningResolveIndexList_.fill(0);
        pendingResolveIndexList_.fill(0);
        taskDfxStatPos_.fill(REG_LOW_TASK_PING);
        pingPongFlag_.fill(0);
        isSendStop = false;
        taskCtrlDequeFinish = false;
        if (IsNeedProcAicpuTask()) {
            aicpuTaskManager_.InitDeviceArgs(deviceArgs);
        }
        context_->Init(deviceArgs, schedIdx);

#if ENABLE_TENSOR_DUMP
        isEnableDump = startArgs->devProg->devArgs.hostPid != 0;
        if (unlikely(isEnableDump)) {
            aicoreDump_.Init(startArgs, schedIdx);
        }
#endif
        (void)startArgs;

        if (deviceArgs->machineConfig != static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH)) {
            if (aicpuNum_ > 1) {
                enableFairSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                                 static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH);
            }
            enableL2CacheSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                                static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH);
        }
        UpdateAiCoreBlockIndexSection(deviceArgs->archInfo);
        if constexpr (IsDeviceMode()) {
            aicoreHal_.MapRegistersForAllCores(aicNum_);
            aicoreProf_.ProfInit(deviceArgs);
        } else {
            aicoreHal_.SetTaskTimeCost([this](uint64_t coreIdx, uint64_t taskId, uint64_t time) {
                return GetCostModelTaskTime(coreIdx, taskId, time);});
        }
        firstLock[static_cast<int>(CoreType::AIC)] = true;
        firstLock[static_cast<int>(CoreType::AIV)] = true;
        aicoreDevTaskInited = false;
        DEV_INFO("Init aicore manager: aicNum=%d, aivNum=%d, schAicpuNum=%d, aicpuIdx=%d, "
            "aicValidNum=%d, aicoreHal.regAddrs=%p, sharedBuffer=%p, machineConfig=%u.",
            aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, aicoreHal_.GetRegAddrs(),
            (void*)aicoreHal_.GetSharedBuffer(), static_cast<uint8_t>(deviceArgs->machineConfig));
    }


    inline SchDeviceTaskContext* HandShakeTryPreFetchDevTask(bool& needSendAic, bool& needSendAiv)
    {
        FillParallelDevtaskCtx();
        if (!context_->DevTaskEmpty()) {
            auto deviceTaskCtx = context_->FrontDevTaskCtx();
            InitDevTask(deviceTaskCtx);
            needSendAic = (deviceTaskCtx->readyAicCoreFunctionQue->tail != deviceTaskCtx->readyAicCoreFunctionQue->head);
            needSendAiv = (deviceTaskCtx->readyAivCoreFunctionQue->tail != deviceTaskCtx->readyAivCoreFunctionQue->head);
            DEV_DEBUG("hand shake prefetch dev task success: needSendAic=%d, needSendAiv=%d", needSendAic, needSendAiv);
            return deviceTaskCtx;
        }
        return nullptr;
    }

    inline void HandShakePostProc(SchDeviceTaskContext* schDeviceTaskCtx, bool needSendAic, bool needSendAiv)
    {
        // send task by left ready core
        if (needSendAic) {
            __sync_synchronize();
            TryBatchSendTask(schDeviceTaskCtx, CoreType::AIC, schDeviceTaskCtx->readyAicCoreFunctionQue, aicStart_, aicEnd_);
        }
        if (needSendAiv) {
            __sync_synchronize();
            TryBatchSendTask(schDeviceTaskCtx, CoreType::AIV, schDeviceTaskCtx->readyAivCoreFunctionQue, aivStart_, aivEnd_);
        }

        if (schDeviceTaskCtx) {
            schDeviceTaskCtx->CountCoreTaskSent();
            DEV_DEBUG("hand shake presend task cnt : aic=%lu, aiv=%lu",
                schDeviceTaskCtx->waitTaskCnt[static_cast<int>(CoreType::AIC)],
                schDeviceTaskCtx->waitTaskCnt[static_cast<int>(CoreType::AIV)]);
        }
    }

    inline void DumpAicoreStatusWhenTimeout(bool* handFlag)
    {
        for (int i = aicStart_; i < aicEnd_; i++) {
            if (handFlag[i]) {
                DEV_INFO("Aic core[%d] hand shake success, phyid=%d.", i, aicoreHal_.GetPhyIdByBlockId(i));
            } else {
                DEV_ERROR(
                    SchedErr::HANDSHAKE_TIMEOUT,
                    "#sche.handshake.timeout: Aic core[%d] hand shake timeout, status=%lu.", i,
                    aicoreHal_.GetAicoreStatus(i));
            }
        }

        for (int i = aivStart_; i < aivEnd_; i++) {
            if (handFlag[i]) {
                DEV_INFO("Aiv core[%d] hand shake success, phyid=%d.", i, aicoreHal_.GetPhyIdByBlockId(i));
            } else {
                DEV_ERROR(
                    SchedErr::HANDSHAKE_TIMEOUT,
                    "#sche.handshake.timeout: Aiv core[%d] hand shake timeout, status=%lu.", i,
                    aicoreHal_.GetAicoreStatus(i));
            }
        }
    }

    inline int HandShakeByGmWithPreSendTask(DevStartArgs* devStartArgs)
    {
        int handShakeNum = 0;
        int mngAicoreNum = aicEnd_ - aicStart_ + aivEnd_ - aivStart_;
        bool handFlag[MAX_AICORE_NUM] = {false};
        uint64_t start_cycles = GetCycles();
        bool needSendAic = false;
        bool needSendAiv = false;
        bool aicAllSuccess = false;
        bool aivAllSuccess = false;
        bool needSetSync = true;
        int aicSucessCnt = 0;
        int aivSucessCnt = 0;
        int aicTreshold = 4;
        int aivThreshold = 4;
        SchDeviceTaskContext* deviceCtx = nullptr;
        while (handShakeNum < mngAicoreNum) {
            if (deviceCtx == nullptr) {
                deviceCtx = HandShakeTryPreFetchDevTask(needSendAic, needSendAiv);
            }

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
                    AddReadyCoreIdx(i, static_cast<int>(CoreType::AIC));
                } else {
                    curIterAllAicSuccess = false;
                }
            }
            aicAllSuccess = curIterAllAicSuccess;

            if (unlikely(needSetSync && (handShakeNum > 0))) {
                devStartArgs->syncFlag = 1;
                needSetSync = false;
            }

            if (needSendAic && aicSucessCnt >= aicTreshold) {
                __sync_synchronize(); // sync  REG_SPR_FAST_PATH_ENABLE
                TryBatchSendTask(deviceCtx, CoreType::AIC, deviceCtx->readyAicCoreFunctionQue, aicStart_, aicEnd_);
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
                    AddReadyCoreIdx(i, static_cast<int>(CoreType::AIV));
                } else {
                    curIterAllAivSuccess = false;
                }
            }
            aivAllSuccess = curIterAllAivSuccess;

            if (needSendAiv && aivSucessCnt >= aivThreshold) {
                __sync_synchronize();
                TryBatchSendTask(deviceCtx, CoreType::AIV, deviceCtx->readyAivCoreFunctionQue, aivStart_, aivEnd_);
                aivSucessCnt = 0;
            }

            if (unlikely(GetCycles() - start_cycles > HAND_SHAKE_TIMEOUT)) {
                DumpAicoreStatusWhenTimeout(handFlag);
                DEV_ERROR(
                    SchedErr::HANDSHAKE_TIMEOUT,
                    "#sche.handshake.timeout: HandShakeByGmWithPreSendTask timeout notHandshakeNum=%d.",
                    mngAicoreNum - handShakeNum);
                return DEVICE_MACHINE_ERROR;
            }
        }

        HandShakePostProc(deviceCtx, needSendAic, needSendAiv);
        return DEVICE_MACHINE_OK;
    }

    inline int HandShake(DevStartArgs* devStartArgs)
    {
        DEV_INFO("aicpu[%d] handshake start.", aicpuIdx_);
        int rc = HandShakeByGmWithPreSendTask(devStartArgs);
        if (rc != DEVICE_MACHINE_OK) {
            DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT, "#sche.handshake.presend: Aicpu[%d] handshake failed.", aicpuIdx_);
            return rc;
        }

        DEV_INFO("Aicpu[%d] handshake success.", aicpuIdx_);
        return 0;
    }

    /* assign aic and aiv core index section for this aicpu */
    inline void UpdateAiCoreBlockIndexSection(ArchInfo archInfo)
    {
        auto f = [](int total, int idx, int part, int count, int& start, int& end) {
            int perCpu = (total / part) * count;
            int remain = total % part;
            start = idx * perCpu + ((idx < remain) ? idx * count : remain * count);
            end = start + perCpu + ((idx < remain) ? count : 0);
        };

        f(aicValidNum_, schedIdx_, aicpuNum_, 1, aicStart_, aicEnd_);
        if (archInfo == ArchInfo::DAV_3510) {
            f(aicValidNum_, schedIdx_, aicpuNum_, AIV_NUM_PER_AI_CORE, aivStart_, aivEnd_);
        } else {
            f(AIV_NUM_PER_AI_CORE * aicValidNum_, schedIdx_, aicpuNum_, 1, aivStart_, aivEnd_);
        }

        aivStart_ += aicValidNum_;
        aivEnd_ += aicValidNum_;

        DEV_IF_NONDEVICE
        {
            context_->corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = aicEnd_ - aicStart_;
            context_->corePendReadyCnt_[static_cast<int>(CoreType::AIV)] = aivEnd_ - aivStart_;
            ForEachManageAicoreReverse([this](int coreIdx) {
                int coreType = static_cast<int>(AicoreType(coreIdx));
                AddReadyCoreIdx(coreIdx, coreType);
            });
        }

        context_->lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIV)] = static_cast<uint32_t>(aivStart_);
        context_->lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIC)] = static_cast<uint32_t>(aicStart_);
        aicoreHal_.SetMngCoreBlockId(aicStart_, aicEnd_, aivStart_, aivEnd_);
        DEV_DEBUG("assign core aic coreindex section: start=%d, end=%d.", aicStart_, aicEnd_);
        DEV_DEBUG("assign core aiv coreindex section: start=%d, end=%d.", aivStart_, aivEnd_);
    }

    inline int GetPhyIdByBlockId(int coreIdx) { return aicoreHal_.GetPhyIdByBlockId(coreIdx); }

    inline void ForEachManageAicore(std::function<void(int coreIdx)> func) const
    {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            func(i);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            func(i);
        }
    }

    inline void ForEachManageAicoreReverse(std::function<void(int coreIdx)> func) const
    {
        for (int i = aicEnd_ - 1; i >= aicStart_; --i) {
            func(i);
        }
        for (int i = aivEnd_ - 1; i >= aivStart_; --i) {
            func(i);
        }
    }

    inline int ForEachManageAicoreWithRet(std::function<int(int coreIdx)> func) const
    {
        int ret = DEVICE_MACHINE_OK;
        for (int i = aicStart_; i < aicEnd_; ++i) {
            ret = func(i);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR(
                    SchedErr::CORE_TASK_PROCESS_FAILED, "#sche.check.aic.process: proc aicore aic[%d] failed.", i);
                return ret;
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            ret = func(i);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR(
                    SchedErr::CORE_TASK_PROCESS_FAILED, "#sche.check.aiv.process: proc aicore aiv[%d] failed.", i);
                return ret;
            }
        }
        return ret;
    }

    inline void AbnormalStop()
    {
        ResetRegAll();
        CheckAndResetReg();
        DEV_INFO("aicore manager[%d] abnormal stopped.", aicpuIdx_);
    }

    inline void NormalStop()
    {
        DEV_INFO("aicore manager[%d] try normal stop.", aicpuIdx_);
        ForEachManageAicore([this](auto coreIdx) { aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1); });
        /* write to MAINBASE reg must be done before close 0x18 */
        __sync_synchronize();
        ForEachManageAicore([this](auto coreIdx) { aicoreHal_.ResetShakeBuf(coreIdx); });
        DEV_INFO("aicore manager[%d] normal stopped.", aicpuIdx_);
    }

    inline void NormalStopSingleCore(int coreIdx)
    {
        aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
        __sync_synchronize();
        aicoreHal_.ResetShakeBuf(coreIdx);
    }

    inline int GetAllAiCoreNum() { return aicNum_ + aivNum_; }
    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }
     inline void SetNextDfxPos(int coreIdx) 
     { 
         taskDfxStatPos_[coreIdx] = 
             taskDfxStatPos_[coreIdx] == REG_LOW_TASK_PING ? REG_LOW_TASK_PONG : REG_LOW_TASK_PING; 
     } 
     inline int GetDfxPos(int coreIdx) { return taskDfxStatPos_[coreIdx]; }

    // DFX
    inline void DfxProcAfterFinishTask(SchDeviceTaskContext* deviceTaskCtx, int coreIdx, uint64_t taskId)
    {
        DEV_TRACE_DEBUG(LEvent(
            LUid(deviceTaskCtx->TaskId(), FuncID(taskId), GetRootIndex(deviceTaskCtx, taskId),
            TaskID(taskId), GetLeafIndex(deviceTaskCtx, taskId)), LActFinish(coreIdx)));
        if constexpr (!IsDeviceMode())
            return;

#if ENABLE_AICORE_PRINT
        DumpAicoreLog(coreIdx);
#endif

        volatile TaskStat* stat = aicoreHal_.GetTaskStat(coreIdx, 0);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1 
         aicoreProf_.ProfGet(coreIdx, stat->subGraphId, stat->taskId, const_cast<TaskStat*>(stat)); 
#endif

#if ENABLE_TENSOR_DUMP
        // dump output tensor
        if (unlikely(isEnableDump)) {
            aicoreDump_.DoDump(deviceTaskCtx->GetDeviceTask(), "output", taskId, GetPhyIdByBlockId(coreIdx), stat->execStart, stat->execEnd);
        }
#endif

        DEV_IF_VERBOSE_DEBUG { recvFinTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId, deviceTaskCtx->TaskId())); }

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1 
         SetNextDfxPos(coreIdx); // pingpong 存储 
#endif
        (void)stat;
    }

    inline bool IsNeedProcAicpuTask() { return aicpuIdx_ == 2; }

private:
    void ReuseUpdateDeviceCtx(SchDeviceTaskContext* devTaskCtx, DeviceTaskCtrl* newDevTask) {
        devTaskCtx->BindTaskCtrl(newDevTask);

        // update version mark parallel context modified
        context_->UpdateParallelVersion();
        devTaskCtx->BindParallelCtxVersion(context_->PrallelVersion());
    }

    void FillParallelDevtaskCtx() {
        auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
        while (!parallelDevTaskCtx.Full()) {
            DeviceTaskCtrl* taskCtrl = nullptr;
            if (!taskQueue_->TempDequeue(taskCtrl)) {
                break; // no device task
            }

            if (taskCtrl == nullptr) {
                taskQueue_->PopFront();
                taskCtrlDequeFinish = true;
                break;
            }

            if (parallelDevTaskCtx.Empty()) {
                PerfMtTrace(PERF_TRACE_DEV_TASK_RCV, aicpuIdx_);
                context_->EnqueueParallelCtx(taskCtrl); // if empty,  equeue directly
                taskQueue_->PopFront();
                continue;
            }

            if (!context_->CurSupportParallel()) {
                DEV_DEBUG("Cur ctx cannot support prallel, fill stop.");
                break; // non-parallel context just support one devtask schedule
            }

            if (!taskCtrl->SupportParallel()) {
                DEV_DEBUG("Device task(%lu) cannot support prallel, fill stop.", taskCtrl->taskId);
                break; // non-parallel devtask cannot scheduled with prallel dev task
            }

            if (!context_->CanParallelWith(taskCtrl)) {
                DEV_DEBUG("Cur ctx cannot prallel with device task, %lu, forid=%u", taskCtrl->taskId, taskCtrl->ParallelForId());
                break; // different parallel-forid devtask cannot scheduled together
            }

            taskCtrlDequeFinish = reinterpret_cast<DynDeviceTask *>(taskCtrl->devTask)->IsLastTask();
            context_->EnqueueParallelCtx(taskCtrl);
            taskQueue_->PopFront();
        }

        if (!aicoreDevTaskInited && !parallelDevTaskCtx.Empty()) {
            DEV_INFO("Begin init aicore parallel devtask data.");
            InitAicoreParallelDevTask(&parallelDevTaskCtx);
            aicoreDevTaskInited = true; // just need init one time
        }
        return;
    }

    inline int32_t ProcessParallelDevTasks() {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
        for (uint32_t i = parallelDevTaskCtx.front; i < parallelDevTaskCtx.rear; ++i) {
            SchDeviceTaskContext* devTaskCtx = parallelDevTaskCtx.Element(i);
            if (devTaskCtx->IsFree()) {
                DEV_VERBOSE_DEBUG("Device task ctx(%u) wait recycle.", devTaskCtx->parallelIdx);
                continue; // maybe have some non-consecutiv free ctx wait recycle
            }
            context_->curSchDevTaskCtx = devTaskCtx;
            PerfMtBegin(PERF_EVT_RUN_TASK, aicpuIdx_);
            ret = RunTask(devTaskCtx);
            PerfMtEnd(PERF_EVT_RUN_TASK, aicpuIdx_);
            if (ret != DEVICE_MACHINE_OK)
                break;

            if (devTaskCtx->IsRunFinish()) {
                if (unlikely(devTaskCtx->IsParallel())) {
                    // continue bind the next parallel devtaskctrl wich have the same forid & iterid to this sch context
                    DeviceTaskCtrl* curTaskCtrl = devTaskCtx->GetDeviceTaskCtrl();
                    if (curTaskCtrl->ExistNextSameIterTask()) {
                        DeviceTaskCtrl* nextTaskCtrl = curTaskCtrl->NextSameIterTaskCtrl();
                        if (nextTaskCtrl != nullptr) {
                            // reuse this task ctrl and task context for next same iterid device task
                            ReuseUpdateDeviceCtx(devTaskCtx, nextTaskCtrl);
                            curTaskCtrl->Free(); // parallel device taskctrl need free manually
                            DEV_DEBUG("Sch dev ctx bind next same parallel iter device task(%lu), forid %u, iterid %u",
                                nextTaskCtrl->taskId, nextTaskCtrl->ParallelForId(), nextTaskCtrl->ParallelIterId());
                        } else {
                            DEV_VERBOSE_DEBUG("Wait ctrl build same parallel iter device task, forid %u, iterid %u.",
                                curTaskCtrl->ParallelForId(), curTaskCtrl->ParallelIterId());
                        }
                    } else {
                        // parallel devicetaskctrl and device context need set free manually, wait recycle
                        devTaskCtx->Free();
                    }
                } else {
                    parallelDevTaskCtx.PopFront(); // non-parallel tasks can only exist one at a time.
                }
            }
        }

        parallelDevTaskCtx.RecycleFreeContexts();
        return ret;
    }

private:
    AicoreHAL aicoreHal_;
    bool aicoreDevTaskInited{false};
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

    std::array<uint32_t, MAX_AICORE_NUM> runningIds_;
    std::array<uint32_t, MAX_AICORE_NUM> pendingIds_;
    std::array<int, MAX_AICORE_NUM> runningResolveIndexList_;
    std::array<int, MAX_AICORE_NUM> pendingResolveIndexList_;

    SchduleContext* context_{nullptr};
    SchThreadStatus &threadStatus;

    bool taskCtrlDequeFinish{false};
    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> *taskQueue_{nullptr};
    AicpuTaskManager &aicpuTaskManager_;

    AiCoreProf aicoreProf_;
    AicoreDump aicoreDump_;
    bool isEnableDump{false};
    int64_t dotStatus_{0};
    bool isSendStop{false};
    std::array<uint8_t, MAX_AICORE_NUM> pingPongFlag_;

    std::array<int, MAX_AICORE_NUM> taskDfxStatPos_;
    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];

    AicoreLogger* logger_{nullptr};
    friend class AiCoreProf;

    bool enableEslModel_;
};
} // namespace npu::tile_fwk::dynamic
