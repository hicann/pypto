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
 * \file device_machine.h
 * \brief
 */

#pragma once

#include <signal.h>
#include <sys/ucontext.h>
#include "device_sche_context.h"
#include "device_common.h"
#include "aicore_manager.h"
#include "aicore_constants.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "tilefwk/aicore_print.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "device_trace.h"
#include "device_sche_alloc_thread.h"
#include "device_sche_wait_ctrl.h"

constexpr uint32_t LAUNCH_AICPU_NUM = 5;
constexpr int MAX_RETRIES = 100;

namespace npu::tile_fwk::dynamic {
struct AicoreLogManager {
    AicoreLogManager()
    {
        data_ = aligned_alloc(PAGE_SIZE, MAX_AICORE_NUM * PRINT_BUFFER_SIZE);
        uint8_t* buf = (uint8_t*)data_;
        for (uint32_t i = 0; i < MAX_AICORE_NUM; i++) {
            logger[i].Init(buf, PRINT_BUFFER_SIZE);
            buf += PRINT_BUFFER_SIZE;
        }
    }
    ~AicoreLogManager() { free(data_); }

    void* data_;
    AicoreLogger logger[MAX_AICORE_NUM];
};

typedef void (*sig_act_f)(int signum, siginfo_t* info, void* act);

class DeviceSchedMachine {
public:
    DeviceSchedMachine()
    {
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; ++i) {
            aicoreManager_[i] = std::make_unique<AiCoreManager>(schThreadStatus);
        }
    }

    void SetStachSchduleContext(int schedIdx, SchduleContext* context)
    {
        aicoreManager_[schedIdx]->SetSchduleContext(context);
    }

    bool CheckAndResetReg() { return aicoreManager_[0]->CheckAndResetReg(); }

    void init(uint32_t schNum)
    {
        schAicpuNum_ = schNum;
        schThreadStatus.Init();
    }

    int RunThread(int threadIdx, DevStartArgs* devStartArgs, DeviceArgs* args, int schedIdx, int arbitratedScheNum)
    {
        int ret = 0;
        if (args->nrAic == 0 || args->nrValidAic == 0 || args->nrAicpu < args->scheCpuNum) {
            DEV_ERROR(
                DevCommonErr::PARAM_INVALID,
                "#sche.thread.init: Device machine run invalid args: aicNum=%u, blockdim=%u, launchAicpuNum=%u, "
                "launchScheAicpuNum=%u",
                args->nrAic, args->nrValidAic, args->nrAicpu, args->scheCpuNum);
            return DEVICE_MACHINE_ERROR;
        }

        if (static_cast<uint32_t>(schedIdx) >= args->scheCpuNum) {
            DEV_INFO("thread start ignore ");
            return DEVICE_MACHINE_OK;
        }
#if ENABLE_AICORE_PRINT
        aicoreManager_[schedIdx]->InitLogger(logManager.logger);
#endif
        ret = aicoreManager_[schedIdx]->RunManager(threadIdx, devStartArgs, args, schedIdx, arbitratedScheNum);
        DEV_INFO("threadIdx=%d end, ret=%d", threadIdx, ret);
        return ret;
    }

    void ResetRegAll()
    {
        sleep(1);
        DEV_INFO("ResetRegAll");
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            aicoreManager_[i]->ResetRegAll();
        }
        sleep(1);
        aicoreManager_[0]->CheckAndResetReg();
        DEV_INFO("Exception reset reg finish.");
    }

private:
    SchThreadStatus schThreadStatus;
    uint32_t schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    std::unique_ptr<AiCoreManager> aicoreManager_[MAX_SCHEDULE_AICPU_NUM];
#if ENABLE_AICORE_PRINT
    AicoreLogManager logManager;
#endif
};

constexpr int CPUS_PER_CLUSTER = 4;
static constexpr uint64_t SIGNAL_DELAY_SECONDS = 2;
constexpr int SCHE_THREAD_START_IDX = 1;

struct DynMachineManager {
    struct KernelCtrlEntry {
        int (*kernelCtrlServerInit)(void* targ);
        int (*kernelCtrlServer)(void* targ);
    };

    static int AllocThreadIdxForDav3510(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx,
        std::atomic<uint64_t>& cpumask, std::atomic<int>& arbitrationLevel)
    {
        return AllocThreadIdxForDav3510Impl(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitrationLevel);
    }

    static int AllocThreadIdxForDav2201(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx,
        std::atomic<uint64_t>& cpumask, int& arbitratedScheNum, std::atomic<int>& arbitrationLevel)
    {
        return AllocThreadIdxForDav2201Impl(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitratedScheNum, arbitrationLevel);
    }

    static int AllocThreadIdx(DeviceArgs* devArgs, int& curThreadIdx, std::atomic<int>& threadIdx, 
        std::atomic<uint64_t>& cpumask, int& arbitratedScheNum, std::atomic<int>& arbitrationLevel, std::atomic<int>& simCpuId)
    {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;

#ifdef __DEVICE__
        int cpu = sched_getcpu();
        (void) simCpuId;
#else
        int cpu = ++simCpuId;
#endif
        if (devArgs->archInfo == ArchInfo::DAV_3510) {
            ret = AllocThreadIdxForDav3510(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitrationLevel);
        } else if (devArgs->archInfo == ArchInfo::DAV_2201) {
            ret = AllocThreadIdxForDav2201(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitratedScheNum, arbitrationLevel);
        } else {
            curThreadIdx = ++threadIdx;
        }
        return ret;
    }

    void SignalReg(const sig_act_f sigAct)
    {
        DEV_INFO("Exception SignalReg.");
        struct sigaction myAct;
        (void)memset_s(&myAct, sizeof(myAct), 0, sizeof(myAct));
        sigemptyset(&myAct.sa_mask);
        myAct.sa_flags = SA_SIGINFO;
        myAct.sa_sigaction = sigAct;
        sigaction(SIGFPE, &myAct, &oriFPEAct_);
        sigaction(SIGBUS, &myAct, &oriBUSAct_);
        sigaction(SIGSEGV, &myAct, &oriSEGVAct_);
        sigaction(SIGPIPE, &myAct, &oriPIPEAct_);
        sigaction(SIGILL, &myAct, &oriILLAct_);
        sigaction(SIGABRT, &myAct, &oriBordAct_);
        return;
    }

    int RunCtrl(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry, int threadIdx)
    {
        DEV_TRACE_DEBUG(schema::CtrlEvent(threadIdx, schema::ThreadStart()));

        DEV_INFO("ThreadCtrlEnter idx=%d", threadIdx);

        int ret = entry.kernelCtrlServer(static_cast<void*>(kargs));

        DEV_INFO("ThreadCtrlLeave idx=%d ret=%d", threadIdx, ret);
        return ret;
    }

    int RunSche(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry, int threadIdx, int arbitratedScheNum)
    {
        UNUSED(entry);

        DeviceArgs* devArgs = PtrToPtr<int64_t, DeviceArgs>(kargs->cfgdata);
        DEV_INFO("DeviceMode=%s, isDeviceMode=%d, stage=%s, threadIdx=%d", IsDeviceMode() ? "device" : "sim", IsDeviceMode(), "RunSche.before", threadIdx);
        DEV_INFO("ThreadScheEnter idx=%d", threadIdx);

        DEV_INFO(
            "TaskType=%d, threadIdx=%d, aicNum=%u, aivNum=%u, aicpuNum=%u, validAicNum=%u.",
            static_cast<int>(devArgs->taskType), threadIdx, devArgs->nrAic, devArgs->nrAiv, devArgs->nrAicpu,
            devArgs->nrValidAic);
        DEV_INFO(
            "devQueueAddr=%#lx, sharedBuffer=%#lx, coreRegAddr=%#lx, corePmuAdr=%#lx.", devArgs->devQueueAddr,
            devArgs->sharedBuffer, devArgs->coreRegAddr, devArgs->corePmuAddr);
        DEV_TRACE_DEBUG(schema::ScheEvent(threadIdx, schema::ThreadStart()));

        devArgs->toSubMachineConfig = kargs->toSubMachineConfig;
        SchduleContext localContext;
        int schedIdx = threadIdx - SCHE_THREAD_START_IDX;
        schMachine_.SetStachSchduleContext(schedIdx, &localContext);
        DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(kargs->cfgdata);
        DevStartArgs* devStartArgs =
            reinterpret_cast<DevStartArgs*>(devProg->GetRuntimeDataList()->GetRuntimeDataCurrent());
        int ret = schMachine_.RunThread(threadIdx, devStartArgs, devArgs, schedIdx, arbitratedScheNum);

        DEV_INFO("ThreadScheLeave idx=%d ret=%d", threadIdx, ret);
        return ret;
    }

    void RunSchInit(DeviceArgs *args)
    {
        if (initSch_.load()) {
            return;
        }
        schMachine_.init(args->scheCpuNum);
        initSch_.store(true);
    }

    void RunSchDeInit()
    {
        cpumask_ = 0;
        schExitNum_ = 0;
        arbitrationLevel_.store(ARBIT_UNSET, std::memory_order_release);
        ctrlWaitLevel_.store(CTRL_WAIT_UNSET, std::memory_order_release);
        initSch_.store(false);
        globalThreadIdx_.store(0);
        simCpuId_.store(0);
    }

    void RunSchPost(DevAscendProgram *devProg)
    {
        ReleaseRuntimeDataRingBuffer(devProg);
        DEV_INFO("All schedule exited, destroy the machine.");
    }

    int SyncSchExit(DevAscendProgram* devProg, const DeviceArgs& devArgs, int ret)
    {
        if (++schExitNum_ == devArgs.nrAicpu) {
            RunSchPost(devProg);
            RunSchDeInit();
            PerfEvtMgr::Instance().AddScheduleTurn();
            DEV_INFO("All sche cpu exited.");
            return ret;
        }
        return ret;
    }

    int RunCtrlInitNoLock(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        int ret = entry.kernelCtrlServerInit(kargs);
        return ret;
    }

    __sighandler_t GetSigHandle(int signum)
    {
        __sighandler_t handle = nullptr;
        if (signum == static_cast<int>(SIGFPE)) {
            handle = oriFPEAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGBUS)) {
            handle = oriBUSAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGSEGV)) {
            handle = oriSEGVAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGPIPE)) {
            handle = oriPIPEAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGILL)) {
            handle = oriILLAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGABRT)) {
            handle = oriBordAct_.sa_handler;
        }
        return handle;
    }

    void SigAct(int signum, siginfo_t* info, void* act)
    {
        (void)info;
        (void)act;
        DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Exception Signum[%d] Act.", signum);
        PrintBacktrace(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "signal " + std::to_string(signum));
        if (reset_.load()) {
            DEV_WARN("#sche.except.reset: Exception Already reset.");
            sleep(SIGNAL_DELAY_SECONDS);
            return;
        }
        reset_.store(true);
        if (!initSch_.load() && !initCtrl_.load()) {
            DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Exception call ori sigact.");
            __sighandler_t handle = GetSigHandle(signum);
            if (handle == SIG_DFL) {
                DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Ori sigact SIG_DFL.");
                signal(signum, SIG_DFL);
                raise(signum);
            } else if (handle == SIG_IGN) {
                DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Ori sigact SIG_IGN.");
            } else if (handle != nullptr) {
                DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Call Ori sigact.");
                handle(signum);
            }
            return;
        }
        schMachine_.ResetRegAll();
        sigaction(SIGFPE, &oriFPEAct_, nullptr);
        sigaction(SIGBUS, &oriBUSAct_, nullptr);
        sigaction(SIGSEGV, &oriSEGVAct_, nullptr);
        sigaction(SIGPIPE, &oriPIPEAct_, nullptr);
        sigaction(SIGILL, &oriILLAct_, nullptr);
        sigaction(SIGABRT, &oriBordAct_, nullptr);
        (void)raise(signum);
        return;
    }

    void ReleaseRuntimeDataRingBuffer(DevAscendProgram* devProg)
    {
        RuntimeDataRingBufferHead* runtimeDataList = devProg->GetRuntimeDataList();
        runtimeDataList->Deallocate(runtimeDataList->GetRuntimeDataCurrent());
        DEV_INFO("Runtimedata: %lu, %lu", runtimeDataList->GetIndexFinished(), runtimeDataList->GetIndexPending());
    }

    int EntrySplittedStreamCtrl(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        // ctrl start only one thread
        DEV_INFO("Ctrl enter round=%d", (int)kargs->parameter.globalRound);
        initCtrl_.store(true);
        int ret = RunCtrlInitNoLock(kargs, entry);
        if (ret != 0) {
            initCtrl_.store(false);
            DeviceTrace::GetInstance().ReportTraceMsg();
            return ret;
        }
        kargs->taskWastTime = GetCycles();
        ret = RunCtrl(kargs, entry, 0);
        PerfMtTrace(PERF_TRACE_BEGIN, 0, kargs->taskWastTime);
        PerfMtTrace(PERF_TRACE_EXIT, 0);
        DEV_INFO("Ctrl leave ret=%d", ret);
        if (ret != DEVICE_MACHINE_OK) {
            DeviceTrace::GetInstance().ReportTraceMsg();
        }
        initCtrl_.store(false);
        PerfEvtMgr::Instance().AddCtrlTurn();
        return ret;
    }

    void ReCalcDevArgsAicoreNum(DeviceKernelArgs* kargs, DevAscendProgram* devProg)
    {
        if (kargs->parameter.ctrlBlockNum != 0 &&
            static_cast<uint32_t>(kargs->parameter.ctrlBlockNum) != devProg->devArgs.nrValidAic) {
            devProg->devArgs.nrValidAic = kargs->parameter.ctrlBlockNum;
            DEV_INFO("control aicore before launch, nrValidAic changed to %lu", kargs->parameter.ctrlBlockNum);
        }
    }

    int EntrySplittedStreamSche(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        DevAscendProgram* devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
        auto beginTime = GetCycles(); // After wait, the devStartArgs should be ready.
        ReCalcDevArgsAicoreNum(kargs, devProg);
        auto devArgs = devProg->devArgs;
        int threadIdx = -1;
        RunSchInit(&devArgs);
        int arbitratedScheNum = devArgs.scheCpuNum;
        if (AllocThreadIdx(&devArgs, threadIdx, globalThreadIdx_, cpumask_, arbitratedScheNum, arbitrationLevel_, simCpuId_) != DEVICE_MACHINE_OK) {
            DEV_ERROR(
                ThreadErr::THREAD_CPU_ALLOC_FAILED, "#sche.thread.init: Current cpu[%d] alloc thread failed.",
                sched_getcpu());
            DEV_ATRACE("Schedule Current cpu[%d] alloc thread failed", sched_getcpu());
            DeviceTrace::GetInstance().ReportTraceMsg();
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_ERROR;
        }
        if (threadIdx != -1) {
            int scheWaitRet = WaitForCtrlDecision(devProg, devArgs.archInfo, threadIdx, arbitratedScheNum, ctrlWaitLevel_);
            if (scheWaitRet != DEVICE_MACHINE_OK) {
                DEV_ERROR(SchedErr::RINGBUFFER_WAIT_TIMEOUT, "#sche.wait: ScheWait failed, ret=%d.", scheWaitRet);
                DeviceTrace::GetInstance().ReportTraceMsg();
                return scheWaitRet;
            }
        }
        PerfMtTrace(PERF_TRACE_ALLOC_THREAD_ID, threadIdx);
        PerfMtTrace(PERF_TRACE_BEGIN, threadIdx, beginTime);
        int ret = DEVICE_MACHINE_OK;
        if (threadIdx != -1 && threadIdx <= arbitratedScheNum) {
            DEV_INFO("SchedThreadEnter idx=%d round=%d", threadIdx, (int)kargs->parameter.globalRound);
            ret = RunSche(kargs, entry, threadIdx, arbitratedScheNum);
            DEV_INFO("SchedThreadLeave idx=%d ret=%d", threadIdx, ret);
            if (ret != DEVICE_MACHINE_OK) {
                DeviceTrace::GetInstance().ReportTraceMsg();
            }
            PerfMtTrace(PERF_TRACE_EXIT, threadIdx);
        }
        return SyncSchExit(devProg, devArgs, ret);
    }

    int Entry(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        switch (kargs->parameter.runMode) {
            case RUN_SPLITTED_STREAM_CTRL:
                return EntrySplittedStreamCtrl(kargs, entry);
                break;
            case RUN_SPLITTED_STREAM_SCHE:
                return EntrySplittedStreamSche(kargs, entry);
                break;
            default:
                DEV_ERROR(
                    DevCommonErr::PARAM_INVALID, "#dev.entry.invalid_mode: Invalid run mode: %d\n",
                    (int)kargs->parameter.runMode);                
                break;
        }
        return DEVICE_MACHINE_INVALID_RUN_MODE;
    }

    int LastFinishThreadIdx_{0};
    std::atomic<uint64_t> cpumask_{0};
    std::atomic<uint32_t> schExitNum_{0};
    std::atomic<int> arbitrationLevel_{ARBIT_UNSET};
    std::atomic<int> ctrlWaitLevel_{CTRL_WAIT_UNSET};
    std::atomic<int> simCpuId_{0};
    DeviceSchedMachine schMachine_;
    struct sigaction oriFPEAct_;
    struct sigaction oriBUSAct_;
    struct sigaction oriSEGVAct_;
    struct sigaction oriPIPEAct_;
    struct sigaction oriILLAct_;
    struct sigaction oriBordAct_;
    std::atomic<bool> reset_{false};
    std::atomic<bool> initCtrl_{false};
    std::atomic<bool> initSch_{false};
    std::atomic<bool> schRunFailed_{false};
    std::atomic<int> globalThreadIdx_{0};

    struct SplittedInfo {
        bool ScheSync(DevStartArgs* devStartArgs, int schNum)
        {
            return ++devStartArgs->devScheState.finished == schNum;
        }
    } splittedInfo_;
};

} // namespace npu::tile_fwk::dynamic
