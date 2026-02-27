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

#include "device_common.h"
#include "aicore_manager.h"
#include "aicore_constants.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "tilefwk/aicore_print.h"
#include "machine/device/dynamic/aicore_prof.h"

constexpr uint32_t LAUNCH_AICPU_NUM = 5;

namespace npu::tile_fwk::dynamic {
struct AicoreLogManager {
    AicoreLogManager() {
        data_ = aligned_alloc(PAGE_SIZE, MAX_AICORE_NUM * PRINT_BUFFER_SIZE);
        uint8_t *buf = (uint8_t *)data_;
        for (uint32_t i = 0; i < MAX_AICORE_NUM; i++) {
            logger[i].Init(buf, PRINT_BUFFER_SIZE);
            buf += PRINT_BUFFER_SIZE;
        }
    }
    ~AicoreLogManager() { free(data_); }

    void *data_;
    AicoreLogger logger[MAX_AICORE_NUM];
};

class DeviceSchedMachine {
public:
    DeviceSchedMachine() {
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; ++i) {
            aicoreManager_[i] = std::make_unique<AiCoreManager>(aicpuTaskManager_);
        }
    }

    void SetStachSchduleContext(int schedIdx, SchduleContext* context) {
        aicoreManager_[schedIdx]->SetSchduleContext(context);
    }

    bool CheckAndResetReg(){
        return aicoreManager_[0]->CheckAndResetReg();
    }

    void init(uint32_t schNum) {
        schAicpuNum_ = schNum;
    }

    int RunThread(int threadIdx, DevStartArgs *devStartArgs, DeviceArgs *args, int schedIdx) {
        int ret = 0;
        if (args->nrAic == 0 || args->nrValidAic == 0 || args->nrAicpu < NEED_LAUNCH_AICPU_MINNUM) {
            DEV_ERROR("Device machinr run invalid args aicnum:%u, blockdim:%u, launchAicpu num:%u",
                args->nrAic, args->nrValidAic, args->nrAicpu);
            return DEVICE_MACHINE_ERROR;
        }

        if (static_cast<uint32_t>(schedIdx) >= args->scheCpuNum) {
            DEV_INFO("thread start ignore ");
            return DEVICE_MACHINE_OK;
        }
#if ENABLE_AICORE_PRINT
        aicoreManager_[schedIdx]->InitLogger(logManager.logger);
#endif
        ret = aicoreManager_[schedIdx]->RunManager(threadIdx, devStartArgs, args, schedIdx);
        DEV_INFO("thread  %d end , ret = %d", threadIdx, ret);
        return ret;
    }

    void ResetRegAll() {
      sleep(1);
      DEV_ERROR("ResetRegAll");
      for (uint32_t i = 0; i < schAicpuNum_; ++i) {
        aicoreManager_[i]->ResetRegAll();
      }
      sleep(1);
      aicoreManager_[0]->CheckAndResetReg();
      DEV_ERROR("Exception reset reg finish.");
    }

    inline void DumpAicorePerfTrace(std::string file = "") {
        (void)file;
#if ENABLE_PERF_TRACE
        std::ostringstream oss;
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            aicoreManager_[i]->DumpAicorePerfTrace(oss);
            oss << (i == schAicpuNum_ - 1 ? "" : ",");
        }

        const std::string& str = oss.str();
        uint32_t totalLength = str.length();
        uint32_t startPos = 0;
        uint32_t batchSize = 600;
        while (startPos < totalLength) {
            uint32_t endPos = std::min(startPos + batchSize, totalLength);
            std::string batch = str.substr(startPos, endPos - startPos);
            DEV_ERROR("tile_fwk aicore prof:%s", batch.c_str());
            startPos = endPos;
        }

        if (file != "") {
            std::ofstream os(file);
            os << "[";
            os << oss.str();
            os << "]";
        }
#endif
    }

private:
    AicpuTaskManager aicpuTaskManager_;
    uint32_t schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    std::unique_ptr<AiCoreManager> aicoreManager_[MAX_SCHEDULE_AICPU_NUM];
#if ENABLE_AICORE_PRINT
    AicoreLogManager logManager;
#endif
};

constexpr int CPUS_PER_CLUSTER = 4;
static constexpr uint64_t SIGNAL_DELAY_SECONDS = 2;

struct DynMachineManager {
    struct KernelCtrlEntry {
        void (*sigAct)(int signum, siginfo_t* info, void* act);
        int (*kernelCtrlServerInit)(void *targ);
        int (*kernelCtrlServer)(void *targ);
    };

    int AllocThreadIdx(int nrAicpu, uint32_t scheCpuNum, std::atomic<int> &threadIdx) { 
        if (scheCpuNum == 1) { 
            return ++threadIdx; 
        } 
        int cpu = sched_getcpu(); 
        cpumask_.fetch_or(1 << cpu, std::memory_order_release); 
        while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != nrAicpu) { 
            sched_yield(); 
        } 
        auto maskval = cpumask_.load(std::memory_order_relaxed); 
        int cpuoff = 0; 
        int clus_id = -1; 
        for (int index = 0; index < static_cast<int>(sizeof(uint64_t)); ++index) { 
            int mask = (maskval >> cpuoff) & 0xF; 
            if (__builtin_popcount(static_cast<uint32_t>(mask)) >= static_cast<int>(scheCpuNum)) { 
                clus_id = index; 
                break; 
            } 
            cpuoff += CPUS_PER_CLUSTER; 
        } 
        if (clus_id == -1) { 
            return ++threadIdx; 
        } 
        if (cpu < cpuoff || cpu >= (cpuoff + CPUS_PER_CLUSTER)) { 
            return -1; 
        } 
        return ++threadIdx; 
    }

    void SignalReg(const KernelCtrlEntry &entry) {
        if (sigReg_) {
            return;
        }
        sigReg_ = true;
        DEV_INFO("Exception SignalReg.");
        struct sigaction myAct;
        (void)memset_s(&myAct, sizeof(myAct), 0, sizeof(myAct));
        sigemptyset(&myAct.sa_mask);
        myAct.sa_flags = SA_SIGINFO;
        myAct.sa_sigaction = entry.sigAct;
        sigaction(SIGFPE, &myAct, &oriFPEAct_);
        sigaction(SIGBUS, &myAct, &oriBUSAct_);
        sigaction(SIGSEGV, &myAct, &oriSEGVAct_);
        sigaction(SIGPIPE, &myAct, &oriPIPEAct_);
        sigaction(SIGILL, &myAct, &oriILLAct_);
        sigaction(SIGABRT, &myAct, &oriBordAct_);
        return;
    }

    int RunCtrl(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry, int threadIdx) {
        DEV_TRACE_DEBUG(schema::CtrlEvent(threadIdx, schema::ThreadStart()));

        DEV_INFO("ThreadCtrlEnter idx=%d", threadIdx);

        int ret = entry.kernelCtrlServer(static_cast<void*>(kargs));

        DEV_INFO("ThreadCtrlLeave idx=%d ret=%d", threadIdx, ret);
        return ret;
    }

    int RunSche(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry, int threadIdx) {
        UNUSED(entry);

        DeviceArgs *devArgs = PtrToPtr<int64_t, DeviceArgs>(kargs->cfgdata);
        DEV_INFO("ThreadScheEnter idx=%d", threadIdx);

        DEV_INFO("TaskType %d threadIdx %d aicNum %u aivNum %u aicpuNum %u validAicNum %u .",
            static_cast<int>(devArgs->taskType), threadIdx, devArgs->nrAic,
            devArgs->nrAiv, devArgs->nrAicpu, devArgs->nrValidAic);
        DEV_INFO("devQueueAddr %lx, sharedBuffer %lx coreRegAddr %lx corePmuAdr %lx .", devArgs->devQueueAddr,
            devArgs->sharedBuffer, devArgs->coreRegAddr, devArgs->corePmuAddr);
        DEV_TRACE_DEBUG(schema::ScheEvent(threadIdx, schema::ThreadStart()));

        devArgs->toSubMachineConfig = kargs->toSubMachineConfig;
        SchduleContext localContext;
        int schedIdx = threadIdx - 1;
        machine_.SetStachSchduleContext(schedIdx, &localContext);
        DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(kargs->cfgdata);
        DevStartArgs *devStartArgs = reinterpret_cast<DevStartArgs *>(devProg->GetRuntimeDataList()->GetRuntimeDataCurrent());
        int ret = machine_.RunThread(threadIdx, devStartArgs, devArgs, schedIdx);

        DEV_INFO("ThreadScheLeave idx=%d ret=%d", threadIdx, ret);
        if (ret != DEVICE_MACHINE_OK) {
            schRunFailed_ = true;
        }
        return ret;
    }

    void RunPost(DevAscendProgram *devProg) {
        ReleaseRuntimeDataRingBuffer(devProg);
        DEV_INFO("All schedule exited, destroy the machine.");
        DeInit();
#if ENABLE_PERF_TRACE
        PerfMtTrace(PERF_TRACE_EXIT, LastFinishThreadIdx_);
        DEV_ERROR("Begin dump machine perf trace:");
        PerfEvtMgr::Instance().DumpPerfTrace(devProg->devArgs.scheCpuNum, "/tmp/tile_fwk_aicpu_perftrace.json");
        DEV_IF_DEVICE {
            machine_.DumpAicorePerfTrace("tmp/tile_fwk_aicore_perftrace.json");
        }
        DEV_ERROR("Finish dump machine perf trace.");
#endif
    }

    int RunUnifiedStream(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        DeviceArgs *devArgs = PtrToPtr<int64_t, DeviceArgs>(kargs->cfgdata);
        if (devArgs->scheCpuNum > devArgs->nrAicpu - 1) {
            DEV_ERROR("Aicpu num[%u] less than sche num[%u].", devArgs->nrAicpu, devArgs->scheCpuNum);
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_ERROR;
        }
        int threadIdx = AllocThreadIdx(devArgs->nrAicpu, devArgs->scheCpuNum, threadIdx_);	 
        uint64_t allocThreadCycle = GetCycles();

        if ((threadIdx != -1) && threadIdx <= static_cast<int>(devArgs->scheCpuNum)) {	 
             ret = RunSche(kargs, entry, threadIdx);
        } else {
            threadIdx = ctrlcpuIdx_.fetch_add(1);	 
            DEV_INFO("TaskType %d.",  static_cast<int>(devArgs->taskType)); 
            if (devArgs->enableCtrl == 1 && threadIdx == CTRL_CPU_THREAD_IDX) { 
                ret = RunCtrl(kargs, entry, threadIdx);
            } else {
                threadIdx += devArgs->scheCpuNum;
                SignalReg(entry); 
            }
        }

        PerfMtTrace(PERF_TRACE_BEGIN, threadIdx, kargs->taskWastTime);
        PerfMtTrace(PERF_TRACE_ALLOC_THREAD_ID, threadIdx, allocThreadCycle);

        DEV_INFO("ThreadLeave idx=%d ret=%d", threadIdx, ret);

        PerfMtTrace(PERF_TRACE_EXIT, threadIdx);
        if (++finished_ == static_cast<std::atomic<int>>(devArgs->nrAicpu)) {
            LastFinishThreadIdx_ = threadIdx;
            if (unlikely(!machine_.CheckAndResetReg())) {
                DEV_WARN("Some registers force closed!");
            }
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED;
        }
        return ret;
    }

    int RunCtrlInitNoLock(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
#ifdef __DEVICE__
        auto devArgs = PtrToPtr<int64_t, DeviceArgs>(kargs->cfgdata);
        if (devArgs->aicpuPerfAddr != 0) {
            PerfEvtMgr::Instance().SetIsOpenProf(true, devArgs->aicpuPerfAddr);
        }
#endif
        int ret = entry.kernelCtrlServerInit(kargs);
        return ret;
    }

    int RunCtrlInit(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
        int ret = DEVICE_MACHINE_OK;
        mutex_.lock();
        if (!initCtrl_.load()) {
            initCtrl_.store(true);
            ret = RunCtrlInitNoLock(kargs, entry);
        }
        mutex_.unlock();
        return ret;
    }

    void Init(DeviceArgs *args) {
        if (init_.load()) {
            return;
        }
        init_.store(true);
        ctrlcpuIdx_.store(0);
        machine_.init(args->scheCpuNum);
        schRunFailed_ = false;
    }

    void DeInit() {
        threadIdx_ = 0;
        finished_ = 0;
        cpumask_ = 0;
        ctrlcpuIdx_ = 0;
        init_.store(false);
        initCtrl_.store(false);
    }

    __sighandler_t GetSigHandle(int signum) {
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

    void SigAct(int signum, siginfo_t* info, void* act) {
        (void)info;
        (void)act;
        DEV_ERROR("Exception Signum[%d] Act.", signum);
        PrintBacktrace("signal " + std::to_string(signum));
        if (reset_.load()) {
            DEV_ERROR("Exception Already reset.");
            sleep(SIGNAL_DELAY_SECONDS);
            return;
        }
        reset_.store(true);
        if (!init_.load()) {
            DEV_ERROR("Exception call ori sigact.");
            __sighandler_t handle = GetSigHandle(signum);
            if (handle == SIG_DFL) {
                DEV_ERROR("Ori sigact SIG_DFL.");
                signal(signum, SIG_DFL);
                raise(signum);
            } else if (handle == SIG_IGN) {
                DEV_ERROR("Ori sigact SIG_IGN.");
            } else if (handle != nullptr) {
                DEV_ERROR("Call Ori sigact.");
                handle(signum);
            }
            return;
        }
        machine_.ResetRegAll();
        sigaction(SIGFPE, &oriFPEAct_, nullptr);
        sigaction(SIGBUS, &oriBUSAct_, nullptr);
        sigaction(SIGSEGV, &oriSEGVAct_, nullptr);
        sigaction(SIGPIPE, &oriPIPEAct_, nullptr);
        sigaction(SIGILL, &oriILLAct_, nullptr);
        sigaction(SIGABRT, &oriBordAct_, nullptr);
        (void)raise(signum);
        return;
    }

    void ReleaseRuntimeDataRingBuffer(DevAscendProgram *devProg) {
        RuntimeDataRingBufferHead *runtimeDataList = devProg->GetRuntimeDataList();
        runtimeDataList->Deallocate(runtimeDataList->GetRuntimeDataCurrent());
    }

    int EntryUnifiedStream(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
        auto ret = RunCtrlInit(kargs, entry);
        if (ret != DEVICE_MACHINE_OK) {
            DEV_ERROR("Server init failed");
            return ret;
        }
        DevAscendProgram *devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
        kargs->taskWastTime = GetCycles();
        Init(&devProg->devArgs);
        int rc = RunUnifiedStream(kargs, entry);
        if (rc == npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED) {
            RunPost(devProg);
            return DEVICE_MACHINE_OK;
        }
        return rc;
    }

    int EntrySplittedStreamCtrl(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
        int ret = 0;
        constexpr int ctrlThreadIdx = 0;
        uint64_t ctrlStep = splittedInfo_.ctrlStep++;
        // ctrl start 2 threads: one for ctrl, one for registering signal
        if (ctrlStep % 2 == 0) {
            DEV_INFO("CtrlThreadEnter idx=%d round=%d", ctrlThreadIdx, (int)kargs->parameter.globalRound);
            ret = RunCtrlInitNoLock(kargs, entry);
            if (ret != 0) {
                return ret;
            }
            ret = RunCtrl(kargs, entry, ctrlThreadIdx);
            DEV_INFO("CtrlThreadLeave idx=%d ret=%d", ctrlThreadIdx, ret);
        } else {
            SignalReg(entry);
        }
        return ret;
    }

    int EntrySplittedStreamSche(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
        DevAscendProgram *devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);

        splittedInfo_.ScheWait(devProg);
        // After wait, the devStartArgs should be ready.

        DevStartArgs *runtimeDataCurrent = reinterpret_cast<DevStartArgs *>(devProg->GetRuntimeDataList()->GetRuntimeDataCurrent());
        auto devArgs = devProg->devArgs;
        int threadIdx = AllocThreadIdx(devArgs.nrAicpu, devArgs.scheCpuNum, runtimeDataCurrent->devScheState.threadIdx);
        int ret = DEVICE_MACHINE_OK;
        if (threadIdx != -1 && threadIdx <= static_cast<int>(devArgs.scheCpuNum)) {
            DEV_INFO("SchedThreadEnter idx=%d round=%d", threadIdx, (int)kargs->parameter.globalRound);
            ret = RunSche(kargs, entry, threadIdx);
            DEV_INFO("SchedThreadLeave idx=%d ret=%d", threadIdx, ret);

            if (splittedInfo_.ScheSync(runtimeDataCurrent, devArgs.scheCpuNum)) {
                if (unlikely(!machine_.CheckAndResetReg())) {
                    DEV_WARN("Some registers force closed!");
                }
                RunPost(devProg);
                DEV_INFO("All schedule exited, destroy the machine.");
                return DEVICE_MACHINE_OK;
            }
        }
        return ret;
    }

    int Entry(DeviceKernelArgs *kargs, const KernelCtrlEntry &entry) {
        switch (kargs->parameter.runMode) {
            case RUN_UNIFIED_STREAM:
                return EntryUnifiedStream(kargs, entry);
                break;
            case RUN_SPLITTED_STREAM_CTRL:
                return EntrySplittedStreamCtrl(kargs, entry);
                break;
            case RUN_SPLITTED_STREAM_SCHE:
                return EntrySplittedStreamSche(kargs, entry);
                break;
            default:
                DEV_ERROR("Invalid run mode: %d\n", (int)kargs->parameter.runMode);
                break;
        }
        return DEVICE_MACHINE_INVALID_RUN_MODE;
    }

    int LastFinishThreadIdx_{0};
    std::atomic<int> threadIdx_{0};
    std::atomic<int> finished_{0};
    std::atomic<uint64_t> cpumask_{0};
    std::atomic<int> ctrlcpuIdx_{0};
    DeviceSchedMachine machine_;
    bool sigReg_{false};
    struct sigaction oriFPEAct_;
    struct sigaction oriBUSAct_;
    struct sigaction oriSEGVAct_;
    struct sigaction oriPIPEAct_;
    struct sigaction oriILLAct_;
    struct sigaction oriBordAct_;
    std::atomic<bool> reset_{false};
    std::atomic<bool> init_{false};
    std::atomic<bool> initCtrl_{false};
    std::mutex mutex_;
    std::atomic<bool> schRunFailed_{false};

    struct SplittedInfo {
        std::atomic<uint64_t> ctrlStep{0};
        std::atomic<uint64_t> currentRound{0};

        void ScheWait(DevAscendProgram *devProg) {
            while (unlikely(!devProg->runtimeDataRingBufferInited)) {
                /* In the first launch, sche must wait for ctrl's ring buffer's initialization.
                 * Otherwise, the ringBufferHead->Empty() is not legal. */
                RuntimeYield(0);
            }

            RuntimeDataRingBufferHead *ringBufferHead = devProg->GetRuntimeDataList();
            while (unlikely(ringBufferHead->Empty())) {
                /* Sche must wait until the current devStarArgs has been initialized. */
                RuntimeYield(0);
            }
        }

        bool ScheSync(DevStartArgs *devStartArgs, int schNum) {
            return ++devStartArgs->devScheState.finished == schNum;
        }
    } splittedInfo_;
};

} // namespace npu::tile_fwk
