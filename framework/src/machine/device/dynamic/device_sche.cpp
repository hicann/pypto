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
 * \file device_machine.cpp
 * \brief
 */

#include "device_sche.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sched.h>
#include <signal.h>
#include <sys/ucontext.h>
#include "machine/device/dynamic/device_utils.h"
#include "machine/utils/device_log.h"
#include "device_utils.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
constexpr int CPUS_PER_CLUSTER = 4;
constexpr uint64_t SIGNAL_DELAY_SECONDS = 2;

extern void SigAct(int signum, siginfo_t* info, void* act);
extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServerInit(void *targ);
extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServer(void *targ);

struct DynMachineManager {
    int allocThreadIdx(int nrAicpu) {
        if (schAicpuNum_ == 1) {
            return threadIdx_++;
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
            if (__builtin_popcount(static_cast<uint32_t>(mask)) >= schAicpuNum_) {
                clus_id = index;
                break;
            }
            cpuoff += CPUS_PER_CLUSTER;
        }
        if (clus_id == -1) {
            return threadIdx_++;
        }
        if (cpu < cpuoff || cpu >= (cpuoff + CPUS_PER_CLUSTER)) {
            return -1;
        }
        return threadIdx_++;
    }

    void SignalReg() {
        DEV_INFO("Exception SignalReg.");
        struct sigaction myAct;
        (void)memset_s(&myAct, sizeof(myAct), 0, sizeof(myAct));
        sigemptyset(&myAct.sa_mask);
        myAct.sa_flags = SA_SIGINFO;
        myAct.sa_sigaction = SigAct;
        sigaction(SIGFPE, &myAct, &oriFPEAct_);
        sigaction(SIGBUS, &myAct, &oriBUSAct_);
        sigaction(SIGSEGV, &myAct, &oriSEGVAct_);
        sigaction(SIGPIPE, &myAct, &oriPIPEAct_);
        sigaction(SIGILL, &myAct, &oriILLAct_);
        sigaction(SIGABRT, &myAct, &oriBordAct_);
        return;
    }

    int Run(AstKernelArgs *args) {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        auto devArgs = PtrToPtr<int64_t, DeviceArgs>(args->cfgdata);
        if ((uint32_t)schAicpuNum_ > devArgs->nrAicpu - 1) {
            DEV_ERROR("Aicpu num[%u] less than sche num[%d].", devArgs->nrAicpu, schAicpuNum_);
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_ERROR;
        }
        int threadIdx = allocThreadIdx(devArgs->nrAicpu);
        uint64_t allocThreadCycle = GetCycles();
        if ((threadIdx != -1) && threadIdx < schAicpuNum_) {
            CreateLogFile(LogType::LOG_TYPE_SCHEDULER, threadIdx);
            DEV_INFO("TaskType %d threadIdx %d aicNum %u aivNum %u aicpuNum %u validAicNum %u .",
                static_cast<int>(devArgs->taskType), threadIdx, devArgs->nrAic,
                devArgs->nrAiv, devArgs->nrAicpu, devArgs->nrValidAic);
            DEV_INFO("devQueueAddr %lx, sharedBuffer %lx coreRegAddr %lx corePmuAdr %lx .", devArgs->devQueueAddr,
                devArgs->sharedBuffer, devArgs->coreRegAddr, devArgs->corePmuAddr);
            DEV_TRACE_DEBUG(schema::ScheEvent(threadIdx, schema::ThreadStart()));
            ret = machine_.Run(threadIdx, devArgs);
            if (ret != DEVICE_MACHINE_OK) {
                schRunFailed_ = true;
            }
        } else {
            threadIdx = ctrlcpuIdx_.fetch_add(1);
            DEV_INFO("TaskType %d.",  static_cast<int>(devArgs->taskType));
            if (devArgs->enableCtrl == 1 && threadIdx == schAicpuNum_) {
                CreateLogFile(LogType::LOG_TYPE_CONTROLLER, 0);
                DEV_TRACE_DEBUG(schema::CtrlEvent(threadIdx, schema::ThreadStart()));
                ret = PyptoKernelCtrlServer(static_cast<void*>(args));
            } else {
                SignalReg();
            }
        }
        PerfMtTrace(PERF_TRACE_BEGIN, threadIdx, args->taskWastTime);
        PerfMtTrace(PERF_TRACE_ALLOC_THREAD_ID, threadIdx, allocThreadCycle);
        DEV_INFO("ThreadIdx %d finished, ret %d .", threadIdx, ret);
        GetLogger().Flush();
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

    void Init(DeviceArgs *args) {
        if (init_.load()) {
            return;
        }
        init_.store(true);
        schAicpuNum_ = args->scheCpuNum;
        ctrlcpuIdx_.store(schAicpuNum_);
        machine_.init(schAicpuNum_);
        schRunFailed_ = false;
    }

    void DeInit() {
      threadIdx_ = 0;
      finished_ = 0;
      cpumask_ = 0;
      ctrlcpuIdx_ = 0;
      init_.store(false);
    }

    int LastFinishThreadIdx_{0};
    std::atomic<int> threadIdx_{0};
    std::atomic<int> finished_{0};
    std::atomic<uint64_t> cpumask_{0};
    std::atomic<int> ctrlcpuIdx_{0};
    int schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    DeviceMachine machine_;
    struct sigaction oriFPEAct_;
    struct sigaction oriBUSAct_;
    struct sigaction oriSEGVAct_;
    struct sigaction oriPIPEAct_;
    struct sigaction oriILLAct_;
    struct sigaction oriBordAct_;
    std::atomic<bool> reset_{false};
    std::atomic<bool> init_{false};
    std::atomic<bool> schRunFailed_{false};
};

DynMachineManager g_machine_mgr;

__sighandler_t GetSigHandle(int signum) {
    __sighandler_t handle = nullptr;
    if (signum == static_cast<int>(SIGFPE)) {
        handle = g_machine_mgr.oriFPEAct_.sa_handler;
    } else if (signum == static_cast<int>(SIGBUS)) {
        handle = g_machine_mgr.oriBUSAct_.sa_handler;
    } else if (signum == static_cast<int>(SIGSEGV)) {
        handle = g_machine_mgr.oriSEGVAct_.sa_handler;
    } else if (signum == static_cast<int>(SIGPIPE)) {
        handle = g_machine_mgr.oriPIPEAct_.sa_handler;
    } else if (signum == static_cast<int>(SIGILL)) {
        handle = g_machine_mgr.oriILLAct_.sa_handler;
    } else if (signum == static_cast<int>(SIGABRT)) {
        handle = g_machine_mgr.oriBordAct_.sa_handler;
    }
    return handle;
}

void SigAct(int signum, siginfo_t* info, void* act) {
    (void)info;
    (void)act;
    DEV_ERROR("Exception Signum[%d] Act.", signum);
    PrintBacktrace("signal " + std::to_string(signum));
    if (g_machine_mgr.reset_.load()) {
        DEV_ERROR("Exception Already reset.");
        sleep(SIGNAL_DELAY_SECONDS);
        return;
    }
    g_machine_mgr.reset_.store(true);
    if (!g_machine_mgr.init_.load()) {
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
    g_machine_mgr.machine_.ResetRegAll();
    sigaction(SIGFPE, &g_machine_mgr.oriFPEAct_, nullptr);
    sigaction(SIGBUS, &g_machine_mgr.oriBUSAct_, nullptr);
    sigaction(SIGSEGV, &g_machine_mgr.oriSEGVAct_, nullptr);
    sigaction(SIGPIPE, &g_machine_mgr.oriPIPEAct_, nullptr);
    sigaction(SIGILL, &g_machine_mgr.oriILLAct_, nullptr);
    sigaction(SIGABRT, &g_machine_mgr.oriBordAct_, nullptr);
    (void)raise(signum);
    return;
}
}


extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *targ) {
    return PyptoKernelCtrlServerInit(targ);
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *targ) {
    auto kargs = (AstKernelArgs *)targ;
    auto devArgs = PtrToPtr<int64_t, DeviceArgs>(kargs->cfgdata);
    kargs->taskWastTime = GetCycles();
    g_machine_mgr.Init(devArgs);
    int rc = g_machine_mgr.Run(kargs);
    if (rc == npu::tile_fwk::dynamic::DEVICE_MACHINE_FINISHED) {
        DEV_INFO("All schedule exited, destroy the machine.");
        g_machine_mgr.DeInit();
#if ENABLE_PERF_TRACE
        PerfMtTrace(PERF_TRACE_EXIT, g_machine_mgr.LastFinishThreadIdx_);
        DEV_ERROR("Begin dump machine perf trace:");
        PerfEvtMgr::Instance().DumpPerfTrace("/tmp/tile_fwk_aicpu_perftrace.json");
        DEV_IF_DEVICE {
            g_machine_mgr.machine_.DumpAicorePerfTrace("tmp/tile_fwk_aicore_perftrace.json");
        }
        DEV_ERROR("Finish dump machine perf trace.");
#endif
        return DEVICE_MACHINE_OK;
    }
    return rc;
}
