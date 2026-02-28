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
 * \file device_runner.cpp
 * \brief
 */
#ifdef BUILD_WITH_CANN
#include "machine/runtime/device_runner.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <limits.h>
#include "securec.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/load_aicpu_op.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_common.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "runtime/mem.h"
#include "machine/utils/device_switch.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "toolchain/prof_api.h"
#include "prof_common.h"
#include "load_aicpu_op.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "machine/platform/platform_manager.h"
#include "machine/runtime/device_error_tracking.h"
#include "nlohmann/json.hpp"
#include "dump_device_perf.h"
#include "machine/host/perf_analysis.h"
#include "log_types.h"
#include "tilefwk/pypto_fwk_log.h"

using json = nlohmann::json;
extern char _binary_kernel_o_start[];
extern char _binary_kernel_o_end[];

constexpr int32_t AICORE_ADDR_TYPE = 2; // nocache Addr type for aicore/aicpu map
constexpr int32_t PMU_ADDR_TYPE = 3;    // nGnRnE Addr type for Geting pmuInfo
constexpr int32_t PATH_LENGTH = 64;
constexpr uint32_t LOG_BUF_SIZE = 64 * 1024;
bool g_IsNullLaunched = false;
constexpr uint32_t MIX_BLOCK_DIM = 2;
constexpr uint32_t HIGHT_BIT = 16;

constexpr uint32_t SUB_CORE = 3;
constexpr uint32_t AIV_PER_AICORE = 2;

extern "C"{
    __attribute__((weak)) int AdxDataDumpServerUnInit();
    __attribute__((weak)) int dlog_getlevel(int32_t moduled, int32_t *enableEvent);
}
namespace npu::tile_fwk {

namespace {

void ExchangeCaputerMode(const bool &isCapture) {
    if (isCapture) {
        aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        aclmdlRICaptureThreadExchangeMode(&mode);
        MACHINE_LOGI("captureMode is: %d", mode);
    }
}

void SyncStreams(rtStream_t aicpuStream, rtStream_t aicoreStream, bool useSyncFlag) {
    aclrtEvent event;
    int rc;

    if (useSyncFlag) {
        rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    } else {
        rc = aclrtCreateEvent(&event);
    }
    
    if (rc < 0) {
        MACHINE_LOGI("CreateEvent failed rc=%d, useSyncFlag=%d", rc, useSyncFlag);
    }

    rc = aclrtRecordEvent(event, aicpuStream);
    if (rc < 0) {
        MACHINE_LOGI("RecordEvent failed rc=%d", rc);
    }

    rc = aclrtStreamWaitEvent(aicoreStream, event);
    if (rc < 0) {
        MACHINE_LOGI("StreamWaitEvent failed rc=%d", rc);
    }
}
}

DeviceRunner &DeviceRunner::Get() {
    static DeviceRunner runner;
    std::call_once(runner.once_, [&]() { runner.Init(); });
    return runner;
}

HostProf& DeviceRunner::GetHostProfInstance() {
    return hostProf_;
}

void *DeviceRunner::DevAlloc(int size) {
    uint8_t *devPtr = nullptr;
    machine::GetRA()->AllocDevAddr(&devPtr, size);
    int rc = rtMemset(devPtr, size, 0, size);
    if (rc != 0) {
        machine::GetRA()->FreeTensor(devPtr);
        MACHINE_LOGE("rtMemset failed size=%d rc=%d\n", size, rc);
        return nullptr;
    }
    return devPtr;
}

void DeviceRunner::GetModuleLogLevel(DeviceArgs &args) {
    int logLevel= -1;
    if (dlog_getlevel != nullptr) {
        int32_t enableLog = -1;
        logLevel = dlog_getlevel(PYPTO, &enableLog);
    }
    DevDfxArgs devDfxArg;
    devDfxArg.logLevel = logLevel;
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        devDfxArg.isOpenSwim = PRO_LEVEL2;
    }
    MACHINE_LOGI("Get PYPTO log level is: %d, openSwimLevel: %d", logLevel, devDfxArg.isOpenSwim);
    auto size = sizeof(DevDfxArgs);
    args.devDfxArgAddr = args_.devDfxArgAddr;
    auto ret = rtMemcpy(reinterpret_cast<void *>(args.devDfxArgAddr), size, &devDfxArg, size, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        MACHINE_LOGW("Rt mem cpy failed, so couldn't get device log");
    }
}

void DeviceRunner::InitDynamicArgs(DeviceArgs &args) {
    devArgs_ = reinterpret_cast<DeviceArgs *>(DevAlloc(sizeof(DeviceArgs)));
    rtMemcpy(reinterpret_cast<void *>(devArgs_), sizeof(DeviceArgs), &args, sizeof(DeviceArgs),
        RT_MEMCPY_HOST_TO_DEVICE);

    for (uint64_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        perfData_.push_back(DevAlloc(MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics)));
    }
}

void DeviceRunner::ResetPerData() {
    auto size = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
    for (uint64_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        int rc = rtMemset(perfData_[i], size, 0, size);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %lu, rtMemSet failed, rc: %d", i, rc);
        }
    }
}

void DeviceRunner::InitMetaData(DeviceArgs &devArgs) {
    auto shmAddr = args_.runtimeDataRingBufferAddr;
    devArgs.runtimeDataRingBufferAddr = shmAddr;
    devArgs.sharedBuffer = args_.sharedBuffer;
    devArgs.coreRegAddr = args_.coreRegAddr;
    devArgs.nrAic = args_.nrAic;
    devArgs.nrAiv = args_.nrAiv;
    devArgs.corePmuRegAddr = args_.corePmuRegAddr;
    devArgs.corePmuAddr = args_.corePmuAddr;
    devArgs.taskWastTime = args_.taskWastTime;
    devArgs.pmuEventAddr = args_.pmuEventAddr;
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        args_.aicpuPerfAddr = npu::tile_fwk::dynamic::PtrToValue(DevAlloc(sizeof(MetricPerf)));
        if (args_.aicpuPerfAddr == 0) {
            MACHINE_LOGW("Aicpu per addr malloc failed");
        }
    }
    devArgs.aicpuPerfAddr = args_.aicpuPerfAddr;
    GetModuleLogLevel(devArgs);
}

int DeviceRunner::InitDeviceArgsCore(DeviceArgs &args, const std::vector<int64_t> &regs, const std::vector<int64_t> &regsPmu) {
    uint32_t totalCoreCount = regs.size();
    uint32_t aicCount = totalCoreCount / SUB_CORE;
    uint32_t aivCount = aicCount * AIV_PER_AICORE;
    args.nrAic = aicCount;
    args.nrAiv = aivCount;
    blockDim_ = dynamic::GetCfgBlockdim();
    args.nrValidAic = blockDim_;
    args.nrAicpu = aicpuNum_;
    int nrCore = regs.size() + AICPU_NUM_OF_RUN_AICPU_TASKS;
    args.sharedBuffer = reinterpret_cast<uint64_t>(DevAlloc(nrCore * SHARED_BUFFER_SIZE));
    args.coreRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * PMU_BUFFER_SIZE));
    args.taskWastTime = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(sizeof(uint64_t))));
    size_t shmSize = sizeof(dynamic::RuntimeDataRingBufferHead) + dynamic::DEVICE_SHM_SIZE + dynamic::DEVICE_TASK_QUEUE_SIZE * aicpuNum_;
    uint64_t shmAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(shmSize)));
    args.runtimeDataRingBufferAddr = shmAddr;
    PmuCommon::InitPmuEventType(args.archInfo, pmuEvtType_);
    args.pmuEventAddr = reinterpret_cast<uint64_t>(DevAlloc(pmuEvtType_.size() * sizeof(int64_t)));
    args.devDfxArgAddr = reinterpret_cast<uint64_t>(DevAlloc(sizeof(DevDfxArgs)));

    if (args.devDfxArgAddr == 0) {
        ALOG_ERROR_F("Alloc devDfx info failed");
        return -1;
    }

    if (args.sharedBuffer == 0 || args.coreRegAddr == 0 || args.corePmuAddr == 0 || args.corePmuRegAddr == 0) {
        return -1;
    }
    size_t size = nrCore * sizeof(uint64_t);
    rtMemcpy(reinterpret_cast<void *>(args.coreRegAddr), size, regs.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
    rtMemcpy(reinterpret_cast<void *>(args.corePmuRegAddr), size, regsPmu.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
    size = pmuEvtType_.size() * sizeof(int64_t);
    rtMemcpy(reinterpret_cast<void *>(args.pmuEventAddr), size, pmuEvtType_.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
    MACHINE_LOGI("aic %u aiv %u  blockDim_ %d sharedBuffer %lx coreRegAddr %lx corePmuRegAddr %lx\n", args.nrAic,
        args.nrAiv, blockDim_, args.sharedBuffer, args.coreRegAddr, args.corePmuRegAddr);
    InitDynamicArgs(args);

    return 0;
}

int DeviceRunner::InitDeviceArgs(DeviceArgs &args) {
    hostProf_.RegHostProf();

    addressMappingTable_[ArchInfo::DAV_2201] = [&args](std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu) {
        std::vector<int64_t> aiv;
        std::vector<int64_t> aic;
        std::vector<int64_t> aivPmu;
        std::vector<int64_t> aicPmu;
        if (machine::GetRA()->GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
            return -1;
        }
        if (machine::GetRA()->GetAicoreRegInfo(aicPmu, aivPmu, ADDR_MAP_TYPE_REG_AIC_PMU_CTRL) != 0) {
            return 0;
        }
        regs.insert(regs.end(), aic.begin(), aic.end());
        regs.insert(regs.end(), aiv.begin(), aiv.end());
        regsPmu.insert(regsPmu.end(), aicPmu.begin(), aicPmu.end());
        regsPmu.insert(regsPmu.end(), aivPmu.begin(), aivPmu.end());
        return 0;
    };

    addressMappingTable_[ArchInfo::DAV_3510] = [](std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu) {
        return machine::GetRA()->GetAicoreRegInfoForDAV3510(regs, regsPmu);
    };

    memset_s(&args, sizeof(args), 0, sizeof(args));
    std::vector<int64_t> regs;
    std::vector<int64_t> regsPmu;

    args.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    if (args.archInfo == ArchInfo::DAV_3510) {
        aicpuNum_ = npu::tile_fwk::dynamic::DEVICE_MAX_AICPU_NUM;
    }
    int cpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum() - 1);
    aicpuNum_ = aicpuNum_ < cpuNum ? aicpuNum_ : cpuNum;
    auto it = addressMappingTable_.find(args.archInfo);
    if (it != addressMappingTable_.end()){
        if (it->second(regs, regsPmu) != 0) {
            return -1;
        }
    }
    InitAiCpuSoBin(args);
    return InitDeviceArgsCore(args, regs, regsPmu);
}

uint64_t DeviceRunner::GetTasksTime() const {
    uint64_t buffer;
    int rc = rtMemcpy(reinterpret_cast<void *>(&buffer), sizeof(uint64_t),
                      reinterpret_cast<void *>(static_cast<uintptr_t>(args_.taskWastTime)),
                      sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST);
    (void)rc;
    return buffer;
}


bool DeviceRunner::GetValidGetPgMask() const {
    return machine::GetRA()->GetValidGetPgMask();
}

int DeviceRunner::Run(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, uint64_t taskData, int taskType) {
    int rc;

    rc = RunAsync(aicpuStream, aicoreStream, taskId, taskData, taskType);
    if (rc < 0) {
        return rc;
    }
    rc = rtStreamSynchronize(aicoreStream);
    if (rc != 0) {
        MACHINE_LOGI("aicore stream sync failed");
    }
    rc = rtStreamSynchronize(aicpuStream);
    if (rc != 0) {
        MACHINE_LOGI("aicpu stream sync failed");
    }
    ASSERT(rc == 0);
    if (IsPtoDataDumpEnabled()) {
        MACHINE_LOGD("DataDumpServerInit is called \n");
        rc = AdxDataDumpServerUnInit();
        if (rc != 0) {
            MACHINE_LOGE("AdxDataDumpServerUnInit is failed %d \n", rc);
        }
    }
    uint64_t taskWastTime = GetTasksTime();
    MACHINE_LOGI("task wast time %lu\n", taskWastTime);
    return rc;
}

int DeviceRunner::LaunchAiCore(rtStream_t aicoreStream, int taskType) {
    struct Args {
        int64_t *ctrlFlowCache = nullptr;
        int64_t *inputs = nullptr;
        int64_t *outputs = nullptr;
        int64_t *workspace = nullptr;
        int64_t *tilingData = nullptr;
        DeviceArgs *cfgdata = nullptr;
        int64_t *logBuf = nullptr;
    };

    auto localArgs = args_;
    auto size = sizeof(args_);
    localArgs.taskType = taskType;
    if (devArgs_ == nullptr) {
        MACHINE_LOGE("devArgs_ is null..");
        return -1;
    }
    int rc = rtMemcpy(devArgs_, size, &localArgs, size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        MACHINE_LOGE("rtmemcpy failed %p rc %d\n", devArgs_, rc);
        return rc;
    }
    Args args{nullptr, nullptr, nullptr, nullptr, nullptr, devArgs_};
    rtArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    return rtKernelLaunchWithHandleV2(binHdl_, 0, blockDim_, &rtArgs, nullptr, aicoreStream, nullptr);
}

int DeviceRunner::LaunchAiCpu(
    const rtStream_t aicpuStream, const uint64_t taskId, const uint64_t taskData, int taskType) const {
    struct Args {
        DeviceArgs devArgs;
        const char kernelName[32] = {"StaticTileFwkKernelServer"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.devArgs = args_;
    args.devArgs.taskType = taskType;
    args.devArgs.taskId = taskId;
    args.devArgs.taskData = taskData;

    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_AICPU", aicpuNum_, &rtArgs, nullptr, aicpuStream, 0);
}

void DeviceRunner::AllocDfxMetricMemory() {
    for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        KernelArgs kernelArgs;
        memset_s(&kernelArgs, sizeof(kernelArgs), 0, sizeof(kernelArgs));
        kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] =
            reinterpret_cast<int64_t>(DevAlloc(MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics)));
        rtMemcpy((reinterpret_cast<uint8_t *>(args_.sharedBuffer)) + i * SHARED_BUFFER_SIZE, sizeof(kernelArgs),
            reinterpret_cast<uint8_t *>(&kernelArgs), sizeof(kernelArgs), RT_MEMCPY_HOST_TO_DEVICE);
        MACHINE_LOGI("aicore %u , dfxaddr 0x%ld \n", i, kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    }
}

int DeviceRunner::RunAsync(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, uint64_t taskData, int taskType) {
    int rc;
    SyncStreams(aicpuStream, aicoreStream, false);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
    AllocDfxMetricMemory();
#else
    int size = (args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS) * SHARED_BUFFER_SIZE;
    rtMemset(reinterpret_cast<void *>(args_.sharedBuffer), size, 0, size);
#endif

    std::lock_guard<FileLock> lock(lock_);
    rc = LaunchAiCore(aicoreStream, DEVICE_TASK_TYPE_STATIC);
    if (rc < 0) {
        MACHINE_LOGI("launch aicpu failed %d\n", rc);
        return rc;
    }

    InitAiCpuSoBin(args_);
    rc = LaunchAiCpu(aicpuStream, taskId, taskData, taskType);
    if (rc < 0) {
        MACHINE_LOGI("launch aicpu failed %d\n", rc);
        return rc;
    }

    return 0;
}

void DeviceRunner::Dump() {
    MACHINE_LOGI("======== aicore status ========");

    int coreNum = args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS;
    uint64_t size = coreNum * SHARED_BUFFER_SIZE;
    std::vector<uint64_t> buffer(size / sizeof(uint64_t));
    int rc =
        rtMemcpy(buffer.data(), size, reinterpret_cast<void *>(args_.sharedBuffer), size, RT_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) {
        MACHINE_LOGI("rtmemcpy failed");
        return;
    }

    uint64_t buffAddr = reinterpret_cast<uint64_t>(buffer.data());
    for (int i = 0; i < coreNum; i++) {
        KernelArgs *arg = reinterpret_cast<KernelArgs *>(buffAddr + i * SHARED_BUFFER_SIZE);
        MACHINE_LOGI("aicore %d hello status %ld", i, arg->shakeBuffer[0]);
        MACHINE_LOGI("last_taskId %ld", arg->shakeBuffer[1]);
        MACHINE_LOGI("task status %ld", arg->shakeBuffer[2]);

        for (int k = 0; k < static_cast<int>(sizeof(arg->taskStat) / sizeof(TaskStat)); k++) {
            MACHINE_LOGI("task rsp index %d: taskId %d, subGraphID %d execStart %ld execEnd %ld\n", k,
                arg->taskStat[k].taskId, arg->taskStat[k].subGraphId, arg->taskStat[k].execStart,
                arg->taskStat[k].execEnd);
        }
    }
}

/**************************** DynamicFunction *****************************/
void DeviceRunner::DumpAiCoreExecutionTimeData() {
    npu::tile_fwk::dynamic::DumpAicoreTaskExectInfo(args_, perfData_);
}

void DeviceRunner::DumpAiCorePmuData() {
    MACHINE_LOGI("TODO: DumpAiCorePmuData");
}

void DeviceRunner::SynchronizeDeviceToHostProfData() {
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        DumpAiCoreExecutionTimeData();
    }
}

int DeviceRunner::DynamicLaunchSynchronize(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream) {
    int rcAicore = rtStreamSynchronize(aicoreStream);
    int rcAicpu = rtStreamSynchronize(aicpuStream);
    int rcCtrl = 0;
    if (ctrlStream != nullptr) {
        rcCtrl = rtStreamSynchronize(aicpuStream);
    }
    if (IsPtoDataDumpEnabled()) {
        MACHINE_LOGD("DataDumpServerInit is called \n");
        AdxDataDumpServerUnInit();
    }
    if (rcAicore != 0 || rcAicpu != 0 || rcCtrl != 0) {
        MACHINE_LOGW("sync stream failed aicpu:%d aicore:%d ctrl cpu:%d", rcAicpu, rcAicore, rcCtrl);
    }
    return rcAicore + rcAicpu + rcCtrl;
}

int DeviceRunner::launchDynamicAiCore(rtStream_t aicoreStream, DeviceKernelArgs *kernelArgs) {
    rtArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void *> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;
    return rtKernelLaunchWithHandleV2(binHdl_, tilingKey, blockDim_, &rtArgs, nullptr, aicoreStream, &cfg);
}

int DeviceRunner::launchDynamicAiCpu(rtStream_t aicpuStream, DeviceKernelArgs *kArgs) {
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, aicpuNum_, "PyptoRun");
#endif
    // use inputs/outputs store argsaddr/argsSize(aicpu task info + tensorInfo size)
    auto args = reinterpret_cast<dynamic::AiCpuArgs*>(kArgs->inputs);
    uint64_t argsSize = reinterpret_cast<uint64_t>(kArgs->outputs);
    kArgs->inputs = nullptr;
    args->kArgs = *kArgs;
    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = argsSize;
    rtArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
    rtArgs.hostInputInfoNum = 1;
    rtHostInputInfo_t hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
    rtArgs.hostInputInfoPtr = &hostInputInfo;
    MACHINE_LOGI("Copy flow addrOffset %u argsSize %u", hostInputInfo.addrOffset, hostInputInfo.dataOffset);
    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum_, &rtArgs, nullptr, aicpuStream, 0);
}

void DeviceRunner::InitAiCpuSoBin(DeviceArgs &devArgs) {
    std::vector<char> buffer;
    std::string fileName = GetCurrentSharedLibPath() + "/libtilefwk_backend_server.so";
    if (!ReadBytesFromFile(fileName, buffer)) {
        MACHINE_LOGE("Read bin form tilefwk_backend_server.so failed, please check the so[%s]", fileName.c_str());
        return;
    }
    size_t aicpuDataLength = buffer.size();
    auto dAicpuData = DevAlloc(aicpuDataLength);
    rtMemcpy(dAicpuData, aicpuDataLength, reinterpret_cast<void *>(buffer.data()),
             aicpuDataLength, RT_MEMCPY_HOST_TO_DEVICE);
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
    devArgs.aicpuSoLen = buffer.size();
    devArgs.deviceId = GetLogDeviceId();
    if (IsPtoDataDumpEnabled()) {
       devArgs.hostPid = getpid();
    }
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitAicpuSo);
}

int DeviceRunner::InitAicpuServer() {
    auto aicpuStream = machine::GetRA()->GetScheStream();
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, 1, "PyptoInit");
#endif
    struct Args {
        DeviceKernelArgs kArgs;
        const char kernelName[32] = {"DynTileFwkKernelServerInit"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.kArgs.cfgdata = (int64_t *)devArgs_;

    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    int ret = rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
        "AST_DYN_AICPU", 1, &rtArgs, nullptr, aicpuStream, 0);
    if (ret != RT_ERROR_NONE) {
        MACHINE_LOGE("Aicpu server init failed %d", ret);
        return ret;
    }
    // for triple stream schedule, must wait aicpu server init done
    return rtStreamSynchronize(aicpuStream);
}

void DeviceRunner::SetDebugEnable() {
    for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv; i++) {
        rtMemset(perfData_[i], sizeof(Metrics), 0, sizeof(Metrics));
        rtMemcpy((reinterpret_cast<uint8_t *>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) + i * SHARED_BUFFER_SIZE,
            sizeof(uint64_t),
            reinterpret_cast<uint8_t *>(&perfData_[i]),
            sizeof(uint64_t),
            RT_MEMCPY_HOST_TO_DEVICE);
    }
}

int DeviceRunner::RunPrepare() {
    int ret = 0;
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL || ENABLE_PERF_TRACE == 1 || PMU_COLLECT == 1) {
        for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
           auto preCoreShareadBufferAddr = (reinterpret_cast<uint8_t *>(args_.sharedBuffer +
                                            sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) + i * SHARED_BUFFER_SIZE;
            ret = rtMemcpy(preCoreShareadBufferAddr,
                       sizeof(uint64_t),
                       reinterpret_cast<uint8_t *>(&perfData_[i]),
                       sizeof(uint64_t),
                       RT_MEMCPY_HOST_TO_DEVICE);
        }
    }
    return ret;
}

int DeviceRunner::RunPreSync(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    aclrtEvent event;
    int rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    if (rc < 0) {
        MACHINE_LOGE("aclrtCreateEvent failed %d\n", rc);
        return rc;
    }

    rc = aclrtRecordEvent(event, aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE("aclrtRecordEvent failed %d\n", rc);
        return rc;
    }

    rc = aclrtStreamWaitEvent(aicpuStream, event);
    if (rc < 0) {
        MACHINE_LOGE("aclrtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

int DeviceRunner::RunPost(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    SyncStreams(aicpuStream, aicoreStream, true);
    return 0;
}

int DeviceRunner::DynamicKernelLaunch(rtStream_t aicpuStream, rtStream_t aicoreStream, DeviceKernelArgs *kernelArgs, int blockdim) {
    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuInit);
    uint64_t startTime = MsprofSysCycleTime();
    auto rc = launchDynamicAiCpu(aicpuStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, aicpuNum_, MSPROF_GE_TASK_TYPE_AI_CPU);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuRun);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_MIX_AIC, true);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAIcore);
    return rc;
}

int DeviceRunner::DynamicSeparateLaunch(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream,
    DeviceKernelArgs *kernelArgs, int blockdim) {
    LoadAicpuOp::GetInstance().CustomAiCpuSoLoad();
    std::string initKernel =  OpInfoManager::GetInstance().GetOpFuncName() + "Init";
    std::string mainKernel =  OpInfoManager::GetInstance().GetOpFuncName() + "Run";
    uint64_t startTime = MsprofSysCycleTime();
    int rc = LoadAicpuOp::GetInstance().LaunchCustomOp(ctrlStream, kernelArgs, initKernel);
    if (rc < 0) {
        MACHINE_LOGE("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_AI_CPU, true);

    rc = RunPreSync(ctrlStream, aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE("prepare failed %d\n", rc);
        return rc;
    }

    startTime = MsprofSysCycleTime();
    rc = LoadAicpuOp::GetInstance().LaunchCustomOp(ctrlStream, kernelArgs, mainKernel);
    if (rc < 0) {
        MACHINE_LOGE("launch custom aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_AI_CPU, true);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCpu(aicpuStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, aicpuNum_, MSPROF_GE_TASK_TYPE_AI_CPU);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE("launch aicore failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_MIX_AIC, true);

    rc = RunPost(ctrlStream, aicoreStream);
    return rc;
}

int DeviceRunner::DynamicLaunch(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream, [[maybe_unused]] int64_t taskId,
    DeviceKernelArgs *kernelArgs, int blockdim, int launchAicpuNum) {
    #ifdef BUILD_WITH_NEW_CANN
    if (!g_IsNullLaunched) {
        auto ret = LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, 1, "PyptoNull");
        if (ret != 0) {
            MACHINE_LOGE("launch built null failed");
            return ret;
        }
        g_IsNullLaunched = true;
    }
    #endif
    int rc = RunPrepare();
    if (rc < 0) {
        MACHINE_LOGE("Prepare failed.");
        return rc;
    }
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitRunPrepare);

    lastLaunchToSubMachineConfig_ = kernelArgs->toSubMachineConfig;
    blockDim_ = blockdim;
    aicpuNum_ = launchAicpuNum;
    // for dump perfInfo update device args
    args_.nrValidAic = blockdim;
    args_.nrAicpu = launchAicpuNum;
    args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(blockDim_, aicpuNum_, args_.archInfo);
    ExchangeCaputerMode(isCapture_);
    if (ctrlStream == nullptr) {
        return DynamicKernelLaunch(aicpuStream, aicoreStream, kernelArgs, blockDim_);
    } else {
        return DynamicSeparateLaunch(aicpuStream, ctrlStream, aicoreStream, kernelArgs, blockDim_);
    }
}

void DeviceRunner::ReportHostProfInfo(uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore) {
    if (hostProf_.GetProfType() == PROF_COMMANDHANDLE_TYPE_START) {
        uint64_t endTime = MsprofSysCycleTime();
        if (isCore) {
            uint32_t mixBlockDim = MIX_BLOCK_DIM;
            blockDim = (mixBlockDim << HIGHT_BIT) | blockDim;
            hostProf_.HostProfReportContextInfo(endTime);
        }
        if ((hostProf_.GetProfSwitch() & PROF_TASK_TIME_L1_MASK) != 0) {
            hostProf_.HostProfReportNodeInfo(endTime, blockDim, taskType);
        }
        endTime = MsprofSysCycleTime();
        hostProf_.HostProfReportApi(startTime, endTime);
    }
}

int DeviceRunner::DynamicRun(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream, int64_t taskId, DeviceKernelArgs *kernelArgs, int blockdim, int launchAicpuNum) {
    int rc = DynamicLaunch(aicpuStream, ctrlStream, aicoreStream, taskId, kernelArgs, blockdim, launchAicpuNum);
    if (rc < 0) {
        return rc;
    }
    if (isCapture_) {
        return 0;
    }
    return DynamicLaunchSynchronize(aicpuStream, ctrlStream, aicoreStream);
}

/**************************** DynamicFunction *****************************/
std::vector<uint8_t> g_binBuf;

void DeviceRunner::SetBinData(const std::vector<uint8_t> &binBuf) {
  g_binBuf = binBuf;
  MACHINE_LOGD("Set kernel size:%zu", g_binBuf.size());
  return;
}

int DeviceRunner::RegisterKernelBin(void **hdl, std::vector<uint8_t> *funcBinBuf) {
    if (*hdl) {
        binHdl_ = *hdl;
        MACHINE_LOGD("RegisterKernelBin reuse cache.");
        return 0;
    }
    void *bin = nullptr;
    size_t binSize = 0;
    std::vector<uint8_t> *binBuf = (funcBinBuf == nullptr) ? &g_binBuf : funcBinBuf;
    if (binBuf->size() != 0) {
        bin = binBuf->data();
        binSize = binBuf->size();
        MACHINE_LOGD("Reg dynamic bin size %zu.", binSize);
    } else {
        bin = _binary_kernel_o_start;
        binSize = _binary_kernel_o_end - _binary_kernel_o_start;
        MACHINE_LOGD("Reg static bin size %zu.", binSize);
    }
    rtDevBinary_t binary{.magic = RT_DEV_BINARY_MAGIC_ELF, .version = 0, .data = bin, .length = binSize};
    int rc = rtRegisterAllKernel(&binary, hdl);
    if (rc != 0) {
        MACHINE_LOGE("RegisterKernelBin failed\n");
    }
    binHdl_ = *hdl;
    MACHINE_LOGD("finish RegisterKernelBin.");
    return rc;
}

int DeviceRunner::Init(void) {
    char path[PATH_LENGTH];
    sprintf_s(path, PATH_LENGTH, "/tmp/aicpu%d.lock", devId_);
    lock_.Init(path);
    std::string builtInOpPath = config::LogTopFolder() + "/built_in";
    CreateMultiLevelDir(builtInOpPath);
    LoadAicpuOp::GetInstance().GenBuiltInOpInfo(builtInOpPath);
    if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle() != 0) {
        MACHINE_LOGE("Get builtInOp Funchandle failed\n");
        return -1;
    }

    InitializeErrorCallback();

    if (InitDeviceArgs(args_) != 0) {
        MACHINE_LOGE("prepareArgs failed\n");
        return -1;
    }
    if (RegisterKernelBin(&binHdl_) != 0) {
        MACHINE_LOGE("RegisterKernelBin failed\n");
        return -1;
    }
    InitAicpuServer();
    return 0;
}
} // namespace npu::tile_fwk

#else // stub

#include "machine/runtime/device_runner.h"

namespace npu::tile_fwk {
DeviceRunner &DeviceRunner::Get() {
    static DeviceRunner runner;
    return runner;
}
void DeviceRunner::InitMetaData(DeviceArgs &devArgs) {
    (void)devArgs;
}
bool DeviceRunner::GetValidGetPgMask() const {
    return true;
}
}

#endif // BUILD_WITH_CANN
