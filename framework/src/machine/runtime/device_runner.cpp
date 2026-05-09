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

#include "machine/runtime/device_runner.h"
#include <cstdint>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <limits.h>
#include "securec.h"
#include "machine/runtime/runtime_agent.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/load_aicpu_op.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_common.h"
#include "interface/utils/file_utils.h"
#include "machine/utils/device_switch.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "load_aicpu_op.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "machine/platform/platform_manager.h"
#include "machine/runtime/device_error_tracking.h"
#include "nlohmann/json.hpp"
#include "dump_device_perf.h"
#include "machine/host/perf_analysis.h"
#include "tilefwk/pypto_fwk_log.h"
#include "interface/machine/host/host_machine.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"

using json = nlohmann::json;

constexpr int32_t AICORE_ADDR_TYPE = 2; // nocache Addr type for aicore/aicpu map
constexpr int32_t PMU_ADDR_TYPE = 3;    // nGnRnE Addr type for Geting pmuInfo
constexpr int32_t PATH_LENGTH = 64;
constexpr uint32_t LOG_BUF_SIZE = 64 * 1024;
bool g_IsNullLaunched = false;
bool g_is_machine_trace_addr_inited = false;
constexpr uint32_t MIX_BLOCK_DIM = 2;
constexpr uint32_t HIGHT_BIT = 16;

constexpr uint32_t SUB_CORE = 3;
constexpr uint32_t AIV_PER_AICORE = 2;

extern "C" {
__attribute__((weak)) int AdxDataDumpServerUnInit();
__attribute__((weak)) int dlog_getlevel(int32_t moduled, int32_t* enableEvent);
__attribute__((weak)) int drvDeviceGetPhyIdByIndex(uint32_t logicDevId, uint32_t* phyDevId);
}
namespace npu::tile_fwk {

namespace {

void ExchangeCaputerMode(const bool& isCapture)
{
    if (isCapture) {
        AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
        AclMdlRICaptureThreadExchangeMode(&mode);
        MACHINE_LOGI("captureMode is: %d", static_cast<int>(mode));
    }
}

void* MachinePerfTraceDevMalloc(int size)
{
    uint8_t* devPtr = nullptr;
    auto alignSize = MemSizeAlign(size);
    if (RuntimeMalloc(reinterpret_cast<void**>(&devPtr), alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0) != 0) {
        MACHINE_LOGW("Mem alloc failed");
        return nullptr;
    }
    return devPtr;
}

void SyncStreams(RtStream aicpuStream, RtStream aicoreStream, bool useSyncFlag)
{
    AclRtEvent event;
    int rc;

    if (useSyncFlag) {
        rc = AclRtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    } else {
        rc = AclRtCreateEvent(&event);
    }

    if (rc < 0) {
        MACHINE_LOGI("CreateEvent failed rc=%d, useSyncFlag=%d", rc, useSyncFlag);
    }

    rc = AclRtRecordEvent(event, aicpuStream);
    if (rc < 0) {
        MACHINE_LOGI("RecordEvent failed rc=%d", rc);
    }

    rc = AclRtStreamWaitEvent(aicoreStream, event);
    if (rc < 0) {
        MACHINE_LOGI("StreamWaitEvent failed rc=%d", rc);
    }
}
} // namespace

DeviceRunner& DeviceRunner::Get()
{
    static DeviceRunner runner;
    std::call_once(runner.once_, [&]() { ASSERT(DevCommonErr::INIT_FAILED, runner.Init() == 0); });
    return runner;
}

HostProf& DeviceRunner::GetHostProfInstance() { return hostProf_; }

void* DeviceRunner::DevAlloc(int size)
{
    uint8_t* devPtr = nullptr;
    machine::GetRA()->AllocDevAddr(&devPtr, size);
    int rc = RuntimeMemset(devPtr, size, 0, size);
    if (rc != 0) {
        machine::GetRA()->FreeTensor(devPtr);
        MACHINE_LOGE(RtErr::RT_MEMSET_FAILED, "RuntimeMemset failed size=%d rc=%d\n", size, rc);
        return nullptr;
    }
    return devPtr;
}

void DeviceRunner::GetModuleLogLevel(DeviceArgs& args)
{
    int logLevel = -1;
    if (dlog_getlevel != nullptr) {
        int32_t enableLog = -1;
        logLevel = dlog_getlevel(PYPTO, &enableLog);
    }
    DevDfxArgs devDfxArg;
    devDfxArg.logLevel = logLevel;
    uint32_t logicalDevId = GetLogDeviceId();
    uint32_t phyDevId = 0;
    if (drvDeviceGetPhyIdByIndex != nullptr) {
        drvDeviceGetPhyIdByIndex(logicalDevId, &phyDevId);
    } else {
        MACHINE_LOGW("Get device Local deviceId failed");
    }
    MACHINE_LOGI("Current device info: logical devId: %u, phyDevId: %u", logicalDevId, phyDevId);
    devDfxArg.deviceId = phyDevId;
    if (enableDumpMachinePerfTrace_) {
        devDfxArg.isOpenPerfTrace = 1;
    }
    MACHINE_LOGI("Get PYPTO dfxAddr: %lu log level is: %d, openPerTrace: %d, deviceId: %u\n",
        args_.devDfxArgAddr, logLevel, devDfxArg.isOpenPerfTrace, devDfxArg.deviceId);
    auto size = sizeof(DevDfxArgs);
    auto ret = RuntimeMemcpy(reinterpret_cast<void*>(args.devDfxArgAddr), size, &devDfxArg, size, RtMemcpyKind::HOST_TO_DEVICE);
    if (ret != 0) {
        MACHINE_LOGW("rtmemcpy failed, so couldn't get device log");
    }
}

void DeviceRunner::InitDynamicArgs(DeviceArgs& args)
{
    devArgs_ = reinterpret_cast<DeviceArgs*>(DevAlloc(sizeof(DeviceArgs)));
    RuntimeMemcpy(
        reinterpret_cast<void*>(devArgs_), sizeof(DeviceArgs), &args, sizeof(DeviceArgs), RtMemcpyKind::HOST_TO_DEVICE);

    for (uint64_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        perfData_.push_back(MachinePerfTraceDevMalloc(PERF_DATA_TOTAL_SIZE));
    }

    if (GetEnvVar("DUMP_DEVICE_PERF") == "true") {
        auto aicpuDevPtr = MachinePerfTraceDevMalloc(MAX_ROUND_NUM * sizeof(MetricPerf));
        if (aicpuDevPtr == 0) {
            MACHINE_LOGW("Aicpu per addr malloc failed");
            return;
        }
        args_.aicpuPerfAddr = npu::tile_fwk::dynamic::PtrToValue(aicpuDevPtr);
        enableDumpMachinePerfTrace_ = true;
    }
}

void DeviceRunner::ResetPerData()
{
    auto size = PERF_DATA_TOTAL_SIZE;
    for (uint64_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        int rc = RuntimeMemset(perfData_[i], size, 0, size);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %lu, rtMemSet failed, rc: %d", i, rc);
        }
    }
}

void DeviceRunner::InitMetaData(DeviceArgs& devArgs)
{
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
    devArgs.aicpuPerfAddr = args_.aicpuPerfAddr;
    devArgs.devDfxArgAddr = args_.devDfxArgAddr;
}

int DeviceRunner::InitDeviceArgsCore(
    DeviceArgs& args, const std::vector<int64_t>& regs, const std::vector<int64_t>& regsPmu)
{
    uint32_t totalCoreCount = regs.size();
    uint32_t aicCount = totalCoreCount / SUB_CORE;
    uint32_t aivCount = aicCount * AIV_PER_AICORE;
    args.nrAic = aicCount;
    args.nrAiv = aivCount;
    blockDim_ = dynamic::GetCfgBlockdim();
    args.nrValidAic = blockDim_;
    args.nrAicpu = aicpuNum_;
    args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(blockDim_, aicpuNum_, args.archInfo);
    int nrCore = regs.size() + AICPU_NUM_OF_RUN_AICPU_TASKS;
    args.sharedBuffer = reinterpret_cast<uint64_t>(DevAlloc(nrCore * SHARED_BUFFER_SIZE));
    args.coreRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * PMU_BUFFER_SIZE));
    args.taskWastTime = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(sizeof(uint64_t))));
    size_t shmSize = sizeof(dynamic::RuntimeDataRingBufferHead) + dynamic::DEVICE_SHM_SIZE +
                     dynamic::DEVICE_TASK_QUEUE_SIZE * aicpuNum_;
    uint64_t shmAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(shmSize)));
    args.runtimeDataRingBufferAddr = shmAddr;
    PmuCommon::InitPmuEventType(args.archInfo, pmuEvtType_);
    args.pmuEventAddr = reinterpret_cast<uint64_t>(DevAlloc(pmuEvtType_.size() * sizeof(int64_t)));
    args.devDfxArgAddr = reinterpret_cast<uint64_t>(DevAlloc(sizeof(DevDfxArgs)));

    if (args.devDfxArgAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc devDfx info failed");
        return -1;
    }

    if (args.sharedBuffer == 0 || args.coreRegAddr == 0 || args.corePmuAddr == 0 || args.corePmuRegAddr == 0) {
        return -1;
    }
    size_t size = nrCore * sizeof(uint64_t);
    RuntimeMemcpy(reinterpret_cast<void*>(args.coreRegAddr), size, regs.data(), size, RtMemcpyKind::HOST_TO_DEVICE);
    RuntimeMemcpy(reinterpret_cast<void*>(args.corePmuRegAddr), size, regsPmu.data(), size,
                  RtMemcpyKind::HOST_TO_DEVICE);
    size = pmuEvtType_.size() * sizeof(int64_t);
    RuntimeMemcpy(reinterpret_cast<void*>(args.pmuEventAddr), size, pmuEvtType_.data(), size,
                  RtMemcpyKind::HOST_TO_DEVICE);
    MACHINE_LOGI(
        "aic %u aiv %u  blockDim_ %d sharedBuffer %lx coreRegAddr %lx corePmuRegAddr %lx\n", args.nrAic, args.nrAiv,
        blockDim_, args.sharedBuffer, args.coreRegAddr, args.corePmuRegAddr);
    InitDynamicArgs(args);
    GetModuleLogLevel(args);
    return 0;
}

int DeviceRunner::InitDeviceArgs(DeviceArgs& args)
{
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
    int cpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum());
    args.maxAicpuNum = static_cast<uint32_t>(cpuNum);
    aicpuNum_ = aicpuNum_ < cpuNum ? aicpuNum_ : cpuNum;
    auto it = addressMappingTable_.find(args.archInfo);
    if (it != addressMappingTable_.end()) {
        if (it->second(regs, regsPmu) != 0) {
            return -1;
        }
    }
    InitAiCpuSoBin(args);
    return InitDeviceArgsCore(args, regs, regsPmu);
}

uint64_t DeviceRunner::GetTasksTime() const
{
    uint64_t buffer;
    int rc = RuntimeMemcpy(
        reinterpret_cast<void*>(&buffer), sizeof(uint64_t),
        reinterpret_cast<void*>(static_cast<uintptr_t>(args_.taskWastTime)), sizeof(uint64_t),
        RtMemcpyKind::DEVICE_TO_HOST);
    (void)rc;
    return buffer;
}

bool DeviceRunner::GetValidGetPgMask() const { return machine::GetRA()->GetValidGetPgMask(); }

void DeviceRunner::AllocDfxMetricMemory()
{
    for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        KernelArgs kernelArgs;
        memset_s(&kernelArgs, sizeof(kernelArgs), 0, sizeof(kernelArgs));
        kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] =
            reinterpret_cast<int64_t>(DevAlloc(PERF_DATA_TOTAL_SIZE));
        RuntimeMemcpy(
            (reinterpret_cast<uint8_t*>(args_.sharedBuffer)) + i * SHARED_BUFFER_SIZE, sizeof(kernelArgs),
            reinterpret_cast<uint8_t*>(&kernelArgs), sizeof(kernelArgs), RtMemcpyKind::HOST_TO_DEVICE);
        MACHINE_LOGI("aicore %u , dfxaddr 0x%ld \n", i, kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    }
}

/**************************** DynamicFunction *****************************/
void DeviceRunner::DumpAiCoreExecutionTimeData()
{
    // 多轮控核，nrValidAic和scheCpuNum需实时刷新，否则泳道图会出错
    args_.nrValidAic = dynamic::GetCfgBlockdim();
    args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args_.nrValidAic, aicpuNum_, args_.archInfo);
    npu::tile_fwk::dynamic::DumpAicoreTaskExectInfo(args_, perfData_);
}

void DeviceRunner::DumpAiCorePmuData() const { MACHINE_LOGI("TODO: DumpAiCorePmuData"); }

void DeviceRunner::SynchronizeDeviceToHostProfData()
{
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        DumpAiCoreExecutionTimeData();
    }
}

int DeviceRunner::DynamicLaunchSynchronize(RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream)
{
    int rcAicore = RuntimeStreamSynchronize(aicoreStream);
    int rcAicpu = RuntimeStreamSynchronize(aicpuStream);
    int rcCtrl = 0;
    if (ctrlStream != nullptr) {
        rcCtrl = RuntimeStreamSynchronize(aicpuStream);
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

int DeviceRunner::launchDynamicAiCore(RtStream aicoreStream, DeviceKernelArgs* kernelArgs) const
{
    RtArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void*> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    RtTaskCfgInfo cfg = {};
    cfg.schemMode = static_cast<uint8_t>(npu::tile_fwk::RtSchemModeType::BATCH);
    return RuntimeKernelLaunchWithHandleV2(binHdl_, tilingKey, blockDim_, &rtArgs, nullptr, aicoreStream, &cfg);
}

int DeviceRunner::launchDynamicAiCpu(RtStream aicpuStream, DeviceKernelArgs* kArgs) const
{
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, aicpuNum_, "PyptoRun");
#endif
    // use inputs/outputs store argsaddr/argsSize(aicpu task info + tensorInfo size)
    auto args = reinterpret_cast<dynamic::AiCpuArgs*>(kArgs->inputs);
    RtAicpuArgsEx rtArgs;
    uint64_t argsSize = reinterpret_cast<uint64_t>(kArgs->outputs);
    kArgs->inputs = nullptr;
    args->kArgs = *kArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = argsSize;
    rtArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
    rtArgs.hostInputInfoNum = 1;
    RtHostInputInfo hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
    rtArgs.hostInputInfoPtr = &hostInputInfo;
    rtArgs.timeout = dynamic::AICPU_EXECUTE_TIMEOUT;
    MACHINE_LOGI("Copy flow addrOffset %u argsSize %u", hostInputInfo.addrOffset, hostInputInfo.dataOffset);
    return RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", aicpuNum_, &rtArgs, nullptr,
        aicpuStream, RT_KERNEL_USE_SPECIAL_TIMEOUT);
}

void DeviceRunner::InitAiCpuSoBin(DeviceArgs& devArgs)
{
    std::vector<char> buffer;
    std::string fileName = GetCurrentSharedLibPath() + "/libtilefwk_backend_server.so";
    if (!ReadBytesFromFile(fileName, buffer)) {
        MACHINE_LOGE(
            DevCommonErr::FILE_ERROR, "Read bin form tilefwk_backend_server.so failed, please check the so[%s]",
            fileName.c_str());
        return;
    }
    size_t aicpuDataLength = buffer.size();
    auto dAicpuData = DevAlloc(aicpuDataLength);
    RuntimeMemcpy(dAicpuData, aicpuDataLength, reinterpret_cast<void*>(buffer.data()), aicpuDataLength,
                  RtMemcpyKind::HOST_TO_DEVICE);
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
    devArgs.aicpuSoLen = buffer.size();
    devArgs.deviceId = GetLogDeviceId();
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitAicpuSo);
}

int DeviceRunner::InitAicpuServer()
{
    auto aicpuStream = GetStreamContext().GetScheStream();
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, 1, "PyptoInit");
#endif
    struct Args {
        DeviceKernelArgs kArgs;
        const char kernelName[32] = {"DynTileFwkKernelServerInit"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.kArgs.cfgdata = (int64_t*)devArgs_;

    RtAicpuArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    int ret = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr,
        aicpuStream, 0);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_LAUNCH_FAILED, "Aicpu server init failed %d", ret);
        return ret;
    }
    // for triple stream schedule, must wait aicpu server init done
    return RuntimeStreamSynchronize(aicpuStream);
}

bool DeviceRunner::GetEnableDumpDevPref() const { return enableDumpMachinePerfTrace_; }

void DeviceRunner::ResetMetrics(const uint32_t& coreId)
{
    if (perfData_.empty()) {
        return;
    }
    if (enableDumpMachinePerfTrace_) {
        if (!g_is_machine_trace_addr_inited) {
            RuntimeMemset(perfData_[coreId], sizeof(Metrics), 0, sizeof(Metrics));
            g_is_machine_trace_addr_inited = true;
        }
    } else {
        RuntimeMemset(perfData_[coreId], sizeof(Metrics), 0, sizeof(Metrics));
    }
}

void DeviceRunner::SetDebugEnable()
{
    for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        ResetMetrics(i);
        RuntimeMemcpy(
            (reinterpret_cast<uint8_t*>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) +
                i * SHARED_BUFFER_SIZE,
            sizeof(uint64_t), reinterpret_cast<uint8_t*>(&perfData_[i]), sizeof(uint64_t),
            RtMemcpyKind::HOST_TO_DEVICE);
    }
    MACHINE_LOGD("Set debug enable aicore 0 devPtr: %p", perfData_[0]);
}

int DeviceRunner::RunPrepare() const
{
    int ret = 0;
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL || ENABLE_PERF_TRACE == 1 ||
        PMU_COLLECT == 1) {
        for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
            auto preCoreShareadBufferAddr =
                (reinterpret_cast<uint8_t*>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) +
                i * SHARED_BUFFER_SIZE;
            ret = RuntimeMemcpy(
                preCoreShareadBufferAddr, sizeof(uint64_t), reinterpret_cast<const uint8_t*>(&perfData_[i]),
                sizeof(uint64_t), RtMemcpyKind::HOST_TO_DEVICE);
        }
    }
    return ret;
}

int DeviceRunner::RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream)
{
    int rc = AclRtRecordEvent(event_, aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtRecordEvent failed %d\n", rc);
        return rc;
    }
    rc = AclRtStreamWaitEvent(scheStream, event_);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    rc = AclRtStreamWaitEvent(ctrlStream, event_);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

int DeviceRunner::RunPost(RtStream aicpuStream, RtStream aicoreStream)
{
    SyncStreams(aicpuStream, aicoreStream, true);
    return 0;
}

int DeviceRunner::DynamicKernelLaunch(
    RtStream aicpuStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockdim)
{
    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuInit);
    uint64_t startTime = MspfSysCycleTime();
    auto rc = launchDynamicAiCpu(aicpuStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicpuStream, startTime, aicpuNum_, MSPF_GE_TASK_TYPE_AI_CPU);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuRun);

    startTime = MspfSysCycleTime();
    rc = launchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicoreStream, startTime, blockdim, MSPF_GE_TASK_TYPE_MIX_AIC, true);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAIcore);
    return rc;
}

int DeviceRunner::DynamicTripleStreamLaunch(
    RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockdim)
{
    LoadAicpuOp::GetInstance().CustomAiCpuSoLoad();
    uint64_t startTime = MspfSysCycleTime();
    auto args = reinterpret_cast<dynamic::AiCpuArgs*>(kernelArgs->inputs);
    RtAicpuArgsEx rtArgs;
    uint64_t argsSize = reinterpret_cast<uint64_t>(kernelArgs->outputs);
    kernelArgs->inputs = nullptr;
    args->kArgs = *kernelArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = argsSize;
    rtArgs.hostInputInfoNum = 1;
    rtArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
    RtHostInputInfo hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
    rtArgs.hostInputInfoPtr = &hostInputInfo;
    MACHINE_LOGI("Copy flow addrOffset %u argsSize %u", hostInputInfo.addrOffset, hostInputInfo.dataOffset);
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    int rc = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr,
        (AclRtStream)ctrlStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch ctrl aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(ctrlStream, startTime, 1, MSPF_GE_TASK_TYPE_AI_CPU, false);

    startTime = MspfSysCycleTime();
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    rc = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", aicpuNum_, &rtArgs, nullptr,
        (AclRtStream)schedStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch sche aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(schedStream, startTime, aicpuNum_, MSPF_GE_TASK_TYPE_AI_CPU, false);

    startTime = MspfSysCycleTime();
    rc = launchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICORE_FAILED, "triple stream launch aicore failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicoreStream, startTime, blockdim, MSPF_GE_TASK_TYPE_MIX_AIC, true);

    rc = RunPost(ctrlStream, aicoreStream);
    return rc;
}

int DeviceRunner::DynamicLaunch(
    RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, [[maybe_unused]] int64_t taskId,
    DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum)
{
#ifdef BUILD_WITH_NEW_CANN
    if (!g_IsNullLaunched) {
        auto ret = LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, 1, "PyptoNull");
        if (ret != 0) {
            MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "launch built null failed");
            return ret;
        }
        g_IsNullLaunched = true;
    }
#endif
    int rc = RunPrepare();
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_PREPARE_FAILED, "Prepare failed.");
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
    }
    return DynamicTripleStreamLaunch(aicpuStream, ctrlStream, aicoreStream, kernelArgs, blockDim_);
}

void DeviceRunner::ReportHostProfInfo(
    RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore)
{
    if (hostProf_.GetProfType() == MSPF_COMMANDHANDLE_TYPE_START) {
        uint64_t endTime = MspfSysCycleTime();
        if (isCore) {
            uint32_t mixBlockDim = MIX_BLOCK_DIM;
            blockDim = (mixBlockDim << HIGHT_BIT) | blockDim;
            hostProf_.HostProfReportContextInfo(endTime);
        }
        if ((hostProf_.GetProfSwitch() & MSPF_TASK_TIME_L1_MASK) != 0) {
            hostProf_.HostProfReportNodeInfo(endTime, blockDim, taskType);
        }
        endTime = MspfSysCycleTime();
        hostProf_.HostProfReportApi(startTime, endTime);
    }
    if (taskType == MSPF_GE_TASK_TYPE_MIX_AIC) {
        hostProf_.HostProfReportCacheTaskInfo(stream, blockDim, taskType);
    }
}

int DeviceRunner::DynamicRun(
    RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, int64_t taskId,
    DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum)
{
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

void DeviceRunner::SetBinData(const std::vector<uint8_t>& binBuf)
{
    g_binBuf = binBuf;
    MACHINE_LOGD("Set kernel size:%zu", g_binBuf.size());
    return;
}

int DeviceRunner::RegisterKernelBin(void** hdl, std::vector<uint8_t>* funcBinBuf)
{
    if (*hdl) {
        binHdl_ = *hdl;
        MACHINE_LOGD("RegisterKernelBin reuse cache.");
        return 0;
    }
    void* bin = nullptr;
    size_t binSize = 0;
    std::vector<uint8_t>* binBuf = (funcBinBuf == nullptr) ? &g_binBuf : funcBinBuf;
    if (binBuf == nullptr || binBuf->size() == 0) {
        return 0;
    }
    if (binBuf->size() != 0) {
        bin = binBuf->data();
        binSize = binBuf->size();
        MACHINE_LOGD("Reg dynamic bin size %zu.", binSize);
    }
    RtDevBinary binary{.magic = RT_DEV_BINARY_MAGIC_ELF, .version = 0, .data = bin, .length = binSize};
    int rc = RuntimeRegisterAllKernel(&binary, hdl);
    if (rc != 0) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "RegisterKernelBin failed\n");
    }
    binHdl_ = *hdl;
    MACHINE_LOGD("finish RegisterKernelBin.");
    return rc;
}

int DeviceRunner::Init(void)
{
    char path[PATH_LENGTH];
    sprintf_s(path, PATH_LENGTH, "/tmp/aicpu%d.lock", devId_);
    lock_.Init(path);
    std::string builtInOpPath = config::LogTopFolder() + "/built_in";
    CreateMultiLevelDir(builtInOpPath);
    LoadAicpuOp::GetInstance().GenBuiltInOpInfo(builtInOpPath);
    if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle() != 0) {
        MACHINE_LOGE(DevCommonErr::GET_HANDLE_FAILED, "Get builtInOp Funchandle failed\n");
        return -1;
    }

    InitializeErrorCallback();

    if (AclRtCreateEventExWithFlag(&event_, ACL_EVENT_SYNC) < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtCreateEvent failed.");
        return -1;
    }
    if (InitDeviceArgs(args_) != 0) {
        MACHINE_LOGE(HostLauncherErr::PREPARE_ARGS_FAILED, "prepareArgs failed\n");
        return -1;
    }
    if (RegisterKernelBin(&binHdl_) != 0) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "RegisterKernelBin failed\n");
        return -1;
    }
    if (!(config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM
            && config::GetSimConfig(KEY_ACCURACY_LEVEL, 2) == 2)) {
        InitAicpuServer();
    }
    StartMachinePerfTraceDumpThread();
    return 0;
}

void DeviceRunner::StartMachinePerfTraceDumpThread()
{
    if (!enableDumpMachinePerfTrace_) {
        return;
    }
    if (dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(false);
    dumpThread_ = std::thread(&DeviceRunner::MachinePerfTraceDumpThread, this);
    MACHINE_LOGI("Dump thread started");
}

void DeviceRunner::StopMachinePerfTraceDumpThread()
{
    if (!dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(true);
    if (dumpThread_.joinable()) {
        dumpThread_.join();
    }
    MACHINE_LOGD("Dump thread stopped");

    if (args_.aicpuPerfAddr != 0) {
        void* ptr = npu::tile_fwk::dynamic::ValueToPtr(args_.aicpuPerfAddr);
        if (ptr != nullptr) {
            RuntimeFree(ptr);
            args_.aicpuPerfAddr = 0;
        }
    }
}

void DeviceRunner::MachinePerfTraceDumpThread()
{
    MACHINE_LOGD("Dump thread start to machine perf trace data");
    while (!dumpThreadStopFlag_.load()) {
        usleep(10000);
        npu::tile_fwk::dynamic::DumpDevTaskPerfData(args_, perfData_, false);
    }
    MACHINE_LOGD("Dump thread final dump");
    npu::tile_fwk::dynamic::DumpDevTaskPerfData(args_, perfData_, true);
}

DeviceRunner::~DeviceRunner()
{
    MACHINE_LOGD("Start to cleanup perfData");
    StopMachinePerfTraceDumpThread();
    for (size_t i = 0; i < perfData_.size(); i++) {
        if (perfData_[i] != nullptr) {
            RuntimeFree(perfData_[i]);
            perfData_[i] = nullptr;
        }
    }
    perfData_.clear();
}
} // namespace npu::tile_fwk
