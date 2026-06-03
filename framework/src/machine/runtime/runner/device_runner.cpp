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

#include "machine/runtime/runner/device_runner.h"
#include <cstdlib>
#include <mutex>
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"
#include "interface/utils/common.h"
#include "interface/utils/op_info_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/machine/host/host_machine.h"
#include "interface/utils/file_utils.h"
#include "machine/runtime/runner/runtime_agent.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/runner/load_aicpu_op.h"
#include "machine/runtime/runner/device_error_tracking.h"
#include "machine/runtime/runner/dump_device_perf.h"
#include "machine/runtime/runner/pmu_common.h"
#include "machine/host/perf_analysis.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_switch.h"
#include "securec.h"
#include "nlohmann/json.hpp"
#include "device_exception_dump.h"

using json = nlohmann::json;

extern "C" {
__attribute__((weak)) int AdxDataDumpServerUnInit();
__attribute__((weak)) int dlog_getlevel(int32_t moduled, int32_t* enableEvent);
__attribute__((weak)) int drvDeviceGetPhyIdByIndex(uint32_t logicDevId, uint32_t* phyDevId);
}
namespace npu::tile_fwk {
namespace {
constexpr uint32_t MIX_BLOCK_DIM = 2;
constexpr uint32_t HIGHT_BIT = 16;
constexpr uint32_t SUB_CORE = 3;
constexpr uint32_t AIV_PER_AICORE = 2;
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
void* DevAlloc(const uint64_t size)
{
    uint8_t* devPtr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&devPtr, size);
    if (devPtr == nullptr) {
        MACHINE_LOGE(RtErr::RT_MALLOC_FAILED, "Failed to alloc dev addr of size[%lu].", size);
        return nullptr;
    }
    if (RuntimeMemset(devPtr, size, 0, size) != RT_SUCCESS) {
        DevMemoryPool::Instance().FreeDevAddr(devPtr);
        MACHINE_LOGE(RtErr::RT_MEMSET_FAILED, "RuntimeMemset failed size=%lu.", size);
        return nullptr;
    }
    return devPtr;
}
void* CopyDataToDevice(const void *dataPtr, const uint64_t dataSize)
{
    void* devAddr = DevAlloc(dataSize);
    if (devAddr == nullptr) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Failed to alloc dev memory of size %lu", dataSize);
        return nullptr;
    }
    if (RuntimeMemcpy(devAddr, dataSize, dataPtr, dataSize, RtMemcpyKind::HOST_TO_DEVICE) != RT_SUCCESS) {
        DevMemoryPool::Instance().FreeDevAddr(devAddr);
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Failed to copy data to dev of size %lu", dataSize);
        return nullptr;
    }
    return devAddr;
}
}
DeviceRunner& DeviceRunner::Get()
{
    static DeviceRunner runner;
    std::call_once(runner.once_, [&]() { ASSERT(DevCommonErr::INIT_FAILED, runner.Init() == 0); });
    return runner;
}

void DeviceRunner::SetHostProfFunction(Function* function, const std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors)
{
    hostProf_.SetProfFunction(function, tensors);
}

uint32_t DeviceRunner::GetHostProfType() const
{
    return hostProf_.GetProfType();
}

void DeviceRunner::InitDevDfxArgs(const bool isPerfTrace, DevDfxArgs &devDfxArg)
{
    int logLevel = -1;
    if (dlog_getlevel != nullptr) {
        int32_t enableLog = -1;
        logLevel = dlog_getlevel(PYPTO, &enableLog);
    }
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
    if (isPerfTrace) {
        devDfxArg.isOpenPerfTrace = 1;
    }
    MACHINE_LOGI("Dfx info: log level is: %d, openPerTrace: %d, deviceId: %u",
                 logLevel, devDfxArg.isOpenPerfTrace, devDfxArg.deviceId);
}

void DeviceRunner::ResetPerData() const
{
    auto size = PERF_DATA_TOTAL_SIZE;
    for (uint64_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        int rc = RuntimeMemset(perfData_[i], size, 0, size);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %lu, rtMemSet failed, rc: %d", i, rc);
        }
    }
}

void DeviceRunner::InitMetaData(DeviceArgs& devArgs) const
{
    devArgs.runtimeDataRingBufferAddr = args_.runtimeDataRingBufferAddr;
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

int DeviceRunner::InitDeviceArgsCore(DeviceArgs& args)
{
    std::vector<int64_t> regs;
    std::vector<int64_t> regsPmu;
    GetAicoreRegs(args.archInfo, regs, regsPmu);

    args.nrAic = regs.size() / SUB_CORE;
    args.nrAiv = args.nrAic * AIV_PER_AICORE;
    uint64_t nrCore = regs.size() + AICPU_NUM_OF_RUN_AICPU_TASKS;
    args.sharedBuffer = reinterpret_cast<uint64_t>(DevAlloc(nrCore * SHARED_BUFFER_SIZE));
    args.corePmuAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * PMU_BUFFER_SIZE));
    if (args.sharedBuffer == 0 || args.corePmuAddr == 0) {
        return -1;
    }

    // core reg
    size_t regSize = regs.size() * sizeof(uint64_t);
    args.coreRegAddr = reinterpret_cast<uint64_t>(CopyDataToDevice(regs.data(), regSize));
    if (args.coreRegAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Fail to copy aicore reg data from host to device.");
        return -1;
    }

    // core reg pmu
    args.corePmuRegAddr = reinterpret_cast<uint64_t>(CopyDataToDevice(regsPmu.data(), regSize));
    if (args.corePmuRegAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Fail to copy aicore pmu reg data from host to device.");
        return -1;
    }
    MACHINE_LOGI("Dev args :aic %u aiv %u, sharedBuffer %lx coreRegAddr %lx corePmuRegAddr %lx",
        args.nrAic, args.nrAiv, args.sharedBuffer, args.coreRegAddr, args.corePmuRegAddr);

    args.taskWastTime = reinterpret_cast<uint64_t>(DevAlloc(sizeof(uint64_t)));
    size_t shmSize = sizeof(dynamic::RuntimeDataRingBufferHead) + dynamic::DEVICE_SHM_SIZE +
                     dynamic::DEVICE_TASK_QUEUE_SIZE * args.nrAicpu;
    args.runtimeDataRingBufferAddr = reinterpret_cast<uint64_t>(DevAlloc(shmSize));

    // pmu evt info
    std::vector<int64_t> pmuEvtType;
    PmuCommon::InitPmuEventType(args.archInfo, pmuEvtType);
    args.pmuEventAddr =
        reinterpret_cast<uint64_t>(CopyDataToDevice(pmuEvtType.data(), pmuEvtType.size() * sizeof(int64_t)));
    if (args.pmuEventAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Fail to copy pmu evt type from host to device.");
        return -1;
    }

    InitAicpuPerfAddr(args);
    // dfx info
    DevDfxArgs dfxArgs;
    InitDevDfxArgs(args.aicpuPerfAddr != 0, dfxArgs);
    args.devDfxArgAddr = reinterpret_cast<uint64_t>(CopyDataToDevice(&dfxArgs, sizeof(DevDfxArgs)));
    if (args.devDfxArgAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Fail to copy dfx info from host to device.");
        return -1;
    }
    return 0;
}

void DeviceRunner::InitAicpuPerfAddr(DeviceArgs& args)
{
    if (GetEnvVar("DUMP_DEVICE_PERF") == "true") {
        auto aicpuDevPtr = MachinePerfTraceDevMalloc(MAX_ROUND_NUM * sizeof(MetricPerf), TWO_MB_HUGE_PAGE_FLAGS);
        if (aicpuDevPtr == nullptr) {
            MACHINE_LOGW("Aicpu per addr malloc failed");
        } else {
            args.aicpuPerfAddr = npu::tile_fwk::dynamic::PtrToValue(aicpuDevPtr);
        }
    }
}

void DeviceRunner::GetAicoreRegs(const ArchInfo archInfo, std::vector<int64_t> &regs, std::vector<int64_t> &regsPmu)
{
    if (archInfo == ArchInfo::DAV_3510) {
        RuntimeAgent::GetAgent()->GetAicoreRegInfoForDAV3510(regs, regsPmu);
        return;
    }
    if (archInfo == ArchInfo::DAV_2201) {
        std::vector<int64_t> aiv;
        std::vector<int64_t> aic;
        if (RuntimeAgent::GetAgent()->GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
            return;
        }
        regs.insert(regs.end(), aic.begin(), aic.end());
        regs.insert(regs.end(), aiv.begin(), aiv.end());

        std::vector<int64_t> aivPmu;
        std::vector<int64_t> aicPmu;
        if (RuntimeAgent::GetAgent()->GetAicoreRegInfo(aicPmu, aivPmu, ADDR_MAP_TYPE_REG_AIC_PMU_CTRL) != 0) {
            return;
        }
        regsPmu.insert(regsPmu.end(), aicPmu.begin(), aicPmu.end());
        regsPmu.insert(regsPmu.end(), aivPmu.begin(), aivPmu.end());
    }
}

int DeviceRunner::InitDeviceArgs(DeviceArgs& args)
{
    memset_s(&args, sizeof(args), 0, sizeof(args));
    args.deviceId = GetLogDeviceId();
    args.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    uint32_t aicpuNum = args.archInfo == ArchInfo::DAV_3510 ? dynamic::DEVICE_MAX_AICPU_NUM : 5;
    args.maxAicpuNum = static_cast<uint32_t>(Platform::Instance().GetSoc().GetAICPUNum());
    args.nrValidAic = GetCfgBlockdim();
    args.nrAicpu = std::min(aicpuNum, args.maxAicpuNum);
    args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args.nrValidAic, args.nrAicpu, args.archInfo);
    MACHINE_LOGD("DevArgs: block dim[%u], aicpu num[%u], max aicpu num[%u], sche cpu num[%u].",
        args.nrValidAic, args.nrAicpu, args.maxAicpuNum, args.scheCpuNum);

    InitAiCpuSoBin(args);

    return InitDeviceArgsCore(args);
}

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
    args_.nrValidAic = GetCfgBlockdim();
    args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(GetBlockDim(), GetAicpuNum(), args_.archInfo);
    dynamic::DumpAicoreTaskExectInfo(args_, perfData_);
}

void DeviceRunner::DumpAiCorePmuData() { MACHINE_LOGI("TODO: DumpAiCorePmuData"); }

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

int DeviceRunner::LaunchDynamicAiCore(RtStream aicoreStream, DeviceKernelArgs* kernelArgs) const
{
    RtArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void*> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    RtTaskCfgInfo cfg = {};
    cfg.schemMode = static_cast<uint8_t>(npu::tile_fwk::RtSchemModeType::BATCH);
    return RuntimeKernelLaunchWithHandleV2(binHdl_, tilingKey, GetBlockDim(), &rtArgs, nullptr, aicoreStream, &cfg);
}

int DeviceRunner::LaunchDynamicAiCpu(RtStream aicpuStream, DeviceKernelArgs* kernelArgs) const
{
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, GetAicpuNum(), "PyptoRun");
#endif
    // use inputs/outputs store argsaddr/argsSize(aicpu task info + tensorInfo size)
    auto args = reinterpret_cast<AiCpuArgs*>(kernelArgs->inputs);
    kernelArgs->inputs = nullptr;
    args->kArgs = *kernelArgs;
    RtAicpuArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = reinterpret_cast<uint64_t>(kernelArgs->outputs);
    rtArgs.kernelNameAddrOffset = offsetof(AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(AiCpuArgs, soName);
    rtArgs.hostInputInfoNum = 1;
    RtHostInputInfo hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(AiCpuArgs);
    rtArgs.hostInputInfoPtr = &hostInputInfo;
    rtArgs.timeout = dynamic::AICPU_EXECUTE_TIMEOUT;
    MACHINE_LOGI("Copy flow addrOffset %u argsSize %u", hostInputInfo.addrOffset, hostInputInfo.dataOffset);
    return RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", GetAicpuNum(), &rtArgs, nullptr,
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
    void* devBufferPtr = CopyDataToDevice(buffer.data(), buffer.size());
    if (devBufferPtr == nullptr) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Failed to copy buffer of [%s] to device.", fileName.c_str());
        return;
    }
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(devBufferPtr);
    devArgs.aicpuSoLen = buffer.size();
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitAicpuSo);
}

int DeviceRunner::InitAicpuServer(int64_t *devArgsAddr)
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

    args.kArgs.cfgdata = devArgsAddr;

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

bool DeviceRunner::GetEnableDumpDevPref() const { return args_.aicpuPerfAddr != 0; }

void DeviceRunner::ResetMetrics(const uint32_t& coreId)
{
    if (perfData_.empty() || perfData_.size() <= static_cast<size_t>(coreId)) {
        return;
    }
    if (GetEnableDumpDevPref()) {
        if (!isPerfDataInited_) {
            RuntimeMemset(perfData_[coreId], sizeof(Metrics), 0, sizeof(Metrics));
            isPerfDataInited_ = true;
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

int DeviceRunner::RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream) const
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

void DeviceRunner::RunPost(RtStream aicpuStream, RtStream aicoreStream)
{
    AclRtEvent event;
    int rc = AclRtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    if (rc < 0) {
        MACHINE_LOGI("CreateEvent failed rc=%d.", rc);
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

int DeviceRunner::DynamicKernelLaunch(RtStream aicpuStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs) const
{
    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuInit);
    uint64_t startTime = MspfSysCycleTime();
    auto rc = LaunchDynamicAiCpu(aicpuStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicpuStream, startTime, GetAicpuNum(), MSPF_GE_TASK_TYPE_AI_CPU);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuRun);

    startTime = MspfSysCycleTime();
    rc = LaunchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicoreStream, startTime, GetBlockDim(), MSPF_GE_TASK_TYPE_MIX_AIC, true);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAIcore);
    return rc;
}

int DeviceRunner::DynamicTripleStreamLaunch(
    RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs)
{
    LoadAicpuOp::GetInstance().CustomAiCpuSoLoad();
    uint64_t startTime = MspfSysCycleTime();
    auto args = reinterpret_cast<AiCpuArgs*>(kernelArgs->inputs);
    kernelArgs->inputs = nullptr;
    args->kArgs = *kernelArgs;
    RtAicpuArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = reinterpret_cast<uint64_t>(kernelArgs->outputs);
    rtArgs.hostInputInfoNum = 1;
    rtArgs.kernelNameAddrOffset = offsetof(AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(AiCpuArgs, soName);
    RtHostInputInfo hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(AiCpuArgs);
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
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", GetAicpuNum(), &rtArgs, nullptr,
        (AclRtStream)schedStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch sche aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(schedStream, startTime, GetAicpuNum(), MSPF_GE_TASK_TYPE_AI_CPU, false);

    startTime = MspfSysCycleTime();
    rc = LaunchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICORE_FAILED, "triple stream launch aicore failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicoreStream, startTime, GetBlockDim(), MSPF_GE_TASK_TYPE_MIX_AIC, true);

    RunPost(ctrlStream, aicoreStream);
    return rc;
}

int DeviceRunner::DynamicLaunch(
    RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, [[maybe_unused]] int64_t taskId,
    DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum)
{
#ifdef BUILD_WITH_NEW_CANN
    if (!isPyptoNullLaunched_) {
        auto ret = LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, 1, "PyptoNull");
        if (ret != 0) {
            MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "launch built null failed");
            return ret;
        }
        isPyptoNullLaunched_ = true;
    }
#endif
    int rc = RunPrepare();
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_PREPARE_FAILED, "Prepare failed.");
        return rc;
    }
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitRunPrepare);

    // for dump perfInfo update device args
    args_.nrValidAic = blockdim;
    args_.nrAicpu = launchAicpuNum;
    args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(GetBlockDim(), GetAicpuNum(), args_.archInfo);

    ExchangeCaptureMode(isCapture_);
    if (ctrlStream == nullptr) {
        return DynamicKernelLaunch(aicpuStream, aicoreStream, kernelArgs);
    }
    return DynamicTripleStreamLaunch(aicpuStream, ctrlStream, aicoreStream, kernelArgs);
}

void DeviceRunner::ReportHostProfInfo(
    RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore) const
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
        (void)hostProf_.HostProfReportApi(startTime, endTime);
    }
    if (taskType == MSPF_GE_TASK_TYPE_MIX_AIC) {
        hostProf_.HostProfReportCacheTaskInfo(stream, blockDim, taskType);
    }
}

int DeviceRunner::DynamicRun(int64_t taskId, DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum)
{
    int rc = DynamicLaunch(GetContextScheStream(), GetContextCtrlStream(), GetContextAiCoreStream(), taskId, kernelArgs,
        blockdim, launchAicpuNum);
    if (rc < 0) {
        return rc;
    }
    if (isCapture_) {
        return 0;
    }
    return DynamicLaunchSynchronize(GetContextScheStream(), GetContextCtrlStream(), GetContextAiCoreStream());
}

/**************************** DynamicFunction *****************************/
int DeviceRunner::RegisterKernelBin(const std::vector<uint8_t>& binBuffer)
{
    binHdl_ = RegisterKernelBinary(binBuffer);
    MACHINE_LOGD("Finish registering kernel bin whose size is [%zu].", binBuffer.size());
    return binHdl_ == nullptr ? -1 : 0;
}

int DeviceRunner::RegisterKernelBin(void** hdl, const std::vector<uint8_t>& binBuffer)
{
    if (*hdl) {
        binHdl_ = *hdl;
        MACHINE_LOGD("Kernel bin has been registered.");
        return 0;
    }
    return RegisterKernelBin(binBuffer);
}

int DeviceRunner::Init()
{
    std::string builtInOpPath = config::LogTopFolder() + "/built_in";
    CreateMultiLevelDir(builtInOpPath);
    LoadAicpuOp::GetInstance().GenBuiltInOpInfo(builtInOpPath);
    if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle() != 0) {
        MACHINE_LOGE(DevCommonErr::GET_HANDLE_FAILED, "Get builtInOp Funchandle failed\n");
        return -1;
    }

    hostProf_.RegHostProf();
    InitializeErrorCallback();

    if (AclRtCreateEventExWithFlag(&event_, ACL_EVENT_SYNC) < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtCreateEvent failed.");
        return -1;
    }
    if (InitDeviceArgs(args_) != 0) {
        MACHINE_LOGE(HostLauncherErr::PREPARE_ARGS_FAILED, "prepareArgs failed\n");
        return -1;
    }
    int64_t *devArgsAddr = static_cast<int64_t*>(CopyDataToDevice(&args_, sizeof(DeviceArgs)));
    if (devArgsAddr == nullptr) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Failed to copy args to device.");
        return -1;
    }
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM) {
        InitAicpuServer(devArgsAddr);
    }
    InitPerfData();

    StartMachinePerfTraceDumpThread();
    npu::tile_fwk::dynamic::AdumpRegExceptionDump();
    return 0;
}

void DeviceRunner::InitPerfData()
{
    // init perf data
    for (uint64_t i = 0; i < args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        perfData_.push_back(MachinePerfTraceDevMalloc(PERF_DATA_TOTAL_SIZE, TWO_MB_HUGE_PAGE_FLAGS));
    }
}

void DeviceRunner::StartMachinePerfTraceDumpThread()
{
    if (!GetEnableDumpDevPref()) {
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
    int32_t deviceId = static_cast<int32_t>(args_.deviceId);
    if (RuntimeSetDevice(deviceId) != 0) {
        MACHINE_LOGW("Dump perf thread set Device[%d] not success", deviceId);
    }
    while (!dumpThreadStopFlag_.load()) {
        usleep(10000);
        dynamic::DumpDevTaskPerfData(args_, perfData_, false);
    }
    MACHINE_LOGD("Dump thread final dump");
    dynamic::DumpDevTaskPerfData(args_, perfData_, true);
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
