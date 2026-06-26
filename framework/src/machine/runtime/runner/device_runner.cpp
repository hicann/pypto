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
#include "utils/file_utils.h"
#include "machine/runtime/runner/runtime_agent.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/runner/pmu_common.h"
#include "machine/runtime/runner/load_aicpu_op.h"
#include "machine/runtime/runner/device_error_tracking.h"
#include "machine/runtime/runner/device_exception_dump.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/host/perf_analysis.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/utils/machine_ws_intf.h"
#include "securec.h"

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
    if (RuntimeMemcpyDirect(devAddr, dataSize, dataPtr, dataSize, RtMemcpyKind::HOST_TO_DEVICE) != RT_SUCCESS) {
        DevMemoryPool::Instance().FreeDevAddr(devAddr);
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Failed to copy data to dev of size %lu", dataSize);
        return nullptr;
    }
    return devAddr;
}
}

DeviceRunner::~DeviceRunner() {}

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

void DeviceRunner::ResetPerData() const { devicePerf_.ResetPerData(); }

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
        auto aicpuDevPtr = DevMallocWithAlignSize(MAX_ROUND_NUM * sizeof(MetricPerf), TWO_MB_HUGE_PAGE_FLAGS);
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
        RuntimeAgent::GetAicoreRegInfoForDAV3510(regs, regsPmu);
        return;
    }
    if (archInfo == ArchInfo::DAV_2201) {
        std::vector<int64_t> aiv;
        std::vector<int64_t> aic;
        if (RuntimeAgent::GetAgent().GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
            return;
        }
        regs.insert(regs.end(), aic.begin(), aic.end());
        regs.insert(regs.end(), aiv.begin(), aiv.end());

        std::vector<int64_t> aivPmu;
        std::vector<int64_t> aicPmu;
        if (RuntimeAgent::GetAgent().GetAicoreRegInfo(aicPmu, aivPmu, ADDR_MAP_TYPE_REG_AIC_PMU_CTRL) != 0) {
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
    uint32_t maxAicpuNum = static_cast<uint32_t>(Platform::Instance().GetSoc().GetAICPUNum());
    args.nrValidAic = GetCfgBlockdim();
    args.nrAicpu = std::min(aicpuNum, maxAicpuNum);
    args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args.nrValidAic, args.nrAicpu, args.archInfo);
    MACHINE_LOGD("DevArgs: block dim[%u], aicpu num[%u], max aicpu num[%u], sche cpu num[%u].",
        args.nrValidAic, args.nrAicpu, maxAicpuNum, args.scheCpuNum);

    InitAiCpuSoBin(args);

    return InitDeviceArgsCore(args);
}

void DeviceRunner::SyncProfData(bool debugEnable) { devicePerf_.SyncProfData(debugEnable); }

/**************************** DynamicFunction *****************************/
int DeviceRunner::DynamicLaunchSynchronize(RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream)
{
    int rcAicore = RuntimeStreamSynchronize(aicoreStream);
    int rcAicpu = RuntimeStreamSynchronize(schedStream);
    int rcCtrl = 0;
    if (ctrlStream != nullptr) {
        rcCtrl = RuntimeStreamSynchronize(ctrlStream);
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

int DeviceRunner::LaunchDynamicAiCore(void *binHandle, RtStream aicoreStream, uint32_t blockDim,
                                      DeviceKernelArgs* kernelArgs)
{
    RtArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void*> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    RtTaskCfgInfo cfg = {};
    cfg.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);
    return RuntimeKernelLaunchWithHandleV2(binHandle, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &cfg);
}

int DeviceRunner::LaunchDynamicAiCpu(RtStream aicpuStream, uint32_t aicpuNum, DeviceKernelArgs* kernelArgs)
{
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, aicpuNum, "PyptoRun");
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
    return RuntimeAicpuKernelLaunchExWithArgs(static_cast<uint32_t>(RtKernelType::AICPU_KFC), "AST_DYN_AICPU", aicpuNum,
        &rtArgs, nullptr, aicpuStream, RT_KERNEL_USE_SPECIAL_TIMEOUT);
}

void DeviceRunner::InitAiCpuSoBin(DeviceArgs& devArgs)
{
    std::string fileName = GetPyptoLibPath() + "/libtilefwk_backend_server.so";
    auto buffer = ReadFile(fileName);
    if (buffer.empty()) {
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

int DeviceRunner::LaunchAicpuServerInit(int64_t *devArgsAddr)
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
        static_cast<uint32_t>(RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr, aicpuStream, 0);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_LAUNCH_FAILED, "Aicpu server init failed %d", ret);
        return ret;
    }
    // for triple stream schedule, must wait aicpu server init done
    return RuntimeStreamSynchronize(aicpuStream);
}

bool DeviceRunner::GetEnableDumpDevPref() const { return args_.aicpuPerfAddr != 0; }

void DeviceRunner::SetDebugEnable() { devicePerf_.SetDebugEnable(); }

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

int DeviceRunner::DynamicKernelLaunch(const KernelLaunchInfo &launchInfo, DeviceKernelArgs* kernelArgs) const
{
    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuInit);
    uint64_t startTime = MspfSysCycleTime();
    auto rc = LaunchDynamicAiCpu(launchInfo.schedStream, launchInfo.aicpuNum, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(launchInfo.aicoreStream, startTime, launchInfo.aicpuNum, MSPF_GE_TASK_TYPE_AI_CPU);
    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuRun);

    startTime = MspfSysCycleTime();
    rc = LaunchDynamicAiCore(launchInfo.binHandle, launchInfo.aicoreStream, launchInfo.blockDim, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(launchInfo.aicoreStream, startTime, launchInfo.blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAIcore);
    return rc;
}

int DeviceRunner::DynamicTripleStreamLaunch(const KernelLaunchInfo &launchInfo, DeviceKernelArgs* kernelArgs) const
{
    LoadAicpuOp::GetInstance().CustomAiCpuSoLoad();
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

    uint64_t startTime = MspfSysCycleTime();
    int rc = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr, launchInfo.ctrlStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch ctrl aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(launchInfo.ctrlStream, startTime, 1, MSPF_GE_TASK_TYPE_AI_CPU, false);

    startTime = MspfSysCycleTime();
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    rc = RuntimeAicpuKernelLaunchExWithArgs(static_cast<uint32_t>(RtKernelType::AICPU_KFC), "AST_DYN_AICPU",
                                            launchInfo.aicpuNum, &rtArgs, nullptr, launchInfo.schedStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch sche aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(launchInfo.schedStream, startTime, launchInfo.aicpuNum, MSPF_GE_TASK_TYPE_AI_CPU, false);

    startTime = MspfSysCycleTime();
    rc = LaunchDynamicAiCore(launchInfo.binHandle, launchInfo.aicoreStream, launchInfo.blockDim, kernelArgs);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICORE_FAILED, "triple stream launch aicore failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(launchInfo.aicoreStream, startTime, launchInfo.blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);

    RunPost(launchInfo.ctrlStream, launchInfo.aicoreStream);
    return rc;
}

int DeviceRunner::DynamicLaunch(const KernelLaunchInfo &launchInfo, DeviceKernelArgs* kernelArgs)
{
#ifdef BUILD_WITH_NEW_CANN
    auto ret = LoadAicpuOp::GetInstance().LaunchPyptoNullOp(launchInfo.schedStream, kernelArgs, 1);
    if (ret != 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "launch built null failed");
        return ret;
    }
#endif
    if (!devicePerf_.RunPrepare()) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_PREPARE_FAILED, "Prepare failed.");
        return -1;
    }
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitRunPrepare);

    if (launchInfo.isCaptureActivate) {
        ExchangeCaptureModeGlobal();
    }
    if (launchInfo.ctrlStream == nullptr) {
        return DynamicKernelLaunch(launchInfo, kernelArgs);
    }
    return DynamicTripleStreamLaunch(launchInfo, kernelArgs);
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

int DeviceRunner::DynamicRun(const KernelLaunchInfo &launchInfo, DeviceKernelArgs* kernelArgs)
{
    int rc = DynamicLaunch(launchInfo, kernelArgs);
    if (rc < 0) {
        return rc;
    }
    if (launchInfo.isCaptureActivate) {
        return 0;
    }
    return DynamicLaunchSynchronize(launchInfo.schedStream, launchInfo.ctrlStream, launchInfo.aicoreStream);
}

int DeviceRunner::Init()
{
    LoadAicpuOp::GetInstance().GenBuiltInOpInfo();
    if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle() != 0) {
        MACHINE_LOGE(DevCommonErr::GET_HANDLE_FAILED, "Get builtInOp Funchandle failed\n");
        return -1;
    }

    hostProf_.RegHostProf();
    InitializeErrorCallback();

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
        LaunchAicpuServerInit(devArgsAddr);
        devicePerf_.InitAndStartDumpThread(args_);
        npu::tile_fwk::dynamic::AdumpRegExceptionDump();
    }
    return 0;
}
} // namespace npu::tile_fwk
