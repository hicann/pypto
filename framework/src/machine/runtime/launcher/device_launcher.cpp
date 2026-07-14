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
 * \file device_launcher.cpp
 * \brief
 */

#include "machine/runtime/launcher/device_launcher.h"

#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/adump_api.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/memory_utils/eslmodel_memory_utils.h"
#include "interface/configs/config_manager_ng.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/context/device_launcher_context.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/launcher/device_launcher_driver_gate.h"
#include "machine/host/perf_analysis.h"

extern "C" __attribute__((weak)) int AdxDataDumpServerUnInit();
extern "C" __attribute__((weak)) int AdxDataDumpServerInit();

namespace npu::tile_fwk::dynamic {
bool DeviceLauncher::inited_ = false;
std::vector<uint8_t> DeviceLauncher::tensorInfo_(kDefaultTensorinfoSize);
std::unordered_map<Function*, DeviceLauncher::DeviceRunCacheInfo> DeviceLauncher::cacheInfoDict_;

void DeviceLauncher::CheckAscendDriverVersionOnboard()
{
    AscendDriverVersionGate::EnsureDriverVersionForOnboardOnce();
}

int DeviceLauncher::SetCaptureStream(RtStream aicoreStream, RtStream aicpuStream, bool& isCapture)
{
    AclMdlRI rtModel = nullptr;

    if (!GetStreamCaptureInfo(aicoreStream, rtModel, isCapture)) {
        return -1;
    }
    DeviceLauncherContext::Get().SetCaptureMode(isCapture);

    if (isCapture) {
        if (rtModel == nullptr) {
            MACHINE_LOGE(DevCommonErr::NULLPTR, "rtModel is null!");
            return -1;
        }
        RtError ret = RuntimeStreamAddToModel(aicpuStream, rtModel);
        if (ret != 0) {
            MACHINE_LOGE(RtErr::RT_LAUNCH_FAILED, "RuntimeStreamAddToModel failed, return[%d]", ret);
            return -1;
        }
    }
    return 0;
}

int DeviceLauncher::RunWithProfile(RtStream aicoreStream, RtStream aicpuStream, bool isCapture)
{
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        if (isCapture) {
            MACHINE_LOGW("The swimlane function is not currently supported in CaptureMode. The contents of "
                         "tilefwk_L1_prof_data may be empty.");
            return 0;
        }
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
        if (rc < 0) {
            return rc;
        }
        DeviceRunner::Get().SyncProfData(true);
        DeviceRunner::Get().ResetPerData();
    }
    return 0;
}

int DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    RtStream aicoreStream, bool streamSynchronize, CachedOperator* cachedOperator,
    DevControlFlowCache* inputDevCtrlCache, const DeviceLauncherConfig& config)
{
    MACHINE_LOGI("Kernel Launch");
    aicoreStream = aicoreStream == nullptr ? GetContextAiCoreStream() : aicoreStream;
    KernelLaunchInfo launchInfo(GetContextScheStream(), GetContextCtrlStream(), aicoreStream,
        config.blockdim, config.aicpuNum);
    // 1.Add stream to capture model
    int rc = SetCaptureStream(launchInfo.aicoreStream, launchInfo.schedStream, launchInfo.isCaptureActivate);
    if (rc < 0) {
        return rc;
    }

    // 2. Change capture mode to relaxed
    if (launchInfo.isCaptureActivate) {
        ExchangeCaptureModeRelax();
    }
    HOST_PERF_TRACE(TracePhase::RunDeviceSetCapture);

    DeviceRunner::Get().SetHostProfFunction(function);
    rc = AclInit(nullptr);
    if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }
    HOST_PERF_TRACE(TracePhase::RunDeviceInit);

    CheckAscendDriverVersionOnboard();

    if (cachedOperator == nullptr) {
        // Not python cached operator mode, consider kernel reuse mode
        if (IsDevRunCacheKernelEnable(function)) {
            cachedOperator = GetDevRunCacheOperator(function);
        }
    }

    if (function != nullptr && function->GetDyndevAttribute() != nullptr) {
        launchInfo.binHandle = *reinterpret_cast<RtBinHandle*>(CachedOperator::GetBinHandleHolder(cachedOperator));
        if (launchInfo.binHandle == nullptr) {
            launchInfo.binHandle = RegisterKernelBin(function->GetDyndevAttribute()->kernelBinary);
        }
        if (launchInfo.binHandle == nullptr) {
            MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Register kernel bin failed.");
            return -1;
        }
    }
    HOST_PERF_TRACE(TracePhase::RunDevRegistKernelBin);

    auto dynAttr = function->GetDyndevAttribute();
    CheckDeviceId();
    DeviceKernelArgs kArgs;
    DeviceLauncherConfigFillDeviceInfo(config);
    DeviceMemoryUtils devMemoryUtilis;
    DeviceInitDistributedContext(devMemoryUtilis, dynAttr->commGroupNames, kArgs);

    HOST_PERF_TRACE(TracePhase::RunDevEnvReady);
    DeviceInitTilingData(devMemoryUtilis, kArgs, dynAttr->devProgBinary, inputDevCtrlCache, config, cachedOperator);
    HOST_PERF_TRACE(TracePhase::RunDevInitTiling);

    SetDevRunCacheKernel(function, (uint8_t*)kArgs.cfgdata);
    DeviceInitKernelInOuts(devMemoryUtilis, kArgs, inputList, outputList, dynAttr->disableL2List);

    HOST_PERF_TRACE(TracePhase::RunDevInitInOutTensor);

    DataDumpInit();
    launchInfo.blockDim = config.blockdim;
    launchInfo.aicpuNum = config.aicpuNum;
    rc = DeviceRunner::Get().DynamicLaunch(launchInfo, &kArgs);
    if (rc < 0) {
        return rc;
    }
    rc = RunWithProfile(aicoreStream, launchInfo.schedStream, launchInfo.isCaptureActivate);
    if (rc < 0) {
        return rc;
    }
    if (streamSynchronize) {
        rc = DeviceRunner::Get().DynamicLaunchSynchronize(launchInfo.schedStream, launchInfo.ctrlStream, aicoreStream);
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, DevMemoryPool::Instance().CheckAllSentinels());
    }
    MACHINE_LOGI("finish Kernel Launch.");

    HOST_PERF_TRACE(TracePhase::RunDevRunProfile);
    DataDumpUnInit();
    return rc;
}

int DeviceLauncher::DeviceSynchronize(RtStream aicpuStream, RtStream aicoreStream)
{
    int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
    return rc;
}

int DeviceLauncher::DeviceRunOnce(
    Function* function, DevControlFlowCache* hostCtrlCache, const DeviceLauncherConfig& config)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    DeviceMemoryUtils devMemoryUtilis(true);
    std::tie(inputDeviceDataList, outputDeviceDataList) =
        BuildInputOutputFromHost(devMemoryUtilis, inputDataList, outputDataList);

    DeviceMemoryUtils devMemory(false);
    uint8_t* devCtrlCache = nullptr;
    if (hostCtrlCache) {
        devCtrlCache =
            devMemory.CopyToDev(reinterpret_cast<uint8_t*>(hostCtrlCache), hostCtrlCache->usedCacheSize, nullptr);
    }

    int rc = DeviceLaunchOnceWithDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr, true,
        nullptr, reinterpret_cast<DevControlFlowCache*>(devCtrlCache), config);
    CopyFromDev(DeviceMemoryUtils(), outputDataList);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        CopyFromDev(DeviceMemoryUtils(), inputDataList);
    }
    for (const auto& data : inputDeviceDataList) {
        devMemoryUtilis.Free(static_cast<uint8_t*>(data.GetAddr()));
    }
    for (const auto& data : outputDeviceDataList) {
        devMemoryUtilis.Free(static_cast<uint8_t*>(data.GetAddr()));
    }
    devMemory.Free(devCtrlCache);
    return rc;
}

void DeviceLauncher::SetDevRunCacheKernelEnable(Function* func, bool enabled)
{
    cacheInfoDict_[func].devProgEnabled = enabled;
}

bool DeviceLauncher::IsDevRunCacheKernelEnable(Function* func)
{
    return cacheInfoDict_[func].devProgEnabled;
}

void DeviceLauncher::SetDevRunCacheKernel(Function* func, uint8_t* devProg)
{
    if (!IsDevRunCacheKernelEnable(func)) {
        return;
    }
    *CachedOperator::GetCfgDataDevAddrHolder(&(cacheInfoDict_[func].cacheOperator)) = devProg;
}

CachedOperator* DeviceLauncher::GetDevRunCacheOperator(Function* func)
{
    if (!IsDevRunCacheKernelEnable(func)) {
        return nullptr;
    }
    return &(cacheInfoDict_[func].cacheOperator);
}

void DeviceLauncher::DataDumpInit()
{
    if (IsPtoDataDumpEnabled()) {
        if (!AdxDataDumpServerInit) {
            MACHINE_LOGW("AdxDataDumpServerInit function not found.");
            return;
        }
        MACHINE_LOGD("DataDumpServerInit is called \n");
        int sf = AdxDataDumpServerInit();
        if (sf != 0) {
            MACHINE_LOGW("ERROR AdxDataDumpServerInit failed \n");
        }
    }
}

void DeviceLauncher::DataDumpUnInit()
{
    if (IsPtoDataDumpEnabled()) {
        if (!AdxDataDumpServerUnInit) {
            MACHINE_LOGW("AdxDataDumpServerUnInit function not found.");
            return;
        }
        MACHINE_LOGD("DataDumpServerUnInit is called \n");
        int sf = AdxDataDumpServerUnInit();
        if (sf != 0) {
            MACHINE_LOGW("AdxDataDumpServerUnInit is failed %d \n", sf);
        }
    }
}

int32_t DataFormat2CannFormat(const TileOpFormat format)
{
    constexpr int32_t GE_FORMAT_ND = 2;
    constexpr int32_t GE_FORMAT_NZ = 29;
    switch (format) {
        case TileOpFormat::TILEOP_ND:
            return GE_FORMAT_ND;
        case TileOpFormat::TILEOP_NZ:
            return GE_FORMAT_NZ;
        default:
            throw std::invalid_argument("Unknown Format");
    }
}

void DeviceLauncher::DumpIOTensorsWithCann(AclRtStream stream, std::vector<DeviceTensorData>& tensors,
    const std::string& funcName)
{
    if (AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR) != 0) {
        std::vector<AdxTensorInfoV2> dumpTensors;
        for (auto& tensor : tensors) {
            AdxTensorInfoV2 info;
            info.type = AdxTensorType::INPUT;
            info.addrType = AdxAddressType::TRADITIONAL;
            info.tensorSize = static_cast<size_t>(tensor.GetDataSize());
            info.format = DataFormat2CannFormat(tensor.Format());
            info.dataType = static_cast<int32_t>(DataType2CannType(tensor.GetDataType()));
            info.tensorAddr = static_cast<int64_t *>(tensor.GetAddr());
            info.placement = static_cast<int32_t>(AdxTensorPlacement::kOnDeviceHbm);
            info.shape = tensor.GetShape();
            info.originShape = tensor.GetShape();
            dumpTensors.push_back(info);
        }
        AdxDumpDumpTensorV2(funcName, funcName, dumpTensors, stream);
    }
}

uint8_t* DeviceLauncher::CopyControlFlowCache(DevControlFlowCache* ctrlCache)
{
    uint8_t* devCache = nullptr;
    auto cacheSize = ctrlCache->usedCacheSize;
    auto bufNum = DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT;

    int ret = RuntimeMalloc((void**)&devCache, cacheSize * bufNum, RT_MEMORY_HBM, 0);
    if (devCache == nullptr) {
        MACHINE_LOGE(RtErr::RT_MALLOC_FAILED, "control flow cache malloc failed");
        return nullptr;
    }

    for (int i = 0; i < bufNum; ++i) {
        ret = static_cast<int>(RuntimeMemcpyDirect(devCache + i * cacheSize, cacheSize, ctrlCache, cacheSize,
                                                   RtMemcpyKind::HOST_TO_DEVICE));
        if (ret != 0) {
            MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED, "control flow cache memcpy failed, ret: %d", ret);
            RuntimeFree(devCache);
            return nullptr;
        }
    }
    return devCache;
}

void DeviceLauncher::FreeControlFlowCache(uint8_t* ctrlCache)
{
    if (ctrlCache != nullptr) {
        RuntimeFree(ctrlCache);
    }
}

void DeviceLauncher::AddAicpuStream(const bool isCapture, AclMdlRI& rtModel)
{
    if (isCapture) {
        RuntimeStreamAddToModel(GetContextCtrlStream(), rtModel);
        RuntimeStreamAddToModel(GetContextScheStream(), rtModel);
    }
}

void DeviceLauncher::SaveStream(AclRtStream aicoreStream)
{
    // 存储 current stream，后续控核接口需使用current stream
    GetStreamContext().SetCurrentStream(aicoreStream);
}

void DeviceLauncher::GetCaptureInfo(AclRtStream aicoreStream, AclMdlRI& rtModel)
{
    bool isCapture = false;
    (void)GetStreamCaptureInfo(aicoreStream, rtModel, isCapture);
    DeviceLauncherContext::Get().SetCaptureMode(isCapture);
}

bool DeviceLauncher::IsCaptureMode() { return DeviceLauncherContext::Get().IsCaptureMode(); }

void DeviceLauncher::SetDevPerfAddr([[maybe_unused]] const bool debugEnable, [[maybe_unused]] const bool isCaptureMode)
{
    auto& devRunner = DeviceRunner::Get();
    if (debugEnable || devRunner.GetEnableDumpDevPref() || devRunner.GetHostProfType() == 1) {
        if (isCaptureMode) {
            ExchangeCaptureModeRelax();
        }
        devRunner.SetDebugEnable();
        if (isCaptureMode) {
            ExchangeCaptureModeGlobal();
        }
    }
}

int DeviceLauncher::LaunchSyncTask(AclRtStream aicoreStream, bool isCaptureMode, int launchEarlyMode)
{
    if (launchEarlyMode == 1) { // 1 ： early launch in all modes
        return 0;
    }
    if (launchEarlyMode == 0 && isCaptureMode) {  // 0 : early launch only in capture mode
        return 0;
    }

    //  close early launch
    auto schedStream = GetStreamContext().GetScheStream();
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    return RunPreSync(schedStream, ctrlStream, aicoreStream);
}

int DeviceLauncher::RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream)
{
    AclRtEvent event;
    if (AclRtCreateEventExWithFlag(&event, ACL_EVENT_SYNC) < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtCreateEvent failed.");
        return -1;
    }
    int rc = AclRtRecordEvent(event, aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtRecordEvent failed %d\n", rc);
        return rc;
    }
    rc = AclRtStreamWaitEvent(scheStream, event);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    rc = AclRtStreamWaitEvent(ctrlStream, event);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

int DeviceLauncher::LaunchAicpuKernel(
    RtAicpuArgsEx& rtArgs, [[maybe_unused]] bool debugEnable, [[maybe_unused]] Function* function,
    const std::vector<DeviceTensorData>& tensors)
{
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    auto schedStream = GetStreamContext().GetScheStream();
    auto& devRunner = DeviceRunner::Get();
    devRunner.SetHostProfFunction(function, tensors);
    int ret = 0;
    auto args = (AiCpuArgs*)rtArgs.args;
    const int nrAicpu = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.nrAicpu);
    const bool launchSchedSameCluster = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.launchSchedSameCluster);
    if (launchSchedSameCluster) {
        MACHINE_LOGW("When available AICPUs are insufficient, execute export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false"
            "to disable the constraint that forces scheduling threads onto the same cluster.");
    }
    args->kArgs.parameter.ctrlBlockNum = static_cast<int>(DeviceLauncher::GetDevProg(function)->ctrlBlockDim);
    auto startTime = MspfSysCycleTime();    
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    ret = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr, ctrlStream,
        RT_KERNEL_USE_SPECIAL_TIMEOUT);
    devRunner.ReportHostProfInfo(ctrlStream, startTime, 1, MSPF_GE_TASK_TYPE_AI_CPU, false);
    if (ret != RT_SUCCESS) {
        return ret;
    }
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    startTime = MspfSysCycleTime();
    const int scheCpuNum = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.scheCpuNum);
    ret = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", nrAicpu, &rtArgs, nullptr,
        schedStream, RT_KERNEL_USE_SPECIAL_TIMEOUT);
    devRunner.ReportHostProfInfo(schedStream, startTime, scheCpuNum, MSPF_GE_TASK_TYPE_AI_CPU, false);    
    return ret;
}

int DeviceLauncher::LaunchAicoreKernel(
    AclRtStream aicoreStream, void* kernel, RtArgsEx& rtArgs, RtTaskCfgInfo& rtTaskCfg,
    bool debugEnable, [[maybe_unused]] Function* function)
{
    auto& devRunner = DeviceRunner::Get();
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    int blockDim = static_cast<int>(DeviceLauncher::GetDevProg(function)->ctrlBlockDim);
    if (blockDim == 0) {
        blockDim = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.nrValidAic);
    }
    auto startTime = MspfSysCycleTime();
    auto ret = RuntimeKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    devRunner.ReportHostProfInfo(aicoreStream, startTime, blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);
    if (debugEnable) {
        auto scheStream = GetStreamContext().GetScheStream();
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(scheStream, nullptr, aicoreStream);
        if (rc != 0) {
            MACHINE_LOGE(HostLauncherErr::SYNC_FAILED, "sync failed");
            return rc;
        }
        devRunner.SyncProfData(debugEnable);
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, DevMemoryPool::Instance().CheckAllSentinels());
    }
    if (IsPtoDataDumpEnabled()) {
        auto scheStream = GetStreamContext().GetScheStream();
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(scheStream, nullptr, aicoreStream);
        if (rc != 0) {
            MACHINE_LOGE(HostLauncherErr::SYNC_FAILED, "sync failed");
            return rc;
        }
        uint32_t hostPid = GetProcessId();
        std::string sourceDir = "output/dump_tensor_" + std::to_string(hostPid);
        std::string targetDir = config::LogTopFolder() + "/dump_tensor_" + std::to_string(hostPid);
        if (IsPathExist(sourceDir)) {
            std::rename(sourceDir.c_str(), targetDir.c_str());
        }
    }
    return ret;
}
} // namespace npu::tile_fwk::dynamic
