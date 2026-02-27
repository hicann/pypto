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

#include "machine/runtime/device_launcher.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/host/backend.h"
#include "machine/runtime/host_prof.h"
#include "machine/host/perf_analysis.h"
#include "interface/utils/op_info_manager.h"

extern "C" __attribute__((weak)) int AdxDataDumpServerUnInit();
extern "C" __attribute__((weak)) int AdxDataDumpServerInit();

namespace npu::tile_fwk::dynamic {
namespace {
    constexpr uint32_t kMinDefaultDim = 20;
}
int GetCfgBlockdim() {
#ifdef BUILD_WITH_CANN
    auto blk = Platform::Instance().GetSoc().GetAICoreNum();
    blk = blk > 0 ? blk : kMinDefaultDim;
    ALOG_DEBUG_F("Get blockdim[%d].", blk);
    return blk;
#else
    return kMinDefaultDim;
#endif
}

void (*forceLinkLibraryCompiler)() = &npu::tile_fwk::ForceLinkLibraryCompiler;

DeviceLauncherContext &DeviceLauncherContext::Get() {
    static DeviceLauncherContext context;
    return context;
}

std::vector<uint8_t> DeviceLauncher::tensorInfo_(kDefaultTensorinfoSize);

#ifdef BUILD_WITH_CANN
static const std::unordered_map<int, std::function<void(bool&)>> captureStatusHandlers = {
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, [](bool& isCapture) {isCapture = true;}},
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE,
        [](bool& isCapture) {(void)isCapture; ALOG_DEBUG_F("GetStreamCaptureInfo: status NONE");}},
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
        [](bool& isCapture) {(void)isCapture; ALOG_DEBUG_F("GetStreamCaptureInfo: status invalidated");}}
};

int DeviceLauncher::GetStreamCaptureInfo(rtStream_t aicoreStream, aclmdlRI &rtModel, bool &isCapture)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(aicoreStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        ALOG_WARN_F("Stream capture not support");
        return 0;
    } else if (ret != ACL_SUCCESS) {
        ALOG_ERROR_F("aclmdlRICaptureGetInfo failed, return[%d]", ret);
        return -1;
    }

    auto it = captureStatusHandlers.find(captureStatus);
    if (it != captureStatusHandlers.end()) {
        it->second(isCapture);
    } else {
        ALOG_ERROR_F("GetStreamCaptureInfo get unsupport capture status");
        return -1;
    }
    ALOG_INFO_F("capture mode[%d]", isCapture);
    return 0;
}

void DeviceLauncher::ChangeCaptureModeRelax()
{
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;   // aclgraph does not support rtmemcpy / rtmemset, set to relaxed mode
    aclmdlRICaptureThreadExchangeMode(&mode);
}

void DeviceLauncher::ChangeCaptureModeGlobal()
{
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
    aclmdlRICaptureThreadExchangeMode(&mode);
}

int DeviceLauncher::SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream, bool &isCapture)
{
    aclmdlRI rtModel = nullptr;

    if (GetStreamCaptureInfo(aicoreStream, rtModel, isCapture) < 0) {
        return -1;
    }

    if (isCapture) {
        if (rtModel ==  nullptr) {
            ALOG_ERROR_F("rtModel is null!");
            return -1;;
        }
        rtError_t ret = rtStreamAddToModel(aicpuStream, rtModel);
        if (ret != 0) {
            ALOG_ERROR_F("rtStreamAddToModel failed, return[%d]", ret);
            return -1;
        }
    }
    return 0;
}

int DeviceLauncher::RunWithProfile(rtStream_t aicoreStream, rtStream_t aicpuStream, bool isCapture) {
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        if (isCapture) {
            ALOG_WARN("The swimlane function is not currently supported in CaptureMode. The contents of tilefwk_L1_prof_data may be empty.");
            return 0;
        }
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
        if (rc < 0) {
            return rc;
        }
        DeviceRunner::Get().SynchronizeDeviceToHostProfData();
        DeviceRunner::Get().ResetPerData();
    }
    return 0;
}

int DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator,
        DevControlFlowCache* inputDevCtrlCache, const DeviceLauncherConfig &config) {
    bool isCapture = false;
    ALOG_INFO_F("Kernel Launch");

    HOST_PERF_TRACE(TracePhase::RunDeviceInit);

    if (cachedOperator == nullptr) { // st scene
        if (function != nullptr && function->GetDyndevAttribute() != nullptr) {
            DeviceRunner::SetBinData(function->GetDyndevAttribute()->kernelBinary);
        }
    }

    /* 1.Add stream to capture model*/
    int rc = SetCaptureStream(aicoreStream, aicpuStream, isCapture);
    if (rc < 0) {
        return rc;
    }

    /* 2. Change capture mode to relaxed*/
    if (isCapture) {
        ChangeCaptureModeRelax();
    }
    DeviceRunner::Get().SetCaptureFlag(isCapture);

    HOST_PERF_TRACE(TracePhase::RunDeviceSetCapture);

    DeviceRunner::Get().GetHostProfInstance().SetProfFunction(function);
    rc = aclInit(nullptr);
    if (rc != 0 && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }

    if (cachedOperator == nullptr) {
        // Not python cached operator mode, consider kernel reuse mode
        if (DeviceRunCacheKernelEnable(function)) {
            cachedOperator = DeviceRunCacheOperatorGet(function);
        }
    }

    auto dynAttr = function->GetDyndevAttribute();
    CheckDeviceId();
    DeviceKernelArgs kArgs;
    DeviceLauncherConfigFillDeviceInfo(config);
    DeviceInitDistributedContext(dynAttr->commGroupNames, GetDevProg(function));

    HOST_PERF_TRACE(TracePhase::RunDevEnvReady);
    DeviceInitTilingData(DeviceMemoryUtils(), kArgs, dynAttr->devProgBinary, inputDevCtrlCache, config, cachedOperator);
    HOST_PERF_TRACE(TracePhase::RunDevInitTiling);

    DeviceRunCacheKernelSet(function, (uint8_t *)kArgs.cfgdata);
    DeviceInitKernelInOuts(DeviceMemoryUtils(), kArgs, inputList, outputList, dynAttr->disableL2List);

    HOST_PERF_TRACE(TracePhase::RunDevInitInOutTensor);

    rc = DeviceRunner::Get().RegisterKernelBin(&(*reinterpret_cast<rtBinHandle *>(CachedOperator::GetBinHandleHolder(cachedOperator))),
            cachedOperator == nullptr ? nullptr : &(function->GetDyndevAttribute()->kernelBinary));
    if (rc < 0) {
        ALOG_ERROR_F("Register kernel bin failed.");
        return rc;
    }

    HOST_PERF_TRACE(TracePhase::RunDevRegistKernelBin);

    DataDumpInit();
    rc = DeviceRunner::Get().DynamicLaunch(aicpuStream, nullptr, aicoreStream, 0, &kArgs, config.blockdim, config.aicpuNum);
    if (rc < 0) {
        return rc;
    }
    rc = RunWithProfile(aicoreStream, aicpuStream, isCapture);
    if (rc < 0) {
        return rc;
    }
    if (streamSynchronize) {
        rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
        ASSERT(machine::GetRA()->CheckAllSentinels());
    }
    ALOG_INFO_F("finish Kernel Launch.");

    HOST_PERF_TRACE(TracePhase::RunDevRunProfile);
    DataDumpUnInit();
    return rc;
}

int DeviceLauncher::DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
    return rc;
}
#endif

int DeviceLauncher::DeviceRunOnce(Function *function, DevControlFlowCache* hostCtrlCache, const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    auto aicpuStream = machine::GetRA()->GetScheStream();
    auto aicoreStream = machine::GetRA()->GetStream();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = BuildInputOutputFromHost(DeviceMemoryUtils(), inputDataList, outputDataList);

    uint8_t* devCtrlCache = nullptr;
    DeviceMemoryUtils devMemory(false);
    if (hostCtrlCache) {
        devCtrlCache = devMemory.CopyToDev(reinterpret_cast<uint8_t *>(hostCtrlCache), hostCtrlCache->usedCacheSize, nullptr);
    }

    int rc = DeviceLaunchOnceWithDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList,
        aicpuStream, aicoreStream, true, nullptr, reinterpret_cast<DevControlFlowCache*>(devCtrlCache), config);
    CopyFromDev(DeviceMemoryUtils(), outputDataList);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        CopyFromDev(DeviceMemoryUtils(), inputDataList);
    }
    devMemory.Free(devCtrlCache);
    machine::GetRA()->FreeTmpMemory();
    return rc;
#else
    (void)hostCtrlCache;
    (void)function;
    (void)config;
    return 0;
#endif
}

struct DeviceRunCacheInfo {
    /* By default: devProg cache is enabled */
    bool devProgEnabled{true};
    CachedOperator cacheOperator;
};
static std::unordered_map<Function *, DeviceRunCacheInfo> &DeviceRunCacheInfoDict() {
    static std::unordered_map<Function *, DeviceRunCacheInfo> cacheInfoDict;
    return cacheInfoDict;
}
void DeviceLauncher::DeviceRunCacheKernelEnable(Function *func, bool enabled) {
    auto &dict = DeviceRunCacheInfoDict();
    dict[func].devProgEnabled = enabled;
}
bool DeviceLauncher::DeviceRunCacheKernelEnable(Function *func) {
    auto &dict = DeviceRunCacheInfoDict();
    return dict[func].devProgEnabled;
}
void DeviceLauncher::DeviceRunCacheKernelSet(Function *func, uint8_t *devProg) {
    if (!DeviceRunCacheKernelEnable(func)) {
        return;
    }
    auto &dict = DeviceRunCacheInfoDict();
    *CachedOperator::GetCfgDataDevAddrHolder(&(dict[func].cacheOperator)) = devProg;
}

uint8_t *DeviceLauncher::DeviceRunCacheKernelGet(Function *func) {
    if (!DeviceRunCacheKernelEnable(func)) {
        return nullptr;
    }
    auto &dict = DeviceRunCacheInfoDict();
    return *CachedOperator::GetCfgDataDevAddrHolder(&(dict[func].cacheOperator));
}

CachedOperator* DeviceLauncher::DeviceRunCacheOperatorGet(Function *func) {
    if (!DeviceRunCacheKernelEnable(func)) {
        return nullptr;
    }
    auto &dict = DeviceRunCacheInfoDict();
    return &(dict[func].cacheOperator);
}

DeviceStream DeviceGetAicpuStream() {
#ifdef BUILD_WITH_CANN
    rtStream_t aicpuStreamValue = machine::GetRA()->GetScheStream();
    return reinterpret_cast<DeviceStream>(aicpuStreamValue);
#else
    return 0;
#endif
}

DeviceStream DeviceGetAicoreStream() {
#ifdef BUILD_WITH_CANN
    rtStream_t aicoreStreamValue = machine::GetRA()->GetStream();
    return reinterpret_cast<DeviceStream>(aicoreStreamValue);
#else
    return 0;
#endif
}

int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(
        ExportedOperator *op, const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList,
        DeviceStream aicpuStream, DeviceStream aicoreStream, bool streamSynchronize, uint8_t* devCtrlCache,
        const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
    rtStream_t aicpuStreamValue = reinterpret_cast<rtStream_t>(aicpuStream);
    rtStream_t aicoreStreamValue = reinterpret_cast<rtStream_t>(aicoreStream);
    return DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(op->GetFunction(), inputList, outputList,
        aicpuStreamValue, aicoreStreamValue, streamSynchronize, op,
        reinterpret_cast<DevControlFlowCache*>(devCtrlCache), config);
#else
    (void)devCtrlCache;
    (void)op;
    (void)inputList;
    (void)outputList;
    (void)aicpuStream;
    (void)aicoreStream;
    (void)streamSynchronize;
    (void)devCtrlCache;
    (void)config;
    return 0;
#endif
}

int DeviceSynchronize(DeviceStream aicpuStream, DeviceStream aicoreStream) {
#ifdef BUILD_WITH_CANN
    rtStream_t aicpuStreamValue = reinterpret_cast<rtStream_t>(aicpuStream);
    rtStream_t aicoreStreamValue = reinterpret_cast<rtStream_t>(aicoreStream);
    return DeviceLauncher::DeviceSynchronize(aicpuStreamValue, aicoreStreamValue);
#else
    (void)aicpuStream;
    (void)aicoreStream;
    return 0;
#endif
}

int DeviceRunOnce(Function *function, uint8_t* hostCtrlCache, const DeviceLauncherConfig &config) {
    return DeviceLauncher::DeviceRunOnce(function, reinterpret_cast<DevControlFlowCache*>(hostCtrlCache), config);
}

int HasInplaceArgs(Function *function) {
    return DeviceLauncher::HasInplaceArgs(function);
}

void DeviceLauncherInit() {
    DeviceLauncherContext::Get().DeviceInit();
}

void DeviceLauncherFini() {
    DeviceLauncherContext::Get().DeviceFini();
}


void ChangeCaptureModeRelax() {
    DeviceLauncher::ChangeCaptureModeRelax();
}

void ChangeCaptureModeGlobal() {
    DeviceLauncher::ChangeCaptureModeGlobal();
}

static std::unordered_map<ExportedOperator *, std::shared_ptr<ExportedOperator>> exportedOperatorDict;

ExportedOperator *ExportedOperatorBegin() {
    std::shared_ptr<ExportedOperator> op = std::make_shared<ExportedOperator>();
    exportedOperatorDict[op.get()] = op;
    return op.get();
}

void ExportedOperatorEnd(ExportedOperator *op) {
    op->ResetFunction(Program::GetInstance().GetLastFunction());
}

void DataDumpInit() {
    if (IsPtoDataDumpEnabled()) {
        if (!AdxDataDumpServerInit) {
            ALOG_ERROR_F("AdxDataDumpServerInit function not found.");
            return;
        }
        ALOG_DEBUG_F("DataDumpServerInit is called \n");
        int sf = AdxDataDumpServerInit();
        if (sf != 0) {
            ALOG_ERROR_F("ERROR AdxDataDumpServerInit failed \n");
        }
    }
}

void DataDumpUnInit() {
    if (IsPtoDataDumpEnabled()) {
        if (!AdxDataDumpServerUnInit) {
            ALOG_ERROR_F("AdxDataDumpServerUnInit function not found.");
            return;
        }
        ALOG_DEBUG_F("DataDumpServerUnInit is called \n");
        int sf = AdxDataDumpServerUnInit();
        if (sf != 0) {
            ALOG_ERROR_F("AdxDataDumpServerUnInit is failed %d \n", sf);
        }
    }
}

void CopyDevToHost(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
#ifdef BUILD_WITH_CANN
    DeviceMemoryUtils().CopyFromDev((uint8_t *)hostTensor.GetAddr(), (uint8_t *)devTensor.GetAddr(), devTensor.GetDataSize());
#else
    (void)devTensor;
    (void)hostTensor;
#endif
}

void CopyHostToDev(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
#ifdef BUILD_WITH_CANN
    DeviceMemoryUtils().CopyToDev((uint8_t *)devTensor.GetAddr(), (uint8_t *)hostTensor.GetAddr(), devTensor.GetDataSize());
#else
    (void)devTensor;
    (void)hostTensor;
#endif
}

uint8_t* CopyHostToDev(uint8_t* data, uint64_t size) {
#ifdef BUILD_WITH_CANN
    return DeviceMemoryUtils(false).CopyToDev((uint8_t *)data, size, nullptr);
#else
    (void)data;
    (void)size;
    return nullptr;
#endif
}

DeviceGuard::DeviceGuard(int32_t devId) : nDevId(devId) {
#ifdef BUILD_WITH_CANN
    (void)rtGetDevice(&oDevId);
    if (nDevId != oDevId) {
        rtSetDevice(nDevId);
    }
#endif
}

DeviceGuard::~DeviceGuard() {
#ifdef BUILD_WITH_CANN
    if (nDevId != oDevId) {
        rtSetDevice(oDevId);
    }
#endif
}

AclModeGuard::AclModeGuard(aclmdlRICaptureMode tmode) : mode(tmode) {
#ifdef BUILD_WITH_CANN
    aclmdlRICaptureThreadExchangeMode(&mode);
#endif
}
AclModeGuard::~AclModeGuard() {
#ifdef BUILD_WITH_CANN
    aclmdlRICaptureThreadExchangeMode(&mode);
#endif
}

void DeviceLauncher::FillDeviceKernelArgs(std::vector<uint8_t> &devProgData, DeviceKernelArgs &kargs) {
#ifdef BUILD_WITH_CANN
    DeviceLauncherConfig config;
    CachedOperator cache;
    DeviceLauncherConfigFillDeviceInfo(config);
    DeviceInitTilingData(DeviceMemoryUtils(), kargs, devProgData, nullptr, config, &cache);
#else
    (void)devProgData;
    (void)kargs;
#endif
}

int64_t DeviceLauncher::GetL2Offset() {
#ifdef BUILD_WITH_CANN
    return machine::GetRA()->GetL2Offset();
#else
    return 0;
#endif
}

uint8_t *DeviceLauncher::CopyControlFlowCache(DevControlFlowCache *ctrlCache) {
#ifdef BUILD_WITH_CANN
    uint8_t *devCache = nullptr;
    auto cacheSize = ctrlCache->usedCacheSize;
    auto bufNum = DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT;

    int ret = rtMalloc((void **)&devCache, cacheSize * bufNum, RT_MEMORY_HBM, 0);
    if (devCache == nullptr) {
        ALOG_ERROR("control flow cache malloc failed");
        return nullptr;
    }

    for (int i = 0; i < bufNum; ++i) {
        ret = rtMemcpy(devCache + i * cacheSize, cacheSize, ctrlCache, cacheSize, RT_MEMCPY_HOST_TO_DEVICE);
        if (ret != 0) {
            ALOG_ERROR("control flow cache memcpy failed", ret);
            rtFree(devCache);
            return nullptr;
        }
    }
    return devCache;
#else
    (void)ctrlCache;
    return nullptr;
#endif
}

void DeviceLauncher::FreeControlFlowCache(uint8_t *ctrlCache) {
#ifdef BUILD_WITH_CANN
    if (ctrlCache != nullptr) {
        rtFree(ctrlCache);
    }
#else
    (void)ctrlCache;
#endif
}

bool DeviceLauncher::AddAicpuStream(aclrtStream aicoreStream, bool tripleStream) {
#ifdef BUILD_WITH_CANN
    auto ctrlStream = (aclrtStream)machine::GetRA()->GetCtrlStream();
    auto schedtream = (aclrtStream)machine::GetRA()->GetScheStream();
    aclmdlRI rtModel;
    aclmdlRICaptureStatus status = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    auto ret = aclmdlRICaptureGetInfo(aicoreStream, &status, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        return false;
    } else if (ret != ACL_SUCCESS) {
        ALOG_ERROR("get capture info failed: ", ret);
        return false;
    }
    if (status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        if (tripleStream) {
            rtStreamAddToModel(ctrlStream, rtModel);
        }
        rtStreamAddToModel(schedtream, rtModel);
        return true;
    }
    return false;
#else
    (void)aicoreStream;
    (void)tripleStream;
    return false;
#endif
}

void *DeviceLauncher::RegisterKernelBin(const std::vector<uint8_t> &kernelBinary) {
#ifdef BUILD_WITH_CANN
    void *hdl = nullptr;
    rtDevBinary_t binary = {
        .magic = RT_DEV_BINARY_MAGIC_ELF,
        .version = 0,
        .data = kernelBinary.data(),
        .length = kernelBinary.size(),
    };

    int ret = rtRegisterAllKernel(&binary, &hdl);
    if (ret != RT_ERROR_NONE) {
        ALOG_ERROR_F("register kernel failed, ret: %d", ret);
    }
    return hdl;
#else
    (void)kernelBinary;
    return nullptr;
#endif
}

void DeviceLauncher::UnregisterKernelBin(void *hdl) {
#ifdef BUILD_WITH_CANN
    int ret = rtDevBinaryUnRegister(hdl);
    if (ret != RT_ERROR_NONE) {
        ALOG_ERROR_F("unregister kernel failed, ret: %d", ret);
    }
#else
    (void)hdl;
#endif
}

int DeviceLauncher::LaunchAicpuKernel(rtAicpuArgsEx_t &rtArgs, bool tripleStream,
                                      bool debugEnable, [[maybe_unused]]Function *function) {
#ifdef BUILD_WITH_CANN
    auto ctrlStream = (aclrtStream)machine::GetRA()->GetCtrlStream();
    auto schedStream = (aclrtStream)machine::GetRA()->GetScheStream();
    auto &devRunner = DeviceRunner::Get();
    devRunner.GetHostProfInstance().SetProfFunction(function);
    if (debugEnable) {
        devRunner.SetDebugEnable();
    }
    int ret = 0;
    auto args = (AiCpuArgs *)rtArgs.args;
    const int nrAicpu = DeviceLauncher::GetDevProg(function)->devArgs.nrAicpu;
    if (tripleStream) {
        auto startTime = MsprofSysCycleTime();
        args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
        ret = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 2, &rtArgs, nullptr, ctrlStream, 0);
        devRunner.ReportHostProfInfo(startTime, 2, MSPROF_GE_TASK_TYPE_AI_CPU, false);
        if (ret != RT_ERROR_NONE) {
            return ret;
        }
        args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
        startTime = MsprofSysCycleTime();
        ret = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", nrAicpu, &rtArgs, nullptr, schedStream, 0);
        devRunner.ReportHostProfInfo(startTime, DeviceLauncher::GetDevProg(function)->devArgs.scheCpuNum, MSPROF_GE_TASK_TYPE_AI_CPU, false);
        return ret;
    } else {
        args->kArgs.parameter.runMode = RUN_UNIFIED_STREAM;
        auto startTime = MsprofSysCycleTime();
        ret = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", nrAicpu, &rtArgs, nullptr, schedStream, 0);
        devRunner.ReportHostProfInfo(startTime, nrAicpu, MSPROF_GE_TASK_TYPE_AI_CPU, false);
        return ret;
    }
#else
    (void)rtArgs;
    (void)tripleStream;
    (void)debugEnable;
    return 0;
#endif
}

int DeviceLauncher::LaunchAicoreKernel(
    aclrtStream aicoreStream, void *kernel, rtArgsEx_t &rtArgs, rtTaskCfgInfo_t &rtTaskCfg, bool debugEnable) {
#ifdef BUILD_WITH_CANN
    auto &devRunner = DeviceRunner::Get();
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    auto blockDim = dynamic::GetCfgBlockdim();
    auto startTime = MsprofSysCycleTime();
    auto ret = rtKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    devRunner.ReportHostProfInfo(startTime, blockDim, MSPROF_GE_TASK_TYPE_MIX_AIC, true);
    if (debugEnable) {
        auto scheStream = (aclrtStream)machine::GetRA()->GetScheStream();
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(scheStream, nullptr, aicoreStream);
        if (rc != 0) {
            ALOG_ERROR("sync failed");
            return rc;
        }
        devRunner.SynchronizeDeviceToHostProfData();
        ASSERT(machine::GetRA()->CheckAllSentinels());
    }
    return ret;
#else
    (void)aicoreStream;
    (void)kernel;
    (void)rtArgs;
    (void)rtTaskCfg;
    (void)debugEnable;
    return 0;
#endif
}
}
