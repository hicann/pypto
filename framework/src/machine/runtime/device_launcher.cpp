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

void DeviceLauncher::ChangeCaptureMode()
{
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;   // aclgraph does not support rtmemcpy / rtmemset, set to relaxed mode
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

int DeviceLauncher::RunWithProfile(rtStream_t aicoreStream, rtStream_t aicpuStream) {
    aclmdlRI rtModel = nullptr;
    bool isCapture = false;
    if (GetStreamCaptureInfo(aicoreStream, rtModel, isCapture) < 0) {
        return -1;
    }
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        if (isCapture) {
            ALOG_WARN("The swimlane function is not currently supported in CaptureMode. The contents of tilefwk_L1_prof_data may be empty.");
        }
        aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
        aclmdlRICaptureThreadExchangeMode(&mode);
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
        aclmdlRICaptureThreadExchangeMode(&mode);
        if (rc < 0) {
            return rc;
        }
        DeviceRunner::Get().SynchronizeDeviceToHostProfData();
    }
    return 0;
}

int DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator,
        const DeviceLauncherConfig &config) {
    bool isCapture = false;
    std::cout << "!!! Kernel Launch " << "\n";
    if (function != nullptr && function->GetDyndevAttribute() != nullptr) {
        DeviceRunner::SetBinData(function->GetDyndevAttribute()->kernelBinary);
    }
    /* 1.Add stream to capture model*/
    int rc = SetCaptureStream(aicoreStream, aicpuStream, isCapture);
    if (rc < 0) {
        return rc;
    }
    /* 2. Change capture mode to relaxed*/
    if (isCapture) {
        ChangeCaptureMode();
    }
    DeviceRunner::Get().SetCaptureFlag(isCapture);
    DeviceRunner::Get().GetHostProfInstance().SetProfFunction(function);
    rc = aclInit(nullptr);
    if (rc != 0 && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }

    CachedOperator cachedOperatorData;
    if (cachedOperator == nullptr) {
        // Not python cached operator mode, consider kernel reuse mode
        if (DeviceRunCacheKernelEnable(function)) {
            *CachedOperator::GetCfgDataDevAddrHolder(&cachedOperatorData) = DeviceRunCacheKernelGet(function);
            cachedOperator = &cachedOperatorData;
        }
    }
    CheckDeviceId();
    AstKernelArgs kArgs;
    DeviceLauncherConfigFillDeviceInfo(config);
    DeviceInitTilingData(DeviceMemoryUtils(), kArgs, function->GetDyndevAttribute()->devProgBinary, config, cachedOperator);
    DeviceRunCacheKernelSet(function, (uint8_t *)kArgs.cfgdata);
    DeviceInitKernelInOuts(DeviceMemoryUtils(), kArgs, inputList, outputList,
        function->GetDyndevAttribute()->disableL2List, config.isGETensorList);
    rc = DeviceRunner::Get().RegisterKernelBin(&(*reinterpret_cast<rtBinHandle *>(CachedOperator::GetBinHandleHolder(cachedOperator))));
    if (rc < 0) {
        ALOG_ERROR_F("Register kernel bin failed.");
        return rc;
    }
    rc = DeviceRunner::Get().DynamicLaunch(aicpuStream, nullptr, aicoreStream, 0, &kArgs, config.blockdim, config.aicpuNum);
    if (rc < 0) {
        return rc;
    }
    rc = RunWithProfile(aicoreStream, aicpuStream);
    if (rc < 0) {
        return rc;
    }
    if (streamSynchronize) {
        rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
    }
    return rc;
}

int DeviceLauncher::DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
    return rc;
}
#endif

int DeviceLauncher::DeviceRunOnce(Function *function, const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    auto aicpuStream = machine::GetRA()->GetScheStream();
    auto aicoreStream = machine::GetRA()->GetStream();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = BuildInputOutputFromHost(DeviceMemoryUtils(), inputDataList, outputDataList);
    int rc = DeviceLaunchOnceWithDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, aicpuStream, aicoreStream, true, nullptr, config);
    CopyFromDev(DeviceMemoryUtils(), outputDataList);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        CopyFromDev(DeviceMemoryUtils(), inputDataList);
    }
    return rc;
#else
    (void)function;
    (void)config;
    return 0;
#endif
}

struct DeviceRunCacheInfo {
    /* By default: devProg cache is enabled */
    bool devProgEnabled{true};
    uint8_t *devProgAddr{nullptr};
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
    dict[func].devProgAddr = devProg;
}
uint8_t *DeviceLauncher::DeviceRunCacheKernelGet(Function *func) {
    if (!DeviceRunCacheKernelEnable(func)) {
        return nullptr;
    }
    auto &dict = DeviceRunCacheInfoDict();
    return dict[func].devProgAddr;
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
        ExportedOperator *op, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        DeviceStream aicpuStream, DeviceStream aicoreStream, bool streamSynchronize,
        const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
    rtStream_t aicpuStreamValue = reinterpret_cast<rtStream_t>(aicpuStream);
    rtStream_t aicoreStreamValue = reinterpret_cast<rtStream_t>(aicoreStream);
    return DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(op->GetFunction(), inputList, outputList, aicpuStreamValue, aicoreStreamValue, streamSynchronize, op, config);
#else
    (void)op;
    (void)inputList;
    (void)outputList;
    (void)aicpuStream;
    (void)aicoreStream;
    (void)streamSynchronize;
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

int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config) {
    return DeviceLauncher::DeviceRunOnce(function, config);
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

static std::unordered_map<ExportedOperator *, std::shared_ptr<ExportedOperator>> exportedOperatorDict;

ExportedOperator *ExportedOperatorBegin() {
    std::shared_ptr<ExportedOperator> op = std::make_shared<ExportedOperator>();
    exportedOperatorDict[op.get()] = op;
    return op.get();
}

void ExportedOperatorEnd(ExportedOperator *op) {
    op->ResetFunction(Program::GetInstance().GetLastFunction());
}

DeviceTensorData CopyDevToHost(const DeviceTensorData &tensorData) {
#ifdef BUILD_WITH_CANN
    uint8_t *data = (uint8_t *)malloc(tensorData.GetDataSize());
    auto rawData = RawTensorData::CreateTensor(tensorData.GetDataType(), tensorData.GetShape(), data);
    rawData->SetDevPtr((uint8_t *)tensorData.GetAddr());
    DeviceMemoryUtils().CopyFromDev(*rawData);
    auto hostTensor =std::make_shared<DeviceTensorData>(tensorData.GetDataType(), (uintdevptr_t)rawData->data(), tensorData.GetShape());
    return *hostTensor;
#else
    auto hostTensor =std::make_shared<DeviceTensorData>(tensorData.GetDataType(), 0, tensorData.GetShape());
    return *hostTensor;
#endif
}

}
