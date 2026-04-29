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

#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/adump_api.h"
#include "interface/utils/op_info_manager.h"

#include "machine/runtime/device_launcher.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/host/backend.h"
#include "machine/host/perf_analysis.h"
#include "tilefwk/error_code.h"

struct process_sign {
    pid_t tgid;
    char sign[49]; // 49 is PROCESS_SIGN_LENGTH
    char resv[4];  // 4 is PROCESS_RESV_LENGTH
};
extern "C" __attribute__((weak)) int AdxDataDumpServerUnInit();
extern "C" __attribute__((weak)) int AdxDataDumpServerInit();
extern "C" __attribute__((weak)) int drvGetProcessSign(process_sign* sign);

namespace npu::tile_fwk::dynamic {
namespace {
constexpr uint32_t kMinDefaultDim = 20;
// AIC:AIV的比例系数
constexpr uint32_t AICAIVRATIO = 2;
} // namespace
int GetCfgBlockdim()
{
    auto blk = Platform::Instance().GetSoc().GetAICoreNum();
    blk = blk > 0 ? blk : kMinDefaultDim;

    // 通过GetMaxBlockdim接口获取设置的最大核数，如果设置的最大核数大于硬件物理最大核数时，控核不生效
    // 如果未进行控核，GetMaxBlockdim接口将通过AclRtGetStreamResLimit函数返回硬件物理最大核数
    auto maxBlk = GetMaxBlockdim();
    blk = (maxBlk > 0 && maxBlk < static_cast<int>(blk)) ? maxBlk : blk;
    MACHINE_LOGD("Get blockdim[%zu].", blk);
    return blk;
}

int GetMaxBlockdim()
{
    uint32_t cubeBlockDim = 0;
    uint32_t vectorBlockDim = 0;
    // 若未进行控核，AclRtGetStreamResLimit返回的是满核
    auto aicoreStream = machine::GetRA()->GetCurrentStream();
    AclRtGetStreamResLimit(aicoreStream, AclRtDevResLimitType::CUBE_CORE, &cubeBlockDim);
    AclRtGetStreamResLimit(aicoreStream, AclRtDevResLimitType::VECTOR_CORE, &vectorBlockDim);
    // 若不满足AIC和AIV的比例，手动处理成为符合AIC和AIV的比例最大值
    if (vectorBlockDim != cubeBlockDim * AICAIVRATIO) {
        auto rtsMaxBlockDim = std::min(cubeBlockDim, vectorBlockDim / AICAIVRATIO);
        MACHINE_LOGW(
            "The cubeBlockDim[%u] and vectorBlockDim[%u] do not conform to the 1: %u ratio of AIC and AIV, "
            "and will be set to values that conform to the ratio of AIC and AIV. "
            "The cubeBlockDim and vectorBlockDim are set at %u and %u",
            cubeBlockDim, vectorBlockDim, AICAIVRATIO, rtsMaxBlockDim, rtsMaxBlockDim * AICAIVRATIO);
        return rtsMaxBlockDim;
    } else {
        return cubeBlockDim;
    }
}

void (*forceLinkLibraryCompiler)() = &npu::tile_fwk::ForceLinkLibraryCompiler;

DeviceLauncherContext& DeviceLauncherContext::Get()
{
    static DeviceLauncherContext context;
    return context;
}

std::vector<uint8_t> DeviceLauncher::tensorInfo_(kDefaultTensorinfoSize);
bool DeviceLauncher::captureMode_ = false;

static const std::unordered_map<AclMdlRICaptureStatus, std::function<void(bool&)>> captureStatusHandlers = {
    {npu::tile_fwk::AclMdlRICaptureStatus::ACTIVE, [](bool& isCapture) { isCapture = true; }},
    {npu::tile_fwk::AclMdlRICaptureStatus::NONE,
     [](bool& isCapture) {
         (void)isCapture;
         MACHINE_LOGD("GetStreamCaptureInfo: status NONE");
     }},
    {npu::tile_fwk::AclMdlRICaptureStatus::INVALIDATED, [](bool& isCapture) {
         (void)isCapture;
         MACHINE_LOGD("GetStreamCaptureInfo: status invalidated");
     }}};

int DeviceLauncher::GetStreamCaptureInfo(RtStream aicoreStream, AclMdlRI& rtModel, bool& isCapture)
{
    AclMdlRICaptureStatus captureStatus = AclMdlRICaptureStatus::NONE;
    AclError ret = AclMdlRICaptureGetInfo(aicoreStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        MACHINE_LOGW("Stream capture not support");
        return 0;
    } else if (ret != ACLRT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_CAPTURE_FAILED, "AclMdlRICaptureGetInfo failed, return[%d]", ret);
        return -1;
    }

    auto it = captureStatusHandlers.find(captureStatus);
    if (it != captureStatusHandlers.end()) {
        it->second(isCapture);
    } else {
        MACHINE_LOGE(RtErr::RT_CAPTURE_FAILED, "GetStreamCaptureInfo get unsupport capture status");
        return -1;
    }
    MACHINE_LOGI("capture mode[%d]", isCapture);
    return 0;
}

void DeviceLauncher::ChangeCaptureModeRelax()
{
    AclMdlRICaptureMode mode =
        AclMdlRICaptureMode::RELAXED; // aclgraph does not support rtmemcpy / rtmemset, set to relaxed mode
    AclMdlRICaptureThreadExchangeMode(&mode);
}

void DeviceLauncher::ChangeCaptureModeGlobal()
{
    AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
    AclMdlRICaptureThreadExchangeMode(&mode);
}

int DeviceLauncher::SetCaptureStream(RtStream aicoreStream, RtStream aicpuStream, bool& isCapture)
{
    AclMdlRI rtModel = nullptr;

    if (GetStreamCaptureInfo(aicoreStream, rtModel, isCapture) < 0) {
        return -1;
    }

    if (isCapture) {
        if (rtModel == nullptr) {
            MACHINE_LOGE(DevCommonErr::NULLPTR, "rtModel is null!");
            return -1;
            ;
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
        DeviceRunner::Get().SynchronizeDeviceToHostProfData();
        DeviceRunner::Get().ResetPerData();
    }
    return 0;
}

int DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, bool streamSynchronize,
    CachedOperator* cachedOperator, DevControlFlowCache* inputDevCtrlCache, const DeviceLauncherConfig& config)
{
    bool isCapture = false;
    MACHINE_LOGI("Kernel Launch");

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
    rc = AclInit(nullptr);
    if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
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
    DeviceMemoryUtils devMemoryUtilis;
    DeviceInitDistributedContext(devMemoryUtilis, dynAttr->commGroupNames, kArgs);

    HOST_PERF_TRACE(TracePhase::RunDevEnvReady);
    DeviceInitTilingData(devMemoryUtilis, kArgs, dynAttr->devProgBinary, inputDevCtrlCache, config, cachedOperator);
    HOST_PERF_TRACE(TracePhase::RunDevInitTiling);

    DeviceRunCacheKernelSet(function, (uint8_t*)kArgs.cfgdata);
    DeviceInitKernelInOuts(devMemoryUtilis, kArgs, inputList, outputList, dynAttr->disableL2List);

    HOST_PERF_TRACE(TracePhase::RunDevInitInOutTensor);

    rc = DeviceRunner::Get().RegisterKernelBin(
        &(*reinterpret_cast<RtBinHandle*>(CachedOperator::GetBinHandleHolder(cachedOperator))),
        cachedOperator == nullptr ? nullptr : &(function->GetDyndevAttribute()->kernelBinary));
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Register kernel bin failed.");
        return rc;
    }

    HOST_PERF_TRACE(TracePhase::RunDevRegistKernelBin);

    DataDumpInit();
    rc = DeviceRunner::Get().DynamicLaunch(
        aicpuStream, ctrlStream, aicoreStream, 0, &kArgs, config.blockdim, config.aicpuNum);
    if (rc < 0) {
        return rc;
    }
    rc = RunWithProfile(aicoreStream, aicpuStream, isCapture);
    if (rc < 0) {
        return rc;
    }
    if (streamSynchronize) {
        rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, ctrlStream, aicoreStream);
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, machine::GetRA()->CheckAllSentinels());
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
    auto aicpuStream = machine::GetRA()->GetScheStream();
    auto aicoreStream = machine::GetRA()->GetStream();
    auto ctrlStream = machine::GetRA()->GetCtrlStream();
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

    int rc = DeviceLaunchOnceWithDeviceTensorData(
        function, inputDeviceDataList, outputDeviceDataList, aicpuStream, ctrlStream, aicoreStream, true, nullptr,
        reinterpret_cast<DevControlFlowCache*>(devCtrlCache), config);
    CopyFromDev(DeviceMemoryUtils(), outputDataList);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        CopyFromDev(DeviceMemoryUtils(), inputDataList);
    }
    devMemory.Free(devCtrlCache);
    return rc;
}

struct DeviceRunCacheInfo {
    /* By default: devProg cache is enabled */
    bool devProgEnabled{true};
    CachedOperator cacheOperator;
};
static std::unordered_map<Function*, DeviceRunCacheInfo>& DeviceRunCacheInfoDict()
{
    static std::unordered_map<Function*, DeviceRunCacheInfo> cacheInfoDict;
    return cacheInfoDict;
}
void DeviceLauncher::DeviceRunCacheKernelEnable(Function* func, bool enabled)
{
    auto& dict = DeviceRunCacheInfoDict();
    dict[func].devProgEnabled = enabled;
}
bool DeviceLauncher::DeviceRunCacheKernelEnable(Function* func)
{
    auto& dict = DeviceRunCacheInfoDict();
    return dict[func].devProgEnabled;
}
void DeviceLauncher::DeviceRunCacheKernelSet(Function* func, uint8_t* devProg)
{
    if (!DeviceRunCacheKernelEnable(func)) {
        return;
    }
    auto& dict = DeviceRunCacheInfoDict();
    *CachedOperator::GetCfgDataDevAddrHolder(&(dict[func].cacheOperator)) = devProg;
}

uint8_t* DeviceLauncher::DeviceRunCacheKernelGet(Function* func)
{
    if (!DeviceRunCacheKernelEnable(func)) {
        return nullptr;
    }
    auto& dict = DeviceRunCacheInfoDict();
    return *CachedOperator::GetCfgDataDevAddrHolder(&(dict[func].cacheOperator));
}

CachedOperator* DeviceLauncher::DeviceRunCacheOperatorGet(Function* func)
{
    if (!DeviceRunCacheKernelEnable(func)) {
        return nullptr;
    }
    auto& dict = DeviceRunCacheInfoDict();
    return &(dict[func].cacheOperator);
}

DeviceStream DeviceGetAicpuStream()
{
    RtStream aicpuStreamValue = machine::GetRA()->GetScheStream();
    return reinterpret_cast<DeviceStream>(aicpuStreamValue);
}

DeviceStream DeviceGetCtrlStream()
{
    RtStream ctrlStreamValue = machine::GetRA()->GetCtrlStream();
    return reinterpret_cast<DeviceStream>(ctrlStreamValue);
}

DeviceStream DeviceGetAicoreStream()
{
    RtStream aicoreStreamValue = machine::GetRA()->GetStream();
    return reinterpret_cast<DeviceStream>(aicoreStreamValue);
}

int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(
    ExportedOperator* op, const std::vector<DeviceTensorData>& inputList,
    const std::vector<DeviceTensorData>& outputList, DeviceStream aicpuStream, DeviceStream ctrlStream,
    DeviceStream aicoreStream, bool streamSynchronize, uint8_t* devCtrlCache, const DeviceLauncherConfig& config)
{
    RtStream aicpuStreamValue = reinterpret_cast<RtStream>(aicpuStream);
    RtStream ctrlStreamValue = reinterpret_cast<RtStream>(ctrlStream);
    RtStream aicoreStreamValue = reinterpret_cast<RtStream>(aicoreStream);
    return DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
        op->GetFunction(), inputList, outputList, aicpuStreamValue, ctrlStreamValue, aicoreStreamValue,
        streamSynchronize, op, reinterpret_cast<DevControlFlowCache*>(devCtrlCache), config);
}

int DeviceSynchronize(DeviceStream aicpuStream, DeviceStream aicoreStream)
{
    RtStream aicpuStreamValue = reinterpret_cast<RtStream>(aicpuStream);
    RtStream aicoreStreamValue = reinterpret_cast<RtStream>(aicoreStream);
    return DeviceLauncher::DeviceSynchronize(aicpuStreamValue, aicoreStreamValue);
}

int DeviceRunOnce(Function* function, uint8_t* hostCtrlCache, const DeviceLauncherConfig& config)
{
    return DeviceLauncher::DeviceRunOnce(function, reinterpret_cast<DevControlFlowCache*>(hostCtrlCache), config);
}

int HasInplaceArgs(Function* function) { return DeviceLauncher::HasInplaceArgs(function); }

void DeviceLauncherInit() { DeviceLauncherContext::Get().DeviceInit(); }

void DeviceLauncherFini() { DeviceLauncherContext::Get().DeviceFini(); }

void ChangeCaptureModeRelax() { DeviceLauncher::ChangeCaptureModeRelax(); }

void ChangeCaptureModeGlobal() { DeviceLauncher::ChangeCaptureModeGlobal(); }

static std::unordered_map<ExportedOperator*, std::shared_ptr<ExportedOperator>> exportedOperatorDict;

ExportedOperator* ExportedOperatorBegin()
{
    std::shared_ptr<ExportedOperator> op = std::make_shared<ExportedOperator>();
    exportedOperatorDict[op.get()] = op;
    return op.get();
}

void ExportedOperatorEnd(ExportedOperator* op) { op->ResetFunction(Program::GetInstance().GetLastFunction()); }

void DataDumpInit()
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

void DataDumpUnInit()
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
    constexpr int32_t GE_FORAMT_ND = 2;
    constexpr int32_t GE_FORAMT_NZ = 29;
    switch (format) {
        case TileOpFormat::TILEOP_ND:
            return GE_FORAMT_ND;
        case TileOpFormat::TILEOP_NZ:
            return GE_FORAMT_NZ;
        default:
            throw std::invalid_argument("Unknown Format");
    }
}

void DumpIOTensorsWithCann(AclRtStream stream, std::vector<DeviceTensorData>& tensors, const std::string& funcName)
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

uint32_t GetProcessId()
{
    if (drvGetProcessSign != nullptr) {
        process_sign processSign;
        auto ret = drvGetProcessSign(&processSign);
        if (ret == 0) {
            MACHINE_LOGD("Got process sign from drv: tgid=%d", processSign.tgid);
            return static_cast<uint32_t>(processSign.tgid);
        }
        MACHINE_LOGW("drvGetProcessSign failed, ret=%d, falling back to getpid()", ret);
    } else {
        MACHINE_LOGW("drvGetProcessSign is nullptr, falling back to getpid()");
    }

    uint32_t pid = static_cast<uint32_t>(getpid());
    MACHINE_LOGD("Using getpid(): pid=%u", pid);
    return pid;
}

void CopyDevToHost(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor)
{
    DeviceMemoryUtils().CopyFromDev(
        (uint8_t*)hostTensor.GetAddr(), (uint8_t*)devTensor.GetAddr(), devTensor.GetDataSize());
}

void CopyHostToDev(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor)
{
    DeviceMemoryUtils().CopyToDev(
        (uint8_t*)devTensor.GetAddr(), (uint8_t*)hostTensor.GetAddr(), devTensor.GetDataSize());
}

uint8_t* CopyHostToDev(uint8_t* data, uint64_t size)
{
    return DeviceMemoryUtils(false).CopyToDev((uint8_t*)data, size, nullptr);
}

DeviceGuard::DeviceGuard(int32_t devId) : nDevId(devId)
{
    (void)RuntimeGetDevice(&oDevId);
    if (nDevId != oDevId) {
        RuntimeSetDevice(nDevId);
    }
}

DeviceGuard::~DeviceGuard()
{
    if (nDevId != oDevId) {
        RuntimeSetDevice(oDevId);
    }
}

AclModeGuard::AclModeGuard(AclMdlRICaptureMode tmode) : mode(tmode)
{
    AclMdlRICaptureThreadExchangeMode(&mode);
}
AclModeGuard::~AclModeGuard()
{
    AclMdlRICaptureMode mod = AclMdlRICaptureMode::GLOBAL;
    AclMdlRICaptureThreadExchangeMode(&mod);
}

void DeviceLauncher::FillDeviceKernelArgs(
    std::vector<uint8_t>& devProgData, DeviceKernelArgs& kargs, const std::vector<std::string>& groupNames)
{
    DeviceLauncherConfig config;
    CachedOperator cache;
    DeviceLauncherConfigFillDeviceInfo(config);
    DeviceMemoryUtils deviceMemoryUtils;
    DeviceInitTilingData(deviceMemoryUtils, kargs, devProgData, nullptr, config, &cache);
    DeviceInitDistributedContext(deviceMemoryUtils, groupNames, kargs);
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
        ret = RuntimeMemcpy(devCache + i * cacheSize, cacheSize, ctrlCache, cacheSize, RtMemcpyKind::HOST_TO_DEVICE);
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

void DeviceLauncher::AddAicpuStream(AclMdlRI& rtModel)
{
    auto ctrlStream = (AclRtStream)machine::GetRA()->GetCtrlStream();
    auto schedtream = (AclRtStream)machine::GetRA()->GetScheStream();

    if (IsCaptureMode()) {
        RuntimeStreamAddToModel(ctrlStream, rtModel);
        RuntimeStreamAddToModel(schedtream, rtModel);
    }
}

void DeviceLauncher::SaveStream(AclRtStream aicoreStream)
{
    // 存储 current stream，后续控核接口需使用current stream
    machine::GetRA()->SetCurrentStream(aicoreStream);
}

void DeviceLauncher::GetCaptureInfo(AclRtStream aicoreStream, AclMdlRI& rtModel)
{
    SetCaptureMode(false);
    AclMdlRICaptureStatus status = AclMdlRICaptureStatus::NONE;
    auto ret = AclMdlRICaptureGetInfo(aicoreStream, &status, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        return;
    } else if (ret != ACLRT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_CAPTURE_FAILED, "get capture info failed: %d", ret);
        return;
    }
    if (status == AclMdlRICaptureStatus::ACTIVE) {
        SetCaptureMode(true);
        MACHINE_LOGI("The current mode is capture mode");
    }
}

void DeviceLauncher::SetCaptureMode(bool captureMode) { captureMode_ = captureMode; }

bool DeviceLauncher::IsCaptureMode() { return captureMode_; }

void* DeviceLauncher::RegisterKernelBin(const std::vector<uint8_t>& kernelBinary)
{
    void* hdl = nullptr;
    RtDevBinary binary = {
        .magic = RT_DEV_BINARY_MAGIC_ELF,
        .version = 0,
        .data = kernelBinary.data(),
        .length = kernelBinary.size(),
    };

    int ret = RuntimeRegisterAllKernel(&binary, &hdl);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "register kernel failed, ret: %d", ret);
    }
    return hdl;
}

void DeviceLauncher::UnregisterKernelBin(void* hdl)
{
    int ret = RuntimeDevBinaryUnRegister(hdl);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_REGISTER_FAILED, "unregister kernel failed, ret: %d", ret);
    }
}

void DeviceLauncher::SetDevPerfAddr(
    [[maybe_unused]] const bool& debugEnable, [[maybe_unused]] const bool& isCaptureMode)
{
    auto& devRunner = DeviceRunner::Get();
    if (debugEnable || devRunner.GetEnableDumpDevPref() || devRunner.GetHostProfInstance().GetProfType() == 1) {
        if (isCaptureMode) {
            ChangeCaptureModeRelax();
        }
        devRunner.SetDebugEnable();
        if (isCaptureMode) {
            ChangeCaptureModeGlobal();
        }
    }
}

int DeviceLauncher::LaunchSyncTask(AclRtStream aicoreStream, bool isCaptureMode)
{
    if (isCaptureMode) {
        return 0;
    }
    auto schedStream = machine::GetRA()->GetScheStream();
    auto ctrlStream = machine::GetRA()->GetCtrlStream();
    return DeviceRunner::Get().RunPreSync(schedStream, ctrlStream, (RtStream)aicoreStream);
}

int DeviceLauncher::LaunchAicpuKernel(
    RtAicpuArgsEx& rtArgs, [[maybe_unused]] bool debugEnable, [[maybe_unused]] Function* function)
{
    auto ctrlStream = (AclRtStream)machine::GetRA()->GetCtrlStream();
    auto schedStream = (AclRtStream)machine::GetRA()->GetScheStream();
    auto& devRunner = DeviceRunner::Get();
    devRunner.GetHostProfInstance().SetProfFunction(function);
    int ret = 0;
    auto args = (AiCpuArgs*)rtArgs.args;
    const int nrAicpu = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.nrAicpu);
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
    AclRtStream aicoreStream, void* kernel, RtArgsEx& rtArgs, RtTaskCfgInfo& rtTaskCfg, bool debugEnable)
{
    auto& devRunner = DeviceRunner::Get();
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    auto blockDim = dynamic::GetCfgBlockdim();
    auto startTime = MspfSysCycleTime();
    auto ret = RuntimeKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    devRunner.ReportHostProfInfo(aicoreStream, startTime, blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);
    if (debugEnable) {
        auto scheStream = (AclRtStream)machine::GetRA()->GetScheStream();
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(scheStream, nullptr, aicoreStream);
        if (rc != 0) {
            MACHINE_LOGE(HostLauncherErr::SYNC_FAILED, "sync failed");
            return rc;
        }
        devRunner.DumpAiCoreExecutionTimeData();
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, machine::GetRA()->CheckAllSentinels());
    }
    if (IsPtoDataDumpEnabled()) {
        auto scheStream = (AclRtStream)machine::GetRA()->GetScheStream();
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
