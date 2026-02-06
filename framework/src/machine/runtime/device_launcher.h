/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_MACHINE_DEVICE_LAUNCHER_H
#define SRC_MACHINE_DEVICE_LAUNCHER_H

#include <cstdint>
#include <cinttypes>

#ifdef BUILD_WITH_CANN
#include "machine/runtime/device_runner.h"
#endif

#include "machine/runtime/device_launcher_binding.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_tensor_creator.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/runtime/device_memory_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/platform.h"
#include "machine/runtime/distributed_context.h"

#ifndef BUILD_WITH_CANN
enum aclmdlRICaptureMode {};
using rtStream_t = uint64_t;
using aclmdlRI = void *;
using aclrtStream = void *;
typedef struct tagRtArgsEx rtArgsEx_t;
typedef struct tagRtAicpuArgsEx rtAicpuArgsEx_t;
typedef struct tagRtTaskCfgInfo rtTaskCfgInfo_t;
#endif

namespace npu::tile_fwk::dynamic {

struct AiCpuArgs {
    DeviceKernelArgs kArgs;
    const char kernelName[32] = {"DynTileFwkKernelServer"};
    const char soName[32] = {"libaicpu_extend_kernels.so"};
    const char opName[32] = {""};
};

int GetCfgBlockdim();

class DeviceLauncherContext {
public:
    void DeviceInit() {
        // 使能 Aihac 后端
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
#ifdef ENABLE_STEST_BINARY_CACHE
        // BinaryCache
        oriEnableBinaryCache = config::GetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
#endif
#ifdef ENABLE_STEST_DUMP_JSsON
        oriEnableDumpJson = config::GetPassConfig(KEY_PRINT_GRAPH, oriEnableDumpJson);
        config::GetPassConfig(KEY_PRINT_GRAPH, true);
#endif
        // Reset Program

        Program::GetInstance().Reset();
        ProgramData::GetInstance().Reset();
    }

    void DeviceFini() {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
#ifdef ENABLE_STEST_BINARY_CACHE
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
#endif
#ifdef ENABLE_STEST_DUMO_JSON
        config::SetHostConfig(KEY_PRINT_GRAPH, oriEnablePrintJson);
#endif
    }
    static DeviceLauncherContext &Get();

protected:
    bool oriEnableAihacBackend = false;
#ifdef ENABLE_STEST_BINARY_CACHE
    bool oriEnableBinaryCache = false;
#endif
#ifdef ENABLE_STEST_DUMO_JSON
    bool oriEnableDumpJson = false;
#endif
};

class DeviceLauncher {
public:
    static constexpr uint32_t kDefaultAicNum = 25;
    static constexpr uint32_t kDefaultAivNum = 50;
    static constexpr uint32_t kDefaultTensorinfoSize = 16384;
    static DevAscendProgram *GetDevProg(Function *func) {
        return reinterpret_cast<DevAscendProgram *>(func->GetDyndevAttribute()->devProgBinary.data());
    }

    static bool HasInplaceArgs(Function *function) {
        return GetDevProg(function)->outputInplaceSlotList.size() != 0;
    }

    static void DeviceLauncherConfigFillDeviceInfo(const DeviceLauncherConfig &config) {
        DeviceLauncherConfig &devConfig = const_cast<DeviceLauncherConfig &>(config);
#ifdef BUILD_WITH_CANN
        int maxBlockDim = GetCfgBlockdim();
        int maxAicpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum() - 1);
#else
        int maxBlockDim = 25; // 25:maxblockDim
        int maxAicpuNum = 5; // 5:maxaicpuNUm
#endif
        if (devConfig.blockdim == 0 || devConfig.blockdim > maxBlockDim) {
            devConfig.blockdim = maxBlockDim;
        }

        if (devConfig.aicpuNum == 0 || devConfig.aicpuNum > maxAicpuNum) {
            devConfig.aicpuNum = maxAicpuNum;
        }
    }

    template<typename DeviceMemoryTy>
    static void AssignMetaAddr(DeviceMemoryTy devMem, DeviceKernelArgs &kArgs, DevAscendProgram *devProg, CachedOperator *cachedOperator) {
        (void)kArgs;

        FillDeviceRuntimeOffset(devProg, DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT);
        size_t runtimeDataSize = devProg->GetDeviceRuntimeOffset().size;
        size_t runtimeDataCount = devProg->GetDeviceRuntimeOffset().count;
        size_t runtimeDataRingBufferSize = RuntimeDataRingBufferHead::GetRingBufferSize(runtimeDataSize, runtimeDataCount);
        uint64_t runtimeDataRingBufferAddr = (uint64_t)devMem.AllocDev(runtimeDataRingBufferSize, CachedOperator::GetMetaDataDevAddrHolder(cachedOperator));
        devProg->devArgs.runtimeDataRingBufferAddr = runtimeDataRingBufferAddr;

        uint64_t generalSize = devProg->memBudget.metadata.general;
        uint64_t stitchPoolSize = devProg->memBudget.metadata.stitchPool;
        ALOG_DEBUG_F("generalSize:%lu stitchPoolSize:%lu generalOffset:%lx stitchPoolOffset:%lx.", generalSize, stitchPoolSize,
            devProg->deviceRuntimeOffset.generalOffset, devProg->deviceRuntimeOffset.stitchPoolOffset);
        return;
    }

    // Prepare device program scheduling and memory budget related args (keeps <= 50 lines)
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ASSERT(config.blockdim != 0) << "Invalid blockdim: " << config.blockdim << ", must not be zero";
        devProg->devArgs.taskId = 0;
        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.nrValidAic = config.blockdim;
        devProg->devArgs.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;

        int aiCpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum()) - 1;
        devProg->devArgs.scheCpuNum = CalcSchAicpuNumByBlockDim(config.blockdim, aiCpuNum, devProg->devArgs.archInfo);
        config.aicpuNum = devProg->devArgs.scheCpuNum + dynamic::MAX_OTHER_AICPU_NUM;
        devProg->devArgs.nrAicpu = config.aicpuNum;
#ifdef BUILD_WITH_CANN
        if (isDevice) {
            devProg->devArgs.validGetPgMask = DeviceRunner::Get().GetValidGetPgMask();
        }
#endif
        devProg->devArgs.disableSync = config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_NO_DEVICE_TENSOR_DEPEND ? 1 : 0;
        ALOG_DEBUG_F("Set aicore blockdim:%d aicpu blockdim:%d.", config.blockdim, config.aicpuNum);

        devProg->devArgs.enableCtrl = 1; // need set 0 if use custom cpu launch ctrl cpu
        if (config.dynWorkspaceSize != 0) {
            ALOG_ERROR_F("[Deprecated] User provided dynamic workspace: %" PRId64, config.dynWorkspaceSize);
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = std::max(
                static_cast<int64_t>(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem),
                AlignUp(config.dynWorkspaceSize, TENSOR_ADDR_ALIGNMENT));
        }
#ifdef BUILD_WITH_CANN
        if (isDevice) {
            DeviceRunner::Get().InitMetaData(devProg->devArgs);
        }
#endif
        devProg->workspaceSize = devProg->memBudget.Total();
        ALOG_INFO_F("workspaceSize=%lu, tensor=%lu, metadata=%lu, aicoreSpillen=%lu, debug.DumpTensor=%lu",
            devProg->workspaceSize, devProg->memBudget.tensor.Total(), devProg->memBudget.metadata.Total(),
            devProg->memBudget.aicoreSpilled, devProg->memBudget.debug.dumpTensor);
        ALOG_INFO_F("Tensor:rootInner=%lu, devTaskInnerOutCasts=%lu, slotted=%lux%lu(slots).",
            devProg->memBudget.tensor.rootInner,
            devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts, devProg->memBudget.tensor.MaxOutcastMem(),
            devProg->memBudget.tensor.devTaskBoundaryOutcastNum);
    }

    // Fill metadata and kArgs (templated because it uses DeviceMemoryTy) (keeps <= 50 lines)
    template<typename DeviceMemoryTy>
    static void FillKernelMeta(DeviceMemoryTy devMem, DeviceKernelArgs &kArgs, DevAscendProgram *devProg,
            const std::vector<uint8_t> &devProgData, bool isCtrlCacheRecording, const DeviceLauncherConfig &config, CachedOperator *cachedOperator) {
        AssignMetaAddr(devMem, kArgs, devProg, cachedOperator);
        devProg->l2CacheOffset = devMem.GetL2Offset();
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        if (isCtrlCacheRecording) {
            kArgs.cfgdata = (int64_t *)devProg;
        } else if (CachedOperator::GetCfgDataDevAddrHolder(cachedOperator) && *CachedOperator::GetCfgDataDevAddrHolder(cachedOperator)) {
            /* Already copied, do not copy again. */
            kArgs.cfgdata = (int64_t *)*CachedOperator::GetCfgDataDevAddrHolder(cachedOperator);
        } else {
            kArgs.cfgdata = (int64_t *)devMem.CopyToDev(devProgData, CachedOperator::GetCfgDataDevAddrHolder(cachedOperator));
        }
        kArgs.machineConfig = devProg->devArgs.machineConfig;
        if (config::GetPlatformConfig(KEY_ENABLE_PROF_FUNC, false)) {
            kArgs.toSubMachineConfig.profConfig.Add(ProfConfig::AICPU_FUNC);
        }
        if (config::GetPlatformConfig(KEY_ENABLE_PROF_AICORE_TIME, false) || config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL)  {
            kArgs.toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_TIME);
        }
        if (config::GetPlatformConfig(KEY_ENABLE_PROF_AICORE_PMU, false)) {
            kArgs.toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_PMU);
        }
        devProg->devArgs.toSubMachineConfig = kArgs.toSubMachineConfig;
    }

    static void PrepareHcclContext(const std::vector<uint64_t> &hcclContext, DevAscendProgram *devProg) {
        ASSERT(devProg->commGroupNum == hcclContext.size())
            << "commGroupNum mismatch. commGroupNum = "
            <<devProg->commGroupNum << ", hcclContext size = " << hcclContext.size();
        ASSERT(devProg->commGroupNum <= (sizeof(devProg->hcclContext) / sizeof(uint64_t)))
            << "commGroupNum exceeds array size. commGroupNum = "
            << devProg->commGroupNum << ", max allowed = " << sizeof(devProg->hcclContext) / sizeof(uint64_t);
        for (size_t i = 0; i < devProg->commGroupNum; i++) {
            devProg->hcclContext[i] = hcclContext[i];
        }
    }

     static void DeviceInitDistributedContextToHost(const std::vector<std::string> &groupNames, DevAscendProgram *devProg) {
        if (devProg->hcclContext[0] != 0) {
            return;
        }
        auto hcclContext = DistributedContext::GetHcclContextToHost(groupNames);
        PrepareHcclContext(hcclContext, devProg);
    }

    static void DeviceInitDistributedContext(const std::vector<std::string> &groupNames,
        DevAscendProgram *devProg) {
        auto hcclContext = DistributedContext::GetHcclContext(groupNames);
        if ((hcclContext.size() == 0) || (devProg->hcclContext[0] == hcclContext[0])) {
            return;
        }
        PrepareHcclContext(hcclContext, devProg);
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitTilingData(DeviceMemoryTy devMem, DeviceKernelArgs &kArgs, const std::vector<uint8_t> &devProgData,
            DevControlFlowCache* ctrlFlowCache, const DeviceLauncherConfig &config, CachedOperator *cachedOperator) {
        auto &mutableConfig = const_cast<DeviceLauncherConfig &>(config);
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data()));
        PrepareDevProgArgs(devProg, mutableConfig, devMem.IsDevice());
        // Fill all metadata and kernel args
        bool isCtrlCacheRecording  = false;
        if (!devMem.IsDevice()) {
            isCtrlCacheRecording =  ctrlFlowCache != nullptr ? ctrlFlowCache->IsRecording() : devProg->controlFlowCache.IsRecording();
        }
        FillKernelMeta(devMem, kArgs, devProg, devProgData, isCtrlCacheRecording, config, cachedOperator);
        kArgs.ctrlFlowCache = reinterpret_cast<int64_t*>(ctrlFlowCache);
    }

    static void InitAicpuTaskInfo() {
        static bool inited = false;
        if (!inited) {
            AiCpuArgs initArgs;
            (void)memcpy_s(tensorInfo_.data(), sizeof(AiCpuArgs), &initArgs, sizeof(AiCpuArgs));
            inited = true;
        }
    }

    /*
     *  inputs          |  inputSize  |
     *  outputs         |  outputSize |
     *                  |     ...     |
     * DevTensorData*   |    input0   |
     *                  |    input1   |
     *                  |     ...     |
     *                  |    output0  |
     *                  |     ...     |
     */
    template<typename DeviceMemoryTy>
    static void DeviceInitKernelInOuts(DeviceMemoryTy devMem, DeviceKernelArgs &kArgs,
            const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            const std::vector<uint8_t>& disableL2List) {
        size_t l2InfoSize = disableL2List.size();
        auto buildInouts = [&](const std::vector<DeviceTensorData> &tensorDataList, DevTensorData* data,
            size_t &tensorIdx) {
            for (size_t k = 0; k < tensorDataList.size(); ++k) {
                auto &tensorData = tensorDataList[k];
                uint64_t addr = reinterpret_cast<uint64_t>(tensorData.GetAddr());
                if (unlikely(addr != 0 && tensorIdx < l2InfoSize && disableL2List[tensorIdx] == 1)) {
                    ALOG_INFO_F("Tneosr[%zu] ori:%lx, l2offset[%lu].", tensorIdx, addr, devMem.GetL2Offset());
                    addr += devMem.GetL2Offset();
                }
                DevAscendTensorDataCreator::Init(data, addr, tensorData.GetShape().data(), tensorData.GetShape().size());
                data++;
                tensorIdx++;
            }
            return;
        };
        size_t inputSize = inputList.size() * sizeof(DevTensorData);
        size_t outputSize = outputList.size() * sizeof(DevTensorData);
        size_t tensorSize = inputSize + outputSize + 2 * sizeof(uint64_t);
        size_t allSize = tensorSize + sizeof(AiCpuArgs);
        if (unlikely(allSize > tensorInfo_.size())) {
            tensorInfo_.resize(allSize);
        }
        InitAicpuTaskInfo();
        auto data = reinterpret_cast<uint64_t*>(tensorInfo_.data() + sizeof(AiCpuArgs));
        *data = inputList.size();
        data++;
        *data = outputList.size();
        data++;
        auto dataPtr = reinterpret_cast<DevTensorData*>(data);
        size_t tensorIdx = 0;
        buildInouts(inputList, dataPtr, tensorIdx);
        dataPtr += inputList.size();
        buildInouts(outputList, dataPtr, tensorIdx);
        if (devMem.IsDevice()) {
            kArgs.inputs = reinterpret_cast<int64_t*>(tensorInfo_.data());
            kArgs.outputs = (int64_t *)allSize;
        } else {;
            kArgs.inputs = reinterpret_cast<int64_t*>(tensorInfo_.data() + sizeof(AiCpuArgs));
            kArgs.outputs = kArgs.inputs + 1;
        }
        ALOG_DEBUG_F("Inputs %p outputs %p workspace %p cfgdata %p tensorSize %zu", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata, tensorSize);
    }

    template<typename DeviceMemoryTy>
    static std::pair<std::vector<DeviceTensorData>, std::vector<DeviceTensorData>> BuildInputOutputFromHost(
            DeviceMemoryTy devMem,
            const std::vector<RawTensorDataPtr> &inputDataList,
            const std::vector<RawTensorDataPtr> &outputDataList) {
        std::vector<DeviceTensorData> inputDeviceDataList;
        std::vector<DeviceTensorData> outputDeviceDataList;
        for (size_t k = 0; k < inputDataList.size(); k++) {
            auto &inputData = inputDataList[k];
            std::vector<int64_t> shape;
            if (inputData) {
                inputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), inputData->GetShape().begin(), inputData->GetShape().end());
                auto inAddr = devMem.CopyToDev(*inputData);
                inputDeviceDataList.emplace_back(inputData->GetDataType(), inAddr, shape);
            } else {
                inputDeviceDataList.emplace_back(DT_UINT8, nullptr, shape);
            }
        }
        for (size_t k = 0; k < outputDataList.size(); k++) {
            auto &outputData = outputDataList[k];
            std::vector<int64_t> shape;
            if (outputData) {
                outputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), outputData->GetShape().begin(), outputData->GetShape().end());
                auto outAddr = devMem.CopyToDev(*outputData);
                outputDeviceDataList.emplace_back(outputData->GetDataType(), outAddr, shape);
            } else {
                outputDeviceDataList.emplace_back(DT_UINT8, nullptr, shape);
            }
        }
        return std::make_pair(inputDeviceDataList, outputDeviceDataList);
    }

    template<typename DeviceMemoryTy>
    static void CopyFromDev(
            DeviceMemoryTy devMem,
            const std::vector<RawTensorDataPtr> &outputs) {
        for (auto &output : outputs) {
            if (output) {
                devMem.CopyFromDev(*output);
            }
        }
    }

#ifdef BUILD_WITH_CANN
    static void ChangeCaptureModeRelax();
    static void ChangeCaptureModeGlobal();
    static int GetStreamCaptureInfo(rtStream_t aicoreStream, aclmdlRI &rtModel, bool &isCapture);
    static int SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream, bool &isCapture);
    static int RunWithProfile(rtStream_t aicoreStream, rtStream_t aicpuStream, bool isCapture);
    static int DeviceLaunchOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList,
            const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize,
            CachedOperator *cachedOperator, DevControlFlowCache* ctrlCache = nullptr,
            const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static int DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream);
#else
    static void ChangeCaptureModeRelax() {}
    static void ChangeCaptureModeGlobal() {}
    static int GetStreamCaptureInfo(rtStream_t, aclmdlRI &, bool &) { return 0; }
    static int SetCaptureStream(rtStream_t, rtStream_t, bool &) { return 0; }
    static int RunWithProfile(rtStream_t, rtStream_t, bool) { return 0; }
    static int DeviceLaunchOnceWithDeviceTensorData(Function *, const std::vector<DeviceTensorData> &,
        const std::vector<DeviceTensorData> &, rtStream_t, rtStream_t, bool, CachedOperator *, uintptr_t,
        const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        (void)config;
        return 0;
    }
    static int DeviceSynchronize(rtStream_t, rtStream_t) { return 0; }
#endif
    static void FillDeviceKernelArgs(std::vector<uint8_t> &devProgData, DeviceKernelArgs &kargs);
    static int64_t GetL2Offset();
    static uint8_t *CopyControlFlowCache(DevControlFlowCache *ctrlCache);
    static void FreeControlFlowCache(uint8_t *ctrlCache);
    static void *RegisterKernelBin(const std::vector<uint8_t> &kernelBinary);
    static void UnregisterKernelBin(void *hdl);
    static bool AddAicpuStream(aclrtStream aicoreStream, bool tripleStream);
    static int LaunchAicpuKernel(rtAicpuArgsEx_t &rtArgs, bool tripleStream, bool debugEnable);
    static int LaunchAicoreKernel(
        aclrtStream aicoreStream, void *kernel, rtArgsEx_t &rtArgs, rtTaskCfgInfo_t &rtTaskCfg, bool debugEnable);
    static int DeviceRunOnce(Function *function, DevControlFlowCache* hostCtrlCache = nullptr,
        const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static void DeviceRunCacheKernelEnable(Function *func, bool enabled);
    static bool DeviceRunCacheKernelEnable(Function *func);
    static void DeviceRunCacheKernelSet(Function *func, uint8_t *devProg);
    static uint8_t *DeviceRunCacheKernelGet(Function *func);
    static CachedOperator* DeviceRunCacheOperatorGet(Function *func);
 public:
    static std::vector<uint8_t> tensorInfo_;
};

void DataDumpInit();
void DataDumpUnInit();

class DeviceGuard {
public:
    DeviceGuard(int32_t devId);
    ~DeviceGuard();
private:
    int32_t oDevId{0};
    int32_t nDevId{0};
};

class AclModeGuard {
public:
    AclModeGuard(aclmdlRICaptureMode tmode);
    ~AclModeGuard();
private:
    aclmdlRICaptureMode mode;
};
}
#endif//SRC_MACHINE_DEVICE_LAUNCHER_H
