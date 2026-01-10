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
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_MACHINE_DEVICE_LAUNCHER_H
#define SRC_MACHINE_DEVICE_LAUNCHER_H

#include <cstdint>

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

namespace npu::tile_fwk::dynamic {

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

        config::SetHostOption(ONLY_CODEGEN, true);
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
    static std::vector<uint8_t>& GetDevProg(Function *func) {
        return func->GetDyndevAttribute()->devProgBinary;
    }

    static bool HasInplaceArgs(Function *function) {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(GetDevProg(function).data()));
        return devProg->outputInplaceSlotList.size() != 0;
    }

    static void DeviceLauncherConfigFillDeviceInfo(const DeviceLauncherConfig &config) {
        DeviceLauncherConfig &devConfig = const_cast<DeviceLauncherConfig &>(config);
#ifdef BUILD_WITH_CANN
        int maxBlockDim = GetCfgBlockdim();
#else
        int maxBlockDim = 25;
#endif
        if (devConfig.blockdim == 0 || devConfig.blockdim > maxBlockDim) {
            devConfig.blockdim = maxBlockDim;
        }
    }

    template<typename DeviceMemoryTy>
    static void AssignMetaAddr(AstKernelArgs &kArgs, DeviceMemoryTy devMem, DevAscendProgram *devProg, CachedOperator *cachedOperator) {
        uint64_t generalSize = devProg->memBudget.metadata.general;
        uint64_t stitchPoolSize = devProg->memBudget.metadata.stitchPool;
        size_t shmSize = DEVICE_SHM_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum +
            generalSize + stitchPoolSize;
        uint64_t shmAddr = (uint64_t)devMem.AllocDev(shmSize, CachedOperator::GetMetaDataDevAddrHolder(cachedOperator));
        devProg->devArgs.startArgsAddr = shmAddr;
        shmAddr += DEV_ARGS_SIZE;
        devProg->devArgs.taskCtrl = shmAddr;
        shmAddr += DEVICE_TASK_CTRL_SIZE;
        devProg->devArgs.taskQueue = shmAddr;
        shmAddr += DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        devProg->devArgs.generalAddr = shmAddr;
        kArgs.opMetaAddrs.generalAddr = shmAddr;
        shmAddr += generalSize;
        devProg->devArgs.stitchPoolAddr = shmAddr;
        kArgs.opMetaAddrs.stitchPoolAddr = shmAddr;
        ALOG_DEBUG_F("generalSize:%lu stitchPoolSize:%lu generalAddr:%lx stitchPoolAddr:%lx.", generalSize, stitchPoolSize,
            devProg->devArgs.generalAddr, devProg->devArgs.stitchPoolAddr);
        return;
    }

    // Prepare device program scheduling and memory budget related args (keeps <= 50 lines)
    static void PrepareDevProgArgs(DevAscendProgram *devProg, const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
        int maxBlockDim = GetCfgBlockdim();
#else
        int maxBlockDim = 25;
#endif
        int blockdim = config.blockdim;
        // Validate and clamp blockdim locally (do not mutate config)
        int effectiveBlockdim = (blockdim == 0 || blockdim > maxBlockDim) ? maxBlockDim : blockdim;

        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.nrValidAic = effectiveBlockdim;
        devProg->devArgs.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());

        int aicpuNum = config.aicpuNum;
        int platformMaxAicpu = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum()) - 1;
        int clampedAicpu = (aicpuNum < platformMaxAicpu) ? aicpuNum : platformMaxAicpu;
        devProg->devArgs.scheCpuNum = CalcSchAicpuNumByBlockDim(effectiveBlockdim, clampedAicpu);

        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        devProg->devArgs.isGETensorList = config.isGETensorList ? 1 : 0;

        int minCpuNum = devProg->devArgs.scheCpuNum + 1;
        int effectiveAicpuNum = (aicpuNum < minCpuNum || aicpuNum > DEVICE_MAX_AICPU_NUM) ? (minCpuNum + 1) : aicpuNum;
        devProg->devArgs.nrAicpu = effectiveAicpuNum;

        ALOG_DEBUG_F("Set aicore blockdim:%d aicpu blockdim:%d.", effectiveBlockdim, effectiveAicpuNum);

        devProg->devArgs.enableCtrl = 1; // need set 0 if use custom cpu launch ctrl cpu
        if (config.dynWorkspaceSize) {
            ALOG_ERROR("[Deprecated] User provided dynamic workspace: %zu", config.dynWorkspaceSize);
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = std::max(
                static_cast<int64_t>(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem),
                AlignUp(config.dynWorkspaceSize, TENSOR_ADDR_ALIGNMENT));
        }

        devProg->workspaceSize = devProg->memBudget.Total();
        ALOG_INFO_F("workspaceSize=%lu, tensor=%lu, metadata=%lu, aicoreSpillen=%lu, debug.DumpTensor=%lu",
            devProg->workspaceSize, devProg->memBudget.tensor.Total(), devProg->memBudget.metadata.Total(),
            devProg->memBudget.aicoreSpilled, devProg->memBudget.debug.dumpTensor);
        ALOG_INFO_F("Tensor:rootInner=%lu, devTaskInnerOutCasts=%lu, slotted=%lux%lu(slots).",
            devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts, devProg->memBudget.tensor.MaxOutcastMem(),
            devProg->memBudget.tensor.devTaskBoundaryOutcastNum);
    }

    // Fill metadata and kArgs (templated because it uses DeviceMemoryTy) (keeps <= 50 lines)
    template<typename DeviceMemoryTy>
    static void FillKernelMeta(DeviceMemoryTy devMem, AstKernelArgs &kArgs, DevAscendProgram *devProg,
            const std::vector<uint8_t> &devProgData, const DeviceLauncherConfig &config, CachedOperator *cachedOperator) {
        AssignMetaAddr(kArgs, devMem, devProg, cachedOperator);
        devProg->l2CacheOffset = devMem.GetL2Offset();
        ASSERT(devProg->commGroupNum == config.hcclContext.size()) << "commGroupNum mismatch. commGroupNum = " <<
               devProg->commGroupNum << ", hcclContext size = " << config.hcclContext.size();
        ASSERT(devProg->commGroupNum <= (sizeof(devProg->hcclContext) / sizeof(uint64_t))) << "commGroupNum exceeds array size. commGroupNum = "
               << devProg->commGroupNum << ", max allowed = " << sizeof(devProg->hcclContext) / sizeof(uint64_t);
        for (size_t i = 0; i < devProg->commGroupNum; i++) {
            devProg->hcclContext[i] = config.hcclContext[i];
        }
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        if (devProg->controlFlowCache.isRecording && !devMem.IsDevice()) {
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
        kArgs.toSubMachineConfig.isGETensorList = config.isGETensorList ? 1 : 0;
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitTilingData(DeviceMemoryTy devMem, AstKernelArgs &kArgs, const std::vector<uint8_t> &devProgData,
        const DeviceLauncherConfig &config, CachedOperator *cachedOperator) {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data()));
        PrepareDevProgArgs(devProg, config);
        // Fill all metadata and kernel args
        FillKernelMeta(devMem, kArgs, devProg, devProgData, config, cachedOperator);
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitTensorLists(
            DeviceMemoryTy devMem,
            AstKernelArgs &kArgs,
            const std::vector<DeviceTensorData> &inputList,
            const std::vector<DeviceTensorData> &outputList) {
        auto buildInouts = [&](const std::vector<DeviceTensorData> &tensorDataList) {
            std::vector<DevTensorData> geTensors;
            for (size_t k = 0; k < tensorDataList.size(); k++) {
                auto &tensorData = tensorDataList[k];
                uint64_t addr = 0;
                if (tensorData.GetAddr() != 0) {
                    addr = (uint64_t)tensorData.GetAddr();
                }
                geTensors.emplace_back(DevAscendTensorDataCreator::Create(addr, tensorData.GetShape()));
            }
            std::vector<int64_t> encoded = DevAscendTensorDataCreator::Encode(geTensors);
            return encoded;
        };
        std::vector<int64_t> encodedInputList = buildInouts(inputList);
        std::vector<int64_t> encodedOutputList = buildInouts(outputList);
        kArgs.inputs = devMem.CopyToDev(encodedInputList, nullptr);
        kArgs.outputs = devMem.CopyToDev(encodedOutputList, nullptr);
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
        return;
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
    static void DeviceInitKernelInOuts(DeviceMemoryTy devMem, AstKernelArgs &kArgs,
            const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            const std::vector<uint8_t>& disableL2List, bool isGETensorList) {
        if (isGETensorList) {
            return DeviceInitTensorLists(devMem, kArgs, inputList, outputList);
        }
        size_t l2InfoSize = disableL2List.size();
        auto buildInouts = [&](const std::vector<DeviceTensorData> &tensorDataList, uint8_t* data, size_t size,
            size_t &tensorIdx) {
            std::vector<DevTensorData> tensors;
            for (size_t k = 0; k < tensorDataList.size(); k++) {
                auto &tensorData = tensorDataList[k];
                uint64_t addr = 0;
                if (tensorData.GetAddr() != 0) {
                    addr = (uint64_t)tensorData.GetAddr();
                }
                if (addr != 0 && tensorIdx < l2InfoSize && disableL2List[tensorIdx] == 1) {
                    ALOG_INFO_F("Tneosr[%zu] ori:%lx, l2offset[%lu].", tensorIdx, addr, devMem.GetL2Offset());
                    addr += devMem.GetL2Offset();
                }
                tensors.emplace_back(DevAscendTensorDataCreator::Create(addr, tensorData.GetShape()));
                tensorIdx++;
            }
            (void)memcpy_s(data, size, tensors.data(), size);
            return;
        };
        size_t inputSize = inputList.size() * sizeof(DevTensorData);
        size_t outputSize = outputList.size() * sizeof(DevTensorData);
        size_t allSize = inputSize + outputSize + 2 * sizeof(uint64_t);
        std::vector<int64_t> tensorInfo(allSize);
        auto* data = tensorInfo.data();
        *data = inputList.size();
        data++;
        *data = outputList.size();
        data++;
        uint8_t* dataPtr = reinterpret_cast<uint8_t*>(data);
        size_t tensorIdx = 0;
        buildInouts(inputList, dataPtr, inputSize, tensorIdx);
        dataPtr += inputSize;
        buildInouts(outputList, dataPtr, outputSize, tensorIdx);
        dataPtr += outputSize;
        kArgs.inputs = devMem.CopyToDev(tensorInfo, nullptr);
        kArgs.outputs = kArgs.inputs + 1;
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
        return;
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
    static void ChangeCaptureMode();
    static int GetStreamCaptureInfo(rtStream_t aicoreStream, aclmdlRI &rtModel, bool &isCapture);
    static int SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream, bool &isCapture);
    static int RunWithProfile(rtStream_t aicoreStream, rtStream_t aicpuStream);
    static int DeviceLaunchOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator,
            const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static int DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream);
#else
using aclmdlRICaptureMode = uint32_t;
using rtStream_t = uint64_t;
using aclmdlRI = void *;
    static void ChangeCaptureMode() {
        return;
    }
    static int GetStreamCaptureInfo(rtStream_t aicoreStream, aclmdlRI &rtModel, bool &isCapture) {
        (void)aicoreStream;
        (void)rtModel;
        (void)isCapture;
        return 0;
    }
    static int SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream, bool &isCapture) {
        (void)aicoreStream;
        (void)aicpuStream;
        (void)isCapture;
        return 0;
    }
    static int RunWithProfile(rtStream_t aicoreStream, rtStream_t aicpuStream) {
        (void)aicoreStream;
        (void)aicpuStream;
        return 0;
    }
    static int DeviceLaunchOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator, uintptr_t workspacePtr,
            const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        (void)function;
        (void)inputList;
        (void)outputList;
        (void)aicpuStream;
        (void)aicoreStream;
        (void)streamSynchronize;
        (void)cachedOperator;
        (void)workspacePtr;
        (void)config;
        return 0;
    }

    static int DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream) {
        (void)aicoreStream;
        (void)aicpuStream;
        return 0;
    }
#endif
    static int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static void DeviceRunCacheKernelEnable(Function *func, bool enabled);
    static bool DeviceRunCacheKernelEnable(Function *func);
    static void DeviceRunCacheKernelSet(Function *func, uint8_t *devProg);
    static uint8_t *DeviceRunCacheKernelGet(Function *func);
};
}
#endif//SRC_MACHINE_DEVICE_LAUNCHER_H
