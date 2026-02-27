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

#include "machine/runtime/emulation_launcher.h"

#include <thread>
#include "machine/host/backend.h"
#include "machine/runtime/device_launcher.h"

extern "C" int DynTileFwkBackendKernelServer(void *targ);

namespace npu::tile_fwk::dynamic {

static int EmulationLaunchOnce(DeviceKernelArgs &kArgs) {
    constexpr int threadNum = 7;
    std::thread aicpuThreadList[threadNum];
    int aicpuResultList[threadNum] = {0};
    std::atomic<int> idx{0};
    auto *devProg = (DevAscendProgram *)(kArgs.cfgdata);
    size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    devProg->devArgs.aicpuPerfAddr = 0UL;
    for (int i = 0; i < static_cast<int>(devProg->devArgs.nrAicpu); i++) {
        aicpuThreadList[i] = std::thread([&](int threadIndex) {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            (void)sprintf_s(name, sizeof(name), "aicput%d", tidx);
            ALOG_DEBUG_F("start thread: %s ", name);
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            aicpuResultList[threadIndex] = DynTileFwkBackendKernelServer(&kArgs);
        }, i);
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuThreadList[i].joinable()) {
            aicpuThreadList[i].join();
        }
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuResultList[i] != 0) {
            return aicpuResultList[i];
        }
    }
    return 0;
}

int EmulationLauncher::EmulationLaunchOnceWithHostTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        DevControlFlowCache* ctrlCache, const DeviceLauncherConfig &config) {
    ALOG_DEBUG_F("!!! Emulation Launch\n");
    DeviceKernelArgs kArgs;
    auto dynAttr = function->GetDyndevAttribute();
    auto devProg = DeviceLauncher::GetDevProg(function);
    DeviceLauncher::DeviceInitDistributedContextToHost(dynAttr->commGroupNames, devProg);
    DeviceLauncher::DeviceInitTilingData(EmulationMemoryUtils(), kArgs, dynAttr->devProgBinary, ctrlCache, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(EmulationMemoryUtils(), kArgs, inputList, outputList, dynAttr->disableL2List);
    int rc = EmulationLaunchOnce(kArgs);
    return rc;
}

int EmulationLauncher::EmulationRunOnce(Function *function, DevControlFlowCache* inputCtrlCache, const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(EmulationMemoryUtils(), inputDataList, outputDataList);
    DevControlFlowCache* launchCtrlFlowCache = nullptr;
    if (inputCtrlCache) {
        launchCtrlFlowCache = reinterpret_cast<DevControlFlowCache*>(RuntimeHostAgent::GetAgent()->AllocHostAddr(inputCtrlCache->usedCacheSize, false, false));
        if (launchCtrlFlowCache) {
            memcpy_s(launchCtrlFlowCache, inputCtrlCache->usedCacheSize, inputCtrlCache, inputCtrlCache->usedCacheSize);
        }
    }
    int rc = EmulationLaunchOnceWithHostTensorData(function, inputDeviceDataList, outputDeviceDataList, launchCtrlFlowCache, config);
    RuntimeHostAgent::GetAgent()->Free(reinterpret_cast<uint8_t*>(launchCtrlFlowCache));
    return rc;
}

DevControlFlowCache* EmulationLauncher::CreateHostCtrlFlowCache(DevAscendProgram *devProg, Function *function) {
    DevControlFlowCache encodeCtrlCache;
    uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(encodeCtrlCache.data);
    encodeCtrlCache.Init(function->GetDyndevAttribute().get(),
        devProg->ctrlFlowCacheSize, devProg->runtimeOutcastPoolSize, initOffset);
    uint32_t ctrlCacheAllocSize = encodeCtrlCache.GetSize();
    DevControlFlowCache* hostCtrlFlowCache = reinterpret_cast<DevControlFlowCache*>(RuntimeHostAgent::GetAgent()->AllocHostAddr(ctrlCacheAllocSize, false, false));
    if (hostCtrlFlowCache == nullptr) {
        return nullptr;
    }
    hostCtrlFlowCache->allCacheSize = ctrlCacheAllocSize;
    initOffset = reinterpret_cast<uintdevptr_t>(hostCtrlFlowCache->data);
    hostCtrlFlowCache->Init(function->GetDyndevAttribute().get(),
        devProg->ctrlFlowCacheSize, devProg->runtimeOutcastPoolSize, initOffset);
    return hostCtrlFlowCache;
}

int EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        CachedOperator *cachedOperator,  DevControlFlowCache **outCtrlFlowCache,
        const DeviceLauncherConfig &config) {
    (void)cachedOperator;
    auto dynAttr = function->GetDyndevAttribute();
    DevAscendProgram *devProg = DeviceLauncher::GetDevProg(function);
    DevControlFlowCache* hostCtrlFlowCache = CreateHostCtrlFlowCache(devProg, function);
    hostCtrlFlowCache->isRecording = true;
    hostCtrlFlowCache->isCacheOriginShape = config.isCacheOriginShape;
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceInitDistributedContextToHost(dynAttr->commGroupNames, devProg);
    DeviceLauncher::DeviceInitTilingData(EmulationMemoryUtils(), kArgs, dynAttr->devProgBinary, hostCtrlFlowCache, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(EmulationMemoryUtils(), kArgs, inputList, outputList, dynAttr->disableL2List);
    int rc = EmulationLaunchOnce(kArgs);

    hostCtrlFlowCache->isRecording = false;
    hostCtrlFlowCache->CalcUsedCacheSize();
    uint64_t contextWorkspaceAddr = hostCtrlFlowCache->contextWorkspaceAddr;
    hostCtrlFlowCache->IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    hostCtrlFlowCache->RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr);
    hostCtrlFlowCache->RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    hostCtrlFlowCache->TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    hostCtrlFlowCache->TaskAddrRelocProgramAndCtrlCache(reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(hostCtrlFlowCache), 0, 0);
    hostCtrlFlowCache->RelocMetaCache(reinterpret_cast<uint64_t>(hostCtrlFlowCache), 0);
    hostCtrlFlowCache->isActivated = true;
    devProg->ctrlFlowCacheAnchor = nullptr;
    devProg->ResetFromLaunch();
    if (outCtrlFlowCache) {
        *outCtrlFlowCache = hostCtrlFlowCache;
    }
    return rc;
}

int EmulationLauncher::BuildControlFlowCache(Function *function, DevControlFlowCache **outCtrlFlowCache,
                                             const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(EmulationMemoryUtils(), inputDataList, outputDataList);
    return BuildControlFlowCacheWithEmulationTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr, outCtrlFlowCache, config);
}

int EmulationLauncher::BuildControlFlowCache(
        Function *function,
        const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList, DevControlFlowCache **outCtrlFlowCache,
        const DeviceLauncherConfig &config) {

    auto getShapeString = [&](const std::vector<DeviceTensorData> &inputTensor) {
        std::stringstream ss;
        for (size_t i = 0; i < inputTensor.size(); ++i) {
            const auto &shape = inputTensor[i].GetShape();

            ss << "[";
            for (size_t j = 0; j < shape.size(); ++j) {
                ss << shape[j];
                if (j != shape.size() - 1) {
                    ss << ",";
                }
            }
            ss << "]";

            if (i != inputTensor.size() - 1) {
                ss << " ";
            }
        }
        return ss.str();
    };
    ALOG_INFO_F("!!! Emulation ControlFlowCache shape {%s}\n", getShapeString(inputList).c_str());

    /* python front end use inputs/output as unified tensors, outputList is always null */
    if (inputList.size() == 0 && outputList.size() == 0) {
        return BuildControlFlowCache(function, outCtrlFlowCache, config);
    } else {
        std::vector<DeviceTensorData> inputDeviceDataList;
        std::vector<DeviceTensorData> outputDeviceDataList;
        uintptr_t index = 0;
        for (auto &input : inputList) {
            inputDeviceDataList.emplace_back(input.GetDataType(), CONTROL_FLOW_CACHE_BASE_ADDR + index * CONTROL_FLOW_CACHE_TENSOR_SIZE, input.GetShape());
            index++;
        }
        for (auto &output : outputList) {
            outputDeviceDataList.emplace_back(output.GetDataType(), CONTROL_FLOW_CACHE_BASE_ADDR + index * CONTROL_FLOW_CACHE_TENSOR_SIZE, output.GetShape());
            index++;
        }
        return BuildControlFlowCacheWithEmulationTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr, outCtrlFlowCache, config);
    }
}

static std::vector<DeviceTensorData> toHostTensorData(const std::vector<DeviceTensorData> &devDataList, bool isInput) {
    std::vector<DeviceTensorData> hostDataList;
    for (auto &devData : devDataList) {
        auto size = devData.GetDataSize();
        void *ptr = malloc(size);
        if (isInput) {
#ifdef BUILD_WITH_CANN
            rtMemcpy(ptr, size, devData.GetAddr(), size, RT_MEMCPY_DEVICE_TO_HOST);
#endif
        }
        hostDataList.emplace_back(devData.GetDataType(), ptr, devData.GetShape());
    }
    return hostDataList;
}

static void freeHostTensorData(const std::vector<DeviceTensorData> &hostDataList) {
    for (auto &hostData : hostDataList) {
        free(hostData.GetAddr());
    }
}

int EmulationLauncher::EmulationLaunchDeviceTensorData(Function *function,
    const std::vector<DeviceTensorData> &inDevList, const std::vector<DeviceTensorData> &outDevList,
    const DeviceLauncherConfig &config) {
    DeviceLauncher::ChangeCaptureModeRelax();
    auto inList = toHostTensorData(inDevList, true);
    auto outList = toHostTensorData(outDevList, false);
    DeviceLauncher::ChangeCaptureModeGlobal();
    int rc = EmulationLaunchOnceWithHostTensorData(function, inList, outList, nullptr, config);
    freeHostTensorData(inList);
    freeHostTensorData(outList);
    return rc;
}
}
