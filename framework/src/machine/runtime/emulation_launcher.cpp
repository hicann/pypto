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

extern "C" int DynTileFwkBackendKernelServer(void *targ);
extern "C" int DynTileFwkBackendKernelServerInit(void *targ);

namespace npu::tile_fwk::dynamic {

static int EmulationLaunchOnce(DeviceKernelArgs &kArgs) {
    constexpr int threadNum = 6;
    std::thread aicpuThreadList[threadNum];
    int aicpuResultList[threadNum] = {0};
    std::atomic<int> idx{0};
    auto *devProg = (DevAscendProgram *)(kArgs.cfgdata);
    auto rc = DynTileFwkBackendKernelServerInit(&kArgs);
    if (rc != 0) {
        return rc;
    }
    for (int i = 0; i < static_cast<int>(devProg->devArgs.nrAicpu); i++) {
        aicpuThreadList[i] = std::thread([&](int threadIndex) {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            (void)sprintf_s(name, sizeof(name), "aicput%d", tidx);
            std::cout << "start thread: " << name << std::endl;
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
        const DeviceLauncherConfig &config) {
    std::cout << "!!! Emulation Launch\n";

    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceInitTilingData(EmulationMemoryUtils(), kArgs, function->GetDyndevAttribute()->devProgBinary,
                                         config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(EmulationMemoryUtils(), kArgs, inputList, outputList,
        function->GetDyndevAttribute()->disableL2List, config.isGETensorList);
    int rc = EmulationLaunchOnce(kArgs);
    return rc;
}

int EmulationLauncher::EmulationRunOnce(Function *function, const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(EmulationMemoryUtils(), inputDataList, outputDataList);
    int rc = EmulationLaunchOnceWithHostTensorData(function, inputDeviceDataList, outputDeviceDataList, config);
    return rc;
}

int EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        CachedOperator *cachedOperator,
        const DeviceLauncherConfig &config) {
    (void)cachedOperator;
    std::cout << "!!! Emulation ControlFlowCache\n";
    std::vector<uint8_t> &devProgData = DeviceLauncher::GetDevProg(function);
    DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data()));
    devProg->controlFlowCache.isRecording = true;
    devProg->controlFlowCache.deviceTaskCount = 0;
    devProg->controlFlowCache.cacheDataOffset = 0;
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceInitTilingData(EmulationMemoryUtils(), kArgs, devProgData, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(EmulationMemoryUtils(), kArgs, inputList, outputList,
        function->GetDyndevAttribute()->disableL2List, config.isGETensorList);
    int rc = EmulationLaunchOnce(kArgs);

    devProg->controlFlowCache.isRecording = false;
    uint64_t contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->ResetFromLaunch();
    devProg->controlFlowCache.isActivated = true;
    return rc;
}

int EmulationLauncher::BuildControlFlowCache(Function *function,
                                             const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(EmulationMemoryUtils(), inputDataList, outputDataList);
    return BuildControlFlowCacheWithEmulationTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr, config);
}

int EmulationLauncher::BuildControlFlowCache(
        Function *function,
        const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList,
        const DeviceLauncherConfig &config) {
    /* python front end use inputs/output as unified tensors, outputList is always null */
    if (inputList.size() == 0 && outputList.size() == 0) {
        return BuildControlFlowCache(function, config);
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
        return BuildControlFlowCacheWithEmulationTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr, config);
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
    auto inList = toHostTensorData(inDevList, true);
    auto outList = toHostTensorData(outDevList, false);
    int rc = EmulationLaunchOnceWithHostTensorData(function, inList, outList, config);
    freeHostTensorData(inList);
    freeHostTensorData(outList);
    return rc;
}
}
