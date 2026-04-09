/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include "machine/runtime/eslmodel_launcher.h"
#include "machine/runtime/device_launcher.h"
#include "interface/utils/op_info_manager.h"

extern "C" int DynTileFwkBackendKernelServer(void *targ);
namespace npu::tile_fwk::dynamic {

int EslModelLauncher::EslModelLaunchAicore(aclrtStream aicoreStream, void *kernel, DeviceKernelArgs *kernelArgs) {
#ifdef BUILD_WITH_CANN
    rtArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void *> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;
    auto *devProg = (dynamic::DevAscendProgram *)(kernelArgs->cfgdata);
    auto blockDim = devProg->devArgs.nrValidAic;
    return rtKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &cfg);
#else
    (void) aicoreStream;
    (void) kernel;
    (void) kernelArgs;
    return 0;
#endif
}

void EslModelLauncher::CopyInputOutputData() {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    for (size_t k = 0; k < inputDataList.size(); k++) {
        auto &inputData = inputDataList[k];
        if (inputData) {
            memcpy_s(inputData->GetDevPtr(), inputData->size(), 
                     (uint8_t *)inputData->data(), inputData->size());
        }
    }
    for (size_t k = 0; k < outputDataList.size(); k++) {
        auto &outputData = outputDataList[k];
        if (outputData) {
            memcpy_s(outputData->GetDevPtr(), outputData->size(), 
                     (uint8_t *)outputData->data(), outputData->size());
        }
    }
}

int EslModelLauncher::DynamicKernelLaunchEsl(DeviceKernelArgs *kArgs, aclrtStream aicoreStream, void *kernel) {
#ifdef BUILD_WITH_CANN
    auto *devProg = (dynamic::DevAscendProgram *)(kArgs->cfgdata);
    devProg->devArgs.nrAic = 32;
    devProg->devArgs.nrAiv = 64;
    EslModelLaunchAicore(aicoreStream, kernel, kArgs);
    CopyInputOutputData();
    devProg->devArgs.enableEslModel = true;
    size_t shmSize = dynamic::DEVICE_TASK_CTRL_POOL_SIZE + dynamic::DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->devArgs.runtimeDataRingBufferAddr + sizeof(RuntimeDataRingBufferHead) + DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    int threadNum = static_cast<int>(devProg->devArgs.nrAicpu);
    threadNum = (devProg->devArgs.enableCtrl == 1) ? threadNum : threadNum + 1;
    std::vector<std::thread> aicpus(threadNum);
    std::atomic<int> idx{0};
    std::this_thread::sleep_for(std::chrono::seconds(10));
    for (int i = 0; i < threadNum; i++) {
        aicpus[i] = std::thread([&]() {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            std::string name = "aicput" + std::to_string(tidx);
            pthread_setname_np(pthread_self(), name.c_str());
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            (void)DynTileFwkBackendKernelServer(kArgs);
        });
    }
    for (int i = 0; i < threadNum; i++) {
        if (aicpus[i].joinable()) {
            aicpus[i].join();
        }
    }
    EslModelMemoryUtils::UnmapAllMappings();
    return 0;
#else
    (void) kArgs;
    (void) aicoreStream;
    (void) kernel;
    return 0;
#endif
}

void EslModelLauncher::ExchangeCaputerMode(const bool &isCapture) {
#ifdef BUILD_WITH_CANN
    if (isCapture) {
        aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        aclmdlRICaptureThreadExchangeMode(&mode);
        MACHINE_LOGI("captureMode is: %d", mode);
    }
#else
    (void) isCapture;
#endif
}

int EslModelLauncher::EslModelLaunchDeviceTensorData(Function *function,
    const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
    rtStream_t aicpuStream, rtStream_t aicoreStream, void *kernel, const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
    MACHINE_LOGI("Kernel Launch");
    bool isCapture = false;

    DeviceLauncher::SetCaptureStream(aicoreStream, aicpuStream, isCapture);

    if (isCapture) {
        DeviceLauncher::ChangeCaptureModeRelax();
    }

    auto rc = aclInit(nullptr);
    if (rc != 0 && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }

    auto dynAttr = function->GetDyndevAttribute();
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    EslModelMemoryUtils eslMemoryUtil;
 	DeviceLauncher::DeviceInitDistributedContext(eslMemoryUtil, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(eslMemoryUtil, kArgs, dynAttr->devProgBinary, nullptr, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(eslMemoryUtil, kArgs, inputList, outputList, dynAttr->disableL2List);
    ExchangeCaputerMode(isCapture);

    rc = DynamicKernelLaunchEsl(&kArgs, aicoreStream, kernel);
    if (rc < 0) {
        return rc;
    }
    rc = rtStreamSynchronize(aicoreStream);
    return rc;
#else
    (void) function;
    (void) inputList;
    (void) outputList;
    (void) aicpuStream;
    (void) aicoreStream;
    (void) kernel;
    (void) config;
    return 0;
#endif
}

int EslModelLauncher::EslModelRunOnce(void *kernel, const DeviceLauncherConfig &config) {
#ifdef BUILD_WITH_CANN
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    auto aicpuStream = machine::GetRA()->GetScheStream();
    auto aicoreStream = machine::GetRA()->GetStream();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    EslModelMemoryUtils devMemoryHugePage(true);
    EslModelMemoryUtils devMemoryNotHugePage(false);
    Function *function = Program::GetInstance().GetLastFunction();
 	std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(devMemoryHugePage, inputDataList, outputDataList);
    int rc = EslModelLaunchDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, aicpuStream, aicoreStream, kernel, config);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        DeviceLauncher::CopyFromDev(devMemoryNotHugePage, inputDataList);
    }
    return rc;
#else
    (void) kernel;
    (void) config;
    return 0;
#endif
}
} 