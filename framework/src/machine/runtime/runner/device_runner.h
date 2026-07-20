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
 * \file device_runner.h
 * \brief
 */

#pragma once

#include <vector>
#include <mutex>
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/device_perf.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk {
struct KernelLaunchInfo {
    RtStream schedStream;
    RtStream ctrlStream;
    RtStream aicoreStream;
    uint32_t blockDim;
    uint32_t aicpuNum;
    RtBinHandle binHandle;
    bool isCaptureActivate;
    KernelLaunchInfo(RtStream sStream, RtStream cStream, RtStream aStream, uint32_t blkDim, uint32_t cpuNum)
        : schedStream(sStream),
          ctrlStream(cStream),
          aicoreStream(aStream),
          blockDim(blkDim),
          aicpuNum(cpuNum),
          binHandle(nullptr),
          isCaptureActivate(false)
    {}
};
class DeviceRunner {
public:
    static DeviceRunner& Get();
    static int DynamicLaunchSynchronize(RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream);

    void SetHostProfFunction(Function* function,
                             const std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors = {});
    uint32_t GetHostProfType() const;
    void ReportHostProfInfo(RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType,
                            bool isCore = false) const;

    void InitMetaData(DeviceArgs& devArgs) const;
    bool GetEnableDumpDevPref() const;
    void ResetPerData() const;
    void SyncProfData(bool debugEnable);
    void SetDebugEnable();

private:
    DeviceRunner() = default;
    ~DeviceRunner();
    int Init();
    int InitDeviceArgs(DeviceArgs& args);
    static void InitAiCpuSoBin(DeviceArgs& devArgs);
    static void InitDevDfxArgs(const bool isPerfTrace, DevDfxArgs& devDfxArg);
    static void GetAicoreRegs(const ArchInfo archInfo, std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu);
    static int InitDeviceArgsCore(DeviceArgs& args);
    static void InitAicpuPerfAddr(DeviceArgs& args);
    static int LaunchAicpuServerInit(int64_t* devArgsAddr);

private:
    std::once_flag once_;
    DeviceArgs args_;
    HostProf hostProf_;
    DevicePerf devicePerf_;
};
} // namespace npu::tile_fwk
