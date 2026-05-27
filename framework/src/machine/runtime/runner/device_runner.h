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
#include <thread>
#include <atomic>
#include "machine/runtime/runner/host_prof.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk {
class DeviceRunner {
public:
    static DeviceRunner& Get();

    int DynamicLaunch(
        RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, int64_t taskId,
        DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum);
    static int DynamicLaunchSynchronize(RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream);
    int DynamicRun(int64_t taskId, DeviceKernelArgs* kernelArgs, int blockdim = 25, int launchAicpuNum = 5);
    int RegisterKernelBin(void** hdl, const std::vector<uint8_t>& binBuffer);
    int RegisterKernelBin(const std::vector<uint8_t>& binBuffer);

    void SetHostProfFunction(Function* function);
    uint32_t GetHostProfType() const;
    void SetCaptureFlag(bool isCapture) { isCapture_ = isCapture; }

    void SetDebugEnable();
    void ResetMetrics(const uint32_t& coreId);
    void ResetPerData() const;
    void DumpAiCoreExecutionTimeData();
    static void DumpAiCorePmuData();
    void SynchronizeDeviceToHostProfData();
    void InitMetaData(DeviceArgs& devArgs) const;
    static void InitAiCpuSoBin(DeviceArgs& devArgs);
    void ReportHostProfInfo(
        RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore = false) const;
    bool GetEnableDumpDevPref() const;
    void StartMachinePerfTraceDumpThread();
    void StopMachinePerfTraceDumpThread();
    int RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream) const;

private:
    DeviceRunner() = default;
    ~DeviceRunner();
    uint32_t GetBlockDim() const { return args_.nrValidAic; }
    uint32_t GetAicpuNum() const { return args_.nrAicpu; }
    static void InitDevDfxArgs(const bool isPerfTrace, DevDfxArgs &devDfxArg);
    static void GetAicoreRegs(const ArchInfo archInfo, std::vector<int64_t> &regs, std::vector<int64_t> &regsPmu);
    static int InitDeviceArgsCore(DeviceArgs& args);
    static void InitAicpuPerfAddr(DeviceArgs& args);
    int InitDeviceArgs(DeviceArgs& args);
    void InitPerfData();
    int Init();

    void AllocDfxMetricMemory();
    /**************DynamicFunction**************/
    int LaunchDynamicAiCore(RtStream aicoreStream, DeviceKernelArgs* kernelArgs) const;
    int LaunchDynamicAiCpu(RtStream aicpuStream, DeviceKernelArgs* kernelArgs) const;
    int RunPrepare() const;
    static void RunPost(RtStream aicpuStream, RtStream aicoreStream);
    int InitAicpuServer(int64_t *devArgsAddr);
    int DynamicKernelLaunch(RtStream aicpuStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs) const;
    int DynamicTripleStreamLaunch(
        RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs);
    void MachinePerfTraceDumpThread();

private:
    std::once_flag once_;
    DeviceArgs args_;
    RtBinHandle binHdl_;
    AclRtEvent event_;
    HostProf hostProf_;
    bool isCapture_{false};
    bool isPerfDataInited_{false};
    std::vector<void*> perfData_;
    bool isPyptoNullLaunched_{false};
    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};
} // namespace npu::tile_fwk
