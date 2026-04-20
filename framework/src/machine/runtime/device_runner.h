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

#ifndef SRC_MACHINE_DEVICE_RUNNER_H
#define SRC_MACHINE_DEVICE_RUNNER_H

#include <cstdint>
#include <fcntl.h>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <sys/file.h>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "tilefwk/platform.h"
#include "machine/runtime/host_prof.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/runtime/pmu_common.h"

constexpr int CORE_DEFAULT_NUM = 70;
namespace npu::tile_fwk {
struct FileLock {
    FileLock() : fd(-1){};

    bool Init(const char* path)
    {
        fd = open(path, O_RDWR | O_CREAT, S_IRWXU | S_IRUSR | S_IXUSR | S_IROTH | S_IXOTH);
        return fd >= 0;
    }

    void lock() const { flock(fd, LOCK_EX); }

    void unlock() const { flock(fd, LOCK_UN); }

    ~FileLock()
    {
        if (fd != -1) {
            close(fd);
        }
    }

    int fd;
};
class DeviceRunner {
public:
    static DeviceRunner& Get();

    uint64_t GetTasksTime() const;
    int DynamicLaunch(
        RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, int64_t taskId,
        DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum);
    int DynamicLaunchSynchronize(RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream);
    int DynamicRun(
        RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, int64_t taskId,
        DeviceKernelArgs* kernelArgs, int blockdim = 25, int launchAicpuNum = 5);
    void InitDynamicArgs(DeviceArgs& args);
    int RegisterKernelBin(void** hdl, std::vector<uint8_t>* funcBinBuf = nullptr);
    static void SetBinData(const std::vector<uint8_t>& binBuf);
    HostProf& GetHostProfInstance();
    inline void SetCaptureFlag(bool isCapture) { isCapture_ = isCapture; }

    void SetDebugEnable();
    void ResetMetrics(const uint32_t& coreId);
    void ResetPerData();
    void DumpAiCoreExecutionTimeData();
    void DumpAiCorePmuData();
    void SynchronizeDeviceToHostProfData();
    void InitMetaData(DeviceArgs& devArgs);
    void InitAiCpuSoBin(DeviceArgs& devArgs);
    bool GetValidGetPgMask() const;
    void ReportHostProfInfo(
        RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore = false);
    bool GetEnableDumpDevPref() const;
    void StartMachinePerfTraceDumpThread();
    void StopMachinePerfTraceDumpThread();
    int RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream);

private:
    DeviceRunner() = default;
    ~DeviceRunner();
    void* DevAlloc(int size);
    void GetModuleLogLevel(DeviceArgs& args);
    int InitDeviceArgsCore(DeviceArgs& args, const std::vector<int64_t>& regs, const std::vector<int64_t>& regsPmu);
    int InitDeviceArgs(DeviceArgs& args);
    int Init();

    void Dump();
    void AllocDfxMetricMemory();
    /**************DynamicFunction**************/
    int launchDynamicAiCore(RtStream aicoreStream, DeviceKernelArgs* kernelArgs);
    int launchDynamicAiCpu(RtStream aicpuStream, DeviceKernelArgs* kArgs);
    int RunPrepare();
    int RunPost(RtStream aicpuStream, RtStream aicoreStream);
    int launchDynamicAiCpuInit(RtStream aicpuStream, DeviceKernelArgs* kArgs);
    int InitAicpuServer();
    int DynamicKernelLaunch(
        RtStream aicpuStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockdim);
    int DynamicTripleStreamLaunch(
        RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs,
        int blockdim);
    int ConstrutDeviceArgs(DeviceArgs& args, const std::vector<int64_t>& regs, const std::vector<int64_t>& regsPmu);
    void MachinePerfTraceDumpThread();

private:
    int devId_;
    int aicpuNum_{5};
    int blockDim_{24};
    std::vector<int64_t> pmuEvtType_;
    DeviceArgs args_;
    ToSubMachineConfig lastLaunchToSubMachineConfig_;
    DeviceArgs* devArgs_;
    std::vector<void*> perfData_;
    std::once_flag once_;
    RtBinHandle binHdl_;
    FileLock lock_;
    HostProf hostProf_;
    AclRtEvent event_;
    std::unordered_map<ArchInfo, std::function<int(std::vector<int64_t>&, std::vector<int64_t>&)>> addressMappingTable_;
    bool isCapture_{false};
    bool initFlag_{false};
    bool enableDumpMachinePerfTrace_{false};

    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};
} // namespace npu::tile_fwk
#endif // SRC_MACHINE_DEVICE_RUNNER_H
