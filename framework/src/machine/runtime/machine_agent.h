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
 * \file machine_agent.h
 * \brief
 */

#ifndef MACHINE_AGENT_H
#define MACHINE_AGENT_H
#include <iostream>
#include "interface/machine/host/machine_task.h"
#include "interface/cache/function_cache.h"
#include "machine/host/machine_compiler.h"
#include "machine/host/device_agent_task.h"
#include "machine/runtime/runtime.h"

namespace npu::tile_fwk {
constexpr int64_t MACHINE_DEBUG = 1;
constexpr int64_t MACHINE_ERROR = -1;
constexpr int64_t MACHINE_OK = 0;

#if defined(MACHINE_DEBUG) && MACHINE_DEBUG == 1
#define MACHINE_ASSERT(exp) ASSERT(exp)
#else
#define MACHINE_ASSERT(exp)
#endif

class MachineAgent {
public:
    MachineAgent() = default;
    static void AgentProc(DeviceAgentTask* task);
private:
    static void DumpData(const std::string &fileName, const char* data,size_t len);
    static void Validate(DeviceAgentTask* task);
    static int PrepareWorkSpace(DeviceAgentTask* task);
    static int PrepareInvokeEntry(DeviceAgentTask* task);
    static int PrepareTopo(DeviceAgentTask* task);
    static int PrepareCoreFunctionBin(DeviceAgentTask* task);
    static int PrepareReadyCoreFunction(DeviceAgentTask* task);
    static int PrepareHcclContext(DeviceAgentTask *task);
    static int PrepareReadyState(DeviceAgentTask* task);
    static void FillL2PrefetchInfo(DeviceAgentTask *task, DeviceTask &devTask);
    static void FillDeviceTask(DeviceAgentTask *task, DeviceTask &devTask, MachineDeviceAgentInfo &devInfo);
    static void DumpDeviceTaskInfo(const DeviceAgentTask *task, uint8_t *deviceTaskGmAddr, const DeviceTask &devTask);
    static int ConstructDeviceTask(DeviceAgentTask* task);
    static void SetDumpTensorInfo(InvokeParaOffset &elm, TensorInfo &tensorInfo, MachineTask *task);
    static void FillVirtualFunction(DeviceAgentTask *task);
};

class MachinePipe {
public:
    MachinePipe() = default;

    static void PipeProc(DeviceAgentTask* task);
};
int Run(const void *stream, const void *workSpaceGmAddr, DeviceAgentTask *deviceAgentTask,
    const std::vector<void *> &opOriginArgs, const std::vector<size_t> &argsSize, bool isAsync);
}
#endif // MACHINE_AGENT_H
