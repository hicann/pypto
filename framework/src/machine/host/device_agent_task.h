/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "interface/machine/host/machine_task.h"
#include "tilefwk/core_func_data.h"
#include "machine/host/machine_compiler.h"

namespace npu::tile_fwk {
/* 每次device agent 处理后的信息, 如所有在workspace申请的内存, 每个AscendFunction一个 */
struct MachineDeviceAgentInfo {
    //uint32_t aicoreCnt{AICORE_NUM}; // 从全局配置获取
    uint8_t* workspaceGmAddr{nullptr};
    uint8_t* invokeEntryOffsetsGmAddr{nullptr};
    uint8_t* topoGmAddr{nullptr};
    uint8_t* functionBinGmAddr{nullptr};
    uint8_t* readyStateGmAddr{nullptr};
    uint8_t* coreFuncWsAddrGmAddr{nullptr};
    uint8_t* deviceTaskGmAddr{nullptr};
    uint8_t* readyAicQueElmGmAddr{nullptr};
    uint8_t* readyAivQueElmGmAddr{nullptr};
    uint8_t* readyAicpuQueElmGmAddr{nullptr};
    uint8_t* readyAicQueGmAddr{nullptr};
    uint8_t* readyAivQueGmAddr{nullptr};
    uint8_t* readyAicpuQueGmAddr{nullptr};

    std::vector<uint64_t> coreFunctionInvokeEntryAddr;
    std::vector<uint64_t> coreFunctionInvokeEntryInfo;
    std::vector<uint64_t> coreFunctionTopoAddr;
    std::vector<uint64_t> coreFuncBinAddr;
    std::vector<npu::tile_fwk::CoreFunctionWsAddr> coreFunctionWsAddr;
    npu::tile_fwk::DeviceTask devceTask;
    std::map<int, uint8_t *> stubOutRawTensorAddr;
    std::vector<uint64_t> coreFunctionInvokeEntryOriAddr;
};

class DeviceAgentTask {
public:
    explicit DeviceAgentTask(std::shared_ptr<npu::tile_fwk::MachineTask> task) : compileTask(task) {}

    ~DeviceAgentTask() {}

    uint64_t GetTaskId() const { return this->compileTask->GetTaskId(); }

    npu::tile_fwk::Function *GetFunction() const { return this->compileTask->GetFunction(); }

    std::optional<npu::tile_fwk::CacheValue> GetFuncCacheValue() const { return this->cacheValue_;}

    void SetFunctionCache(std::optional<npu::tile_fwk::CacheValue> value) { this->cacheValue_ = value; }

    uint64_t GetWorkSpaceSize() const {
        return compileInfo.aicoreCnt * compileInfo.workSpaceStackSize + compileInfo.invokeParaWorkSpaceSize;
    }

    uint8_t* GetDeviceTaskGmAddr() const { return this->deviceInfo.deviceTaskGmAddr; }

    void SetDeviceWorkSpaceAddr(uint8_t* addr) { this->deviceInfo.workspaceGmAddr = addr; }

    void SetAicpuStream(void* stream) { this->aicpuStream_ = stream; }

    void SetOpOriginArgsInfo(const std::vector<OriArgInfo>& originArgs) {
        this->opOriginArgs_ = originArgs;
    }

    uint8_t* GetOpOriginArgsRawTensorAddr(size_t seq) {
        ASSERT(seq < this->opOriginArgs_.size())<<"Sequence index="<<seq<<" out of bounds. Size: "<<this->opOriginArgs_.size();
        return reinterpret_cast<uint8_t*>(this->opOriginArgs_[seq].addr);
    }

    void SetAsync(bool isAsync) { this->isAsync_ = isAsync; }

    bool IsAsync() const { return this->isAsync_; }

    void ProcessReadyCoreFunctions(const CacheValue &cacheValue);

    void UpdateCoreFunction(const CacheValue &cacheValue);

    void UpdateCompileInfo();

    void Validate();

private:
    void SetDumpTensorInfo(const InvokeParaOffset &elm, TensorInfo &tensorInfo, const DeviceAgentTask *task) const;

public:
    std::vector<OriArgInfo> opOriginArgs_;
    void* aicpuStream_{nullptr};
    std::shared_ptr<npu::tile_fwk::MachineTask> compileTask{nullptr};
    MachineCompileInfo compileInfo;
    MachineDeviceAgentInfo deviceInfo;
private:
    bool isAsync_{false};
    std::optional<CacheValue> cacheValue_;
};
using DeviceAgentTaskPtr = std::shared_ptr<DeviceAgentTask>;
extern DeviceAgentTaskPtr gDeviceAgentTaskPtr;
}
