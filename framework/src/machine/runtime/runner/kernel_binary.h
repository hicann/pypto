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
 * \file kernel_binary.h
 * \brief KernelBinary class for managing compiled kernel binary and its runtime resources.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "interface/function/function.h"
#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"
#include "machine/runtime/launcher/device_launcher_types.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/utils/machine_ws_intf.h"


using namespace npu::tile_fwk;
namespace npu::tile_fwk::dynamic {

class KernelBinary {
public:
    KernelBinary(std::shared_ptr<Function> func);
    ~KernelBinary();

    ToSubMachineConfig& GetMachineConfig();

    uint8_t* FindCtrlFlowCache(std::vector<std::vector<int64_t>>& inputs, bool isOriginShape);
    uint8_t* FindCtrlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape);
    uint8_t* BuildControlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape);

    int64_t GetWorkspaceSize(const std::vector<DeviceTensorData>& tensors);

    std::pair<AiCpuArgs*, int64_t> BuildKernelArgs(const std::vector<DeviceTensorData>& tensors);

    bool CheckArgs(const std::vector<DeviceTensorData>& tensors) const;

    void* GetKernelBin();
    auto& GetArgTypes() { return argTypes; }
    Function* GetFunction();
    const std::string& GetKernelname() const;
    bool DisableHostCtrlFlowCacheBuild() const;
    uint64_t GetMaxDynamicAssembleOutcastMem() const;
    uint64_t GetMaxDynamicCellMatchTableMem() const;
    uint64_t GetRuntimeDynamicCellMatchAddr() const;
    uint64_t GetRuntimeDynamicCellMatchCapacity() const;

    void SetSyncMode(uint8_t syncModel);
    uint8_t GetSyncMode();

    void PatchHostDynamicCellMatchAddr(DevAscendProgram* hostProg);

private:
    void InitCachedArgs();
    void RefreshRuntimeDynamicCellMatchMeta(uint64_t needBytes);

    std::shared_ptr<Function> dynFunc;
    DyndevFunctionAttribute* dynAttr{nullptr};
    DevAscendProgram* devProg{nullptr};
    void* kernelBin{nullptr};
    int64_t workspaceSize{0}; // static workspace size
    std::vector<ControlFlowCache> inferShapeCaches;
    std::vector<ControlFlowCache> originShapeCaches;

    std::vector<int64_t> aicpuArgBuf;
    uint64_t l2Offset{0};
    std::vector<DeviceTensorData> argTypes;
    std::vector<DevDynamicCellMatchStridePatch> dynamicCellMatchDescPatches_;
    uint64_t lastPreparedDynamicCellMatchBytes_{0};
    uint64_t runtimeDynamicCellMatchAddr_{0};
    uint64_t runtimeDynamicCellMatchHostAddr_{0};
    uint64_t runtimeDynamicCellMatchCapacity_{0};
    bool runtimeDynamicCellMatchOwned_{false};
    bool runtimeDynamicCellMatchHostOwned_{false};
    std::string kernelName_;
    ToSubMachineConfig toSubMachineConfig_;
    uint8_t scheSyncModel_{0};
};

} // namespace npu::tile_fwk::dynamic
