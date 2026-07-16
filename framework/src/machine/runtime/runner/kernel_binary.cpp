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
 * \file kernel_binary.cpp
 * \brief Implementation of KernelBinary class.
 */

#include "machine/runtime/runner/kernel_binary.h"

#include <cstdlib>
#include <sstream>

#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "interface/function/rebuildable_attribute.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime/memory_utils/device_memory_utils.h"
#include "machine/runtime/memory_utils/eslmodel_memory_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "interface/configs/config_manager_ng.h"

namespace npu::tile_fwk::dynamic {

KernelBinary::KernelBinary(std::shared_ptr<Function> func) : dynFunc(func)
{
    dynAttr = dynFunc->GetDyndevAttribute().get();
    devProg = (DevAscendProgram*)dynAttr->devProgBinary.data();
    kernelBin = RegisterKernelBinary(dynAttr->kernelBinary);
    workspaceSize = devProg->memBudget.Total();
    InitCachedArgs();
    InitLaunchArgs();
    auto aicpuArgs = (AiCpuArgs*)aicpuArgBuf.data();
    DeviceLauncher::FillSwimLaneEnableInfo(toSubMachineConfig_);
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
        EslModelMemoryUtils eslMemoryUtils{true, false};
        DeviceLauncher::FillDeviceKernelArgs(eslMemoryUtils, dynAttr->devProgBinary, aicpuArgs->kArgs,
                                             dynAttr->commGroupNames);
    } else {
        DeviceMemoryUtils deviceMemoryUtils;
        DeviceLauncher::FillDeviceKernelArgs(deviceMemoryUtils, dynAttr->devProgBinary, aicpuArgs->kArgs,
                                             dynAttr->commGroupNames);
    }
    runtimeDynamicCellMatchAddr_ = devProg->devArgs.dynamicCellMatchAddr;
    runtimeDynamicCellMatchCapacity_ = devProg->devArgs.dynamicCellMatchCapacity;
    lastPreparedDynamicCellMatchBytes_ = runtimeDynamicCellMatchCapacity_;
    kernelName_ = "PyPTO_" + dynFunc->GetOriginalRawName();
}

KernelBinary::~KernelBinary()
{
    if (runtimeDynamicCellMatchOwned_ && runtimeDynamicCellMatchAddr_ != 0) {
        DevMemoryPool::Instance().FreeDevAddr(reinterpret_cast<uint8_t*>(runtimeDynamicCellMatchAddr_));
    }
    if (runtimeDynamicCellMatchHostOwned_ && runtimeDynamicCellMatchHostAddr_ != 0) {
        std::free(reinterpret_cast<void*>(runtimeDynamicCellMatchHostAddr_));
    }
    UnregisterKernelBinary(kernelBin);
    for (auto& cache : originShapeCaches) {
        DeviceLauncher::FreeControlFlowCache(cache.devCache);
    }
    for (auto& cache : inferShapeCaches) {
        DeviceLauncher::FreeControlFlowCache(cache.devCache);
    }
}

ToSubMachineConfig& KernelBinary::GetMachineConfig() { return toSubMachineConfig_; }

uint8_t* KernelBinary::FindCtrlFlowCache(std::vector<std::vector<int64_t>>& inputs, bool isOriginShape)
{
    int64_t inHash = ControlFlowCache::Hash(inputs);
    auto& caches = isOriginShape ? originShapeCaches : inferShapeCaches;
    for (auto& cache : caches) {
        if (cache.hash == inHash) {
            return cache.devCache;
        }
    }
    return nullptr;
}

uint8_t* KernelBinary::FindCtrlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape)
{
    int64_t inHash = ControlFlowCache::Hash(inputs);
    auto& caches = isOriginShape ? originShapeCaches : inferShapeCaches;
    for (auto& cache : caches) {
        if (cache.hash == inHash) {
            return cache.devCache;
        }
    }
    return nullptr;
}

uint8_t* KernelBinary::BuildControlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape)
{
    DeviceLauncherConfig config;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    DevControlFlowCache* ctrlCache = nullptr;
    devProg->ctrlFlowCacheSize = DEFAULT_STITCH_CFGCACHE_SIZE;
    config.isCacheOriginShape = isOriginShape;
    EmulationMemoryUtils memUtils;
    int ret = EmulationLauncher::BuildControlFlowCache(dynFunc.get(), memUtils, inputs, {}, &ctrlCache, config);
    if (ret != 0) {
        COMPILER_LOGE(CtrlErr::DEVICE_TASK_BUILD_FAILED, "control flow cache failed %d", ret);
        return nullptr;
    }

    uint8_t* devCache = DeviceLauncher::CopyControlFlowCache(ctrlCache);
    COMPILER_LOGD("control flow cache: %p", devCache);
    if (isOriginShape) {
        originShapeCaches.emplace_back(inputs, devCache);
    } else {
        inferShapeCaches.emplace_back(inputs, devCache);
    }
    return devCache;
}

int64_t KernelBinary::GetWorkspaceSize(const std::vector<DeviceTensorData>& tensors)
{
    static const std::vector<DeviceTensorData> kEmptyOutputs;
    Evaluator eval{dynAttr->inputSymbolDict, &tensors, &kEmptyOutputs};
    dynamicCellMatchDescPatches_ = PrepareDynamicCellMatchDescPatches(*dynAttr, eval);
    PatchHostDynamicCellMatchTableDesc(devProg, dynamicCellMatchDescPatches_);
    if (dynAttr->maxDynamicAssembleOutcastMem.IsValid() || dynAttr->maxDynamicCellMatchTableMem.IsValid()) {
        if (dynAttr->maxDynamicAssembleOutcastMem.IsValid()) {
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = eval.Evaluate(
                dynAttr->maxDynamicAssembleOutcastMem);
        }
        if (dynAttr->maxDynamicCellMatchTableMem.IsValid()) {
            devProg->memBudget.metadata.maxDynamicCellMatchTableMem = eval.Evaluate(
                dynAttr->maxDynamicCellMatchTableMem);
            uint64_t totalDynamicCellMatchSlotNum = devProg->memBudget.metadata.dynamicCellMatchSlotNum;
            devProg->memBudget.metadata.dynamicCellMatch = totalDynamicCellMatchSlotNum *
                                                           devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
            ValidateDynamicCellMatchTableMemBudget(*dynAttr, devProg);
        }
        if (devProg->memBudget.metadata.dynamicCellMatch != lastPreparedDynamicCellMatchBytes_) {
            RefreshRuntimeDynamicCellMatchMeta(devProg->memBudget.metadata.dynamicCellMatch);
            lastPreparedDynamicCellMatchBytes_ = devProg->memBudget.metadata.dynamicCellMatch;
        }
        PatchHostDynamicCellMatchAddr(devProg);
        workspaceSize = devProg->memBudget.Total();

        // check and pretty print total workspace consumption
        auto* wsChecker = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableWorkspaceDesc>(dynFunc.get());
        MACHINE_LOGI_FULL("Memory Consumption: size=%ld\n%s\n", workspaceSize,
                          wsChecker
                              ->PrettyDumpSize(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem,
                                               devProg->memBudget.debug.Total())
                              .c_str());
        MACHINE_ASSERT(uint64_t(workspaceSize) ==
                       wsChecker->GetSizeForCheckOnly(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem,
                                                      devProg->memBudget.debug.Total()));
    }
    return workspaceSize;
}

std::pair<AiCpuArgs*, int64_t> KernelBinary::BuildKernelArgs(const std::vector<DeviceTensorData>& tensors)
{
    auto& disableL2List = dynAttr->disableL2List;
    auto aicpuArgs = (AiCpuArgs*)aicpuArgBuf.data();
    int64_t* inputp = (int64_t*)(aicpuArgs + 1);
    auto tensorData = (DevTensorData*)(inputp + 2);
    MACHINE_ASSERT((int64_t)tensors.size() == inputp[0]) << "mismatch tensor size";
    for (size_t i = 0; i < (size_t)inputp[0]; ++i) {
        auto& t = tensors[i];
        auto addr = (uint64_t)t.GetAddr();
        if (unlikely(addr && disableL2List.size() && disableL2List[i])) {
            COMPILER_LOGI("mismatch tensor addr");
            addr += l2Offset;
        }
        tensorData->address = addr;
        tensorData->dataType = tensors[i].GetDataType();
        auto& shape = t.GetShape();
        tensorData->shape.dimSize = shape.size();
        for (int j = 0; j < tensorData->shape.dimSize; ++j) {
            tensorData->shape.dim[j] = shape[j];
        }
        tensorData++;
    }

    WriteDynamicCellMatchStridePatchesToLaunchArgs(inputp, dynamicCellMatchDescPatches_);

    return {aicpuArgs, aicpuArgBuf.size() * sizeof(int64_t)};
}

bool KernelBinary::CheckArgs(const std::vector<DeviceTensorData>& tensors) const
{
    if (tensors.size() != argTypes.size()) {
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto& t = tensors[i];
        auto& type = argTypes[i];
        if (unlikely(t.GetDataType() != type.GetDataType())) {
            return false;
        }
        if (unlikely(t.Format() != type.Format())) {
            return false;
        }
        auto& shape1 = type.GetShape();
        auto& shape2 = t.GetShape();
        if (unlikely(shape1.size() != shape2.size())) {
            return false;
        }
        for (size_t j = 0; j < shape1.size(); ++j) {
            if (unlikely((shape1[j] != -1) && (shape1[j] != shape2[j]))) {
                return false;
            }
        }
    }
    return true;
}

void* KernelBinary::GetKernelBin() { return kernelBin; }

Function* KernelBinary::GetFunction() { return dynFunc.get(); }

const std::string& KernelBinary::GetKernelname() const { return kernelName_; }

bool KernelBinary::DisableHostCtrlFlowCacheBuild() const
{
    return devProg != nullptr && devProg->disableCtrlFlowCache != 0;
}

uint64_t KernelBinary::GetMaxDynamicAssembleOutcastMem() const
{
    return devProg->memBudget.tensor.maxDynamicAssembleOutcastMem;
}

uint64_t KernelBinary::GetMaxDynamicCellMatchTableMem() const
{
    return devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
}

uint64_t KernelBinary::GetRuntimeDynamicCellMatchAddr() const { return runtimeDynamicCellMatchAddr_; }

uint64_t KernelBinary::GetRuntimeDynamicCellMatchCapacity() const { return runtimeDynamicCellMatchCapacity_; }

void KernelBinary::SetSyncMode(uint8_t syncModel) { scheSyncModel_ = syncModel; }

uint8_t KernelBinary::GetSyncMode() { return scheSyncModel_; }

void KernelBinary::PatchHostDynamicCellMatchAddr(DevAscendProgram* hostProg)
{
    if (hostProg == nullptr) {
        return;
    }
    hostProg->devArgs.dynamicCellMatchAddr = runtimeDynamicCellMatchHostAddr_;
    hostProg->devArgs.dynamicCellMatchCapacity = runtimeDynamicCellMatchCapacity_;
}

void KernelBinary::InitCachedArgs()
{
    auto argNum = dynAttr->startArgsInputLogicalTensorList.size() + dynAttr->startArgsOutputLogicalTensorList.size();
    const uint64_t maxPatchCount = dynAttr->dynamicCellMatchLaunchMetaList.size();
    auto argSize = sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + argNum * sizeof(DevTensorData) + sizeof(uint64_t) +
                   maxPatchCount * sizeof(DevDynamicCellMatchStridePatch);
    MACHINE_ASSERT(argSize % 0x8 == 0);
    aicpuArgBuf.resize(argSize / 0x8);

    auto aicpuArgs = new (aicpuArgBuf.data()) AiCpuArgs();
    aicpuArgs->kArgs.inputs = nullptr;
    aicpuArgs->kArgs.outputs = nullptr;

    int64_t* inputp = (int64_t*)(aicpuArgs + 1);
    inputp[0] = dynAttr->startArgsInputLogicalTensorList.size();
    inputp[1] = dynAttr->startArgsOutputLogicalTensorList.size();
    const uint64_t tensorCount = static_cast<uint64_t>(inputp[0]) + static_cast<uint64_t>(inputp[1]);
    *reinterpret_cast<uint64_t*>(reinterpret_cast<DevTensorData*>(inputp + 2) + tensorCount) = 0;

    l2Offset = GetRuntimeL2Offset();

    for (auto& t : dynAttr->startArgsInputLogicalTensorList) {
        argTypes.emplace_back(t->Datatype(), nullptr, t->GetShape(), t->Format());
    }
    for (auto& t : dynAttr->startArgsOutputLogicalTensorList) {
        argTypes.emplace_back(t->Datatype(), nullptr, t->GetShape(), t->Format());
    }
}

void KernelBinary::InitLaunchArgs()
{
    memset_s(&rtAicpuArgs_, sizeof(RtAicpuArgsEx), 0, sizeof(RtAicpuArgsEx));
    rtAicpuArgs_.kernelNameAddrOffset = offsetof(AiCpuArgs, kernelName);
    rtAicpuArgs_.soNameAddrOffset = offsetof(AiCpuArgs, soName);
    rtAicpuArgs_.hostInputInfoNum = 1;
    hostInfo_.addrOffset = offsetof(AiCpuArgs, kArgs.inputs);
    hostInfo_.dataOffset = sizeof(AiCpuArgs);
    rtAicpuArgs_.hostInputInfoPtr = &hostInfo_;
    rtAicpuArgs_.timeout = AICPU_EXECUTE_TIMEOUT;
    memset_s(&rtAicoreArgs_, sizeof(RtArgsEx), 0, sizeof(RtArgsEx));
    kernelArgs_.resize(0x7, nullptr);
    rtAicoreArgs_.args = kernelArgs_.data();
    rtAicoreArgs_.argsSize = kernelArgs_.size() * sizeof(void*);

    memset_s(&rtTaskCfg_, sizeof(RtTaskCfgInfo), 0, sizeof(RtTaskCfgInfo));
    rtTaskCfg_.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);
}

void KernelBinary::RefreshRuntimeDynamicCellMatchMeta(uint64_t needBytes)
{
    if (needBytes == 0) {
        if (runtimeDynamicCellMatchOwned_ && runtimeDynamicCellMatchAddr_ != 0) {
            DevMemoryPool::Instance().FreeDevAddr(reinterpret_cast<uint8_t*>(runtimeDynamicCellMatchAddr_));
        }
        if (runtimeDynamicCellMatchHostOwned_ && runtimeDynamicCellMatchHostAddr_ != 0) {
            std::free(reinterpret_cast<void*>(runtimeDynamicCellMatchHostAddr_));
        }
        runtimeDynamicCellMatchAddr_ = 0;
        runtimeDynamicCellMatchHostAddr_ = 0;
        runtimeDynamicCellMatchCapacity_ = 0;
        runtimeDynamicCellMatchOwned_ = false;
        runtimeDynamicCellMatchHostOwned_ = false;
        return;
    }
    if (runtimeDynamicCellMatchAddr_ != 0 && runtimeDynamicCellMatchHostAddr_ != 0 &&
        runtimeDynamicCellMatchCapacity_ >= needBytes) {
        return;
    }
    uint64_t oldAddr = runtimeDynamicCellMatchAddr_;
    uint64_t oldHostAddr = runtimeDynamicCellMatchHostAddr_;
    bool oldOwned = runtimeDynamicCellMatchOwned_;
    bool oldHostOwned = runtimeDynamicCellMatchHostOwned_;
    DeviceMemoryUtils deviceMemoryUtils;
    auto* newPtr = deviceMemoryUtils.AllocDev(needBytes, nullptr);
    if (newPtr == nullptr) {
        ASSERT(false) << "alloc dynamic cell match meta failed, needBytes=" << needBytes;
        return;
    }
    auto* newHostPtr = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(needBytes)));
    if (newHostPtr == nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(newPtr);
        ASSERT(false) << "alloc host dynamic cell match meta failed, needBytes=" << needBytes;
        return;
    }
    runtimeDynamicCellMatchAddr_ = reinterpret_cast<uint64_t>(newPtr);
    runtimeDynamicCellMatchHostAddr_ = reinterpret_cast<uint64_t>(newHostPtr);
    runtimeDynamicCellMatchCapacity_ = needBytes;
    runtimeDynamicCellMatchOwned_ = true;
    runtimeDynamicCellMatchHostOwned_ = true;
    if (oldOwned && oldAddr != 0) {
        DevMemoryPool::Instance().FreeDevAddr(reinterpret_cast<uint8_t*>(oldAddr));
    }
    if (oldHostOwned && oldHostAddr != 0) {
        std::free(reinterpret_cast<void*>(oldHostAddr));
    }
}

} // namespace npu::tile_fwk::dynamic
