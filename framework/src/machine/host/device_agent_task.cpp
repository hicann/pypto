/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/host/device_agent_task.h"

namespace npu::tile_fwk {
DeviceAgentTaskPtr gDeviceAgentTaskPtr = nullptr;

void DeviceAgentTask::ProcessReadyCoreFunctions(const CacheValue &cacheValue) {
    ReadyCoreFunctionCache *readyFunction = cacheValue.readyListCache.get();
    for (uint64_t i = 0; i < cacheValue.header.readyCoreFunctionNum; i++) {
        if (readyFunction->readyCoreFunction[i].coreType == static_cast<uint64_t>(CoreType::AIC)) {
            this->compileInfo.readyAicIdVec.emplace_back(readyFunction->readyCoreFunction[i].id);
            ALOG_DEBUG_F("ready aic function: %lu", readyFunction->readyCoreFunction[i].id);
        } else if (readyFunction->readyCoreFunction[i].coreType == static_cast<uint64_t>(CoreType::AICPU)) {
            this->compileInfo.readyAicpuIdVec.emplace_back(readyFunction->readyCoreFunction[i].id);
            ALOG_DEBUG_F("ready aicpu function: %lu", readyFunction->readyCoreFunction[i].id);
        } else {
            this->compileInfo.readyAivIdVec.emplace_back(readyFunction->readyCoreFunction[i].id);
            ALOG_DEBUG_F("ready aiv function: %lu", readyFunction->readyCoreFunction[i].id);
        }
    }
}

void DeviceAgentTask::UpdateCoreFunction(const CacheValue &cacheValue) {
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache.get();
    uint64_t coreFuncNum = cacheValue.header.coreFunctionNum;
    uint64_t *topoOffset = cacheTopo->coreFunctionTopoOffsets;
    uint64_t *binOffset = cacheValue.binCache->coreFunctionBinOffsets;
    for (uint64_t i = 0; i < coreFuncNum; i++) {
        CoreFunctionTopo *oneTopo = reinterpret_cast<CoreFunctionTopo *>(
                reinterpret_cast<uint8_t *>(cacheTopo) + topoOffset[i]);
        this->compileInfo.coreFunctionIdToProgramId.insert({i, oneTopo->psgId}); // 缓存下来后面functionbin偏移会用
        this->compileInfo.coreFunctionReadyState.emplace_back(
            CoreFunctionReadyState(oneTopo->readyCount, oneTopo->coreType));
        ALOG_DEBUG_F("core function : topoAddr %lx readyCount %ld coreType %lu.", i,
            oneTopo->readyCount, oneTopo->coreType);
        ASSERT((oneTopo->coreType == static_cast<uint64_t>(MachineType::AIC)) ||
                (oneTopo->coreType == static_cast<uint64_t>(MachineType::AIV)) ||
                (oneTopo->coreType == static_cast<uint64_t>(MachineType::HUB)) ||
                (oneTopo->coreType == static_cast<uint64_t>(MachineType::AICPU)))<<"Invalid core type: "<<oneTopo->coreType;
        uint64_t offset = binOffset[oneTopo->psgId] + sizeof(uint64_t);
        this->compileInfo.coreFuncBinOffset.emplace_back(offset);
    }
    for (uint64_t idx = coreFuncNum; idx < cacheValue.header.virtualFunctionNum + coreFuncNum; idx++) {
        CoreFunctionTopo *oneTopo = reinterpret_cast<CoreFunctionTopo *>(
                reinterpret_cast<uint8_t *>(cacheTopo) + topoOffset[idx]);
        ASSERT((oneTopo->coreType == static_cast<uint64_t>(MachineType::VIRTUAL_PURE)) ||
            (oneTopo->coreType == static_cast<uint64_t>(MachineType::VIRTUAL_MIX)))<<"Invalid core type: "<<oneTopo->coreType;
        this->compileInfo.coreFunctionReadyState.emplace_back(
            CoreFunctionReadyState(oneTopo->readyCount, oneTopo->coreType));
        ALOG_DEBUG_F("virtual core function : topoAddr %lx readyCount %ld coreType %lu", idx,
            oneTopo->readyCount, oneTopo->coreType);
    }
}

void DeviceAgentTask::UpdateCompileInfo() {
    CacheValue cacheValue = this->GetFuncCacheValue().value();
    UpdateCoreFunction(cacheValue);
    ProcessReadyCoreFunctions(cacheValue);
    auto &coreTensorInfoVec = this->compileInfo.coreTensorInfoVec;
    size_t invokeOffsetSize = 0;
    for (auto &mapEntry : this->compileInfo.invokeParaOffset) {
        std::vector<uint64_t> argsOffset;
        std::vector<int64_t> tensorsIdx;
        std::list<InvokeParaOffset> &invokeParaOffsetList = mapEntry.second;
        ALOG_DEBUG_F("Tensornum[%zu].", invokeParaOffsetList.size());
        this->compileInfo.coreTensorNum.emplace_back(invokeParaOffsetList.size());
        this->compileInfo.coreFunctionTensorInfoOffset.emplace_back(coreTensorInfoVec.size() * sizeof(TensorInfo));
        this->compileInfo.coreFunctionInvokeEntryOffset.emplace_back((invokeOffsetSize * sizeof(uint64_t)));
        for (auto &elm : invokeParaOffsetList) {
            invokeOffsetSize++;
            TensorInfo tensorInfo;
            SetDumpTensorInfo(elm, tensorInfo, this);
            ALOG_DEBUG_F("Current tensor info paramType is %d, dims is %u, tensorInforpid is %d.\n",
                tensorInfo.paramType, tensorInfo.dims, tensorInfo.hostpid);
            argsOffset.emplace_back(elm.offset);
            if (elm.opOriginArgsSeq == INVALID_IN_OUT_INDEX) {
                tensorsIdx.emplace_back(-1);
            } else {
                tensorsIdx.emplace_back(elm.opOriginArgsSeq);
            }
            ALOG_DEBUG_F("offset %lu  opOriginArgsSeq %zu.\n", elm.offset, elm.opOriginArgsSeq);
            coreTensorInfoVec.emplace_back(tensorInfo);
        }
        this->compileInfo.invokeArgsOffset.emplace_back(argsOffset);
        this->compileInfo.invokeTensorsIdx.emplace_back(tensorsIdx);
    }
    this->compileInfo.invokeOffsetSize = invokeOffsetSize;
    return;
}

void DeviceAgentTask::Validate() {
    ASSERT(this->GetFuncCacheValue() != std::nullopt)<<"Function cache value is empty!";
    ASSERT(this->compileInfo.coreFunctionCnt != 0)<<"Core function count is 0!";
    ASSERT(this->GetFuncCacheValue().value().header.coreFunctionNum == this->compileInfo.coreFunctionCnt)<<"Core function number is mismatch: cache="<<this->GetFuncCacheValue().value().header.coreFunctionNum<<", compileInfo="<<this->compileInfo.coreFunctionCnt;
    ASSERT(this->compileInfo.coreFunctionCnt == this->compileInfo.invokeParaOffset.size())<<"Core function count mismatch with invoke para offset size: count="<<this->compileInfo.coreFunctionCnt<<", offset="<<this->compileInfo.invokeParaOffset.size();
}

void DeviceAgentTask::SetDumpTensorInfo(const InvokeParaOffset &elm, TensorInfo &tensorInfo, const DeviceAgentTask *task) const {
    tensorInfo.functionMagic = elm.funcitonMagic;
    tensorInfo.dataType = static_cast<uint16_t>(elm.datatype);
    tensorInfo.dims = elm.tensorShape.size();
    tensorInfo.paramType = elm.paramType;
    tensorInfo.idx = elm.ioIndex;
    if (IsPtoDataDumpEnabled()) {
        tensorInfo.hostpid = getpid();
    }
    tensorInfo.subgraphId = task->GetFunction()->Operations()[0].GetSubgraphID();
    tensorInfo.deviceId = 0;
    tensorInfo.rawMagic = elm.rawMagic;
    tensorInfo.opMagic = elm.opMagic;
    tensorInfo.dataByte = BytesOf(elm.datatype);
    ALOG_DEBUG_F("Current tile tensor rawMagic %u, opMagic %u.", tensorInfo.rawMagic, tensorInfo.opMagic);
    for (size_t idx = 0; idx < elm.tensorShape.size(); idx++) {
        tensorInfo.shape[idx] = elm.tensorShape[idx];
        tensorInfo.stride[idx] = elm.rawTensorShape[idx];
        ALOG_DEBUG_F("tensor shape[%zu] = %d, stride[%zu] = %d.", idx, tensorInfo.shape[idx], idx, tensorInfo.stride[idx]);
    }
}
}
