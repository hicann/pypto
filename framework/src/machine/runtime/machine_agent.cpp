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
 * \file machine_agent.cpp
 * \brief
 */

#include "machine/runtime/machine_agent.h"
#include <cstdint>
#include <list>
#include <map>
#include <vector>
#include <iostream>
#include "machine/utils/machine_ws_intf.h"
#include "interface/utils/common.h"
#include "machine/runtime/device_runner.h"

#ifdef BUILD_WITH_CANN
#include "securec.h"
extern "C" __attribute__((weak)) int AdxDataDumpServerInit();
#endif


namespace npu::tile_fwk {
void MachineAgent::DumpData(const std::string &fileName,const char *data, size_t len) {
    std::ofstream fout(fileName, std::ios::binary);
    if (!fout) {
        ALOG_INFO_F("can not open file.");
        return;
    }
    fout.write(data, len);
}

/* alloc workspace memory and prepare device task */
void MachineAgent::AgentProc(DeviceAgentTask *task) {
#ifdef BUILD_WITH_CANN
    aclInit(nullptr);
    CheckDeviceId();
    // 使能了dump功能
    if (IsPtoDataDumpEnabled()) {
        int sf = AdxDataDumpServerInit();
        if (sf != 0) {
            printf("ERROR AdxDataDumpServerInit failed \n");
        }
    }
#endif
    
    if (task->compileInfo.coreFunctionCnt == 0) {
        return;
    }

    int ret = PrepareWorkSpace(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    ret = PrepareInvokeEntry(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    ret = PrepareTopo(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    ret = PrepareCoreFunctionBin(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    ret = PrepareReadyCoreFunction(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    ret = PrepareReadyState(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    ret = ConstructDeviceTask(task);
    MACHINE_ASSERT(ret == MACHINE_OK); // 先assert 待异常处理

    Validate(task);
    (void)ret;
}

int MachineAgent::PrepareWorkSpace(DeviceAgentTask *task) {
    if (task->deviceInfo.workspaceGmAddr != nullptr) {
        /* in API mode, task->deviceInfo.workspaceGmAddr is provided  by outside sources */
        return MACHINE_OK;
    }

    uint64_t workSpaceSize =
        task->compileInfo.invokeParaWorkSpaceSize +
        task->compileInfo.aicoreCnt * task->compileInfo.workSpaceStackSize;
    ALOG_EVENT_F("[DEVICE AGENT] workSpaceSize: %lu, stacksize: %lu", workSpaceSize,
        task->compileInfo.workSpaceStackSize);

    if (workSpaceSize == 0) {
        task->deviceInfo.workspaceGmAddr = nullptr;
        ALOG_INFO_F("no need alloc workspace memory .");
        return MACHINE_OK;
    }

    uint8_t *workSpaceAddr = nullptr;
#ifdef BUILD_WITH_CANN
    machine::GetRA()->AllocDevAddr(&workSpaceAddr, workSpaceSize);
    if (workSpaceAddr == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate workspace memory !" << std::endl;
        return MACHINE_ERROR;
    }
#endif
    task->deviceInfo.workspaceGmAddr = workSpaceAddr;

    ALOG_INFO_F("[DEVICE AGENT] Allocated workSpaceAddr: %lu", reinterpret_cast<uint64_t>(workSpaceAddr));
    return MACHINE_OK;
}

void ProcessInvokeParaOffset(DeviceAgentTask *task, InvokeParaOffset &elm,
    uint8_t *paraWorkSpaceAddr, std::vector<uint64_t> &invokeOffsetVec, 
    std::vector<uint64_t> &invokeOffsetOriVec)
{
    uint64_t value;
    uint64_t oriValue;
    if (elm.rawTensorAddr != nullptr) {
        /* rawTensorAddr 不为空，插入 rawTensorAddr + offset ,直接使用op层传入的workspace地址*/
        value = reinterpret_cast<uint64_t>(elm.rawTensorAddr) + elm.offset;
        oriValue = reinterpret_cast<uint64_t>(elm.rawTensorAddr);
        ALOG_INFO_F("[DEVICE AGENT] Added op rawTensorAddr offset: %lu", value);
    } else if (elm.isTensorParam) {
        if (elm.opOriginArgsSeq != INVALID_IN_OUT_INDEX) {
            elm.rawTensorAddr = task->GetOpOriginArgsRawTensorAddr(elm.opOriginArgsSeq);
            MACHINE_ASSERT(elm.rawTensorAddr != nullptr);
            value = reinterpret_cast<uint64_t>(elm.rawTensorAddr) + elm.offset;
            oriValue = reinterpret_cast<uint64_t>(elm.rawTensorAddr);
            ALOG_INFO_F("[DEVICE AGENT] Get op origin args raw tensor addr:segno %zu, base addr %p, offset "
                        "%lu, addr+offset %lx",
                elm.opOriginArgsSeq, elm.rawTensorAddr, elm.offset, value);
        } else {
            ALOG_INFO_F("prepare stub output rawtensor gm addr, rawMagic = %d symbol %s", elm.rawMagic,
                elm.rawSymbol.c_str());
            auto addr = task->deviceInfo.stubOutRawTensorAddr.find(elm.rawMagic);
            if (addr == task->deviceInfo.stubOutRawTensorAddr.end()) {
                elm.rawTensorAddr = paraWorkSpaceAddr + elm.rawTensorOffset;
                task->deviceInfo.stubOutRawTensorAddr[elm.rawMagic] = elm.rawTensorAddr;
                ALOG_INFO_F("[DEVICE AGENT] alloc stub workspace raw tensor addr: rawmagic = %d", elm.rawMagic);
            } else {
                ALOG_INFO_F("Use exist stub out raw tensor addr.");
            }
            value = reinterpret_cast<uint64_t>(paraWorkSpaceAddr) + elm.offset;
            oriValue = reinterpret_cast<uint64_t>(paraWorkSpaceAddr);
            ALOG_INFO_F("[DEVICE AGENT] Added stub op rawTensorAddr offset: %lx", value);
        }
    } else {
        /* raw_tensor_addr_ 为空代表是incast outcast，插入新申请的workspace地址偏移 */
        value = reinterpret_cast<uint64_t>(paraWorkSpaceAddr) + elm.offset;
        oriValue = reinterpret_cast<uint64_t>(paraWorkSpaceAddr);
        ALOG_INFO_F("[DEVICE AGENT] Added incast outcast workSpaceAddr: %lx", value);
    }
    invokeOffsetVec.push_back(value);
    invokeOffsetOriVec.push_back(oriValue);
}

void ProcessCoreFunction(DeviceAgentTask *task, uint8_t *paraWorkSpaceAddr, 
    std::vector<uint64_t> &invokeOffsetVec, std::vector<uint64_t> &invokeOffsetOriVec) {
    std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffsetMap =
        task->compileInfo.invokeParaOffset;
    for (auto &mapEntry : invokeParaOffsetMap) {
        uint64_t coreFuncId = mapEntry.first;
        std::list<InvokeParaOffset> &invokeParaOffsetList = mapEntry.second;
        ALOG_INFO_F("[DEVICE AGENT] Processing core function: %lu", coreFuncId);
        /* 写入corefunction入参地址 */
        int i = 0;
        for (auto &elm : invokeParaOffsetList) {
            ALOG_INFO_F("idx is %d, ele rawMagic: %d, rawSymbol %s, offset: %lu", i++, elm.rawMagic,
                elm.rawSymbol.c_str(), elm.offset);
                ProcessInvokeParaOffset(task, elm, paraWorkSpaceAddr, invokeOffsetVec, invokeOffsetOriVec);
        }
    }    
}

bool AllocateDeviceMemory(uint8_t*& invokeEntyDev, uint8_t*& invokeEntyDevOri, 
        uint8_t*& invokeTensorsInfoDev, size_t invokeOffsetVecSize, 
        size_t invokeOffsetOriSize, size_t invokeTensorsInfoSize)
{
    (void)invokeEntyDev;
    (void)invokeEntyDevOri;
    (void)invokeTensorsInfoDev;
    (void)invokeOffsetVecSize;
    (void)invokeOffsetOriSize;
    (void)invokeTensorsInfoSize;
#ifdef BUILD_WITH_CANN
    machine::GetRA()->AllocDevAddr(&invokeEntyDev, invokeOffsetVecSize);
    if (invokeEntyDev == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate memory for invokeEntyDev!" << std::endl;
        return false;
    }
    machine::GetRA()->AllocDevAddr(&invokeEntyDevOri, invokeOffsetOriSize);
    if (invokeEntyDevOri == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate memory for invokeEntyDev!" << std::endl;
        return false;
    }
    machine::GetRA()->AllocDevAddr(&invokeTensorsInfoDev, invokeTensorsInfoSize);
    if (invokeTensorsInfoDev == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate memory for invokeEntyInfo!" << std::endl;
        return false;
    }
    return true;
#else
    return true;
#endif
}

void CopyDataToDevice(uint8_t* invokeEntyDev, uint8_t* invokeEntyDevOri, uint8_t* invokeTensorsInfoDev,
        size_t invokeOffsetVecSize, size_t invokeTensorsInfoSize, std::vector<uint64_t>& invokeOffsetVec, 
        std::vector<uint64_t>& invokeOffsetOriVec, std::vector<TensorInfo>& coreTensorInfoVec)
{
    (void)invokeEntyDev;
    (void)invokeEntyDevOri;
    (void)invokeTensorsInfoDev;
    (void)invokeOffsetVecSize;
    (void)invokeTensorsInfoSize;
    (void)invokeOffsetVec;
    (void)invokeOffsetOriVec;
    (void)coreTensorInfoVec;
#ifdef BUILD_WITH_CANN
    machine::GetRA()->CopyToDev(
        invokeEntyDev, reinterpret_cast<uint8_t *>(invokeOffsetVec.data()), invokeOffsetVecSize);
    machine::GetRA()->CopyToDev(invokeTensorsInfoDev,
        reinterpret_cast<uint8_t *>(coreTensorInfoVec.data()),invokeTensorsInfoSize);
    machine::GetRA()->CopyToDev(invokeEntyDevOri,
        reinterpret_cast<uint8_t *>(invokeOffsetOriVec.data()), invokeOffsetOriVec.size() * sizeof(uint64_t));
    ALOG_INFO_F("[DEVICE AGENT] Copied invokeOffsetVec data to invokeEntyDev, size: %lu bytes", invokeOffsetVecSize);
#endif  
}

int MachineAgent::PrepareInvokeEntry(DeviceAgentTask *task) {
    uint8_t *paraWorkSpaceAddr = task->deviceInfo.workspaceGmAddr;
    ALOG_INFO_F("paraWorkSpaceAddr base addr %p", paraWorkSpaceAddr);
    std::vector<uint64_t> invokeOffsetVec;
    std::vector<uint64_t> invokeOffsetOriVec;
    ProcessCoreFunction(task, paraWorkSpaceAddr, invokeOffsetVec, invokeOffsetOriVec);
    ALOG_INFO_F("[DEVICE AGENT] invokeOffsetVec size: %zu", invokeOffsetVec.size());
    size_t invokeOffsetVecSize = invokeOffsetVec.size() * sizeof(uint64_t);
    size_t invokeOffsetOriSize = invokeOffsetOriVec.size() * sizeof(uint64_t);
    size_t invokeTensorsInfoSize = task->compileInfo.coreTensorInfoVec.size() * sizeof(TensorInfo);
    uint8_t *invokeEntyDev = nullptr;
    uint8_t *invokeTensorsInfoDev = nullptr;
    uint8_t *invokeEntyDevOri = nullptr;
    if (!AllocateDeviceMemory(invokeEntyDev, invokeEntyDevOri, invokeTensorsInfoDev,
        invokeOffsetVecSize, invokeOffsetOriSize, invokeTensorsInfoSize)) {
            return MACHINE_ERROR;
    }
    task->deviceInfo.invokeEntryOffsetsGmAddr = invokeEntyDev;
    ALOG_INFO_F("[DEVICE AGENT] Allocated invokeEntyDev: %lx", reinterpret_cast<uint64_t>(invokeEntyDev));
    ALOG_INFO_F("PrepareInvokeEntry invokeEntyDev: %p, invokeTensorsInfoDev: %p", invokeEntyDev, invokeTensorsInfoDev);
    DumpData("invokeEntyDev.data", reinterpret_cast<const char *>(&invokeEntyDev), sizeof(uint8_t *));
    DumpData("invokeOffsetVec.data", reinterpret_cast<const char *>(invokeOffsetVec.data()), invokeOffsetVecSize);
    CopyDataToDevice(invokeEntyDev, invokeEntyDevOri, invokeTensorsInfoDev, invokeOffsetVecSize, 
        invokeTensorsInfoSize, invokeOffsetVec, invokeOffsetOriVec, task->compileInfo.coreTensorInfoVec);
    /* cache core function absolute addr */
    for (auto &elm : task->compileInfo.coreFunctionInvokeEntryOffset) {
        task->deviceInfo.coreFunctionInvokeEntryAddr.push_back(reinterpret_cast<uint64_t>(invokeEntyDev + elm));
        task->deviceInfo.coreFunctionInvokeEntryOriAddr.push_back(reinterpret_cast<uint64_t>(invokeEntyDevOri + elm));
    }
    for (auto &elm : task->compileInfo.coreFunctionTensorInfoOffset) {
        task->deviceInfo.coreFunctionInvokeEntryInfo.push_back(reinterpret_cast<uint64_t>(invokeTensorsInfoDev + elm));
    }

    return MACHINE_OK;
}

int MachineAgent::PrepareTopo(DeviceAgentTask *task) {
    /* topo 信息从cache里获取 */
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache.get();
    uint64_t coreFuncNum = cacheValue.header.coreFunctionNum + cacheValue.header.virtualFunctionNum;
    uint8_t *topoGmAddr = nullptr;
#ifdef BUILD_WITH_CANN
    uint64_t allocSize = cacheTopo->dataSize + sizeof(uint64_t); // datasize字段头也一起加上
    machine::GetRA()->AllocDevAddr(&topoGmAddr, allocSize);
    if (topoGmAddr == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate topo memory!" << std::endl;
        return MACHINE_ERROR;
    }
    machine::GetRA()->CopyToDev(topoGmAddr, reinterpret_cast<uint8_t *>(cacheTopo), allocSize);
#endif
    task->deviceInfo.topoGmAddr = topoGmAddr;

    uint64_t *topoOffset = cacheTopo->coreFunctionTopoOffsets;
    for (uint64_t i = 0; i < coreFuncNum; i++) {
        uint64_t curTopoAddr = reinterpret_cast<uint64_t>(topoGmAddr + topoOffset[i]);
        task->deviceInfo.coreFunctionTopoAddr.push_back(curTopoAddr);
    }
    return MACHINE_OK;
}

int MachineAgent::PrepareCoreFunctionBin(DeviceAgentTask *task) {
    /* function bin 信息从cache里获取 */
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    uint64_t coreFuncNum = cacheValue.header.coreFunctionNum;
    uint8_t *binGmAddr = nullptr;
#ifdef BUILD_WITH_CANN
    CoreFunctionBinCache *cacheBin = cacheValue.binCache.get();
    uint64_t allocSize = cacheBin->dataSize + sizeof(uint64_t); // datasize字段头也一起加上
    machine::GetRA()->AllocDevAddr(&binGmAddr, allocSize);
    if (binGmAddr == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate function bin memory!" << std::endl;
        return MACHINE_ERROR;
    }
    ALOG_INFO_F("PrepareCoreFunctionBin binGmAddr: %p", binGmAddr);
    DumpData("binGmAddr.data", reinterpret_cast<const char *>(&binGmAddr), sizeof(uint8_t *));
    DumpData("cacheBin.data", reinterpret_cast<const char *>(cacheBin), allocSize);
    machine::GetRA()->CopyToDev(binGmAddr, reinterpret_cast<uint8_t *>(cacheBin), allocSize);
#endif
    task->deviceInfo.functionBinGmAddr = binGmAddr;
    for (uint64_t i = 0; i < coreFuncNum; i++) {
        uint64_t curBinAddr =
            reinterpret_cast<uint64_t>(binGmAddr + task->compileInfo.coreFuncBinOffset[i]);
        task->deviceInfo.coreFuncBinAddr.push_back(curBinAddr);
        ALOG_INFO_F("[DEVICE AGENT] core function: %lu binAddr %lx ", i, curBinAddr);
    }

    return MACHINE_OK;
}

int MachineAgent::PrepareReadyCoreFunction(DeviceAgentTask *task) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();

#ifdef BUILD_WITH_CANN
    machine::GetRA()->AllocDevAddr(&task->deviceInfo.readyAicQueElmGmAddr,
        cacheValue.header.coreFunctionNum * sizeof(uint64_t));
    machine::GetRA()->AllocDevAddr(&task->deviceInfo.readyAivQueElmGmAddr,
        cacheValue.header.coreFunctionNum * sizeof(uint64_t));
    machine::GetRA()->AllocDevAddr(&task->deviceInfo.readyAicpuQueElmGmAddr,
        cacheValue.header.coreFunctionNum * sizeof(uint64_t));
    machine::GetRA()->AllocDevAddr(
        &task->deviceInfo.readyAicQueGmAddr, sizeof(StaticReadyCoreFunctionQueue));
    machine::GetRA()->AllocDevAddr(
        &task->deviceInfo.readyAivQueGmAddr, sizeof(StaticReadyCoreFunctionQueue));
    machine::GetRA()->AllocDevAddr(
        &task->deviceInfo.readyAicpuQueGmAddr, sizeof(StaticReadyCoreFunctionQueue));
    if (task->deviceInfo.readyAicQueElmGmAddr == nullptr || task->deviceInfo.readyAivQueElmGmAddr == nullptr ||
        task->deviceInfo.readyAicQueGmAddr == nullptr || task->deviceInfo.readyAivQueGmAddr == nullptr ||
        task->deviceInfo.readyAicpuQueElmGmAddr == nullptr || task->deviceInfo.readyAicpuQueGmAddr == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate ready que memory!" << std::endl;
        return MACHINE_ERROR;
    }

    machine::GetRA()->CopyToDev(task->deviceInfo.readyAicQueElmGmAddr,
        reinterpret_cast<uint8_t *>(task->compileInfo.readyAicIdVec.data()),
        task->compileInfo.readyAicIdVec.size() * sizeof(uint64_t));
    machine::GetRA()->CopyToDev(task->deviceInfo.readyAivQueElmGmAddr,
        reinterpret_cast<uint8_t *>(task->compileInfo.readyAivIdVec.data()),
        task->compileInfo.readyAivIdVec.size() * sizeof(uint64_t));
    machine::GetRA()->CopyToDev(task->deviceInfo.readyAicpuQueElmGmAddr,
        static_cast<uint8_t *>(static_cast<void*>(task->compileInfo.readyAicpuIdVec.data())),
        task->compileInfo.readyAicpuIdVec.size() * sizeof(uint64_t));

    auto funcSetQue = [](uint8_t *elm, uint8_t *que, uint64_t elmCnt) {
        StaticReadyCoreFunctionQueue rq;
        rq.head = 0;
        rq.tail = elmCnt;
        rq.elem = reinterpret_cast<uint64_t *>(elm);
        rq.lock = 0;
        machine::GetRA()->CopyToDev(que, reinterpret_cast<uint8_t *>(&rq), sizeof(rq));
        ALOG_INFO_F("[DEVICE AGENT] set que ready function cnt: %lu", elmCnt);
    };
    funcSetQue(task->deviceInfo.readyAicQueElmGmAddr, task->deviceInfo.readyAicQueGmAddr,
        task->compileInfo.readyAicIdVec.size());
    funcSetQue(task->deviceInfo.readyAivQueElmGmAddr, task->deviceInfo.readyAivQueGmAddr,
        task->compileInfo.readyAivIdVec.size());
    funcSetQue(task->deviceInfo.readyAicpuQueElmGmAddr, task->deviceInfo.readyAicpuQueGmAddr,
        task->compileInfo.readyAicpuIdVec.size());
#endif

    return MACHINE_OK;
}

int MachineAgent::PrepareHcclContext(DeviceAgentTask *task) {
    (void)task;
    return MACHINE_OK;
}

int MachineAgent::PrepareReadyState(DeviceAgentTask *task) {
    uint8_t *readyState = nullptr;
#ifdef BUILD_WITH_CANN
    uint64_t allocSize = task->compileInfo.coreFunctionReadyState.size() * sizeof(CoreFunctionReadyState);
    machine::GetRA()->AllocDevAddr(&readyState, allocSize);
    if (readyState == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate ready state memory!" << std::endl;
        return MACHINE_ERROR;
    }
#endif
    task->deviceInfo.readyStateGmAddr = readyState;
#ifdef BUILD_WITH_CANN
    machine::GetRA()->CopyToDev(readyState,
        reinterpret_cast<uint8_t *>(task->compileInfo.coreFunctionReadyState.data()), allocSize);
#endif
    return MACHINE_OK;
}

void MachineAgent::FillL2PrefetchInfo(DeviceAgentTask *task, DeviceTask &devTask) {
    size_t num = 0;
    for (size_t i = 0; i < task->opOriginArgs_.size(); ++i) {
      if (num >= MAX_PREFETCH_NUM) {
        ALOG_WARN_F("Prefetch max num is 4.");
        break;
      }
      if (task->opOriginArgs_[i].needPrefetch && (task->opOriginArgs_[i].size != 0)) {
        devTask.l2Info.prefetchAddrs[num] = reinterpret_cast<uint64_t>(task->opOriginArgs_[i].addr);
        devTask.l2Info.prefetchSizes[num] = task->opOriginArgs_[i].size;
        ALOG_INFO_F("Prefetch addr:%lx size:%lu", devTask.l2Info.prefetchAddrs[num], devTask.l2Info.prefetchSizes[num]);
        ++num;
      }
    }
    devTask.l2Info.prefetchNum = num;
    ALOG_INFO_F("prefetchNum:%lu", devTask.l2Info.prefetchNum);
}

void MachineAgent::FillDeviceTask(DeviceAgentTask *task, DeviceTask &devTask, MachineDeviceAgentInfo &devInfo) {
    devTask.coreFunctionCnt = task->compileInfo.coreFunctionCnt;
    devTask.coreFuncData.stackWorkSpaceAddr =
        reinterpret_cast<uint64_t>(devInfo.workspaceGmAddr + task->compileInfo.invokeParaWorkSpaceSize);
    devTask.coreFuncData.stackWorkSpaceSize = task->compileInfo.workSpaceStackSize;
    devTask.coreFunctionReadyStateAddr = reinterpret_cast<uint64_t>(devInfo.readyStateGmAddr);
    devTask.readyAicCoreFunctionQue = reinterpret_cast<uint64_t>(devInfo.readyAicQueGmAddr);
    devTask.readyAivCoreFunctionQue = reinterpret_cast<uint64_t>(devInfo.readyAivQueGmAddr);
    (void)memcpy_s(&(devTask.readyAicpuFunctionQue), sizeof(uint64_t), &(devInfo.readyAicpuQueGmAddr), sizeof(uint8_t*));
    FillL2PrefetchInfo(task, devTask);
}

void MachineAgent::DumpDeviceTaskInfo(
    const DeviceAgentTask *task, uint8_t *deviceTaskGmAddr, const DeviceTask &devTask) {
    ALOG_INFO_F("DeviceTask: %lu", task->GetTaskId());
    ALOG_INFO_F(" deviceTaskGmAddr: %lx", reinterpret_cast<uint64_t>(deviceTaskGmAddr));
    ALOG_INFO_F(" coreFunctionCnt: %lu", devTask.coreFunctionCnt);
    ALOG_INFO_F(" coreFunctionReadyStateAddr:  %lx", devTask.coreFunctionReadyStateAddr);
    ALOG_INFO_F(" readyAicQueAddr:  %lx", devTask.readyAicCoreFunctionQue);
    ALOG_INFO_F(" readyAivQueAddr:  %lx", devTask.readyAivCoreFunctionQue);
    ALOG_INFO_F(" coreFunctionWsAddr:  %lx", devTask.coreFuncData.coreFunctionWsAddr);
    ALOG_INFO_F(" stackWorkSpaceAddr:  %lx", devTask.coreFuncData.stackWorkSpaceAddr);
    ALOG_INFO_F(" stackWorkSpaceSize:  %lu", devTask.coreFuncData.stackWorkSpaceSize);
}

void MachineAgent::FillVirtualFunction(DeviceAgentTask *task) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    MachineDeviceAgentInfo &devInfo = task->deviceInfo;
    for (uint64_t i = cacheValue.header.coreFunctionNum;
        i < cacheValue.header.coreFunctionNum + cacheValue.header.virtualFunctionNum; i++) {
        devInfo.coreFunctionWsAddr.push_back(
            CoreFunctionWsAddr(0, 0, 0xFFFFFFF, devInfo.coreFunctionTopoAddr.at(i), 0, 0));
        ALOG_INFO_F("virtual function ws addr : coreFuncID:%lu, topoAddr: %lx", i, devInfo.coreFunctionTopoAddr.at(i));
    }
}

int MachineAgent::ConstructDeviceTask(DeviceAgentTask *task) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    MachineDeviceAgentInfo &devInfo = task->deviceInfo;
    DeviceTask &devTask = devInfo.devceTask;
    FillDeviceTask(task, devTask, devInfo);
    for (uint64_t i = 0; i < devInfo.coreFunctionInvokeEntryAddr.size(); i++) {
        devInfo.coreFunctionWsAddr.push_back(
            CoreFunctionWsAddr(devInfo.coreFuncBinAddr.at(i), devInfo.coreFunctionInvokeEntryAddr.at(i),
                task->compileInfo.coreFunctionIdToProgramId[i], devInfo.coreFunctionTopoAddr.at(i),
                devInfo.coreFunctionInvokeEntryInfo.at(i), task->compileInfo.coreTensorNum.at(i),
                devInfo.coreFunctionInvokeEntryOriAddr.at(i)));
        ALOG_INFO_F(
            "function ws addr : coreFuncID:%lu, binaddr: %lx, invokeEntryAddr: %lx, topoAddr: %lx, Num: %lu, psgid:%lu",
            i, devInfo.coreFuncBinAddr.at(i), devInfo.coreFunctionInvokeEntryAddr.at(i),
            devInfo.coreFunctionTopoAddr.at(i), task->compileInfo.coreTensorNum.at(i),
            task->compileInfo.coreFunctionIdToProgramId[i]);
    }

    FillVirtualFunction(task); // add virtual subgraph

    uint8_t *coreFuncWsGmAddr = nullptr;
#ifdef BUILD_WITH_CANN
    uint64_t allocSize = devInfo.coreFunctionWsAddr.size() * sizeof(CoreFunctionWsAddr);
    machine::GetRA()->AllocDevAddr(&coreFuncWsGmAddr, allocSize);
    if (coreFuncWsGmAddr == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate  core func ws addr memory!" << std::endl;
        return MACHINE_ERROR;
    }
#endif
    task->deviceInfo.coreFuncWsAddrGmAddr = coreFuncWsGmAddr;
    devTask.coreFuncData.coreFunctionWsAddr = reinterpret_cast<uint64_t>(coreFuncWsGmAddr);
#ifdef BUILD_WITH_CANN
    DumpData("coreFunctionWsAddr.data", reinterpret_cast<const char *>(devInfo.coreFunctionWsAddr.data()),
        allocSize);
    machine::GetRA()->CopyToDev(
        coreFuncWsGmAddr, reinterpret_cast<uint8_t *>(devInfo.coreFunctionWsAddr.data()), allocSize);
#endif
    uint8_t *deviceTaskGmAddr = nullptr;
#ifdef BUILD_WITH_CANN
    machine::GetRA()->AllocDevAddr(&deviceTaskGmAddr, sizeof(DeviceTask));
    if (deviceTaskGmAddr == nullptr) {
        std::cerr << "[DEVICE AGENT] Error: Failed to allocate  devicetask addr memory!" << std::endl;
        return MACHINE_ERROR;
    }
#endif
    task->deviceInfo.deviceTaskGmAddr = deviceTaskGmAddr;
#ifdef BUILD_WITH_CANN
    machine::GetRA()->CopyToDev(deviceTaskGmAddr, reinterpret_cast<uint8_t *>(&devTask), sizeof(DeviceTask));
#endif
    DumpDeviceTaskInfo(task, deviceTaskGmAddr, devTask);
    DumpData("coreFuncData.data", reinterpret_cast<const char *>(&devTask.coreFuncData), sizeof(CoreFunctionData));

    return MACHINE_OK;
}

void MachineAgent::Validate(DeviceAgentTask *task) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    MachineDeviceAgentInfo &devInfo = task->deviceInfo;
    MACHINE_ASSERT(devInfo.coreFunctionInvokeEntryAddr.size() + cacheValue.header.virtualFunctionNum ==
        devInfo.coreFunctionTopoAddr.size());
    MACHINE_ASSERT(devInfo.coreFunctionInvokeEntryAddr.size() == devInfo.coreFuncBinAddr.size());
    MACHINE_ASSERT(devInfo.coreFunctionInvokeEntryAddr.size() == task->compileInfo.coreFunctionIdToProgramId.size());
    MACHINE_ASSERT(devInfo.coreFunctionInvokeEntryAddr.size() + cacheValue.header.virtualFunctionNum ==
        task->compileInfo.coreFunctionReadyState.size());
    MACHINE_ASSERT(
        task->compileInfo.invokeParaOffset.size() == task->GetFuncCacheValue().value().header.coreFunctionNum);
    (void)devInfo;
}

void MachinePipe::PipeProc(DeviceAgentTask *task) {
    /* send to device machine */
#ifdef BUILD_WITH_CANN
    auto &runner = DeviceRunner::Get();
    rtStream_t aicpuStream = task->aicpuStream_ == nullptr ? machine::GetRA()->GetScheStream() : task->aicpuStream_;
    rtStream_t aicoreStream = machine::GetRA()->GetStream();
    if (task->IsAsync()) {
        runner.RunAsync(aicpuStream, aicoreStream, task->GetTaskId(), reinterpret_cast<int64_t>(task->GetDeviceTaskGmAddr()));
    } else {
        runner.Run(aicpuStream, aicoreStream, task->GetTaskId(), reinterpret_cast<int64_t>(task->GetDeviceTaskGmAddr()));
    }
#endif
    ALOG_INFO_F("Recv task id: %lu", task->GetTaskId());
}

int Run(const void *stream, const void *workSpaceGmAddr, DeviceAgentTask *deviceAgentTask, const std::vector<void *> &opOriginArgs,
    const std::vector<size_t> &argsSize, bool isAsync) {
    MACHINE_ASSERT(stream != nullptr);
    if (deviceAgentTask->GetWorkSpaceSize() != 0) {
        MACHINE_ASSERT(workSpaceGmAddr != nullptr);
    }
    MACHINE_ASSERT(opOriginArgs.size() != 0);
    deviceAgentTask->SetDeviceWorkSpaceAddr(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(workSpaceGmAddr)));
    deviceAgentTask->SetAicpuStream(const_cast<void *>(stream));
    // Construct args info
    std::vector<OriArgInfo> argsInfo;
    for (size_t i = 0; i < opOriginArgs.size(); ++i) {
      OriArgInfo info;
      info.addr = reinterpret_cast<uint64_t>(opOriginArgs[i]);
      info.size = 0;
      info.needPrefetch = false;
      if (i < argsSize.size() && argsSize[i] > 0) {
        info.size = argsSize[i];
        info.needPrefetch = true;
      }
      argsInfo.emplace_back(info);
    }
    deviceAgentTask->SetOpOriginArgsInfo(argsInfo);
    deviceAgentTask->SetAsync(isAsync);

    MachineAgent agent;
    agent.AgentProc(deviceAgentTask);

    MachinePipe piple;
    piple.PipeProc(deviceAgentTask);
    // finishQueue_.push(std::unique_ptr<MachineTask>(task)); //  存在 aicore icache无法刷新function
    // bin问题，先不能释放，此处先注释掉，后续在FreeHandle里去做 curTask = nullptr;
    return 0;
}
} // namespace npu::tile_fwk
