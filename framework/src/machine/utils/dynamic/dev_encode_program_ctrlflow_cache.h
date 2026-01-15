/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dev_encode_program_ctrlflow_cache.h
 * \brief
 */

#pragma once

#include <cinttypes>
#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/dynamic/item_pool.h"
#include "machine/utils/dynamic/runtime_outcast_tensor.h"

namespace npu::tile_fwk::dynamic {
#define ADDRESS_CACHE_KIND_WORKSPACE         0
#define ADDRESS_CACHE_KIND_INPUT             1
#define ADDRESS_CACHE_KIND_OUTPUT            2
#define INVALID_STITCH_IDX      (static_cast<uint32_t>(-1))

constexpr size_t READY_QUEUE_SIZE = 3UL;
inline constexpr size_t MAX_CACHED_FUNC_NUM = 128;

struct ReadyQueueCache {
    uint32_t coreFunctionCnt;
    struct Queue {
        uint32_t head;
        uint32_t tail;
        uint32_t capacity;
        uint32_t *elem;
    } queueList[READY_QUEUE_SIZE];
    uint32_t readyTaskNum;
};

struct MixTaskDataCache {
    WrapInfoQueue queue;
    uint32_t* wrapTasklist;
    uint64_t wrapIdNum;
    uint64_t opWrapList[MAX_CACHED_FUNC_NUM];
    uint64_t opWrapTaskNumList[MAX_CACHED_FUNC_NUM];
};

struct DynFuncDataCache {
    DevAscendFunction *devFunc;
    predcount_t *predCount;
    int *calleeList;
    DevAscendFunctionDuppedData *duppedData;

    const DynFuncDataCache &At(size_t index) const { return this[index]; }
    DynFuncDataCache &At(size_t index) { return this[index]; }
};

struct DynFuncDataWorkspaceAddressBackup {
    uint64_t runtimeWorkspace;
    uint64_t runtimeOutcastWorkspace;
    uint64_t workspaceAddr;
    uint64_t stackWorkspaceAddr;
};

struct DynFuncDataBackup {
    predcount_t *predCountBackup;
    uint64_t *rawTensorAddrBackup;

    DynFuncDataWorkspaceAddressBackup workspaceAddressBackup;

    const DynFuncDataBackup &At(size_t index) const { return this[index]; }
    DynFuncDataBackup &At(size_t index) { return this[index]; }
};

struct DynDeviceTaskBase {
    DeviceTask devTask;
    DynFuncHeader* dynFuncDataList{nullptr};

    ReadyCoreFunctionQueue *readyQueue[READY_QUEUE_SIZE];
    DynFuncDataCache dynFuncDataCacheList[MAX_CACHED_FUNC_NUM];
    uint64_t dynFuncDataCacheListSize;

    const DevCceBinary *cceBinary;
    const DevAicpuLeafBinary *aicpuLeafBinary;

    ReadyQueueCache *readyQueueBackup;
    MixTaskDataCache *mixTaskDataBackup{nullptr};
    DynFuncDataBackup dynFuncDataBackupList[MAX_CACHED_FUNC_NUM];
    bool isLastTask{false};

    DynFuncHeader *GetDynFuncDataList() const { return dynFuncDataList; }
    DynFuncHeader *GetDynFuncDataList() { return dynFuncDataList; }
    const DynFuncDataCache *GetDynFuncDataCacheList() const { return dynFuncDataCacheList; }
    DynFuncDataCache *GetDynFuncDataCacheList() { return dynFuncDataCacheList; }

    uint64_t GetIndex() { return GetDynFuncDataList()->GetIndex(); }
    inline bool IsLastTask() const { return isLastTask;}
    void SetLastTask(bool b) { isLastTask = b;}
};

struct DeviceTaskCache {
    DynDeviceTaskBase *dynTaskBase;
};

struct DeviceExecuteSlot {
    ItemPoolIter rtOutcastIter{ITEM_POOL_INVALID_INDEX};
    bool isOutputSlot{false};
    bool isAssembleSlot{false};
    bool isAssembleSlotNeedAlloc{false};
    bool isPartialUpdateStitch{false};
    bool isPartialUpdateDirty{false};
    uint32_t stitchDupIdx{INVALID_STITCH_IDX};
    uint32_t stitchOutcastIdx;

    DevAscendProgramPartialUpdate *partialUpdate{nullptr};

    bool IsOutputAddress() const {
        return isOutputSlot;
    }
    bool IsAssembleAddress() const {
        return isAssembleSlot;
    }
};

struct DevProgramControlFlowCacheRuntime {
    struct DeviceWorkspaceAllocator {
        struct {
            SeqWsAllocator rootInner;
            SeqWsAllocator devTaskInnerExclusiveOutcasts;
            WsSlotAllocator devTaskBoundaryOutcasts;
            DevRelocVector<WsSlotAllocator::BlockHeader> slottedOutcastsBlockList;
        } tensorAllocators;
        DevRelocVector<ItemPool<RuntimeOutcastTensor>::ItemBlock> runtimeOutcastTensorPool;
    } workspace;
    struct DeviceSlotContext {
        DevRelocVector<DeviceExecuteSlot> slotList;
    } slotContext;
};

template<typename T>
inline T *RelocControlFlowCachePointer(T *&ptrRef, const RelocRange &relocProgram) {
    T *result = nullptr;
    if (relocProgram.GetDst() == 0) {
        result = ptrRef;
        relocProgram.Reloc(ptrRef);
    } else {
        relocProgram.Reloc(ptrRef);
        result = ptrRef;
    }
    return result;
}

struct DevProgramControlFlowCache {
    /* Filled by user, true means try to allocate in cache. */
    bool isRecording;
    /* Filled by user, true means activate in cache. */
    bool isActivated;

    /* Filled in caching */
    DevRelocVector<DevTensorData> inputTensorDataList;
    /* Filled in caching */
    DevRelocVector<DevTensorData> outputTensorDataList;
    /* Filled in caching for runtime */
    DevProgramControlFlowCacheRuntime runtimeBackup;

    /* Filled in caching, true means some metadata is not cached. */
    bool isRecordingStopped;
    /* Filled in caching */
    uint64_t deviceTaskCount;
    /* Filled in caching */
    uint64_t rootTaskCount;
    /* Filled in caching */
    uint64_t cacheDataOffset;
    /* Filled in caching */
    uint64_t deviceTaskSkippedCount;
    /* Filled in caching */
    uint64_t contextWorkspaceAddr;
    /* Filled in caching */
    DevRelocVector<DeviceTaskCache> deviceTaskCacheList;
    /* Filled in caching */
    DevRelocVector<uint8_t> cacheData;

    uint64_t workspaceAddr;

    bool inline IsRecording() const {
        if (IsDeviceMode()) {
            return false;
        }
        return isRecording;
    }

    bool inline IsRecordingStopped() const {
        return isRecordingStopped;
    }

    void inline StopRecording() {
        isRecordingStopped = true;
    }

#define CFGCACHE_ALIGN      8
    void *AllocateCache(uint64_t size) {
        void *result = nullptr;
        if (cacheDataOffset + size < cacheData.size()) {
            result = &cacheData[cacheDataOffset];
            /* make cache 8 byte aligned */
            cacheDataOffset += (size + CFGCACHE_ALIGN - 1) / CFGCACHE_ALIGN * CFGCACHE_ALIGN;
            DEV_VERBOSE_DEBUG("cacheDataOffset is: %lu", cacheDataOffset);
        } else {
            isRecordingStopped = true;
            DEV_DEBUG("Recording is stopped");
        }
        return result;
    }

    bool AppendDeviceTask(DynDeviceTaskBase *base) {
        if (!isRecordingStopped && (deviceTaskCount < deviceTaskCacheList.size())) {
            deviceTaskCacheList[deviceTaskCount].dynTaskBase = base;
            deviceTaskCount += 1;
            rootTaskCount += base->dynFuncDataList->Size();
            DEV_DEBUG("deviceTaskCount is: %lu", deviceTaskCount);
            return true;
        } else {
            deviceTaskSkippedCount += 1;
            return false;
        }
    }

    void InitInputOutput(DevStartArgsBase *startArgs) {
        for (size_t i = 0; i < inputTensorDataList.size(); i++) {
            inputTensorDataList[i] = startArgs->GetInputTensor(i);
        }
        for (size_t i = 0; i < outputTensorDataList.size(); i++) {
            outputTensorDataList[i] = startArgs->GetOutputTensor(i);
        }
    }

    void MatchInputOutputDump(DevStartArgsBase *startArgs) const {
        DEV_VERBOSE_DEBUG("matchio cache input size: %d", (int)inputTensorDataList.size());
        for (size_t k = 0; k < inputTensorDataList.size(); k++) {
            DEV_VERBOSE_DEBUG("matchio cache input %d: %s", (int)k, DevAscendFunction::DumpShape(inputTensorDataList[k].shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio cache output size: %d", (int)outputTensorDataList.size());
        for (size_t k = 0; k < outputTensorDataList.size(); k++) {
            DEV_VERBOSE_DEBUG("matchio cache output %d: %s", (int)k, DevAscendFunction::DumpShape(outputTensorDataList[k].shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio real input size: %d", (int)startArgs->inputTensorSize);
        for (size_t k = 0; k < startArgs->inputTensorSize; k++) {
            DEV_VERBOSE_DEBUG("matchio real input %d: %s", (int)k, DevAscendFunction::DumpShape(startArgs->GetInputTensor(k).shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio real output size: %d", (int)startArgs->outputTensorSize);
        for (size_t k = 0; k < startArgs->outputTensorSize; k++) {
            DEV_VERBOSE_DEBUG("matchio real output %d: %s", (int)k, DevAscendFunction::DumpShape(startArgs->GetOutputTensor(k).shape).c_str());
        }
    }

    inline bool MatchInputOutput(DevStartArgsBase *startArgs) const {
        MatchInputOutputDump(startArgs);

        if (inputTensorDataList.size() != startArgs->inputTensorSize) {
            return false;
        }
        if (outputTensorDataList.size() != startArgs->outputTensorSize) {
            return false;
        }
        for (size_t k = 0; k < inputTensorDataList.size(); k++) {
            if (!inputTensorDataList[k].shape.Equal(startArgs->GetInputTensor(k).shape)) {
                return false;
            }
        }
        for (size_t k = 0; k < outputTensorDataList.size(); k++) {
            if (!outputTensorDataList[k].shape.Equal(startArgs->GetOutputTensor(k).shape)) {
                return false;
            }
        }
        return true;
    }

    inline bool IsActivatedFullCache(DevStartArgsBase *startArgs) const {
        if (!isActivated) {
            return false;
        }
        if (deviceTaskSkippedCount != 0) {
            return false;
        }
        if (!MatchInputOutput(startArgs)) {
            return false;
        }
        return true;
    }

    inline bool IsActivatedPartialCache(DevStartArgsBase *startArgs) const {
        if (!isActivated) {
            return false;
        }
        if (deviceTaskCount == 0) {
            return false;
        }
        if (!MatchInputOutput(startArgs)) {
            return false;
        }
        return true;
    }

    void PredCountDataBackup(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

            predcount_t *predCountBackup = reinterpret_cast<predcount_t *>(AllocateCache(backupSize));
            if (predCountBackup == nullptr) {
                return;
            }
            dynDataBackup->predCountBackup = predCountBackup;

            memcpy_s(dynDataBackup->predCountBackup, backupSize, &duppedData->GetOperationCurrPredCount(0), backupSize);
        }
    }

    void PredCountDataRestore(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

            memcpy_s(&duppedData->GetOperationCurrPredCount(0), backupSize, dynDataBackup->predCountBackup, backupSize);
        }
    }

    void ReadyQueueDataBackup(DynDeviceTaskBase *base) {
        ReadyQueueCache *readyQueueBackup = reinterpret_cast<ReadyQueueCache *>(AllocateCache(sizeof(ReadyQueueCache)));
        if (readyQueueBackup == nullptr) {
            return;
        }
        readyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
        uint32_t readyTaskNum = 0;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->capacity;
            uint32_t *readyQueueBackupElem = reinterpret_cast<uint32_t *>(AllocateCache(backupSize));
            if (readyQueueBackupElem == nullptr) {
                return;
            }

            readyQueueBackup->queueList[i].head = base->readyQueue[i]->head;
            readyQueueBackup->queueList[i].tail = base->readyQueue[i]->tail;
            readyQueueBackup->queueList[i].capacity = base->readyQueue[i]->capacity;
            readyQueueBackup->queueList[i].elem = readyQueueBackupElem;
            memcpy_s(readyQueueBackup->queueList[i].elem, backupSize, base->readyQueue[i]->elem, backupSize);

            readyTaskNum += base->readyQueue[i]->tail - base->readyQueue[i]->head;
        }
        readyQueueBackup->readyTaskNum = readyTaskNum;
        base->readyQueueBackup = readyQueueBackup;
    }

    void ReadyQueueDataRestore(DynDeviceTaskBase *base) {
        ReadyQueueCache *readyQueueBackup = base->readyQueueBackup;
        base->devTask.coreFunctionCnt = readyQueueBackup->coreFunctionCnt;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->capacity;

            base->readyQueue[i]->head = readyQueueBackup->queueList[i].head;
            base->readyQueue[i]->tail = readyQueueBackup->queueList[i].tail;
            memcpy_s(base->readyQueue[i]->elem, backupSize, readyQueueBackup->queueList[i].elem, backupSize);
        }
    }

    void MixTaskDataBackup(DynDeviceTaskBase *base) {
        if (base->devTask.mixTaskData.wrapIdNum == 0) {
            return;
        }
        MixTaskDataCache *mixTaskDataBackup = reinterpret_cast<MixTaskDataCache *>(AllocateCache(sizeof(MixTaskDataCache)));
        if (mixTaskDataBackup == nullptr) {
            return;
        }
        mixTaskDataBackup->wrapIdNum = base->devTask.mixTaskData.wrapIdNum;
        WrapInfoQueue *wrapInfoQueue = reinterpret_cast<WrapInfoQueue *>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
        size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->Size();
        WrapInfo *wrapQueueBackupElem = reinterpret_cast<WrapInfo *>(AllocateCache(wrapInfoBackupSize));
        if (wrapQueueBackupElem == nullptr) {
            return;
        }
        size_t tasklistBackupSize = base->devTask.coreFunctionCnt;
        uint32_t *tasklistAddr = reinterpret_cast<uint32_t *>(AllocateCache(tasklistBackupSize));
        if (tasklistAddr == nullptr) {
            return;
        }
        mixTaskDataBackup->wrapTasklist = tasklistAddr;
        mixTaskDataBackup->queue.head = wrapInfoQueue->head;
        mixTaskDataBackup->queue.tail = wrapInfoQueue->tail;
        mixTaskDataBackup->queue.capacity = wrapInfoQueue->capacity;
        mixTaskDataBackup->queue.elem = wrapQueueBackupElem;
        memcpy_s(mixTaskDataBackup->queue.elem, wrapInfoBackupSize, wrapInfoQueue->elem, wrapInfoBackupSize);

        uint32_t tasklistOffset = 0;
        for (uint32_t i = mixTaskDataBackup->queue.head; i < mixTaskDataBackup->queue.tail; i++) {
            WrapInfo* srcWrapInfo = &wrapInfoQueue->elem[i];
            WrapInfo* dstWrapInfo = &mixTaskDataBackup->queue.elem[i];
            dstWrapInfo->tasklist.elem = tasklistAddr + tasklistOffset;
            uint32_t tasklistSize = srcWrapInfo->tasklist.Size();
            tasklistOffset += tasklistSize;
            memcpy_s(dstWrapInfo->tasklist.elem, tasklistSize, srcWrapInfo->tasklist.elem, tasklistSize);
        }

        memcpy_s(mixTaskDataBackup->opWrapList, MAX_CACHED_FUNC_NUM, base->devTask.mixTaskData.opWrapList, MAX_CACHED_FUNC_NUM);
        memcpy_s(mixTaskDataBackup->opWrapTaskNumList, MAX_CACHED_FUNC_NUM, base->devTask.mixTaskData.opWrapTaskNumList, MAX_CACHED_FUNC_NUM);
        base->mixTaskDataBackup = mixTaskDataBackup;
    }

    void MixTaskDataRestore(DynDeviceTaskBase *base) {
        if (base->mixTaskDataBackup == nullptr) {
            return;
        }
        MixTaskDataCache *mixTaskDataBackup = base->mixTaskDataBackup;
        base->devTask.mixTaskData.wrapIdNum = mixTaskDataBackup->wrapIdNum;
        base->devTask.mixTaskData.wrapTasklist = PtrToValue(mixTaskDataBackup->wrapTasklist);

        WrapInfoQueue *wrapInfoQueue = reinterpret_cast<WrapInfoQueue *>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
        wrapInfoQueue->head = mixTaskDataBackup->queue.head;
        wrapInfoQueue->tail = mixTaskDataBackup->queue.tail;
        wrapInfoQueue->capacity = mixTaskDataBackup->queue.capacity;
        wrapInfoQueue->elem = mixTaskDataBackup->queue.elem;

        size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->Size();
        memcpy_s(wrapInfoQueue->elem, wrapInfoBackupSize, mixTaskDataBackup->queue.elem, wrapInfoBackupSize);

        uint32_t tasklistOffset = 0;
        for (uint32_t i = mixTaskDataBackup->queue.head; i < mixTaskDataBackup->queue.tail; i++) {
            WrapInfo* srcWrapInfo = &mixTaskDataBackup->queue.elem[i];
            WrapInfo* dstWrapInfo = &wrapInfoQueue->elem[i];
            dstWrapInfo->tasklist.elem = mixTaskDataBackup->wrapTasklist + tasklistOffset;
            uint32_t tasklistSize = srcWrapInfo->tasklist.Size();
            tasklistOffset += tasklistSize;
            memcpy_s(dstWrapInfo->tasklist.elem, tasklistSize, srcWrapInfo->tasklist.elem, tasklistSize);
        }

        memcpy_s(base->devTask.mixTaskData.opWrapList, MAX_CACHED_FUNC_NUM, mixTaskDataBackup->opWrapList, MAX_CACHED_FUNC_NUM);
        memcpy_s(base->devTask.mixTaskData.opWrapTaskNumList, MAX_CACHED_FUNC_NUM, mixTaskDataBackup->opWrapTaskNumList, MAX_CACHED_FUNC_NUM);
    }

    static void RelocBuildInputOutputDesc(
            std::unordered_map<uint64_t, AddressDescriptor> &cacheInputOutputDict,
            DevStartArgsBase *devStartArgs) {
        for (uint64_t i = 0; i < devStartArgs->inputTensorSize; i++) {
            uint64_t addr = devStartArgs->GetInputTensor(i).address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_INPUT, i);
        }
        for (uint64_t i = 0; i < devStartArgs->outputTensorSize; i++) {
            uint64_t addr = devStartArgs->GetOutputTensor(i).address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_OUTPUT, i);
        }
    }

    static void RelocBuildInputOutputDesc(
            std::unordered_map<uint64_t, AddressDescriptor> &cacheInputOutputDict,
            DevRelocVector<DevTensorData> inputTensorDataList,
            DevRelocVector<DevTensorData> outputTensorDataList) {
        for (uint64_t i = 0; i < inputTensorDataList.size(); i++) {
            uint64_t addr = inputTensorDataList[i].address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_INPUT, i);
        }
        for (uint64_t i = 0; i < outputTensorDataList.size(); i++) {
            uint64_t addr = outputTensorDataList[i].address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_OUTPUT, i);
        }
    }

    static void RelocDescToCache(
            AddressDescriptor &desc,
            const RelocRange &relocWorkspace,
            std::unordered_map<uint64_t, AddressDescriptor> &cacheInputOutputDict) {
        AddressDescriptor resultDesc;
        uint64_t addr = desc.GetAddressValue();
        if (cacheInputOutputDict.count(addr)) {
            resultDesc = cacheInputOutputDict[addr];
        } else {
            relocWorkspace.Reloc(addr);
            resultDesc = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_WORKSPACE, addr);
        }
        desc = resultDesc;
    }

    static void RelocDescFromCache(
            AddressDescriptor &desc,
            const RelocRange &relocWorkspace,
            DevStartArgsBase *devStartArgs) {
        uint64_t resultAddr = 0;
        switch (desc.cacheKind) {
            case ADDRESS_CACHE_KIND_WORKSPACE:
                resultAddr = desc.cacheValue;
                relocWorkspace.Reloc(resultAddr);
                break;
            case ADDRESS_CACHE_KIND_INPUT:
                resultAddr = devStartArgs->GetInputTensor(desc.cacheValue).address;
                break;
            case ADDRESS_CACHE_KIND_OUTPUT:
                resultAddr = devStartArgs->GetOutputTensor(desc.cacheValue).address;
                break;
            default:
                DEV_ERROR("[RelocDescFromCache] Invalid kind: %lu\n", (unsigned long)desc.cacheKind);
                break;
        }
        AddressDescriptor resultDesc = AddressDescriptor::MakeFromAddress(resultAddr);
        desc = resultDesc;
    }

    void IncastOutcastAddrBackup(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

            uint64_t *rawTensorAddrBackup = reinterpret_cast<uint64_t *>(AllocateCache(backupSize));
            if (rawTensorAddrBackup == nullptr) {
                return;
            }
            dynDataBackup->rawTensorAddrBackup = rawTensorAddrBackup;
            memcpy_s(dynDataBackup->rawTensorAddrBackup, backupSize, dynData->rawTensorAddr, backupSize);
        }
    }

    void IncastOutcastAddrRestore(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

            memcpy_s(dynData->rawTensorAddr, backupSize, dynDataBackup->rawTensorAddrBackup, backupSize);
        }
    }

    void IncastOutcastAddrRestore() {
        for (size_t i = 0; i < deviceTaskCount; i++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
            IncastOutcastAddrRestore(dynTaskBase);
        }
    }

    void TaskAddrBackupWorkspace(DynDeviceTaskBase * base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;

            dynDataBackup->workspaceAddressBackup.runtimeWorkspace = duppedData->runtimeWorkspace_;
            dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace = duppedData->runtimeOutcastWorkspace_;
            dynDataBackup->workspaceAddressBackup.workspaceAddr = dynData->workspaceAddr;
            dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr = dynData->stackWorkSpaceAddr;
        }
    }

    void TaskAddrRestoreWorkspace(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;

            duppedData->runtimeWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeWorkspace;
            duppedData->runtimeOutcastWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace;
            dynData->workspaceAddr = dynDataBackup->workspaceAddressBackup.workspaceAddr;
            dynData->stackWorkSpaceAddr = dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr;
        }
    }

    void TaskAddrRestoreWorkspace() {
        for (size_t i = 0; i < deviceTaskCount; i++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
            TaskAddrRestoreWorkspace(dynTaskBase);
        }
    }

    void TaskAddrRelocWorkspace(
            uint64_t srcWorkspace, uint64_t dstWorkspace,
            DevStartArgsBase *devStartArgs) {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;

            DynFuncHeader *dynFuncDataList = dynTaskBase->dynFuncDataList;
            DynFuncDataCache *dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup *dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
                DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                if (devStartArgs == nullptr) {
                    // Host: addr uses backup

                    // Reloc Dupped
                    relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeWorkspace);
                    relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace);

                    // Reloc DynFuncData
                    relocWorkspace.Reloc(dynDataBackup->workspaceAddressBackup.workspaceAddr);
                    relocWorkspace.Reloc(dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr);
                } else {
                    // Device: addr uses actual

                    // Reloc Dupped
                    relocWorkspace.RelocNullable(duppedData->runtimeWorkspace_);
                    relocWorkspace.RelocNullable(duppedData->runtimeOutcastWorkspace_);

                    // Reloc DynFuncData
                    relocWorkspace.Reloc(dynData->workspaceAddr);
                    relocWorkspace.Reloc(dynData->stackWorkSpaceAddr);
                }
            }
        }
    }

    void IncastOutcastAddrReloc(
            uint64_t srcWorkspace, uint64_t dstWorkspace,
            DevStartArgsBase *devStartArgs) {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        /* empty constructor's overhead should be negligible */
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        if (devStartArgs == nullptr) {
            /* only run on host */
            RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
        }
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;
            DynFuncHeader *dynFuncDataList = dynTaskBase->dynFuncDataList;
            DynFuncDataCache *dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup *dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
                if (devStartArgs == nullptr) {
                    // Host: addr uses backup
                    for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                        AddressDescriptor *addr = reinterpret_cast<AddressDescriptor *>(dynDataBackup->rawTensorAddrBackup + i);
                        RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                    }
                    for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                        AddressDescriptor *addr = reinterpret_cast<AddressDescriptor *>(dynDataBackup->rawTensorAddrBackup + duppedData->GetIncastSize() + i);
                        RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                    }
                } else {
                    // Device: addr uses actual
                    for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                        AddressDescriptor *addr = &duppedData->GetIncastAddress(i);
                        RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                    }
                    for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                        AddressDescriptor *addr = &duppedData->GetOutcastAddress(i);
                        RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                    }
                }

                dynData->startArgs = devStartArgs;
            }
        }
    }

    void RuntimeAddrBackup(
            DeviceExecuteSlot *runtimeSlotList, const ItemPool<RuntimeOutcastTensor>::ItemBlock *runtimeOutcastTensorPool,
            uint64_t slotSize, uint64_t runtimeOutcastTensorSize, TensorAllocator &allocator) {
        uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
        uint64_t runtimeOutcastPoolDataSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
        (void)memcpy_s(runtimeBackup.slotContext.slotList.Data(), slotDataSize, runtimeSlotList, slotDataSize);
        (void)memcpy_s(runtimeBackup.workspace.runtimeOutcastTensorPool.Data(), runtimeOutcastPoolDataSize, runtimeOutcastTensorPool, runtimeOutcastPoolDataSize);

        struct Backup {
            static void BackupBlockHeader(WsSlotAllocator::BlockHeader *&ptr, WsSlotAllocator::BlockHeader *base) {
                ptr = reinterpret_cast<WsSlotAllocator::BlockHeader *>(static_cast<uintptr_t>(ptr - base));
            }
        };
        runtimeBackup.workspace.tensorAllocators.rootInner = allocator.rootInner;
        runtimeBackup.workspace.tensorAllocators.devTaskInnerExclusiveOutcasts = allocator.devTaskInnerExclusiveOutcasts;
        runtimeBackup.workspace.tensorAllocators.devTaskBoundaryOutcasts = allocator.devTaskBoundaryOutcasts;

        uint64_t backupSize = sizeof(WsSlotAllocator::BlockHeader) * allocator.devTaskBoundaryOutcasts.slotNum_;
        (void)memcpy_s(runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.Data(), backupSize, allocator.devTaskBoundaryOutcasts.GetBlockHeaderBase(), backupSize);

        WsSlotAllocator::BlockHeader *base = allocator.devTaskBoundaryOutcasts.GetBlockHeaderBase();
        Backup::BackupBlockHeader(runtimeBackup.workspace.tensorAllocators.devTaskBoundaryOutcasts.freeListHeader_, base);
        Backup::BackupBlockHeader(runtimeBackup.workspace.tensorAllocators.devTaskBoundaryOutcasts.notInUseHeaders_, base);
        WsSlotAllocator::BlockHeader *checkpointBase = runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.Data();
        for (uint64_t k = 0; k < allocator.devTaskBoundaryOutcasts.slotNum_; k++) {
            Backup::BackupBlockHeader(checkpointBase[k].listNext, base);
        }
    }

    void RuntimeAddrRestore(
            DeviceExecuteSlot *runtimeSlotList, ItemPool<RuntimeOutcastTensor>::ItemBlock *runtimeOutcastTensorPool,
            uint64_t slotSize, uint64_t runtimeOutcastTensorSize, TensorAllocator &allocator) {
        uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
        uint64_t runtimeOutcastPoolDataSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
        (void)memcpy_s(runtimeSlotList, slotDataSize, runtimeBackup.slotContext.slotList.Data(), slotDataSize);
        (void)memcpy_s(runtimeOutcastTensorPool, runtimeOutcastPoolDataSize, runtimeBackup.workspace.runtimeOutcastTensorPool.Data(), runtimeOutcastPoolDataSize);

        struct Restore {
            static void RestoreBlockHeader(WsSlotAllocator::BlockHeader *&ptr, WsSlotAllocator::BlockHeader *base, WsSlotAllocator::BlockHeader *index) {
                ptr = base + (uintptr_t)index;
            }
            static void RestoreSeqAllocator(SeqWsAllocator &dst, SeqWsAllocator &src) {
                dst.allocated_ = src.allocated_;
                dst.resetTimes_ = src.resetTimes_;
            }
        };
        Restore::RestoreSeqAllocator(allocator.rootInner, runtimeBackup.workspace.tensorAllocators.rootInner);
        Restore::RestoreSeqAllocator(allocator.devTaskInnerExclusiveOutcasts, runtimeBackup.workspace.tensorAllocators.devTaskInnerExclusiveOutcasts);
        allocator.devTaskBoundaryOutcasts.availableSlots_ = runtimeBackup.workspace.tensorAllocators.devTaskBoundaryOutcasts.availableSlots_;

        WsSlotAllocator::BlockHeader *base = allocator.devTaskBoundaryOutcasts.GetBlockHeaderBase();
        Restore::RestoreBlockHeader(allocator.devTaskBoundaryOutcasts.freeListHeader_, base, runtimeBackup.workspace.tensorAllocators.devTaskBoundaryOutcasts.freeListHeader_);
        Restore::RestoreBlockHeader(allocator.devTaskBoundaryOutcasts.notInUseHeaders_, base, runtimeBackup.workspace.tensorAllocators.devTaskBoundaryOutcasts.notInUseHeaders_);
        WsSlotAllocator::BlockHeader *checkpointBase = runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.Data();
        for (uint64_t k = 0; k < allocator.devTaskBoundaryOutcasts.slotNum_; k++) {
            Restore::RestoreBlockHeader(base[k].listNext, base, checkpointBase[k].listNext);
        }
    }

    void RuntimeAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram) {
        RelocRange relocProgram(srcProgram, dstProgram);
        {
            auto &slotList = runtimeBackup.slotContext.slotList;
            DeviceExecuteSlot *base = slotList.Data();
            uint64_t size = slotList.size();
            for (uint64_t k = 0; k < size; k++) {
                relocProgram.RelocNullable(base[k].partialUpdate);
            }
        }
    }

    void RuntimeAddrRelocWorkspace(
            uint64_t srcWorkspace, uint64_t dstWorkspace,
            DevStartArgsBase *devStartArgs, DeviceExecuteSlot *runtimeSlotList,
            ItemPool<RuntimeOutcastTensor>::ItemBlock *runtimeOutcastTensorPool) {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        /* empty constructor's overhead should be negligible */
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        if (devStartArgs == nullptr) {
            /* only run on host */
            RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
        }
        {
            auto &slottedOutcastsBlockList = runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList;
            WsSlotAllocator::BlockHeader *base = slottedOutcastsBlockList.Data();
            uint64_t size = slottedOutcastsBlockList.size();
            for (uint64_t k = 0; k < size; k++) {
                relocWorkspace.RelocNullable(base[k].ptr);
            }
        }
        {
            auto &slotList = runtimeBackup.slotContext.slotList;
            DeviceExecuteSlot *base = slotList.Data();
            ItemPool<RuntimeOutcastTensor>::ItemBlock *backupRtOutcastPool = runtimeBackup.workspace.runtimeOutcastTensorPool.Data();
            uint64_t size = slotList.size();
            for (uint64_t k = 0; k < size; k++) {
                static_assert(sizeof(AddressDescriptor) == sizeof(uintdevptr_t),
                    "Please review the following logics when the condition does not hold anymore.");
                if (devStartArgs == nullptr) {
                    // Host: addr uses backup
                    if (base[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) { continue; }

                    auto &rtOutcast = backupRtOutcastPool[base[k].rtOutcastIter].Item();
                    if (rtOutcast.isCache) { continue; } // To avoid duplicate reloc
                    rtOutcast.isCache = true;

                    uintdevptr_t addr = rtOutcast.addr;
                    AddressDescriptor *desc = reinterpret_cast<AddressDescriptor *>(&rtOutcast.addr);
                    *desc = AddressDescriptor::MakeFromAddress(addr);
                    RelocDescToCache(*desc, relocWorkspace, cacheInputOutputDict);
                } else {
                    // Device: addr uses actual
                    if (runtimeSlotList[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) { continue; }

                    auto &rtOutcast = runtimeOutcastTensorPool[runtimeSlotList[k].rtOutcastIter].Item();
                    if (!rtOutcast.isCache) { continue; } // To avoid duplicate reloc
                    rtOutcast.isCache = false;

                    AddressDescriptor *desc = reinterpret_cast<AddressDescriptor *>(&rtOutcast.addr);
                    RelocDescFromCache(*desc, relocWorkspace, devStartArgs);
                    rtOutcast.addr = desc->GetAddressValue();
                }
            }
        }
    }

    void MixTaskDataReloc(RelocRange &relocProgram, DynDeviceTaskBase *dynTaskBase, DynFuncHeader *dynFuncDataList) {
        if (dynTaskBase->devTask.mixTaskData.wrapIdNum == 0) {
            return;
        }
        relocProgram.Reloc(dynTaskBase->devTask.mixTaskData.wrapTasklist);
        WrapInfoQueue *tmpWrapInfoQueue = reinterpret_cast<WrapInfoQueue *>(dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);
        WrapInfoQueue *&wrapInfoQueueRef = tmpWrapInfoQueue;
        WrapInfoQueue *wrapInfoQueue = RelocControlFlowCachePointer(wrapInfoQueueRef, relocProgram);

        WrapInfo *&wrapInfoElemRef = wrapInfoQueue->elem;
        WrapInfo *wrapInfoElem = RelocControlFlowCachePointer(wrapInfoElemRef, relocProgram);
        for (uint32_t i = wrapInfoQueue->head; i < wrapInfoQueue->tail; i++) {
            WrapInfo *wrapInfo = wrapInfoElem + i;
            relocProgram.Reloc(wrapInfo->tasklist.elem);
        }

        MixTaskDataCache *&mixTaskDataBackupRef = dynTaskBase->mixTaskDataBackup;
        MixTaskDataCache *mixTaskDataBackup = RelocControlFlowCachePointer(mixTaskDataBackupRef, relocProgram);
        relocProgram.Reloc(mixTaskDataBackup->wrapTasklist);
        WrapInfo *&wrapInfoBackupElemRef = mixTaskDataBackup->queue.elem;
        WrapInfo *wrapInfoBackupElem = RelocControlFlowCachePointer(wrapInfoBackupElemRef, relocProgram);
        for (uint32_t i = 0; i < wrapInfoQueue->tail; i++) {
            WrapInfo *wrapInfo = wrapInfoBackupElem + i;
            relocProgram.Reloc(wrapInfo->tasklist.elem);
        }

        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            relocProgram.Reloc(dynTaskBase->devTask.mixTaskData.opWrapList[dupIndex]);
            relocProgram.Reloc(dynTaskBase->devTask.mixTaskData.opWrapTaskNumList[dupIndex]);
            relocProgram.Reloc(mixTaskDataBackup->opWrapList[dupIndex]);
            relocProgram.Reloc(mixTaskDataBackup->opWrapTaskNumList[dupIndex]);
        }
    }

    /* Host-to-cache: devStartArgs should be nullptr. Cache-to-Device: devStartArgs should be filled */
    void TaskAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram) {
        RelocRange relocProgram(srcProgram, dstProgram);
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            /* When cached, the pointer is always legal */
            DynDeviceTaskBase *&dynTaskBaseRef = deviceTaskCacheList[deviceIndex].dynTaskBase;
            DynDeviceTaskBase *dynTaskBase = RelocControlFlowCachePointer(dynTaskBaseRef, relocProgram);
            relocProgram.Reloc(dynTaskBase->devTask.readyAivCoreFunctionQue);
            relocProgram.Reloc(dynTaskBase->devTask.readyAicCoreFunctionQue);
            relocProgram.Reloc(dynTaskBase->devTask.readyAicpuFunctionQue);
            for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
                ReadyCoreFunctionQueue *&readyQueueRef = dynTaskBase->readyQueue[i];
                ReadyCoreFunctionQueue *readyQueue = RelocControlFlowCachePointer(readyQueueRef, relocProgram);
                relocProgram.Reloc(readyQueue->elem);
            }
            relocProgram.Reloc(dynTaskBase->cceBinary);
            relocProgram.Reloc(dynTaskBase->aicpuLeafBinary);

            ReadyQueueCache *&readyQueueBackupRef = dynTaskBase->readyQueueBackup;
            ReadyQueueCache *readyQueueBackup = RelocControlFlowCachePointer(readyQueueBackupRef, relocProgram);
            for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
                relocProgram.Reloc(readyQueueBackup->queueList[i].elem);
            }

            DynFuncHeader *&dynFuncDataListRef = dynTaskBase->dynFuncDataList;
            DynFuncHeader *dynFuncDataList = RelocControlFlowCachePointer(dynFuncDataListRef, relocProgram);
            DynFuncDataCache *dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup *dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            MixTaskDataReloc(relocProgram, dynTaskBase, dynFuncDataList);
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                DevAscendFunctionDuppedData *&duppedDataRef = dynDataCache->duppedData;
                DevAscendFunctionDuppedData *duppedData = RelocControlFlowCachePointer(duppedDataRef, relocProgram);

                // Reloc Stitch
                for (uint32_t i = 0; i < duppedData->GetStitchSize(); i++) {
                    DevAscendFunctionDuppedStitchList &stitchList = duppedData->GetStitch(i);
                    DevAscendFunctionDuppedStitch *&stitchRef = stitchList.Head();
                    for (DevAscendFunctionDuppedStitch **nodePtr = &stitchRef; *nodePtr != nullptr; ) {
                        DevAscendFunctionDuppedStitch *node = RelocControlFlowCachePointer(*nodePtr, relocProgram);
                        nodePtr = &node->Next();
                    }
                }

                // Reloc Dupped
                relocProgram.Reloc(duppedData->source_);

                // Reloc DynFuncData
                relocProgram.Reloc(dynData->opAttrs);
                relocProgram.Reloc(dynData->opAtrrOffsets);
                relocProgram.Reloc(dynData->exprTbl);
                relocProgram.Reloc(dynData->rawTensorDesc);
                relocProgram.Reloc(dynData->rawTensorAddr);

                relocProgram.Reloc(dynDataCache->devFunc);
                relocProgram.Reloc(dynDataCache->predCount);
                relocProgram.Reloc(dynDataCache->calleeList);
                relocProgram.RelocNullable(dynDataBackup->predCountBackup);
                relocProgram.RelocNullable(dynDataBackup->rawTensorAddrBackup);
            }
        }
    }
};

#define ControlFlowAllocateSlab(devProg, size, expr) \
    ({ \
        WsAllocation ws; \
        DevProgramControlFlowCache *c = (devProg)->GetControlFlowCache(); \
        if (c->IsRecording()) { \
            void *ptr = c->AllocateCache(size); \
            if (ptr != nullptr) { \
                ws.ptr = reinterpret_cast<uintdevptr_t>(ptr); \
            } else { \
                ws = (expr); \
            } \
        } else { \
            ws = (expr); \
        } \
        ws; \
    })
}