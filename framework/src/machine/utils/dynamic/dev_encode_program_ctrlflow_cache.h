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
#define ADDRESS_CACHE_KIND_WORKSPACE 0
#define ADDRESS_CACHE_KIND_INPUT 1
#define ADDRESS_CACHE_KIND_OUTPUT 2
#define ADDRESS_CACHE_KIND_COMM 3
#define INVALID_STITCH_IDX (static_cast<uint32_t>(-1))

constexpr size_t READY_QUEUE_SIZE = 3UL;
constexpr size_t DIE_READY_QUEUE_SIZE = 2UL;
inline constexpr size_t MAX_STITCH_FUNC_NUM = 1024;

inline constexpr size_t DEFAULT_STITCH_CFGCACHE_SIZE = 100000000;

struct ReadyQueueCache {
    uint32_t coreFunctionCnt;
    ReadyCoreFunctionQueueUnsafe queueList[READY_QUEUE_SIZE];
    uint32_t readyTaskNum;
};

struct DieReadyQueueCache {
    uint32_t coreFunctionCnt;
    ReadyCoreFunctionQueueUnsafe queueList[DIE_READY_QUEUE_SIZE * DIE_NUM];
    uint32_t readyTaskNum;
};

struct MixTaskDataCache {
    WrapInfoQueue queue;
    uint64_t wrapIdNum;
    uint64_t opWrapList[MAX_STITCH_FUNC_NUM];
    uint16_t* opWrapOffsetList[MAX_STITCH_FUNC_NUM];
    StaticReadyCoreFunctionQueue wrapQueueForThread[MAX_SCHEDULE_AICPU_NUM];
};

struct DynFuncDataCache {
    DevAscendFunction* devFunc;
    predcount_t* predCount;
    int* calleeList;
    DevAscendFunctionDuppedData* duppedData;

    const DynFuncDataCache& At(size_t index) const { return this[index]; }
    DynFuncDataCache& At(size_t index) { return this[index]; }
};

struct DynFuncDataWorkspaceAddressBackup {
    uint64_t runtimeWorkspace;
    uint64_t runtimeOutcastWorkspace;
    uint64_t workspaceAddr;
    uint64_t stackWorkspaceAddr;
};

struct DynFuncDataBackup {
    predcount_t* predCountBackup;
    uint64_t* rawTensorAddrBackup;
    uint64_t* deadEndHubBitmapBackup{nullptr};
    uint64_t* tailTaskBitmapBackup{nullptr};
    size_t bitmapByteSize{0};

    DynFuncDataWorkspaceAddressBackup workspaceAddressBackup;

    const DynFuncDataBackup& At(size_t index) const { return this[index]; }
    DynFuncDataBackup& At(size_t index) { return this[index]; }
};

struct ParallelInfo {
    uint32_t parallelism{1};
    uint32_t forId{0};
    uint32_t iterId{0};
    uint32_t wsId{0}; // parallel dev task use different workspace
};

struct DynDeviceTaskBase {
    DeviceTask devTask;
    DynFuncHeader* dynFuncDataList{nullptr};

    ReadyCoreFunctionQueue* readyQueue[READY_QUEUE_SIZE];
    DynFuncDataCache dynFuncDataCacheList[MAX_STITCH_FUNC_NUM];
    uint64_t dynFuncDataCacheListSize;

    const DevCceBinary* cceBinary;
    const DevAicpuLeafBinary* aicpuLeafBinary;

    ReadyQueueCache* readyQueueBackup;
    DieReadyQueueCache* dieReadyQueueBackup{nullptr};
    MixTaskDataCache* mixTaskDataBackup{nullptr};
    DynFuncDataBackup dynFuncDataBackupList[MAX_STITCH_FUNC_NUM];
    void* shmemWaitUntilCacheBackup{nullptr};
    bool isLastTask{false};
    bool isParallelSameIterLastTask{false};
    ParallelInfo parallelInfo;

    DynFuncHeader* GetDynFuncDataList() const { return dynFuncDataList; }
    DynFuncHeader* GetDynFuncDataList() { return dynFuncDataList; }
    const DynFuncDataCache* GetDynFuncDataCacheList() const { return dynFuncDataCacheList; }
    DynFuncDataCache* GetDynFuncDataCacheList() { return dynFuncDataCacheList; }

    uint64_t GetIndex() { return GetDynFuncDataList()->GetIndex(); }
    inline bool IsLastTask() const { return isLastTask; }
    void SetLastTask(bool b) { isLastTask = b; }
    uint32_t ParallelForId() { return parallelInfo.forId; }
    uint32_t ParallelIterId() { return parallelInfo.iterId; }
    uint32_t ParallelWsId() { return parallelInfo.wsId; }
    bool SupportParallel() { return parallelInfo.forId != 0; }
    void SetParallelInfo(ParallelInfo info) { parallelInfo = info; }
    bool IsParallelSameIterLastDevTask() { return isParallelSameIterLastTask; }
    void SetParallelSameIterLastDevTask(bool isLast) { isParallelSameIterLastTask = isLast; }
    uint32_t maxC_{0};
    uint32_t maxV_{0};

    uint32_t GetMaxC() const { return maxC_; }
    uint32_t GetMaxV() const { return maxV_; }
    void SetMaxCV(uint32_t maxC, uint32_t maxV) { maxC_ = maxC; maxV_ = maxV; }
};

struct DeviceTaskCache {
    DynDeviceTaskBase* dynTaskBase;
};

struct DeviceExecuteSlot {
    ItemPoolIter rtOutcastIter{ITEM_POOL_INVALID_INDEX};
    bool isOutputSlot{false};
    bool isAssembleSlot{false};
    bool isAssembleSlotNeedAlloc{false};
    bool isOutputTensorNeedCellMatch{false};
    bool isPartialUpdateStitch{false};
    uint32_t stitchDupIdx{INVALID_STITCH_IDX};
    uint32_t stitchOutcastIdx;
    uint32_t slotAllocIterId{0}; // when alloc new tensor memory ,change it for cell match tag check

    DevAscendProgramPartialUpdate* partialUpdate{nullptr};

    bool IsOutputAddress() const { return isOutputSlot; }
    bool IsAssembleAddress() const { return isAssembleSlot; }
    void ChangeSlotAllocIterId() {
        slotAllocIterId++;
        if (slotAllocIterId == MAX_STITCH_FUNC_NUM -1) {
            slotAllocIterId = 0;
        }
    }
};

struct DevControlFlowCacheRuntime {
    struct DeviceWorkspaceAllocator {
        struct {
            SeqWsAllocator rootInner;
            SeqWsAllocator devTaskInnerExclusiveOutcasts;
            WsSlotAllocator devTaskBoundaryOutcasts;
            DevRelocVector<WsSlotAllocator::BlockHeader> slottedOutcastsBlockList;
        } tensorAllocators[SCH_DEVTASK_MAX_PARALLELISM];
        DevRelocVector<ItemPool<RuntimeOutcastTensor>::ItemBlock> runtimeOutcastTensorPool;
        ItemPoolMeta itemPoolMeta;
    } workspace;
    struct DeviceSlotContext {
        DevRelocVector<DeviceExecuteSlot> slotList;
    } slotContext;
};

template <typename T>
inline T* RelocControlFlowCachePointer(T*& ptrRef, const RelocRange& relocProgram)
{
    T* result = nullptr;
    if (relocProgram.GetDst() == 0) {
        result = ptrRef;
        relocProgram.Reloc(ptrRef);
    } else {
        relocProgram.Reloc(ptrRef);
        result = ptrRef;
    }
    return result;
}

struct DevControlFlowCache {
    uint64_t allCacheSize{0};
    /* actual used cache size */
    uint64_t usedCacheSize{0};
    /* Filled by user, true means try to allocate in cache. */
    bool isRecording{false};
    /* Filled by user, true means activate in cache. */
    bool isActivated{false};
    /* reloc meta at device */
    bool isRelocMetaDev{false};
    /* reloc data at device */
    bool isRelocDataDev{false};
    /* cache shape is origin or infer shape */
    bool isCacheOriginShape{true};
    /* Filled in caching */
    DevRelocVector<DevTensorData> inputTensorDataList;
    /* Filled in caching */
    DevRelocVector<DevTensorData> outputTensorDataList;
    /* Filled in caching for runtime */
    DevControlFlowCacheRuntime runtimeBackup;

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
#define ctrlFlowLastField cacheData
    uint64_t dataSize;
    uint8_t data[0];

    bool inline IsRecording() const
    {
        if (IsDeviceMode()) {
            return false;
        }
        return isRecording;
    }

    bool inline IsRecordingStopped() const { return isRecordingStopped; }

    bool inline IsCacheOriginShape() const { return isCacheOriginShape; }

    void inline StopRecording() { isRecordingStopped = true; }

    void inline CalcUsedCacheSize()
    {
        usedCacheSize =
            reinterpret_cast<uintptr_t>(cacheData.Data()) + cacheDataOffset - reinterpret_cast<uintptr_t>(this);
    }

    void Init(
        void* dyndevAttrPtr, uint64_t cacheSize, uint64_t runtimeOutcastPoolSize, uint64_t& initOffset,
        uint32_t stitchMaxFunctionNum);
    uint64_t GetSize() const
    {
        return reinterpret_cast<uintptr_t>(ctrlFlowLastField.End()) - reinterpret_cast<uintptr_t>(this);
    }

#define CFGCACHE_ALIGN 8
    void* AllocateCache(uint64_t size)
    {
        void* result = nullptr;
        if (cacheDataOffset + size < cacheData.size()) {
            result = &cacheData[cacheDataOffset];
            /* make cache 8 byte aligned */
            cacheDataOffset += (size + CFGCACHE_ALIGN - 1) / CFGCACHE_ALIGN * CFGCACHE_ALIGN;
            DEV_VERBOSE_DEBUG("cacheDataOffset is: %lu", cacheDataOffset);
        } else {
            DEV_DEBUG(
                "[ctrl.cache.record] Recording is stopped, requestSize=%lu, "
                "cacheDataOffset=%lu, cacheDataSize=%lu.",
                size, cacheDataOffset,cacheData.size());
            
            isRecordingStopped = true;
        }
        return result;
    }

    bool AppendDeviceTask(DynDeviceTaskBase* base)
    {
        if (!isRecordingStopped && (deviceTaskCount < deviceTaskCacheList.size())) {
            deviceTaskCacheList[deviceTaskCount].dynTaskBase = base;
            deviceTaskCount += 1;
            rootTaskCount += base->dynFuncDataList->Size();
            DEV_DEBUG("deviceTaskCount=%lu", deviceTaskCount);
            return true;
        } else {
            deviceTaskSkippedCount += 1;
            return false;
        }
    }

    void InitInputOutput(DevStartArgsBase* startArgs)
    {
        for (size_t i = 0; i < inputTensorDataList.size(); i++) {
            inputTensorDataList[i] = startArgs->GetInputTensor(i);
        }
        for (size_t i = 0; i < outputTensorDataList.size(); i++) {
            outputTensorDataList[i] = startArgs->GetOutputTensor(i);
        }
    }

    void MatchInputOutputDump(DevStartArgsBase* startArgs) const
    {
        DEV_VERBOSE_DEBUG("matchio cache input size: %d", (int)inputTensorDataList.size());
        for (size_t k = 0; k < inputTensorDataList.size(); k++) {
            DEV_VERBOSE_DEBUG("matchio cache input %d: %s", (int)k, DumpShape(inputTensorDataList[k].shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio cache output size: %d", (int)outputTensorDataList.size());
        for (size_t k = 0; k < outputTensorDataList.size(); k++) {
            DEV_VERBOSE_DEBUG("matchio cache output %d: %s", (int)k, DumpShape(outputTensorDataList[k].shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio real input size: %d", (int)startArgs->inputTensorSize);
        for (size_t k = 0; k < startArgs->inputTensorSize; k++) {
            DEV_VERBOSE_DEBUG(
                "matchio real input %d: %s", (int)k, DumpShape(startArgs->GetInputTensor(k).shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio real output size: %d", (int)startArgs->outputTensorSize);
        for (size_t k = 0; k < startArgs->outputTensorSize; k++) {
            DEV_VERBOSE_DEBUG(
                "matchio real output %d: %s", (int)k, DumpShape(startArgs->GetOutputTensor(k).shape).c_str());
        }
    }

    inline bool MatchInputOutput(DevStartArgsBase* startArgs) const
    {
        MatchInputOutputDump(startArgs);

        if (inputTensorDataList.size() != startArgs->inputTensorSize) {
            return false;
        }
        if (outputTensorDataList.size() != startArgs->outputTensorSize) {
            return false;
        }
        // support infer controlflow cache now, cached shape and realshape may not match now
        return true;
    }

    inline bool IsActivatedFullCache(DevStartArgsBase* startArgs) const
    {
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

    inline bool IsActivatedPartialCache(DevStartArgsBase* startArgs) const
    {
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

    void BitmapDataBackup(DynFuncDataCache* dynDataCache, DynFuncDataBackup* dynDataBackup)
    {
        auto* devFunc = dynDataCache->devFunc;
        size_t bitmapBytes = devFunc->GetBitmapByteSize();
        if (bitmapBytes == 0) {
            return;
        }

        uint64_t* deadEndBuf = reinterpret_cast<uint64_t*>(AllocateCache(bitmapBytes));
        uint64_t* tailBuf = reinterpret_cast<uint64_t*>(AllocateCache(bitmapBytes));
        if (deadEndBuf == nullptr || tailBuf == nullptr) {
            return;
        }

        dynDataBackup->deadEndHubBitmapBackup = deadEndBuf;
        dynDataBackup->tailTaskBitmapBackup = tailBuf;
        dynDataBackup->bitmapByteSize = bitmapBytes;
        devFunc->BackupBitmapTo(deadEndBuf, tailBuf, bitmapBytes);
    }

    void BitmapDataRestore(DynFuncDataCache* dynDataCache, DynFuncDataBackup* dynDataBackup)
    {
        if (dynDataBackup->bitmapByteSize == 0) {
            return;
        }
        dynDataCache->devFunc->RestoreBitmapFrom(
            dynDataBackup->deadEndHubBitmapBackup, dynDataBackup->tailTaskBitmapBackup,
            dynDataBackup->bitmapByteSize);
    }

    void PredCountDataBackup(DynDeviceTaskBase* base)
    {
        DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

            predcount_t* predCountBackup = reinterpret_cast<predcount_t*>(AllocateCache(backupSize));
            if (predCountBackup == nullptr) {
                return;
            }
            dynDataBackup->predCountBackup = predCountBackup;
            DevMemcpyS(dynDataBackup->predCountBackup, backupSize, &duppedData->GetOperationCurrPredCount(0), backupSize);

            BitmapDataBackup(dynDataCache, dynDataBackup);
        }
    }

    void PredCountDataRestore(DynDeviceTaskBase* base)
    {
        DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();
            DevMemcpyS(&duppedData->GetOperationCurrPredCount(0), backupSize, dynDataBackup->predCountBackup, backupSize);

            BitmapDataRestore(dynDataCache, dynDataBackup);
        }
    }

    void ReadyQueueDataBackup(DynDeviceTaskBase* base)
    {
        ReadyQueueCache* readyQueueBackup = reinterpret_cast<ReadyQueueCache*>(AllocateCache(sizeof(ReadyQueueCache)));
        if (readyQueueBackup == nullptr) {
            return;
        }
        readyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
        uint32_t readyTaskNum = 0;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->Capacity();
            uint32_t* readyQueueBackupElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
            if (readyQueueBackupElem == nullptr) {
                return;
            }

            new (&readyQueueBackup->queueList[i])
                ReadyCoreFunctionQueueUnsafe(base->readyQueue[i]->Capacity(), readyQueueBackupElem);
            readyQueueBackup->queueList[i] = *base->readyQueue[i];
            readyTaskNum += base->readyQueue[i]->UnsafeSize();
        }
        readyQueueBackup->readyTaskNum = readyTaskNum;
        base->readyQueueBackup = readyQueueBackup;
    }

    void ReadyQueueDataRestore(DynDeviceTaskBase* base)
    {
        ReadyQueueCache* readyQueueBackup = base->readyQueueBackup;
        base->devTask.coreFunctionCnt = readyQueueBackup->coreFunctionCnt;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            *base->readyQueue[i] = readyQueueBackup->queueList[i];
        }
    }

    void DieReadyQueueDataBackup(DynDeviceTaskBase* base)
    {
        DieReadyQueueCache* dieReadyQueueBackup =
            reinterpret_cast<DieReadyQueueCache*>(AllocateCache(sizeof(DieReadyQueueCache)));
        if (dieReadyQueueBackup == nullptr) {
            return;
        }
        dieReadyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
        uint32_t readyTaskNum = 0;
        for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
            ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
                (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                                base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
            if (dieReadyQueue == nullptr) {
                new (&dieReadyQueueBackup->queueList[i]) ReadyCoreFunctionQueueUnsafe(0, nullptr);
                continue;
            }
            size_t backupSize = sizeof(uint32_t) * dieReadyQueue->Capacity();
            uint32_t* dieReadyQueueBackupElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
            if (dieReadyQueueBackupElem == nullptr) {
                return;
            }

            new (&dieReadyQueueBackup->queueList[i])
                ReadyCoreFunctionQueueUnsafe(dieReadyQueue->Capacity(), dieReadyQueueBackupElem);
            dieReadyQueueBackup->queueList[i] = *dieReadyQueue;
            readyTaskNum += dieReadyQueue->UnsafeSize();
        }
        dieReadyQueueBackup->readyTaskNum = readyTaskNum;
        base->dieReadyQueueBackup = dieReadyQueueBackup;
    }

    void DieReadyQueueDataRestore(DynDeviceTaskBase* base)
    {
        DieReadyQueueCache* dieReadyQueueBackup = base->dieReadyQueueBackup;
        if (dieReadyQueueBackup == nullptr) {
            return;
        }
        base->devTask.coreFunctionCnt = dieReadyQueueBackup->coreFunctionCnt;
        for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
            ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
                (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                                base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
            if (dieReadyQueue == nullptr) {
                continue;
            }
            *dieReadyQueue = dieReadyQueueBackup->queueList[i];
        }
    }

    bool BackupOpWrapOffsetList(DynDeviceTaskBase* base, MixTaskDataCache* mixTaskDataBackup)
    {
        for (uint32_t dupIndex = 0; dupIndex < base->dynFuncDataList->funcNum; dupIndex++) {
            auto funcWrapIdNum = base->dynFuncDataCacheList[dupIndex].devFunc->wrapIdNum_;
            if (funcWrapIdNum == 0) {
                continue;
            }
            size_t wrapOffsetBackupSize = funcWrapIdNum * sizeof(uint16_t);
            uint16_t* wrapOffsetBackupElem = reinterpret_cast<uint16_t*>(AllocateCache(wrapOffsetBackupSize));
            if (wrapOffsetBackupElem == nullptr) {
                return false;
            }
            mixTaskDataBackup->opWrapOffsetList[dupIndex] = wrapOffsetBackupElem;
            uint16_t* list = base->devTask.mixTaskData.opWrapOffsetList[dupIndex];
            DevMemcpyS(wrapOffsetBackupElem, wrapOffsetBackupSize, list, wrapOffsetBackupSize);
        }
        return true;
    }

    void MixTaskDataBackup(DynDeviceTaskBase* base)
    {
        if (base->devTask.mixTaskData.wrapIdNum == 0) {
            return;
        }
        MixTaskDataCache* mixTaskDataBackup =
            reinterpret_cast<MixTaskDataCache*>(AllocateCache(sizeof(MixTaskDataCache)));
        if (mixTaskDataBackup == nullptr) {
            return;
        }
        mixTaskDataBackup->wrapIdNum = base->devTask.mixTaskData.wrapIdNum;
        WrapInfoQueue* wrapInfoQueue =
            reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
        size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->capacity;
        WrapInfo* wrapQueueBackupElem = reinterpret_cast<WrapInfo*>(AllocateCache(wrapInfoBackupSize));
        if (wrapQueueBackupElem == nullptr) {
            return;
        }
        mixTaskDataBackup->queue.head = wrapInfoQueue->head;
        mixTaskDataBackup->queue.tail = wrapInfoQueue->tail;
        mixTaskDataBackup->queue.capacity = wrapInfoQueue->capacity;
        mixTaskDataBackup->queue.elem = wrapQueueBackupElem;
        DevMemcpyS(mixTaskDataBackup->queue.elem, wrapInfoBackupSize, wrapInfoQueue->elem, wrapInfoBackupSize);

        size_t wrapPtrBackupSize = sizeof(uint64_t) * wrapInfoQueue->capacity; // 与wrapInfoQueue元素数量相同
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; i++) {
            StaticReadyCoreFunctionQueue* wrapQueueForThread =
                reinterpret_cast<StaticReadyCoreFunctionQueue*>(base->devTask.mixTaskData.wrapQueueForThread[i]);
            uint64_t* wrapPtrQueueBackupElem = reinterpret_cast<uint64_t*>(AllocateCache(wrapPtrBackupSize));
            if (wrapPtrQueueBackupElem == nullptr) {
                return;
            }
            mixTaskDataBackup->wrapQueueForThread[i].head = wrapQueueForThread->head;
            mixTaskDataBackup->wrapQueueForThread[i].tail = wrapQueueForThread->tail;
            mixTaskDataBackup->wrapQueueForThread[i].elem = wrapPtrQueueBackupElem;
            DevMemcpyS(
                mixTaskDataBackup->wrapQueueForThread[i].elem, wrapPtrBackupSize, wrapQueueForThread->elem,
                wrapPtrBackupSize);
        }

        constexpr size_t arrSize = sizeof(uint64_t) * MAX_STITCH_FUNC_NUM;
        DevMemcpyS(mixTaskDataBackup->opWrapList, arrSize, base->devTask.mixTaskData.opWrapList, arrSize);

        if (!BackupOpWrapOffsetList(base, mixTaskDataBackup)) {
            return;
        }
        base->mixTaskDataBackup = mixTaskDataBackup;
    }

    void MixTaskDataRestore(DynDeviceTaskBase* base)
    {
        if (base->mixTaskDataBackup == nullptr) {
            return;
        }
        MixTaskDataCache* mixTaskDataBackup = base->mixTaskDataBackup;
        base->devTask.mixTaskData.wrapIdNum = mixTaskDataBackup->wrapIdNum;

        WrapInfoQueue* wrapInfoQueue =
            reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
        wrapInfoQueue->head = mixTaskDataBackup->queue.head;
        wrapInfoQueue->tail = mixTaskDataBackup->queue.tail;
        wrapInfoQueue->capacity = mixTaskDataBackup->queue.capacity;

        size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->capacity;
        DevMemcpyS(wrapInfoQueue->elem, wrapInfoBackupSize, mixTaskDataBackup->queue.elem, wrapInfoBackupSize);

        size_t wrapPtrBackupSize = sizeof(uint64_t) * wrapInfoQueue->capacity; // 与wrapInfoQueue元素数量相同
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; i++) {
            StaticReadyCoreFunctionQueue* wrapQueueForThread =
                reinterpret_cast<StaticReadyCoreFunctionQueue*>(base->devTask.mixTaskData.wrapQueueForThread[i]);
            wrapQueueForThread->head = mixTaskDataBackup->wrapQueueForThread[i].head;
            wrapQueueForThread->tail = mixTaskDataBackup->wrapQueueForThread[i].tail;
            DevMemcpyS(
                wrapQueueForThread->elem, wrapPtrBackupSize, mixTaskDataBackup->wrapQueueForThread[i].elem,
                wrapPtrBackupSize);
        }

        constexpr size_t arrSize = sizeof(uint64_t) * MAX_STITCH_FUNC_NUM;
        DevMemcpyS(base->devTask.mixTaskData.opWrapList, arrSize, mixTaskDataBackup->opWrapList, arrSize);

        for (uint32_t dupIndex = 0; dupIndex < base->dynFuncDataList->funcNum; dupIndex++) {
            auto funcWrapIdNum = base->dynFuncDataCacheList[dupIndex].devFunc->wrapIdNum_;
            auto& list = base->devTask.mixTaskData.opWrapOffsetList[dupIndex];
            if (funcWrapIdNum == 0) {
                list = nullptr;
                continue;
            }
            size_t wrapOffsetBackupSize = funcWrapIdNum * sizeof(uint16_t);
            DevMemcpyS(list, wrapOffsetBackupSize, mixTaskDataBackup->opWrapOffsetList[dupIndex], wrapOffsetBackupSize);
        }
    }

    static void RelocBuildInputOutputDesc(
        std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict, DevStartArgsBase* devStartArgs)
    {
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
        std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict,
        DevRelocVector<DevTensorData> inputTensorDataList, DevRelocVector<DevTensorData> outputTensorDataList)
    {
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
        AddressDescriptor& desc, const RelocRange& relocWorkspace,
        std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict)
    {
        AddressDescriptor resultDesc;
        uint64_t addr = desc.GetAddressValue();
        if (cacheInputOutputDict.count(addr)) {
            resultDesc = cacheInputOutputDict[addr];
        } else if (addr & (1UL << 58)) {
            resultDesc = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_COMM, addr);
        } else {
            relocWorkspace.Reloc(addr);
            resultDesc = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_WORKSPACE, addr);
        }
        desc = resultDesc;
    }

    static void RelocDescFromCache(
        AddressDescriptor& desc, const RelocRange& relocWorkspace, DevStartArgsBase* devStartArgs)
    {
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
            case ADDRESS_CACHE_KIND_COMM:
                resultAddr = desc.cacheValue;
                break;
            default:
                DEV_ERROR(
                    ProgEncodeErr::CACHE_RELOC_KIND_INVALID,
                    "#ctrl.task.pre.cache.reloc: [RelocDescFromCache] Invalid kind: %lu\n",
                    (unsigned long)desc.cacheKind);
                break;
        }
        AddressDescriptor resultDesc = AddressDescriptor::MakeFromAddress(resultAddr);
        desc = resultDesc;
    }

    void IncastOutcastAddrBackup(DynDeviceTaskBase* base)
    {
        DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

            uint64_t* rawTensorAddrBackup = reinterpret_cast<uint64_t*>(AllocateCache(backupSize));
            if (rawTensorAddrBackup == nullptr) {
                return;
            }
            dynDataBackup->rawTensorAddrBackup = rawTensorAddrBackup;
            DevMemcpyS(dynDataBackup->rawTensorAddrBackup, backupSize, dynData->rawTensorAddr, backupSize);
        }
    }

    void IncastOutcastAddrRestore(DynDeviceTaskBase* base)
    {
        DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

            DevMemcpyS(dynData->rawTensorAddr, backupSize, dynDataBackup->rawTensorAddrBackup, backupSize);
        }
    }

    void IncastOutcastAddrRestore()
    {
        for (size_t i = 0; i < deviceTaskCount; i++) {
            DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
            IncastOutcastAddrRestore(dynTaskBase);
        }
    }

    void TaskAddrBackupWorkspace(DynDeviceTaskBase* base)
    {
        DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

            dynDataBackup->workspaceAddressBackup.runtimeWorkspace = duppedData->runtimeWorkspace_;
            dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace = duppedData->runtimeOutcastWorkspace_;
            dynDataBackup->workspaceAddressBackup.workspaceAddr = dynData->workspaceAddr;
            dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr = dynData->stackWorkSpaceAddr;
        }
    }

    void TaskAddrRestoreWorkspace(DynDeviceTaskBase* base)
    {
        DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

            duppedData->runtimeWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeWorkspace;
            duppedData->runtimeOutcastWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace;
            dynData->workspaceAddr = dynDataBackup->workspaceAddressBackup.workspaceAddr;
            dynData->stackWorkSpaceAddr = dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr;
        }
    }

    void TaskAddrRestoreWorkspace()
    {
        for (size_t i = 0; i < deviceTaskCount; i++) {
            DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
            TaskAddrRestoreWorkspace(dynTaskBase);
        }
    }

    void TaskAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace, DevStartArgsBase* devStartArgs)
    {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;

            DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
            DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
                DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

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

    void IncastOutcastAddrReloc(uint64_t srcWorkspace, uint64_t dstWorkspace, DevStartArgsBase* devStartArgs)
    {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        /* empty constructor's overhead should be negligible */
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        if (devStartArgs == nullptr) {
            /* only run on host */
            RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
        }
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;
            DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
            DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
                if (devStartArgs == nullptr) {
                    // Host: addr uses backup
                    for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                        AddressDescriptor* addr =
                            reinterpret_cast<AddressDescriptor*>(dynDataBackup->rawTensorAddrBackup + i);
                        RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                    }
                    for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                        AddressDescriptor* addr = reinterpret_cast<AddressDescriptor*>(
                            dynDataBackup->rawTensorAddrBackup + duppedData->GetIncastSize() + i);
                        RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                    }
                } else {
                    // Device: addr uses actual
                    for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                        AddressDescriptor* addr = &duppedData->GetIncastAddress(i);
                        RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                    }
                    for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                        AddressDescriptor* addr = &duppedData->GetOutcastAddress(i);
                        RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                    }
                }

                dynData->startArgs = devStartArgs;
            }
        }
    }

    void RuntimeAddrBackup(
        DeviceExecuteSlot* runtimeSlotList, ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool,
        uint64_t slotSize, uint64_t runtimeOutcastTensorSize, TensorAllocator* allocator, uint32_t parallelism)
    {
        uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
        uint64_t runtimeOutcastPoolDataSize =
            sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
        (void)memcpy_s(runtimeBackup.slotContext.slotList.Data(), slotDataSize, runtimeSlotList, slotDataSize);
        auto itemBlockBase = reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool->At(0));
        (void)memcpy_s(
            runtimeBackup.workspace.runtimeOutcastTensorPool.Data(), runtimeOutcastPoolDataSize,
            itemBlockBase, runtimeOutcastPoolDataSize);
        runtimeBackup.workspace.itemPoolMeta = runtimeOutcastTensorPool->GetMetaData();
        struct Backup {
            static void BackupBlockHeader(WsSlotAllocator::BlockHeader*& ptr, WsSlotAllocator::BlockHeader* base)
            {
                ptr = reinterpret_cast<WsSlotAllocator::BlockHeader*>(static_cast<uintptr_t>(ptr - base));
            }
        };

        for (uint32_t i = 0; i < parallelism; i++) {
            runtimeBackup.workspace.tensorAllocators[i].rootInner = allocator[i].rootInner;
            runtimeBackup.workspace.tensorAllocators[i].devTaskInnerExclusiveOutcasts =
                allocator[i].devTaskInnerExclusiveOutcasts;
            runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts = allocator[i].devTaskBoundaryOutcasts;

            uint64_t backupSize = sizeof(WsSlotAllocator::BlockHeader) * allocator[i].devTaskBoundaryOutcasts.slotNum_;
            (void)memcpy_s(
                runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList.Data(), backupSize,
                allocator[i].devTaskBoundaryOutcasts.GetBlockHeaderBase(), backupSize);

            WsSlotAllocator::BlockHeader* base = allocator[i].devTaskBoundaryOutcasts.GetBlockHeaderBase();
            Backup::BackupBlockHeader(
                runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.freeListHeader_, base);
            Backup::BackupBlockHeader(
                runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.notInUseHeaders_, base);
            WsSlotAllocator::BlockHeader* checkpointBase =
                runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList.Data();
            for (uint64_t k = 0; k < allocator[i].devTaskBoundaryOutcasts.slotNum_; k++) {
                Backup::BackupBlockHeader(checkpointBase[k].listNext, base);
            }
        }
    }

    void RuntimeAddrRestore(
        DeviceExecuteSlot* runtimeSlotList, ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool,
        uint64_t slotSize, uint64_t runtimeOutcastTensorSize, TensorAllocator* allocator, uint32_t parallelism)
    {
        uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
        uint64_t runtimeOutcastPoolDataSize =
            sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
        (void)memcpy_s(runtimeSlotList, slotDataSize, runtimeBackup.slotContext.slotList.Data(), slotDataSize);
        auto itemBlockBase = reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool->At(0));
        (void)memcpy_s(
            itemBlockBase, runtimeOutcastPoolDataSize,
            runtimeBackup.workspace.runtimeOutcastTensorPool.Data(), runtimeOutcastPoolDataSize);
        runtimeOutcastTensorPool->RestoreMetaData(runtimeBackup.workspace.itemPoolMeta);
        struct Restore {
            static void RestoreBlockHeader(
                WsSlotAllocator::BlockHeader*& ptr, WsSlotAllocator::BlockHeader* base,
                WsSlotAllocator::BlockHeader* index)
            {
                ptr = base + (uintptr_t)index;
            }
            static void RestoreSeqAllocator(SeqWsAllocator& dst, SeqWsAllocator& src)
            {
                dst.allocated_ = src.allocated_;
                dst.resetTimes_ = src.resetTimes_;
            }
        };

        for (uint32_t i = 0; i < parallelism; i++) {
            Restore::RestoreSeqAllocator(allocator[i].rootInner, runtimeBackup.workspace.tensorAllocators[i].rootInner);
            Restore::RestoreSeqAllocator(
                allocator[i].devTaskInnerExclusiveOutcasts,
                runtimeBackup.workspace.tensorAllocators[i].devTaskInnerExclusiveOutcasts);
            allocator[i].devTaskBoundaryOutcasts.availableSlots_ =
                runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.availableSlots_;

            WsSlotAllocator::BlockHeader* base = allocator[i].devTaskBoundaryOutcasts.GetBlockHeaderBase();
            Restore::RestoreBlockHeader(
                allocator[i].devTaskBoundaryOutcasts.freeListHeader_, base,
                runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.freeListHeader_);
            Restore::RestoreBlockHeader(
                allocator[i].devTaskBoundaryOutcasts.notInUseHeaders_, base,
                runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.notInUseHeaders_);
            WsSlotAllocator::BlockHeader* checkpointBase =
                runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList.Data();
            for (uint64_t k = 0; k < allocator[i].devTaskBoundaryOutcasts.slotNum_; k++) {
                Restore::RestoreBlockHeader(base[k].listNext, base, checkpointBase[k].listNext);
            }
        }
    }

    void RuntimeAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram)
    {
        RelocRange relocProgram(srcProgram, dstProgram);
        {
            auto& slotList = runtimeBackup.slotContext.slotList;
            DeviceExecuteSlot* base = slotList.Data();
            uint64_t size = slotList.size();
            for (uint64_t k = 0; k < size; k++) {
                relocProgram.RelocNullable(base[k].partialUpdate);
            }
        }
    }

    void RuntimeAddrRelocWorkspace(
        uint64_t srcWorkspace, uint64_t dstWorkspace, DevStartArgsBase* devStartArgs,
        DeviceExecuteSlot* runtimeSlotList, ItemPool<RuntimeOutcastTensor>::ItemBlock* runtimeOutcastTensorPool,
        uint32_t parallelism)
    {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        /* empty constructor's overhead should be negligible */
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        if (devStartArgs == nullptr) {
            /* only run on host */
            RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
        }
        {
            for (uint32_t i = 0; i < parallelism; i++) {
                auto& slottedOutcastsBlockList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
                WsSlotAllocator::BlockHeader* base = slottedOutcastsBlockList.Data();
                uint64_t size = slottedOutcastsBlockList.size();
                for (uint64_t k = 0; k < size; k++) {
                    relocWorkspace.RelocNullable(base[k].ptr);
                }
            }
        }
        {
            auto& slotList = runtimeBackup.slotContext.slotList;
            DeviceExecuteSlot* base = slotList.Data();
            ItemPool<RuntimeOutcastTensor>::ItemBlock* backupRtOutcastPool =
                runtimeBackup.workspace.runtimeOutcastTensorPool.Data();
            uint64_t size = slotList.size();
            for (uint64_t k = 0; k < size; k++) {
                static_assert(
                    sizeof(AddressDescriptor) == sizeof(uintdevptr_t),
                    "Please review the following logics when the condition does not hold anymore.");
                if (devStartArgs == nullptr) {
                    // Host: addr uses backup
                    if (base[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                        continue;
                    }

                    auto& rtOutcast = backupRtOutcastPool[base[k].rtOutcastIter].Item();
                    if (rtOutcast.isCache) {
                        continue;
                    } // To avoid duplicate reloc
                    rtOutcast.isCache = true;

                    uintdevptr_t addr = rtOutcast.Addr();
                    AddressDescriptor* desc = reinterpret_cast<AddressDescriptor*>(&rtOutcast.Addr());
                    *desc = AddressDescriptor::MakeFromAddress(addr);
                    RelocDescToCache(*desc, relocWorkspace, cacheInputOutputDict);
                } else {
                    // Device: addr uses actual
                    if (runtimeSlotList[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                        continue;
                    }

                    auto& rtOutcast = runtimeOutcastTensorPool[runtimeSlotList[k].rtOutcastIter].Item();
                    if (!rtOutcast.isCache) {
                        continue;
                    } // To avoid duplicate reloc
                    rtOutcast.isCache = false;

                    AddressDescriptor* desc = reinterpret_cast<AddressDescriptor*>(&rtOutcast.allocation.ptr);
                    RelocDescFromCache(*desc, relocWorkspace, devStartArgs);
                    rtOutcast.Addr() = desc->GetAddressValue();
                }
            }
        }
    }

    void MixTaskDataReloc(
        RelocRange& relocCtrlCache, RelocRange& relocProgram, DynDeviceTaskBase* dynTaskBase,
        DynFuncHeader* dynFuncDataList)
    {
        if (dynTaskBase->devTask.mixTaskData.wrapIdNum == 0) {
            return;
        }

        WrapInfoQueue* tmpWrapInfoQueue =
            reinterpret_cast<WrapInfoQueue*>(dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);
        WrapInfoQueue*& wrapInfoQueueRef = tmpWrapInfoQueue;
        WrapInfoQueue* wrapInfoQueue = RelocControlFlowCachePointer(wrapInfoQueueRef, relocCtrlCache);
        relocCtrlCache.Reloc(dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);

        WrapInfo*& wrapInfoElemRef = wrapInfoQueue->elem;
        WrapInfo* wrapInfoElem = RelocControlFlowCachePointer(wrapInfoElemRef, relocCtrlCache);
        (void)wrapInfoElem;

        MixTaskDataCache*& mixTaskDataBackupRef = dynTaskBase->mixTaskDataBackup;
        MixTaskDataCache* mixTaskDataBackup = RelocControlFlowCachePointer(mixTaskDataBackupRef, relocCtrlCache);
        WrapInfo*& wrapInfoBackupElemRef = mixTaskDataBackup->queue.elem;
        WrapInfo* wrapInfoBackupElem = RelocControlFlowCachePointer(wrapInfoBackupElemRef, relocCtrlCache);
        (void)wrapInfoBackupElem;

        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; i++) {
            StaticReadyCoreFunctionQueue* tmpWrapQueueForThread =
                reinterpret_cast<StaticReadyCoreFunctionQueue*>(dynTaskBase->devTask.mixTaskData.wrapQueueForThread[i]);
            StaticReadyCoreFunctionQueue*& wrapQueueForThreadRef = tmpWrapQueueForThread;
            StaticReadyCoreFunctionQueue* wrapQueueForThread =
                RelocControlFlowCachePointer(wrapQueueForThreadRef, relocCtrlCache);
            relocCtrlCache.Reloc(dynTaskBase->devTask.mixTaskData.wrapQueueForThread[i]);

            uint64_t*& wrapPtrElemRef = wrapQueueForThread->elem;
            uint64_t* wrapPtrElem = RelocControlFlowCachePointer(wrapPtrElemRef, relocCtrlCache);
            (void)wrapPtrElem;

            uint64_t*& wrapPtrBackupElemRef = mixTaskDataBackup->wrapQueueForThread[i].elem;
            uint64_t* wrapPtrBackupElem = RelocControlFlowCachePointer(wrapPtrBackupElemRef, relocCtrlCache);
            (void)wrapPtrBackupElem;
        }

        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            relocProgram.Reloc(dynTaskBase->devTask.mixTaskData.opWrapList[dupIndex]);
            relocProgram.Reloc(mixTaskDataBackup->opWrapList[dupIndex]);

            relocCtrlCache.Reloc(dynTaskBase->devTask.mixTaskData.opWrapOffsetList[dupIndex]);
            (void)RelocControlFlowCachePointer(mixTaskDataBackup->opWrapOffsetList[dupIndex], relocCtrlCache);
        }
    }

    void DieReadyQueueReloc(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase)
    {
        auto processQueue = [&](uint64_t& queuePtrValue) {
            if (queuePtrValue == 0) {
                return;
            }
            ReadyCoreFunctionQueue* queueRef = reinterpret_cast<ReadyCoreFunctionQueue*>(queuePtrValue);
            ReadyCoreFunctionQueue* queue = RelocControlFlowCachePointer(queueRef, relocCtrlCache);
            relocCtrlCache.Reloc(queuePtrValue);
            if (queue != nullptr) {
                queue->Reloc(relocCtrlCache);
            }
        };
        for (size_t i = 0; i < DIE_NUM; i++) {
            processQueue(dynTaskBase->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
            processQueue(dynTaskBase->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        }

        DieReadyQueueCache*& dieReadyQueueBackupRef = dynTaskBase->dieReadyQueueBackup;
        DieReadyQueueCache* dieReadyQueueBackup = RelocControlFlowCachePointer(dieReadyQueueBackupRef, relocCtrlCache);
        if (dieReadyQueueBackup != nullptr) {
            for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
                dieReadyQueueBackup->queueList[i].Reloc(relocCtrlCache);
            }
        }
    }
    void RelocDuppedDataAndDynFuncData(
        RelocRange& relocProgram, RelocRange& relocCtrlCache, DevAscendFunctionDuppedData* duppedData,
        DynFuncData* dynData, DynFuncDataCache* dynDataCache, DynFuncDataBackup* dynDataBackup)
    {
        // Reloc Dupped
        relocProgram.Reloc(duppedData->source_);

        // Reloc DynFuncData
        relocProgram.Reloc(dynData->opAttrs);
        relocProgram.Reloc(dynData->opAtrrOffsets);
        relocProgram.Reloc(dynData->rawTensorDesc);

        relocCtrlCache.Reloc(dynData->exprTbl);
        relocCtrlCache.Reloc(dynData->rawTensorAddr);

        relocProgram.Reloc(dynDataCache->devFunc);
        relocProgram.Reloc(dynDataCache->calleeList);

        relocCtrlCache.Reloc(dynDataCache->predCount);
        relocCtrlCache.RelocNullable(dynDataBackup->predCountBackup);
        relocCtrlCache.RelocNullable(dynDataBackup->rawTensorAddrBackup);
        relocCtrlCache.RelocNullable(dynDataBackup->deadEndHubBitmapBackup);
        relocCtrlCache.RelocNullable(dynDataBackup->tailTaskBitmapBackup);
    }

    /* Host-to-cache: devStartArgs should be nullptr. Cache-to-Device: devStartArgs should be filled */
    void TaskAddrRelocProgramAndCtrlCache(
        uint64_t srcProgram, uint64_t srcCtrlCache, uint64_t dstProgram, uint64_t dstCtrlCache)
    {
        RelocRange relocCtrlCache(srcCtrlCache, dstCtrlCache);
        RelocRange relocProgram(srcProgram, dstProgram);
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            /* When cached, the pointer is always legal */
            DynDeviceTaskBase*& dynTaskBaseRef = deviceTaskCacheList[deviceIndex].dynTaskBase;
            DynDeviceTaskBase* dynTaskBase = RelocControlFlowCachePointer(dynTaskBaseRef, relocCtrlCache);
            relocCtrlCache.Reloc(dynTaskBase->devTask.readyAivCoreFunctionQue);
            relocCtrlCache.Reloc(dynTaskBase->devTask.readyAicCoreFunctionQue);
            relocCtrlCache.Reloc(dynTaskBase->devTask.readyAicpuFunctionQue);

            for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
                ReadyCoreFunctionQueue*& readyQueueRef = dynTaskBase->readyQueue[i];
                ReadyCoreFunctionQueue* readyQueue = RelocControlFlowCachePointer(readyQueueRef, relocCtrlCache);
                readyQueue->Reloc(relocCtrlCache);
            }
            relocProgram.Reloc(dynTaskBase->cceBinary);
            relocProgram.Reloc(dynTaskBase->aicpuLeafBinary);

            ReadyQueueCache*& readyQueueBackupRef = dynTaskBase->readyQueueBackup;
            ReadyQueueCache* readyQueueBackup = RelocControlFlowCachePointer(readyQueueBackupRef, relocCtrlCache);
            for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
                readyQueueBackup->queueList[i].Reloc(relocCtrlCache);
            }

            DynFuncHeader*& dynFuncDataListRef = dynTaskBase->dynFuncDataList;
            DynFuncHeader* dynFuncDataList = RelocControlFlowCachePointer(dynFuncDataListRef, relocCtrlCache);
            relocProgram.Reloc(dynFuncDataList->cceBinary);
            DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            MixTaskDataReloc(relocCtrlCache, relocProgram, dynTaskBase, dynFuncDataList);
            DieReadyQueueReloc(relocCtrlCache, dynTaskBase);
            void*& shmemWaitUntilCacheBackupRef = dynTaskBase->shmemWaitUntilCacheBackup;
 	        RelocControlFlowCachePointer(shmemWaitUntilCacheBackupRef, relocCtrlCache);
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                DevAscendFunctionDuppedData*& duppedDataRef = dynDataCache->duppedData;
                DevAscendFunctionDuppedData* duppedData = RelocControlFlowCachePointer(duppedDataRef, relocCtrlCache);

                // Reloc Stitch
                for (uint32_t i = 0; i < duppedData->GetStitchSize(); i++) {
                    DevAscendFunctionDuppedStitchList& stitchList = duppedData->GetStitch(i);
                    DevAscendFunctionDuppedStitch*& stitchRef = stitchList.Head();
                    for (DevAscendFunctionDuppedStitch** nodePtr = &stitchRef; *nodePtr != nullptr;) {
                        DevAscendFunctionDuppedStitch* node = RelocControlFlowCachePointer(*nodePtr, relocCtrlCache);
                        nodePtr = &node->Next();
                    }
                }
                RelocDuppedDataAndDynFuncData(
                    relocProgram, relocCtrlCache, duppedData, dynData, dynDataCache, dynDataBackup);
            }
        }
    }

    template <typename Ty>
    typename Ty::ElementType* RelocOffset(intptr_t shift, void*& offset, Ty& list)
    {
        typename Ty::ElementType* ptr = reinterpret_cast<typename Ty::ElementType*>(offset);
        offset = (void*)((uintptr_t)(offset) + list.ElementSize() * list.size());
        list.DeviceRelocData(shift);
        return ptr;
    }

    void RelocMetaCache(uint64_t srcCache, uint64_t dstCache)
    {
        intptr_t shift = static_cast<int64_t>(dstCache) - static_cast<int64_t>(srcCache);
        void* offset = data;
        RelocOffset(shift, offset, inputTensorDataList);
        RelocOffset(shift, offset, outputTensorDataList);
        for (uint32_t i = 0; i < SCH_DEVTASK_MAX_PARALLELISM; i++) {
            RelocOffset(shift, offset, runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList);
        }
        RelocOffset(shift, offset, runtimeBackup.slotContext.slotList);
        RelocOffset(shift, offset, runtimeBackup.workspace.runtimeOutcastTensorPool);
        RelocOffset(shift, offset, deviceTaskCacheList);
        RelocOffset(shift, offset, cacheData);
    }
};

#define ControlFlowAllocateSlab(devProg, size, expr)               \
    ({                                                             \
        WsAllocation ws;                                           \
        DevControlFlowCache* c = (devProg)->GetControlFlowCache(); \
        if (c->IsRecording()) {                                    \
            void* ptr = c->AllocateCache(size);                    \
            if (ptr != nullptr) {                                  \
                ws.ptr = reinterpret_cast<uintdevptr_t>(ptr);      \
            } else {                                               \
                ws = (expr);                                       \
            }                                                      \
        } else {                                                   \
            ws = (expr);                                           \
        }                                                          \
        ws;                                                        \
    })
} // namespace npu::tile_fwk::dynamic
