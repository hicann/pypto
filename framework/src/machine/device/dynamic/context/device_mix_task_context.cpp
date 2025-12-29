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
 * \file device_mix_task_context.cpp
 * \brief
 */

 #include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {
#ifdef SUPPORT_MIX_SUBGRAPH_SCHE
void DeviceTaskContext::ProcessWrapQueue(DynDeviceTask *dyntask, uint32_t wrapId, int funcIndex, size_t opIndex, WrapInfoQueue *wrapQueue, uint32_t *wrapTasklistAddr) {
    DEV_VERBOSE_DEBUG("add task to wrap queue, wrapId = %u, funcIndex = %d, opIndex = %lu", wrapId, funcIndex, opIndex);
    if (wrapQueue == nullptr || wrapTasklistAddr == nullptr) {
        DEV_VERBOSE_DEBUG("wrapQueue or wrapTasklistAddr = nullptr");
        return;
    }

    for (uint32_t idx = wrapQueue->head; idx < wrapQueue->tail; idx++) {
        if (wrapQueue->elem[idx].wrapId == wrapId) {
            ReadyCoreFunctionQueue* tasklist = &wrapQueue->elem[idx].tasklist;
            tasklist->elem[tasklist->tail++] = MakeTaskID(funcIndex, opIndex);
            return;
        }
    }

    auto opWrapTaskNumList = reinterpret_cast<int32_t*>(dyntask->devTask.opWrapTaskNumList[funcIndex]);
    auto cceBinary = dyntask->cceBinary;
    auto callList = dyntask->dynFuncDataCacheList[funcIndex].calleeList;

    // add new wrap id to wrapQueue
    WrapInfo *info = &wrapQueue->elem[wrapQueue->tail];
    info->wrapId = wrapId;
    info->aicCoreIdx = 0;
    info->aivCoreIdxZero = 0;
    info->aivCoreIdxOne = 0;
    info->taskCnt = opWrapTaskNumList[opIndex];
    info->mixResourceType = cceBinary[callList[opIndex]].mixResourceType;
    info->tasklist.head = 0;
    info->tasklist.tail = 0;
    info->tasklist.capacity = opWrapTaskNumList[opIndex];
    if (wrapQueue->tail == 0) {
        info->tasklist.elem = wrapTasklistAddr;
    } else {
        WrapInfo *preQueueInfo = &wrapQueue->elem[wrapQueue->tail - 1];
        info->tasklist.elem = preQueueInfo->tasklist.elem + preQueueInfo->tasklist.capacity;
    }
    info->tasklist.elem[info->tasklist.tail++] = MakeTaskID(funcIndex, opIndex);
    wrapQueue->tail++;
}

uint32_t* DeviceTaskContext::AllocWrapTasklist(DynDeviceTask *dyntask) {
    uint32_t size = dyntask->devTask.coreFunctionCnt; // can be optimized by wrapTaskNum
    WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_TASKLIST));
    uint32_t *wrapTasklistAddr = qalloc.As<uint32_t>();
    return wrapTasklistAddr;
}

WrapInfoQueue* DeviceTaskContext::AllocWrapQueue(DynDeviceTask *dyntask) {
    uint32_t size = sizeof(WrapInfoQueue) + dyntask->devTask.wrapIdNum * sizeof(WrapInfo);
    WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_QUEUE));
    WrapInfoQueue *q = qalloc.As<WrapInfoQueue>();
    q->head = 0;
    q->tail = 0;
    q->lock = 0;
    q->capacity = dyntask->devTask.wrapIdNum;
    q->elem = reinterpret_cast<WrapInfo *>(q + 1);
    return q;
}

int DeviceTaskContext::BuildReadyQueueWithMixTask(DynDeviceTask *dyntask, DevAscendProgram *devProg) {
    PerfBegin(PERF_EVT_READY_QUEUE_IN);
    uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
    if (dyntask->devTask.coreFunctionCnt > devProg->stitchFunctionsize) {
        DEV_ERROR("coreFunctionCnt (%lu) exceeds stitchFunctionsize (%u).", dyntask->devTask.coreFunctionCnt, devProg->stitchFunctionsize);
        return DEVICE_MACHINE_ERROR;
    }
    DEV_ASSERT(dyntask->devTask.coreFunctionCnt <= devProg->stitchFunctionsize);
    ReadyCoreFunctionQueue *queue[READY_QUEUE_SIZE];
    for (size_t index = 0; index < READY_QUEUE_SIZE; ++index) {
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::READY_QUE));
        ReadyCoreFunctionQueue *q = qalloc.As<ReadyCoreFunctionQueue>();
        q->lock = 0;
        q->head = 0;
        q->tail = 0;
        q->capacity = dyntask->devTask.coreFunctionCnt;
        q->elem = reinterpret_cast<taskid_t *>(q + 1);
        queue[index] = q;
        dyntask->readyQueue[index] = q;
    }

    ReadyCoreFunctionQueue *aicpuQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)];
    ReadyCoreFunctionQueue *aivQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue *aicQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];

    dyntask->devTask.wrapIdNum = 0;
    for (size_t funcIndex = 0; funcIndex < dyntask->dynFuncDataCacheListSize; ++funcIndex) {
        dyntask->devTask.wrapIdNum += dyntask->dynFuncDataCacheList[funcIndex].devFunc->wrapIdNum_;
    }
    bool isNeedWrap = dyntask->devTask.wrapIdNum > 0;
    uint32_t *wrapTasklistAddr = isNeedWrap ? AllocWrapTasklist(dyntask) : nullptr;
    WrapInfoQueue *wrapQueue = isNeedWrap ? AllocWrapQueue(dyntask) : nullptr;

    int wrapTaskNum = 0;
    int aivQueueTail = 0;
    int aicQueueTail = 0;
    int aicpuQueueTail = 0;

    uint32v8 one = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32v8 base = {0, 1, 2, 3, 4, 5, 6, 7};
    DynFuncDataCache *dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
    size_t funcSize = dyntask->dynFuncDataCacheListSize;
    for (size_t funcIndex = 0; funcIndex < funcSize; ++funcIndex) {
        int32_t* opWrapList = reinterpret_cast<int32_t *>(dyntask->devTask.opWrapList[funcIndex]);
        DevAscendFunctionDuppedData *duppedData = dynFuncDataCacheList->At(funcIndex).duppedData;
        predcount_t *dupPredCountList = &duppedData->GetOperationCurrPredCount(0);
        auto &predInfo = duppedData->GetSource()->GetPredInfo();

        size_t totalZeroPredAIVBatchEnd = isNeedWrap ? 0 :predInfo.totalZeroPredAIV & ~0x7; // wrap doesnt support batch process
        taskid_t *aivQueueElemList = reinterpret_cast<taskid_t *>(aivQueue->elem);
        const size_t DUP_PRED_COUNT_LOOP_MAX = 8;
        const size_t DUP_PRED_COUNT_PRE_LOOP_CNT = 4;
        for (size_t opIndex = 0; opIndex < totalZeroPredAIVBatchEnd; opIndex += DUP_PRED_COUNT_LOOP_MAX) {
            if (likely((*reinterpret_cast<uint64_t *>(&dupPredCountList[opIndex]) | *reinterpret_cast<uint64_t *>(&dupPredCountList[opIndex + DUP_PRED_COUNT_PRE_LOOP_CNT]))) == 0) {
                uint32v8 taskidv8 = (one * MakeTaskID(funcIndex, 0)) | (base + static_cast<uint32_t>(opIndex));
#ifdef __x86_64__
                memcpy_s(&aivQueueElemList[aivQueueTail], sizeof(taskidv8), &taskidv8, sizeof(taskidv8));
#else
                *reinterpret_cast<uint32v8 *>(&aivQueueElemList[aivQueueTail]) = taskidv8;
#endif
                aivQueueTail += DUP_PRED_COUNT_LOOP_MAX;
            } else {
                for (size_t idx = 0; idx < DUP_PRED_COUNT_LOOP_MAX; ++idx) {
                    if (likely(dupPredCountList[opIndex + idx] == 0)) {
                        aivQueueElemList[aivQueueTail] = MakeTaskID(funcIndex, opIndex + idx);
                        aivQueueTail++;
                    }
                }
            }
        }
        for (size_t opIndex = totalZeroPredAIVBatchEnd; opIndex < predInfo.totalZeroPredAIV; ++opIndex) {
            if (likely(dupPredCountList[opIndex] == 0)) {
                if (isNeedWrap && opWrapList[opIndex] != -1) {
                    ProcessWrapQueue(dyntask, MakeMixWrapID(funcIndex, static_cast<uint32_t>(opWrapList[opIndex])), funcIndex, opIndex, wrapQueue, wrapTasklistAddr);
                    wrapTaskNum++;
                } else {
                    aivQueue->elem[aivQueueTail++] = MakeTaskID(funcIndex, opIndex);
                }
            }
        }

        auto aicEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC;
        for (size_t opIndex = predInfo.totalZeroPredAIV; opIndex < aicEnd; ++opIndex) {
            if (likely(dupPredCountList[opIndex] == 0)) {
                if (isNeedWrap && opWrapList[opIndex] != -1) {
                    ProcessWrapQueue(dyntask, MakeMixWrapID(funcIndex, static_cast<uint32_t>(opWrapList[opIndex])), funcIndex, opIndex, wrapQueue, wrapTasklistAddr);
                    wrapTaskNum++;
                } else {
                    aicQueue->elem[aicQueueTail++] = MakeTaskID(funcIndex, opIndex);
                }
            }
        }

        auto aicpuEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC + predInfo.totalZeroPredAicpu;
        for (size_t opIndex = aicEnd; opIndex < aicpuEnd; ++opIndex) {
            if (likely(dupPredCountList[opIndex] == 0)) {
                aicpuQueue->elem[aicpuQueueTail++] = MakeTaskID(funcIndex, opIndex);
            }
        }
    }

    aicpuQueue->tail = static_cast<uint32_t>(aicpuQueueTail);
    aivQueue->tail = static_cast<uint32_t>(aivQueueTail);
    aicQueue->tail = static_cast<uint32_t>(aicQueueTail);
    dyntask->devTask.readyAivCoreFunctionQue = PtrToValue(aivQueue);
    dyntask->devTask.readyAicCoreFunctionQue = PtrToValue(aicQueue);
    dyntask->devTask.readyAicpuFunctionQue = PtrToValue(aicpuQueue);
    dyntask->devTask.readyWrapCoreFunctionQue = PtrToValue(wrapQueue);
    dyntask->devTask.wrapTasklist = PtrToValue(wrapTasklistAddr);
    readyTaskNum += static_cast<uint64_t>(aivQueueTail + aicQueueTail + aicpuQueueTail + wrapTaskNum);
    PerfEnd(PERF_EVT_READY_QUEUE_IN);
    return DEVICE_MACHINE_OK;
}

#endif
}