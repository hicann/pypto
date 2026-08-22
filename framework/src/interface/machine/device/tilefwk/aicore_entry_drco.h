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
 * \file aicore_entry_drco.h
 * \brief DRCO (Dependency Resolving by aiCOre) scheduling flow for aicore_entry.h.
 */

#ifndef AICORE_ENTRY_DRCO_H
#define AICORE_ENTRY_DRCO_H

// device switch head file begin
namespace npu::tile_fwk {

#if defined(__AIV__)
constexpr CoreType drcoCoreType = CoreType::AIV;
#elif defined(__AIC__)
constexpr CoreType drcoCoreType = CoreType::AIC;
#else
constexpr CoreType drcoCoreType = CoreType::MIX;
#endif

} // namespace npu::tile_fwk
// device switch head file end

#define DRCO_DCCI_SINGLE_CACHE_LINE(ptr) dcci((__gm__ uint8_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT)
#define DRCO_DCCI_ENTIRE_DATA_CACHE() dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT)

#define DRCO_BUSY_BACKOFF_CYC 1000

INLINE void DrcoBusyBackOff()
{
    uint64_t t0 = get_sys_cnt();
    while (get_sys_cnt() - t0 < DRCO_BUSY_BACKOFF_CYC) {
    }
}

using DrcoDeviceTask = npu::tile_fwk::DrcoDeviceTask;
using DrcoDeviceTaskReadyQueue = npu::tile_fwk::DrcoDeviceTaskReadyQueue;
using DrcoGlobalReadyQueue = npu::tile_fwk::DrcoGlobalReadyQueue;
using DrcoGlobalReadyQueuePtr = npu::tile_fwk::DrcoGlobalReadyQueuePtr;
using DrcoLocalReadyQueue = npu::tile_fwk::DrcoLocalReadyQueue;

INLINE __gm__ DrcoDeviceTask* GetCurrentDeviceTask(__gm__ DrcoDeviceTaskReadyQueue* queue)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(queue);
    if (queue->head >= queue->size) {
        return nullptr;
    }
    __gm__ DrcoDeviceTask* elem = &queue->dynFuncDataListList[queue->head];
    DRCO_DCCI_SINGLE_CACHE_LINE(elem);
    if (elem->dynFuncDataList == nullptr) {
        return nullptr;
    }
    return elem;
}

INLINE uint32_t DrcoAtomicAddToSigned(__gm__ int32_t* ptr, int32_t value)
{
    return static_cast<uint32_t>(atomicAdd(ptr, value));
}

INLINE uint32_t DrcoAtomicAddTo(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicAdd(ptr, value));
}

INLINE uint32_t DrcoAtomicCasTo(__gm__ uint32_t* ptr, uint32_t compare, uint32_t value)
{
    return static_cast<uint32_t>(atomicCAS(ptr, compare, value));
}

INLINE uint32_t DrcoAtomicExchTo(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicExch(ptr, value));
}

INLINE void DrcoLocalReadyQueuePushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    uint32_t tail = DrcoAtomicAddTo(&queue->tail, 1);
    DrcoAtomicExchTo(&queue->taskList[tail], DRCO_ENCODE_TASK(readyTask));
}

INLINE uint32_t DrcoLocalReadyQueueGetFirstTask(__gm__ DrcoLocalReadyQueue* queue)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(queue);
    uint32_t head = queue->head;
    uint32_t tail = queue->tail;
    if (head >= tail) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    uint32_t headPrev = DrcoAtomicCasTo(&queue->head, head, head + 1);
    if (headPrev != head) {
        return static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    }
    uint32_t taskId = DrcoAtomicAddTo(&queue->taskList[headPrev], 0);
    while (taskId == 0) {
        taskId = DrcoAtomicAddTo(&queue->taskList[headPrev], 0);
    }
    return DRCO_DECODE_TASK(taskId);
}

INLINE uint32_t DrcoPerCorePendingQueueGetFirstTask(__gm__ npu::tile_fwk::PerCorePendingQueue* queue)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(&queue->size);
    uint32_t head = queue->head;
    if (head >= queue->size) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    DRCO_DCCI_SINGLE_CACHE_LINE(&queue->taskList[head]);
    uint32_t taskId = queue->taskList[head];
    queue->head = head + 1;
    return taskId;
}

INLINE int DrcoGlobalReadyQueueTryPushLastTask(__gm__ DrcoGlobalReadyQueue* queue, uint32_t readyTask)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicAddTo(tailPtr, 0);
    uint32_t tailPrev = DrcoAtomicCasTo(&queue->tail, tail, tail + 1);
    if (tailPrev == tail) {
        DrcoAtomicExchTo(&queue->taskList[tailPrev], DRCO_ENCODE_TASK(readyTask));
        return 0;
    }
    return 1;
}

INLINE void DrcoGlobalReadyQueuePushLastTask(__gm__ DrcoGlobalReadyQueue* queue, uint32_t readyTask)
{
    int result = DrcoGlobalReadyQueueTryPushLastTask(queue, readyTask);
    while (result == 1) {
        result = DrcoGlobalReadyQueueTryPushLastTask(queue, readyTask);
    }
}

INLINE uint32_t DrcoGlobalReadyQueueTryGetFirstTask(__gm__ DrcoGlobalReadyQueue* queue)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(queue);
    volatile __gm__ uint32_t* headAddr = &queue->head;
    volatile __gm__ uint32_t* tailAddr = &queue->tail;
    uint32_t head = *headAddr;
    uint32_t tail = *tailAddr;
    if (head < tail) {
        uint32_t headPrev = DrcoAtomicCasTo(&queue->head, head, head + 1);
        if (headPrev == head) {
            uint32_t taskId = DrcoAtomicAddTo(&queue->taskList[headPrev], 0);
            while (taskId == 0) {
                taskId = DrcoAtomicAddTo(&queue->taskList[headPrev], 0);
            }
            return DRCO_DECODE_TASK(taskId);
        }
        return static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    } else if (head < queue->size) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    return static_cast<uint32_t>(AICORE_TASK_ALL_FINISH);
}

INLINE uint32_t DrcoGlobalReadyQueueGetFirstTask(__gm__ DrcoGlobalReadyQueue* queue)
{
    uint32_t taskId = static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    while (taskId == static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT)) {
        taskId = DrcoGlobalReadyQueueTryGetFirstTask(queue);
        if (taskId == static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
            break;
        }
    }
    return taskId;
}

INLINE __gm__ DrcoGlobalReadyQueue* GetDrcoGlobalReadyQueue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                            uint32_t readyQueueCoreType)
{
    if (readyQueueCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIV)) {
        return rootFuncList->globalReadyQueueList[npu::tile_fwk::DRCO_QUEUE_AIV].ptr;
    } else if (readyQueueCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC)) {
        return rootFuncList->globalReadyQueueList[npu::tile_fwk::DRCO_QUEUE_AIC].ptr;
    }
    return nullptr;
}

INLINE __gm__ DrcoLocalReadyQueue* GetDrcoLocalReadyQueue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                          uint32_t readyQueueCoreType)
{
    if (readyQueueCoreType < npu::tile_fwk::NUM_CORE_TYPES) {
        return rootFuncList->localReadyQueueArray[readyQueueCoreType][0];
    }
    return nullptr;
}

struct DrcoDynFuncDataListPush {
    INLINE static void Push(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t succTaskId,
                            uint32_t succCoreType)
    {
        __gm__ DrcoLocalReadyQueue* localQueue = GetDrcoLocalReadyQueue(rootFuncList, succCoreType);
        if (localQueue != nullptr) {
            DRCO_DCCI_SINGLE_CACHE_LINE(localQueue);
            if (localQueue->tail < localQueue->size) {
                DrcoLocalReadyQueuePushTask(localQueue, succTaskId);
                return;
            }
        }
        __gm__ DrcoGlobalReadyQueue* globalQueue = GetDrcoGlobalReadyQueue(rootFuncList, succCoreType);
        if (globalQueue != nullptr) {
            DrcoGlobalReadyQueuePushLastTask(globalQueue, succTaskId);
        }
    }
};

template <typename GlobalReadyQueueHandler>
INLINE void ExecDrcoResolve(ExecuteContext* ctx, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t taskId)
{
    constexpr uint32_t HUB_STACK_SIZE = 64;
    uint32_t hubStack[HUB_STACK_SIZE];
    int32_t hubStackTop = -1;
    hubStack[++hubStackTop] = taskId;

    while (hubStackTop >= 0) {
        uint32_t curTaskId = hubStack[hubStackTop--];
        uint32_t funcIdx = npu::tile_fwk::FuncID(curTaskId);
        auto funcData = &ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].funcDataList[funcIdx];

        __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = &funcData->drcoRootFuncData;
        __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfoList = rootFuncData->succInfoList;
        __gm__ int32_t* succStaticList = rootFuncData->succStaticList;
        __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode** succStitchList = rootFuncData->succStitchList;
        __gm__ int32_t* predCount = rootFuncData->predCount;

        uint32_t operationIndex = npu::tile_fwk::TaskID(curTaskId);
        volatile __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfo = &succInfoList[operationIndex];
        uint16_t staticIndex = succInfo->staticIndex;
        uint16_t staticSize = succInfo->staticSize;
        uint32_t stitchIndex = succInfo->stitchIndex;

        for (uint16_t i = staticIndex; i < staticIndex + staticSize; i++) {
            uint32_t succOpIdx = succStaticList[i];
            int32_t old = DrcoAtomicAddToSigned(&predCount[succOpIdx], -1);
            if (old == 1) {
                uint32_t succTaskId = npu::tile_fwk::MakeTaskID(funcIdx, succOpIdx);
                int cceBinaryIndex = funcData->cceBinaryIndexList[succOpIdx];
                uint32_t
                    succCoreType = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx].cceBinary[cceBinaryIndex].coreType;
                if (succCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB)) {
                    if (hubStackTop + 1 < HUB_STACK_SIZE) {
                        hubStack[++hubStackTop] = succTaskId;
                    }
                } else if (succCoreType != static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB_MIX)) {
                    GlobalReadyQueueHandler::Push(rootFuncList, succTaskId, succCoreType);
                }
            }
        }

        if (stitchIndex != 0) {
            for (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node = succStitchList[stitchIndex];
                 node != nullptr; node = node->nodeNext) {
                for (uint32_t i = 0; i < node->nodeSize; i++) {
                    uint32_t succTaskId = node->nodeTaskList[i];
                    uint32_t succFuncId = npu::tile_fwk::FuncID(succTaskId);
                    uint32_t succOpIdx = npu::tile_fwk::TaskID(succTaskId);
                    __gm__ npu::tile_fwk::DrcoRootFuncData*
                        succRootFuncData = &ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx]
                                                .funcDataList[succFuncId]
                                                .drcoRootFuncData;
                    int32_t old = DrcoAtomicAddToSigned(&succRootFuncData->predCount[succOpIdx], -1);
                    if (old == 1) {
                        int cceBinaryIndex = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx]
                                                 .funcDataList[succFuncId]
                                                 .cceBinaryIndexList[succOpIdx];
                        uint32_t succCoreType = ctx->cachedDevTasks[ctx->curLeafTaskParallelIdx]
                                                    .cceBinary[cceBinaryIndex]
                                                    .coreType;
                        if (succCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB)) {
                            if (hubStackTop + 1 < HUB_STACK_SIZE) {
                                hubStack[++hubStackTop] = succTaskId;
                            }
                        } else if (succCoreType != static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB_MIX)) {
                            GlobalReadyQueueHandler::Push(rootFuncList, succTaskId, succCoreType);
                        }
                    }
                }
            }
        }
    }
}

INLINE uint32_t DrcoDynFuncDataListGetFirstTask(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    uint64_t t0 = get_sys_cnt();
    while (true) {
        if (get_sys_cnt() - t0 > AICORE_LEAF_TASK_RUN_TIMEOUT) {
            Trap();
        }
        __gm__ DrcoLocalReadyQueue* localQueue = GetDrcoLocalReadyQueue(
            rootFuncList, static_cast<uint32_t>(npu::tile_fwk::drcoCoreType));
        if (localQueue != nullptr) {
            DRCO_DCCI_SINGLE_CACHE_LINE(localQueue);
            uint32_t taskId = static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
            while (taskId == static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT)) {
                taskId = DrcoLocalReadyQueueGetFirstTask(localQueue);
            }
            if (taskId != static_cast<uint32_t>(AICORE_TASK_NO_INCOME)) {
                return taskId;
            }
        }

        __gm__ DrcoGlobalReadyQueue* globalReadyQueue = GetDrcoGlobalReadyQueue(
            rootFuncList, static_cast<uint32_t>(npu::tile_fwk::drcoCoreType));
        if (globalReadyQueue != nullptr) {
            DRCO_DCCI_SINGLE_CACHE_LINE(globalReadyQueue);
            uint32_t taskId = DrcoGlobalReadyQueueGetFirstTask(globalReadyQueue);
            if (taskId != static_cast<uint32_t>(AICORE_TASK_NO_INCOME) &&
                taskId != static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
                return taskId;
            }
        }

        DRCO_DCCI_SINGLE_CACHE_LINE(rootFuncList->executedTaskCount);
        if (*rootFuncList->executedTaskCount >= rootFuncList->totalTaskCount) {
            return static_cast<uint32_t>(AICORE_TASK_ALL_FINISH);
        }
    }
}

INLINE void KernelEntryDrco(int64_t ffts_addr, int64_t inputs, int64_t outputs, int64_t workspace, int64_t tilingdata,
                            int64_t cfgdata)
{
    uint64_t start = get_sys_cnt();
    UNUSED(ffts_addr);
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(workspace);
    UNUSED(tilingdata);
#if defined(__AIV__) && defined(__MIX__)
    int32_t blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    int32_t blockIdx = get_block_idx();
#endif
    auto devArgs = (DeviceArgs*)cfgdata;
    __gm__ KernelArgs* args = (__gm__ KernelArgs*)(devArgs->sharedBuffer + blockIdx * SHARED_BUFFER_SIZE);
    __gm__ Metrics* metric = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    ExecuteContext ctx = {};
    ctx.args = args;
    ctx.blockIdx = blockIdx;
    __gm__ DevDfxArgs* devDfxAddr = (__gm__ DevDfxArgs*)devArgs->devDfxArgAddr;
    ctx.aicoreDevTaskMetric.devTaskMetricEnable = devDfxAddr->isOpenPerfTrace != 0;
    uint8_t aicoreLogLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
#if ENABLE_AICORE_PRINT
    if (devDfxAddr->logLevel >= 0) {
        aicoreLogLevel = static_cast<uint8_t>(devDfxAddr->logLevel);
    }
#endif
    if (ctx.aicoreDevTaskMetric.devTaskMetricEnable && metric->turnNum < MAX_ROUND_NUM) {
        uint64_t round = metric->turnNum;
        ctx.aicoreDevTaskMetric.devTaskMetric = &(metric->aicoreDevTaskInfo[round]);
        PerfTraceRecord(INVALID_DEV_TASK_ID, ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_BEGIN, start);
    }

    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args->shakeBuffer);
    SetStatus(args, STAGE_HANDSHAKE_END);
    set_mask_norm();
    uint8_t lastMixResourceType = static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN);

    PerfTraceRecord(INVALID_DEV_TASK_ID, ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_INIT);

    InitCtx(&ctx, metric, nullptr, aicoreLogLevel);

    __gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData*
        runtimeDataRingBufferHeadData = (__gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData*)
                                            devArgs->runtimeDataRingBufferAddr;
    DRCO_DCCI_SINGLE_CACHE_LINE(runtimeDataRingBufferHeadData);
    __gm__ npu::tile_fwk::DevStartArgsBase* base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
        runtimeDataRingBufferHeadData);
    DRCO_DCCI_SINGLE_CACHE_LINE(base);
    DRCO_DCCI_SINGLE_CACHE_LINE(&base->drcoDeviceTaskReadyQueue);
    __gm__ DrcoDeviceTaskReadyQueue* deviceTaskReadyQueue = base->drcoDeviceTaskReadyQueue;

    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    while (true) {
        AICORE_TIMEOUT_CHECK_RETURN_VOID(t0, loop_count, AICORE_LEAF_TASK_RUN_TIMEOUT, STAGE_RUN_LEAFTASK_TIMEOUT);
        if (deviceTaskReadyQueue == nullptr) {
            DRCO_DCCI_SINGLE_CACHE_LINE(runtimeDataRingBufferHeadData);
            base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(runtimeDataRingBufferHeadData);
            DRCO_DCCI_SINGLE_CACHE_LINE(base);
            DRCO_DCCI_SINGLE_CACHE_LINE(&base->drcoDeviceTaskReadyQueue);
            deviceTaskReadyQueue = base->drcoDeviceTaskReadyQueue;
            if (deviceTaskReadyQueue == nullptr) {
                DrcoBusyBackOff();
                continue;
            }
        }

        __gm__ DrcoDeviceTask* deviceTask = GetCurrentDeviceTask(deviceTaskReadyQueue);
        if (deviceTask == nullptr) {
            DRCO_DCCI_SINGLE_CACHE_LINE(deviceTaskReadyQueue);
            if (deviceTaskReadyQueue->head < deviceTaskReadyQueue->size) {
                __gm__ DrcoDeviceTask* elem = &deviceTaskReadyQueue->dynFuncDataListList[deviceTaskReadyQueue->head];
                DRCO_DCCI_SINGLE_CACHE_LINE(elem);
                if (elem->dynFuncDataList == nullptr) {
                    break;
                }
            }
            DrcoBusyBackOff();
            continue;
        }

        __gm__ DynFuncHeader* deviceTaskDynFuncDataList = deviceTask->dynFuncDataList;
        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList = deviceTask->drcoRootFuncList;

        UpdateCacheDevTask(&ctx, ctx.curLeafTaskParallelIdx, (int64_t)deviceTaskDynFuncDataList);

        __gm__ npu::tile_fwk::PerCorePendingQueue* myPerCoreQueue = rootFuncList->perCorePendingQueueArray[blockIdx];
        DRCO_DCCI_SINGLE_CACHE_LINE(myPerCoreQueue);
        DRCO_DCCI_SINGLE_CACHE_LINE(&myPerCoreQueue->size);
        ctx.profLevel = 0;

        if (rootFuncList->totalTaskCount == 0) {
            uint32_t oldHead = deviceTaskReadyQueue->head;
            DrcoAtomicCasTo(&deviceTaskReadyQueue->head, oldHead, oldHead + 1);
            continue;
        }

        while (myPerCoreQueue->size > myPerCoreQueue->head) {
            uint32_t taskId = DrcoPerCorePendingQueueGetFirstTask(myPerCoreQueue);
            if (taskId == static_cast<uint32_t>(AICORE_TASK_NO_INCOME)) {
                break;
            }
            ExecCoreFunctionKernel(&ctx, taskId, lastMixResourceType);
#ifdef __HAS_SUB_FUNC__
            ExecDrcoResolve<DrcoDynFuncDataListPush>(&ctx, rootFuncList, taskId);
#endif
            uint32_t oldCount = DrcoAtomicAddTo(rootFuncList->executedTaskCount, 1);
            if (oldCount == (rootFuncList->totalTaskCount - 1)) {
                uint32_t oldHead = deviceTaskReadyQueue->head;
                DrcoAtomicCasTo(&deviceTaskReadyQueue->head, oldHead, oldHead + 1);
            }
        }

        uint32_t taskId = DrcoDynFuncDataListGetFirstTask(rootFuncList);
        while (taskId != static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
            if ((taskId & AICORE_FIN_MASK) == 0) {
                ExecCoreFunctionKernel(&ctx, taskId, lastMixResourceType);
#ifdef __HAS_SUB_FUNC__
                ExecDrcoResolve<DrcoDynFuncDataListPush>(&ctx, rootFuncList, taskId);
#endif
                uint32_t oldCount = DrcoAtomicAddTo(rootFuncList->executedTaskCount, 1);
                if (oldCount == (rootFuncList->totalTaskCount - 1)) {
                    uint32_t oldHead = deviceTaskReadyQueue->head;
                    DrcoAtomicCasTo(&deviceTaskReadyQueue->head, oldHead, oldHead + 1);
                }
            }
            taskId = DrcoDynFuncDataListGetFirstTask(rootFuncList);
        }

        DRCO_DCCI_SINGLE_CACHE_LINE(rootFuncList->executedTaskCount);
        while (*rootFuncList->executedTaskCount < rootFuncList->totalTaskCount) {
            DRCO_DCCI_SINGLE_CACHE_LINE(rootFuncList->executedTaskCount);
        }
    }
    if (blockIdx == 0) {
        DRCO_DCCI_SINGLE_CACHE_LINE(runtimeDataRingBufferHeadData);
        uint64_t finished = runtimeDataRingBufferHeadData->indexFinished.value + 1;
        runtimeDataRingBufferHeadData->indexFinished.value = finished;
        DRCO_DCCI_SINGLE_CACHE_LINE(runtimeDataRingBufferHeadData);
    }

    FlushMetricStatistic(args);
    return;
}

#endif
