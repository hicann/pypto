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
 * \file aicore.cpp
 * \brief
 */

#include <stdint.h>
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aicore_runtime.h"
#include "tilefwk/aicore_print.h"
#include "tilefwk/core_func_data.h"
#include "machine/kernel/aicore.h"
#include "machine/utils/device_switch.h"

#define TO_STRING_IMPL(str) #str
#define TO_STRING(str) TO_STRING_IMPL(str)

#ifdef __HAS_SUB_FUNC__
#if defined(__MIX__) && defined(__AIV__)
#include TO_STRING(__HEAD_FILE__)
#else
#include TO_STRING(__HEAD_FILE__)
#endif
#endif

using npu::tile_fwk::DynFuncHeader;
using npu::tile_fwk::DynFuncData;
using npu::tile_fwk::DynFuncBin;
using npu::tile_fwk::DevRawTensorDesc;
using npu::tile_fwk::CoreFunctionData;

constexpr uint32_t STATUS_TASKID_SHIFT = 32;

#if defined(__MIX__) && defined(__AIV__)
#define blockIdx __v_blockIdx
#define GmWorkspace __v_GmWorkspace
#endif

[[block_local]] int blockIdx;
[[block_local]] int64_t GmWorkspace;

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_GET_COREFUNC_DATA_STOP = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8
};

struct ExecuteContext {
    __gm__ KernelArgs *args;
    uint32_t seqNo;
    __gm__ DynFuncData *funcDataList;
    __gm__ CoreFunctionData *staticFuncData;
#if ENABLE_AICORE_PRINT
    AicoreLogger logger;
#endif
};

typedef void (*StaticKernelFunc)(__gm__ int64_t *param, int64_t gmStackAddr, __gm__ int64_t *hcclContext, __gm__ int64_t *oriAddr);

INLINE uint32_t GetNextTask(uint32_t lastTaskIdx) {
    uint32_t nextLowIdx;
    uint64_t coreStatus;
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    do {
        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
        nextLowIdx = coreStatus & 0xFFFFFFFF;
        nextLowIdx -= 1;
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 500000000)) {
            break;
        }
    } while (nextLowIdx == lastTaskIdx);

    return nextLowIdx;
}

INLINE void PipeSync() {
#if defined(__AIV__)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#else
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
#endif
}

INLINE void Barrier()
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

INLINE void HandshakeClient(volatile __gm__ int64_t *shakeBuf) {
    volatile __gm__ int64_t *hello = shakeBuf;

    set_cond(AICORE_TASK_INIT);
    *hello = (int64_t)get_coreid() << 32 | AICORE_SAY_HELLO;
    Barrier();
    dcci(hello, SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}


INLINE void SetStatus(__gm__ KernelArgs *args, int64_t val) {
#if ENABLE_COMPILE_VERBOSE_LOG
        Barrier();
        args->shakeBuffer[2] = val;
        dcci(args->shakeBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
}

INLINE void SendRegFinsh(uint32_t curTaskIdx) {
    set_cond(curTaskIdx | AICORE_FIN_MASK);
}

INLINE void SendRegDevTaskStop(uint32_t dTaskId) {
    set_cond(((uint64_t)dTaskId << REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP | AICORE_FIN_MASK));
}

INLINE void SendRegAck(uint32_t taskIdx) {
    set_cond(taskIdx);
}

INLINE void SetTaskStatistic(__gm__ KernelArgs *args, int32_t& dfxPose,
                             int32_t taskId, int32_t subGraphId, int64_t tStart)
{
    __gm__ volatile TaskStat *stat = &args->taskStat[dfxPose];
    stat->subGraphId = subGraphId;
    stat->taskId = taskId;
    stat->execStart = tStart;
    stat->execEnd = get_sys_cnt();
    dcci(stat, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void AddMetricStatistic(__gm__ KernelArgs *args, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
    auto m = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId;
        m->tasks[m->taskCount].execStart = t1;
        m->tasks[m->taskCount].execEnd = get_sys_cnt();
        m->taskCount++;
    }
#endif
}

INLINE void FlushMetricStatistic(__gm__ volatile KernelArgs* args) {
    __gm__ volatile Metrics* m = (__gm__ volatile Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < m->taskCount; i++) {
        dcci(&m->tasks[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    m->isMetricStop = 1;
    dcci(m, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE uint64_t getCoreFuncionData(__gm__ KernelArgs *args, int64_t lastFunc) {
    uint32_t nextLowIdx;
    uint64_t coreStatus;
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    while (true) {
        // check if stop
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 500000000)) {
            break;
        }
        volatile __gm__ int64_t *shakebuffer = args->shakeBuffer;
        dcci(shakebuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
        auto newFunc = args->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX];
        if (newFunc != lastFunc && newFunc != 0) {
            dcci((__gm__ void *)newFunc, SINGLE_CACHE_LINE, CACHELINE_OUT);
            return newFunc;
        }

        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
        nextLowIdx = coreStatus & 0xFFFFFFFF;
        nextLowIdx -= 1;

        if (nextLowIdx == AICORE_TASK_STOP) {
            return 0;
        }
    }
    return 0;
}

INLINE void PmuTestBegin(__gm__ KernelArgs *args) {
#if PERF_PMU_TEST_SWITCH
    if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
        set_ctrl((uint64_t) get_ctrl() | 0x1);
    }
#endif
}

INLINE void PmuTestEnd(__gm__ KernelArgs *args) {
#if PERF_PMU_TEST_SWITCH
        if (args->taskEntry.reserved[0] == PRO_LEVEL2) {
            set_ctrl((uint64_t) get_ctrl() - 1);
        }
#endif
}

#define FuncNum(id)      TaskID(id)
INLINE void ExecStaticCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    static int32_t taskDfxPos = REG_LOW_TASK_PING;
#endif
    __gm__ CoreFunctionData* coreFuncData = ctx->staticFuncData;
    uint64_t t1 = get_sys_cnt();
    SetStatus(ctx->args,  ((uint64_t)taskId << STATUS_TASKID_SHIFT) | STAGE_PRE_EXEC_COREFUNC_KERNEL);
    __gm__ npu::tile_fwk::CoreFunctionWsAddr* functionInfo =
            &((__gm__ npu::tile_fwk::CoreFunctionWsAddr*)coreFuncData->coreFunctionWsAddr)[taskId];
    StaticKernelFunc kernel = (StaticKernelFunc)functionInfo->functionBinAddr;
    kernel((__gm__ int64_t *)functionInfo->invokeEntryAddr,
           coreFuncData->stackWorkSpaceAddr + blockIdx * coreFuncData->stackWorkSpaceSize,
           (__gm__ int64_t *)coreFuncData->hcclContextAddr,
           (__gm__ int64_t *)functionInfo->invokeEntryOriAddr);

    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    SetTaskStatistic(ctx->args, taskDfxPos, taskId, (int32_t)functionInfo->psgId, t1);
#endif

    AddMetricStatistic(ctx->args, 0, taskId, (int32_t)functionInfo->psgId, t1);
}

#ifdef __HAS_SUB_FUNC__
INLINE void ExecDynCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
    uint64_t t1 = get_sys_cnt();

    SetStatus(ctx->args, ((uint64_t)taskId << STATUS_TASKID_SHIFT) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId

    auto funcData = &ctx->funcDataList[FuncID(taskId)];
    auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[TaskID(taskId)]];
#if ENABLE_AICORE_PRINT
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, ctx->logger.context()};
#else
    CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, nullptr};
#endif
    CallSubFuncTask(opAttrs[0], &param, funcData->stackWorkSpaceAddr + blockIdx * funcData->stackWorkSpaceSize,
                    (__gm__ int64_t *)funcData->hcclContext);
    SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
    PipeSync();
    SetStatus(ctx->args, STAGE_FINISH_PIPE_SYNC);
    AddMetricStatistic(ctx->args, ctx->seqNo, taskId, opAttrs[0], t1);
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    static int32_t taskDfxPos = REG_LOW_TASK_PING;
    SetTaskStatistic(ctx->args, taskDfxPos, taskId, opAttrs[0], t1);
#endif
}
#endif

INLINE void InitCtx(ExecuteContext *ctx, uint64_t coreFuncData, bool isDyn) {
    if (isDyn) {
        __gm__ DynFuncHeader *header = (__gm__ DynFuncHeader *)coreFuncData;
        ctx->seqNo = header->seqNo;
        ctx->funcDataList = (__gm__ npu::tile_fwk::DynFuncData *)(header + 1);
#if ENABLE_AICORE_PRINT
        auto buffer = reinterpret_cast<__gm__ uint8_t *>(ctx->args->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX]);
        if (ctx->logger.GetBuffer() != buffer) {
            ctx->logger.Init(buffer, PRINT_BUFFER_SIZE);
        }
#endif
        dcci((__gm__ void *)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        return;
    }

    ctx->staticFuncData = (__gm__ npu::tile_fwk::CoreFunctionData*)coreFuncData;
}

INLINE void ExecCoreFunctionKernel(ExecuteContext *ctx, uint32_t curTaskId, bool isDyn) {
#ifdef __HAS_SUB_FUNC__
    if (isDyn) {
        ExecDynCoreFunctionKernel(ctx, curTaskId);
        return;
    }
#endif
    ExecStaticCoreFunctionKernel(ctx, curTaskId);
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(__OPTYPE__, __TILINGKEY__)(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
#if defined(__AIV__) and defined(__MIX__)
    blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    blockIdx = get_block_idx();
#endif
    auto devArgs = (DeviceArgs*)cfgdata;
    __gm__ KernelArgs *args = (__gm__ KernelArgs *)(devArgs->sharedBuffer + blockIdx * SHARED_BUFFER_SIZE);
    bool isDyn = devArgs->taskType == DEVICE_TASK_TYPE_DYN ? true : false;

    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args->shakeBuffer);
    SetStatus(args, STAGE_HANDSHAKE_END);
    set_mask_norm();
    uint32_t curTaskIdx;
    int64_t coreFuncData = 0;
    ExecuteContext ctx = {.args = args };
    //get core task data
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    while (true) {
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 3000000000)) {
            break;
        }
        uint32_t lastTaskIdx = AICORE_TASK_INIT;
        coreFuncData = getCoreFuncionData(args, coreFuncData);
        if (coreFuncData == 0) {
            FlushMetricStatistic(args);
            SetStatus(args, STAGE_GET_COREFUNC_DATA_STOP);
            return; // no data exit
        }
        InitCtx(&ctx, coreFuncData, isDyn);
        uint64_t t1 = get_sys_cnt();
        uint64_t inner_loop_count = 0;
        while (true) {
            ++inner_loop_count;
            if ((inner_loop_count % 1000 == 0) && (get_sys_cnt() - t1 > 3000000000)) {
                break;
            }
            curTaskIdx = GetNextTask(lastTaskIdx);
            if (curTaskIdx == AICORE_TASK_STOP || curTaskIdx == AICORE_FUNC_STOP) {
                SetStatus(args, STAGE_GET_NEXT_TASK_STOP);
                if (isDyn) {
                    SendRegDevTaskStop(ctx.seqNo);
                    break;
                } else {
                    FlushMetricStatistic(args);
                    return;
                }
            }

            SendRegAck(curTaskIdx);
            PmuTestBegin(args);
            ExecCoreFunctionKernel(&ctx, curTaskIdx, isDyn);
            PmuTestEnd(args);
            SendRegFinsh(curTaskIdx);
            lastTaskIdx = curTaskIdx;
            SetStatus(args, STAGE_FINISH_CUR_TASK);
        }
    }
}
