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
 * \file gen_aicore_code.cpp
 * \brief
 */

#include "machine/kernel/gen_aicore_code.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {
namespace {
const std::string kKernelEntryStr = "KERNEL_ENTRY";
const size_t kKernelEntryStrSize = 12;
const std::string kAicoreSrcCode = R"!!!(
#include <stdint.h>
#include <cstdint>
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aicore_runtime.h"
#include "tilefwk/aicore_print.h"
#include "tilefwk/core_func_data.h"

#define TO_ENTRY_IMPL(name, line, key, type) (name##line##key##type)
#define TO_ENTRY(name, key, type) TO_ENTRY_IMPL(name, _, key, type)

#ifdef __MIX__
#ifdef __AIV__
#define KERNEL_ENTRY(x, y) TO_ENTRY(x, y, _mix_aiv)
#else
#define KERNEL_ENTRY(x, y) TO_ENTRY(x, y, _mix_aic)
#endif
#else
#define KERNEL_ENTRY(x, y) x
#endif
#define unlikely(expr) __builtin_expect(!!(expr), 0)

constexpr uint32_t REG_HIGH_DTASKID_SHIFT = 32;
enum class TASK_POS : size_t { LOW_REG = 0, HIGH_REG = 1, ALL_REG = 2, REG_POS_BUTT = 3 };

struct TaskStat {
    int16_t seqNo;
    int16_t subGraphId;
    int32_t taskId;
    int64_t execStart;
    int64_t execEnd;
    int64_t waitStart; // 2.0 dfx 当前未使用
};

constexpr uint32_t PERF_TRACE_INST_MAX_NUM_EVERY_TYPE = 10;
constexpr uint32_t INVALID_DEV_TASK_ID = 0xFFFFFFFF;
enum AicorePerfTrace {
    PERF_TRACE_CORE_BEGIN = 0,
    PERF_TRACE_CORE_INIT,
    PERF_TRACE_CORE_DEV_TASK_RCV_MODEL,
    PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK,
    PERF_TRACE_CORE_DEV_TASK_CALLOP_TASK_EXEC,
    PERF_TRACE_CORE_DEV_TASK_WAIT_SYNC_STOP_NOTIFY,
    PERF_TRACE_CORE_WAIT_ALL_DEV_TASK_CALLOP_EXEC_FINISH,
    PERF_TRACE_CORE_WAIT_EXIT_NOTIFY,
    PERF_TRACE_CORE_MAX
};

struct Metrics {
  int64_t isMetricStop;
  int64_t taskCount; 
  int64_t perfTrace[PERF_TRACE_CORE_MAX][PERF_TRACE_INST_MAX_NUM_EVERY_TYPE];
  uint32_t perfTraceDevTaskId[PERF_TRACE_CORE_MAX][PERF_TRACE_INST_MAX_NUM_EVERY_TYPE];
  uint32_t perfTraceCnt[PERF_TRACE_CORE_MAX];
  TaskStat tasks[];
};

struct TaskEntry {
    int32_t subGraphId;
    int32_t taskId;
    int64_t funcAddr;
    int64_t tensorAddrs;
    int64_t gmStackSize;
    int64_t gmStackBase;
    int64_t reserved2[2];
    uint32_t tensorSize;
    uint32_t reserved[1];
};

struct KernelArgs {
    int64_t shakeBuffer[8];
    int64_t shakeBufferCpuToCore[8];
    TaskEntry taskEntry;
    TaskStat taskStat[2];
};

static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);
// aicore head file end

// device switch head file begin
namespace npu::tile_fwk {
#define PERF_PMU_TEST_SWITCH 0

#define DEBUG_SWITCH 0

#define ENABLE_AICORE_PRINT 0

#define ENABLE_AICORE_PERF_TRACE  0

// whether to support hand shake by reg
#define ENABLE_HAND_SHAKE_BY_REG 0

/* The DFX swimlane performance statistics use host pre-allocated memory mode, which avoids data collection during
   AICPU scheduling to minimize scheduling interference. However, each AICore only supports tracking up to
   MAX_DFX_TASK_NUM_PER_CORE tasks, with excess tasks being discarded.
*/
#define PROF_DFX_HOST_PREPARE_MEMORY_MODE 1
}
// device switch head file end

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

#if defined(__MIX__) && defined(__AIV__)
#define blockIdx __v_blockIdx
#define GmWorkspace __v_GmWorkspace
#endif

[[block_local]] int blockIdx;
[[block_local]] int64_t GmWorkspace;

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_CORE_EXIT = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8,
    STAGE_GET_COREFUNC_DATA_TIMEOUT = 9,
    STAGE_GET_NEXT_TASK_TIMEOUT = 10
};

struct ExecuteContext {
    __gm__ KernelArgs *args;
    uint32_t seqNo;
    __gm__ DynFuncData *funcDataList;
    uint64_t lastTaskFinishCycle{0};
#if ENABLE_AICORE_PRINT
    AicoreLogger logger;
#endif
};

INLINE uint32_t GetNextTask(uint32_t lastTaskIdx, uint32_t curDevTaskId) {
    uint32_t nextLowIdx = 0;
    uint64_t coreStatus = 0;
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    do {
        __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(coreStatus));
        nextLowIdx = coreStatus & 0xFFFFFFFF;
        nextLowIdx -= 1;

        if ((nextLowIdx == AICORE_FUNC_STOP) &&
            (curDevTaskId == (uint32_t)(coreStatus >> REG_HIGH_DTASKID_SHIFT))) {
            return AICORE_FUNC_STOP;
        }

        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 500000000)) {
            return AICORE_TASK_STOP;
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
    set_cond(AICORE_TASK_INIT);
#if ENABLE_HAND_SHAKE_BY_REG
    uint64_t AICORE_REG_SAY_HELLO = 0xF000000080000000;
    set_cond(((int64_t)blockIdx << 48) | ((int64_t)get_coreid() << 32) | AICORE_REG_SAY_HELLO);
#endif
    volatile __gm__ int64_t *hello = shakeBuf;
    *hello = (int64_t)get_coreid() << 32 | AICORE_SAY_HELLO;
    Barrier();
    dcci(hello, SINGLE_CACHE_LINE, CACHELINE_OUT);
    Barrier();
}

INLINE void SetStatus(__gm__ KernelArgs *args, int64_t val) {
#if DEBUG_SWITCH
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

INLINE void PerfTraceRecord(uint32_t devTaskId, __gm__ Metrics* metric, AicorePerfTrace type, uint64_t cycle = 0) {
#if ENABLE_AICORE_PERF_TRACE
    uint32_t cnt = metric->perfTraceCnt[type];
    if (cnt < PERF_TRACE_INST_MAX_NUM_EVERY_TYPE) {
        metric->perfTrace[type][cnt] = cycle == 0 ? get_sys_cnt() : cycle;
        metric->perfTraceDevTaskId[type][cnt] = devTaskId;
        metric->perfTraceCnt[type]++;
    }
#endif
}

INLINE void SetTaskStatistic(__gm__ KernelArgs *args, int32_t& dfxPose,
                             int32_t taskId, int32_t subGraphId, int64_t tStart, uint16_t seqNo = 0)
{
    __gm__ volatile TaskStat *stat = &args->taskStat[dfxPose];
    stat->subGraphId = subGraphId;
    stat->taskId = taskId;
    stat->execStart = tStart;
    stat->execEnd = get_sys_cnt();
    stat->seqNo = seqNo;
    dcci(stat, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void AddMetricStatistic(ExecuteContext *ctx, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
    auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId;
        m->tasks[m->taskCount].execStart = t1;
        ctx->lastTaskFinishCycle = get_sys_cnt();
        m->tasks[m->taskCount].execEnd = ctx->lastTaskFinishCycle;
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
    dcci((__gm__ void *)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}

INLINE void DfxProcWhenCoreExit(ExecuteContext *ctx, __gm__ KernelArgs *args, __gm__ Metrics* metric) {
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_WAIT_EXIT_NOTIFY);
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(INVALID_DEV_TASK_ID, metric,
            PERF_TRACE_CORE_WAIT_ALL_DEV_TASK_CALLOP_EXEC_FINISH, ctx->lastTaskFinishCycle);
    }
    if (unlikely(args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1)) {
        FlushMetricStatistic(args);
    }
}

INLINE void DfxProcWhenDevTaskStop(ExecuteContext *ctx, __gm__ KernelArgs *args, __gm__ Metrics* metric) {
    PerfTraceRecord(ctx->seqNo, metric, PERF_TRACE_CORE_DEV_TASK_WAIT_SYNC_STOP_NOTIFY);
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(ctx->seqNo, metric, PERF_TRACE_CORE_DEV_TASK_CALLOP_TASK_EXEC, ctx->lastTaskFinishCycle);
    }
    SetStatus(args, STAGE_GET_NEXT_TASK_STOP);
}

INLINE uint64_t getCoreFuncionData(__gm__ KernelArgs *args, int64_t lastFunc) {
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    while (true) {
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 500000000)) {
            SetStatus(args, STAGE_GET_COREFUNC_DATA_TIMEOUT);
            break;
        }
        volatile __gm__ int64_t *shakebufferCpuToCore = args->shakeBufferCpuToCore;
        dcci(shakebufferCpuToCore, SINGLE_CACHE_LINE, CACHELINE_OUT);
        auto newFunc = shakebufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX];
        if (newFunc != lastFunc && newFunc != 0) {
            dcci((__gm__ void *)newFunc, SINGLE_CACHE_LINE, CACHELINE_OUT);
            return newFunc;
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

#ifdef __HAS_SUB_FUNC__
INLINE void ExecDynCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
    uint64_t t1 = get_sys_cnt();
    SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId
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
    if (unlikely(ctx->args->taskEntry.reserved[0] == PRO_LEVEL2 || ctx->args->taskEntry.reserved[0] == PRO_LEVEL1)) {
        AddMetricStatistic(ctx, ctx->seqNo, taskId, opAttrs[0], t1);
    }
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
    static int32_t taskDfxPos = REG_LOW_TASK_PING;
    SetTaskStatistic(ctx->args, taskDfxPos, taskId, opAttrs[0], t1, ctx->seqNo);
#endif
}
#endif

INLINE void InitCtx(ExecuteContext *ctx, __gm__ Metrics* metric, uint64_t coreFuncData) {
    __gm__ DynFuncHeader *header = (__gm__ DynFuncHeader *)coreFuncData;
    ctx->seqNo = header->seqNo;
    PerfTraceRecord(ctx->seqNo, metric, PERF_TRACE_CORE_DEV_TASK_RCV_MODEL);
    ctx->funcDataList = (__gm__ npu::tile_fwk::DynFuncData *)(header + 1);
    ctx->lastTaskFinishCycle = 0;
#if ENABLE_AICORE_PRINT
    auto buffer = reinterpret_cast<__gm__ uint8_t *>(ctx->args->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX]);
    if (ctx->logger.GetBuffer() != buffer) {
        ctx->logger.Init(buffer, PRINT_BUFFER_SIZE);
    }
#endif
    dcci((__gm__ void *)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return;
}

INLINE void ExecCoreFunctionKernel(ExecuteContext *ctx, uint32_t curTaskIdx) {
#ifdef __HAS_SUB_FUNC__
    ExecDynCoreFunctionKernel(ctx, curTaskIdx);
    return;
#endif
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
    __gm__ Metrics* metric = (__gm__ Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_BEGIN);
    bool isFirstTask = true;
    SetStatus(args, STAGE_HANDSHAKE_START);
    HandshakeClient(args->shakeBuffer);
    SetStatus(args, STAGE_HANDSHAKE_END);
    set_mask_norm();
    uint32_t curTaskIdx;
    uint32_t lastTaskIdx;
    int64_t coreFuncData = 0;
    ExecuteContext ctx = {.args = args };
    //get core task data
    uint64_t t0 = get_sys_cnt();
    uint64_t loop_count = 0;
    bool bIsExit = false;
    PerfTraceRecord(INVALID_DEV_TASK_ID, metric, PERF_TRACE_CORE_INIT);
    while (true) {
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > 3000000000)) {
            break;
        }
        lastTaskIdx = AICORE_TASK_INIT;
        if (bIsExit) {
            DfxProcWhenCoreExit(&ctx, args, metric);
            return; // no data exit
        }
        coreFuncData = getCoreFuncionData(args, coreFuncData);
        if (coreFuncData == 0) {
            DfxProcWhenCoreExit(&ctx, args, metric);
            return; // no data exit
        }
        InitCtx(&ctx, metric, coreFuncData);
        uint64_t t1 = get_sys_cnt();
        uint64_t inner_loop_count = 0;
        isFirstTask = true;
        while (true) {
            ++inner_loop_count;
            if ((inner_loop_count % 1000 == 0) && (get_sys_cnt() - t1 > 3000000000)) {
                break;
            }
            curTaskIdx = GetNextTask(lastTaskIdx, ctx.seqNo);
            if (curTaskIdx == AICORE_TASK_STOP) {
                DfxProcWhenDevTaskStop(&ctx, args, metric);
                SetStatus(args, STAGE_CORE_EXIT);
                bIsExit = true;
                break;
            } else if (curTaskIdx == AICORE_FUNC_STOP) {
                DfxProcWhenDevTaskStop(&ctx, args, metric);
                SendRegDevTaskStop(ctx.seqNo);
                break;
            }

            if (isFirstTask) {
                PerfTraceRecord(ctx.seqNo, metric, PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK);
                isFirstTask = false;
            }

            SendRegAck(curTaskIdx);
            PmuTestBegin(args);
            ExecCoreFunctionKernel(&ctx, curTaskIdx);
            PmuTestEnd(args);
            SendRegFinsh(curTaskIdx);
            lastTaskIdx = curTaskIdx;
            SetStatus(args, STAGE_FINISH_CUR_TASK);
        }
    }
}
)!!!";
}

bool GenAicoreSrcFile(const std::string &codeSrcPath, const std::string &funcHash) {
    std::string newSrcCode = kAicoreSrcCode;
    size_t pos = newSrcCode.find(kKernelEntryStr);
    while (pos != std::string::npos) {
        newSrcCode.replace(pos, kKernelEntryStrSize, kKernelEntryStr + "_" + funcHash);
        pos = newSrcCode.find(kKernelEntryStr, pos + kKernelEntryStrSize + 1);
    }
    if (RealPath(codeSrcPath).empty()) {
        DumpFile(kAicoreSrcCode, codeSrcPath);
    }
    return !RealPath(codeSrcPath).empty();
}
}
