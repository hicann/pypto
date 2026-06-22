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
 * \file aicore_entry_dfx.h
 * \brief DFX/perf helpers for aicore_entry.h (include only after ExecuteContext is defined).
 */

#pragma once

INLINE void PerfTraceRecord(
    uint32_t devTaskId, __gm__ AicoreMetric* metric, AicorePerfTrace type, uint64_t cycle = 0)
{
    if (metric != nullptr) {
        uint64_t cnt = metric->cnt;
        if (cnt < PERF_TRACE_CORE_MAX * PERF_TRACE_INST_MAX_NUM_EVERY_TYPE) {
            __gm__ AicoreDevTaskPerf* perf = &(metric->aicoreEveryDevTypeTimeStamp[cnt]);
            perf->type = static_cast<uint64_t>(type);
            perf->aicoreDevTimeStamp = (cycle == 0) ? get_sys_cnt() : cycle;
            perf->devTaskIdx = static_cast<uint64_t>(devTaskId);
            
            metric->cnt = cnt + 1;
        }
    }
}

INLINE void AddMetricStatistic(ExecuteContext* ctx, uint32_t seqNo, uint32_t taskId, int32_t subGraphId, int64_t t1)
{
    auto m = (__gm__ Metrics*)(ctx->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m && m->taskCount < MAX_DFX_TASK_NUM_PER_CORE) {
        m->tasks[m->taskCount].subGraphId = subGraphId;
        m->tasks[m->taskCount].seqNo = seqNo;
        m->tasks[m->taskCount].taskId = taskId & TASKID_FROM_CTRL_TOPO_MASK;
        m->tasks[m->taskCount].execStart = t1;
        ctx->lastTaskFinishCycle = get_sys_cnt();
        m->tasks[m->taskCount].execEnd = ctx->lastTaskFinishCycle;
        m->taskCount++;
    }
}

INLINE void FlushMetricStatistic(__gm__ volatile KernelArgs* args)
{
    __gm__ volatile Metrics* m = (__gm__ volatile Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    if (m == nullptr) {
        return;
    }

    for (uint32_t i = 0; i < m->taskCount; i++) {
        dcci(&m->tasks[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }

    m->isMetricStop = 1;
    dcci(m, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}

INLINE void DfxProcWhenCoreExit(ExecuteContext* ctx, __gm__ KernelArgs* args, __gm__ Metrics* metric)
{
    PerfTraceRecord(INVALID_DEV_TASK_ID, ctx->aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_WAIT_EXIT_NOTIFY);
    if (unlikely(
            args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1 ||
            ctx->aicoreDevTaskMetric.devTaskMetricEnable)) {
        metric->turnNum++;
        FlushMetricStatistic(args);
    }
}

INLINE void DfxProcWhenDevTaskStop(ExecuteContext *ctx, __gm__ KernelArgs *args, __gm__ Metrics* metric)
{
    if (ctx->lastTaskFinishCycle > 0) {
        PerfTraceRecord(
            ctx->SeqNo(), ctx->aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_DEV_TASK_LEAF_TASK_EXEC, ctx->lastTaskFinishCycle);
    }
    (void)metric;
    (void)args;
}
