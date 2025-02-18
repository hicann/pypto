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
 * \file aicore.h
 * \brief
 */

#pragma once
#include <cstdint>
#include "tilefwk/aicpu_common.h"
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

const uint64_t AICORE_REG_SAY_HELLO = 0xF000000080000000;
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

inline const char *AicorePerfTraceName[] = {
    "BEGIN",
    "INIT",
    "DEV_TASK_RCV_MODEL",
    "DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK",
    "DEV_TASK_ALL_CALLOP_TASK_EXEC",
    "DEV_TASK_WAIT_SYNC_STOP_NOTIFY",
    "WAIT_ALL_DEV_TASK_CALLOP_EXEC_FINISH",
    "WAIT_EXIT_NOTIFY"
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
    TaskStat taskStat[2]; // 寄存器高低32位，两个task 和 pending & running task存储： 2 * 2 个
};

static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);
