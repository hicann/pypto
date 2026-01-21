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
 * \file device_utils.h
 * \brief
 */

#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <time.h>
#include <atomic>
#include <vector>
#include <mutex>
#include <sstream>
#include <fstream>
#include <ostream>
#include "securec.h"
#include "machine/utils/device_log.h"

#ifndef CONFIG_MAX_DEVICE_TASK_NUM
#define CONFIG_MAX_DEVICE_TASK_NUM 64
#endif

#define MAX_DEVICE_TASK_NUM CONFIG_MAX_DEVICE_TASK_NUM

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define BITS_PER_BYTES  8
#define BITS_PER_INT    32
#define BITS_PER_LONG   64

#ifndef UNUSED
#define UNUSED(n)       (void)(n)
#endif

namespace npu::tile_fwk::dynamic {
inline constexpr bool IsDeviceMode() {
#ifdef __DEVICE__
    return true;
#else
    return false;
#endif // __DEVICE__
}

constexpr int32_t DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH = -6;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH = -5;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_AIV = -4;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_AIC = -3;
constexpr int32_t DEVICE_MACHINE_TIMEOUT_CORETASK = -2;
constexpr int32_t DEVICE_MACHINE_ERROR = -1;
constexpr int32_t DEVICE_MACHINE_OK = 0;
constexpr int32_t DEVICE_MACHINE_FINISHED = 1;
constexpr int32_t TIME_OUT_THRESHOLD = 1000000; // 超时阈值 1s
constexpr int32_t DFX_TIME_OUT_THRESHOLD = 50000000; // 超时阈值 50s
constexpr uint32_t MAX_SCHEDULE_AICPU_NUM = 5;          // 真正负责调度aicore的最大aicpu个数
constexpr uint32_t MAX_OTHER_AICPU_NUM = 2; // 除调度cpu以外的其它aicpu数量
constexpr uint32_t MAX_USED_AICPU_NUM = MAX_SCHEDULE_AICPU_NUM + MAX_OTHER_AICPU_NUM;
constexpr uint32_t CTRL_CPU_THREAD_IDX = 0;
constexpr int32_t START_AICPU_NUM = 3;
constexpr uint64_t NUM_FIFTY = 50;
constexpr uint64_t US_PER_SEC = 1000000;
constexpr uint64_t NSEC_PER_USEC = 1000;
constexpr uint64_t NSEC_PER_SEC = 1000000000;
constexpr uint64_t HAND_SHAKE_TIMEOUT = 48000000000; // aicpu stream wait hccl finish
constexpr int32_t MAX_MNG_AICORE_AVG_NUM = 8;
constexpr uint32_t NEED_LAUNCH_AICPU_MINNUM = 3;
constexpr uint32_t CORE_IDX_AIV = 0;
constexpr uint32_t CORE_IDX_AIC = 1;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const int INVALID_CORE_IDX = 0xFF;

#ifdef __aarch64__
constexpr uint64_t TIMEOUT_CYCLES = 500 * 1000 * 1000;
#else
constexpr uint64_t TIMEOUT_CYCLES = NSEC_PER_SEC;
#endif

constexpr uint64_t PROF_DUMP_TIMEOUT_CYCLES = TIMEOUT_CYCLES;

#define PERF_LEVEL 0
#define PERF_AICORE_THREAD_START 100

#define PERF_EVENTS                             \
    X(0, EXEC_DYN)                              \
    X(1, INIT)                                  \
    X(2, CONTROL_FLOW_MAPEXE)                   \
    X(3, CONTROL_FLOW_MAPEXE_MEMCPY)            \
    X(1, CONTROL_FLOW_CALL)                     \
    X(1, CONTROL_FLOW_INIT)                     \
    X(1, CONTROL_FLOW)                          \
    X(2, ROOT_FUNC)                             \
    X(3, STAGE_DUP_ROOT)                        \
    X(3, ALLOCATE_WORKSPACE)                    \
    X(3, STAGE_STITCH)                          \
    X(4, FAST_STITCH)                           \
    X(4, UPDATE_SLOT)                           \
    X(3, SUBMIT_AICORE)                         \
    X(4, DECIDE_SLOT_ADDRESS)                   \
    X(4, DECIDE_INCAST_ADDRESS)                 \
    X(4, RELEASE_FINISH_TASK)                   \
    X(4, DEALLOCATE_TASK)                       \
    X(4, STAGE_BUILD_TASK)                      \
    X(5, ALLOCATE_TASK)                         \
    X(5, BUILD_TASK_DATA)                       \
    X(6, READY_QUEUE)                           \
    X(7, READY_QUEUE_IN)                        \
    X(6, RESOLVE_EARLY)                         \
    X(6, CORE_FUNCDATA)                         \
    X(5, SLAB_MEM_SUBMIT)                        \
    X(4, DEALLOCATE_WORKSPACE)                  \
    X(4, STAGE_PUSH_TASK)                       \
    X(1, STAGE_TASK_SYNC)                       \
    X(2, RELEASE_FINISH_TASK_INSYNC)            \
    X(3, DEALLOCATE_TASK_INSYNC)                \
    X(1, STAGE_STOP_AICORE)                     \
    X(1, DEVICE_MACHINE_INIT_DYN)               \
    X(1, DEVICE_MACHINE_SERVER_DYN)             \
    X3(1, STAGE_SCHEDULE)                       \
    X3(2, RUN_TASK)                             \
    X3(3, POLLING_AICORES)                      \
    X3(3, RESOLVE_DEPENDENCE)                   \
    X3(3, SEND_AIC_TASK)                        \
    X3(3, SEND_AIV_TASK)                        \
    X3(3, WAIT_AICORE_FINISH)                   \
    X3(3, DISPATCH_TASK)                             \
    X3(2, SYNC_AICORE)                          \
    X3(1, TASK)                                 \
    X3(2, RECV_TASK)                            \
    X3(2, SEND_TASK)                            \
    X3(2, RESOLVE_DEP)                          \
    X_L2(1, WSALLOC_CORE_A)                     \
    X_L2(1, WSALLOC_CORE_D)                     \
    X_L2(1, WSALLOC_CPU_A)                      \
    X_L2(1, WSALLOC_CPU_D)                      \
    X_L2(1, WSRTALLOC_CPU_A)                    \
    X_L2(1, WSRTALLOC_CPU_D)                    \
    X(0, MAX)

#define X3(a, b) \
    X(a, b)      \
    X(a, b##1)   \
    X(a, b##2)

enum PerfEventType {
#define X(ind, evt) PERF_EVT_##evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
};

inline const char *PerfEventName[] = {
#define SPACE_0 ""
#define SPACE_1 "  " SPACE_0
#define SPACE_2 "  " SPACE_1
#define SPACE_3 "  " SPACE_2
#define SPACE_4 "  " SPACE_3
#define SPACE_5 "  " SPACE_4
#define SPACE_6 "  " SPACE_5
#define SPACE_7 "  " SPACE_6
#define X(ind, evt) SPACE_##ind #evt,
#define X_L2(idx, evt) X(idx, evt)
    PERF_EVENTS
#undef X_L2
#undef X
#undef SPACE_7
#undef SPACE_6
#undef SPACE_5
#undef SPACE_4
#undef SPACE_3
#undef SPACE_2
#undef SPACE_1
#undef SPACE_0
};

inline bool PerfEvtEnable[] = {
#define X(ind, evt)  PERF_LEVEL >= 0,
#define X_L2(ind, evt) PERF_LEVEL >= 2,
    PERF_EVENTS
#undef X_L2
#undef X
};


#define PERF_TRACES                             \
    X(BEGIN)                                    \
    X(ALLOC_THREAD_ID)                          \
    X(INIT)                                     \
    X(CORE_HAND_SHAKE)                          \
    XDEVTASK(DEV_TASK_BUILD)                    \
    XDEVTASK(DEV_TASK_RCV)                      \
    XDEVTASK(DEV_TASK_SEND_FIRST_CALLOP_TASK)   \
    XDEVTASK(DEV_TASK_SCHED_EXEC)               \
    XDEVTASK(DEV_TASK_SYNC_CORE_STOP)           \
    XDEVTASK(DEV_TASK_RSP)                      \
    X(WAIT_ALL_DEV_TASK_FINISH)                 \
    X(WAIT_CORE_EXIT)                           \
    X(EXIT)                                     \
    X(MAX)                                      \


inline bool PerfTraceIsDevTask[] = {
#define X(trace)  0,
#define XDEVTASK(trace) 1,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

enum PerfTraceType {
#define X(trace) PERF_TRACE_##trace,
#define XDEVTASK(trace) PERF_TRACE_##trace,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

inline const char *PerfTraceName[] = {
#define X(trace) #trace,
#define XDEVTASK(trace) #trace,
    PERF_TRACES
#undef XDEVTASK
#undef X
};

#define DEVTASK_PERF_ARRY_INDEX(type) (type - PERF_TRACE_DEV_TASK_BUILD)
inline constexpr uint32_t DEVTASK_PERF_TYPE_NUM (PERF_TRACE_DEV_TASK_RSP - PERF_TRACE_DEV_TASK_BUILD + 1);
inline constexpr uint32_t PERF_TRACE_COUNT_DEVTASK_MAX_NUM = 20;

// common of ptr
template<typename TI, typename TO>
inline TO *PtrToPtr(TI *const ptr) {
  return reinterpret_cast<TO *>(ptr);
}

template<typename TI, typename TO>
inline const TO *PtrToPtr(const TI *const ptr) {
  return reinterpret_cast<const TO *>(ptr);
}

inline uint64_t PtrToValue(const void *const ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

inline std::vector<uint64_t> VPtrToValue(const std::vector<void *> v_ptr) {
  std::vector<uint64_t> v_value;
  for (const auto &ptr : v_ptr) {
    v_value.emplace_back(PtrToValue(ptr));
  }
  return v_value;
}

inline uint64_t GetTimeMonotonic() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

inline uint64_t GetCycles() {
    uint64_t cycles;
#ifdef __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#else
    cycles = GetTimeMonotonic();
#endif
    return cycles;
}

inline uint64_t GetFreq() {
    uint64_t freq;
#ifdef __aarch64__
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
#else
    freq = NSEC_PER_SEC;
#endif
    return freq;
}

inline uint64_t CurrentTime() {
    uint64_t mono = GetTimeMonotonic();
    return mono / NSEC_PER_USEC;
}

inline int CheckTimeOut(const uint64_t &tStart, uint64_t &tCnt, uint64_t &tCur, const std::string &opString) {
    tCnt++;
    if (tCnt % 50 == 0) { // 50 is loop count to check whether timeout occurs
        tCur = CurrentTime();
        DEV_IF_VERBOSE_DEBUG {
            if (tCur - tStart > DFX_TIME_OUT_THRESHOLD) {
                DEV_ERROR("%s dfx_timeout, aicpu force exit, ttl=%lu.", opString.c_str(), tCnt);
                return DEVICE_MACHINE_ERROR;
            }
        } else {
            if (tCur - tStart > TIME_OUT_THRESHOLD) {
                DEV_ERROR("%s timeout, aicpu force exit, ttl=%lu.", opString.c_str(), tCnt);
                return DEVICE_MACHINE_ERROR;
            }
        }
    }
    return DEVICE_MACHINE_OK;
}

struct TimeCheck {
    uint64_t startTime = CurrentTime();
    uint64_t curTime = 0;
    uint64_t count = 0;
};

inline int CheckTimeOut(const std::string &operation, TimeCheck &timeCheck) {
    return CheckTimeOut(timeCheck.startTime, timeCheck.count, timeCheck.curTime, operation);
}
}
#endif
