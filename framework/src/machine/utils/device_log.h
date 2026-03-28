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
 * \file device_log.h
 * \brief
 */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>
#include <cassert>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <execinfo.h>
#include "securec.h"
#include "tilefwk/aikernel_define.h"
#include "machine/utils/device_switch.h"
#include "machine/utils/machine_error.h"

#ifndef ERROR_CODE_UNDEFINED
#define ERROR_CODE_UNDEFINED 0xFFFFFU
#endif

#ifdef __DEVICE__
#include "dlog_pub.h"
#else
#include "tilefwk/pypto_fwk_log.h"
#endif

namespace npu::tile_fwk {
#define DEV_IF_NONDEVICE     if constexpr (!IsDeviceMode())

#define DEV_IF_DEVICE        if constexpr (IsDeviceMode())

#define DEV_IF_DEBUG         if (IsDebugMode())

#define DEV_IF_VERBOSE_DEBUG if constexpr (IsCompileVerboseLog())

inline constexpr bool IsCompileVerboseLog() {
#if ENABLE_COMPILE_VERBOSE_LOG
    return true;
#else
    return false;
#endif
}

#ifdef __DEVICE__
#define GET_TID() syscall(__NR_gettid)
#define LOG_MOD_ID AICPU

inline bool g_isLogEnableDebug = false;
inline bool g_isLogEnableInfo = false;
inline bool g_isLogEnableWarn = false;
inline bool g_isLogEnableError = false;

inline void InitLogSwitch() {
    g_isLogEnableDebug = CheckLogLevel(LOG_MOD_ID, DLOG_DEBUG);
    g_isLogEnableInfo = CheckLogLevel(LOG_MOD_ID, DLOG_INFO);
    g_isLogEnableWarn = CheckLogLevel(LOG_MOD_ID, DLOG_WARN);
    g_isLogEnableError = CheckLogLevel(LOG_MOD_ID, DLOG_ERROR);
}

inline bool IsLogEnableDebug() { return g_isLogEnableDebug; }
inline bool IsLogEnableInfo() { return g_isLogEnableInfo; }
inline bool IsLogEnableWarn() { return g_isLogEnableWarn; }
inline bool IsLogEnableError() { return g_isLogEnableError; }

inline bool IsDebugMode() {
    return g_isLogEnableDebug;
}

template<typename... Args>
inline void DeviceLogSplitDebug(const char* func, const char* format, Args... args) {
    if (!IsLogEnableDebug()) {
        return;
    }
    constexpr size_t MAX_LOG_CHUNK = 824;
    char *formatted_str = nullptr;
    int len = asprintf(&formatted_str, format, args...);
    if (len < 0 || formatted_str == nullptr) {
        return;
    }
    std::string log_content(formatted_str);
    free(formatted_str);
    // 分段输出
    if (log_content.size() <= MAX_LOG_CHUNK) {
        dlog_debug(LOG_MOD_ID, "%lu %s\n%s", GET_TID(), func, log_content.c_str());
    } else {
        size_t start = 0;
        int segment_num = 0;
        size_t total_len = log_content.size();
        size_t total_segments = (total_len + MAX_LOG_CHUNK - 1) / MAX_LOG_CHUNK;
        while (start < total_len) {
            size_t chunk_size = (total_len - start > MAX_LOG_CHUNK) ? MAX_LOG_CHUNK : total_len - start;
            std::string segment = log_content.substr(start, chunk_size);
            dlog_debug(LOG_MOD_ID, "%lu %s [Segment %d/%lu]\n%s",
                       GET_TID(), func, segment_num + 1, total_segments, segment.c_str());
            start += chunk_size;
            segment_num++;
        }
    }
}

#define D_DEV_LOGD(fmt, ...)                                                               \
  do {                                                                                     \
      if (IsLogEnableDebug()) {                                                            \
        dlog_debug(LOG_MOD_ID, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                                    \
  } while (false)

#define D_DEV_LOGI(fmt, ...)                                                               \
  do {                                                                                     \
      if (IsLogEnableInfo()) {                                                             \
        dlog_info(LOG_MOD_ID, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                                    \
  } while(false)

#define D_DEV_LOGW(fmt, ...)                                                               \
  do {                                                                                     \
      if (IsLogEnableWarn()) {                                                             \
        dlog_warn(LOG_MOD_ID, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                                    \
  } while(false)

#define D_DEV_LOGE(errCode, fmt, ...)                                                               \
  do {                                                                                     \
    if (IsLogEnableError()) {                                                              \
        dlog_error(LOG_MOD_ID, "%lu %s\nErrCode: F%05X! " #fmt, GET_TID(), __FUNCTION__,        \
        static_cast<uint32_t>(errCode) & 0xFFFFF, ##__VA_ARGS__);                          \
    }                                                                                    \
  } while(false)

#define D_DEV_LOGD_SPLIT(fmt, ...)                                                    \
    do {                                                                              \
        if (IsLogEnableDebug()) {                                                     \
            DeviceLogSplitDebug(__FUNCTION__, fmt, ##__VA_ARGS__);                    \
        }                                                                             \
    } while (false)

#define DEV_VERBOSE_DEBUG(fmt, args...)                                               \
  do {                                                                                \
    if constexpr (IsCompileVerboseLog())  {                                           \
        D_DEV_LOGD(fmt, ##args);                                                      \
    }                                                                                 \
  } while(0)

#define DEV_DEBUG(fmt, args...) D_DEV_LOGD(fmt, ##args)
#define DEV_INFO(fmt, args...) D_DEV_LOGI(fmt, ##args)
#define DEV_WARN(fmt, args...) D_DEV_LOGW(fmt, ##args)
#define DEV_ERROR(errCode, fmt, args...) D_DEV_LOGE(errCode, fmt, ##args)
#define DEV_DEBUG_SPLIT(fmt, args...) D_DEV_LOGD_SPLIT(fmt, ##args)

#define DEV_ASSERT_MSG(errCode, expr, fmt, args...)                              \
    do {                                                                \
        if (!(expr)) {                                                  \
            DEV_ERROR(errCode, "Assertion failed (%s): " fmt, #expr, ##args);    \
            assert(0);                                                  \
        }                                                               \
    } while (0)

#define DEV_ASSERT(errCode, expr)                                                \
    do {                                                                \
        if (!(expr)) {                                                  \
            DEV_ERROR(errCode, "Assertion failed (%s)", #expr);                  \
            assert(0);                                                  \
        }                                                               \
    } while (0)

#define DEV_MEM_DUMP(fmt, args...)

#else  // none device
inline bool IsDebugMode() {
    return true;
}

#define DEV_VERBOSE_DEBUG(fmt, args...) PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_DEBUG_SPLIT(fmt, args...)   PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_DEBUG(fmt, args...)         PYPTO_SIM_LOG(DLOG_DEBUG, MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...)          PYPTO_SIM_LOG(DLOG_INFO, MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...)          PYPTO_SIM_LOG(DLOG_WARN, MACHINE, fmt, ##args)
#define DEV_ERROR(errCode, fmt, args...)         PYPTO_SIM_LOGE(MACHINE, errCode, fmt, ##args)

#if DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE
#define DEV_MEM_DUMP(fmt, args...) MACHINE_LOGD("[WsMem Statistics] " fmt, ##args)
#else
#define DEV_MEM_DUMP(fmt, args...)
#endif // DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE

#define DEV_ASSERT_MSG(errCode, expr, fmt, args...)           \
    do {                                             \
        if (!(expr)) {                               \
            MACHINE_LOGE(errCode, "%s :" fmt, #expr, ##args); \
            assert(0);                               \
        }                                            \
    } while (0)

#define DEV_ASSERT(errCode, expr)               \
    do {                               \
        if (!(expr)) {                 \
            MACHINE_LOGE(errCode, "%s", #expr); \
            assert(0);                 \
        }                              \
    } while (0)

#endif

#define BACKTRACE_STACK_COUNT 64

static inline void PrintBacktrace[[maybe_unused]](const npu::tile_fwk::ThreadErr errcode, const std::string &prefix = "", int count = BACKTRACE_STACK_COUNT) {
    std::vector<void *> backtraceStack(count);
    int backtraceStackCount = backtrace(backtraceStack.data(), static_cast<int>(backtraceStack.size()));
    DEV_ERROR(errcode, "backtrace %s count:%d", prefix.c_str(), backtraceStackCount);
    char **backtraceSymbolList = backtrace_symbols(backtraceStack.data(), backtraceStackCount);
    for (int i = 0; i < backtraceStackCount; i++) {
        DEV_ERROR(errcode, "backtrace %s frame[%d]: %s", prefix.c_str(), i, backtraceSymbolList[i]);
    }
    free(backtraceSymbolList);
}
} // namespace npu::tile_fwk