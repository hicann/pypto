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
#ifdef __DEVICE__
#include "dlog_pub.h"
#endif
namespace npu::tile_fwk {

#define DEV_IF_NONDEVICE                                                        \
    if constexpr (!IsDeviceMode())

#define DEV_IF_DEVICE                                                           \
    if constexpr (IsDeviceMode())

#define DEV_IF_DEBUG                                                        \
    if (IsDebugMode())

#define DEV_IF_VERBOSE_DEBUG                                            \
    if constexpr (IsCompileVerboseLog())

#if ENABLE_TMP_LOG == 0
#define DEBUG_PLOG 1
#else
#define DEBUG_PLOG 0
#endif/*DEBUG_PLOG*/

inline constexpr bool IsCompileVerboseLog() {
#if ENABLE_COMPILE_VERBOSE_LOG
    return true;
#else
    return false;
#endif
}

constexpr int LOG_LEVEL_DEBUG = 0;
constexpr int LOG_LEVEL_INFO = 1;
constexpr int LOG_LEVEL_WARN = 2;
constexpr int LOG_LEVEL_ERROR = 3;

static const char *g_levelName[] = {"DEBUG", "INFO", "WARN", "ERROR"};

class DeviceLogger {
public:
    explicit DeviceLogger(int level = LOG_LEVEL_INFO) : level_(level){};

    int Level() const { return level_; }

    void Log(int level, const char *file, int line, const char *fmt, ...) const __attribute__((format(printf, 5, 6))) {
        if (level < level_) {
            return;
        }

        struct timeval tv;
        gettimeofday(&tv, nullptr);
        const char *fileName = strrchr(file, '/');
        if (fileName != nullptr) {
            file = fileName + 1;
        }
        va_list ap;
        va_start(ap, fmt);
        if (fp_) {
            fprintf(fp_, "%ld.%06ld [%s] %s:%d", tv.tv_sec, tv.tv_usec, g_levelName[level], file, line);
            vfprintf(fp_, fmt, ap);
            fprintf(fp_, "\n");
        } else {
            printf("%ld.%06ld [%s] %s:%d", tv.tv_sec, tv.tv_usec, g_levelName[level], file, line);
            vprintf(fmt, ap);
            printf("\n");
        }
        va_end(ap);
        Flush();
    }

    void Flush() const {
        if (fp_) {
            fflush(fp_);
        }
    }

    void SetLogFile(const char *logfile) {
        if (strcmp(logfile, logfile_.c_str()) == 0) {
            return;
        }

        if (fp_) {
            fclose(fp_);
        }

        fp_ = fopen(logfile, "wb+");
        logfile_ = logfile;
    }

    ~DeviceLogger() {
        if (fp_) {
            fclose(fp_);
        }
    }

private:
    FILE *fp_{nullptr};
    std::string logfile_;
    int level_;
};

inline DeviceLogger &GetLogger(const char *logfile = nullptr, int level = LOG_LEVEL_DEBUG) {
    (void)logfile;
    (void)level;
    thread_local DeviceLogger devLogger(level);
#if ENABLE_TMP_LOG || !defined(__DEVICE__)
    if (logfile != nullptr) {
        devLogger.SetLogFile(logfile);
    }
#endif
    return devLogger;
}

enum class LogType {
    LOG_TYPE_SCHEDULER,    // 调度器日志
    LOG_TYPE_CONTROLLER,   // 控制器日志
    LOG_TYPE_PREFETCH      // 预取日志
};

// 创建日志文件
void SetLogFilePrefix(const std::string &prefix);
void CreateLogFile(LogType type, int threadIdx);

void InitLogSwitch();

extern bool g_isLogEnableDebug;
extern bool g_isLogEnableInfo;
extern bool g_isLogEnableWarn;
extern bool g_isLogEnableError;

#if DEBUG_PLOG && defined(__DEVICE__)
static inline bool IsLogEnableDebug() { return g_isLogEnableDebug; }
static inline bool IsLogEnableInfo() { return g_isLogEnableInfo; }
static inline bool IsLogEnableWarn() { return g_isLogEnableWarn; }
static inline bool IsLogEnableError() { return g_isLogEnableError; }
#else
#if ENABLE_TMP_LOG
static inline bool IsLogEnableDebug() { return true; }
static inline bool IsLogEnableInfo() { return true; }
static inline bool IsLogEnableWarn() { return true; }
static inline bool IsLogEnableError() { return true; }
#else
static inline bool IsLogEnableDebug() { return false; }
static inline bool IsLogEnableInfo() { return false; }
static inline bool IsLogEnableWarn() { return false; }
static inline bool IsLogEnableError() { return true; }
#endif
#endif

#if DEBUG_PLOG && defined(__DEVICE__)
#define GET_TID() syscall(__NR_gettid)
const std::string TILE_FWK_DEVICE_MACHINE = "AI_CPU";

inline bool IsDebugMode() {
    return g_isLogEnableDebug;
}

template<typename... Args>
inline void DeviceLogSplitDebug([[maybe_unused]] const std::string& mode_name,
                                const char* func, const char* format, Args... args)
    {
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
            dlog_debug(AICPU, "%lu %s\n%s", GET_TID(), func, log_content.c_str());
        } else {
            size_t start = 0;
            int segment_num = 0;
            size_t total_len = log_content.size();
            size_t total_segments = (total_len + MAX_LOG_CHUNK - 1) / MAX_LOG_CHUNK;
            while (start < total_len) {
                size_t chunk_size = (total_len - start > MAX_LOG_CHUNK) ? MAX_LOG_CHUNK : total_len - start;
                std::string segment = log_content.substr(start, chunk_size);
                dlog_debug(AICPU, "%lu %s [Segment %d/%lu]\n%s",
                           GET_TID(), func, segment_num + 1, total_segments, segment.c_str());
                start += chunk_size;
                segment_num++;
            }
        }
    }

#define D_DEV_LOGD(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogEnableDebug()) {                                                  \
        dlog_debug(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                               \
  } while (false)

#define D_DEV_LOGI(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogEnableInfo()) {                                                   \
        dlog_info(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                               \
  } while(false)

#define D_DEV_LOGW(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogEnableWarn()) {                                                   \
        dlog_warn(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                               \
  } while(false)

#define D_DEV_LOGE(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
    if (IsLogEnableError()) {                                                  \
        dlog_error(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                               \
  } while(false)

#define D_DEV_LOGD_SPLIT(MODE_NAME, fmt, ...)                                         \
    do {                                                                                \
        if (IsLogEnableDebug()) {                                                       \
            DeviceLogSplitDebug(MODE_NAME, __FUNCTION__, fmt, ##__VA_ARGS__);             \
        }                                                                               \
    } while (false)

#define DEV_VERBOSE_DEBUG(fmt, args...)                                  \
  do {                                                                  \
    if constexpr (IsCompileVerboseLog())  {                          \
        D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args);               \
    }                                                                   \
  } while(0)
#define DEV_DEBUG(fmt, args...) D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...) D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...) D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_DEBUG_SPLIT(fmt, args...) D_DEV_LOGD_SPLIT(TILE_FWK_DEVICE_MACHINE, fmt, ##args)

#define DEV_ASSERT_MSG(expr, fmt, args...)                              \
    do {                                                                \
        if (!(expr)) {                                                  \
            DEV_ERROR("Assertion failed (%s): " fmt, #expr, ##args);    \
            assert(0);                                                  \
        }                                                               \
    } while (0)

#define DEV_ASSERT(expr)                                                \
    do {                                                                \
        if (!(expr)) {                                                  \
            DEV_ERROR("Assertion failed (%s)", #expr);                  \
            assert(0);                                                  \
        }                                                               \
    } while (0)

#define DEV_MEM_DUMP(fmt, args...)

#else

inline bool IsDebugMode() {
    return true;
}

#define DEV_VERBOSE_DEBUG(fmt, args...)  \
    do { \
        if (IsLogEnableDebug()) { \
            GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args); \
        } \
    } while (0)
#define DEV_DEBUG(fmt, args...) \
    do { \
        if (IsLogEnableDebug()) { \
            GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args); \
        } \
    } while (0)
#define DEV_INFO(fmt, args...) \
    do { \
        if (IsLogEnableInfo()) { \
            GetLogger().Log(LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, ##args); \
        } \
    } while (0)
#define DEV_WARN(fmt, args...) \
    do { \
        if (IsLogEnableWarn()) { \
            GetLogger().Log(LOG_LEVEL_WARN, __FILE__, __LINE__, fmt, ##args); \
        } \
    } while (0)
#define DEV_ERROR(fmt, args...) \
    do { \
        if (IsLogEnableError()) { \
            GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args); \
        } \
    } while (0)
#define DEV_DEBUG_SPLIT(fmt, args...) \
    do { \
        if (IsLogEnableDebug()) { \
            GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args); \
        } \
    } while (0)

#if DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE
#define DEV_MEM_DUMP(fmt, args...) GetLogger().Log(LOG_LEVEL_DEBUG, "/memdump", 0, "[WsMem Statistics] " fmt, ##args)
#else
#define DEV_MEM_DUMP(fmt, args...)
#endif // DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE

#define DEV_ASSERT_MSG(expr, fmt, args...)                                                   \
    do {                                                                                     \
        if (!(expr)) {                                                                       \
            GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, "%s :" fmt, #expr, ##args); \
            GetLogger().Flush();                                                             \
            assert(0);                                                                       \
        }                                                                                    \
    } while (0)

#define DEV_ASSERT(expr)                                                       \
    do {                                                                       \
        if (!(expr)) {                                                         \
            GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, "%s", #expr); \
            GetLogger().Flush();                                               \
            assert(0);                                                         \
        }                                                                      \
    } while (0)

#endif // DEBUG_PLOG

#define BACKTRACE_STACK_COUNT 64

static inline void PrintBacktrace(const std::string &prefix = "", int count = BACKTRACE_STACK_COUNT) {
    std::vector<void *> backtraceStack(count);
    int backtraceStackCount = backtrace(backtraceStack.data(), static_cast<int>(backtraceStack.size()));
    DEV_ERROR("backtrace %s count:%d", prefix.c_str(), backtraceStackCount);
    char **backtraceSymbolList = backtrace_symbols(backtraceStack.data(), backtraceStackCount);
    for (int i = 0; i < backtraceStackCount; i++) {
        DEV_ERROR("backtrace %s frame[%d]: %s", prefix.c_str(), i, backtraceSymbolList[i]);
    }
    free(backtraceSymbolList);
}

} // namespace npu::tile_fwk