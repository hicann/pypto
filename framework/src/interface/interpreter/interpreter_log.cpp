/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file interpreter_log.cpp
 * \brief Interpreter logging helpers and macros implementation.
 */

#include "interface/interpreter/interpreter_log.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>

#include "securec.h"

namespace npu::tile_fwk::interpreter {
namespace {
bool IsEnvEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct LogContext {
    std::mutex mutex;
    std::string logFilePath = "output/interpreter.log";
    FILE* logFile = nullptr;
    bool printToStdout = false;
    bool writeInfoToFile = false;

    LogContext()
        : printToStdout(IsEnvEnabled("ASCEND_SLOG_PRINT_TO_STDOUT")),
          writeInfoToFile(IsEnvEnabled("ASCEND_INTERPRETER_LOG_INFO_TO_FILE"))
    {
    }

    ~LogContext()
    {
        if (logFile != nullptr) {
            fclose(logFile);
            logFile = nullptr;
        }
    }
};

LogContext& GetLogContext()
{
    static LogContext context;
    return context;
}

bool ShouldWriteLevel(LogLevel level, const LogContext& context)
{
    switch (level) {
        case LogLevel::kError:
        case LogLevel::kEvent:
        case LogLevel::kWarn:
            return true;
        case LogLevel::kInfo:
            return context.writeInfoToFile;
        case LogLevel::kDebug:
        default:
            return false;
    }
}

const char* LevelToString(LogLevel level)
{
    switch (level) {
        case LogLevel::kDebug:
            return "DEBUG";
        case LogLevel::kInfo:
            return "INFO";
        case LogLevel::kWarn:
            return "WARN";
        case LogLevel::kEvent:
            return "EVENT";
        case LogLevel::kError:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

void WriteLine(LogLevel level, const char* fmt, va_list args) __attribute__((format(printf, 2, 0)));
void WriteLine(LogLevel level, const char* fmt, va_list args)
{
    constexpr int kInitialBufSize = 1024;
    char stackBuf[kInitialBufSize];
    const char* msgPtr = stackBuf;
    std::string dynamicBuf;

    va_list argsCopy;
    va_copy(argsCopy, args);
    int msgLength = vsnprintf_s(stackBuf, sizeof(stackBuf), sizeof(stackBuf) - 1, fmt, argsCopy);
    va_end(argsCopy);
    if (msgLength < 0) {
        return;
    }
    // count 为 sizeof(stackBuf)-1，栈上最多容纳 kInitialBufSize-1 个字符加 '\0'。
    // 若返回值 >= kInitialBufSize（含等于），表示未完整写入或仍需更大缓冲，须走动态路径。
    if (msgLength >= kInitialBufSize) {
        dynamicBuf.resize(static_cast<size_t>(msgLength) + 1);
        va_copy(argsCopy, args);
        const int ret = vsnprintf_s(dynamicBuf.data(), dynamicBuf.size(), dynamicBuf.size() - 1, fmt, argsCopy);
        va_end(argsCopy);
        if (ret < 0) {
            return;
        }
        msgPtr = dynamicBuf.c_str();
    }

    std::time_t now = std::time(nullptr);
    std::tm localTm {};
    (void)localtime_r(&now, &localTm);
    char timeBuf[32] = {0};
    (void)std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &localTm);
    const uint64_t threadId = static_cast<uint64_t>(syscall(SYS_gettid));
    const char* levelStr = LevelToString(level);

    auto& context = GetLogContext();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.logFile == nullptr) {
        context.logFile = fopen(context.logFilePath.c_str(), "a");
        if (context.logFile == nullptr) {
            return;
        }
    }

    fprintf(context.logFile, "[%s][%s][tid:%" PRIu64 "] %s\n", timeBuf, levelStr, threadId, msgPtr);
    if (level == LogLevel::kError) {
        fflush(context.logFile);
    }
    if (context.printToStdout) {
        fprintf(stdout, "[%s][%s][tid:%" PRIu64 "] %s\n", timeBuf, levelStr, threadId, msgPtr);
    }
}
} // namespace

const std::string& LogFilePath()
{
    return GetLogContext().logFilePath;
}

void SetLogFilePath(const std::string& path)
{
    auto& context = GetLogContext();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (path.empty() || context.logFilePath == path) {
        return;
    }
    if (context.logFile != nullptr) {
        fclose(context.logFile);
        context.logFile = nullptr;
    }
    context.logFilePath = path;
}

void Log(LogLevel level, const char* fmt, ...)
{
    auto& context = GetLogContext();
    if (!ShouldWriteLevel(level, context)) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    WriteLine(level, fmt, args);
    va_end(args);
}
} // namespace npu::tile_fwk::interpreter
