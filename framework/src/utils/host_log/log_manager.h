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
 * \file log_manager.h
 * \brief
 */

#pragma once

#include <string>
#include <mutex>
#include <queue>
#include <cstdarg>
#include <fstream>

namespace npu::tile_fwk {
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    EVENT = 4,
    NONE = 5
};
const size_t kMsgMaxLen = 1024;
struct LogMsg {
    char msg[kMsgMaxLen];
    size_t length;
};
struct LogAttr {
    bool isDevice{false};
    int32_t deviceId{0};
};
class LogManager {
public:
    static LogManager &Instance();
    bool CheckLevel(const LogLevel logLevel) const;
    void Record(const LogLevel logLevel, const char *fmt, va_list list);
    void EnableHostLog() { attr_.isDevice = false; }
    void EnableDeviceLog(const int32_t deviceId = 0) {
        attr_.isDevice = true;
        attr_.deviceId = deviceId;
    }
private:
    LogManager();
    ~LogManager();
    void SetLogLevel(const LogLevel logLevel);
    static void ConstructMessage(const LogLevel logLevel, const char *fmt, va_list list, LogMsg &logMsg);
    static void ConstructMsgHeader(const LogLevel logLevel, LogMsg &logMsg);
    static void ConstructMsgTail(LogMsg &logMsg);
    void WriteMessage(const LogMsg &logMsg);
    void WriteToStdOut(const LogMsg &logMsg);
    void WriteToFile(const LogMsg &logMsg);
    void CreateAndOpenNewLogFile();
    static void CheckAndCloseLogFile(std::ofstream &currentFileStream, std::queue<std::string> &logFilesQueue);
    const std::string& GetLogDir() const { return attr_.isDevice ? deviceLogDir_ : hostLogDir_; }
    std::ofstream& GetCurrentFileStream() { return attr_.isDevice ? devFileStream_ : hostFileStream_; }
    std::queue<std::string>& GetLogFilesQueue() { return attr_.isDevice ? devLogFiles_ : hostLogFiles_; }
    const std::queue<std::string>& GetLogFilesQueue() const { return attr_.isDevice ? devLogFiles_ : hostLogFiles_; }
private:
    LogLevel level_{LogLevel::ERROR};
    bool enableEvent_{false};
    bool enableStdOut_{true};
    std::string hostLogDir_;
    std::string deviceLogDir_;
    std::ofstream hostFileStream_;
    std::ofstream devFileStream_;
    std::queue<std::string> hostLogFiles_;
    std::queue<std::string> devLogFiles_;
    std::mutex writeMutex_;
    LogAttr attr_;
};
}
