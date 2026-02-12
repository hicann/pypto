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
 * \file log_manager.cpp
 * \brief
 */

#include "host_log/log_manager.h"
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/syscall.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <map>
#include "securec.h"

namespace npu::tile_fwk {
namespace {
constexpr const char *kEnvGlobalLogLevel = "ASCEND_GLOBAL_LOG_LEVEL";
constexpr const char *kEnvModuleLogLevel = "ASCEND_MODULE_LOG_LEVEL";
constexpr const char *kEnvGlobalLogEvent = "ASCEND_GLOBAL_EVENT_ENABLE";
constexpr const char *kEnvProcessLogPath = "ASCEND_PROCESS_LOG_PATH";
constexpr const char *kModuleName = "PYPTO";
constexpr const char *kModulePrefix = "PYPTO=";

const std::string kLogLevelNoneStr = "NONE";
const std::map<LogLevel, std::string> logLevelStrMap = {
    {LogLevel::DEBUG, "DEBUG"},
    {LogLevel::INFO, "INFO"},
    {LogLevel::WARN, "WARN"},
    {LogLevel::ERROR, "ERROR"},
    {LogLevel::EVENT, "EVENT"}
};

uint64_t GetTid() {
    thread_local uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

const std::string& GetLogLevelStr(const LogLevel logLevel) {
    const auto iter = logLevelStrMap.find(logLevel);
    return iter != logLevelStrMap.end() ? iter->second : kLogLevelNoneStr;
}

bool GetEnvStr(const char *envName, std::string &envValue) {
    const size_t envValueMaxLen = 1024UL;
    const char *envTemp = std::getenv(envName);
    if (envTemp == nullptr || strnlen(envTemp, envValueMaxLen) >= envValueMaxLen) {
        return false;
    }
    envValue = envTemp;
    return true;
}

std::string GetCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm* nowTm = std::localtime(&nowTime);
    std::stringstream ss;
    ss << std::put_time(nowTm, "%Y-%m-%d %H:%M:%S");
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    ss << "." << std::setfill('0') << std::setw(3) << milliseconds.count();
    return ss.str();
}

int ParseStrToInt(const std::string &str) {
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << '\n';
    } catch (const std::out_of_range& oor) {
        std::cerr << "Out of Range error: " << oor.what() << '\n';
    }
    return -1;
}

int GetModLogLevel(const std::string &str) {
    if (str.empty()) {
        return -1;
    }
    size_t posLeft = str.find(kModulePrefix);
    if (posLeft == std::string::npos) {
        return -1;
    }
    size_t posRight = str.find(kModulePrefix, posLeft);
    if (posRight == std::string::npos) {
        return ParseStrToInt(str.substr(posLeft + strlen(kModulePrefix)));
    }
    return ParseStrToInt(str.substr(posLeft + strlen(kModulePrefix), posRight - posLeft - strlen(kModulePrefix)));
}
}
LogManager &LogManager::Instance() {
    static LogManager instance;
    return instance;
}

LogManager::LogManager() {
    std::string envGlobalLogLevel;
    if (GetEnvStr(kEnvGlobalLogLevel, envGlobalLogLevel)) {
        SetLogLevel(static_cast<LogLevel>(ParseStrToInt(envGlobalLogLevel)));
    }

    std::string envModuleLogLevel;
    if (GetEnvStr(kEnvModuleLogLevel, envModuleLogLevel)) {
        SetLogLevel(static_cast<LogLevel>(GetModLogLevel(envModuleLogLevel)));
    }

    std::string envGlobalEvent;
    if (GetEnvStr(kEnvGlobalLogEvent, envGlobalEvent)) {
        enableEvent_ = ParseStrToInt(envGlobalEvent) != 0;
    }

    std::string envProcessLogPath;
    if (GetEnvStr(kEnvProcessLogPath, envProcessLogPath)) {
        fileDir_ = envProcessLogPath;
    }
}

LogManager::~LogManager() {
    level_ = LogLevel::ERROR;
    enableStdOut_ = true;
    fileDir_.clear();
    std::queue<std::string> tmp_files;
    logFiles_.swap(tmp_files);
}

void LogManager::SetLogLevel(const LogLevel logLevel) {
    if (logLevel >= LogLevel::DEBUG && logLevel < LogLevel::NONE) {
        level_ = logLevel;
    }
}

bool LogManager::CheckLevel(const LogLevel logLevel) const {
    if (logLevel == LogLevel::EVENT) {
        return enableEvent_;
    }
    return logLevel >= level_;
}

void LogManager::Record(const LogLevel logLevel, const char *fmt, va_list list) {
    LogMsg logMsg{};
    ConstructMessage(logLevel, fmt, list, logMsg);
    WriteMessage(logMsg);
}

void LogManager::ConstructMessage(const LogLevel logLevel, const char *fmt, va_list list, LogMsg &logMsg) {
    ConstructMsgHeader(logLevel, logMsg);
    int ret = vsnprintf_truncated_s(logMsg.msg + logMsg.length, kMsgMaxLen - logMsg.length, fmt, list);
    if (ret < 0) {
        std::cerr << "Constrcut message failed: " << ret << std::endl;
        return;
    }
    logMsg.length += static_cast<size_t>(ret);

    ConstructMsgTail(logMsg);
}

void LogManager::ConstructMsgHeader(const LogLevel logLevel, LogMsg &logMsg) {
    int ret = snprintf_s(logMsg.msg, kMsgMaxLen, kMsgMaxLen - 1, "[%s] %s(%lu):%s ",
                         GetLogLevelStr(logLevel).c_str(), kModuleName, GetTid(), GetCurrentTime().c_str());
    if (ret < 0) {
        std::cerr << "Construct log msg hader failed: " << ret << std::endl;
        return;
    }
    logMsg.length = static_cast<size_t>(ret);
}

void LogManager::ConstructMsgTail(LogMsg &logMsg) {
    if (logMsg.msg[logMsg.length - 1] != '\n') {
        if (logMsg.length < kMsgMaxLen) {
            logMsg.msg[logMsg.length] = '\n';
            logMsg.length++;
        } else {
            logMsg.msg[kMsgMaxLen - 1] = '\n';
        }
    }
}

void LogManager::WriteMessage(const LogMsg &logMsg) {
    const std::lock_guard lockGuard(writeMutex_);
    if (enableStdOut_) {
        WriteToStdOut(logMsg);
    }
}

void LogManager::WriteToStdOut(const LogMsg &logMsg) {
    int fd = fileno(stdout);
    if (fd <= 0) {
        std::cerr << "Cannot get fileno of stdout" << std::endl;
    }
    int ret = write(fd, logMsg.msg, logMsg.length);
    if (ret < 0) {
        std::cerr << "Cannot write to stdout: " << ret << std::endl;
    }
}
}
