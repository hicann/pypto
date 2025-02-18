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
 * \file log.h
 * \brief
 */

#pragma once
#ifndef LOG_H
#define LOG_H

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <cstdio>
#include "securec.h"

namespace npu::tile_fwk {

class TTYCmd {
public:
    unsigned char code;

    explicit TTYCmd(int codeIn) : code(codeIn) {}
    std::string Str() const { return "\033[" + std::to_string(code) + "m"; }
};

#define TTY_COLOR(n, ...) TTYCmd(n), ##__VA_ARGS__, TTYCmd(0)
#define TTY_RED(...) TTY_COLOR(31, __VA_ARGS__)
#define TTY_GREEN(...) TTY_COLOR(32, __VA_ARGS__)
#define TTY_YELLOW(...) TTY_COLOR(33, __VA_ARGS__)
#define TTY_BLUE(...) TTY_COLOR(34, __VA_ARGS__)
#define TTY_MAGENTA(...) TTY_COLOR(35, __VA_ARGS__)
#define TTY_CYAN(...) TTY_COLOR(36, __VA_ARGS__)
#define TTY_WHITE(...) TTY_COLOR(37, __VA_ARGS__)

enum class LoggerLevel {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    FATAL,
    EVENT,
    NONE,
};

class StdLogger {
public:
    StdLogger &Log(TTYCmd &&cmd) {
        std::cout << cmd.Str();
        return *this;
    }
    template <typename T>
    StdLogger &Log(T &&t) {
        std::cout << (std::forward<T>(t));
        return *this;
    }

    StdLogger() = default;

private:
    StdLogger(const StdLogger &) = delete;
    StdLogger &operator=(const StdLogger &) = delete;
};

class FileLogger {
public:
    std::ofstream ofs;

    FileLogger(const std::string &filepath, bool append) {
        if (append) {
            ofs.open(filepath, std::ios_base::app);
        } else {
            ofs.open(filepath);
        }
    }

    FileLogger &Log([[maybe_unused]] TTYCmd &&cmd) { return *this; }

    template <typename T>
    FileLogger &Log(T &&t) {
        ofs << (std::forward<T>(t));
        return *this;
    }

private:
    FileLogger(const FileLogger &) = delete;
    FileLogger &operator=(const FileLogger &) = delete;
};

class LineLogger : public std::vector<std::string> {
public:
    LineLogger &Log([[maybe_unused]] TTYCmd &&cmd) { return *this; }
    LineLogger &Log(std::string &&t) {
        this->emplace_back(t);
        return *this;
    }
};

class LoggerManager {
public:
    std::mutex logMtx;
    LoggerLevel level{LoggerLevel::ERROR};
    bool stdEnabled{true};
    StdLogger stdLogger;
    std::unordered_map<std::string, std::unique_ptr<FileLogger>> fileLoggerDict;
    std::unordered_map<std::string, std::shared_ptr<LineLogger>> lineLoggerDict;

    LoggerManager() = default;

    template <typename T>
    void Log(LoggerLevel l, T &&t, T &&tRich) {
        std::lock_guard lock(logMtx);
        if (l >= level) {
            if (stdEnabled) {
                stdLogger.Log(std::forward<T>(tRich));
            }
        }
        for (auto &[filepath, logger] : fileLoggerDict) {
            (void)filepath;
            logger->Log(std::forward<T>(t));
        }
        for (auto &[name, logger] : lineLoggerDict) {
            (void)name;
            logger->Log(std::forward<T>(t));
        }
    }

    static void ResetLevel(LoggerLevel l) { GetManager().level = l; }

    static void StdLoggerEnable(bool enabled) { GetManager().stdEnabled = enabled; }

    static void FileLoggerRegister(const std::string &filepath, bool append) {
        GetManager().fileLoggerDict.try_emplace(filepath, std::make_unique<FileLogger>(filepath, append));
    }

    static void FileLoggerUnregister(const std::string &filepath) { GetManager().fileLoggerDict.erase(filepath); }

    static void FileLoggerReplace(const std::string &oldfilepath, const std::string &newfilepath, bool append) {
        FileLoggerUnregister(oldfilepath);
        FileLoggerRegister(newfilepath, append);
    }

    static std::shared_ptr<LineLogger> LineLoggerRegister(const std::string &name) {
        auto logger = std::make_shared<LineLogger>();
        GetManager().lineLoggerDict[name] = logger;
        return logger;
    }

    static void LineLoggerUnregister(const std::string &name) { GetManager().lineLoggerDict.erase(name); }

    friend class Logger;
    static LoggerManager &GetManager() {
        static LoggerManager manager;
        return manager;
    }
};
constexpr uint32_t MAX_LOG_BUF_SIZE = 1024;
class Logger {
private:
    std::stringstream ss;
    std::stringstream ssRich;
    LoggerLevel level{LoggerLevel::ERROR};
    bool enableLog = false;

public:
    Logger(LoggerLevel levelIn, [[maybe_unused]] const std::string &func, [[maybe_unused]] int line) : level(levelIn) {
        enableLog = LoggerManager::GetManager().level <= level;
        if (enableLog) {
            static const char *MSG = "DIWEFVN";
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            auto tm = *std::localtime(&time);
            Log(std::put_time(&tm, "%F %T."));

            char buf[MAX_LOG_BUF_SIZE];
            auto epoch = now.time_since_epoch();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count() % 1000;
            sprintf_s(buf, MAX_LOG_BUF_SIZE, "%03d %c | ", static_cast<int>(ms), MSG[static_cast<int>(level)]);

            Log(buf);
            // Log(func + ":" + std::to_string(line) + " | ");
        }
    }

    ~Logger() {
        if (enableLog) {
            Log("\n");

            LoggerManager::GetManager().Log(level, ss.str(), ssRich.str());
        }
    }

    Logger &Log(TTYCmd &&val) {
        ssRich << val.Str();
        return *this;
    }

    template <typename T>
    Logger &Log(T &&val) {
        ss << (std::forward<T>(val));
        ssRich << (std::forward<T>(val));
        return *this;
    }

    template <typename T>
    Logger &operator<<(T &&val) {
        if (enableLog) {
            return Log(std::forward<T>(val));
        } else {
            return *this;
        }
    }

    template <typename... Tys>
    Logger &operator()(Tys &&...vals) {
        if (enableLog) {
            if constexpr (sizeof...(Tys) > 0) {
                (Log(std::forward<Tys>(vals)), ...);
            }
        }
        return *this;
    }
};
} // namespace npu::tile_fwk

#define ALOG_LEVEL(lvl) npu::tile_fwk::Logger(lvl, __func__, __LINE__)
#define ALOG_DEBUG ALOG_LEVEL(npu::tile_fwk::LoggerLevel::DEBUG)
#define ALOG_INFO ALOG_LEVEL(npu::tile_fwk::LoggerLevel::INFO)
#define ALOG_WARN ALOG_LEVEL(npu::tile_fwk::LoggerLevel::WARN)
#define ALOG_ERROR ALOG_LEVEL(npu::tile_fwk::LoggerLevel::ERROR)
#define ALOG_FATAL ALOG_LEVEL(npu::tile_fwk::LoggerLevel::FATAL)
#define ALOG_EVENT ALOG_LEVEL(npu::tile_fwk::LoggerLevel::EVENT)

#define ALOG_F(lvl, args...)                                                                \
    do {                                                                                    \
        if (LoggerManager::GetManager().level <= LoggerLevel::lvl) {                        \
            constexpr int defaultBufSize = 1024;                                            \
            std::string buf(defaultBufSize, '\0');                                          \
            int msgLength = snprintf_s(buf.data(), buf.size(), buf.size() - 1, ##args) + 1; \
            if (msgLength > defaultBufSize) {                                               \
                buf.resize(msgLength, '\0');                                                \
                snprintf_s(buf.data(), buf.size(), buf.size() - 1, ##args);                 \
            }                                                                               \
            ALOG_##lvl(buf.data());                                                         \
        }                                                                                   \
    } while (false)

#define ALOG_DEBUG_F(fmt, args...) ALOG_F(DEBUG, fmt, ##args)
#define ALOG_INFO_F(fmt, args...) ALOG_F(INFO, fmt, ##args)
#define ALOG_WARN_F(fmt, args...) ALOG_F(WARN, fmt, ##args)
#define ALOG_ERROR_F(fmt, args...) ALOG_F(ERROR, fmt, ##args)
#define ALOG_EVENT_F(fmt, args...) ALOG_F(EVENT, fmt, ##args)

#define ASLOGI ALOG_INFO_F
#define ASLOGE ALOG_ERROR_F

#endif
