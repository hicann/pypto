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
 * \file ModelLogger.h
 * \brief
 */

#pragma once
#ifndef COST_MODEL_LOG_H
#define COST_MODEL_LOG_H

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <map>

namespace CostModel {

class TTYCmd {
public:
    unsigned char code;

    explicit TTYCmd(int c) : code(c) {}
    [[nodiscard]] std::string Str() const
    {
        return "\033[" + std::to_string(code) + "m";
    }
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
    NONE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL,
    DEFAULT = INFO,
};

class StdLogger {
public:
    StdLogger &Log(TTYCmd &&cmd)
    {
        std::cout << cmd.Str();
        return *this;
    }
    template <typename T>
    StdLogger &Log(T &&t)
    {
        std::cout << (std::forward<T>(t));
        return *this;
    }

    StdLogger() = default;
    StdLogger operator=(const StdLogger &) = delete;

private:
    StdLogger(const StdLogger &) = delete;
};

class FileLogger {
public:
    std::ofstream ofs;

    FileLogger(const std::string &filepath, bool append)
    {
        if (append) {
            ofs.open(filepath, std::ios_base::app);
        } else {
            ofs.open(filepath);
        }
    }

    FileLogger &Log()
    {
        return *this;
    }

    template <typename T>
    FileLogger &Log(T &&t)
    {
        ofs << (std::forward<T>(t));
        return *this;
    }
    FileLogger operator=(const FileLogger &) = delete;
private:
    FileLogger(const FileLogger &) = delete;
};

class LineLogger : public std::vector<std::string> {
public:
    LineLogger &Log()
    {
        return *this;
    }
    LineLogger &Log(std::string &&t)
    {
        this->emplace_back(t);
        return *this;
    }
};

class LoggerManager {
public:
    std::mutex logMtx;
    LoggerLevel level{LoggerLevel::DEFAULT};
    bool stdEnabled{true};
    StdLogger stdLogger;
    std::unordered_map<std::string, std::unique_ptr<FileLogger>> fileLoggerDict;
    std::unordered_map<std::string, std::shared_ptr<LineLogger>> lineLoggerDict;
    std::map<int, CostModel::LoggerLevel> levelMap = {
        {1, CostModel::LoggerLevel::DEBUG}, {2, CostModel::LoggerLevel::INFO},  {3, CostModel::LoggerLevel::WARN},
        {4, CostModel::LoggerLevel::ERROR}, {5, CostModel::LoggerLevel::FATAL},
    };
    LoggerManager() = default;

    template <typename T>
    void Log(LoggerLevel l, T &&t, T &&tRich)
    {
        std::lock_guard lock(logMtx);
        if (l >= level) {
            if (stdEnabled) {
                std::cout << tRich.c_str() << std::endl;
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
    }

    static LoggerLevel ConvertLevel(int logLevel)
    {
        return GetManager().levelMap[logLevel];
    }

    static void ResetLevel(LoggerLevel l)
    {
        GetManager().level = l;
    }

    static void StdLoggerEnable(bool enabled)
    {
        GetManager().stdEnabled = enabled;
    }

    static void FileLoggerRegister(const std::string &filepath, bool append)
    {
        GetManager().fileLoggerDict.try_emplace(filepath, std::make_unique<FileLogger>(filepath, append));
    }

    static void FileLoggerUnregister(const std::string &filepath)
    {
        GetManager().fileLoggerDict.erase(filepath);
    }

    static std::shared_ptr<LineLogger> LineLoggerRegister(const std::string &name)
    {
        auto logger = std::make_shared<LineLogger>();
        GetManager().lineLoggerDict[name] = logger;
        return logger;
    }

    static void LineLoggerUnregister(const std::string &name)
    {
        GetManager().lineLoggerDict.erase(name);
    }

    friend class Logger;

private:
    static LoggerManager &GetManager()
    {
        static LoggerManager manager;
        return manager;
    }
};

class Logger {
private:
    std::stringstream ss;
    std::stringstream ssRich;
    std::stringstream ssFunc;
    LoggerLevel level{LoggerLevel::INFO};

public:
    Logger(LoggerLevel lev, const std::string &func, int line) : level(lev)
    {
        ssFunc << func << " line:" << std::to_string(line);
    }

    ~Logger()
    {
        LoggerManager::GetManager().Log(level, ss.str(), ssRich.str());
    }

    Logger &Log(TTYCmd &&val)
    {
        ssRich << val.Str();
        return *this;
    }

    template <typename T>
    Logger &Log(T &&val)
    {
        ss << (std::forward<T>(val));
        ssRich << (std::forward<T>(val));
        return *this;
    }

    template <typename... Tys>
    Logger &operator()(Tys &&...vals)
    {
        if constexpr (sizeof...(Tys) > 0) {
            (Log(std::forward<Tys>(vals)), ...);
        }
        return *this;
    }
};
}  // namespace CostModel

#define MLOG_DEBUG CostModel::Logger(CostModel::LoggerLevel::DEBUG, __func__, __LINE__)
#define MLOG_INFO CostModel::Logger(CostModel::LoggerLevel::INFO, __func__, __LINE__)
#define MLOG_WARN CostModel::Logger(CostModel::LoggerLevel::WARN, __func__, __LINE__)
#define MLOG_ERROR CostModel::Logger(CostModel::LoggerLevel::ERROR, __func__, __LINE__)
#define MLOG_FATAL CostModel::Logger(CostModel::LoggerLevel::FATAL, __func__, __LINE__)

#endif