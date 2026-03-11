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
 * \file interpreter_log_test_utils.h
 * \brief Common helpers for interpreter/log related unit tests:
 *        - CaptureLogFileAndEcho: capture log output from log files (落盘形式，与 LogManager 路径一致)
 *        - VerifyLogContainsFailed: check VERIFY log failures
 *        - VerifyLogContainsIndex0Failed: check VERIFY index 0 failures
 */

#pragma once

#include <functional>
#include <string>
#include <cstdio>
#include <regex>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <map>
#include <sys/syscall.h>
#include <unistd.h>

// 与 LogManager 落盘路径一致（仅本头文件内使用）
static constexpr const char *kInterpLogTestEnvProcessLogPath = "ASCEND_PROCESS_LOG_PATH";
static constexpr const char *kInterpLogTestHostLogFilePrefix = "pypto-log-";
static constexpr const char *kInterpLogTestLogFileSuffix = ".log";
static constexpr const char *kInterpLogTestHostLogSubDir = "/debug/plog";
static constexpr const char *kInterpLogTestDefaultLogSubDir = "/ascend/log";

static inline std::string InterpLogTestGetHostLogDir() {
    const char *envPath = std::getenv(kInterpLogTestEnvProcessLogPath);
    if (envPath != nullptr && envPath[0] != '\0') {
        return std::string(envPath) + kInterpLogTestHostLogSubDir;
    }
    const char *home = std::getenv("HOME");
    std::string base = (home != nullptr && home[0] != '\0') ? std::string(home) : ".";
    return base + kInterpLogTestDefaultLogSubDir + kInterpLogTestHostLogSubDir;
}

// 与 LogManager 一致：日志文件名形如 pypto-log-<tid>-<timestamp>.log，这里按当前线程 tid 过滤
static inline std::string InterpLogTestGetThreadLogPrefix() {
    return std::string(kInterpLogTestHostLogFilePrefix) + std::to_string(getpid()) + "_";
}

static inline std::map<std::string, size_t> InterpLogTestListHostLogFilesWithSize(const std::string &dir,
                                                                                  const std::string &threadPrefix) {
    std::map<std::string, size_t> result;
    DIR *d = opendir(dir.c_str());
    if (d == nullptr) {
        return result;
    }
    struct dirent *dp = nullptr;
    while ((dp = readdir(d)) != nullptr) {
        if (dp->d_name[0] == '.') {
            continue;
        }
        std::string name = dp->d_name;
        if (name.find(threadPrefix) != 0 ||
            name.rfind(kInterpLogTestLogFileSuffix) != name.size() - std::strlen(kInterpLogTestLogFileSuffix)) {
            continue;
        }
        std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            result[path] = static_cast<size_t>(st.st_size);
        }
    }
    closedir(d);
    return result;
}

// 从日志落盘目录捕获本次 func() 执行产生的新增日志内容（与 LogManager 落盘路径一致）
static inline std::string CaptureLogFileAndEcho(std::function<void()> func) {
    std::string logDir = InterpLogTestGetHostLogDir();
    std::string threadPrefix = InterpLogTestGetThreadLogPrefix();
    std::map<std::string, size_t> before = InterpLogTestListHostLogFilesWithSize(logDir, threadPrefix);

    func();

    std::map<std::string, size_t> after = InterpLogTestListHostLogFilesWithSize(logDir, threadPrefix);

    // 一个用例的日志只会落在单个文件中：这里选择“本次增长字节数最大”的那个文件，仅读它的增量部分
    std::string targetPath;
    size_t targetOldSize = 0;
    size_t targetDelta = 0;
    for (const auto &p : after) {
        const std::string &path = p.first;
        size_t newSize = p.second;
        size_t oldSize = 0;
        auto it = before.find(path);
        if (it != before.end()) {
            oldSize = it->second;
        }
        if (newSize <= oldSize) {
            continue;
        }
        size_t delta = newSize - oldSize;
        if (delta > targetDelta) {
            targetDelta = delta;
            targetOldSize = oldSize;
            targetPath = path;
        }
    }

    if (targetPath.empty()) {
        return "";
    }

    std::ifstream ifs(targetPath, std::ios::binary);
    if (!ifs) {
        return "";
    }
    ifs.seekg(static_cast<std::streamoff>(targetOldSize));
    std::string captured;
    captured.append(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    return captured;
}

// 捕获 stdout 输出，用于校验日志内容
static inline std::string CaptureStdoutAndEcho(std::function<void()> func) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return "";
    }

    int old_stdout = dup(STDOUT_FILENO);
    if (old_stdout == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }

    if (dup2(pipefd[1], STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        close(old_stdout);
        return "";
    }

    close(pipefd[1]);
    func();
    fflush(stdout);

    if (dup2(old_stdout, STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(old_stdout);
        return "";
    }

    char buffer[8192] = {0};
    ssize_t len = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);

    std::string captured(len > 0 ? static_cast<size_t>(len) : 0, '\0');
    if (len > 0) {
        captured.assign(buffer, static_cast<size_t>(len));
        // 同时打印到控制台，便于调试查看
        ssize_t written = write(old_stdout, buffer, static_cast<size_t>(len));
        (void)written;
    }
    close(old_stdout);

    return captured;
}

// 仅检查 [VERIFY] 日志行中是否出现 FAILED，其他模块日志不参与判断
inline bool VerifyLogContainsFailed(const std::string& logOutput) {
    // 匹配形如："...[VERIFY]...FAILED..."，且 [VERIFY] 与 FAILED 必须在同一行（中间不允许换行）
    static const std::regex kVerifyFailedPattern(R"(\[VERIFY][^\n]*FAILED)");
    return std::regex_search(logOutput, kVerifyFailedPattern);
}

// 仅检查 [VERIFY] 日志行中 index 0 是否出现 FAILED，用于 Topk 用例
inline bool VerifyLogContainsIndex0Failed(const std::string& logOutput) {
    // 只关心 flow_verifier 打印的 index 0 结果行：
    // "... [VERIFY]: ... Verify for ... index 0 result FAILED"
    static const std::regex kVerifyIndex0FailedPattern(R"(\[VERIFY][^\n]*index 0 result FAILED)");
    return std::regex_search(logOutput, kVerifyIndex0FailedPattern);
}

