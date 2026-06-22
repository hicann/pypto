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
 * \file pass_log.cpp
 * \brief
 */

#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include "passes/pass_log/pass_log.h"

#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "utils/file_utils.h"
#include "utils/host_log/log_manager.h"

#undef MODULE_NAME
#define MODULE_NAME "PassLog"

namespace npu::tile_fwk {
namespace {
const char* kExtractPassLogScriptName = "extract_pass_log.py";
const char* kExtractPassLogScriptInSource = "tools/scripts/extract_pass_log.py";
const char* kComputationGraphFolder = "computation_graph";

bool IsFileAccessible(const std::string& path) { return !path.empty() && access(path.c_str(), F_OK) == 0; }

void DeleteDirIfEmpty(const std::string& folder)
{
    if (folder.empty() || access(folder.c_str(), F_OK) != 0) {
        return;
    }
    auto files = GetFiles(folder, "");
    if (files.empty()) {
        (void)DeleteDir(folder);
    }
}

std::string ResolveExtractPassLogScriptPath()
{
    // Installed layout: <lib>/scripts/extract_pass_log.py
    const std::string installedScriptPath = GetPyptoLibPath() + "/scripts/" + kExtractPassLogScriptName;
    if (IsFileAccessible(installedScriptPath)) {
        return installedScriptPath;
    }

    // Source-tree layout (mainly for local development and UT).
    if (IsFileAccessible(kExtractPassLogScriptInSource)) {
        return std::string(kExtractPassLogScriptInSource);
    }
    return "";
}
} // namespace

std::string EscapeShellArg(const std::string& arg)
{
    std::string escaped = "'";
    for (char c : arg) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped += c;
        }
    }
    escaped += "'";
    return escaped;
}

// 入参为Operation对象
std::string GetFormatBacktrace(const Operation& op)
{
    auto span = op.GetSpan();
    if (!span.IsUnknown()) {
        return "";
    }

    std::ostringstream oss;
    oss << "[FuncMagic:" << op.BelongTo()->GetFuncMagic() << "]"
        << "[OpMagic:" << op.opmagic << "].";
    return oss.str();
}

// 入参为智能指针
std::string GetFormatBacktrace(const OperationPtr& op)
{
    if (!op) {
        return "";
    }
    return GetFormatBacktrace(*op);
}

// 入参为普通指针
std::string GetFormatBacktrace(const Operation* op)
{
    if (!op) {
        return "";
    }
    return GetFormatBacktrace(*op);
}

void LogPassRuntime(
    const std::string& identifier, Program& program, Function& function,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& start)
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    APASS_LOG_INFO_F(
        Elements::Function, "The Runtime of pass %s for program %s function %s is %ld us.", identifier.c_str(),
        program.Name().c_str(), function.GetMagicName().c_str(), duration.count());
}

void ExtractPassLogByFunction(const Function& function, const std::string& strategy)
{
    const std::string scriptPath = ResolveExtractPassLogScriptPath();
    if (scriptPath.empty()) {
        APASS_LOG_WARN_F(
            Elements::Function, "%s not found under install(%s/scripts), or source(%s).", kExtractPassLogScriptName,
            GetPyptoLibPath().c_str(), kExtractPassLogScriptInSource);
        return;
    }

    const std::string hostLogDir = LogManager::Instance().GetHostLogDir();
    const std::string logPattern = hostLogDir + "/pypto-log-" + std::to_string(getpid()) + "_*.log";
    const std::string functionName = function.GetMagicName();
    std::string outputDir = config::LogTopFolder();
    if (!strategy.empty()) {
        outputDir = outputDir + "/" + kComputationGraphFolder + "/" + strategy;
    }
    const std::string command = "python3 " + EscapeShellArg(scriptPath) + " " + EscapeShellArg(logPattern) + " -f " +
                                EscapeShellArg(functionName) + " -o " + EscapeShellArg(outputDir) + " --silentmode";
    int ret = std::system(command.c_str());
    if (ret != 0) {
        APASS_LOG_WARN_F(Elements::Function, "extract_pass_log.py failed(ret=%d), command: %s.", ret, command.c_str());
    }
}

PassLogUtil::PassLogUtil(Pass& pass, Function& function, size_t passIndex)
    : PassLogUtil(pass, function, "", passIndex)
{}

PassLogUtil::PassLogUtil(Pass& pass, Function& function, const std::string& strategy, size_t passIndex)
{
    originLogOutPath_ = config::LogFile();
    if (!strategy.empty()) {
        computationGraphFolder_ = config::LogTopFolder() + "/" + kComputationGraphFolder;
        strategyFolder_ = computationGraphFolder_ + "/" + strategy;
    }
    logFolder_ = pass.LogFolder(config::LogTopFolder(), strategy, passIndex);
    logFilePath_ = logFolder_ + "/" + (pass.GetName() + function.GetMagicName() + ".log");
}

PassLogUtil::~PassLogUtil()
{
    DeleteDirIfEmpty(logFolder_);
    DeleteDirIfEmpty(strategyFolder_);
    DeleteDirIfEmpty(computationGraphFolder_);
}

} // namespace npu::tile_fwk
