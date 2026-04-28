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
 * \file source_location.cpp
 * \brief
 */

#include "source_location.h"
#include "interface/utils/common.h"

#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>

namespace npu::tile_fwk {

static std::map<std::pair<std::string, uint64_t>, std::vector<uint64_t>> BuildDlMap(const std::unordered_set<uint64_t>& pcSet)
{
    Dl_info dlinfo;
    std::map<std::pair<std::string, uint64_t>, std::vector<uint64_t>> dlMap;
    for (auto pc : pcSet) {
        if (dladdr((void*)pc, &dlinfo) != 0) {
            dlMap[{dlinfo.dli_fname, (uint64_t)dlinfo.dli_fbase}].push_back(pc);
        }
    }
    return dlMap;
}

static std::vector<std::string> BuildAddr2LineArgs(
    const std::string& elfName, const std::vector<uint64_t>& pcs, uint64_t baseAddr)
{
    std::vector<std::string> args = {"addr2line", "-i", "-p", "-e", elfName};
    for (auto pc : pcs) {
        std::stringstream ss;
        ss << std::hex << (pc - (intptr_t)baseAddr);
        args.push_back(ss.str());
    }
    return args;
}

static bool ParseLineInfo(const std::string& lineStr, std::string& fname, int& lineno)
{
    if (lineStr.empty()) {
        return false;
    }
    size_t colonPos = lineStr.find(":");
    if (colonPos != std::string::npos) {
        fname = lineStr.substr(0, colonPos);
        long parsedLineno = strtol(lineStr.substr(colonPos + 1).c_str(), nullptr, 10);
        lineno = static_cast<int>(parsedLineno);
        return true;
    } else {
        fname = lineStr;
        lineno = 0;
        return true;
    }
}

static void SetFallbackLocation(std::string& fname, int& lineno, const std::string& elfName, uint64_t pc, uint64_t baseAddr)
{
    std::stringstream os;
    os << elfName << "(+" << std::hex << pc - (intptr_t)baseAddr << ")";
    fname = os.str();
    lineno = 0;
}

void SourceLocation::Init() const
{
    std::lock_guard<std::mutex> lock(mutex);
    if (pcSet.empty()) {
        return;
    }

    auto dlMap = BuildDlMap(pcSet);
    pcSet.clear();

    for (auto& [info, pcs] : dlMap) {
        auto args = BuildAddr2LineArgs(info.first, pcs, info.second);
        std::string output = SafeExecCommandWithOutput(args);
        std::istringstream iss(output);
        std::string lineStr;

        for (auto pc : pcs) {
            if (!std::getline(iss, lineStr)) {
                SetFallbackLocation(locMap[pc]->fname_, locMap[pc]->lineno_, info.first, pc, info.second);
                continue;
            }

            if (lineStr.find("inlined by") != std::string::npos && !std::getline(iss, lineStr)) {
                lineStr.clear();
            }

            if (!ParseLineInfo(lineStr, locMap[pc]->fname_, locMap[pc]->lineno_)) {
                SetFallbackLocation(locMap[pc]->fname_, locMap[pc]->lineno_, info.first, pc, info.second);
            }
        }
    }
}

int SourceLocation::GetLineno() const
{
    Init();
    return lineno_;
}

const std::string& SourceLocation::GetFileName() const
{
    Init();
    return fname_;
}

const std::string& SourceLocation::GetBacktrace() const { return backtrace_; }

bool SourceLocation::isCppMode_ = false;
std::mutex SourceLocation::mutex;
std::stack<std::shared_ptr<SourceLocation>> SourceLocation::callStack;
std::unordered_set<uint64_t> SourceLocation::pcSet;
std::unordered_map<uint64_t, std::shared_ptr<SourceLocation>> SourceLocation::locMap;
} // namespace npu::tile_fwk
