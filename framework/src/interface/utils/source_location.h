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
 * \file source_location.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <stack>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace npu::tile_fwk {

class SourceLocation {
public:
    SourceLocation() = default;
    SourceLocation(const std::string &fname, int lineno) : fname_(fname), lineno_(lineno) {}
    SourceLocation(const std::string &fname, int lineno, const std::string &backtrace) : fname_(fname), lineno_(lineno), backtrace_(backtrace){}
    explicit SourceLocation(uint64_t pc) : fname_("??"), lineno_(-1), pc_(pc){};

    int GetLineno() const;
    const std::string& GetFileName() const;
    const std::string& GetBacktrace() const;
    std::string ToString() const { return GetFileName() + ":" + std::to_string(GetLineno()); }

public:
    static void SetLocation(const void *pc) { callStack.push(GetLocation(reinterpret_cast<uint64_t>(pc))); }
    static void SetLocation(std::shared_ptr<SourceLocation> loc) { callStack.push(loc); }
    static void SetLocation(const std::string &fname, int lineno) { callStack.push(std::make_shared<SourceLocation>(fname, lineno)); }
    static void SetLocation(const std::string &fname, int lineno, const std::string &backtrace) { callStack.push(std::make_shared<SourceLocation>(fname, lineno, backtrace)); }
    static void ClearLocation() { callStack.pop(); }
    static auto GetLocation() { return callStack.size() > 0 ? callStack.top() : nullptr; }

    static void SetCppMode(bool val) { isCppMode_ = val; }
    static bool IsCppMode() { return isCppMode_; }

private:
    static std::shared_ptr<SourceLocation> GetLocation(uint64_t pc) {
        std::lock_guard<std::mutex> lock(mutex);
        if (locMap.find(pc) != locMap.end()) {
            return locMap[pc];
        }
        auto loc = std::make_shared<SourceLocation>(pc);
        locMap[pc] = loc;
        pcSet.insert(pc);
        return loc;
    }

    void Init() const;

private:
    mutable std::string fname_;
    mutable int lineno_;
    std::string backtrace_;
    uint64_t pc_;
    static bool isCppMode_;
    static std::mutex mutex;
    static std::stack<std::shared_ptr<SourceLocation>> callStack;
    static std::unordered_set<uint64_t> pcSet;
    static std::unordered_map<uint64_t, std::shared_ptr<SourceLocation>> locMap;
};

using SourceLocationPtr = std::shared_ptr<SourceLocation>;

struct SourceLocationHelper {
    // lr is return address, we need find caller address, minus 4 here
    SourceLocationHelper(const void *lr) {
        if (SourceLocation::IsCppMode()) {
            SourceLocation::SetLocation(static_cast<const uint8_t *>(lr) - 4);
        }
    }
    ~SourceLocationHelper() {
        if (SourceLocation::IsCppMode()) {
            SourceLocation::ClearLocation();
        }
    }
};

#define DEFINE_SOURCE_LOCATION() auto __loc = SourceLocationHelper(__builtin_return_address(0))

} // namespace npu::tile_fwk
