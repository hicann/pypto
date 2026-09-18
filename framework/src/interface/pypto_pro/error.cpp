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
 * \file error.cpp
 * \brief First-line rendering and the ambient DSL location.
 */

#include "pypto_pro/error.h"

#include <iomanip>
#include <sstream>

namespace pypto {
namespace pro {
namespace {

/**
 * \brief The DSL location in effect on this thread.
 *
 * thread_local because compilation may run on several threads and a span names
 * one kernel statement.
 */
thread_local ir::Span g_currentSpan;

constexpr unsigned CODE_MASK = 0xFFFFFU;
constexpr int CODE_WIDTH = 5;

} // namespace

const ir::Span& CurrentSpan() { return g_currentSpan; }

SpanScope::SpanScope(const ir::Span& span) : prev_(g_currentSpan)
{
    // An invalid span would erase the enclosing statement's location: synthesised
    // nodes have no source of their own.
    if (!span.IsUnknown() && span.IsValid()) {
        g_currentSpan = span;
    }
}

SpanScope::~SpanScope() { g_currentSpan = prev_; }

std::string ErrHead(const char* file, int line, const char* module, unsigned code, const char* codeName)
{
    std::ostringstream oss;
    oss << "[" << file << ":" << line << "][" << module << "]:"
        << "ErrCode: F" << std::uppercase << std::hex << std::setw(CODE_WIDTH) << std::setfill('0')
        << (code & CODE_MASK) << std::dec << "! Enum: " << codeName << ". ";
    return oss.str();
}

std::string LocLine(const ir::Span& span)
{
    if (span.IsUnknown() || !span.IsValid()) {
        return "";
    }
    std::ostringstream oss;
    oss << "\n  --> " << span.Filename() << ":" << span.BeginLine() << ":" << span.BeginColumn();
    return oss.str();
}

} // namespace pro
} // namespace pypto
