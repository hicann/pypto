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
 * \file error.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <exception>
#include <sstream>
#include <string>
#include <memory>
#include <cstring>
#include <cassert>

#include "lazy.h"

namespace npu::tile_fwk {
using Backtrace = std::shared_ptr<LazyValue<std::string>>;

Backtrace GetBacktrace(size_t skipFrames, size_t maxFrames);

struct ErrorMessage {
public:
    std::string Message() { return ss.str(); }

    template <typename T>
    ErrorMessage &operator<<(const T &value) {
        ss << value;
        return *this;
    }

    // support std::endl etc.
    ErrorMessage &operator<<(std::ostream &(*manipulator)(std::ostream &)) {
        ss << manipulator;
        return *this;
    }

    std::stringstream ss;
};

class Error : public std::exception {
public:
    Error(const char *func, const char *file, size_t line, const std::string &msg, Backtrace backtrace)
        : func_(func), file_(file), line_(line), msg_(msg), backtrace_(backtrace) {}

    Error(const char *func, const char *file, size_t line, Backtrace backtrace)
        : func_(func), file_(file), line_(line), backtrace_(backtrace) {}

    const char *what() const noexcept override;

    int operator=(ErrorMessage &msg) {
        msg_ = msg.Message();
        /* avoid nested throw */
        if (std::uncaught_exceptions() == 0) {
            throw *this;
        }
        return 0;
    }

private:
    const char *func_;
    const char *file_;
    size_t line_;
    std::string msg_;
    std::string umsg_;
    Backtrace backtrace_;
    mutable LazyShared<std::string> what_;
};

class AssertInfo {
public:
    [[noreturn]] int operator=(ErrorMessage &msg) {
        (void)fprintf(stderr, "%s\n", msg.Message().c_str());
        abort();
    }
};

#ifndef __DEVICE__
/* used for internal check */
#define ASSERT(cond)                                                                                                           \
   (cond) ? 0 : npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) = \
        npu::tile_fwk::ErrorMessage() << "ASSERTION FAILED: " #cond << "\n"

/* check for user input */
#define CHECK(cond)                                                                                                           \
   (cond) ? 0 : npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) = \
        npu::tile_fwk::ErrorMessage() << "CHECK FAILED: " #cond << "\n"

#define TILEFWK_ERROR()                                                                                            \
    npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) = \
        npu::tile_fwk::ErrorMessage()
#else

#define ASSERT(cond)                             \
    (cond) ? 0 : AssertInfo() = npu::tile_fwk::ErrorMessage() \
        << "ASSERTION FAILED: " #cond " file " << __FILE__ << ", line " << __LINE__ << "\n"

#define CHECK(cond)                             \
    (cond) ? 0 : AssertInfo() = npu::tile_fwk::ErrorMessage() \
        << "CHECK FAILED: " #cond " file " << __FILE__ << ", line " << __LINE__ << "\n"
#endif
} // namespace npu::tile_fwk
