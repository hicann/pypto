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
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

#include "lazy.h"

#ifndef ERROR_CODE_UNDEFINED
#define ERROR_CODE_UNDEFINED 0xFFFFFU
#endif

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

    template <typename T>
    ErrorMessage &operator<<(const std::vector<T> &vec) {
        ss << "[";
        for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
            if (iter != vec.begin()) {
                ss << ", ";
            }
            ss << *iter;
        }
        ss << "]";
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

    Error(const char *func, const char *file, size_t line, Backtrace backtrace = nullptr)
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
#define ASSERT_WITH_CODE(errcode, cond)                                                                                     \
    (cond) ?                                                                                                                \
        0 :                                                                                                                 \
        npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) =      \
            npu::tile_fwk::ErrorMessage()                                                                                   \
            << "Errcode: F"                                                                                                 \
            << std::uppercase << std::hex << std::setw(5) << std::setfill('0')                                             \
            << (static_cast<unsigned>(errcode) & 0xFFFFF) << std::dec << "!\n"

#define CHECK_WITH_CODE(errcode, cond)                                                                                      \
    (cond) ?                                                                                                                \
        0 :                                                                                                                 \
        npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) =      \
            npu::tile_fwk::ErrorMessage()                                                                                   \
            << "Errcode: F"                                                                                                 \
            << std::uppercase << std::hex << std::setw(5) << std::setfill('0')                                             \
            << (static_cast<unsigned>(errcode) & 0xFFFFF) << std::dec << "!\n"

#define TILEFWK_ERROR()                                                                                            \
    npu::tile_fwk::Error(__func__, __FILE__, __LINE__, npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) = \
        npu::tile_fwk::ErrorMessage()
#else
#define ASSERT_WITH_CODE(errcode, cond)                                                                                     \
    (cond) ? 0 :                                                                                                            \
             AssertInfo() = npu::tile_fwk::ErrorMessage()                                                                   \
             << "Errcode: F" << std::uppercase << std::hex << std::setw(5) << std::setfill('0')                            \
             << (static_cast<unsigned>(errcode) & 0xFFFFF) << std::dec << "!\n"

#define CHECK_WITH_CODE(errcode, cond)                                                                                      \
    (cond) ? 0 :                                                                                                            \
             AssertInfo() = npu::tile_fwk::ErrorMessage()                                                                   \
             << "Errcode: F" << std::uppercase << std::hex << std::setw(5) << std::setfill('0')                            \
             << (static_cast<unsigned>(errcode) & 0xFFFFF) << std::dec << "!\n"
#endif

#define ASSERT_OVERLOAD_SELECT(_1, _2, NAME, ...) NAME
#define ASSERT_WITHOUT_ERR_CODE(cond)          ASSERT_WITH_CODE(ERROR_CODE_UNDEFINED, cond)
#define ASSERT_WITH_ERR_CODE(errcode, cond)    ASSERT_WITH_CODE(errcode, cond)
#define ASSERT(...) ASSERT_OVERLOAD_SELECT(__VA_ARGS__, ASSERT_WITH_ERR_CODE, ASSERT_WITHOUT_ERR_CODE)(__VA_ARGS__)

#define CHECK_OVERLOAD_SELECT(_1, _2, NAME, ...) NAME
#define CHECK_WITHOUT_ERR_CODE(cond)           CHECK_WITH_CODE(ERROR_CODE_UNDEFINED, cond)
#define CHECK_WITH_ERR_CODE(errcode, cond)     CHECK_WITH_CODE(errcode, cond)
#define CHECK(...) CHECK_OVERLOAD_SELECT(__VA_ARGS__, CHECK_WITH_ERR_CODE, CHECK_WITHOUT_ERR_CODE)(__VA_ARGS__)

} // namespace npu::tile_fwk
