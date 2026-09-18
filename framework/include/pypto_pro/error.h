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
 * \file error.h
 * \brief pypto_pro error macros: spec first line plus the user's DSL location.
 *
 * They differ from core/logging.h's IRCHECK / INTERNAL_CHECK in two ways: they
 * render the mandatory first line (error code and module) and append the DSL
 * location. The exception type each site throws is unchanged.
 *
 * \code
 *   PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
 *       << "block.load requires 3 args (out_tile, tensor, offsets)";
 * \endcode
 */

#ifndef PYPTO_PRO_ERROR_H_
#define PYPTO_PRO_ERROR_H_

#include <sstream>
#include <string>
#include <utility>

#include "core/error.h"
#include "ir/span.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace pypto {
namespace pro {

/**
 * \brief The DSL location in effect, or Span::Unknown() when none was published.
 *
 * Separate from ir::Span::Current(), which pypto owns and which is not
 * thread-local.
 */
const ir::Span& CurrentSpan();

/**
 * \brief Publish the DSL location for a scope, restoring the previous one.
 *
 * Published at two gates: OpRegistry::Create and CCECodegen's VisitStmt /
 * VisitExpr. An invalid span is a no-op, so synthesised nodes -- which have no
 * source of their own -- cannot erase the enclosing statement's location.
 */
class SpanScope {
public:
    explicit SpanScope(const ir::Span& span);
    ~SpanScope();

    SpanScope(const SpanScope&) = delete;
    SpanScope& operator=(const SpanScope&) = delete;
    SpanScope(SpanScope&&) = delete;
    SpanScope& operator=(SpanScope&&) = delete;

private:
    ir::Span prev_;
};

/**
 * \brief The first-line prefix: [file:line][MODULE]:ErrCode: FXXXXX! Enum: NAME.
 *
 * \a file is passed \c __FILE__ directly; the build already redefines it to the
 * translation unit's basename.
 */
std::string ErrHead(const char* file, int line, const char* module, unsigned code, const char* codeName);

/**
 * \brief The location line "\n  --> <dsl_file>:<line>:<col>", empty for an
 * invalid span. The source preview around it is rendered on the Python side.
 */
std::string LocLine(const ir::Span& span);

/**
 * \brief How to throw. Specialisations keep each site's exception type and
 * behaviour unchanged.
 */
template <typename ExceptionType>
struct Thrower {
    [[noreturn]] static void Throw(const char*, const char*, int, const std::string& text)
    {
        throw ExceptionType(text);
    }
};

/**
 * \brief npu::tile_fwk::Error keeps its operator=, so backtrace, slog and
 * ErrorManager behave as before.
 */
template <>
struct Thrower<npu::tile_fwk::Error> {
    [[noreturn]] static void Throw(const char* func, const char* file, int line, const std::string& text)
    {
        npu::tile_fwk::ErrorMessage message;
        message << text;
        npu::tile_fwk::Error error(func, file, static_cast<size_t>(line),
                                   npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64));
        error = message;
        // operator= throws only when std::uncaught_exceptions() == 0; this fallback
        // is unreachable in practice and exists to satisfy [[noreturn]].
        throw npu::tile_fwk::Error(func, file, static_cast<size_t>(line), text, nullptr);
    }
};

/**
 * \brief Collects the streamed message and throws it on destruction, ordered as
 * first line, the caller's message, then the location line.
 */
template <typename ExceptionType>
class ErrLogger {
public:
    ErrLogger(const char* func, const char* file, int line, const char* module, unsigned code, const char* codeName,
              const ir::Span& span)
        : func_(func), file_(file), line_(line), head_(ErrHead(file, line, module, code, codeName)), loc_(LocLine(span))
    {}

    [[noreturn]] ~ErrLogger() noexcept(false)
    {
        Thrower<ExceptionType>::Throw(func_, file_, line_, head_ + ss_.str() + loc_);
    }

    template <typename T>
    ErrLogger& operator<<(T&& value)
    {
        ss_ << std::forward<T>(value);
        return *this;
    }

    ErrLogger(const ErrLogger&) = delete;
    ErrLogger& operator=(const ErrLogger&) = delete;
    ErrLogger(ErrLogger&&) = delete;
    ErrLogger& operator=(ErrLogger&&) = delete;

private:
    std::stringstream ss_;
    const char* func_;
    const char* file_;
    int line_;
    std::string head_;
    std::string loc_;
};

} // namespace pro
} // namespace pypto

/*
 * Error macros are named `PRO_<MODULE>_<MECHANISM>`.
 *
 * The module is part of the macro name, as in pypto's PASS_LOGE / CODEGEN_LOGE:
 * picking the macro picks the module. This is what lets a check inside a header
 * report its own module rather than that of the including .cpp. pypto_pro needs
 * the tag because its external codes are all F0XXXX and carry no stage.
 *
 * The mechanism selects the exception type, one per existing macro:
 *   _CHECK           -> npu::tile_fwk::Error      (= CHECK)
 *   _IRCHECK         -> pypto::ir::ValueError     (= IRCHECK)
 *   _INTERNAL_CHECK  -> pypto::ir::InternalError  (= INTERNAL_CHECK)
 *   _THROW           -> named at the call site    (= throw exc(...))
 *
 * The location defaults to the ambient span. Use _WITH_SPAN only when the call
 * chain does not pass a gate, or to point at a subexpression rather than the
 * whole statement.
 */

/** \brief Builds an ErrLogger. Call sites use the PRO_* macros below. */
#define PRO_ERR_(exc, module, errcode, span) \
    ::pypto::pro::ErrLogger<exc>(__func__, __FILE__, __LINE__, module, static_cast<unsigned>(errcode), #errcode, span)

/**
 * \brief Shared shape of a conditional check.
 *
 * `if (!!(cond)) ; else` rather than `if (!(cond))`, which would swallow an
 * else at the call site.
 */
#define PRO_CHECK_IMPL_(exc, module, errcode, cond, span) \
    if (!!(cond))                                         \
        ;                                                 \
    else                                                  \
        PRO_ERR_(exc, module, errcode, span)

#define PRO_CHECK_M_(module, errcode, cond) \
    PRO_CHECK_IMPL_(::npu::tile_fwk::Error, module, errcode, cond, ::pypto::pro::CurrentSpan())
#define PRO_IRCHECK_M_(module, errcode, cond) \
    PRO_CHECK_IMPL_(::pypto::ir::ValueError, module, errcode, cond, ::pypto::pro::CurrentSpan())
#define PRO_INTERNAL_CHECK_M_(module, errcode, cond) \
    PRO_CHECK_IMPL_(::pypto::ir::InternalError, module, errcode, cond, ::pypto::pro::CurrentSpan())
#define PRO_INTERNAL_CHECK_SPAN_M_(module, errcode, cond, span) \
    PRO_CHECK_IMPL_(::pypto::ir::InternalError, module, errcode, cond, span)
#define PRO_THROW_M_(exc, module, errcode) PRO_ERR_(exc, module, errcode, ::pypto::pro::CurrentSpan())

// PRO_IR: IR construction and operator definition.
#define PRO_IR_CHECK(errcode, cond) PRO_CHECK_M_("PRO_IR", errcode, cond)
#define PRO_IR_IRCHECK(errcode, cond) PRO_IRCHECK_M_("PRO_IR", errcode, cond)
#define PRO_IR_INTERNAL_CHECK(errcode, cond) PRO_INTERNAL_CHECK_M_("PRO_IR", errcode, cond)
#define PRO_IR_INTERNAL_CHECK_WITH_SPAN(errcode, cond, span) PRO_INTERNAL_CHECK_SPAN_M_("PRO_IR", errcode, cond, span)
#define PRO_IR_THROW(exc, errcode) PRO_THROW_M_(exc, "PRO_IR", errcode)

// PRO_CODEGEN: code generation, including operator emit and VF.
#define PRO_CODEGEN_CHECK(errcode, cond) PRO_CHECK_M_("PRO_CODEGEN", errcode, cond)
#define PRO_CODEGEN_IRCHECK(errcode, cond) PRO_IRCHECK_M_("PRO_CODEGEN", errcode, cond)
#define PRO_CODEGEN_INTERNAL_CHECK(errcode, cond) PRO_INTERNAL_CHECK_M_("PRO_CODEGEN", errcode, cond)
#define PRO_CODEGEN_INTERNAL_CHECK_WITH_SPAN(errcode, cond, span) \
    PRO_INTERNAL_CHECK_SPAN_M_("PRO_CODEGEN", errcode, cond, span)
#define PRO_CODEGEN_THROW(exc, errcode) PRO_THROW_M_(exc, "PRO_CODEGEN", errcode)

// PRO_PASS: IR transforms. Internal errors only -- a pass consumes the previous
// stage's output, not what the user wrote, so its failures are framework bugs.
#define PRO_PASS_INTERNAL_CHECK(errcode, cond) PRO_INTERNAL_CHECK_M_("PRO_PASS", errcode, cond)
#define PRO_PASS_INTERNAL_CHECK_WITH_SPAN(errcode, cond, span) \
    PRO_INTERNAL_CHECK_SPAN_M_("PRO_PASS", errcode, cond, span)

#endif // PYPTO_PRO_ERROR_H_
