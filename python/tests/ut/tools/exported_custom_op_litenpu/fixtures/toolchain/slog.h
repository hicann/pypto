/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Minimal mock of CANN ``toolchain/slog.h`` for codegen compile tests.
//
// The real header (in the CANN install) declares the slog backend
// (``CheckLogLevel`` / ``DlogRecord``) and the ``dlog_debug(moduleId, fmt, ...)`` /
// ``dlog_error(moduleId, fmt, ...)`` macros. For compile-only tests we just need those to exist and to
// type-check the caller's format string + args, so we map them to ``printf`` (the
// module id is discarded). Real builds use the install header, which routes to the
// Ascend slog sink instead.
#ifndef PYPTO_MOCK_TOOLCHAIN_SLOG_H
#define PYPTO_MOCK_TOOLCHAIN_SLOG_H

#include <cstdio>
#include "base/log_types.h"

#ifndef DLOG_DEBUG
#define DLOG_DEBUG 1
#endif

#ifndef DLOG_ERROR
#define DLOG_ERROR 0x3
#endif

// Discard the module id; still type-check fmt/args by forwarding to printf.
#define dlog_debug(moduleId, fmt, ...) ((void)(moduleId), printf(fmt, ##__VA_ARGS__))
#define dlog_error(moduleId, fmt, ...) ((void)(moduleId), printf(fmt, ##__VA_ARGS__))

#endif // PYPTO_MOCK_TOOLCHAIN_SLOG_H
