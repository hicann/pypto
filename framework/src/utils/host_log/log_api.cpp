/**
* Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log_api.cpp
 * \brief
 */

#include <map>
#include <cstdarg>

#include "tilefwk/tilefwk_log.h"
#include "host_log/log_manager.h"
#include "host_log/dlog_handler.h"

namespace npu::tile_fwk {
namespace {
const std::map<int32_t, LogLevel> LOG_LEVEL_MAP = {
    {DLOG_DEBUG, LogLevel::DEBUG},
    {DLOG_INFO, LogLevel::INFO},
    {DLOG_WARN, LogLevel::WARN},
    {DLOG_ERROR, LogLevel::ERROR},
    {DLOG_EVENT, LogLevel::EVENT}
};
LogLevel GetLogLevel(const int32_t logLevel) {
    auto iter = LOG_LEVEL_MAP.find(logLevel);
    return iter == LOG_LEVEL_MAP.end() ? LogLevel::ERROR : iter->second;
}
}

int32_t TilefwkCheckLogLevel(int32_t moduleId, int32_t logLevel) {
    (void)moduleId;
    return LogManager::Instance().CheckLevel(GetLogLevel(logLevel)) ? 1 : 0;
}

void TilefwkLogRecord(int32_t moduleId, int32_t logLevel, const char *fmt, ...) {
    (void)moduleId;
    va_list list;
    va_start(list, fmt);
    LogManager::Instance().Record(GetLogLevel(logLevel), fmt, list);
    va_end(list);
}

#ifndef __DEVICE__
TilefwkLogFuncInfo::TilefwkLogFuncInfo() {
    checkLevel = TilefwkCheckLogLevel;
    record = TilefwkLogRecord;
}

TilefwkLogFuncInfo::~TilefwkLogFuncInfo() {
    checkLevel = nullptr;
    record= nullptr;
}
#endif
}
