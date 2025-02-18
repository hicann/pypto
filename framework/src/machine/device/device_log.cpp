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
 * \file device_log.cpp
 * \brief
 */

#include "machine/utils/device_log.h"
namespace npu::tile_fwk {

bool g_isLogEnableDebug = false;
bool g_isLogEnableInfo = false;
bool g_isLogEnableWarn = false;
bool g_isLogEnableError = false;

void InitLogSwitch() {
#if DEBUG_PLOG && defined(__DEVICE__)
    g_isLogEnableDebug = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_isLogEnableInfo = CheckLogLevel(AICPU, DLOG_INFO);
    g_isLogEnableWarn = CheckLogLevel(AICPU, DLOG_WARN);
    g_isLogEnableError = CheckLogLevel(AICPU, DLOG_ERROR);
#endif
}

static std::string logFilePrefix;

void SetLogFilePrefix(const std::string &prefix) {
    logFilePrefix = prefix;
}

// 创建日志文件
void CreateLogFile(LogType type, int threadIdx) {
    (void)type;
    (void)threadIdx;
#if ENABLE_TMP_LOG || !defined(__DEVICE__)
    char logfile[256];
    errno_t rc = EOK;
    switch (type) {
        case LogType::LOG_TYPE_SCHEDULER:
            rc = sprintf_s(logfile, sizeof(logfile), "/tmp/pypto_%saicpu_sch%d.txt", logFilePrefix.c_str(), threadIdx);
            break;
        case LogType::LOG_TYPE_CONTROLLER:
            rc = sprintf_s(logfile, sizeof(logfile), "/tmp/pypto_%saicpu_ctrl.txt", logFilePrefix.c_str());
            break;
        case LogType::LOG_TYPE_PREFETCH:
            rc = sprintf_s(logfile, sizeof(logfile), "/tmp/pypto_%saicpu_prefetch.txt", logFilePrefix.c_str());
            break;
        default:
            return;
    }
    (void)(rc == EOK);
    GetLogger(logfile);
#endif
}

} // namespace npu::tile_fwk