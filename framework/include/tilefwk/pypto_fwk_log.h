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
 * \file pypto_fwk_log.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#define DLOG_DEBUG 0x0
#define DLOG_INFO  0x1
#define DLOG_WARN  0x2
#define DLOG_ERROR 0x3

#define PYPTO 59

#ifndef __DEVICE__
#ifndef __FILE_NAME__
#define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

namespace npu::tile_fwk {
enum class LogModule {
    FUNCTION = 0,
    PASS,
    CODEGEN,
    MACHINE,
    DISTRIBUTED,
    SIMULATION,
    VERIFY,
    COMPILER_MONITOR,
    BOTTOM
};

class LogFuncInfo {
public:
    static LogFuncInfo &Instance();
    int32_t (*checkLevel)(int32_t, int32_t, LogModule);
    void (*record)(int32_t, int32_t, const char *, ...) __attribute__((format(printf, 3, 4)));
    void (*pyptoRecord)(int32_t, int32_t, const char *, ...) __attribute__((format(printf, 3, 4)));
    void (*setAttr)(bool);
private:
    LogFuncInfo();
    ~LogFuncInfo();
};
}

#define PYPTO_HOST_LOG(level, module, fmt, ...)                                                                                              \
    do {                                                                                                                                     \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                     \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                                           \
        }                                                                                                                                    \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr && npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {      \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::module)) {                         \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__,                     \
                                                              #module, ##__VA_ARGS__);                                                       \
            }                                                                                                                                \
        }                                                                                                                                    \
    } while (0)

#define MAX_LOG_LENGTH 880

#define PYPTO_HOST_SPLIT_LOG(level, module, fmt, ...)                                                                                        \
    do {                                                                                                                                     \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                     \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                                           \
        }                                                                                                                                    \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel == nullptr || npu::tile_fwk::LogFuncInfo::Instance().record == nullptr) {      \
            break;                                                                                                                           \
        }                                                                                                                                    \
        if  (!npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::module)) {                           \
            break;                                                                                                                           \
        }                                                                                                                                    \
        char *formatStr = nullptr;                                                                                                           \
        int len = asprintf(&formatStr, fmt, ##__VA_ARGS__);                                                                                  \
        if (len <= 0 || formatStr == nullptr) {                                                                                              \
            break;                                                                                                                           \
        }                                                                                                                                    \
        if (len < MAX_LOG_LENGTH) {                                                                                                          \
            npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:%s", __FILE_NAME__, __LINE__, #module, formatStr);      \
        } else {                                                                                                                             \
            char *msgBegin = formatStr;                                                                                                      \
            char *msgEnd = formatStr + len;                                                                                                  \
            while (msgBegin < msgEnd) {                                                                                                      \
                npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:%.880s", __FILE_NAME__, __LINE__,                   \
                                                              #module, msgBegin);                                                            \
                msgBegin += MAX_LOG_LENGTH;                                                                                                  \
            }                                                                                                                                \
        }                                                                                                                                    \
        free(formatStr);                                                                                                                     \
    } while (0)

#define PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(level, module, fmt, ...)                                                                          \
    do {                                                                                                                                     \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                     \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(false);                                                                           \
        }                                                                                                                                    \
        if (npu::tile_fwk::LogFuncInfo::Instance().record != nullptr) {                                                                      \
            npu::tile_fwk::LogFuncInfo::Instance().record(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__, #module, ##__VA_ARGS__);\
        }                                                                                                                                    \
    } while (0)

#define PYPTO_SIM_LOG(level, module, fmt, ...)                                                                                               \
    do {                                                                                                                                     \
        if (npu::tile_fwk::LogFuncInfo::Instance().setAttr != nullptr) {                                                                     \
            npu::tile_fwk::LogFuncInfo::Instance().setAttr(true);                                                                            \
        }                                                                                                                                    \
        if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel != nullptr && npu::tile_fwk::LogFuncInfo::Instance().pyptoRecord != nullptr) { \
            if (npu::tile_fwk::LogFuncInfo::Instance().checkLevel(PYPTO, level, npu::tile_fwk::LogModule::module)) {                         \
                npu::tile_fwk::LogFuncInfo::Instance().pyptoRecord(PYPTO, level, "[%s:%d][%s]:" fmt, __FILE_NAME__, __LINE__,                \
                                                                   #module, ##__VA_ARGS__);                                                  \
            }                                                                                                                                \
        }                                                                                                                                    \
    } while (0)

#define FUNCTION_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, FUNCTION, __VA_ARGS__)
#define FUNCTION_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, FUNCTION, __VA_ARGS__)
#define FUNCTION_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, FUNCTION, __VA_ARGS__)

#define PASS_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, PASS, __VA_ARGS__)
#define PASS_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, PASS, __VA_ARGS__)
#define PASS_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, PASS, __VA_ARGS__)
#define PASS_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, PASS, __VA_ARGS__)
#define PASS_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, PASS, __VA_ARGS__)
#define PASS_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, PASS, __VA_ARGS__)

#define CODEGEN_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, CODEGEN, __VA_ARGS__)
#define CODEGEN_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, CODEGEN, __VA_ARGS__)
#define CODEGEN_LOGI_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_INFO, CODEGEN, __VA_ARGS__)

#define MACHINE_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, MACHINE, __VA_ARGS__)
#define MACHINE_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, MACHINE, __VA_ARGS__)
#define MACHINE_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, MACHINE, __VA_ARGS__)
#define MACHINE_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, MACHINE, __VA_ARGS__)
#define MACHINE_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, MACHINE, __VA_ARGS__)
#define MACHINE_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, MACHINE, __VA_ARGS__)

#define DISTRIBUTED_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, DISTRIBUTED, __VA_ARGS__)
#define DISTRIBUTED_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, DISTRIBUTED, __VA_ARGS__)

#define SIMULATION_LOGD(...) PYPTO_SIM_LOG(DLOG_DEBUG, SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGI(...) PYPTO_SIM_LOG(DLOG_INFO, SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGW(...) PYPTO_SIM_LOG(DLOG_WARN, SIMULATION, __VA_ARGS__)
#define SIMULATION_LOGE(...) PYPTO_SIM_LOG(DLOG_ERROR, SIMULATION, __VA_ARGS__)

#define VERIFY_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, VERIFY, __VA_ARGS__)
#define VERIFY_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, VERIFY, __VA_ARGS__)
#define VERIFY_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, VERIFY, __VA_ARGS__)
#define VERIFY_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, VERIFY, __VA_ARGS__)
#define VERIFY_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, VERIFY, __VA_ARGS__)
#define VERIFY_LOGE_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_ERROR, VERIFY, __VA_ARGS__)

#define COMPILER_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_EVENT(...) PYPTO_HOST_LOG_WITHOUT_LEVEL_CHECK(DLOG_INFO, COMPILER_MONITOR, __VA_ARGS__)
#define COMPILER_LOGD_FULL(...) PYPTO_HOST_SPLIT_LOG(DLOG_DEBUG, COMPILER_MONITOR, __VA_ARGS__)

#endif