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
 * \file tilefwk_log.h
 * \brief
 */

#pragma once

#include <cstdint>

#define DLOG_DEBUG 0x0
#define DLOG_INFO  0x1
#define DLOG_WARN  0x2
#define DLOG_ERROR 0x3
#define DLOG_EVENT 0x10

#define PYPTO 76

#ifndef __DEVICE__
namespace npu::tile_fwk {
class TilefwkLogFuncInfo {
public:
    TilefwkLogFuncInfo();
    ~TilefwkLogFuncInfo();
    int32_t(*checkLevel)(int32_t, int32_t);
    void(*record)(int32_t, int32_t, const char *, ...);
};
inline TilefwkLogFuncInfo logFuncInfo;
}

#define INNER_PYPTO_LOG(level, module, fmt, ...)                                                                                                       \
    do {                                                                                                                                               \
        if (npu::tile_fwk::logFuncInfo.checkLevel != nullptr && npu::tile_fwk::logFuncInfo.record != nullptr) {                                        \
            if (npu::tile_fwk::logFuncInfo.checkLevel(PYPTO, level)) {                                                                                 \
                npu::tile_fwk::logFuncInfo.record(PYPTO, level, "[%s][%s:%d][%s]:" fmt, module, __FILE_NAME__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
            }                                                                                                                                          \
        }                                                                                                                                              \
    } while (0)

#define FUNCTION_LOGD(...) INNER_PYPTO_LOG(DLOG_DEBUG, "FUNCTION", __VA_ARGS__)
#define FUNCTION_LOGI(...) INNER_PYPTO_LOG(DLOG_INFO, "FUNCTION", __VA_ARGS__)
#define FUNCTION_LOGW(...) INNER_PYPTO_LOG(DLOG_WARN, "FUNCTION", __VA_ARGS__)
#define FUNCTION_LOGE(...) INNER_PYPTO_LOG(DLOG_ERROR, "FUNCTION", __VA_ARGS__)
#define FUNCTION_EVENT(...) INNER_PYPTO_LOG(DLOG_EVENT, "FUNCTION", __VA_ARGS__)

#define PASS_LOGD(...) INNER_PYPTO_LOG(DLOG_DEBUG, "PASS", __VA_ARGS__)
#define PASS_LOGI(...) INNER_PYPTO_LOG(DLOG_INFO, "PASS", __VA_ARGS__)
#define PASS_LOGW(...) INNER_PYPTO_LOG(DLOG_WARN, "PASS", __VA_ARGS__)
#define PASS_LOGE(...) INNER_PYPTO_LOG(DLOG_ERROR, "PASS", __VA_ARGS__)
#define PASS_EVENT(...) INNER_PYPTO_LOG(DLOG_EVENT, "PASS", __VA_ARGS__)

#define CODEGEN_LOGD(...) INNER_PYPTO_LOG(DLOG_DEBUG, "CODEGEN", __VA_ARGS__)
#define CODEGEN_LOGI(...) INNER_PYPTO_LOG(DLOG_INFO, "CODEGEN", __VA_ARGS__)
#define CODEGEN_LOGW(...) INNER_PYPTO_LOG(DLOG_WARN, "CODEGEN", __VA_ARGS__)
#define CODEGEN_LOGE(...) INNER_PYPTO_LOG(DLOG_ERROR, "CODEGEN", __VA_ARGS__)
#define CODEGEN_EVENT(...) INNER_PYPTO_LOG(DLOG_EVENT, "CODEGEN", __VA_ARGS__)

#define MACHINE_LOGD(...) INNER_PYPTO_LOG(DLOG_DEBUG, "MACHINE", __VA_ARGS__)
#define MACHINE_LOGI(...) INNER_PYPTO_LOG(DLOG_INFO, "MACHINE", __VA_ARGS__)
#define MACHINE_LOGW(...) INNER_PYPTO_LOG(DLOG_WARN, "MACHINE", __VA_ARGS__)
#define MACHINE_LOGE(...) INNER_PYPTO_LOG(DLOG_ERROR, "MACHINE", __VA_ARGS__)
#define MACHINE_EVENT(...) INNER_PYPTO_LOG(DLOG_EVENT, "MACHINE", __VA_ARGS__)

#define DISTRIBUTED_LOGD(...) INNER_PYPTO_LOG(DLOG_DEBUG, "DISTRIBUTED", __VA_ARGS__)
#define DISTRIBUTED_LOGI(...) INNER_PYPTO_LOG(DLOG_INFO, "DISTRIBUTED", __VA_ARGS__)
#define DISTRIBUTED_LOGW(...) INNER_PYPTO_LOG(DLOG_WARN, "DISTRIBUTED", __VA_ARGS__)
#define DISTRIBUTED_LOGE(...) INNER_PYPTO_LOG(DLOG_ERROR, "DISTRIBUTED", __VA_ARGS__)
#define DISTRIBUTED_EVENT(...) INNER_PYPTO_LOG(DLOG_EVENT, "DISTRIBUTED", __VA_ARGS__)

#define SIMULATION_LOGD(...) INNER_PYPTO_LOG(DLOG_DEBUG, "SIMULATION", __VA_ARGS__)
#define SIMULATION_LOGI(...) INNER_PYPTO_LOG(DLOG_INFO, "SIMULATION", __VA_ARGS__)
#define SIMULATION_LOGW(...) INNER_PYPTO_LOG(DLOG_WARN, "SIMULATION", __VA_ARGS__)
#define SIMULATION_LOGE(...) INNER_PYPTO_LOG(DLOG_ERROR, "SIMULATION", __VA_ARGS__)
#define SIMULATION_EVENT(...) INNER_PYPTO_LOG(DLOG_EVENT, "SIMULATION", __VA_ARGS__)

#endif