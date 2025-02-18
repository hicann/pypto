/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <sys/syscall.h>
#include <unistd.h>
#include <securec.h>
#include <cstdint>
#include <string>
#include "dlog_pub.h"

const std::string TILE_FWK_MODULE_NAME = "TILEFWK";

inline uint64_t TileFwkGetTid() {
    thread_local static uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

#define D_TF_LOGD(MOD_NAME, fmt, ...) dlog_debug(OP, "%lu %s:" fmt "\n", TileFwkGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_TF_LOGI(MOD_NAME, fmt, ...) dlog_info(OP, "%lu %s:" fmt "\n", TileFwkGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_TF_LOGW(MOD_NAME, fmt, ...) dlog_warn(OP, "%lu %s:" fmt "\n", TileFwkGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_TF_LOGE(MOD_NAME, fmt, ...) dlog_error(OP, "%lu %s:" fmt "\n", TileFwkGetTid(), __FUNCTION__, ##__VA_ARGS__)

#define TILE_FWK_LOGD(...) D_TF_LOGD(TILE_FWK_MODULE_NAME, __VA_ARGS__)
#define TILE_FWK_LOGI(...) D_TF_LOGI(TILE_FWK_MODULE_NAME, __VA_ARGS__)
#define TILE_FWK_LOGW(...) D_TF_LOGW(TILE_FWK_MODULE_NAME, __VA_ARGS__)
#define TILE_FWK_LOGE(...) D_TF_LOGE(TILE_FWK_MODULE_NAME, __VA_ARGS__)
