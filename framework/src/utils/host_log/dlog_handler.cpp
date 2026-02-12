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
 * \file dlog_handler.cpp
 * \brief
 */

#include "host_log/dlog_handler.h"
#include <dlfcn.h>
#include <iostream>
#include <ostream>
#include <string>

namespace npu::tile_fwk {
namespace {
bool GetEnvHomePath(std::string &homePathStr) {
    const char *homePath =  std::getenv("ASCEND_HOME_PATH");
    if (homePath == nullptr) {
        return false;
    }
    homePathStr = std::string(homePath);
    return true;
}
}
DLogHandler &DLogHandler::Instance() {
    static DLogHandler instance;
    return instance;
}

DLogHandler::DLogHandler() {
    std::string homePath;
    if (!GetEnvHomePath(homePath)) {
        return;
    }
    std::string dLogLibPath = homePath + "/lib64/libunified_dlog.so";
    handle_ = dlopen(dLogLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle_ == nullptr) {
        std::cerr << "Fail to dlopen " << dLogLibPath << ", error:" << dlerror() << std::endl;
        return;
    }
    checkLevelFunc_ = reinterpret_cast<int32_t(*)(int32_t, int32_t)>(dlsym(handle_, "CheckLogLevel"));
    if (checkLevelFunc_ == nullptr) {
        std::cerr << "Fail to dlsym CheckLogLevel function from " << dLogLibPath << std::endl;
        CloseHandle();
        return;
    }
    logRecordFunc_ = reinterpret_cast<void(*)(int32_t, int32_t, const char *, ...)>(dlsym(handle_, "DlogRecord"));
    if (logRecordFunc_ == nullptr) {
        std::cerr << "Fail to dlsym DlogRecord function from " << dLogLibPath << std::endl;
        CloseHandle();
        return;
    }
}

DLogHandler::~DLogHandler() {
    CloseHandle();
}

void DLogHandler::CloseHandle() {
    if (handle_ == nullptr) {
        return;
    }
    if (dlclose(handle_) != 0) {
        std::cerr << "Fail to close dlog library, error:" << dlerror() << std::endl;
    }
    handle_ = nullptr;
    checkLevelFunc_ = nullptr;
    logRecordFunc_ = nullptr;
}
}
