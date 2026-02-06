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
 * \file pass_log_util.cpp
 * \brief Implementation of PassLogUtil for redirecting pass logs to dedicated files
 */

#include "pass_log_util.h"

#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {

PassLogUtil::PassLogUtil(Pass &pass, Function &function, size_t passIndex) {
    originLogOutPath_ = config::LogFile();
    logFolder_ = pass.LogFolder(config::LogTopFolder(), passIndex);
    logFilePath_ = logFolder_ + "/" + (pass.GetName() + function.GetMagicName() + ".log");
    LoggerManager::FileLoggerReplace(originLogOutPath_, logFilePath_, true);
}

PassLogUtil::~PassLogUtil() {
    LoggerManager::FileLoggerReplace(logFilePath_, originLogOutPath_, true);
    if (!logFolder_.empty()) {
        auto files = GetFiles(logFolder_, "");
        if (files.empty()) {
            (void)DeleteDir(logFolder_);
        }
    }
}

} // namespace npu::tile_fwk