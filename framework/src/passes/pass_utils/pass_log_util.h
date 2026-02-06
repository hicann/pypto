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
 * \file pass_log_util.h
 * \brief RAII utility class for managing pass-specific log files
 */

#pragma once

#include <string>
#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
/**
 * @brief RAII utility for managing pass-specific log file redirection
 * 
 * Redirects log output to a pass-specific file during construction and
 * restores the original log output on destruction. Automatically cleans
 * up empty log directories.
 */
class PassLogUtil {
public:
    PassLogUtil(Pass &pass, Function &function, size_t passIndex);
    ~PassLogUtil();
    
    PassLogUtil(const PassLogUtil &) = delete;
    PassLogUtil &operator=(const PassLogUtil &) = delete;

private:
    std::string originLogOutPath_;
    std::string logFilePath_;
    std::string logFolder_;
};

} // namespace npu::tile_fwk
