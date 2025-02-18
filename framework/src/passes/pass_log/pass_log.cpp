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
 * \file pass_log.cpp
 * \brief
 */

#include <iostream>
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

    // 入参为Operation对象
    std::string GetFormatBacktrace(const Operation& op) {
        auto location = op.GetLocation();
        if (!location) {
            return "";
        }
        
        std::ostringstream oss;
        oss << "[FuncMagic:" << op.BelongTo()->GetFuncMagic() << "]" << "[OpMagic:" << op.opmagic << "]" << "[Backtrace]:" << location->SourceLocation::GetBacktrace() << ".";
        return oss.str();
    }

    // 入参为智能指针
    std::string GetFormatBacktrace(const OperationPtr& op) {
        if (!op) {
            return "";
        }
        return GetFormatBacktrace(*op);
    }

    // 入参为普通指针
    std::string GetFormatBacktrace(const Operation* op) {
        if (!op) {
            return "";
        }
        return GetFormatBacktrace(*op);
    }

}