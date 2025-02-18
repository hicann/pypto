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
 * \file pass_log.h
 * \brief
 */

#ifndef PASS_LOG_H
#define PASS_LOG_H

#include <string>
#include "interface/utils/log.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

std::string GetFormatBacktrace(const Operation& op);

std::string GetFormatBacktrace(const OperationPtr& op);

std::string GetFormatBacktrace(const Operation* op);

enum class Elements {
    Operation,
    Tensor,
    Function,
    Graph,
    Config,
    Manager
};

inline const char* toString(Elements elem) {
    static const std::unordered_map<Elements, const char*> passElementName = {
        {Elements::Operation, "Operation"},
        {Elements::Tensor, "Tensor"},
        {Elements::Function, "Function"},
        {Elements::Graph, "Graph"},
        {Elements::Config, "Config"},
        {Elements::Manager, "Manager"}
    };

    auto it = passElementName.find(elem);
    return (it != passElementName.end()) ? it->second : "Unknown";
}
}

#define APASS_LOG_F(lvl, MODULE_NAME, opName, fmt, args...)                        \
    do {                                                                        \
        ALOG_F(lvl, \
        "[%s][%s][" #lvl "]: " fmt, MODULE_NAME, opName, ##args);       \
    } while (false)


#define APASS_LOG_DEBUG_F(opEnum, fmt, args...)   APASS_LOG_F(DEBUG, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_INFO_F(opEnum, fmt, args...)    APASS_LOG_F(INFO, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_WARN_F(opEnum, fmt, args...)    APASS_LOG_F(WARN, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_ERROR_F(opEnum, fmt, args...)   APASS_LOG_F(ERROR, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_EVENT_F(opEnum, fmt, args...)   APASS_LOG_F(EVENT, MODULE_NAME, toString(opEnum), fmt, ##args)

#endif // PASSES_LOG_H