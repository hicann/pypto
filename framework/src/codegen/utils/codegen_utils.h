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
 * \file codegen_utils.h
 * \brief
 */

#ifndef CODEGEN_UTILS_H
#define CODEGEN_UTILS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <variant>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {
constexpr int COMMENT_PREFIX_LENGTH = 2;

template <typename T>
inline void FillIntVecWithDummyInHead(std::vector<T> &input, unsigned padNum, T dummy) {
    for (unsigned i = 0; i < padNum; ++i) {
        input.insert(input.begin(), dummy);
    }
}

// only recogonize /* as comment prefix
inline bool StartWithComment(const std::string &str) {
    return str.size() >= COMMENT_PREFIX_LENGTH && str[0] == '/' && str[1] == '*';
}
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::string> ToStringHelper(const T &value) {
    return std::to_string(value);
}
inline std::string ToStringHelper(const std::string &value) {
    return value;
}

template <typename T = std::string>
std::string JoinString(const std::vector<T> &params, const std::string &sep) {
    std::ostringstream oss;

    for (size_t i = 0; i < params.size(); ++i) {
        std::string current = ToStringHelper(params[i]);
        if (current.empty()) {
            continue;
        }
        if (i > 0) {
            bool useEmptySep{false};
            if constexpr (std::is_same_v<T, std::string>) {
                useEmptySep = StartWithComment(params[i - 1]);
            }
            oss << (useEmptySep ? " " : sep);
        }
        oss << current;
    }

    return oss.str();
}

template <typename T = std::string>
std::string PrintParams(
    const std::pair<std::string, std::string> &delimiter, const std::vector<T> &params, const std::string &conj) {
    std::ostringstream oss;
    oss << delimiter.first << JoinString<T>(params, conj) << delimiter.second;
    return oss.str();
}

template <typename T = std::string>
std::string WrapParamByParentheses(const std::vector<T> &params) {
    return PrintParams(DELIMITER_PARENTHESES, params, CONN_COMMA);
}

template <typename T = std::string>
std::string WrapParamByAngleBrackets(const std::vector<T> &params) {
    return PrintParams(DELIMITER_ANGLE_BRACKETS, params, CONN_COMMA);
}

std::vector<int64_t> NormalizeShape(const std::vector<int64_t> &shapeVec, unsigned dim);
std::string FormatFloat(const std::variant<int64_t, uint64_t, double> &v, int precision = 9);

std::string GetTypeForB16B32(const DataType &dtype);

inline std::string GetPipeId(PipeType queue) {
    auto res = PIPE_ID.find(queue);
    return res == PIPE_ID.end() ? "" : res->second;
}

inline std::string GetTileOpName(Opcode opCode) {
    const auto &opCfg = OpcodeManager::Inst().GetTileOpCfg(opCode);
    return opCfg.tileOpCode_;
}

std::string GetAddrTypeByOperandType(OperandType type);

int64_t CalcLinearOffset(const std::vector<int64_t> &shape, const std::vector<int64_t> &offset);

template <typename T>
void FillParamWithInput(std::vector<std::string> &paramList, const std::vector<T> &input, int start, int count) {
    for (int i = start; i < count; ++i) {
        paramList.emplace_back(ToStringHelper(input[i]));
    }
}

} // namespace npu::tile_fwk
#endif
