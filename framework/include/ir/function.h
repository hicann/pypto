/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/reflection/field_traits.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

/**
 * \brief Function type classification
 *
 * Categorizes functions by their execution context and purpose:
 * - Opaque: Unspecified (default)
 * - Orchestration: Runs on host/AICPU for control flow and dependency analysis
 * - InCore: Sub-graph on specific AICore
 */
enum class FunctionType : uint8_t {
    OPAQUE = 0,        ///< Default: unspecified function type
    ORCHESTRATION = 1, ///< Host/AICPU control and coordination
    IN_CORE = 2,       ///< AICore sub-graph execution
    HELPER = 3         ///< Scalar helper function callable from kernels
};

/**
 * \brief Convert FunctionType to string
 * \param type The function type
 * \return String representation ("Opaque", "Orchestration", or "InCore")
 */
inline std::string FunctionTypeToString(FunctionType type)
{
    switch (type) {
        case FunctionType::OPAQUE:
            return "Opaque";
        case FunctionType::ORCHESTRATION:
            return "Orchestration";
        case FunctionType::IN_CORE:
            return "InCore";
        case FunctionType::HELPER:
            return "Helper";
        default:
            return "Unknown";
    }
}

/**
 * \brief Convert string to FunctionType
 * \param str String representation
 * \return FunctionType enum value
 * \throws std::invalid_argument if string is not recognized
 */
inline FunctionType StringToFunctionType(const std::string& str)
{
    if (str == "Opaque") {
        return FunctionType::OPAQUE;
    } else if (str == "Orchestration") {
        return FunctionType::ORCHESTRATION;
    } else if (str == "InCore") {
        return FunctionType::IN_CORE;
    } else if (str == "Helper") {
        return FunctionType::HELPER;
    } else {
        throw std::invalid_argument("Unknown FunctionType: " + str);
    }
}

/**
 * \brief Function definition
 *
 * Represents a complete function definition with name, parameters, return types, and body.
 * Functions are immutable IR nodes.
 */
class Function : public IRNode {
public:
    /**
     * \brief Create a function definition
     *
     * \param name Function name
     * \param params Parameter variables
     * \param returnTypes Return types
     * \param body Function body statement (use SeqStmts for multiple statements)
     * \param span Source location
     * \param type Function type (default: Opaque)
     */
    Function(
        std::string name, std::vector<VarPtr> params, std::vector<TypePtr> returnTypes, StmtPtr body, Span span,
        FunctionType type = FunctionType::OPAQUE, bool entry = false)
        : IRNode(std::move(span)),
          name_(std::move(name)),
          funcType_(type),
          entry_(entry),
          params_(std::move(params)),
          returnTypes_(std::move(returnTypes)),
          body_(SeqStmts::Wrap(body, span))
    {}

    Function(Span span) : IRNode(std::move(span)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Function; }
    [[nodiscard]] std::string TypeName() const override { return "Function"; }

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (params as DEF field, func_type, return_types and body as USUAL
     * fields, name as an IGNORE field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            IRNode::GetFieldDescriptors(),
            std::make_tuple(
                reflection::DefField(&Function::params_, "params"),
                reflection::UsualField(&Function::funcType_, "func_type"),
                reflection::UsualField(&Function::entry_, "entry"),
                reflection::UsualField(&Function::returnTypes_, "return_types"),
                reflection::UsualField(&Function::body_, "body"), reflection::IgnoreField(&Function::name_, "name")));
    }

public:
    std::string name_;                 // Function name
    FunctionType funcType_;            // Function type (incore, or opaque)
    bool entry_{false};                // Whether this is the program entry function
    std::vector<VarPtr> params_;       // Parameter variables
    std::vector<TypePtr> returnTypes_; // Return types
    SeqStmtsPtr body_;                 // Function body statement
};

using FunctionPtr = std::shared_ptr<const Function>;

} // namespace ir
} // namespace pypto
