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
#include <set>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

/**
 * \brief Program definition
 *
 * Represents a complete program with functions.
 * Programs are immutable IR nodes.
 *
 * Functions are stored in a sorted vector (by name) to ensure deterministic
 * ordering for structural equality and hashing.
 *
 * \note The GlobalVar name must match the function name and be unique within the program.
 *       Validation of this constraint may be added in future passes.
 */
class Program : public IRNode {
public:
    /**
     * \brief Create a program from a vector of functions
     *
     * \param functions Vector of functions
     * \param name Program name (optional)
     * \param span Source location
     */
    Program(std::vector<FunctionPtr> functions, std::string name, Span span)
        : IRNode(std::move(span)), name_(std::move(name)), functions_(std::move(functions))
    {
        std::set<std::string> funcNames;
        for (auto func : functions_) {
            if (funcNames.count(func->name_)) {
                throw InternalError("Duplicate function name: " + func->name_);
            }
            funcNames.insert(func->name_);
        }
    }

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Program; }
    [[nodiscard]] std::string TypeName() const override { return "Program"; }

    /**
     * \brief Get a function by name
     *
     * \param name Function name to look up
     * \return Shared pointer to the function, or nullptr if not found
     */
    [[nodiscard]] FunctionPtr GetFunction(const std::string& name) const
    {
        {
            for (auto func : functions_) {
                if (func->name_ == name) {
                    return func;
                }
            }
            return nullptr;
        }
    }

    /**
     * \brief Get a GlobalVar by name
     *
     * \param name GlobalVar name to look up
     * \return Shared pointer to the GlobalVar, or nullptr if not found
     */
    [[nodiscard]] GlobalVarPtr GetGlobalVar(const std::string& name) const;

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors (name as IGNORE field, functions as USUAL field)
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            IRNode::GetFieldDescriptors(), std::make_tuple(
                                               reflection::IgnoreField(&Program::name_, "name"),
                                               reflection::UsualField(&Program::functions_, "functions")));
    }

public:
    std::string name_;                   // Program name
    std::vector<FunctionPtr> functions_; // Vector of Functions
};

using ProgramPtr = std::shared_ptr<const Program>;

} // namespace ir
} // namespace pypto
