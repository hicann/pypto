/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file unary.cpp
 * \brief Unary block operations (Neg, Exp, Recip, Sqrt, Rsqrt, Cast)
 *
 * This file implements unary operations for block-level programming.
 * Unary operations take a TileType and return a TileType with the same shape.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockUnaryType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got " << args.size();

    // Argument must be TileType
    auto tile_type = As<TileType>(args[0]->GetType());
    CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

    // Unary operations preserve shape and data type
    return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_);
}

TypePtr DeduceBlockCastType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got " << args.size();

    // Argument must be TileType
    auto tile_type = As<TileType>(args[0]->GetType());
    CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

    // Read target_type from kwargs
    bool found_target_type = false;
    DataType target_dtype;
    for (const auto& kwarg : kwargs) {
        if (kwarg.first == "target_type") {
            // Handle both DataType and int for backward compatibility
            if (kwarg.second.type() == typeid(DataType)) {
                target_dtype = AnyCast<DataType>(kwarg.second, "kwarg key: target_type");
            } else if (kwarg.second.type() == typeid(int)) {
                target_dtype = static_cast<DataType>(AnyCast<int>(kwarg.second, "kwarg key: target_type"));
            } else {
                CHECK(false) << "target_type must be a DataType or int, but got " << kwarg.second.type().name();
            }
            found_target_type = true;
            break;
        }
    }
    CHECK(found_target_type) << "block.cast requires 'target_type' kwarg";

    // Cast operation preserves shape but changes data type
    return std::make_shared<TileType>(tile_type->shape_, target_dtype);
}

// ============================================================================
// Op Registration
// ============================================================================

} // namespace ir
} // namespace pypto
