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
 * @file elementwise.cpp
 * \brief Element-wise block operations (Mul, Add, Div, Sub, and scalar variants)
 *
 * This file implements element-wise block operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 * Operations are divided into:
 * - Tile-Tile operations (mul, add, div, sub): TileType + TileType
 * - Tile-Scalar operations (muls, adds, divs, subs): TileType + ScalarType
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockOpElementwiseBinaryType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name,
    bool require_int = false)
{
    CHECK(args.size() == 0x2) << "The operator " << op_name << " requires exactly 2 arguments, but got " << args.size();

    // Both arguments must be TileType
    auto tile_type1 = As<TileType>(args[0]->GetType());
    auto tile_type2 = As<TileType>(args[1]->GetType());

    CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();

    if (require_int) {
        CHECK(tile_type1->dtype_.IsInt())
            << "The operator " << op_name << " requires integer tile dtype, but got " << tile_type1->dtype_.ToString();
        CHECK(tile_type2->dtype_.IsInt())
            << "The operator " << op_name << " requires integer tile dtype, but got " << tile_type2->dtype_.ToString();
    }

    // Use broadcasting
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                    << FormatShape(tile_type1->shape_) << " and " << FormatShape(tile_type2->shape_);

    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

// Tile-tile shift ops (shl, shr): RHS is the shift amount, result type equals LHS tile type,
// consistent with scalar variants (shls/shrs) which preserve the LHS tile dtype.
TypePtr DeduceBlockOpShiftBinaryType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 0x2) << "The operator " << op_name << " requires exactly 2 arguments, but got " << args.size();

    auto tile_type1 = As<TileType>(args[0]->GetType());
    auto tile_type2 = As<TileType>(args[1]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();
    CHECK(tile_type1->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                      << tile_type1->dtype_.ToString();
    CHECK(tile_type2->dtype_.IsInt()) << "The operator " << op_name << " requires integer shift tile dtype, but got "
                                      << tile_type2->dtype_.ToString();

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

    return std::make_shared<TileType>(broadcast_result.shape, tile_type1->dtype_);
}

TypePtr DeduceBlockOpScalarBinaryType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    auto tile_scalar = RequireTileScalarArgs(args, op_name);

    // Result has same shape as tile, with promoted dtype
    auto result_dtype = PromoteDataTypes(tile_scalar.tile_type->dtype_, tile_scalar.scalar_type->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << tile_scalar.tile_type->dtype_.ToString() << " and "
                        << tile_scalar.scalar_type->dtype_.ToString();

    return std::make_shared<TileType>(tile_scalar.tile_type->shape_, *result_dtype);
}

TypePtr DeduceBlockOpIntScalarBinaryType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    auto tile_scalar = RequireTileScalarArgs(args, op_name);

    // First argument must be TileType with integer dtype.
    CHECK(tile_scalar.tile_type->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_scalar.tile_type->dtype_.ToString();

    // Second argument must be ScalarType with an integer dtype per ISA spec:
    //   %dst = tshls/tshrs/tands/tors %src, %scalar : !pto.tile<...>, i32
    // The IR allows any integer width (INT8/16/32/64, UINT variants); codegen casts to i32.
    CHECK(tile_scalar.scalar_type->dtype_.IsInt())
        << "The operator " << op_name << " requires shift/bitwise scalar to be an integer type, but got "
        << tile_scalar.scalar_type->dtype_.ToString();

    // Result has the same shape and dtype as the input tile; the shift amount does not change element type.
    return std::make_shared<TileType>(tile_scalar.tile_type->shape_, tile_scalar.tile_type->dtype_);
}

// ============================================================================
// Op Registration
// ============================================================================

// Tile-tile ternary ops with a tmp buffer as the third argument.
// When require_int is true (bitwise ops like xor), both tile dtypes must be integer.
TypePtr DeduceBlockOpTernaryType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name,
    bool require_int = false)
{
    CHECK(args.size() == 0x3) << "The operator " << op_name << " requires exactly 3 arguments, but got " << args.size();

    auto tile_type1 = As<TileType>(args[0]->GetType());
    auto tile_type2 = As<TileType>(args[1]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();
    CHECK(As<TileType>(args[2]->GetType()))
        << "The operator " << op_name << " requires third argument (tmp) to be a TileType, but got "
        << args[2]->GetType()->TypeName();

    if (require_int) {
        CHECK(tile_type1->dtype_.IsInt())
            << "The operator " << op_name << " requires integer tile dtype, but got " << tile_type1->dtype_.ToString();
        CHECK(tile_type2->dtype_.IsInt())
            << "The operator " << op_name << " requires integer tile dtype, but got " << tile_type2->dtype_.ToString();
    }

    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";
    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

// All three tiles are real inputs (addc, subc): promote dtype and broadcast shape across all three.
TypePtr DeduceBlockOpTriTileType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 0x3) << "The operator " << op_name << " requires exactly 3 arguments, but got " << args.size();

    auto tile_type1 = As<TileType>(args[0]->GetType());
    auto tile_type2 = As<TileType>(args[1]->GetType());
    auto tile_type3 = As<TileType>(args[2]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();
    CHECK(tile_type3) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                      << args[2]->GetType()->TypeName();

    auto result_dtype12 = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    CHECK(result_dtype12) << "The operator " << op_name << " requires compatible data types";
    auto result_dtype = PromoteDataTypes(*result_dtype12, tile_type3->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

    auto broadcast12 = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast12.success) << "The operator " << op_name << " requires compatible shapes";
    auto broadcast_result = BroadcastShapes(broadcast12.shape, tile_type3->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

// (Tile, Scalar, Tile) pattern (addsc, subsc): any scalar type, promote output from all three inputs.
TypePtr DeduceBlockOpTileScalarTileType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 0x3) << "The operator " << op_name << " requires exactly 3 arguments, but got " << args.size();

    auto tile_type1 = As<TileType>(args[0]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();

    auto scalar_type = As<ScalarType>(args[1]->GetType());
    CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                       << args[1]->GetType()->TypeName();

    auto tile_type2 = As<TileType>(args[2]->GetType());
    CHECK(tile_type2) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                      << args[2]->GetType()->TypeName();

    auto result_dtype12 = PromoteDataTypes(tile_type1->dtype_, scalar_type->dtype_);
    CHECK(result_dtype12) << "The operator " << op_name << " requires compatible data types";
    auto result_dtype = PromoteDataTypes(*result_dtype12, tile_type2->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

TypePtr MakePromotedBroadcastTileType(
    const TileTypePtr& lhs_type, const TileTypePtr& rhs_type, const std::string& op_name)
{
    auto result_dtype = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << lhs_type->dtype_.ToString() << " and " << rhs_type->dtype_.ToString();

    auto broadcast_result = BroadcastShapes(lhs_type->shape_, rhs_type->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                    << FormatShape(lhs_type->shape_) << " and " << FormatShape(rhs_type->shape_);

    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceBlockOpXorScalarType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    auto tile_scalar = RequireTileScalarArgs(args, op_name, 3);

    CHECK(tile_scalar.tile_type->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_scalar.tile_type->dtype_.ToString();

    // Second argument must be ScalarType with an integer dtype per ISA spec:
    //   %dst = txors %src, %scalar : !pto.tile<...>, i32
    // The IR allows any integer width (INT8/16/32/64, UINT variants); codegen casts to i32.
    CHECK(tile_scalar.scalar_type->dtype_.IsInt())
        << "The operator " << op_name << " requires scalar to be an integer type, but got "
        << tile_scalar.scalar_type->dtype_.ToString();

    CHECK(As<TileType>(args[2]->GetType()))
        << "The operator " << op_name << " requires third argument to be a TileType, but got "
        << args[2]->GetType()->TypeName();

    // Result has the same shape and dtype as the input tile; bitwise ops do not change element type.
    return std::make_shared<TileType>(tile_scalar.tile_type->shape_, tile_scalar.tile_type->dtype_);
}

// Type deduction for block.sel (MaskTile x Tile x Tile -> Tile)
// The mask tile encodes per-element predicates in a target-defined layout; its dtype/shape
// do not influence the output type.  Output type is derived from lhs and rhs only.
TypePtr DeduceBlockSelType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 0x3) << "The operator " << op_name << " requires exactly 3 arguments, but got " << args.size();

    CHECK(As<TileType>(args[0]->GetType()))
        << "The operator " << op_name << " requires first argument (mask) to be a TileType, but got "
        << args[0]->GetType()->TypeName();

    auto tile_type1 = As<TileType>(args[1]->GetType());
    auto tile_type2 = As<TileType>(args[2]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires second argument (lhs) to be a TileType, but got "
                      << args[1]->GetType()->TypeName();
    CHECK(tile_type2) << "The operator " << op_name << " requires third argument (rhs) to be a TileType, but got "
                      << args[2]->GetType()->TypeName();

    return MakePromotedBroadcastTileType(tile_type1, tile_type2, op_name);
}

// Type deduction for block.sels (Tile x Tile x Scalar -> Tile)
TypePtr DeduceBlockSelScalarType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 0x3) << "The operator " << op_name << " requires exactly 3 arguments, but got " << args.size();

    auto tile_type1 = As<TileType>(args[0]->GetType());
    auto tile_type2 = As<TileType>(args[1]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires first argument (lhs) to be a TileType, but got "
                      << args[0]->GetType()->TypeName();
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument (rhs) to be a TileType, but got "
                      << args[1]->GetType()->TypeName();

    CHECK(As<ScalarType>(args[2]->GetType()))
        << "The operator " << op_name << " requires third argument (select_mode) to be a ScalarType, but got "
        << args[2]->GetType()->TypeName();

    return MakePromotedBroadcastTileType(tile_type1, tile_type2, op_name);
}

// Type deduction for block.cmp and block.cmps (comparison operations)
TypePtr DeduceBlockCmpType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name,
    bool is_scalar_rhs = false)
{
    CHECK(args.size() == 0x2) << "The operator " << op_name << " requires exactly 2 arguments, but got " << args.size();

    // Validate cmp_type attribute exists
    bool has_cmp_type = false;
    for (const auto& kwarg : kwargs) {
        if (kwarg.first == "cmp_type") {
            has_cmp_type = true;
            break;
        }
    }
    CHECK(has_cmp_type) << "The operator " << op_name << " requires 'cmp_type' attribute";

    // First argument must be TileType
    auto tile_type1 = As<TileType>(args[0]->GetType());
    CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                      << args[0]->GetType()->TypeName();

    if (is_scalar_rhs) {
        // Second argument MUST be ScalarType
        auto scalar_type = As<ScalarType>(args[1]->GetType());
        CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                           << args[1]->GetType()->TypeName();

        // Result has same shape as tile, with promoted dtype
        auto result_dtype = PromoteDataTypes(tile_type1->dtype_, scalar_type->dtype_);
        CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                            << tile_type1->dtype_.ToString() << " and " << scalar_type->dtype_.ToString();

        return std::make_shared<TileType>(tile_type1->shape_, *result_dtype);
    } else {
        // Second argument must be TileType
        auto tile_type2 = As<TileType>(args[1]->GetType());
        CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                          << args[1]->GetType()->TypeName();

        // Use broadcasting
        auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
        CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                            << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

        auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
        CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                        << FormatShape(tile_type1->shape_) << " and "
                                        << FormatShape(tile_type2->shape_);

        return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
    }
}

} // namespace ir
} // namespace pypto
