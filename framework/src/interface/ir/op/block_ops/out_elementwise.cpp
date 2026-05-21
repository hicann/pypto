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
 * @file block_ops/out_elementwise.cpp
 * \brief Block element-wise operations with explicit output tiles.
 *
 * All ops accept a pre-allocated output tile as the last argument and return
 * that tile's type.  This file covers:
 *   - Tile x Tile binary: add, sub, mul, div, rem, maximum, minimum, and, or, shl, shr
 *   - Tile x Scalar binary: adds, subs, muls, divs, rems, ands, ors, shls, shrs, maxs, mins, lrelu
 *   - Unary: neg, exp, sqrt, rsqrt, recip, log, abs, relu, not, cast
 *   - Ternary: xor/xors (with tmp), prelu (with tmp), addc, subc, addsc, subsc, sel, sels
 *   - Comparison: cmp, cmps
 *   - Scalar-to-tile: expands
 *   - Layout: reshape, transpose
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

// ---------------------------------------------------------------------------
// Shared type deduction helpers
// ---------------------------------------------------------------------------

/// Return the TileType of the last argument (the pre-allocated output tile).
static TypePtr DeduceBlockOutType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name,
    size_t expected_args)
{
    CHECK(args.size() == expected_args) << "The operator " << op_name << " requires exactly " << expected_args
                                        << " arguments, but got " << args.size();
    auto out_type = As<TileType>(args.back()->GetType());
    CHECK(out_type) << "The operator " << op_name << " requires last argument (out) to be TileType, but got "
                    << args.back()->GetType()->TypeName();
    return out_type;
}

// Validate that args[idx] is TileType.
static void CheckTileArg([[maybe_unused]] const std::vector<ExprPtr>& args, size_t idx, const std::string& op_name)
{
    CHECK(As<TileType>(args[idx]->GetType())) << "The operator " << op_name << " requires argument " << idx
                                              << " to be TileType, but got " << args[idx]->GetType()->TypeName();
}

// Validate that args[idx] is ScalarType.
static void CheckScalarArg([[maybe_unused]] const std::vector<ExprPtr>& args, size_t idx, const std::string& op_name)
{
    CHECK(As<ScalarType>(args[idx]->GetType())) << "The operator " << op_name << " requires argument " << idx
                                                << " to be ScalarType, but got " << args[idx]->GetType()->TypeName();
}

// Type deduction for (TileType, TileType, out:TileType) -> out.
static TypePtr DeduceBlockOutBinaryTile(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 3) << op_name << " requires 3 arguments (lhs, rhs, out)";
    CheckTileArg(args, 0, op_name);
    CheckTileArg(args, 1, op_name);
    return DeduceBlockOutType(args, kwargs, op_name, 3);
}

// Type deduction for (TileType, ScalarType, out:TileType) -> out.
static TypePtr DeduceBlockOutBinaryScalar(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 3) << op_name << " requires 3 arguments (tile, scalar, out)";
    CheckTileArg(args, 0, op_name);
    CheckScalarArg(args, 1, op_name);
    return DeduceBlockOutType(args, kwargs, op_name, 3);
}

// Type deduction for (TileType, out:TileType) -> out  (unary).
static TypePtr DeduceBlockOutUnary(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 2) << op_name << " requires 2 arguments (src, out)";
    CheckTileArg(args, 0, op_name);
    return DeduceBlockOutType(args, kwargs, op_name, 2);
}

// ---------------------------------------------------------------------------
// Tile x Tile binary operations
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_OUT_BINARY_TILE(name)                                                              \
    REGISTER_OP("block." #name)                                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output element-wise " #name ": out = lhs " #name " rhs")         \
        .add_argument("lhs", "Left tile (TileType)")                                                      \
        .add_argument("rhs", "Right tile (TileType)")                                                     \
        .add_argument("out", "Pre-allocated output tile (TileType)")                                      \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutBinaryTile(args, kwargs, "block." #name);                                \
        })

REGISTER_BLOCK_OUT_BINARY_TILE(add);
REGISTER_BLOCK_OUT_BINARY_TILE(sub);
REGISTER_BLOCK_OUT_BINARY_TILE(mul);
REGISTER_BLOCK_OUT_BINARY_TILE(div);
REGISTER_BLOCK_OUT_BINARY_TILE(rem);
REGISTER_BLOCK_OUT_BINARY_TILE(maximum);
REGISTER_BLOCK_OUT_BINARY_TILE(minimum);

// Bitwise tile-tile ops (integer only; validated at the Python layer).
REGISTER_BLOCK_OUT_BINARY_TILE(and);
REGISTER_BLOCK_OUT_BINARY_TILE(or);
REGISTER_BLOCK_OUT_BINARY_TILE(shl);
REGISTER_BLOCK_OUT_BINARY_TILE(shr);
REGISTER_BLOCK_OUT_BINARY_TILE(add_relu);
REGISTER_BLOCK_OUT_BINARY_TILE(sub_relu);
REGISTER_BLOCK_OUT_BINARY_TILE(mul_add_dst);
REGISTER_BLOCK_OUT_BINARY_TILE(fused_mul_add);
REGISTER_BLOCK_OUT_BINARY_TILE(fused_mul_add_relu);

#undef REGISTER_BLOCK_OUT_BINARY_TILE

// block.add_relu_cast: (lhs_tile, rhs_tile, out) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.add_relu_cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output add-relu-cast: out = cast(max(0, lhs + rhs), target_dtype, rounding_mode)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<std::string>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.add_relu_cast");
    });

// block.sub_relu_cast: (lhs_tile, rhs_tile, out) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.sub_relu_cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output sub-relu-cast: out = cast(max(0, lhs - rhs), target_dtype, rounding_mode)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<std::string>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.sub_relu_cast");
    });

// block.mul_cast: (lhs_tile, rhs_tile, out) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.mul_cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output mul-cast: out = cast(lhs * rhs, target_dtype_dtype, rounding_mode)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<std::string>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.mul_cast");
    });

// ---------------------------------------------------------------------------
// Tile x Scalar binary operations
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_OUT_BINARY_SCALAR(name)                                                            \
    REGISTER_OP("block." #name)                                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output tile-scalar " #name ": out = tile " #name " scalar")      \
        .add_argument("tile", "Input tile (TileType)")                                                    \
        .add_argument("scalar", "Scalar operand (ScalarType)")                                            \
        .add_argument("out", "Pre-allocated output tile (TileType)")                                      \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutBinaryScalar(args, kwargs, "block." #name);                              \
        })

REGISTER_BLOCK_OUT_BINARY_SCALAR(adds);
REGISTER_BLOCK_OUT_BINARY_SCALAR(subs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(muls);
REGISTER_BLOCK_OUT_BINARY_SCALAR(divs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(rems);
REGISTER_BLOCK_OUT_BINARY_SCALAR(ands);
REGISTER_BLOCK_OUT_BINARY_SCALAR(ors);
REGISTER_BLOCK_OUT_BINARY_SCALAR(shls);
REGISTER_BLOCK_OUT_BINARY_SCALAR(shrs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(maxs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(mins);
REGISTER_BLOCK_OUT_BINARY_SCALAR(lrelu);

#undef REGISTER_BLOCK_OUT_BINARY_SCALAR

// block.gather: Two forms:
//   Index form: (src_tile, indices_tile, out) or (src_tile, indices_tile, tmp, out) -> out's type
//   Compare form: (src_tile, k_value_tile, cdst_tile, tmp_tile, out) + kwargs(cmp_mode, offset) -> out's type
REGISTER_OP("block.gather")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output gather: index form (out[i] = src[indices[i]]) or compare form (indices where src > kth "
        "or src == kth)")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("indices_or_k_value", "Index tile (index form) or k-value tile (compare form) (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("tmp_or_cdst", "Optional tmp tile (index form) or cdst tile (compare form) (TileType)")
    .add_argument("tmp", "Optional tmp tile for compare form (TileType)")
    .set_attr<int>("cmp_mode")
    .set_attr<int>("offset")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        // Index form: 3 or 4 args (src, indices, out) or (src, indices, tmp, out)
        if (args.size() == 3 || args.size() == 4) {
            bool has_cmp_mode = false;
            for (const auto& [key, val] : kwargs) {
                (void)val;
                if (key == "cmp_mode")
                    has_cmp_mode = true;
            }
            if (!has_cmp_mode) {
                return DeduceBlockOutType(args, kwargs, "block.gather", args.size());
            }
        }
        // Compare form: 5 args (src, k_value, cdst, tmp, out) + cmp_mode + offset
        if (args.size() == 5) {
            return DeduceBlockOutType(args, kwargs, "block.gather", 5);
        }
        throw std::runtime_error("block.gather: expected index form (3-4 args) or compare form (5 args + cmp_mode)");
    });

// block.gatherb: (src_tile, offsets_tile, out) -> out's type
REGISTER_OP("block.gatherb")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output gatherb: out[i] = src[byte_offsets[i]]")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("offsets", "Byte offset tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.gatherb", 3);
    });

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_OUT_UNARY(name)                                                                    \
    REGISTER_OP("block." #name)                                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output unary " #name ": out = " #name "(src)")                   \
        .add_argument("src", "Input tile (TileType)")                                                     \
        .add_argument("out", "Pre-allocated output tile (TileType)")                                      \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutUnary(args, kwargs, "block." #name);                                     \
        })

REGISTER_BLOCK_OUT_UNARY(neg);
REGISTER_BLOCK_OUT_UNARY(exp);
REGISTER_BLOCK_OUT_UNARY(sqrt);
REGISTER_BLOCK_OUT_UNARY(rsqrt);
REGISTER_BLOCK_OUT_UNARY(recip);
REGISTER_BLOCK_OUT_UNARY(log);
REGISTER_BLOCK_OUT_UNARY(abs);
REGISTER_BLOCK_OUT_UNARY(relu);
REGISTER_BLOCK_OUT_UNARY(not);

#undef REGISTER_BLOCK_OUT_UNARY

// block.cast: (src_tile, out) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output type-cast: out = cast(src, target_dtype, rounding_mode)")
    .add_argument("src", "Input tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<std::string>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutUnary(args, kwargs, "block.cast");
    });

// ---------------------------------------------------------------------------
// Ternary / multi-input operations
// ---------------------------------------------------------------------------

// XOR with tmp buffer (tile, tile, tmp, out): 4 args.
REGISTER_OP("block.xor")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output bitwise XOR: out = lhs ^ rhs (integer tiles; tmp is scratch buffer)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.xor", 4);
    });

// XOR-scalar with tmp buffer (tile, scalar, tmp, out): 4 args.
REGISTER_OP("block.xors")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output bitwise XOR with scalar: out = lhs ^ scalar (integer tiles)")
    .add_argument("lhs", "Input tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.xors", 4);
    });

// prelu with tmp buffer (tile, slope, tmp, out): 4 args.
REGISTER_OP("block.prelu")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output parametric ReLU: out = prelu(tile, slope); tmp is scratch buffer")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Slope tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.prelu", 4);
    });

// Three-tile arithmetic (tile, tile, tile, out): 4 args.
REGISTER_OP("block.addc")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output three-tile add: out = lhs + rhs + rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("rhs", "Second tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.addc", 4);
    });

REGISTER_OP("block.subc")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output three-tile sub: out = lhs - rhs - rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("rhs", "Second tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.subc", 4);
    });

// (tile, scalar, tile, out): 4 args.
REGISTER_OP("block.addsc")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output tile+scalar+tile add: out = lhs + scalar + rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.addsc", 4);
    });

REGISTER_OP("block.subsc")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output tile-scalar-tile sub: out = lhs - scalar - rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.subsc", 4);
    });

// Selection (mask, lhs, rhs, tmp, out): 5 args.
REGISTER_OP("block.sel")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output per-element selection: out[i]=lhs[i] if mask[i] else rhs[i]; tmp is scratch buffer")
    .add_argument("lhs", "True-branch tile (TileType)")
    .add_argument("mask", "Predicate mask tile (TileType)")
    .add_argument("rhs", "False-branch tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.sel", 5);
    });

// sels (lhs, rhs, scalar_mode, out): 4 args.
REGISTER_OP("block.sels")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output mode-based selection: out = sels(lhs, rhs, mode)")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("rhs", "Second tile (TileType)")
    .add_argument("select_mode", "Scalar mode (ScalarType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.sels", 4);
    });

// ---------------------------------------------------------------------------
// Comparison operations
// ---------------------------------------------------------------------------

REGISTER_OP("block.cmp")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output element-wise tile comparison: out = (lhs cmp_op rhs)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.cmp");
    });

REGISTER_OP("block.cmps")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output element-wise tile-scalar comparison: out = (tile cmp_op scalar)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("scalar", "Scalar comparand (ScalarType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryScalar(args, kwargs, "block.cmps");
    });

// ---------------------------------------------------------------------------
// Scalar-to-tile broadcast
// ---------------------------------------------------------------------------

REGISTER_OP("block.expands")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output scalar broadcast: fill out tile with scalar value (out[i,j] = scalar)")
    .add_argument("scalar", "Fill value (ScalarType or constant)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.expands", 2);
    });

// ---------------------------------------------------------------------------
// Layout operations
// ---------------------------------------------------------------------------

// block.reshape: (src_tile, shape_tuple, out) -> out's type.
REGISTER_OP("block.reshape")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output reshape: reinterpret src tile layout into out's shape")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("shape", "New shape dimensions (MakeTuple)")
    .add_argument("out", "Pre-allocated output tile with target shape (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutType(args, kwargs, "block.reshape", 3);
    });

// block.transpose: (src_tile, out) -> out's type; axis attrs.
REGISTER_OP("block.transpose")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output transpose: swap two axes of src tile into out")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with transposed shape (TileType)")
    .set_attr<int>("axis1")
    .set_attr<int>("axis2")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutUnary(args, kwargs, "block.transpose");
    });

} // namespace ir
} // namespace pypto
