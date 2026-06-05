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
 * @file memory.cpp
 * \brief Memory block operations (get_block_idx, load, store)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockGetBlockIdxType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

    // get_block_idx returns INT64 scalar (for compatibility with arith.index_cast)
    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceBlockCreateTileType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name)
{
    // make_tile signature: (shape)
    // TileType requires static compile-time constant shapes
    CHECK(args.size() == 0x2)
        << "The operator " << op_name << " requires exactly 2 argument, but got " << args.size();

    // Extract dtype attribute
    DataType dtype = GetOpKwarg<DataType>(kwargs, "dtype");

    // First argument must be MakeTuple with static ConstInt elements
    auto shape_tuple = As<MakeTuple>(args[0]);
    CHECK(shape_tuple) << "The operator " << op_name
                       << " requires first argument to be a MakeTuple expression with static shape values, but got "
                       << args[0]->TypeName();

    // Validate all elements are ConstInt (static compile-time constants)
    std::vector<ExprPtr> tile_shape;
    tile_shape.reserve(shape_tuple->elements_.size());

    for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
        auto const_int = As<ConstInt>(shape_tuple->elements_[i]);
        CHECK(const_int) << "The operator " << op_name << " shape element " << i
                         << " must be a compile-time constant (ConstInt), but got "
                         << shape_tuple->elements_[i]->TypeName();
        CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                     << " must be positive, got " << const_int->value_;
        tile_shape.push_back(shape_tuple->elements_[i]);
    }

    CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

    TileView tile_view;

    auto valid_shape_tuple = As<MakeTuple>(args[1]);
    if (valid_shape_tuple)
        tile_view.validShape = valid_shape_tuple->elements_;

    HardwareInfo hw_info;

    int blayout = GetOpKwarg<int>(kwargs, "blayout", -1);
    if (blayout >= 0) {
        hw_info.blayout = static_cast<TileLayout>(blayout);
    }

    int slayout = GetOpKwarg<int>(kwargs, "slayout", -1);
    if (slayout >= 0) {
        hw_info.slayout = static_cast<TileLayout>(slayout);
    }

    int fractal = GetOpKwarg<int>(kwargs, "fractal", -1);
    if (fractal >= 0) {
        hw_info.fractal = static_cast<uint64_t>(fractal);
    }

    int pad = GetOpKwarg<int>(kwargs, "pad", -1);
    if (pad >= 0) {
        hw_info.pad = static_cast<TilePad>(pad);
    }

    int compact = GetOpKwarg<int>(kwargs, "compact", -1);
    if (compact >= 0) {
        hw_info.compact = static_cast<CompactMode>(compact);
    }
    // If explicit memref kwargs are provided (addr + size + id), attach a MemRef to the TileType.
    // This allows the PTO codegen to emit pto.alloc_tile with base_addr directly from the IR,
    // without requiring the init_memref pass.
    MemorySpace target_memory =
        GetOpKwarg<MemorySpace>(kwargs, "target_memory", std::optional<MemorySpace>(MemorySpace::Vec));

    bool has_memref = false;
    for (const auto& kwarg : kwargs) {
        if (kwarg.first == "memref_id") {
            has_memref = true;
            break;
        }
    }
    if (has_memref) {
        int64_t addr_val = GetOpKwarg<int>(kwargs, "memref_addr");
        int64_t size_val = GetOpKwarg<int>(kwargs, "memref_size");
        uint64_t id_val = static_cast<uint64_t>(GetOpKwarg<int>(kwargs, "memref_id"));
        auto addr_expr = std::make_shared<ConstInt>(addr_val, DataType::INDEX, Span::Unknown());
        MemRefPtr memref = std::make_shared<MemRef>(target_memory, addr_expr, static_cast<uint64_t>(size_val), id_val);
        return std::make_shared<TileType>(tile_shape, dtype, std::optional<MemRefPtr>(memref), tile_view, hw_info);
    }

    return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view, hw_info);
}

TypePtr DeduceBlockGetValType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // block.getval: Read a scalar value from a tile at flattened index
    // Args: (tile, index)
    // Returns: ScalarType with tile's element dtype
    CHECK(args.size() == 0x2)
        << "block.getval requires exactly 2 arguments (tile, index), but got " << args.size();

    // First argument must be TileType
    auto tile_type = As<TileType>(args[0]->GetType());
    CHECK(tile_type) << "block.getval requires first argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

    // Second argument must be ScalarType with integer dtype (flattened index)
    auto index_type = As<ScalarType>(args[1]->GetType());
    CHECK(index_type) << "block.getval requires index to be ScalarType, but got " << args[1]->GetType()->TypeName();
    CHECK(index_type->dtype_.IsInt()) << "block.getval index must have integer dtype, but got "
                                      << index_type->dtype_.ToString();

    // Return ScalarType with tile's element dtype
    return std::make_shared<ScalarType>(tile_type->dtype_);
}

TypePtr DeduceBlockSetValType(
    [[maybe_unused]] const std::vector<ExprPtr>& args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // block.setval: Write a scalar value to a tile at flattened index
    // Args: (tile, index, value)
    // Returns: TileType (same as input tile)
    CHECK(args.size() == 0x3)
        << "block.setval requires exactly 3 arguments (tile, index, value), but got " << args.size();

    // First argument must be TileType
    auto tile_type = As<TileType>(args[0]->GetType());
    CHECK(tile_type) << "block.setval requires first argument to be a TileType, but got "
                     << args[0]->GetType()->TypeName();

    // Second argument must be ScalarType with integer dtype (flattened index)
    auto index_type = As<ScalarType>(args[1]->GetType());
    CHECK(index_type) << "block.setval requires index to be ScalarType, but got " << args[1]->GetType()->TypeName();
    CHECK(index_type->dtype_.IsInt()) << "block.setval index must have integer dtype, but got "
                                      << index_type->dtype_.ToString();

    // Third argument must be ScalarType (value to write)
    auto value_type = As<ScalarType>(args[2]->GetType());
    CHECK(value_type) << "block.setval requires value to be ScalarType, but got " << args[2]->GetType()->TypeName();

    // Value type should match tile (or be compatible for implicit conversion)
    // For now, we just return the tile type with same shape, dtype, and memref
    return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_);
}

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("get_block_idx")
    .set_op_category("LanguageOp")
    .set_description("Get the current block index")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_block_idx");
    });

REGISTER_OP("get_block_num")
    .set_op_category("LanguageOp")
    .set_description("Get the current block number")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_block_num");
    });

REGISTER_OP("get_subblock_idx")
    .set_op_category("LanguageOp")
    .set_description("Get the current subblock index")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_subblock_idx");
    });

REGISTER_OP("index_cast")
    .set_op_category("LanguageOp")
    .set_description("Cast scalar to index type")
    .add_argument("idx", "Input scalar (ScalarType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 1) << "index_cast requires 1 argument, but got " << args.size();
        auto scalar_type = As<ScalarType>(args[0]->GetType());
        CHECK(scalar_type) << "index_cast requires argument to be ScalarType, but got "
                           << args[0]->GetType()->TypeName();
        return std::make_shared<ScalarType>(DataType::INDEX);
    });

REGISTER_OP("block.make_tile")
    .set_op_category("BlockOp")
    .set_description("Create a tile")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("valid_shape", "Valid shape dimensions (optional, TupleType)")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<int>("memref_addr")
    .set_attr<int>("memref_size")
    .set_attr<int>("memref_id")
    .set_attr<int>("blayout")
    .set_attr<int>("slayout")
    .set_attr<int>("fractal")
    .set_attr<int>("pad")
    .set_attr<int>("compact")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockCreateTileType(args, kwargs, "block.make_tile");
    });

REGISTER_OP("block.getval")
    .set_op_category("BlockOp")
    .set_description("Read a scalar value from a tile at flattened index")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("index", "Flattened element index in tile layout (ScalarType with integer dtype)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetValType(args, kwargs);
    });

REGISTER_OP("block.setval")
    .set_op_category("BlockOp")
    .set_description("Write a scalar value to a tile at flattened index")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("index", "Flattened element index in tile layout (ScalarType with integer dtype)")
    .add_argument("value", "Scalar value to write (ScalarType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockSetValType(args, kwargs);
    });
} // namespace ir
} // namespace pypto
