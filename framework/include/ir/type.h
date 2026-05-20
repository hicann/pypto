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
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/memory_space.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declaration
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

class MemRef;
using MemRefPtr = std::shared_ptr<const MemRef>;

/**
 * \brief Base class for type representations in the IR
 *
 * Types represent the structure and properties of values in the IR.
 * All types are immutable.
 */
class Type {
public:
    virtual ~Type() = default;

    /**
     * \brief Get the Kind of this type
     *
     * \return The ObjectKind enum value identifying the concrete type
     */
    [[nodiscard]] virtual ObjectKind GetKind() const = 0;

    /**
     * \brief Get the type name of this type
     *
     * \return Human-readable type name (e.g., "ScalarType", "TensorType")
     */
    [[nodiscard]] virtual std::string TypeName() const { return "Type"; }

    static constexpr auto GetFieldDescriptors() { return std::make_tuple(); }
};

using TypePtr = std::shared_ptr<const Type>;

/**
 * \brief Unknown type representation
 *
 * Represents an unknown or unspecified type.
 * Used as the default type for expressions when type information is not available.
 */
class UnknownType : public Type {
public:
    /**
     * \brief Create an unknown type
     */
    UnknownType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::UnknownType; }
    [[nodiscard]] std::string TypeName() const override { return "UnknownType"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using UnknownTypePtr = std::shared_ptr<const UnknownType>;

/**
 * \brief Get a shared pointer to the singleton UnknownType instance
 *
 * \return Shared pointer to UnknownType
 */
inline UnknownTypePtr GetUnknownType()
{
    static const auto unknownType = std::make_shared<UnknownType>();
    return unknownType;
}

/**
 * \brief Scalar type representation
 *
 * Represents a scalar value type with a data type.
 */
class ScalarType : public Type {
public:
    DataType dtype_;

    /**
     * \brief Create a scalar type
     *
     * \param dtype Data type
     */
    explicit ScalarType(DataType dtype) : dtype_(dtype) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ScalarType; }
    [[nodiscard]] std::string TypeName() const override { return "ScalarType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Type::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&ScalarType::dtype_, "dtype")));
    }
};

using ScalarTypePtr = std::shared_ptr<const ScalarType>;

/**
 * @brief Tensor layout enumeration
 *
 * Defines the available tensor layout types:
 * - ND: ND layout
 * - DN: DN layout
 * - NZ: NZ layout
 */
enum class TensorLayout {
    ND, ///< ND layout
    DN, ///< DN layout
    NZ  ///< NZ layout
};

/**
 * \brief Tile view representation
 *
 * Represents the view information for a tile, including valid shape,
 * stride, and start offset. This is used by TileType to track how
 * a tile views its underlying memory.
 */
struct TileView {
    std::vector<ExprPtr> validShape; ///< Valid shape dimensions
    std::vector<ExprPtr> stride;     ///< Stride for each dimension
    ExprPtr startOffset;             ///< Starting offset

    /**
     * \brief Default constructor for aggregate initialization
     */
    TileView() = default;

    /**
     * \brief Constructor with all parameters
     *
     * \param validShape Valid shape dimensions
     * \param stride Stride for each dimension
     * \param startOffset Starting offset
     */
    TileView(std::vector<ExprPtr> validShapeIn, std::vector<ExprPtr> strideIn, ExprPtr startOffsetIn)
        : validShape(std::move(validShapeIn)), stride(std::move(strideIn)), startOffset(std::move(startOffsetIn))
    {}

    /**
     * \brief Get field descriptors for reflection-based visitation
     *
     * \return Tuple of field descriptors
     */
    static constexpr auto GetFieldDescriptors()
    {
        return std::make_tuple(
            reflection::UsualField(&TileView::validShape, "valid_shape"),
            reflection::UsualField(&TileView::stride, "stride"),
            reflection::UsualField(&TileView::startOffset, "start_offset"));
    }
};

/**
 * \brief Base class for shaped types (tensors and tiles)
 *
 * Represents types that have shape dimensions and optional memory references.
 * Both TensorType and TileType inherit from this class.
 */
class ShapedType : public Type {
public:
    DataType dtype_;                  ///< Element data type
    std::vector<ExprPtr> shape_;      ///< Shape dimensions (symbolic or constant)
    std::optional<MemRefPtr> memref_; ///< Optional memory reference (shared pointer)

    /**
     * \brief Create a shaped type without memory reference
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     */
    ShapedType(DataType dtype, std::vector<ExprPtr> shape)
        : dtype_(dtype), shape_(std::move(shape)), memref_(std::nullopt)
    {}

    /**
     * \brief Create a shaped type with constant shape
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     */
    ShapedType(DataType dtype, const std::vector<int64_t>& shape, std::optional<MemRefPtr> memref);

    /**
     * \brief Create a shaped type with optional memory reference (shared_ptr)
     *
     * \param dtype Element data type
     * \param shape Shape dimensions
     * \param memref Optional memory reference (shared pointer)
     */
    ShapedType(DataType dtype, std::vector<ExprPtr> shape, std::optional<MemRefPtr> memref)
        : dtype_(dtype), shape_(std::move(shape)), memref_(std::move(memref))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::ShapedType; }
    [[nodiscard]] std::string TypeName() const override { return "ShapedType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Type::GetFieldDescriptors(), std::make_tuple(
                                             reflection::UsualField(&ShapedType::dtype_, "dtype"),
                                             reflection::UsualField(&ShapedType::shape_, "shape"),
                                             reflection::UsualField(&ShapedType::memref_, "memref")));
    }
};

using ShapedTypePtr = std::shared_ptr<const ShapedType>;

/**
 * \brief Tensor type representation
 *
 * Represents a tensor type with a data type and shape dimensions.
 */
class TensorType : public ShapedType {
public:
    std::vector<ExprPtr> validShape; ///< Valid shape dimensions (symbolic or constant)
    TensorLayout layout;             ///< Tensor layout

    /**
     * \brief Create a tensor type without memory reference
     *
     * \param shape Shape dimensions
     * \param dtype Element data type
     */
    TensorType(std::vector<ExprPtr> shape, DataType dtype) : ShapedType(dtype, std::move(shape)) {}

    /**
     * \brief Create a tensor type with constant shape
     *
     * \param shape Shape dimensions
     * \param dtype Element data type
     */
    TensorType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref)
        : ShapedType(dtype, shape, std::move(memref))
    {}

    /**
     * \brief Create a tensor type with optional memory reference (shared_ptr)
     *
     * \param shape Shape dimensions
     * \param dtype Element data type
     * \param memref Optional memory reference (shared pointer)
     */
    TensorType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
        : ShapedType(dtype, std::move(shape), std::move(memref))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TensorType; }
    [[nodiscard]] std::string TypeName() const override { return "TensorType"; }

    static constexpr auto GetFieldDescriptors()
    {
        auto newFields = std::make_tuple(
            reflection::UsualField(&TensorType::validShape, "valid_shape"),
            reflection::UsualField(&TensorType::layout, "layout"));
        return std::tuple_cat(ShapedType::GetFieldDescriptors(), newFields);
    }
};

using TensorTypePtr = std::shared_ptr<const TensorType>;

/**
 * \brief Tile type representation
 *
 * Represents a tile type (multi-dimensional tensor).
 * Tiles are used for hardware-optimized operations on multi-dimensional data structures.
 * Note: Code generation currently only supports up to 2D tiles.
 */
class TileType : public ShapedType {
public:
    std::optional<TileView> tileView_; ///< Optional tile view information

    /**
     * \brief Create a tile type without memory reference or tile view
     *
     * \param shape Shape dimensions (supports multi-dimensional tensors)
     * \param dtype Element data type
     */
    TileType(std::vector<ExprPtr> shape, DataType dtype) : ShapedType(dtype, std::move(shape)), tileView_(std::nullopt)
    {}

    /**
     * \brief Create a tile type with optional memory reference (shared_ptr)
     *
     * \param shape Shape dimensions (supports multi-dimensional tensors)
     * \param dtype Element data type
     * \param memref Optional memory reference (shared pointer)
     */
    TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tileView_(std::nullopt)
    {
        // No dimension limit at type level; code generation may have constraints
    }

    /**
     * \brief Create a tile type with constant shape
     *
     * \param shape Shape dimensions (supports multi-dimensional tensors)
     * \param dtype Element data type
     * \param memref Optional memory reference (shared pointer)
     * \param tileView Optional tile view information
     */
    TileType(
        const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
        std::optional<TileView> tileView)
        : ShapedType(dtype, shape, std::move(memref)), tileView_(std::move(tileView))
    {}

    /**
     * \brief Create a tile type with optional memory reference and tile view (shared_ptr)
     *
     * \param shape Shape dimensions (supports multi-dimensional tensors)
     * \param dtype Element data type
     * \param memref Optional memory reference (shared pointer)
     * \param tileView Tile view information
     */
    TileType(
        std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref, std::optional<TileView> tileView)
        : ShapedType(dtype, std::move(shape), std::move(memref)), tileView_(std::move(tileView))
    {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TileType; }
    [[nodiscard]] std::string TypeName() const override { return "TileType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            ShapedType::GetFieldDescriptors(),
            std::make_tuple(reflection::UsualField(&TileType::tileView_, "tile_view")));
    }
};

using TileTypePtr = std::shared_ptr<const TileType>;

/**
 * \brief Tuple type representation
 *
 * Represents a tuple type containing multiple types.
 * Tuples are used for multiple return values and structured data.
 */
class TupleType : public Type {
public:
    std::vector<TypePtr> types_; // Types in the tuple

    /**
     * \brief Create a tuple type
     *
     * \param types List of types in the tuple
     */
    explicit TupleType(std::vector<TypePtr> types) : types_(std::move(types)) {}

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TupleType; }
    [[nodiscard]] std::string TypeName() const override { return "TupleType"; }

    static constexpr auto GetFieldDescriptors()
    {
        return std::tuple_cat(
            Type::GetFieldDescriptors(), std::make_tuple(reflection::UsualField(&TupleType::types_, "types")));
    }
};

using TupleTypePtr = std::shared_ptr<const TupleType>;

/**
 * \brief Memory reference type representation
 *
 * Represents a memory reference type for shaped data (tensors and tiles).
 * Used as the type for MemRef variables.
 */
class MemRefType : public Type {
public:
    /**
     * \brief Create a memory reference type
     */
    MemRefType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRefType; }
    [[nodiscard]] std::string TypeName() const override { return "MemRefType"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using MemRefTypePtr = std::shared_ptr<const MemRefType>;

/**
 * \brief Get a shared pointer to the singleton MemRefType instance
 *
 * \return Shared pointer to MemRefType
 */
inline MemRefTypePtr GetMemRefType()
{
    static const auto memrefType = std::make_shared<MemRefType>();
    return memrefType;
}

/**
 * \brief Pointer type for base allocation identity tokens
 *
 * Represents the type of variables returned by tile.alloc / tensor.alloc.
 * A Ptr variable is the allocation identity token that MemRefs reference
 * via their base_ field.
 */
class PtrType : public Type {
public:
    PtrType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::PtrType; }
    [[nodiscard]] std::string TypeName() const override { return "Ptr"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using PtrTypePtr = std::shared_ptr<const PtrType>;

/**
 * @brief Get a shared pointer to the singleton PtrType instance
 *
 * @return Shared pointer to PtrType
 */
inline PtrTypePtr GetPtrType()
{
    static const auto ptr_type = std::make_shared<PtrType>();
    return ptr_type;
}

/**
 * \brief Token type representation
 *
 * Represents an opaque token value used for side-effect ordering.
 * Tokens carry no data and are only used to establish dependencies
 * between operations.
 */
class TokenType : public Type {
public:
    TokenType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TokenType; }
    [[nodiscard]] std::string TypeName() const override { return "Token"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using TokenTypePtr = std::shared_ptr<const TokenType>;

/**
 * @brief Get a shared pointer to the singleton TokenType instance
 *
 * @return Shared pointer to TokenType
 */
inline TokenTypePtr GetTokenType()
{
    static const auto token_type = std::make_shared<TokenType>();
    return token_type;
}

/**
 * \brief Logical tensor type representation
 *
 * Represents a logical tensor with dtype and shape, without memory allocation info.
 * Used for tensor values in the PIL/IR layer before memory planning.
 */
class LogicalTensorType : public Type {
public:
    LogicalTensorType() = default;

    [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::LogicalTensorType; }
    [[nodiscard]] std::string TypeName() const override { return "LogicalTensorType"; }

    static constexpr auto GetFieldDescriptors() { return Type::GetFieldDescriptors(); }
};

using LogicalTensorTypePtr = std::shared_ptr<const LogicalTensorType>;

inline LogicalTensorTypePtr GetLogicalTensorType()
{
    static const auto type = std::make_shared<LogicalTensorType>();
    return type;
}

} // namespace ir
} // namespace pypto
