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
 * \file value.h
 * \brief
 */

#pragma once

#include "ir/utils.h"
#include "ir/type.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <cstdint>
#include <vector>

namespace pto {

// Enumeration for data type categories (Scalar/Tensor/Tile).
enum class ValueKind {
    Scalar,
    Tile,
    Tensor
};

// Enumeration for scalar value kinds: immediate, symbolic.
enum class ScalarValueKind {
    Immediate,          // Immediate value known at compile time (immediate value is a constant value)
    Symbolic,           // Symbolic value known at runtime (symbolic value is a variable name)
};

// Enumeration for cast modes.
enum class CastMode {
    CAST_NONE,       // No rounding mode specified
    CAST_RINT,       // Round to nearest integer (ties to even)
    CAST_ROUND,      // Round to nearest integer
    CAST_FLOOR,      // Round down to nearest integer
    CAST_CEIL,       // Round up to nearest integer
    CAST_TRUNC,      // Truncate towards zero
    CAST_ODD,        // Round to nearest odd integer
};

// Base class for all data types in PTO-IR.
class Value : public Object {
public:
    explicit Value(ValueKind kind, TypePtr type, std::string name="")
        : Object(ObjectType::Value, name), valueKind_(kind), type_(type) {}
    virtual ~Value() = default;

    const std::string GetSSAName() const {
        if (name_.empty()) {
            return "%" + std::to_string(id_);
        }
        // If tensor has a name, return it directly without adding numeric suffix
        return GetPrefixedName() + "_" + std::to_string(id_);
    }
    ObjectType GetObjectType() const override { return ObjectType::Value; }

    ValueKind GetValueKind() const { return valueKind_; }
    TypePtr GetType() const { return type_; }
    DataType GetDataType() const { return type_ ? type_->GetDataType() : DataType::UNKNOWN; }

    // Pretty-print the type with the given indentation.
    virtual void Print(std::ostream& os, int indent = 0) const = 0;

protected:
    ValueKind valueKind_;
    TypePtr type_;
};

using ValuePtr = std::shared_ptr<Value>;
using ValuePtrs = std::vector<ValuePtr>;

using ImmediateType = std::variant<bool, int, int64_t, uint64_t, double>;

// Scalar type: bool, int4, int8, int16, int32, int64, fp8, fp16, bf16, fp32, fp64
class ScalarValue : public Value {
public:
    explicit ScalarValue(DataType type, std::string name="", ScalarValueKind valueKind = ScalarValueKind::Symbolic)
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(type), name), valueKind_(valueKind), immediateValue_(int64_t{0}) {}

    explicit ScalarValue(std::string typeName, std::string name="", ScalarValueKind valueKind = ScalarValueKind::Symbolic)
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(StringToValueType(typeName)), name), valueKind_(valueKind), immediateValue_(int64_t{0}) {}

    explicit ScalarValue(DataType type, std::string name, ScalarValueKind valueKind, ImmediateType constantVal)
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(type), name), valueKind_(valueKind), immediateValue_(constantVal) {}

    // Constant value constructors - DataType is inferred from value type
    explicit ScalarValue(bool value, std::string name="")
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(DataType::BOOL), name), valueKind_(ScalarValueKind::Immediate), immediateValue_(value) {}

    explicit ScalarValue(int value, std::string name="")
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(DataType::INT32), name), valueKind_(ScalarValueKind::Immediate), immediateValue_(value) {}

    explicit ScalarValue(int64_t value, std::string name="")
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(DataType::INT64), name), valueKind_(ScalarValueKind::Immediate), immediateValue_(value) {}

    explicit ScalarValue(double value, std::string name="")
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(DataType::FP64), name), valueKind_(ScalarValueKind::Immediate), immediateValue_(value) {}

    explicit ScalarValue(uint64_t value, std::string name="")
        : Value(ValueKind::Scalar, std::make_shared<ScalarType>(DataType::UINT64), name), valueKind_(ScalarValueKind::Immediate), immediateValue_(value) {}

    ScalarValue() : Value(ValueKind::Scalar, std::make_shared<ScalarType>(DataType::UNKNOWN)) {}

    ScalarValueKind GetScalarValueKind() const { return valueKind_; }

    // Get immediate value (only valid when valueKind_ == Immediate)
    ImmediateType GetImmediateValue() const {
        return immediateValue_;
    }

    bool HasImmediateValue() const { return valueKind_ == ScalarValueKind::Immediate; }

    // Get immediate value as int64_t. Only valid when HasImmediateValue() is true.
    int64_t GetInt64Value() const;

    void Print(std::ostream& os, int indent = 0) const override;

private:
    ScalarValueKind valueKind_;     // Kind of scalar value: immediate, symbolic
    ImmediateType immediateValue_;   // Immediate value storage
};

using ScalarValuePtr = std::shared_ptr<ScalarValue>;

enum class MemSpaceKind {
    UNKNOWN,
    DDR,
    L2,
    UB,
    L1,
    L0A,
    L0B,
    L0C,
    REG,
    SHMEM,
};

class Memory : public Object {
public:
    Memory(uint64_t byteSize) : Object(ObjectType::Memory), byteSize_(byteSize),
         space_(MemSpaceKind::UNKNOWN) {}

    uint64_t GetSize() const { return byteSize_; }
    MemSpaceKind GetSpace() const { return space_; }
    uint64_t GetAddr() const { return addr_; }

    void SetSize(const uint64_t newSize) { byteSize_ = newSize; }
    void SetSpace(const MemSpaceKind kind) { space_ = kind; }
    void SetAddr(const uint64_t newAddr) { addr_ = newAddr; }

private:
    uint64_t byteSize_;
    MemSpaceKind space_;
    uint64_t addr_;
};

// Tile: tile<validshape, tile_shapes, strides, start_offset, elem_type, memory>
class TileValue : public Value {
public:
    TileValue(std::string name, std::vector<ScalarValuePtr> validShapes, std::vector<uint64_t> shape,
            std::vector<uint64_t> strides, ScalarValuePtr startOffset, DataType elementType,
            std::shared_ptr<Memory> mem=nullptr)
        : Value(ValueKind::Tile, std::make_shared<TileType>(elementType, shape), name),
          validShapes_(validShapes),
          strides_(strides),
          startOffset_(startOffset),
          mem_(mem) {}

    TileValue(std::vector<uint64_t> shape, DataType elementType,
         std::vector<ScalarValuePtr> validShapes, std::string name="")
        : Value(ValueKind::Tile, std::make_shared<TileType>(elementType, shape), name),
          validShapes_(validShapes) { }

    TileValue(std::vector<uint64_t> shape, DataType elementType,
         std::string name="")
        : Value(ValueKind::Tile, std::make_shared<TileType>(elementType, shape), name) {
        validShapes_.reserve(shape.size());
        for (size_t i = 0; i < shape.size(); i++) {
            validShapes_.emplace_back(std::make_shared<ScalarValue>(shape[i]));
        }
    }

    const std::vector<ScalarValuePtr>& GetValidShape() const { return validShapes_; }

    // Get shape from Type system (TileType)
    const std::vector<uint64_t>& GetShape() const {
        auto tileType = std::dynamic_pointer_cast<TileType>(GetType());
        if (!tileType) {
            throw std::runtime_error("Tile type is not TileType");
        }
        return tileType->GetShape();
    }

    const std::vector<uint64_t>& GetStrides() const { return strides_; }
    ScalarValuePtr GetStartOffset() const { return *startOffset_; }
    const std::shared_ptr<Memory> GetMemory() const { return mem_; }

    void SetShape(const std::vector<uint64_t>& newShape) {
        // Update Type with new shape
        type_ = std::make_shared<TileType>(GetDataType(), newShape);
    }
    void SetStrides(const std::vector<uint64_t>& newStrides) { strides_ = newStrides; }
    void SetStartOffset(const ScalarValuePtr newStartOffset) { startOffset_ = newStartOffset; }
    void SetMemory(const std::shared_ptr<Memory> newMem) { mem_ = newMem; }

    void Print(std::ostream& os, int indent = 0) const override;

private:
    std::vector<ScalarValuePtr> validShapes_;
    std::vector<uint64_t> strides_;
    std::optional<ScalarValuePtr> startOffset_;
    std::shared_ptr<Memory> mem_;
};

using TileValuePtr = std::shared_ptr<TileValue>;

// Enumeration for tile operation formats.
enum class TileOpFormat {
    TILEOP_ND = 0,  // Dense format
    TILEOP_NZ = 1   // Non-zero (sparse) format
};

class TensorValue : public Value {
public:
    // Construct tensor from a vector of Scalar dimensions.
    TensorValue(const std::vector<ScalarValuePtr>& shape, DataType type, std::string name="",
            TileOpFormat format = TileOpFormat::TILEOP_ND) :
        Value(ValueKind::Tensor, std::make_shared<TensorType>(type), name), shape_(shape), format_(format) {}

    // Convenience constructor for static integer shapes.
    // This is mainly used by Python bindings where shapes are passed as ints.
    // The parameter order is aligned with Python Tensor(dtype, shape, name, format).
    TensorValue(DataType type, const std::vector<uint64_t>& shape, std::string name="",
            TileOpFormat format = TileOpFormat::TILEOP_ND) :
        Value(ValueKind::Tensor, std::make_shared<TensorType>(type), name), format_(format) {

        shape_.reserve(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_.emplace_back(std::make_shared<ScalarValue>(shape[i]));
        }
    }

    const std::vector<ScalarValuePtr>& GetShape() const { return shape_; }

    TileOpFormat GetFormat() const { return format_; }
    void SetFormat(TileOpFormat format) { format_ = format; }

    void Print(std::ostream& os, int indent) const override;
private:
    std::vector<ScalarValuePtr> shape_;
    TileOpFormat format_;
};

static inline std::vector<ValuePtr> CastScalarToValue(const std::vector<ScalarValuePtr> &scalarList) {
    std::vector<ValuePtr> valueList;
    for (auto scalar : scalarList) {
        valueList.emplace_back(std::static_pointer_cast<Value>(scalar));
    }
    return valueList;
}

} // namespace pto
