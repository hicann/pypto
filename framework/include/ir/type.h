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
 * \file type.h
 * \brief
 */

#pragma once

#include "ir/object.h"

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

namespace pto {

// Enumeration for scalar data types.
// This enum must stay aligned with the Python side DataType enum (pypto_impl.DataType).
enum class DataType {
    INT4 = 0,
    INT8 = 1,
    INT16 = 2,
    INT32 = 3,
    INT64 = 4,

    FP8_E4M3FN = 5,
    FP16 = 6,
    FP32 = 7,
    BF16 = 8,
    HF4 = 9,
    HF8 = 10,

    UINT8 = 11,
    UINT16 = 12,
    UINT32 = 13,
    UINT64 = 14,

    BOOL = 15,
    FP64 = 16,
    FP8_E5M2 = 17,

    UNKNOWN
};

struct DTypeInfo {
    DataType dtype;
    const char *name;
    int bits;
    int bytes;
    bool isFloat;
    bool isSigned;
    bool isUnsigned;
};

// Base class for all types in PTO-IR.
// Type represents the structure of data, separate from specific values.
class Type {
public:
    explicit Type(DataType dataType) : dataType_(dataType) {}
    virtual ~Type() = default;

    DataType GetDataType() const { return dataType_; }

    // Get the size of the data type in bytes (e.g., FP32 -> 4 bytes).
    static uint64_t GetDataTypeSize(DataType dataType);

    // Get the size of the data type in bytes for this type instance.
    uint64_t GetDataTypeSize() const { return GetDataTypeSize(dataType_); }

    // Get the total size of this type in bytes.
    // For ScalarType: returns GetDataTypeSize()
    // For TileType/TensorType: returns GetDataTypeSize() * product of shape dimensions
    virtual uint64_t GetTypeSize() const = 0;

    // Pretty-print the type.
    virtual void Print(std::ostream& os) const = 0;

protected:
    DataType dataType_;
};

using TypePtr = std::shared_ptr<Type>;

// Scalar type: represents a single scalar value type (e.g., int32, fp32).
class ScalarType : public Type {
public:
    explicit ScalarType(DataType dataType) : Type(dataType) {}

    uint64_t GetTypeSize() const override { return GetDataTypeSize(); }
    void Print(std::ostream& os) const override;
};

using ScalarTypePtr = std::shared_ptr<ScalarType>;

// Tile type: represents a tile type with static shape and element type.
// TileType represents the type of a tile with a specific (shape, dataType) combination.
class TileType : public Type {
public:
    TileType(DataType elementType, const std::vector<int64_t>& shape)
        : Type(elementType), shape_(shape) {}

    const std::vector<int64_t>& GetShape() const { return shape_; }

    uint64_t GetTypeSize() const override;
    void Print(std::ostream& os) const override;

private:
    std::vector<int64_t> shape_;  // Static shape dimensions
};

using TileTypePtr = std::shared_ptr<TileType>;

// Tensor type: represents a tensor type with shape and element type.
// TensorType represents the type of a tensor with a specific (shape, dataType) combination.
class TensorType : public Type {
public:
    explicit TensorType(DataType dataType) : Type(dataType) {}

    uint64_t GetTypeSize() const override { return GetDataTypeSize(); }

    void Print(std::ostream& os) const override;
};

using TensorTypePtr = std::shared_ptr<TensorType>;

} // namespace pto

