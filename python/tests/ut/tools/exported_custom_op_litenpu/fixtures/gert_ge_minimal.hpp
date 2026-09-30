/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Minimal gert/ge stubs for compiling pypto export codegen tests only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace ge {

/// Matches dtype tokens emitted by export codegen (`ge::DT_FLOAT`, etc.).
enum DataType {
    DT_FLOAT = 0,           // float type
    DT_FLOAT16 = 1,         // fp16 type
    DT_INT8 = 2,            // int8 type
    DT_INT16 = 6,           // int16 type
    DT_UINT16 = 7,          // uint16 type
    DT_UINT8 = 4,           // uint8 type
    DT_INT32 = 3,           // int32 type
    DT_INT64 = 9,           // int64 type
    DT_UINT32 = 8,          // unsigned int32
    DT_UINT64 = 10,         // unsigned int64
    DT_BOOL = 12,           // bool type
    DT_DOUBLE = 11,         // double type
    DT_STRING = 13,         // string type
    DT_DUAL_SUB_INT8 = 14,  // dual output int8 type
    DT_DUAL_SUB_UINT8 = 15, // dual output uint8 type
    DT_COMPLEX64 = 16,      // complex64 type
    DT_COMPLEX128 = 17,     // complex128 type
    DT_QINT8 = 18,          // qint8 type
    DT_QINT16 = 19,         // qint16 type
    DT_QINT32 = 20,         // qint32 type
    DT_QUINT8 = 21,         // quint8 type
    DT_QUINT16 = 22,        // quint16 type
    DT_RESOURCE = 23,       // resource type
    DT_STRING_REF = 24,     // string ref type
    DT_DUAL = 25,           // dual output type
    DT_BF16 = 27,           // bf16 type
    DT_UNDEFINED = 28,      // Used to indicate a DataType field has not been set.
    DT_INT4 = 29,           // int4 type
    DT_UINT1 = 30,          // uint1 type
    DT_INT2 = 31,           // int2 type
    DT_UINT2 = 32,          // uint2 type
    DT_COMPLEX32 = 33,      // complex32 type
    DT_HIFLOAT8 = 34,       // hifloat8 type
};

// Matches real Ascend ``graph/ge_error_codes.h``: ``using graphStatus = uint32_t;``.
using graphStatus = uint32_t;
inline constexpr graphStatus GRAPH_SUCCESS = 0;
inline constexpr graphStatus GRAPH_FAILED = 1;

// Minimal ``ge::AscendString`` (graph/ascend_string.h): the executor's Compile() uses it
// for the ``ge.socVersion`` GetOption round-trip.
class AscendString {
public:
    AscendString() = default;
    AscendString(const char* s) : value_(s ? s : "") {}
    const char* GetString() const { return value_.c_str(); }

private:
    std::string value_;
};

// Minimal stand-in for ``ge::GetSizeByDataType`` from ``graph/types.h`` — only the
// byte-sized dtypes pypto exposes need to round-trip; everything else returns -1
// so a ``< 0`` guard still fires in negative tests.
inline int GetSizeByDataType(DataType dt)
{
    switch (dt) {
        case DT_FLOAT:
            return 4;
        case DT_FLOAT16:
            return 2;
        case DT_INT8:
            return 1;
        case DT_INT16:
            return 2;
        case DT_UINT16:
            return 2;
        case DT_UINT8:
            return 1;
        case DT_INT32:
            return 4;
        case DT_INT64:
            return 8;
        case DT_UINT32:
            return 4;
        case DT_UINT64:
            return 8;
        case DT_BOOL:
            return 1;
        case DT_DOUBLE:
            return 8;
        case DT_COMPLEX64:
            return 8;
        case DT_COMPLEX128:
            return 16;
        case DT_QINT8:
            return 1;
        case DT_QINT16:
            return 2;
        case DT_QINT32:
            return 4;
        case DT_QUINT8:
            return 1;
        case DT_QUINT16:
            return 2;
        case DT_BF16:
            return 2;
        case DT_COMPLEX32:
            return 4;
        default:
            return -1;
    }
}

} // namespace ge

namespace gert {

struct Shape {
    // Public API on the real gert::Shape: the fixed capacity of its dims_ array, which the generated
    // InferShape rank guard compares against.
    static constexpr size_t kMaxDimNum = 25;

    std::vector<int64_t> dims_;

    Shape() = default;

    explicit Shape(std::initializer_list<int64_t> il) : dims_(il) {}

    void SetDimNum(const size_t dim_num) { dims_.resize(dim_num); }

    size_t GetDimNum() const { return dims_.size(); }

    int64_t GetDim(size_t i) const { return dims_[i]; }

    int64_t& operator[](size_t i) { return dims_[i]; }

    const int64_t& operator[](const size_t i) const { return dims_[i]; }
};

/// Storage shape view; ``GetShape()`` returns the logical ``Shape`` (dims).
class StorageShape {
public:
    StorageShape() : shape_({2, 2}) {}

    explicit StorageShape(Shape shape) : shape_(std::move(shape)) {}

    const Shape& GetShape() const { return shape_; }

private:
    Shape shape_;
};

class MockSinkableOpExecutionContext;

/// Minimal tensor stub for sinkable executor compile tests.
class Tensor {
public:
    Tensor() = default;

    const void* GetAddr() const { return addr_; }

    void* GetAddr() { return addr_; }

    ge::DataType GetDataType() const { return dtype_; }

    const StorageShape& GetShape() const { return storage_shape_; }

    // Logical shape (dims) — used by the executor's Compile() to read the runtime
    // input shape for the JIT build.
    const Shape& GetOriginShape() const { return storage_shape_.GetShape(); }

    // Byte size — used for the last-output end-pointer trailing slot.
    size_t GetSize() const { return size_; }

private:
    friend class MockSinkableOpExecutionContext;
    friend class OpCompileContext;

    void* addr_{};
    ge::DataType dtype_{ge::DT_FLOAT16};
    StorageShape storage_shape_{};
    size_t size_{256};
};

class InferShapeContext {
public:
    void SetInput(int idx, std::initializer_list<int64_t> dims)
    {
        const auto u = static_cast<size_t>(idx);
        if (inputs_.size() <= u) {
            inputs_.resize(u + 1);
        }
        inputs_[u] = Shape(dims);
    }

    // nullptr for an index with no input set, mirroring the real header's documented contract, so the
    // generated null guard is exercisable rather than permanently dead.
    const Shape* GetInputShape(int i) const
    {
        const auto u = static_cast<size_t>(i);
        return u < inputs_.size() ? &inputs_[u] : nullptr;
    }

    Shape* GetOutputShape(int i)
    {
        const auto u = static_cast<size_t>(i);
        if (outputs_.size() <= u) {
            outputs_.resize(u + 1);
        }
        return &outputs_[u];
    }

    const Shape& OutputShape(size_t i) const { return outputs_.at(i); }

private:
    std::vector<Shape> inputs_;
    std::vector<Shape> outputs_;
};

class InferDataTypeContext {
public:
    InferDataTypeContext() { input_dtypes_.assign(8, ge::DT_FLOAT16); }

    void SetInputDataType(int i, ge::DataType dt)
    {
        const auto u = static_cast<size_t>(i);
        if (input_dtypes_.size() <= u) {
            input_dtypes_.resize(u + 1, ge::DT_FLOAT16);
        }
        input_dtypes_[u] = dt;
    }

    ge::DataType GetInputDataType(int i) const { return input_dtypes_.at(static_cast<size_t>(i)); }

    void SetOutputDataType(int idx, ge::DataType dtype)
    {
        const auto u = static_cast<size_t>(idx);
        if (output_dtypes_.size() <= u) {
            output_dtypes_.resize(u + 1);
        }
        output_dtypes_[u] = dtype;
    }

    ge::DataType OutputDataType(size_t i) const { return output_dtypes_.at(i); }

private:
    std::vector<ge::DataType> input_dtypes_;
    std::vector<ge::DataType> output_dtypes_;
};

} // namespace gert
