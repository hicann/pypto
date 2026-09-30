/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// COMPILE-ONLY mock of the vendored pto_custom_op.h.
//
// The generated thin executor ``#include "pto_custom_op.h"``; for unit compile tests we resolve
// it to this stub (fixtures/ is on the -I path) instead of the real header, which would drag in
// the GE custom_op headers, Python.h and <nlohmann/json.hpp>. It mirrors the real base's public interface
// with minimal GE/gert stub types so the generated subclass type-checks:
//   * Compile()/DeclareLaunchArgs() are non-pure no-ops (they live in the real .cpp, not built
//     here);
//   * the per-op hook GetCompileModuleStem() stays pure, so the generated subclass must
//     override it;
//   * REG_AUTO_MAPPING_OP is a no-op.
// Keep in sync with vendors/pto_custom_op.h when the base interface changes.
#ifndef PYPTO_PTO_CUSTOM_OP_H
#define PYPTO_PTO_CUSTOM_OP_H

// The real base routes PTO_CUSTOM_LOGD through slog (dlog_debug); for the compile-only test it's a
// no-op so the generated executor's error-path logs compile without the slog backend.
#ifndef PTO_CUSTOM_LOGD
#define PTO_CUSTOM_LOGD(fmt, ...) ((void)0)
#endif

// PTO_CUSTOM_LOGE (dlog_error in the real base) carries the generated infer wrappers' hard-failure
// diagnostics. The thin executor includes this header rather than codegen's slog preamble.
#ifndef PTO_CUSTOM_LOGE
#define PTO_CUSTOM_LOGE(fmt, ...) ((void)0)
#endif

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ge {
using graphStatus = uint32_t;
constexpr graphStatus GRAPH_SUCCESS = 0u;
constexpr graphStatus GRAPH_FAILED = 1u;
enum DataType { DT_FLOAT16 = 0, DT_FLOAT = 1, DT_INT32 = 2 };
} // namespace ge

namespace gert {
class Shape {
public:
    // Public API on the real gert::Shape: the fixed capacity of its dims_ array, which the generated
    // InferShape rank guard compares against.
    static constexpr size_t kMaxDimNum = 25;

    size_t GetDimNum() const { return dim_num_; }
    int64_t GetDim(size_t) const { return 0; }
    void SetDimNum(size_t n) { dim_num_ = n; }
    int64_t& operator[](size_t i) { return dims_[i % 8]; }
    int64_t operator[](size_t i) const { return dims_[i % 8]; }

private:
    size_t dim_num_ = 0;
    int64_t dims_[8] = {0};
};
class Tensor {
public:
    const void* GetAddr() const { return nullptr; }
    const Shape& GetOriginShape() const
    {
        static Shape s;
        return s;
    }
    ge::DataType GetDataType() const { return ge::DT_FLOAT16; }
};
struct InputAddr {
    uint32_t index;
    const void* addr;
};
struct OutputAddr {
    uint32_t index;
    const void* addr;
};
struct WorkspaceAddr {
    uint32_t index;
    const void* addr;
};
class AnnotatedKernelArgs {
public:
    AnnotatedKernelArgs() {}
    template <typename... Args>
    explicit AnnotatedKernelArgs(Args&&...)
    {}
    ge::graphStatus AppendArg(const InputAddr&) { return ge::GRAPH_SUCCESS; }
    ge::graphStatus AppendArg(const OutputAddr&) { return ge::GRAPH_SUCCESS; }
    ge::graphStatus AppendArg(const WorkspaceAddr&) { return ge::GRAPH_SUCCESS; }
    ge::graphStatus AppendArg(uint64_t) { return ge::GRAPH_SUCCESS; }
};
// The continuous-int64 vector gert::RuntimeAttrs::GetListInt returns; GetData()/GetSize() mirror the
// real surface the generated ListInt ExtractAttrs / InferShape reads. Stub is empty.
class TypedContinuousVectorInt64 {
public:
    const int64_t* GetData() const { return nullptr; }
    size_t GetSize() const { return 0; }
};
// Positional attr accessors the generated ExtractAttrs / InferShape read (GetInt/GetFloat/GetStr/
// GetListInt at the declaration index). Stubs return nullptr so the executor's default/null-guard branch
// is taken. Keep in sync with the real gert RuntimeAttrs surface.
class RuntimeAttrs {
public:
    const int64_t* GetInt(size_t) const { return nullptr; }
    const float* GetFloat(size_t) const { return nullptr; }
    const char* GetStr(size_t) const { return nullptr; }
    const TypedContinuousVectorInt64* GetListInt(size_t) const { return nullptr; }
};
// ge 20260717 annotated-args launch info: kernel meta passed to AddLaunch (no builder/task). The
// real struct lives in annotated_args_context.h; the runtime symbols live in liblowering.so.
struct AnnotatedKernelLaunchInfo {
    const char* kernel_name = nullptr;
    const void* kernel_bin = nullptr;
    size_t kernel_bin_size = 0u;
    uint32_t block_dim = 0u;
    uint32_t stream_id = 0u;
};
class AnnotatedArgsContext {
public:
    size_t GetComputeNodeInputNum() const { return 0; }
    size_t GetComputeNodeOutputNum() const { return 0; }
    const Tensor* GetInputTensor(size_t) const
    {
        static Tensor t;
        return &t;
    }
    const Tensor* GetOutputTensor(size_t) const
    {
        static Tensor t;
        return &t;
    }
    const RuntimeAttrs* GetAttrs() const { return nullptr; }
    WorkspaceAddr MallocWorkSpace(size_t) { return WorkspaceAddr{0u, nullptr}; }
    uint32_t GetStreamId() const { return 0u; }
    ge::graphStatus AddLaunch(const AnnotatedKernelLaunchInfo&, AnnotatedKernelArgs&&) { return ge::GRAPH_SUCCESS; }
};
class OpCompileContext {
public:
    size_t GetComputeNodeInputNum() const { return 0; }
    const Tensor* GetInputTensor(size_t) const
    {
        static Tensor t;
        return &t;
    }
    const RuntimeAttrs* GetAttrs() const { return nullptr; }
};
// Compile-time shape/dtype inference contexts (ge::ShapeInferOp member methods use these).
class InferShapeContext {
public:
    const Shape* GetInputShape(size_t) const
    {
        static Shape s;
        return &s;
    }
    Shape* GetOutputShape(size_t)
    {
        static Shape s;
        return &s;
    }
    const RuntimeAttrs* GetAttrs() const { return nullptr; }
};
class InferDataTypeContext {
public:
    ge::DataType GetInputDataType(size_t) const { return ge::DT_FLOAT16; }
    void SetOutputDataType(size_t, ge::DataType) {}
    const RuntimeAttrs* GetAttrs() const { return nullptr; }
};
} // namespace gert

struct CompiledInfo {
    std::string binPath;
    uint32_t blockDim;
    std::string kernelName;
    size_t workspaceSize;
    std::vector<uint8_t> kernelBuffer;
};

// Mirrors ge::ShapeInferOp (graph/custom_op.h): pure-virtual InferShape/InferDataType the executor
// overrides; GE resolves them via CustomOpFactory + dynamic_cast<ShapeInferOp*> at compile time.
namespace ge {
class ShapeInferOp {
public:
    virtual ~ShapeInferOp() {}
    virtual graphStatus InferShape(gert::InferShapeContext*) = 0;
    virtual graphStatus InferDataType(gert::InferDataTypeContext*) = 0;
};
} // namespace ge

// Forward-declare PyObject so ImportKernelModule's PyObject* signature parses without <Python.h> here
// (this mock header is included before the pybind/Python headers in the generated executor TU).
extern "C" {
struct _object;
typedef struct _object PyObject;
}

class PtoCustomOp {
public:
    virtual ~PtoCustomOp() {}
    virtual ge::graphStatus Compile(gert::OpCompileContext*) { return ge::GRAPH_SUCCESS; }
    virtual ge::graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext&) { return ge::GRAPH_SUCCESS; }
    virtual std::string GetCompileModuleStem() const = 0;
    // Keep in sync with vendors/pto_custom_op.h: basename of the shipped dev-editable kernel snippet
    // (the generated subclass overrides this); empty default => no shipped file.
    virtual std::string GetKernelPyBasename() const = 0;
    virtual void ExtractAttrs(const gert::RuntimeAttrs*, std::map<std::string, std::string>&) const {}
    // Statics that the generated infer wrappers / ListInt ExtractAttrs reference (declarations only —
    // -fsyntax-only needs no definitions). Keep in sync with vendors/pto_custom_op.h.
    static std::string ResolveKernelPy(const std::string& basename);
    static PyObject* ImportKernelModule(const std::string& stem, const std::string& basename);
    static std::string ListIntToJsonStr(const std::vector<int64_t>& vals);
};

#define REG_AUTO_MAPPING_OP(cls)

#endif // PYPTO_PTO_CUSTOM_OP_H
