/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// pypto shared custom-op base class (PtoCustomOp).
//
// The per-op executor is a thin subclass; this base owns the
// Compile/DeclareLaunchArgs orchestration, the compile cache, shape/dtype/attr
// extraction and the kernel launch.
//
// Differences vs the reference `pto_custom_op.h`:
//   * pypto SHIPS the per-op kernel Python as `op_kernel/<stem>.py` and imports it
//     BY PATH: Compile() and the infer wrappers resolve+import it via ImportKernelModule()
//     (ENV-first over the ASCEND_CUSTOM_OPP_PATH entries — <entry>/op_kernel/<stem>.py — with a
//     dladdr upward-walk fallback). That .py is the SINGLE SOURCE OF TRUTH and is REQUIRED: an
//     unresolved/absent file is a FATAL error. It does NOT use the reference's
//     GetPythonModulePath/GetVendorPath/GetPythonModuleName (no PyImport_ImportModule BY NAME).
//   * DeclareLaunchArgs() builds the launch args generically from the runtime
//     input/output counts (AnnotatedKernelArgs::AppendArg), so no per-op launch
//     code is generated (ge 20260717 annotated-args launch model).
#ifndef PYPTO_PTO_CUSTOM_OP_H
#define PYPTO_PTO_CUSTOM_OP_H

// Logging routed through the Ascend slog backend, which already prepends [file:line]. Link dep:
// unified_dlog.
//   * PTO_CUSTOM_LOGD (dlog_debug) carries the enter/exit traces: level-gated, emits only at
//     ASCEND_GLOBAL_LOG_LEVEL=0. Same macro the generated OpDef/plugin TUs use, so trace logging is
//     uniform across the .so.
//   * PTO_CUSTOM_LOGE (dlog_error) carries the ERROR-class diagnostics, so a GRAPH_FAILED returned to
//     GE is explainable from a production log with no rerun at debug level.
#include "toolchain/slog.h"
#include "base/log_types.h"
#ifndef PTO_CUSTOM_LOGD
#define PTO_CUSTOM_LOGD(fmt, ...) dlog_debug(OP, fmt, ##__VA_ARGS__)
#endif
#ifndef PTO_CUSTOM_LOGE
#define PTO_CUSTOM_LOGE(fmt, ...) dlog_error(OP, fmt, ##__VA_ARGS__)
#endif

#include "graph/custom_op.h"
#include "exe_graph/runtime/annotated_args_context.h"
#include "exe_graph/runtime/op_compile_context.h"
#include "exe_graph/runtime/runtime_attrs.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Forward-declare PyObject so ImportKernelModule's signature parses without pulling in <Python.h>
// here (this header is included before pybind/Python headers in the generated executor TU).
extern "C" {
struct _object;
typedef struct _object PyObject;
}

// Launch metadata parsed from the kernel's JSON sidecar + the kernel binary bytes.
struct CompiledInfo {
    std::string binPath;
    uint32_t blockDim;
    std::string kernelName;
    size_t workspaceSize;
    std::vector<uint8_t> kernelBuffer;
};

// Inherits AnnotatedArgsOp (not ArgsUpdater): GE's GetArgsRefreshStrategy checks ArgsUpdater FIRST and, if
// present, pins the op to the kUpdateCallback strategy — which the mobile/端侧 OMC path (Kirin9030) rejects
// with "does not implement AnnotatedArgsOp" (custom_ops_kernel_builder.cc). AnnotatedArgsOp's annotated-args
// launch is itself address-refreshable, so GE refreshes I/O without ArgsUpdater (the GE reference add_custom_pto
// omits it too).
class PtoCustomOp : public ge::CompilableOp, public ge::AnnotatedArgsOp {
public:
    PtoCustomOp() {}
    virtual ~PtoCustomOp() {}

    // GE callbacks (defined in pto_custom_op.cpp).
    ge::graphStatus Compile(gert::OpCompileContext* ctx) override;
    ge::graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext& ctx) override;

    // --- per-op hooks (the generated subclass provides these) ---------------

    // Unique stem for the imported module (avoids collisions when several ops live
    // in one .so), e.g. ``pypto_compile_PyptoCustomOpAdd``.
    virtual std::string GetCompileModuleStem() const = 0;

    // Basename of the shipped dev-editable kernel snippet, e.g. ``add.py`` (deployed as
    // ``op_kernel/<stem>.py``). The generated subclass overrides this with the exact codegen basename.
    // The file is REQUIRED: ImportKernelModule() resolves it via the ASCEND_CUSTOM_OPP_PATH entries
    // (<entry>/op_kernel/<basename>) with a dladdr upward-walk fallback and loads it by path (dev edits
    // win); an unresolved/absent file is a FATAL error.
    virtual std::string GetKernelPyBasename() const = 0;

    // Op attributes → {name: value-string}, fed to the cache key and the Python
    // compile call. Default: no attrs. Overridden by the generated subclass when the
    // op declares any.
    virtual void ExtractAttrs(const gert::RuntimeAttrs* attrs, std::map<std::string, std::string>& out) const
    {
        (void)attrs;
        (void)out;
    }

    // --- shared helpers (templated over the GE context type) ----------------

    template <typename Ctx>
    static void ExtractInputShapes(Ctx* ctx, std::vector<std::string>& shape_strs)
    {
        shape_strs.clear();
        size_t n = ctx->GetComputeNodeInputNum();
        for (size_t i = 0; i < n; i++) {
            const gert::Tensor* tensor = ctx->GetInputTensor(i);
            if (tensor == nullptr)
                continue;
            const gert::Shape& shape = tensor->GetOriginShape();
            size_t dims = shape.GetDimNum();
            std::string str = "(";
            for (size_t j = 0; j < dims; j++) {
                if (j > 0)
                    str += ",";
                str += std::to_string(shape.GetDim(j));
            }
            str += ")";
            shape_strs.push_back(str);
        }
    }

    template <typename Ctx>
    static void ExtractInputDtypes(Ctx* ctx, std::vector<std::string>& dtype_strs)
    {
        dtype_strs.clear();
        size_t n = ctx->GetComputeNodeInputNum();
        for (size_t i = 0; i < n; i++) {
            const gert::Tensor* tensor = ctx->GetInputTensor(i);
            if (tensor == nullptr)
                continue;
            const char* str = DataTypeToStr(tensor->GetDataType());
            // Unmapped dtype: push a marker (not a silent "float16") so the cache key is distinct and
            // the Python compile entry raises clearly. DataTypeToStr logs the offending enum.
            dtype_strs.push_back(str ? str : "unsupported");
        }
    }

    static std::string ListIntToJsonStr(const std::vector<int64_t>& vals);
    static const char* DataTypeToStr(ge::DataType dtype);

    // Resolve THIS .so's on-disk directory via dladdr on a base-class symbol (GE dlopened the .so
    // from opp_root, opp_root/op_proto, or opp_root/framework/onnx). Kept only for the dladdr fallback
    // in ResolveKernelPy(). Empty on failure.
    static std::string SelfSoDir();

    // Resolve ``op_kernel/<basename>`` to an absolute path, else "" if not found. ENV-FIRST: probe each
    // ASCEND_CUSTOM_OPP_PATH entry's <entry>/op_kernel/<basename> (the var is a ':'-separated LIST).
    // Fallback: dladdr SelfSoDir() + a bounded upward walk (<so_dir>, /.., /../..) covering the
    // root/op_proto/framework.onnx .so copies. Static so the this-less free infer wrappers can reuse it.
    static std::string ResolveKernelPy(const std::string& basename);

    // Resolve + import the op's kernel module. Returns a NEW reference to the module (exposing
    // __pypto_compile + the infer funcs) or nullptr with a Python error set. The GIL must be held by the
    // caller. *basename* selects the op_kernel .py; an unresolved/absent file is FATAL (a
    // FileNotFoundError naming the basename, the ASCEND_CUSTOM_OPP_PATH list and the dladdr dir). A base
    // static so Compile() and the free infer wrappers share ONE definition.
    static PyObject* ImportKernelModule(const std::string& stem, const std::string& basename);

    // Look the compile result up by cache key, or nullptr. Takes cache_mutex_; the returned pointer stays
    // valid after the lock is released because entries are inserted once and never overwritten or erased,
    // and std::map never relocates a mapped value. The result is const so that invariant is enforced by the
    // compiler rather than by convention: a cached entry must never be mutated once another thread can see it.
    const CompiledInfo* FindCachedResult(const std::string& key);

protected:
    std::string BuildCacheKey(const std::vector<std::string>& shape_strs, const std::vector<std::string>& dtype_strs,
                              const std::map<std::string, std::string>& attrs) const;

    // Insert a compile result under *key* if absent; an existing entry is kept. Takes cache_mutex_.
    void StoreCachedResult(const std::string& key, CompiledInfo&& info);

    // Compile() inserts and both GE callbacks look up. GE memoizes one executor instance per op TYPE and
    // documents no serialization for DeclareLaunchArgs, which it dispatches per node -- concurrent as soon
    // as MAX_COMPILE_CORE_NUMBER > 1 -- so every access to compiled_cache_ goes through cache_mutex_.
    std::mutex cache_mutex_;
    std::map<std::string, CompiledInfo> compiled_cache_;
};

#endif // PYPTO_PTO_CUSTOM_OP_H
