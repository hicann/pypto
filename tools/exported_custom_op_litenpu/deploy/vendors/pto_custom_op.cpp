/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// pypto shared custom-op base class — implementation. See pto_custom_op.h.
#include "pto_custom_op.h"

// Opened here rather than in the header, so the header does not leak them into consuming TUs.
using namespace ge;
using namespace gert;
#include <nlohmann/json.hpp>

#include <Python.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

using json = nlohmann::json;

std::string PtoCustomOp::ListIntToJsonStr(const std::vector<int64_t>& vals)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vals.size(); i++) {
        if (i > 0)
            oss << ",";
        oss << vals[i];
    }
    oss << "]";
    return oss.str();
}

// Emit exactly the dtype basename strings pypto's runtime compile contract accepts; an unmapped
// dtype returns nullptr (logged) so it fails loudly rather than masquerading as float16.
const char* PtoCustomOp::DataTypeToStr(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return "float32";
        case ge::DT_DOUBLE:
            return "float64";
        case ge::DT_FLOAT16:
            return "float16";
        case ge::DT_BF16:
            return "bfloat16";
        case ge::DT_INT8:
            return "int8";
        case ge::DT_INT16:
            return "int16";
        case ge::DT_INT32:
            return "int32";
        case ge::DT_INT64:
            return "int64";
        case ge::DT_UINT8:
            return "uint8";
        case ge::DT_BOOL:
            return "bool";
        default:
            PTO_CUSTOM_LOGE("ERROR: unsupported ge::DataType enum=%d\n", static_cast<int>(dtype));
            return nullptr;
    }
}

// Append one length-prefixed ("<len>:<text>") segment. Attr values are arbitrary strings (a String attr
// holds anything; a ListInt attr's rendering contains commas), so a plain separator-joined key is not
// injective — two different attr sets can render identically, and a collision makes the second node reuse
// the FIRST node's kernel binary and blockDim. A reader consumes exactly <len> bytes, so the segment text
// can never be mistaken for structure.
static void AppendKeySegment(std::string& key, const std::string& segment)
{
    key += std::to_string(segment.size());
    key += ':';
    key += segment;
}

std::string PtoCustomOp::BuildCacheKey(const std::vector<std::string>& shape_strs,
                                       const std::vector<std::string>& dtype_strs,
                                       const std::map<std::string, std::string>& attrs) const
{
    std::string key = "{";
    for (const std::string& shape_str : shape_strs) {
        AppendKeySegment(key, shape_str);
    }
    key += "}|{";
    for (const std::string& dtype_str : dtype_strs) {
        AppendKeySegment(key, dtype_str);
    }
    key += "}|{";
    // std::map iterates in name order, so the same attr set always renders the same key.
    for (const auto& kv : attrs) {
        AppendKeySegment(key, kv.first);
        AppendKeySegment(key, kv.second);
    }
    key += "}";
    return key;
}

const CompiledInfo* PtoCustomOp::FindCachedResult(const std::string& key)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = compiled_cache_.find(key);
    return (it != compiled_cache_.end()) ? &it->second : nullptr;
}

void PtoCustomOp::StoreCachedResult(const std::string& key, CompiledInfo&& info)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    // Insert-if-absent, never overwrite. FindCachedResult hands out a pointer INTO the mapped value and GE
    // reads through it (kernelBuffer.data(), kernelName.c_str()) after cache_mutex_ is released, so an
    // in-place reassignment of an existing key could invalidate a pointer another thread still holds.
    // GE calls Compile() once per NODE, so repeat keys are routine: the first result wins.
    const auto inserted = compiled_cache_.try_emplace(key, std::move(info)).second;
    if (!inserted) {
        PTO_CUSTOM_LOGD("Cache already holds key=%s, keeping the first compile result\n", key.c_str());
    }
}

// dladdr on a base-class symbol yields THIS .so's loaded on-disk path; return its directory.
std::string PtoCustomOp::SelfSoDir()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&PtoCustomOp::SelfSoDir), &info) && info.dli_fname) {
        std::string p(info.dli_fname);
        auto slash = p.find_last_of('/');
        return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
    }
    return {};
}

// Resolve the single canonical <opp_root>/op_kernel/<basename> ENV-FIRST: ASCEND_CUSTOM_OPP_PATH is a
// LIST of entries (GE treats it that way), so split on ':' and probe each <entry>/op_kernel/<basename>,
// first R_OK hit wins. Fallback (env unset/empty or no hit): the dladdr SelfSoDir() + a bounded upward
// walk (<so_dir>, /.., /../..) retargeted to op_kernel/, covering the root/op_proto/framework.onnx copies.
// Static (this-less) so the free infer wrappers reuse it with a codegen-baked basename literal.
std::string PtoCustomOp::ResolveKernelPy(const std::string& name)
{
    if (name.empty())
        return {};

    // Primary: env-based. Split ASCEND_CUSTOM_OPP_PATH on ':' and probe each entry's op_kernel/<name>.
    const char* opp = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (opp != nullptr && *opp != '\0') {
        const std::string list(opp);
        size_t start = 0;
        while (start <= list.size()) {
            size_t sep = list.find(':', start);
            const std::string entry = list.substr(start, sep == std::string::npos ? std::string::npos : sep - start);
            if (!entry.empty()) {
                const std::string cand = entry + "/op_kernel/" + name;
                if (::access(cand.c_str(), R_OK) == 0) {
                    PTO_CUSTOM_LOGD("Resolved dev-editable kernel snippet (env): %s\n", cand.c_str());
                    return cand;
                }
            }
            if (sep == std::string::npos)
                break;
            start = sep + 1;
        }
    }

    // Fallback: dladdr self-path + bounded upward walk, retargeted to op_kernel/.
    const std::string dir = SelfSoDir();
    if (dir.empty())
        return {};
    std::string prefix = dir;
    for (int up = 0; up <= 2; ++up) {
        const std::string cand = prefix + "/op_kernel/" + name;
        if (::access(cand.c_str(), R_OK) == 0) {
            PTO_CUSTOM_LOGD("Resolved dev-editable kernel snippet (dladdr): %s\n", cand.c_str());
            return cand;
        }
        prefix += "/..";
    }
    return {};
}

// --- Compile-time Python helpers (file-local) -------------------------------

// The Python entry the shipped snippet exposes; fixed by our codegen contract.
static constexpr const char* kCompileFuncName = "__pypto_compile";

// Returns nullptr (Python error unset; the caller logs and fails the compile) when an input tensor is
// absent. Failing beats skipping: the cache key is built by the null-skipping extractors, so a skipped
// input here would describe a different input set than the key the compile result is stored under.
static PyObject* BuildTensorShapesTuple(gert::OpCompileContext* ctx, size_t inputNum)
{
    PyObject* shapes_tuple = PyTuple_New(inputNum);
    if (shapes_tuple == nullptr)
        return nullptr;
    for (size_t i = 0; i < inputNum; i++) {
        const gert::Tensor* tensor = ctx->GetInputTensor(i);
        if (tensor == nullptr) {
            PTO_CUSTOM_LOGE("ERROR: null input tensor at index %zu while building the compile shapes\n", i);
            Py_DECREF(shapes_tuple);
            return nullptr;
        }
        const gert::Shape& input_shape = tensor->GetOriginShape();
        size_t dim_num = input_shape.GetDimNum();
        PyObject* shape_tuple = PyTuple_New(dim_num);
        if (shape_tuple == nullptr) {
            Py_DECREF(shapes_tuple);
            return nullptr;
        }
        for (size_t j = 0; j < dim_num; j++) {
            PyTuple_SetItem(shape_tuple, j, PyLong_FromLongLong(input_shape.GetDim(j)));
        }
        PyTuple_SetItem(shapes_tuple, i, shape_tuple);
    }
    return shapes_tuple;
}

// Same null-tensor contract as BuildTensorShapesTuple: a missing input fails the compile.
static PyObject* BuildTensorDtypesTuple(gert::OpCompileContext* ctx, size_t inputNum)
{
    PyObject* dtypes_tuple = PyTuple_New(inputNum);
    if (dtypes_tuple == nullptr)
        return nullptr;
    for (size_t i = 0; i < inputNum; i++) {
        const gert::Tensor* tensor = ctx->GetInputTensor(i);
        if (tensor == nullptr) {
            PTO_CUSTOM_LOGE("ERROR: null input tensor at index %zu while building the compile dtypes\n", i);
            Py_DECREF(dtypes_tuple);
            return nullptr;
        }
        const char* dtype_str = PtoCustomOp::DataTypeToStr(tensor->GetDataType());
        // Unmapped dtype: pass "unsupported" so the Python compile entry raises a clear error
        // (DataTypeToStr already logged the enum) instead of silently compiling for float16.
        PyObject* py_dtype = PyUnicode_FromString(dtype_str ? dtype_str : "unsupported");
        if (py_dtype == nullptr) {
            Py_DECREF(dtypes_tuple);
            return nullptr;
        }
        PyTuple_SetItem(dtypes_tuple, i, py_dtype);
    }
    return dtypes_tuple;
}

// Returns nullptr when an attr value is not valid UTF-8 (PyUnicode_FromString is data-dependent and the
// values come from the graph). Failing the dict build fails the compile, which beats compiling the kernel
// against a silently incomplete attr set.
static PyObject* BuildAttrsDict(const std::map<std::string, std::string>& attrs)
{
    PyObject* py_attrs = PyDict_New();
    if (py_attrs == nullptr)
        return nullptr;
    for (const auto& kv : attrs) {
        PyObject* v = PyUnicode_FromString(kv.second.c_str());
        if (v == nullptr) {
            PTO_CUSTOM_LOGE("ERROR: attr '%s' has a value that is not valid UTF-8\n", kv.first.c_str());
            Py_DECREF(py_attrs);
            return nullptr;
        }
        PyDict_SetItemString(py_attrs, kv.first.c_str(), v);
        Py_DECREF(v);
    }
    return py_attrs;
}

// Calls ``func_name(shapes, dtypes, attrs, soc_version)`` and returns the .o path string.
static std::string CallPythonFunc(PyObject* module, const std::string& func_name, PyObject* shapes_tuple,
                                  PyObject* dtypes_tuple, PyObject* py_attrs, PyObject* py_soc_version)
{
    PyObject* func = PyObject_GetAttrString(module, func_name.c_str());
    if (!func || !PyCallable_Check(func)) {
        PyErr_Print();
        PTO_CUSTOM_LOGE("ERROR: function %s not found", func_name.c_str());
        Py_XDECREF(func);
        return "";
    }
    PyObject* args = PyTuple_Pack(4, shapes_tuple, dtypes_tuple, py_attrs, py_soc_version);
    PyObject* result = PyObject_CallObject(func, args);

    std::string compiled_bin_path;
    if (result) {
        PyObject* str_result = PyUnicode_Check(result) ? result : PyObject_Str(result);
        if (str_result) {
            const char* c = PyUnicode_AsUTF8(str_result);
            compiled_bin_path = c ? c : "";
            if (str_result != result)
                Py_DECREF(str_result);
        }
        Py_DECREF(result);
    } else {
        PyErr_Print();
    }
    Py_DECREF(args);
    Py_DECREF(func);
    return compiled_bin_path;
}

// Parse ``<bin without .o>.json`` (blockDim/kernelName/workspaceSize/kernelBin) and load
// the kernel binary bytes.
//
// The sidecar is produced by the dev-editable op_kernel/<stem>.py, so a malformed or incomplete one is a
// realistic input rather than corruption. Everything that can throw (nlohmann parse/type errors, the
// std::string/std::vector allocations) is contained here and reported as false: Compile() runs inside a
// graphStatus-returning GE callback, and an exception must never unwind across that ABI boundary.
static bool ParseCompiledMetadata(const std::string& compiled_bin_path, CompiledInfo& info)
{
    info.binPath = compiled_bin_path;
    std::string json_name = compiled_bin_path.substr(0, compiled_bin_path.size() - 2) + ".json";
    std::ifstream f(json_name);
    if (!f.is_open()) {
        PTO_CUSTOM_LOGE("ERROR: cannot open json sidecar %s", json_name.c_str());
        return false;
    }
    std::string kernel_bin_path;
    try {
        json j = json::parse(f);
        info.blockDim = j["blockDim"].get<uint32_t>();
        info.kernelName = j["kernelName"].get<std::string>();
        info.workspaceSize = j.value("workspaceSize", static_cast<size_t>(0));

        std::string kernel_bin_name = j["kernelBin"].get<std::string>();
        size_t lastSlash = kernel_bin_name.find_last_of('/');
        if (lastSlash != std::string::npos)
            kernel_bin_name = kernel_bin_name.substr(lastSlash + 1);
        kernel_bin_path = compiled_bin_path.substr(0, compiled_bin_path.find_last_of("/\\") + 1) + kernel_bin_name;
    } catch (const nlohmann::json::exception& e) {
        PTO_CUSTOM_LOGE("ERROR: malformed json sidecar %s: %s\n", json_name.c_str(), e.what());
        return false;
    } catch (const std::exception& e) {
        PTO_CUSTOM_LOGE("ERROR: cannot read json sidecar %s: %s\n", json_name.c_str(), e.what());
        return false;
    }

    std::ifstream file(kernel_bin_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        PTO_CUSTOM_LOGE("ERROR: cannot open kernel bin %s", kernel_bin_path.c_str());
        return false;
    }
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    try {
        info.kernelBuffer.resize(size);
    } catch (const std::exception& e) {
        PTO_CUSTOM_LOGE("ERROR: cannot hold kernel bin %s (%zu bytes): %s\n", kernel_bin_path.c_str(), size, e.what());
        return false;
    }
    file.read(reinterpret_cast<char*>(info.kernelBuffer.data()), static_cast<std::streamsize>(size));
    return true;
}

// The SoC version selects the kernel's target architecture, so the fallback is reported at ERROR level:
// the mobile OMC path may legitimately omit ``ge.socVersion`` (hence a default rather than a hard
// failure), but a kernel built for the wrong architecture must never be a silent outcome.
static std::string GetSocVersion(gert::OpCompileContext* ctx)
{
    ge::AscendString soc_version_val;
    const std::string kDefaultSocVersion = "Kirin9030";
    std::string soc_version_str = kDefaultSocVersion;
    if (ctx->GetOption(ge::AscendString("ge.socVersion"), soc_version_val) == ge::GRAPH_SUCCESS) {
        soc_version_str = soc_version_val.GetString();
    } else {
        PTO_CUSTOM_LOGE("ge.socVersion option unavailable, compiling the kernel for the default SoC %s\n",
                        kDefaultSocVersion.c_str());
    }
    return soc_version_str;
}

// The kernel-module loader source (imports the shipped op_kernel/<stem>.py by path). The
// codegen injects the ``deploy.embed`` module source in place of the placeholder below, so
// the built .so carries the loader INLINE and imports NO pypto Python module at runtime (only stdlib,
// used inside the imported snippet). Keeps the
// "no cpp in pypto.extensions.torch_custom_op_litenpu" boundary at deploy time too.
namespace {
const char* kEmbeddedCompileLoaderSrc = R"PYPTOEMBED(
__PYPTO_EMBED_LOADER_SRC__
)PYPTOEMBED";

// Exec the loader source once into a private globals dict (kept alive process-lifetime via the
// returned function's __globals__) and cache the ``load_embedded_compile_module`` callable. The GIL
// must be held by the caller, which also serializes this lazy initialisation: it is the only mutable
// state on this path, so it needs no separate lock (compiled_cache_ has cache_mutex_ for that reason).
// Returns nullptr on failure (Python error set).
PyObject* GetEmbeddedCompileLoader()
{
    static PyObject* loader = nullptr;
    if (loader != nullptr) {
        return loader;
    }
    PyObject* globals = PyDict_New();
    if (globals == nullptr) {
        return nullptr;
    }
    PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
    PyObject* res = PyRun_String(kEmbeddedCompileLoaderSrc, Py_file_input, globals, globals);
    if (res == nullptr) {
        Py_DECREF(globals);
        return nullptr;
    }
    Py_DECREF(res);
    PyObject* fn = PyDict_GetItemString(globals, "load_embedded_compile_module"); // borrowed
    if (fn != nullptr) {
        Py_INCREF(fn); // hold the loader (and, via its __globals__, its module state) for the process
    }
    Py_DECREF(globals); // fn keeps globals alive through its __globals__
    loader = fn;
    return loader;
}
} // namespace

// Resolve + import of the op's kernel module, shared by Compile() and both infer wrappers. The shipped
// <opp_root>/op_kernel/<basename> is REQUIRED: an unresolved path is a FATAL FileNotFoundError.
// Returns a NEW reference (or nullptr with a Python error set); GIL held by caller.
PyObject* PtoCustomOp::ImportKernelModule(const std::string& stem, const std::string& basename)
{
    const std::string py_path = basename.empty() ? std::string{} : ResolveKernelPy(basename);
    if (py_path.empty()) {
        const char* opp = std::getenv("ASCEND_CUSTOM_OPP_PATH");
        // Descriptive: names the basename + the env list + the dladdr dir that were searched.
        std::string msg = "pypto op_kernel: could not resolve op_kernel/" + basename + " for stem '" + stem +
                          "'. Searched ASCEND_CUSTOM_OPP_PATH=[" + (opp ? opp : "<unset>") + "] and dladdr dir '" +
                          SelfSoDir() + "'. Install via the op-package install step (it ships <opp_root>/op_kernel/" +
                          basename + ").";
        PyErr_SetString(PyExc_FileNotFoundError, msg.c_str());
        PTO_CUSTOM_LOGE("ERROR: %s\n", msg.c_str());
        return nullptr;
    }
    PyObject* loader = GetEmbeddedCompileLoader(); // borrowed, process-lifetime cache
    if (!loader || !PyCallable_Check(loader)) {
        PyErr_Print();
        return nullptr;
    }
    PyObject* mod = PyObject_CallFunction(loader, "ss", stem.c_str(), py_path.c_str());
    // embed.py raises FileNotFoundError if the file vanished between resolve and import, and RuntimeError
    // when the op package carries no pypto_version.info or was built by a newer pypto -> propagates here.
    if (mod) {
        PTO_CUSTOM_LOGD("Resolved kernel module stem=%s path=%s\n", stem.c_str(), py_path.c_str());
    } else {
        // Contract: return nullptr with the error STILL SET — Compile prints it, and the infer wrappers
        // rethrow it through pybind11, which requires the indicator set. So NO PyErr_Print here: fetch +
        // normalize only to log the reason, then PyErr_Restore it unchanged.
        PyObject* etype = nullptr;
        PyObject* evalue = nullptr;
        PyObject* etb = nullptr;
        PyErr_Fetch(&etype, &evalue, &etb);
        PyErr_NormalizeException(&etype, &evalue, &etb);
        std::string reason;
        if (evalue != nullptr) {
            // The file's own Python-object-to-text pattern (see CallPythonFunc).
            PyObject* str_value = PyUnicode_Check(evalue) ? evalue : PyObject_Str(evalue);
            if (str_value != nullptr) {
                const char* c = PyUnicode_AsUTF8(str_value);
                reason = c ? c : "";
                if (str_value != evalue)
                    Py_DECREF(str_value);
            }
        }
        // Drop anything PyObject_Str/PyUnicode_AsUTF8 may have set while building the text, so the
        // restore below hands the consumers the ORIGINAL error rather than a formatting failure.
        PyErr_Clear();
        // The only site putting the REASON in slog: Compile's LOGE names just the stem (its PyErr_Print
        // goes to stderr). The infer wrappers log the text for a genuine user infer exception, so an
        // import failure through InferShape is logged twice on purpose.
        PTO_CUSTOM_LOGE("ERROR: kernel module import failed for stem %s: %s\n", stem.c_str(), reason.c_str());
        PyErr_Restore(etype, evalue, etb);
    }
    return mod; // NEW ref (or nullptr with error set)
}

// --- GE callbacks -----------------------------------------------------------

namespace {
// RAII over PyGILState_Ensure/Release: Compile() runs inside a graphStatus-returning GE callback, so
// every exit from the guarded region — an early return or an unexpected exception — must hand the GIL
// back. A leaked GIL deadlocks every later op that touches Python.
class GilGuard {
public:
    GilGuard() : state_(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state_); }
    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;

private:
    PyGILState_STATE state_;
};
} // namespace

graphStatus PtoCustomOp::Compile(gert::OpCompileContext* ctx)
{
    PTO_CUSTOM_LOGD("Entering PtoCustomOp::Compile\n");
    size_t inputNum = ctx->GetComputeNodeInputNum();
    std::vector<std::string> shape_strs, dtype_strs;
    ExtractInputShapes(ctx, shape_strs);
    ExtractInputDtypes(ctx, dtype_strs);
    std::map<std::string, std::string> attrs;
    ExtractAttrs(ctx->GetAttrs(), attrs);
    std::string cache_key = BuildCacheKey(shape_strs, dtype_strs, attrs);
    if (FindCachedResult(cache_key) != nullptr) {
        PTO_CUSTOM_LOGD("Cache hit for key=%s, skip compile", cache_key.c_str());
        PTO_CUSTOM_LOGD("Exiting PtoCustomOp::Compile (cache hit, ok)\n");
        return SUCCESS;
    }

    // One-time CPython bootstrap for a non-Python host (GE dlopen'd this .so); guarded so a
    // Python host is untouched, and never finalized (re-init is fragile).
    static const bool _bootstrapped = []() {
        if (!Py_IsInitialized()) {
            Py_InitializeEx(0);  // 0: don't install signal handlers (library-friendly)
            PyEval_SaveThread(); // drop the GIL the init thread holds; reacquired below
        }
        return true;
    }();
    (void)_bootstrapped;
    GilGuard gil;

    // Load the kernel-compile snippet via the shared resolve+import. The shipped dev-editable
    // <opp_root>/op_kernel/<stem>.py is REQUIRED (resolved ENV-first via the ASCEND_CUSTOM_OPP_PATH
    // entries, dladdr upward-walk fallback) so a developer can edit it in place and the change takes
    // effect on the next op-compile with NO .so rebuild; an unresolved/absent file is FATAL. The loader
    // source is compiled into this .so — no pypto Python module is imported at runtime.
    PyObject* module = ImportKernelModule(GetCompileModuleStem(), GetKernelPyBasename());
    if (!module) {
        PyErr_Print();
        PTO_CUSTOM_LOGE("ERROR: kernel module load failed for stem %s\n", GetCompileModuleStem().c_str());
        PTO_CUSTOM_LOGD("Exiting PtoCustomOp::Compile (kernel module load error -> GRAPH_FAILED)\n");
        return GRAPH_FAILED; // strict: an unresolved op_kernel .py is FATAL
    }

    // Each builder returns nullptr on a null input tensor, an unencodable attr value, or an allocation
    // failure; any of those means the compile call would be made against an argument set that does not
    // describe this node, so fail the compile instead.
    PyObject* shapes_tuple = BuildTensorShapesTuple(ctx, inputNum);
    PyObject* dtypes_tuple = BuildTensorDtypesTuple(ctx, inputNum);
    PyObject* py_attrs = BuildAttrsDict(attrs);
    std::string soc_version = GetSocVersion(ctx);
    PyObject* py_soc = PyUnicode_FromString(soc_version.c_str());
    if (shapes_tuple == nullptr || dtypes_tuple == nullptr || py_attrs == nullptr || py_soc == nullptr) {
        PyErr_Print(); // prints and clears whichever builder set an error; a no-op when none did
        Py_XDECREF(py_soc);
        Py_XDECREF(py_attrs);
        Py_XDECREF(shapes_tuple);
        Py_XDECREF(dtypes_tuple);
        Py_DECREF(module);
        PTO_CUSTOM_LOGE("ERROR: cannot build the Python compile arguments\n");
        PTO_CUSTOM_LOGD("Exiting PtoCustomOp::Compile (compile-argument build error -> GRAPH_FAILED)\n");
        return ge::GRAPH_FAILED;
    }

    std::string compiled_bin_path = CallPythonFunc(module, kCompileFuncName, shapes_tuple, dtypes_tuple, py_attrs,
                                                   py_soc);

    Py_DECREF(py_soc);
    Py_DECREF(py_attrs);
    Py_DECREF(shapes_tuple);
    Py_DECREF(dtypes_tuple);
    Py_DECREF(module);

    if (compiled_bin_path.empty()) {
        PTO_CUSTOM_LOGE("ERROR: Python compile returned empty path");
        PTO_CUSTOM_LOGD("Exiting PtoCustomOp::Compile (empty compile path -> GRAPH_FAILED)\n");
        return GRAPH_FAILED;
    }
    CompiledInfo info;
    if (!ParseCompiledMetadata(compiled_bin_path, info)) {
        PTO_CUSTOM_LOGD("Exiting PtoCustomOp::Compile (metadata parse error -> GRAPH_FAILED)\n");
        return GRAPH_FAILED;
    }
    const uint32_t block_dim = info.blockDim;
    const size_t workspace_size = info.workspaceSize;
    StoreCachedResult(cache_key, std::move(info));
    PTO_CUSTOM_LOGD("Compiled key=%s path=%s blockDim=%u workspaceSize=%zu", cache_key.c_str(),
                    compiled_bin_path.c_str(), block_dim, workspace_size);
    PTO_CUSTOM_LOGD("Exiting PtoCustomOp::Compile (ok)\n");
    return SUCCESS;
}

// ge 20260717 annotated-args launch: declare the kernel launch args declaratively at compile time. The
// arg pack is arity-generic (inputs, then outputs, then the workspace slot), read from the runtime I/O
// counts, so no per-op launch code is generated. Null tensors are skipped (lenient; static-shape ONNX
// ops have all tensors present). One AddLaunch (the offline TaskDef path supports a single main-stream
// launch).
graphStatus PtoCustomOp::DeclareLaunchArgs(gert::AnnotatedArgsContext& ctx)
{
    PTO_CUSTOM_LOGD("Entering PtoCustomOp::DeclareLaunchArgs\n");
    std::vector<std::string> shape_strs, dtype_strs;
    ExtractInputShapes(&ctx, shape_strs);
    ExtractInputDtypes(&ctx, dtype_strs);
    std::map<std::string, std::string> attrs;
    ExtractAttrs(ctx.GetAttrs(), attrs);

    std::string key = BuildCacheKey(shape_strs, dtype_strs, attrs);
    const CompiledInfo* info = FindCachedResult(key);
    if (info == nullptr) {
        PTO_CUSTOM_LOGE("ERROR: cache miss for key=%s", key.c_str());
        PTO_CUSTOM_LOGD("Exiting PtoCustomOp::DeclareLaunchArgs (cache miss -> GRAPH_FAILED)\n");
        return GRAPH_FAILED;
    }

    AnnotatedKernelArgs args;
    // InputAddr.index / OutputAddr.index are IR-prototype port indices (pr_4046), not flat instance
    // indices. Reusing the flat loop index is correct ONLY because every current pypto op is fixed-arity,
    // all-REQUIRED (no demo op declares optional/dynamic input ports; support_dynamic_aligned is a dynamic
    // SHAPE, a different axis). A future op with optional/dynamic ports MUST revisit this and index via the
    // IR-prototype accessors GetRequiredInputTensor/GetOptionalInputTensor/GetDynamicInputTensor.
    // An AppendArg failure leaves a hole in the argument pack, and the kernel would then read its operands
    // from the wrong slots, so every append is checked and a failure fails the declaration.
    size_t nIn = ctx.GetComputeNodeInputNum();
    for (size_t i = 0; i < nIn; i++) {
        const gert::Tensor* t = ctx.GetInputTensor(i);
        if (t == nullptr)
            continue;
        if (args.AppendArg(InputAddr{static_cast<uint32_t>(i), t->GetAddr()}) != ge::GRAPH_SUCCESS) {
            PTO_CUSTOM_LOGE("ERROR: AppendArg failed for input %zu, key=%s\n", i, key.c_str());
            PTO_CUSTOM_LOGD("Exiting PtoCustomOp::DeclareLaunchArgs (append input failed -> GRAPH_FAILED)\n");
            return ge::GRAPH_FAILED;
        }
    }
    size_t nOut = ctx.GetComputeNodeOutputNum();
    for (size_t o = 0; o < nOut; o++) {
        const gert::Tensor* t = ctx.GetOutputTensor(o);
        if (t == nullptr)
            continue;
        if (args.AppendArg(OutputAddr{static_cast<uint32_t>(o), t->GetAddr()}) != ge::GRAPH_SUCCESS) {
            PTO_CUSTOM_LOGE("ERROR: AppendArg failed for output %zu, key=%s\n", o, key.c_str());
            PTO_CUSTOM_LOGD("Exiting PtoCustomOp::DeclareLaunchArgs (append output failed -> GRAPH_FAILED)\n");
            return ge::GRAPH_FAILED;
        }
    }
    if (info->workspaceSize > 0) {
        gert::WorkspaceAddr ws = ctx.MallocWorkSpace(info->workspaceSize);
        if (ws.addr == nullptr) {
            PTO_CUSTOM_LOGE("ERROR: MallocWorkSpace failed, key=%s size=%zu", key.c_str(), info->workspaceSize);
            PTO_CUSTOM_LOGD("Exiting PtoCustomOp::DeclareLaunchArgs (workspace alloc failed)\n");
            return GRAPH_FAILED;
        }
        if (args.AppendArg(ws) != ge::GRAPH_SUCCESS) {
            PTO_CUSTOM_LOGE("ERROR: AppendArg failed for the workspace slot, key=%s\n", key.c_str());
            PTO_CUSTOM_LOGD("Exiting PtoCustomOp::DeclareLaunchArgs (append workspace failed -> GRAPH_FAILED)\n");
            return ge::GRAPH_FAILED;
        }
    }

    gert::AnnotatedKernelLaunchInfo launch_info{};
    launch_info.kernel_name = info->kernelName.c_str();
    launch_info.kernel_bin = info->kernelBuffer.data();
    launch_info.kernel_bin_size = info->kernelBuffer.size();
    launch_info.block_dim = info->blockDim;
    launch_info.stream_id = ctx.GetStreamId();
    PTO_CUSTOM_LOGD("Exiting PtoCustomOp::DeclareLaunchArgs (AddLaunch)\n");
    return ctx.AddLaunch(launch_info, std::move(args));
}
