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
 * \file controller.cpp
 * \brief
 */

#include "pybind_common.h"

#include <utility>
#include <vector>
#include <string>

using namespace npu::tile_fwk;
using ref_tensors = std::vector<std::reference_wrapper<const Tensor>>;

namespace pypto {
void bind_controller_config(py::module &m) {
    m.def("SetBuildStatic", [](const bool &value) { config::SetBuildStatic(value); }, py::arg("value"));

    m.def("ResetOptions", []() {
        config::Reset();
        ConfigManagerNg::GetInstance().SetScope(std::map<std::string, Any>({{"host.compile_stage", (int64_t)GEN_KERNEL_CODE}}));
    });


    m.def(
        "SetPrintOptions",
        [](int edgeItems, int precision, int threshold, int linewidth) {
            config::SetPrintOptions(edgeItems, precision, threshold, linewidth);
        },
        py::arg("edgeItems"), py::arg("precision"), py::arg("threshold"), py::arg("linewidth"));

    m.def("SetSemanticLabel",
        [](const std::string &label, const std::string &filename, int lineno) {
            config::SetSemanticLabel(label, filename.c_str(), lineno);
        }, py::arg("label"), py::arg("filename"), py::arg("lineno"));

    m.def("IsVerifyEnabled", &calc::IsVerifyEnabled);
    m.def("LogTopFolder", []() { return py::cast(ConfigManager::Instance().LogTopFolder()); });
    m.def("ResetLog", [](const std::string &path) { ConfigManager::Instance().ResetLog(path); });
}


void bind_controller_set_tile(py::module &m) {
    m.def("SetVecTile", [](py::args args) {
        std::vector<int64_t> v;
        v.reserve(args.size());
        for (auto &a : args) {
            v.push_back(a.cast<int64_t>());
        }
        TileShape::Current().SetVecTile(v);
    });
    m.def("GetVecTile", []() { return TileShape::Current().GetVecTile().tile; });
    m.def(
        "SetMatrixSize", [](const std::vector<int64_t> &size) { TileShape::Current().SetMatrixSize(size); },
        py::arg("size"));
    m.def(
        "SetCubeTile",
        [](const std::vector<int64_t> &mvec, const std::vector<int64_t> &kvec, const std::vector<int64_t> &nvec,
            bool enableMultiDataLoad, bool enableSplitK) {
            if (mvec.size() > MAX_M_DIM_SIZE) {
                throw py::value_error(
                    "Parameter 'm' must have exactly " + std::to_string(MAX_M_DIM_SIZE) + " elements");
            }
            if (kvec.size() > MAX_K_DIM_SIZE) {
                throw py::value_error(
                    "Parameter 'k' must have exactly " + std::to_string(MAX_K_DIM_SIZE) + " elements");
            }
            if (nvec.size() > MAX_N_DIM_SIZE) {
                throw py::value_error(
                    "Parameter 'n' must have exactly " + std::to_string(MAX_N_DIM_SIZE) + " elements");
            }

            std::array<int64_t, MAX_M_DIM_SIZE> marr = {0};
            std::array<int64_t, MAX_K_DIM_SIZE> karr = {0};
            std::array<int64_t, MAX_N_DIM_SIZE> narr = {0};

            std::copy(mvec.begin(), mvec.end(), marr.begin());
            std::copy(kvec.begin(), kvec.end(), karr.begin());
            std::copy(nvec.begin(), nvec.end(), narr.begin());
            TileShape::Current().SetCubeTile(marr, karr, narr, enableMultiDataLoad, enableSplitK);
        },
        py::arg("m"), py::arg("k"), py::arg("n"), py::arg("enable_multi_data_load"), py::arg("enable_split_k"), 
        "Set cube tile shapes with specified dimensions");
    m.def("GetCubeTile", []() {
        auto cubeTile = TileShape::Current().GetCubeTile();
        return std::tuple(cubeTile.m, cubeTile.k, cubeTile.n, cubeTile.enableMultiDataLoad, cubeTile.enableSplitK);
    });
}

void bind_controller_function(py::module &m) {
    m.def("BeginFunction", [](const std::string &funcName, GraphType graphType, FunctionType funcType, py::args args) {
        std::vector<std::reference_wrapper<const Tensor>> tensors;
        tensors.reserve(args.size());
        for (auto &a : args) {
            tensors.push_back(a.cast<Tensor &>());
        }
        Program::GetInstance().Reset();
        Program::GetInstance().BeginFunction(FUNCTION_PREFIX + funcName, funcType, graphType, tensors);
    });
    m.def("EndFunction", [](const std::string &funcName, bool generateCall) {
        Program::GetInstance().EndFunction(FUNCTION_PREFIX + funcName, generateCall);
    });
    py::class_<RecordFunc>(m, "RecordFunc")
        .def(py::init<const std::string &>(), py::arg("name"))
        .def(py::init<const std::string &, const std::vector<std::reference_wrapper<const Tensor>> &>(), py::arg("name"),
            py::arg("explicit_op_args"))
        .def(
            py::init<const std::string &, const ref_tensors &, const ref_tensors &,
                const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>>
                    &>(),
            py::arg("name"), py::arg("inputs"), py::arg("outputs"),
            py::arg("in_place_args"))
        .def("EndFunction", &RecordFunc::EndFunction)
        .def("__iter__", [](RecordFunc &c) {
            // Return Python iterator from C++ begin/end
            return py::make_iterator(c.begin(), c.end());
            });;
    py::class_<RecordLoopFunc>(m, "RecordLoopFunc")
        .def(py::init<const std::string &, FunctionType, const std::string &, const LoopRange &, const std::set<int> &,
                 bool>(),
            py::arg("name"), py::arg("func_type"), py::arg("iter_name"), py::arg("loop_range"), py::arg("unroll_List"),
            py::arg("submit_before_loop"))
        .def("__iter__", [](RecordLoopFunc &c) {
            // Return Python iterator from C++ begin/end
            return py::make_iterator(c.begin(), c.end());
        });
}

void bind_controller_loop(py::module &m) {
    py::class_<RecordIfBranch>(m, "RecordIfBranch")
        .def(py::init<SymbolicScalar, const std::string &, int>(), py::arg("cond"), py::arg("file") = "",
            py::arg("line") = 0)
        .def("__bool__", py::overload_cast<>(&RecordIfBranch::operator bool, py::const_));
    py::class_<LoopRange>(m, "LoopRange")
        .def(py::init<const SymbolicScalar &/* rangeBegin */, const SymbolicScalar &/* rangeEnd */,
            const SymbolicScalar &/* rangeStep */>())
        .def(py::init<const SymbolicScalar &/* rangeBegin */, const SymbolicScalar &/* rangeEnd */>())
        .def(py::init<const SymbolicScalar &/* rangeEnd */>())
        .def(py::init<std::int64_t>()) // C++ Implicit conversion int64_t -> SymbolicScalar
        .def("Dump", (std::string(LoopRange::*)()) &LoopRange::Dump)
        .def("Begin", (SymbolicScalar &(LoopRange::*)()) &LoopRange::Begin,
            py::return_value_policy::reference_internal)
        .def("End", (SymbolicScalar &(LoopRange::*)()) &LoopRange::End, py::return_value_policy::reference_internal)
        .def(
            "Step", (SymbolicScalar &(LoopRange::*)()) &LoopRange::Step, py::return_value_policy::reference_internal);

    m.def("IsLoopBegin", &IsLoopBegin, py::arg("symbol"), py::arg("begin"));
    m.def("IsLoopEnd", &IsLoopEnd, py::arg("symbol"), py::arg("end"));
}

void bind_controller_utils(py::module &m) {
    m.def("Dump", []() { return Program::GetInstance().Dump(); });
    m.def("BytesOf", [](DataType t) { return BytesOf(t); });
    m.def("Reset", []() { Program::GetInstance().Reset(); });
    m.def("SetLocation", [](const std::string &fname, int lineno) {
        SourceLocation::SetLocation(fname, lineno);
    }, py::arg("fname"), py::arg("lineno"));
    m.def("SetLocation", [](const std::string &fname, int lineno, std::string &backtrace) {
        SourceLocation::SetLocation(fname, lineno, backtrace);
    }, py::arg("fname"), py::arg("lineno"), py::arg("backtrace"));
    m.def("ClearLocation", &SourceLocation::ClearLocation);
}


std::map<std::string, npu::tile_fwk::Any> ConvertPyDictToCppMap(const py::dict &values) {
    std::map<std::string, npu::tile_fwk::Any> cpp_values;
    for (auto item : values) {
        std::string key = py::str(item.first);
        py::object value = py::reinterpret_borrow<py::object>(item.second);

        if (py::isinstance<py::bool_>(value)) {
            cpp_values[key] = value.cast<bool>();
        } else if (py::isinstance<py::int_>(value)) {
            cpp_values[key] = value.cast<int64_t>();
        } else if (py::isinstance<py::float_>(value)) {
            cpp_values[key] = value.cast<double>();
        } else if (py::isinstance<py::str>(value)) {
            cpp_values[key] = value.cast<std::string>();
        } else if (py::isinstance<CubeTile>(value)) {
            cpp_values[key] = value.cast<CubeTile>();
        } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
            py::list lst = py::cast<py::list>(value);
            if (lst.size() > 0) {
                if (py::isinstance<py::int_>(lst[0])) {
                    cpp_values[key] = value.cast<std::vector<int64_t>>();
                } else if (py::isinstance<py::str>(lst[0])) {
                    cpp_values[key] = value.cast<std::vector<std::string>>();
                } else {
                    throw py::type_error("Unsupported list element type for key: " + key);
                }
            } else {
                cpp_values[key] = std::vector<int64_t>();
            }
        } else if (py::isinstance<py::dict>(value)) {
            cpp_values[key] = value.cast<std::map<int64_t, int64_t>>();
        } else {
            throw py::type_error("Unsupported value type for key: " + key);
        }
    }

    return cpp_values;
}

void bind_controller_scope(py::module &m) {
    m.def("BeginScope",
        [](const std::string &name, const py::dict &values,
        const std::string &filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().BeginScope(name, std::move(cpp_values), filename.c_str(), lineno);
        },
        py::arg("name"),
        py::arg("values"),
        py::arg("filename"),
        py::arg("lineno")
    );

    m.def("EndScope",
        [](const std::string &filename, int lineno) {
            ConfigManagerNg::GetInstance().EndScope(filename.c_str(), lineno);
        },
        py::arg("filename") = "default",
        py::arg("lineno") = -1
    );

    m.def("SetScope",
        [](const py::dict &values, const std::string &filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().SetScope(std::move(cpp_values), filename.c_str(), lineno);
        },
        py::arg("values"),
        py::arg("filename") = "default",
        py::arg("lineno") = -1
    );

    m.def("SetGlobalConfig",
        [](const py::dict &values, const std::string &filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().SetGlobalConfig(std::move(cpp_values), filename.c_str(), lineno);
        },
        py::arg("values"),
        py::arg("filename") = "default",
        py::arg("lineno") = -1
    );

    m.def("CurrentScope",
        []() { return ConfigManagerNg::GetInstance().CurrentScope(); });

    m.def("GlobalScope",
        []() { return ConfigManagerNg::GetInstance().GlobalScope(); });

    m.def("GetOptionsTree",
        []() { return ConfigManagerNg::GetInstance().GetOptionsTree(); });
}

py::object AnyToPyObject(const Any &val) {
    using Fn = std::function<py::object(const Any&)>;
    static const std::unordered_map<std::type_index, Fn> table = {
        {typeid(bool), [](const Any& a){ return py::cast(AnyCast<bool>(a)); }},
        {typeid(int64_t), [](const Any& a){ return py::cast(AnyCast<int64_t>(a)); }},
        {typeid(double), [](const Any& a){ return py::cast(AnyCast<double>(a)); }},
        {typeid(std::string), [](const Any& a){ return py::cast(AnyCast<std::string>(a)); }},
        {typeid(std::vector<int64_t>), [](const Any& a){ return py::cast(AnyCast<std::vector<int64_t>>(a)); }},
        {typeid(std::vector<std::string>), [](const Any& a){ return py::cast(AnyCast<std::vector<std::string>>(a)); }},
        {typeid(std::map<int64_t,int64_t>), [](const Any& a){ return py::cast(AnyCast<std::map<int64_t,int64_t>>(a)); }},
        {typeid(CubeTile), [](const Any& a){ return py::cast(AnyCast<CubeTile>(a)); }},
        {typeid(DistTile), [](const Any& a){ return py::str(AnyCast<DistTile>(a).ToString()); }},
    };

    auto it = table.find(std::type_index(val.Type()));
    if (it != table.end()) return it->second(val);

    throw py::type_error("Unsupported config value type: " + std::string(val.Type().name()));
}

void bind_controller_scope_classes(py::module &m) {
    py::class_<ConfigScope, std::shared_ptr<ConfigScope>>(m, "ConfigScope")
        .def("GetAnyConfig",
            [](const ConfigScope &scope, const std::string &key) -> py::object {
                return AnyToPyObject(scope.GetAnyConfig(key));
            },
            py::arg("key"))
        .def("GetAllConfig",
            [](const ConfigScope &scope) -> py::dict {
                py::dict result;
                auto config_map = scope.GetAllConfig();
                
                for (const auto &[key, val] : config_map) {
                    try {
                        result[py::str(key)] = AnyToPyObject(val);
                    } catch (const py::type_error &e) {
                        py::print("Warning: Skipping key '", key, "' -", e.what());
                    }
                }
                return result;
            })
        .def("HasConfig", &ConfigScope::HasConfig, py::arg("key"))
        .def("Type",
            [](const ConfigScope &scope, const std::string &key) -> std::string {
                return scope.Type(key).name();
            },
            py::arg("key"))
        .def("ToString", &ConfigScope::ToString);

    py::class_<CubeTile>(m, "CubeTile")
    .def(py::init<>())
    .def(py::init<const std::array<int64_t, MAX_M_DIM_SIZE>&,
                   const std::array<int64_t, MAX_K_DIM_SIZE>&,
                   const std::array<int64_t, MAX_N_DIM_SIZE>&,
                   bool, bool>(),
         py::arg("m"),
         py::arg("k"),
         py::arg("n"),
         py::arg("enableMultiDataLoad") = false,
         py::arg("enableSplitK") = false)
    .def_readwrite("m", &CubeTile::m)
    .def_readwrite("k", &CubeTile::k)
    .def_readwrite("n", &CubeTile::n)
    .def_readwrite("enableMultiDataLoad", &CubeTile::enableMultiDataLoad)
    .def_readwrite("enableSplitK", &CubeTile::enableSplitK)
    .def("valid", &CubeTile::valid)
    .def("ToString", &CubeTile::ToString)
    .def("__repr__", [](const CubeTile &t) { return t.ToString(); })
    .def("__str__",  [](const CubeTile &t) { return t.ToString(); });
}

void bind_controller(py::module &m) {
    bind_controller_config(m);
    bind_controller_set_tile(m);
    bind_controller_function(m);
    bind_controller_loop(m);
    bind_controller_utils(m);
    bind_controller_scope(m);
    bind_controller_scope_classes(m);

    // disable cpp mode
    SourceLocation::SetCppMode(false);
    // set default compile stage to codegen
    ConfigManagerNg::GetInstance().SetScope(std::map<std::string, Any>({{"host.compile_stage", (int64_t)GEN_KERNEL_CODE}}));
}
} // namespace pypto
