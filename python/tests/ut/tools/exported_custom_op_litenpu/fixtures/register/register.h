/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Minimal stub of the real Ascend ``register/register.h`` for compiling pypto
// export op_custom_def codegen tests only.
//
// The generated TU emits, inside ``namespace ge { ... }``:
//
//   REG_OP(MyOp)
//       .INPUT(in0, TensorType({DT_FLOAT16}))
//       .OUTPUT(out0, TensorType({DT_FLOAT16}))
//       .OP_END_FACTORY_REG(MyOp);
//
// Only enough is provided here to make that COMPILE. ``ge::DataType`` (the
// ``DT_*`` enumerators) and the ``ge::`` / ``gert::`` types come from
// ``gert_ge_minimal.hpp``, included directly below; this header must NOT redeclare them.
#ifndef PYPTO_EXPORT_FIXTURE_REGISTER_REGISTER_H
#define PYPTO_EXPORT_FIXTURE_REGISTER_REGISTER_H

#include <initializer_list>
#include <vector>

#include "gert_ge_minimal.hpp"

namespace ge {

// Minimal ``ge::Operator`` (graph/operator.h). The domi onnx-plugin ``ParseParam`` reads the source
// op's ONNX ``"attribute"`` JSON off ``op_src`` via ``GetAttr(name, AscendString&)`` and writes each
// parsed attr onto ``op_dest`` via the typed ``SetAttr`` overloads the generated ParseParam emits
// (Int -> int64_t, Float -> float, String -> const char* [``.c_str()``], ListInt -> vector<int64_t>).
// Compile-only: the stub bodies never run, so ``GetAttr`` just reports "absent".
class Operator {
public:
    graphStatus GetAttr(const char*, AscendString&) const { return GRAPH_FAILED; }
    void SetAttr(const char*, int64_t) {}
    void SetAttr(const char*, float) {}
    void SetAttr(const char*, const char*) {}
    void SetAttr(const char*, const std::vector<int64_t>&) {}
};

// ``TensorType({DT_FLOAT16, ...})`` describes the set of dtypes an IO accepts.
// The fixture only needs it to construct from a brace-list of ``ge::DataType``.
class TensorType {
public:
    TensorType(std::initializer_list<DataType> dtypes) : dtypes_(dtypes) {}

    const std::vector<DataType>& GetDataTypes() const { return dtypes_; }

private:
    std::vector<DataType> dtypes_;
};

// Chainable builder returned by ``REG_OP``. Each ``.INPUT`` / ``.OUTPUT`` takes
// the bare IO identifier (passed through the macro as a no-op) plus a
// ``TensorType`` and returns ``*this`` so the calls chain.
class OpReg {
public:
    OpReg& Io(const TensorType&) { return *this; }
    OpReg& EndFactory() { return *this; }
};

} // namespace ge

// ``domi`` registration surface for the onnx-plugin ParseParam TU. The generated plugin emits, inside
// ``namespace domi { ... }``:
//
//   Status ParseParamX(const ge::Operator& op_src, ge::Operator& op_dest) { ...; return SUCCESS; }
//   REGISTER_CUSTOM_OP("X").FrameworkType(ONNX).OriginOpType("...").ParseParamsByOperatorFn(ParseParamX);
//
// Only enough to make that COMPILE (never run).
namespace domi {

// ``domi::Status`` / ``SUCCESS`` / ``FAILED`` (register/register_error_codes.h): the ParseParam fn's
// return type and the two values it returns — ``SUCCESS`` on a parsed node, ``FAILED`` when the attribute
// JSON cannot be read. Values match the real ``DECLARE_ERRORNO`` definitions.
using Status = uint32_t;
inline constexpr Status SUCCESS = 0;
inline constexpr Status FAILED = 0xFFFFFFFFU;

// ``domi::FrameworkType`` — only the ``ONNX`` enumerator the generated ``.FrameworkType(ONNX)`` names is
// needed. The type is spelled ``FmkType`` (not ``FrameworkType``) so it does not clash with the builder's
// ``FrameworkType`` member function below; the enumerators still leak into ``domi`` (unscoped enum).
enum FmkType {
    CAFFE = 0,
    MINDSPORE = 1,
    TENSORFLOW = 3,
    ANDROID_NN = 4,
    ONNX = 5,
};

// The ParseParam callback signature the plugin registers. Real Ascend uses a ``std::function`` typedef;
// a plain function pointer is enough to type-check the ``.ParseParamsByOperatorFn(ParseParamX)`` call
// (the generated free function decays to this pointer).
using ParseParamByOpFunc = Status (*)(const ge::Operator&, ge::Operator&);

// Chainable registrar returned by ``REGISTER_CUSTOM_OP``. Each builder call returns ``*this`` so the
// ``.FrameworkType(...).OriginOpType(...).ParseParamsByOperatorFn(...)`` chain is one namespace-scope
// static-object definition.
class OpRegistrationData {
public:
    explicit OpRegistrationData(const char*) {}
    OpRegistrationData& FrameworkType(FmkType) { return *this; }
    OpRegistrationData& OriginOpType(const char*) { return *this; }
    OpRegistrationData& ParseParamsByOperatorFn(ParseParamByOpFunc) { return *this; }
};

} // namespace domi

// ``inN`` / ``outK`` are bare C++ identifiers inside REG_OP, not strings; the
// macros swallow the identifier and forward the ``TensorType`` to the builder.
// ``REG_OP`` expands to a uniquely-named ``static`` object definition so the
// whole ``REG_OP(X).INPUT(...)...OP_END_FACTORY_REG(X);`` chain is a valid
// declaration at namespace scope.
#define PYPTO_REG_OP_CONCAT_INNER(a, b) a##b
#define PYPTO_REG_OP_CONCAT(a, b) PYPTO_REG_OP_CONCAT_INNER(a, b)
#define REG_OP(op_type) static ::ge::OpReg PYPTO_REG_OP_CONCAT(g_pypto_reg_op_##op_type##_, __LINE__) = ::ge::OpReg()
#define INPUT(name, tensor_type) Io(tensor_type)
#define OUTPUT(name, tensor_type) Io(tensor_type)
#define OP_END_FACTORY_REG(op_type) EndFactory()

// ``REGISTER_CUSTOM_OP("X")`` expands to a uniquely-named ``static`` registrar object so the whole
// ``REGISTER_CUSTOM_OP(...).FrameworkType(...)...ParseParamsByOperatorFn(...);`` chain is a valid
// declaration at namespace (``domi``) scope.
#define REGISTER_CUSTOM_OP(name)                                              \
    static ::domi::OpRegistrationData PYPTO_REG_OP_CONCAT(g_pypto_custom_op_, \
                                                          __LINE__) = ::domi::OpRegistrationData(name)

#endif // PYPTO_EXPORT_FIXTURE_REGISTER_REGISTER_H
