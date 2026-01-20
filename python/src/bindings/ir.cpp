/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "bindings.h"
#include "ir/builder/ir_context.h"
#include "ir/builder/ir_builder.h"
#include "ir/function.h"
#include "ir/object.h"
#include "ir/operation_base.h"
#include "ir/program.h"
#include "ir/statement.h"
#include "ir/type.h"
#include "ir/utils.h"
#include "ir/utils_defop.h"
#include "ir/value.h"
#include "ir/block_call.h"
namespace py = pybind11;

namespace pto {

static void IrBindEnum(py::module &m) {
    py::enum_<ObjectType>(m, "ObjectType")
        .value("Program", ObjectType::Program)
        .value("Function", ObjectType::Function)
        .value("Statement", ObjectType::Statement)
        .value("Operation", ObjectType::Operation)
        .value("Value", ObjectType::Value)
        .value("Memory", ObjectType::Memory);

    py::enum_<DataType>(m, "DataType")
        .value("bool", DataType::BOOL)
        .value("int8", DataType::INT8)
        .value("int16", DataType::INT16)
        .value("int32", DataType::INT32)
        .value("int64", DataType::INT64)
        .value("uint8", DataType::UINT8)
        .value("uint16", DataType::UINT16)
        .value("uint32", DataType::UINT32)
        .value("uint64", DataType::UINT64)
        .value("float16", DataType::FP16)
        .value("bfloat16", DataType::BF16)
        .value("float32", DataType::FP32)
        .value("float", DataType::FP32)
        .value("float64", DataType::FP64)
        .value("double", DataType::FP64)
        .value("float8_e4m3fn", DataType::FP8_E4M3FN)
        .value("float8_e5m2", DataType::FP8_E5M2)
        .export_values()
        .def("bits", [](DataType dtype) { return DTypeInfoOf(dtype).bits; })
        .def("bytes", [](DataType dtype) { return DTypeInfoOf(dtype).bytes; })
        .def("is_float", [](DataType dtype) { return DTypeInfoOf(dtype).isFloat; })
        .def("__str__", [](DataType dtype) { return DTypeInfoOf(dtype).name; });

    py::enum_<FunctionKind>(m, "FunctionKind")
        .value("ControlFlow", FunctionKind::ControlFlow)
        .value("DataFlow", FunctionKind::DataFlow)
        .value("Block", FunctionKind::Block);

    py::enum_<StatementKind>(m, "StatementKind")
        .value("Compound", StatementKind::Compound)
        .value("Op", StatementKind::Op)
        .value("For", StatementKind::For)
        .value("If", StatementKind::If)
        .value("Yield", StatementKind::Yield)
        .value("Call", StatementKind::Call)
        .value("Return", StatementKind::Return);

    py::enum_<ValueKind>(m, "ValueKind")
        .value("Scalar", ValueKind::Scalar)
        .value("Tensor", ValueKind::Tensor)
        .value("Tile", ValueKind::Tile);

    py::enum_<MemSpaceKind>(m, "MemSpaceKind")
        .value("DDR", MemSpaceKind::DDR)
        .value("L2", MemSpaceKind::L2)
        .value("UB", MemSpaceKind::UB)
        .value("L1", MemSpaceKind::L1)
        .value("L0A", MemSpaceKind::L0A)
        .value("L0B", MemSpaceKind::L0B)
        .value("L0C", MemSpaceKind::L0C)
        .value("REG", MemSpaceKind::REG)
        .value("SHMEM", MemSpaceKind::SHMEM);

    py::enum_<Format>(m, "Format").value("ND", Format::ND).value("NZ", Format::NZ);

    py::enum_<Opcode>(m, "Opcode")
#define DEFOP DEFOP_OPCODE_PYENUM
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP
        ;
}

static void IrBindType(py::module &m) {
    py::class_<Type>(m, "Type")
        .def_property_readonly("dtype", &Type::GetDataType)
        .def("elem_size", [](const Type &self) { return self.GetDataTypeSize(); })
        .def("size", &Type::GetTypeSize);

    py::class_<ScalarType, Type>(m, "ScalarType")
        .def(py::init<DataType>(), py::arg("dtype"))
        .def("size", &ScalarType::GetTypeSize);

    py::class_<pto::TileType, Type>(m, "TileType")
        .def(py::init<DataType, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("shape"))
        .def_property_readonly("shape", &TileType::GetShape)
        .def("size", &TileType::GetTypeSize);

    py::class_<pto::TensorType, Type>(m, "TensorType")
        .def(py::init<DataType>(), py::arg("dtype"))
        .def("size", &TensorType::GetTypeSize);
}

static void IrBindObjClass(py::module &m) {
    py::class_<Object, std::shared_ptr<Object>>(m, "Object")
        .def_property_readonly("id", &Object::GetID)
        .def_property_readonly("name", &Object::GetName)
        .def_property_readonly("type", &Object::GetObjectType)
        .def("properties", py::overload_cast<>(&Object::Attributes, py::const_));
}

static void IrBindValue(py::module &m) {
    py::class_<Value, Object, std::shared_ptr<Value>>(m, "Value")
        .def("ssaname", &Value::GetSSAName)
        .def_property_readonly("kind", &Value::GetValueKind)
        .def_property_readonly("type", &Value::GetType);

    py::class_<ScalarValue, Value, std::shared_ptr<ScalarValue>>(m, "Scalar")
        .def(py::init([](DataType dtype, py::object val, std::string name = "") {
            if (val.is_none()) {
                return std::make_shared<ScalarValue>(dtype, name);
            } else if (py::isinstance<py::int_>(val)) {
                auto intVal = py::cast<int64_t>(val);
                if (DTypeInfoOf(dtype).isFloat) {
                    return std::make_shared<ScalarValue>(
                        dtype, name, ScalarValueKind::Immediate, static_cast<double>(intVal));
                } else {
                    return std::make_shared<ScalarValue>(dtype, name, ScalarValueKind::Immediate, intVal);
                }
            } else if (py::isinstance<py::float_>(val)) {
                auto fval = py::cast<double>(val);
                if (DTypeInfoOf(dtype).isFloat) {
                    return std::make_shared<ScalarValue>(dtype, name, ScalarValueKind::Immediate, fval);
                } else {
                    return std::make_shared<ScalarValue>(
                        dtype, name, ScalarValueKind::Immediate, static_cast<int64_t>(fval));
                }
            } else {
                throw py::type_error("Unsupported value type for key: " + name);
            }
        }),
            py::arg("type"), py::arg("val"), py::arg("name") = "")
        .def_property_readonly("type", &ScalarValue::GetType)
        .def("is_constant", &ScalarValue::HasImmediateValue)
        .def("value", [](const ScalarValue &self) {
            if (self.HasImmediateValue()) {
                const auto &val = self.GetImmediateValue();
                if (self.GetDataType() == DataType::BOOL) {
                    return py::cast(static_cast<bool>(std::get<int64_t>(val)));
                } else if (DTypeInfoOf(self.GetDataType()).isFloat) {
                    return py::cast(std::get<double>(val));
                } else {
                    return py::cast(std::get<int64_t>(val));
                }
            } else {
                return py::cast(self.GetSSAName());
            }
        });

    py::class_<Memory, Object, std::shared_ptr<Memory>>(m, "Memory")
        .def(py::init<uint64_t, MemSpaceKind>(), py::arg("size"), py::arg("kind"))
        .def_property("size", &Memory::GetSize, &Memory::SetSize)
        .def_property("kind", &Memory::GetSpace, &Memory::SetSpace)
        .def_property("addr", &Memory::GetAddr, &Memory::SetAddr);

    py::class_<TileValue, Value, std::shared_ptr<TileValue>>(m, "Tile")
        .def(py::init<std::string, std::vector<ScalarValuePtr>, std::vector<int64_t>, std::vector<int64_t>,
                 ScalarValuePtr, DataType, MemoryPtr>(),
            py::arg("name"), py::arg("valid_shape"), py::arg("shape"), py::arg("strides"), py::arg("offset"),
            py::arg("dtype"), py::arg("memory"))
        .def_property("shape", &TileValue::GetShape, &TileValue::SetShape)
        .def_property("strides", &TileValue::GetStrides, &TileValue::SetStrides)
        .def_property("offset", &TileValue::GetStartOffset, &TileValue::SetStartOffset)
        .def_property("memory", &TileValue::GetMemory, &TileValue::SetMemory);

    py::class_<TensorValue, Value, std::shared_ptr<TensorValue>>(m, "Tensor")
        .def(py::init<std::vector<ScalarValuePtr>, DataType, std::string, Format>(), py::arg("shape"), py::arg("dtype"),
            py::arg("name"), py::arg("format"))
        .def_property_readonly("shape", &TensorValue::GetShape)
        .def_property("format", &TensorValue::GetFormat, &TensorValue::SetFormat);
}

static void IrBindOperation(py::module &m) {
    py::class_<Operation, Object, std::shared_ptr<Operation>>(m, "Operation")
        .def_property_readonly("opcode", &Operation::GetOpcode)
        .def("ioperands", &Operation::GetIOperands, py::return_value_policy::reference_internal)
        .def("ooperands", &Operation::GetOOperands, py::return_value_policy::reference_internal);

    py::class_<UnaryOp, Operation, std::shared_ptr<UnaryOp>>(m, "UnaryOp")
        .def_property("combine_axis", &UnaryOp::GetCombineAxis, &UnaryOp::SetCombineAxis);
    py::class_<BinaryOp, Operation, std::shared_ptr<BinaryOp>>(m, "BinaryOp")
        .def_property("combine_axis", &BinaryOp::GetCombineAxis, &BinaryOp::SetCombineAxis);
    py::class_<BinaryScalarMixOp, Operation, std::shared_ptr<BinaryScalarMixOp>>(m, "BinaryScalarMixOp")
        .def_property("combine_axis", &BinaryScalarMixOp::GetCombineAxis, &BinaryScalarMixOp::SetCombineAxis)
        .def_property("reverse", &BinaryScalarMixOp::GetReverse, &BinaryScalarMixOp::SetReverse);

    py::class_<UnaryScalarOp, Operation, std::shared_ptr<UnaryScalarOp>>(m, "UnaryScalarOp");
    py::class_<BinaryScalarOp, Operation, std::shared_ptr<BinaryScalarOp>>(m, "BinaryScalarOp");
}

static void IrBindStatement(py::module &m) {
    py::class_<Statement, Object, std::shared_ptr<Statement>>(m, "Statement")
        .def_property_readonly("kind", &Statement::GetKind);

    py::class_<CompoundStatement, Statement, std::shared_ptr<CompoundStatement>>(m, "CompoundStatement")
        .def(py::init<>())
        .def(py::init<CompoundStatementPtr>(), py::arg("parent"))
        .def_property("parent", &CompoundStatement::GetParent, &CompoundStatement::SetParent)
        .def(
            "stmts", [](CompoundStatement &self) { return self.GetStatements(); },
            py::return_value_policy::reference_internal)
        .def(
            "vars", [](CompoundStatement &self) { return self.GetEnvTable(); },
            py::return_value_policy::reference_internal);

    py::class_<OpStatement, Statement, std::shared_ptr<OpStatement>>(m, "OpStatement")
        .def(py::init<>())
        .def(
            "operations", [](OpStatement &self) { return self.Operations(); },
            py::return_value_policy::reference_internal);

    py::class_<YieldStatement, Statement, std::shared_ptr<YieldStatement>>(m, "YieldStatement")
        .def(py::init<>())
        .def(
            "values", [](const YieldStatement &self) { return self.Values(); },
            py::return_value_policy::reference_internal);

    py::class_<ForStatement, Statement, std::shared_ptr<ForStatement>>(m, "ForStatement")
        .def(py::init<ScalarValuePtr, ScalarValuePtr, ScalarValuePtr, ScalarValuePtr>(), py::arg("var"),
            py::arg("start"), py::arg("end"), py::arg("step"))
        .def_property_readonly("var", &ForStatement::GetIterationVar)
        .def_property_readonly("start", &ForStatement::GetStart)
        .def_property_readonly("end", &ForStatement::GetEnd)
        .def_property_readonly("step", &ForStatement::GetStep)
        .def(
            "stmts", [](ForStatement &self) { return self.GetCompound(); }, py::return_value_policy::reference_internal)
        .def("yield", [](ForStatement &self) { return self.Yield(); });

    py::class_<IfStatement, Statement, std::shared_ptr<IfStatement>>(m, "IfStatement")
        .def(py::init<ScalarValuePtr>(), py::arg("cond"))
        .def("condition", &IfStatement::GetCondition)
        .def(
            "then_stmts", [](IfStatement &self) { return self.GetThenCompound(); },
            py::return_value_policy::reference_internal)
        .def(
            "else_stmts", [](IfStatement &self) { return self.GetElseCompound(); },
            py::return_value_policy::reference_internal)
        .def("results", [](const IfStatement &self) { return self.Results(); });

    py::class_<ReturnStatement, Statement, std::shared_ptr<ReturnStatement>>(m, "ReturnStatement")
        .def(py::init<>())
        .def(
            "values", [](ReturnStatement &self) { return self.Values(); }, py::return_value_policy::reference_internal);
}

static void IrBindModule(py::module &m) {
    py::class_<ProgramModule, Object, std::shared_ptr<ProgramModule>>(m, "module")
        .def(py::init<const std::string &>(), py::arg("name"))
        .def_property("entry", &ProgramModule::GetProgramEntry, &ProgramModule::SetProgramEntry)
        .def("functions", &ProgramModule::GetFunctions, py::return_value_policy::reference_internal)
        .def("add_function", &ProgramModule::AddFunction, py::arg("function"));
}

static void IrBindFunction(py::module &m) {
    py::class_<FunctionSignature>(m, "FunctionSignature")
        .def(py::init<>())
        .def(py::init<const std::vector<pto::TensorValuePtr> &,
                      const std::vector<pto::TensorValuePtr> &>(),
                      py::arg("input_args"), py::arg("output_args"))
        .def_readwrite("arguments", &FunctionSignature::arguments, py::return_value_policy::reference_internal)
        .def_readwrite("returns", &FunctionSignature::results, py::return_value_policy::reference_internal);

    py::class_<Function, Object, std::shared_ptr<Function>>(m, "Function")
        .def(py::init([](FunctionKind kind, const std::string &name, FunctionSignature sig) {
            return std::make_shared<Function>(name, kind, sig);
        }),
            py::arg("kind"), py::arg("name"), py::arg("sig"))
        .def(
            "stmts", [](Function &self) { return self.GetCompound(); }, py::return_value_policy::reference_internal)
        .def_property_readonly("kind", &Function::GetKind);
}

// Scalar operations (from operation.def)
static void IrBuilderBindScalarOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_unary_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr in, ScalarValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateUnaryScalarOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_binary_scalar_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr lhs, ScalarValuePtr rhs, ScalarValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryScalarOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"));
}

// Tile operations (from tile_graph.def)
static void IrBuilderBindUnaryOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_unary_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateUnaryOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_unary_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateUnaryWithTempOp(opcode, in, out, temp));
}, py::arg("opcode"), py::arg("in"), py::arg("out"), py::arg("temp"))
        .def("create_expand_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateExpandOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_transpose_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateTransposeOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_transpose_vnchw_conv_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateTransposeVnchwConvOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_onehot_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateOnehotOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_cast_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateCastOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_cumsum_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateCumSumOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_convert_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateConvertOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_duplicate_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateDuplicateOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_any_data_copy_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateAnyDataCopyOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_argsort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateArgSortOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"));
}

static void IrBuilderBindBinaryOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_binary_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"))
        .def("create_binary_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryWithTempOp(opcode, lhs, rhs, out, temp));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"), py::arg("temp"))
        .def("create_binary_scalar_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBinaryScalarMixOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("scalar"), py::arg("out"))
        .def("create_scatter_elements_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src0, TileValuePtr src1, ScalarValuePtr scatter, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateScatterElementsOp(opcode, src0, src1, scatter, out));
}, py::arg("opcode"), py::arg("src0"), py::arg("src1"), py::arg("scatter"), py::arg("out"))
        .def("create_scatter_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src0, TileValuePtr src1, TileValuePtr src2, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateScatterOp(opcode, src0, src1, src2, out));
}, py::arg("opcode"), py::arg("src0"), py::arg("src1"), py::arg("src2"), py::arg("out"))
        .def("create_gather_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateGatherOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"))
        .def("create_gather_extended_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateGatherExtendedOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"))
        .def("create_broadcast_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateBroadcastWithTempOp(opcode, lhs, rhs, out, temp));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"), py::arg("temp"))
        .def("create_fused_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateFusedOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"))
        .def("create_load_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateLoadOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"))
        .def("create_copy_in_out_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateCopyInOutOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"));
}

static void IrBuilderBindScalarInputOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_range_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr start, ScalarValuePtr step, ScalarValuePtr size, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateRangeOp(opcode, start, step, size, out));
}, py::arg("opcode"), py::arg("start"), py::arg("step"), py::arg("size"), py::arg("out"))
        .def("create_vec_dup_op",
            [](IRBuilder &self, Opcode opcode, ScalarValuePtr value, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateVecDupOp(opcode, value, out));
}, py::arg("opcode"), py::arg("value"), py::arg("out"))
        .def("create_pow_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, ScalarValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreatePowOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"));
}

static void IrBuilderBindScalarListOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_matmul_load_op",
            [](IRBuilder &self, Opcode opcode, TensorValuePtr input, const std::vector<ScalarValuePtr> &offsets, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulLoadOp(opcode, input, offsets, out));
}, py::arg("opcode"), py::arg("input"), py::arg("offsets"), py::arg("out"))
        .def("create_matmul_extract_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulExtractOp(opcode, input, offsets, out));
}, py::arg("opcode"), py::arg("input"), py::arg("offsets"), py::arg("out"))
        .def("create_matmul_store_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets, TensorValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulStoreOp(opcode, input, offsets, out));
}, py::arg("opcode"), py::arg("input"), py::arg("offsets"), py::arg("out"))
        .def("create_matmul_bias_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulBiasOp(opcode, input, offsets, out));
}, py::arg("opcode"), py::arg("input"), py::arg("offsets"), py::arg("out"))
        .def("create_matmul_quant_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input, const std::vector<ScalarValuePtr> &offsets, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulQuantOp(opcode, input, offsets, out));
}, py::arg("opcode"), py::arg("input"), py::arg("offsets"), py::arg("out"))
        .def("create_ub_copy_in_op",
            [](IRBuilder &self, Opcode opcode, TensorValuePtr src, const std::vector<ScalarValuePtr> &offset, TileValuePtr dst) {
                return std::static_pointer_cast<Operation>(self.CreateUBCopyInOp(opcode, src, offset, dst));
}, py::arg("opcode"), py::arg("src"), py::arg("offset"), py::arg("dst"))
        .def("create_ub_copy_out_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src, const std::vector<ScalarValuePtr> &offset, TensorValuePtr dst) {
                return std::static_pointer_cast<Operation>(self.CreateUBCopyOutOp(opcode, src, offset, dst));
}, py::arg("opcode"), py::arg("src"), py::arg("offset"), py::arg("dst"));
}

static void IrBuilderBindMatmulOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_matmul_mmad_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulMmadOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"))
        .def("create_matmul_acc_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr lhs, TileValuePtr rhs, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMatmulAccOp(opcode, lhs, rhs, out));
}, py::arg("opcode"), py::arg("lhs"), py::arg("rhs"), py::arg("out"));
}

static void IrBuilderBindReduceOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_reduce_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateReduceOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_reduce_with_temp_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateReduceWithTempOp(opcode, in, out, temp));
}, py::arg("opcode"), py::arg("in"), py::arg("out"), py::arg("temp"));
}

static void IrBuilderBindGatherMultiInputOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_gather_in_ub_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateGatherInUBOp(opcode, input1, input2, input3, out));
}, py::arg("opcode"), py::arg("input1"), py::arg("input2"), py::arg("input3"), py::arg("out"))
        .def("create_gather_in_l1_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateGatherInL1Op(opcode, input1, input2, input3, out));
}, py::arg("opcode"), py::arg("input1"), py::arg("input2"), py::arg("input3"), py::arg("out"));
}

static void IrBuilderBindIndexOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_index_out_cast_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr src, TileValuePtr index, TileValuePtr dst, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateIndexOutCastOp(opcode, src, index, dst, out));
}, py::arg("opcode"), py::arg("src"), py::arg("index"), py::arg("dst"), py::arg("out"))
        .def("create_index_add_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, ScalarValuePtr alpha, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateIndexAddOp(opcode, input1, input2, input3, alpha, out));
}, py::arg("opcode"), py::arg("input1"), py::arg("input2"), py::arg("input3"), py::arg("alpha"), py::arg("out"));
}

static void IrBuilderBindSortOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_topk_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out1, TileValuePtr out2) {
                return std::static_pointer_cast<Operation>(self.CreateTopKOp(opcode, in, out1, out2));
}, py::arg("opcode"), py::arg("in"), py::arg("out1"), py::arg("out2"))
        .def("create_bitsort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateBitSortOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_mrgsort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateMrgSortOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_extract_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr in, TileValuePtr out) {
                return std::static_pointer_cast<Operation>(self.CreateExtractOp(opcode, in, out));
}, py::arg("opcode"), py::arg("in"), py::arg("out"))
        .def("create_tiled_mrg_sort_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr input3, TileValuePtr input4, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateTiledMrgSortOp(opcode, input1, input2, input3, input4, out, temp));
}, py::arg("opcode"), py::arg("input1"), py::arg("input2"), py::arg("input3"), py::arg("input4"), py::arg("out"), py::arg("temp"));
}

static void IrBuilderBindWhereTernaryOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_ternary_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, TileValuePtr input, TileValuePtr other, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateTernaryOp(opcode, condition, input, other, out, temp));
}, py::arg("opcode"), py::arg("condition"), py::arg("input"), py::arg("other"), py::arg("out"), py::arg("temp"))
        .def("create_where_ts_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, TileValuePtr input, ScalarValuePtr other, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateWhereTSOp(opcode, condition, input, other, out, temp));
}, py::arg("opcode"), py::arg("condition"), py::arg("input"), py::arg("other"), py::arg("out"), py::arg("temp"))
        .def("create_where_st_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, ScalarValuePtr input, TileValuePtr other, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateWhereSTOp(opcode, condition, input, other, out, temp));
}, py::arg("opcode"), py::arg("condition"), py::arg("input"), py::arg("other"), py::arg("out"), py::arg("temp"))
        .def("create_where_ss_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr condition, ScalarValuePtr input, ScalarValuePtr other, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateWhereSSOp(opcode, condition, input, other, out, temp));
}, py::arg("opcode"), py::arg("condition"), py::arg("input"), py::arg("other"), py::arg("out"), py::arg("temp"));
}

static void IrBuilderBindCompareOps(py::class_<IRBuilder> &irBuilder) {
    irBuilder
        .def("create_compare_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, TileValuePtr input2, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateCompareOp(opcode, input1, input2, out, temp));
}, py::arg("opcode"), py::arg("input1"), py::arg("input2"), py::arg("out"), py::arg("temp"))
        .def("create_compare_scalar_op",
            [](IRBuilder &self, Opcode opcode, TileValuePtr input1, ScalarValuePtr input2, TileValuePtr out, TileValuePtr temp) {
                return std::static_pointer_cast<Operation>(self.CreateCompareScalarOp(opcode, input1, input2, out, temp));
}, py::arg("opcode"), py::arg("input1"), py::arg("input2"), py::arg("out"), py::arg("temp"));
}

static void IrBuilderBindOp(py::class_<IRBuilder> &irBuilder) {
    IrBuilderBindScalarOps(irBuilder);
    IrBuilderBindUnaryOps(irBuilder);
    IrBuilderBindBinaryOps(irBuilder);
    IrBuilderBindScalarInputOps(irBuilder);
    IrBuilderBindScalarListOps(irBuilder);
    IrBuilderBindMatmulOps(irBuilder);
    IrBuilderBindReduceOps(irBuilder);
    IrBuilderBindGatherMultiInputOps(irBuilder);
    IrBuilderBindIndexOps(irBuilder);
    IrBuilderBindSortOps(irBuilder);
    IrBuilderBindWhereTernaryOps(irBuilder);
    IrBuilderBindCompareOps(irBuilder);
}

static void IrBindBuilder(py::module &m) {
    py::class_<IRBuilderContext>(m, "IrBuilderContext")
        .def(py::init<>())
        .def("push_scope", &IRBuilderContext::PushScope)
        .def("pop_scope", &IRBuilderContext::PopScope);

    auto irBuilder =
        py::class_<IRBuilder>(m, "IrBuilder")
            .def(py::init<>())
            .def("create_function", &IRBuilder::CreateFunction, py::arg("name"), py::arg("kind"), py::arg("sig"))
            .def("create_tensor", &IRBuilder::CreateTensor, py::arg("ctx"), py::arg("shape"), py::arg("dtype"),
                py::arg("name") = "")
            .def("create_tile", &IRBuilder::CreateTile, py::arg("ctx"), py::arg("shape"), py::arg("dtype"),
                py::arg("name") = "")
            .def("create_scalar", &IRBuilder::CreateScalar, py::arg("ctx"), py::arg("dtype"), py::arg("name") = "")
            .def(
                "create_const",
                [](IRBuilder &self, IRBuilderContext &ctx, py::object value, const std::string &name) {
                    if (py::isinstance<py::int_>(value)) {
                        return self.CreateConst(ctx, py::cast<int64_t>(value), name);
                    } else if (py::isinstance<py::float_>(value)) {
                        return self.CreateConst(ctx, py::cast<double>(value), name);
                    } else {
                        throw py::type_error("Unsupported const value type");
                    }
                },
                py::arg("ctx"), py::arg("value"), py::arg("name") = "")
            .def("create_op", &IRBuilder::CreateOpStmt, py::arg("ctx"))
            .def("create_for", &IRBuilder::CreateForStmt, py::arg("ctx"), py::arg("var"), py::arg("start"),
                py::arg("end"), py::arg("step"))
            .def("create_if", &IRBuilder::CreateIfStmt, py::arg("ctx"), py::arg("cond"))
            .def("create_return", &IRBuilder::CreateReturn, py::arg("ctx"), py::arg("values"))
            .def(
                "enter_function",
                [](IRBuilder &self, IRBuilderContext &ctx, std::shared_ptr<Function> func) {
                    self.EnterFunctionBody(ctx, func);
                },
                py::arg("ctx"), py::arg("func"))
            .def(
                "enter_for",
                [](IRBuilder &self, IRBuilderContext &ctx, ForStatementPtr stmt) { self.EnterForBody(ctx, stmt); },
                py::arg("ctx"), py::arg("stmt"))
            .def(
                "enter_if_then",
                [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.EnterIfThen(ctx, stmt); },
                py::arg("ctx"), py::arg("stmt"))
            .def(
                "enter_if_else",
                [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.EnterIfElse(ctx, stmt); },
                py::arg("ctx"), py::arg("stmt"))
            .def(
                "exit_for",
                [](IRBuilder &self, IRBuilderContext &ctx, ForStatementPtr stmt) { self.ExitForStatement(ctx, stmt); },
                py::arg("ctx"), py::arg("stmt"))
            .def(
                "exit_if",
                [](IRBuilder &self, IRBuilderContext &ctx, IfStatementPtr stmt) { self.ExitIfStatement(ctx, stmt); },
                py::arg("ctx"), py::arg("stmt"))
            .def("emit", &IRBuilder::Emit, py::arg("ctx"), py::arg("operation"));

    IrBuilderBindOp(irBuilder);
}

void IrBindBlockCall(py::module &m) {
    m.def("call_block", [](const pto::FunctionPtr &blockFuncPtr,
                           const std::vector<npu::tile_fwk::Tensor>& inputTensors,
                           const std::vector<npu::tile_fwk::Tensor>& outputTensors,
                           const std::vector<npu::tile_fwk::SymbolicScalar>& indices) {
        // 转换为 reference_wrapper
        std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> inputRefs;
        std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> outputRefs;
        for (const auto& t : inputTensors) inputRefs.emplace_back(t);
        for (const auto& t : outputTensors) outputRefs.emplace_back(t);
        // 调用 C++ 的 CallBlock (通过 blockName)
        return CallBlock(blockFuncPtr, inputRefs, outputRefs, indices);
    }, py::arg("block_func_ptr"),
       py::arg("input_tensors"),
       py::arg("output_tensors"),
       py::arg("indices"),
       "Call a registered block by name (PascalCase).");
}
} // namespace pto

namespace pypto {
void BindIr(py::module &m) {
    auto ir = m.def_submodule("ir", "IR module");
    pto::IrBindEnum(ir);
    pto::IrBindObjClass(ir);
    pto::IrBindType(ir);
    pto::IrBindValue(ir);
    pto::IrBindOperation(ir);
    pto::IrBindStatement(ir);
    pto::IrBindFunction(ir);
    pto::IrBindModule(ir);
    pto::IrBindBuilder(ir);
    pto::IrBindBlockCall(ir);
}
} // namespace pypto
