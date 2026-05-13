/*
 * Copyright (c) PyPTO Contributors.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file error.cpp
 * \brief Python bindings for PyPTO error classes
 */

#include "bindings.h"

#include "ir/expr.h"
#include "ir/memref.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/transforms/printer.h"

namespace pypto {
namespace ir {

void BindDType(py::module& m)
{
    py::class_<ir::DataType>(m, "DataType", "Enumeration of available data types")
        .def_readonly_static("BOOL", &ir::DataType::BOOL)
        .def_readonly_static("INT4", &ir::DataType::INT4)
        .def_readonly_static("INT8", &ir::DataType::INT8)
        .def_readonly_static("INT16", &ir::DataType::INT16)
        .def_readonly_static("INT32", &ir::DataType::INT32)
        .def_readonly_static("INT64", &ir::DataType::INT64)
        .def_readonly_static("UINT4", &ir::DataType::UINT4)
        .def_readonly_static("UINT8", &ir::DataType::UINT8)
        .def_readonly_static("UINT16", &ir::DataType::UINT16)
        .def_readonly_static("UINT32", &ir::DataType::UINT32)
        .def_readonly_static("UINT64", &ir::DataType::UINT64)
        .def_readonly_static("FP4", &ir::DataType::FP4)
        .def_readonly_static("FP8E4M3FN", &ir::DataType::FP8E4M3FN)
        .def_readonly_static("FP8E5M2", &ir::DataType::FP8E5M2)
        .def_readonly_static("FP16", &ir::DataType::FP16)
        .def_readonly_static("FP32", &ir::DataType::FP32)
        .def_readonly_static("FP64", &ir::DataType::FP64)
        .def_readonly_static("BF16", &ir::DataType::BF16)
        .def_readonly_static("HF4", &ir::DataType::HF4)
        .def_readonly_static("HF8", &ir::DataType::HF8)
        .def_readonly_static("INDEX", &ir::DataType::INDEX)
        .def("bits", &ir::DataType::GetBit, "Get the size in bits of this data type.")
        .def("c_type", &ir::DataType::ToCTypeString, "Get C style type string for code generation.")
        .def("is_float", &ir::DataType::IsFloat, "Check if this data type is a floating point type.")
        .def("is_signed", &ir::DataType::IsSignedInt, "Check if this data type is a signed integer type.")
        .def("is_unsigned", &ir::DataType::IsUnsignedInt, "Check if this data type is an unsigned integer type.")
        .def("is_int", &ir::DataType::IsInt, "Check if this data type is an integer type.")
        .def("__int__", &ir::DataType::Code, "Get the underlying type code.")
        .def("__eq__", &ir::DataType::operator==, py::arg("other"))
        .def("__ne__", &ir::DataType::operator!=, py::arg("other"))
        .def("__repr__", &ir::DataType::ToString)
        .def("__str__", &ir::DataType::ToString);
}

void BindSpan(py::module& m)
{
    py::class_<ir::Span>(m, "Span", "Source location information tracking file, line, and column positions")
        .def(
            py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
            py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1, "Create a source span")
        .def("is_unknown", &ir::Span::IsUnknown, "Check if the span is unknown")
        .def_static("unknown", &ir::Span::Unknown, "Create an unknown span", py::return_value_policy::reference)
        .def("__repr__", &ir::Span::ToString)
        .def("__str__", &ir::Span::ToString)
        .def_property_readonly("filename", &ir::Span::Filename, "Source filename")
        .def_property_readonly("begin_line", &ir::Span::BeginLine, "Beginning line (1-indexed)")
        .def_property_readonly("begin_column", &ir::Span::BeginColumn, "Beginning column (1-indexed)")
        .def_property_readonly("end_line", &ir::Span::EndLine, "Ending line (1-indexed)")
        .def_property_readonly("end_column", &ir::Span::EndColumn, "Ending column (1-indexed)");
}

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& py_class, const FieldDesc& desc)
{
    py_class.def_readonly(desc.name, desc.fieldPtr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& py_class, const DescTuple& descriptors, std::index_sequence<Is...>)
{
    (BindField<ClassType>(py_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& py_class)
{
    constexpr auto descriptors = ClassType::GetFieldDescriptors();
    constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
    BindFieldsImpl<ClassType>(py_class, descriptors, std::make_index_sequence<num_fields>{});
}

void BindExpr(py::module& m)
{
    // clang-format off
    auto irnode = py::class_<IRNode, std::shared_ptr<IRNode>>(m, "IRNode", "Base class for all IR nodes")
        .def("__str__", [](py::object self) -> std::string {
            auto* node = py::cast<IRNode*>(self);
            return PythonPrint(std::shared_ptr<const IRNode>(node, [](const IRNode*) {}), "ir");
        });

    BindFields<IRNode>(irnode);

    auto expr = py::class_<Expr, IRNode, std::shared_ptr<Expr>>(m, "Expr", "Base class for all expressions");
    BindFields<Expr>(expr);

    auto var = py::class_<Var, Expr, std::shared_ptr<Var>>(m, "Var", "Variable reference expression")
        .def(py::init<const std::string&, const TypePtr&, const Span&>(), py::arg("name"),
             py::arg("type"), py::arg("span"),
             "Create a variable reference (memory reference is stored in ShapedType for Tensor/Tile types)"
        );
    BindFields<Var>(var);

    auto iterArg = py::class_<IterArg, Var, std::shared_ptr<IterArg>>(m, "IterArg", "Iteration argument variable")
        .def(py::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(),
             py::arg("name"), py::arg("type"), py::arg("initValue"), py::arg("span"),
             "Create an iteration argument with initial value");
    BindFields<IterArg>(iterArg);

    auto memref = py::class_<MemRef, Expr, std::shared_ptr<MemRef>>(m, "MemRef",
            "Memory reference variable for shaped types (inherits from Var)")
        .def(py::init<MemorySpace, ExprPtr, uint64_t, Span>(),
             py::arg("memory_space"), py::arg("offset"), py::arg("size"), py::arg("span") = Span::Unknown(),
             "Create a memory reference from memory space, offset, and size")
        .def_static("same_allocation", &MemRef::SameAllocation, py::arg("a"), py::arg("b"),
                    "Check if two MemRefs share the same allocation (same base_ Ptr)")
        .def_static("may_alias", &MemRef::MayAlias, py::arg("a"), py::arg("b"),
                    "Check if two MemRefs may alias (same base + overlapping byte ranges)");
    BindFields<MemRef>(memref);

    auto constint = py::class_<ConstInt, Expr, std::shared_ptr<ConstInt>>(m, "ConstInt", "Constant integer expression")
        .def(py::init<int64_t, DataType, const Span&>(),
             py::arg("value"), py::arg("dtype"), py::arg("span"),
             "Create a constant integer expression")
        .def_property_readonly("dtype", &ConstInt::dtype, "Data type of the expression");
    BindFields<ConstInt>(constint);

    auto constfloat = py::class_<ConstFloat, Expr, std::shared_ptr<ConstFloat>>(m, "ConstFloat",
            "Constant float expression")
        .def(py::init<double, DataType, const Span&>(),
             py::arg("value"), py::arg("dtype"), py::arg("span"),
             "Create a constant float expression")
        .def_property_readonly("dtype", &ConstFloat::dtype, "Data type of the expression");
    BindFields<ConstFloat>(constfloat);

    auto constbool_class = py::class_<ConstBool, Expr, std::shared_ptr<ConstBool>>(m, "ConstBool",
            "Constant boolean expression")
        .def(py::init<bool, const Span&>(),
             py::arg("value"), py::arg("span"),
             "Create a constant boolean expression")
        .def_property_readonly("dtype", &ConstBool::dtype, "Data type of the expression (always BOOL)");
    BindFields<ConstBool>(constbool_class);

    auto call = py::class_<Call, Expr, std::shared_ptr<Call>>(m, "Call", "Function call expression")
        .def(py::init<std::string, const std::vector<ExprPtr>&, const Span&>(),
             py::arg("op"), py::arg("args"), py::arg("span"),
             "Create a function call expression");
    BindFields<Call>(call);

    auto make_tuple = py::class_<MakeTuple, Expr, std::shared_ptr<MakeTuple>>(m, "MakeTuple",
            "Tuple construction expression")
        .def(py::init<const std::vector<ExprPtr>&, const Span&>(),
             py::arg("elements"), py::arg("span"),
             "Create a tuple construction expression");
    BindFields<MakeTuple>(make_tuple);

    auto tuple_get_item = py::class_<TupleGetItemExpr, Expr, std::shared_ptr<TupleGetItemExpr>>(m, "TupleGetItemExpr",
            "Tuple element access expression")
        .def(py::init<const ExprPtr&, int, const Span&>(),
             py::arg("tuple"), py::arg("index"), py::arg("span"),
             "Create a tuple element access expression");
    BindFields<TupleGetItemExpr>(tuple_get_item);

    auto binary_expr = py::class_<BinaryExpr, Expr, std::shared_ptr<BinaryExpr>>(m, "BinaryExpr",
        "Base class for binary operations");
    BindFields<BinaryExpr>(binary_expr);

    auto unary_expr = py::class_<UnaryExpr, Expr, std::shared_ptr<UnaryExpr>>(m, "UnaryExpr",
        "Base class for unary operations");
    BindFields<UnaryExpr>(unary_expr);

#define BIND_BINARY_EXPR(OpName, Description)                                                                     \
    py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(m, #OpName, Description)                              \
        .def(py::init<const ExprPtr&, const ExprPtr&, DataType, const Span&>(),                                   \
             py::arg("left"), py::arg("right"), py::arg("dtype"), py::arg("span"),                                \
             "Create " Description);

    // Bind all binary expression nodes
    BIND_BINARY_EXPR(Add, "Addition expression (left + right)")
    BIND_BINARY_EXPR(Sub, "Subtraction expression (left - right)")
    BIND_BINARY_EXPR(Mul, "Multiplication expression (left * right)")
    BIND_BINARY_EXPR(FloorDiv, "Floor division expression (left // right)")
    BIND_BINARY_EXPR(FloorMod, "Floor modulo expression (left % right)")
    BIND_BINARY_EXPR(FloatDiv, "Float division expression (left / right)")
    BIND_BINARY_EXPR(Min, "Minimum expression (min(left, right))")
    BIND_BINARY_EXPR(Max, "Maximum expression (max(left, right))")
    BIND_BINARY_EXPR(Pow, "Power expression (left ** right)")
    BIND_BINARY_EXPR(Eq, "Equality expression (left == right)")
    BIND_BINARY_EXPR(Ne, "Inequality expression (left != right)")
    BIND_BINARY_EXPR(Lt, "Less than expression (left < right)")
    BIND_BINARY_EXPR(Le, "Less than or equal to expression (left <= right)")
    BIND_BINARY_EXPR(Gt, "Greater than expression (left > right)")
    BIND_BINARY_EXPR(Ge, "Greater than or equal to expression (left >= right)")
    BIND_BINARY_EXPR(And, "Logical and expression (left and right)")
    BIND_BINARY_EXPR(Or, "Logical or expression (left or right)")
    BIND_BINARY_EXPR(Xor, "Logical xor expression (left xor right)")
    BIND_BINARY_EXPR(BitAnd, "Bitwise and expression (left & right)")
    BIND_BINARY_EXPR(BitOr, "Bitwise or expression (left | right)")
    BIND_BINARY_EXPR(BitXor, "Bitwise xor expression (left ^ right)")
    BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression (left << right)")
    BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression (left >> right)")

#undef BIND_BINARY_EXPR

#define BIND_UNARY_EXPR(OpName, Description)                                                    \
    py::class_<OpName, UnaryExpr, std::shared_ptr<OpName>>(m, #OpName, Description)                              \
        .def(py::init<const ExprPtr&, DataType, const Span&>(),                                     \
             py::arg("operand"), py::arg("dtype"), py::arg("span"),                                 \
             "Create " Description);

    BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
    BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
    BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
    BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")
    BIND_UNARY_EXPR(Cast, "Cast expression (cast operand to dtype)")

#undef BIND_UNARY_EXPR
    // clang-format on
}

std::any ConvertAttr(const py::object& attr)
{
    if (py::isinstance<DataType>(attr)) {
        return py::cast<DataType>(attr);
    } else if (py::isinstance<MemorySpace>(attr)) {
        return py::cast<MemorySpace>(attr);
    } else if (py::isinstance<TensorLayout>(attr)) {
        return py::cast<TensorLayout>(attr);
    } else if (py::isinstance<bool>(attr)) {
        return py::cast<bool>(attr);
    } else if (py::isinstance<py::int_>(attr)) {
        return py::cast<int>(attr);
    } else if (py::isinstance<py::str>(attr)) {
        return py::cast<std::string>(attr);
    } else if (py::isinstance<py::float_>(attr)) {
        return py::cast<double>(attr);
    } else if (py::isinstance<ExprPtr>(attr)) {
        return py::cast<ExprPtr>(attr);
    } else {
        throw TypeError("Unsupported attr type");
    }
}

std::any ConvertListAttr(const py::list& list)
{
    if (list.empty()) {
        return std::any();
    }
    auto item0 = list[0];
    if (py::isinstance<ExprPtr>(item0)) {
        std::vector<ExprPtr> ret;
        for (auto item : list) {
            ret.push_back(py::cast<ExprPtr>(item));
        }
        return ret;
    } else if (py::isinstance<std::string>(item0)) {
        std::vector<std::string> ret;
        for (auto item : list) {
            ret.push_back(py::cast<std::string>(item));
        }
        return ret;
    } else if (py::isinstance<py::int_>(item0)) {
        std::vector<int> ret;
        for (auto item : list) {
            ret.push_back(py::cast<int>(item));
        }
        return ret;
    } else {
        throw TypeError("Unsupported list attr");
    }
}

std::vector<std::pair<std::string, std::any>> ConvertAttrDict(const py::dict& attrs)
{
    std::vector<std::pair<std::string, std::any>> kwargs;
    for (auto item : attrs) {
        std::string key = py::cast<std::string>(item.first);
        std::any value;
        if (py::isinstance<py::list>(item.second)) {
            value = ConvertListAttr(py::cast<py::list>(item.second));
        } else {
            value = ConvertAttr(py::cast<py::object>(item.second));
        }
        if (value.has_value()) {
            kwargs.emplace_back(key, value);
        }
    }
    return kwargs;
}

void BindStmt(py::module& m)
{
    // clang-format off
    auto stmt = py::class_<Stmt, IRNode, std::shared_ptr<Stmt>>(m, "Stmt", "Base class for all statements");
    BindFields<Stmt>(stmt);

    auto assign_stmt = py::class_<AssignStmt, Stmt, std::shared_ptr<AssignStmt>>(m, "AssignStmt",
            "Assignment statement: var = value")
        .def(py::init<const VarPtr&, const ExprPtr&, const Span&>(),
             py::arg("var"), py::arg("value"), py::arg("span"),
             "Create an assignment statement");
    BindFields<AssignStmt>(assign_stmt);

    auto if_stmt = py::class_<IfStmt, Stmt, std::shared_ptr<IfStmt>>(m, "IfStmt",
            "Conditional statement: if condition then then_body else else_body")
        .def(py::init<const ExprPtr&, const StmtPtr&, const std::optional<StmtPtr>&,
                      const std::vector<VarPtr>&, const Span&>(),
             py::arg("condition"), py::arg("then_body"), py::arg("else_body") = py::none(),
             py::arg("return_vars"), py::arg("span"),
             "Create a conditional statement with then and else branches (else_body can be None)");
    BindFields<IfStmt>(if_stmt);

    auto yield_stmt = py::class_<YieldStmt, Stmt, std::shared_ptr<YieldStmt>>(m, "YieldStmt",
            "Yield statement: yield value")
        .def(py::init<const std::vector<ExprPtr>&, const Span&>(),
             py::arg("value"), py::arg("span"),
             "Create a yield statement with a list of expressions")
        .def(py::init<const Span&>(),
             py::arg("span"),
             "Create a yield statement without values");
    BindFields<YieldStmt>(yield_stmt);

    auto return_stmt = py::class_<ReturnStmt, Stmt, std::shared_ptr<ReturnStmt>>(m, "ReturnStmt",
            "Return statement: return value")
        .def(py::init<const std::vector<ExprPtr>&, const Span&>(),
             py::arg("value"), py::arg("span"),
             "Create a return statement with a list of expressions")
        .def(py::init<const Span&>(),
             py::arg("span"),
             "Create a return statement without values");
    BindFields<ReturnStmt>(return_stmt);

    auto for_stmt = py::class_<ForStmt, Stmt, std::shared_ptr<ForStmt>>(m, "ForStmt",
        "For loop statement: for loop_var in range(start, stop, step): body")
        .def(py::init<VarPtr, ExprPtr, ExprPtr, ExprPtr, std::vector<IterArgPtr>&, StmtPtr,
                      std::vector<VarPtr>&, const Span&>(),
             py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("iter_args"),
             py::arg("body"), py::arg("return_vars"), py::arg("span"),
             "Create a for loop statement with a list of expressions");
    BindFields<ForStmt>(for_stmt);

    auto while_stmt = py::class_<WhileStmt, Stmt, std::shared_ptr<WhileStmt>>(m, "WhileStmt",
        "While loop statement: while condition: body")
        .def(py::init<const ExprPtr&, const std::vector<IterArgPtr>&, const StmtPtr&,
                      const std::vector<VarPtr>&, const Span&>(),
            py::arg("condition"), py::arg("iter_args"), py::arg("body"), py::arg("return_vars"), py::arg("span"),
            "Create a while loop statement");
    BindFields<WhileStmt>(while_stmt);

    auto seq_stmt = py::class_<SeqStmts, Stmt, std::shared_ptr<SeqStmts>>(m, "SeqStmts",
        "Sequence of statements: a sequence of statements")
        .def(py::init<const std::vector<StmtPtr>&, const Span&>(),
             py::arg("stmts"), py::arg("span"),
             "Create a sequence of statements")
        .def("__getitem__", [](SeqStmtsPtr& self, int index) {
            int size = static_cast<int>(self->stmts_.size());
            if (index < -size || index >= size) {
                throw IndexError("SeqStmts index " + std::to_string(index) + " out of range [" +
                                 std::to_string(-size) + ", " + std::to_string(size - 1) + "]");
            }
            if (index < 0) index += size;
            return self->stmts_[index];
        }, py::arg("index"), "Get statement by index, supports negative indexing");
    BindFields<SeqStmts>(seq_stmt);

    auto eval_stmt_class = py::class_<EvalStmt, Stmt, std::shared_ptr<EvalStmt>>(m, "EvalStmt",
            "Evaluation statement: expr")
        .def(py::init<const ExprPtr&, const Span&>(),
             py::arg("expr"), py::arg("span"),
             "Create an evaluation statement");
    BindFields<EvalStmt>(eval_stmt_class);

    auto break_stmt = py::class_<BreakStmt, Stmt, std::shared_ptr<BreakStmt>>(m, "BreakStmt",
            "Break statement: break")
        .def(py::init<const Span&>(), py::arg("span"), "Create a break statement")
        .def(py::init<const std::vector<ExprPtr>&, const Span&>(),
             py::arg("operands"), py::arg("span"),
             "Create a break statement with operands");
    BindFields<BreakStmt>(break_stmt);

    auto continue_stmt = py::class_<ContinueStmt, Stmt, std::shared_ptr<ContinueStmt>>(m, "ContinueStmt",
            "Continue statement: continue")
        .def(py::init<const Span&>(), py::arg("span"), "Create a continue statement")
        .def(py::init<const std::vector<ExprPtr>&, const Span&>(),
             py::arg("operands"), py::arg("span"),
             "Create a continue statement with operands");
    BindFields<ContinueStmt>(continue_stmt);

    auto scalarop_stmt = py::class_<ScalarOpStmt, Stmt, std::shared_ptr<ScalarOpStmt>>(m, "ScalarOpStmt",
            "Scalar operation statement: result, result_token = opcode(args, tokens)")
        .def(py::init<VarPtr, VarPtr, std::string, const std::vector<ExprPtr>&, const Span&>(),
             py::arg("result"), py::arg("result_token"), py::arg("opcode"), py::arg("args"), py::arg("span"),
             "Create a scalar operation statement");
    BindFields<ScalarOpStmt>(scalarop_stmt);

    auto tensorop_stmt = py::class_<TensorOpStmt, Stmt, std::shared_ptr<TensorOpStmt>>(m, "TensorOpStmt",
            "Tensor operation statement: results, result_token = opcode(args, attrs, tokens)")
        .def(py::init([](std::vector<VarPtr> results, VarPtr result_token, std::string opcode,
             std::vector<ExprPtr> args, const std::vector<ExprPtr>& tokens, py::dict attrs, Span span){
                auto attr_list = ConvertAttrDict(attrs);
                return std::make_shared<TensorOpStmt>(results, result_token, opcode, args, tokens, attr_list, span);
            }),
            py::arg("results"), py::arg("result_token"), py::arg("opcode"), py::arg("args"),
            py::arg("tokens"), py::arg("attrs"), py::arg("span"),
            "Create a tensor operation statement");
    BindFields<TensorOpStmt>(tensorop_stmt);

    auto function = py::class_<Function, IRNode, std::shared_ptr<Function>>(m, "Function",
        "Function definition with name, parameters, return types, and body")
        .def(py::init<std::string, std::vector<VarPtr>&, std::vector<TypePtr>&, StmtPtr, Span, FunctionType>(),
             py::arg("name"), py::arg("params"), py::arg("return_types"), py::arg("body"), py::arg("span"),
             py::arg("type") = FunctionType::OPAQUE,
             "Create a function definition");
    BindFields<Function>(function);

    auto program = py::class_<Program, IRNode, std::shared_ptr<Program>>(m, "Program",
        "Program definition with functions mapped by GlobalVar references. "
        "Functions are automatically sorted by name for deterministic ordering.")
        .def(py::init<const std::vector<FunctionPtr>&, const std::string&, const Span&>(),
             py::arg("functions"), py::arg("name"), py::arg("span"),
             "Create a program from a list of functions. "
             "GlobalVar references are created automatically from function names.")
        .def("__getitem__", [](const ProgramPtr& self, const std::string& name) {
            return self->GetFunction(name);
        }, py::arg("name"), "Get function by name, returns None if not found")
        .def_readonly("functions", &Program::functions_, "Program functions")
        .def_readonly("name", &Program::name_, "Program name")
        .def_readonly("span", &Program::span_, "Source location");
    // clang-format on
}

void BindType(py::module& m)
{
    py::native_enum<FunctionType>(m, "FunctionType", "enum.IntEnum", "Function type classification")
        .value("Opaque", FunctionType::OPAQUE, "Unspecified function type (default)")
        .export_values()
        .finalize();

    py::native_enum<TensorLayout>(m, "TensorLayout", "enum.IntEnum", "Tensor layout enumeration")
        .value("ND", TensorLayout::ND, "ND layout")
        .value("DN", TensorLayout::DN, "DN layout")
        .value("NZ", TensorLayout::NZ, "NZ layout")
        .export_values()
        .finalize();

    py::native_enum<MemorySpace>(m, "MemorySpace", "enum.IntEnum", "Memory space enumeration")
        .value("DDR", MemorySpace::DDR, "DDR memory (off-chip)")
        .value("Vec", MemorySpace::Vec, "Vector/unified buffer (on-chip)")
        .value("Mat", MemorySpace::Mat, "Matrix/L1 buffer")
        .value("Left", MemorySpace::Left, "Left matrix operand buffer")
        .value("Right", MemorySpace::Right, "Right matrix operand buffer")
        .value("Acc", MemorySpace::Acc, "Accumulator buffer")
        .value("Bias", MemorySpace::Bias, "Bias buffer")
        .export_values()
        .finalize();

    m.attr("Mem") = m.attr("MemorySpace");

    py::native_enum<PipeType>(m, "PipeType", "enum.IntEnum", "Pipeline type enumeration")
        .value("MTE1", PipeType::MTE1, "Memory Transfer Engine 1")
        .value("MTE2", PipeType::MTE2, "Memory Transfer Engine 2")
        .value("MTE3", PipeType::MTE3, "Memory Transfer Engine 3")
        .value("M", PipeType::M, "Matrix Unit")
        .value("V", PipeType::V, "Vector Unit")
        .value("S", PipeType::S, "Scalar Unit")
        .value("FIX", PipeType::FIX, "Fix Pipe")
        .value("ALL", PipeType::ALL, "All Pipes")
        .export_values()
        .finalize();

    py::native_enum<CoreType>(m, "CoreType", "enum.IntEnum", "Core type enumeration")
        .value("VECTOR", CoreType::VECTOR, "Vector Core")
        .value("CUBE", CoreType::CUBE, "Cube Core")
        .export_values()
        .finalize();
}

void BindTypeClass(py::module& m)
{
    py::class_<Type, std::shared_ptr<Type>>(m, "Type", "Base class for type representations")
        .def("__str__", [](const TypePtr& self) { return PythonPrint(self, "ir"); });

    py::class_<UnknownType, Type, std::shared_ptr<UnknownType>>(m, "UnknownType", "Unknown type")
        .def_static("get", []() { return GetUnknownType(); })
        .def(py::init<>());

    py::class_<ScalarType, Type, std::shared_ptr<ScalarType>>(m, "ScalarType", "Scalar type")
        .def(py::init<DataType>(), py::arg("dtype"))
        .def_readonly("dtype", &ScalarType::dtype_);

    py::class_<TensorType, Type, std::shared_ptr<TensorType>>(m, "TensorType", "Tensor type")
        .def(
            py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>>(), py::arg("shape"),
            py::arg("dtype"), py::arg("memref") = py::none())
        .def(
            py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>>(), py::arg("shape"),
            py::arg("dtype"), py::arg("memref") = py::none())
        .def_readonly("dtype", &TensorType::dtype_)
        .def_readonly("shape", &TensorType::shape_)
        .def_readonly("memref", &TensorType::memref_);

    py::class_<TupleType, Type, std::shared_ptr<TupleType>>(m, "TupleType", "Tuple type")
        .def(py::init<std::vector<TypePtr>>(), py::arg("types"))
        .def_readonly("types", &TupleType::types_);

    py::class_<PtrType, Type, std::shared_ptr<PtrType>>(m, "PtrType", "Pointer type").def(py::init<>());

    py::class_<TokenType, Type, std::shared_ptr<TokenType>>(m, "TokenType", "Opaque token type").def(py::init<>());
}
} // namespace ir

void BindIR(py::module& m)
{
    auto m1 = m.def_submodule("ir");
    ir::BindType(m1);
    ir::BindDType(m1);
    ir::BindTypeClass(m1);
    ir::BindSpan(m1);
    ir::BindExpr(m1);
    ir::BindStmt(m1);
    ir::BindIRBuilder(m1);
}
} // namespace pypto
