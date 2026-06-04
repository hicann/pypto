/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <any>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/printer.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

std::string BlockDslDTypeToString(const DataType& dtype)
{
    if (dtype == DataType::BF16) {
        return "BFLOAT16";
    }
    return DTypeToString(dtype);
}

const char* TileLayoutToPythonName(TileLayout layout)
{
    switch (layout) {
        case TileLayout::none_box:
            return "none_box";
        case TileLayout::row_major:
            return "row_major";
        case TileLayout::col_major:
            return "col_major";
        default:
            INTERNAL_CHECK(false) << "Unknown TileLayout in PrintTileView";
            return "";
    }
}

const char* TilePadToPythonName(TilePad pad)
{
    switch (pad) {
        case TilePad::null:
            return "null";
        case TilePad::zero:
            return "zero";
        case TilePad::max:
            return "max";
        case TilePad::min:
            return "min";
        default:
            INTERNAL_CHECK(false) << "Unknown TilePad in PrintTileView";
            return "";
    }
}

const char* CompactModeToPythonName(CompactMode compact)
{
    switch (compact) {
        case CompactMode::null:
            return "null";
        case CompactMode::normal:
            return "normal";
        case CompactMode::row_plus_one:
            return "row_plus_one";
        default:
            INTERNAL_CHECK(false) << "Unknown CompactMode in PrintTileView";
            return "";
    }
}

void PrintPythonIterArgTuple(std::ostringstream& stream, const std::vector<IterArgPtr>& iter_args)
{
    stream << "(";
    bool need_comma = false;
    for (const auto& iter_arg : iter_args) {
        if (need_comma) {
            stream << ", ";
        }
        stream << iter_arg->iterVar_->name_;
        need_comma = true;
    }
    if (iter_args.size() == 1) {
        stream << ",";
    }
    stream << ")";
}

template <typename VisitExprFn>
void PrintPythonInitValuesClause(
    std::ostringstream& stream, const std::vector<IterArgPtr>& iter_args, const VisitExprFn& visit_expr)
{
    stream << "init_values=(";
    bool need_comma = false;
    for (const auto& iter_arg : iter_args) {
        if (need_comma) {
            stream << ", ";
        }
        visit_expr(iter_arg->initValue_);
        need_comma = true;
    }
    if (iter_args.size() == 1) {
        stream << ",";
    }
    stream << ")";
}

template <typename VisitExprFn>
void PrintPythonRangeHeader(
    std::ostringstream& stream, const std::string& prefix, const ForStmtPtr& op, const VisitExprFn& visit_expr)
{
    stream << "for " << op->loopVar_->name_;
    if (!op->iterArgs_.empty()) {
        stream << ", ";
        PrintPythonIterArgTuple(stream, op->iterArgs_);
    }

    stream << " in " << prefix << ".range(";
    visit_expr(op->start_);
    stream << ", ";
    visit_expr(op->stop_);
    stream << ", ";
    visit_expr(op->step_);

    if (!op->iterArgs_.empty()) {
        stream << ", ";
        PrintPythonInitValuesClause(stream, op->iterArgs_, visit_expr);
    }
    stream << "):\n";
}

template <typename VisitExprFn>
void PrintPythonWhileHeader(
    std::ostringstream& stream, const std::string& prefix, const WhileStmtPtr& op, const VisitExprFn& visit_expr)
{
    stream << "for ";
    PrintPythonIterArgTuple(stream, op->iterArgs_);
    stream << " in " << prefix << ".while_(";
    PrintPythonInitValuesClause(stream, op->iterArgs_, visit_expr);
    stream << "):\n";
}

} // namespace

/**
 * \brief Python-style IR printer
 *
 * Prints IR nodes in Python syntax with type annotations and SSA-style control flow.
 * This is the recommended printer for new code that outputs valid Python syntax.
 *
 * Key features:
 * - Type annotations (e.g., x: pl.INT64, a: pl.Tensor[[4, 8], pl.FP32])
 * - SSA-style if/for with pl.yield_() and pl.range()
 * - Op attributes as keyword arguments
 * - Program headers with # pypto_block.program: name
 */
class IRPythonPrinter : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    explicit IRPythonPrinter(std::string prefix = "pl") : prefix_(std::move(prefix)) {}
    ~IRPythonPrinter() override = default;

    /**
     * \brief Print an IR node to a string in Python IR syntax
     *
     * \param node IR node to print (can be Expr, Stmt, Function, or Program)
     * \return Python-style string representation
     */
    std::string Print(const IRNodePtr& node);
    std::string Print(const TypePtr& type);

protected:
    PYPTO_IR_PRINTER_COMMON_VISITOR_OVERRIDES();
    void VisitStmt_(const SectionStmtPtr& op) override;
    void VisitStmt_(const OpStmtsPtr& op) override;

private:
    std::ostringstream stream_;
    int indent_level_ = 0;
    std::string prefix_;                   // Prefix for type names (e.g., "pl" or "ir")
    ProgramPtr current_program_ = nullptr; // Track when printing within Program (for self.method() calls)

    // Helper methods
    std::string GetIndent() const;
    void IncreaseIndent();
    void DecreaseIndent();

    // Print a statement block at current indent level.
    // SeqStmts/OpStmts are transparent containers - recursed into without extra indent.
    void PrintStmtBlock(const StmtPtr& stmt);

    // Statement body visitor with SSA-style handling
    void VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars = {});
    void PrintYieldAssignmentVars(const std::vector<VarPtr>& return_vars);
    void PrintYieldAssignment(const YieldStmtPtr& yield_stmt, const std::vector<VarPtr>& return_vars);
    void VisitSeqStmtBody(const SeqStmtsPtr& seq_stmts, const std::vector<VarPtr>& return_vars);
    void PrintReturnFromYield(const YieldStmtPtr& yield_stmt);
    void PrintFunctionDecorator(const FunctionPtr& func);
    void PrintFunctionParam(const VarPtr& var);
    void PrintFunctionSignature(const FunctionPtr& func);
    void PrintFunctionReturnType(const FunctionPtr& func);
    void PrintFunctionBody(const FunctionPtr& func);

    bool TryPrintProgramCall(const CallPtr& op);
    std::string FormatCallOpName(const std::string& op_name) const;
    void PrintCallArgs(const CallPtr& op);
    void PrintCallArg(const CallPtr& op, size_t index);
    void PrintCallKwargs(const CallPtr& op, bool need_comma);
    void PrintKwargValue(const std::string& key, const std::any& value);

    // Binary/unary operator helpers (reuse precedence logic)
    void PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol);
    void PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name);
    void PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left);
    void PrintPrefixUnaryOp(const UnaryExprPtr& op, const char* op_symbol, Precedence operand_precedence);

    // Shape printing helper
    void PrintShapedTypePrefix(
        std::ostringstream& oss, const char* type_name, const std::vector<ExprPtr>& shape, DataType dtype);
    void PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape);
    void PrintExprList(std::ostringstream& oss, const std::vector<ExprPtr>& exprs);

    // MemRef and TileView printing helpers
    std::string PrintMemRef(const MemRef& memref);
    std::string PrintTileView(const TileView& tile_view);
    std::string PrintHardwareInfo(const HardwareInfo& hw_info);
    std::string PrintTensorView(const TensorView& tensor_view);
};

// Helper function to format float literals with decimal point
std::string FormatFloatLiteral(double value)
{
    // Check if the value is an integer (no fractional part)
    if (std::fabs(value - std::floor(value)) < 1e-15) {
        // For integer values, format as X.0
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << value;
        return oss.str();
    } else {
        // For non-integer values, use default formatting with enough precision
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}

// DataTypeToPythonString removed — now uses DTypeToString from dtype.h

// IRPythonPrinter implementation
std::string IRPythonPrinter::Print(const IRNodePtr& node)
{
    stream_.str("");
    stream_.clear();
    indent_level_ = 0;
    PrintIRNodeWithVisitor(*this, stream_, node);
    return stream_.str();
}

void IRPythonPrinter::PrintShapedTypePrefix(
    std::ostringstream& oss, const char* type_name, const std::vector<ExprPtr>& shape, DataType dtype)
{
    oss << prefix_ << "." << type_name << "[[";
    PrintShapeDims(oss, shape);
    oss << "], " << prefix_ << "." << BlockDslDTypeToString(dtype);
}

std::string IRPythonPrinter::Print(const TypePtr& type)
{
    if (auto scalar_type = As<ScalarType>(type)) {
        // Print as pl.Scalar[pl.INT64] for proper round-trip support
        return prefix_ + ".Scalar[" + prefix_ + "." + BlockDslDTypeToString(scalar_type->dtype_) + "]";
    }

    if (auto ptr_type = As<PtrType>(type)) {
        // Print as pl.Ptr[pl.FP32]
        return prefix_ + ".Ptr[" + prefix_ + "." + BlockDslDTypeToString(ptr_type->dtype_) + "]";
    }

    if (auto tensor_type = As<TensorType>(type)) {
        std::ostringstream oss;
        PrintShapedTypePrefix(oss, "Tensor", tensor_type->shape_, tensor_type->dtype_);

        // Add optional tensor_view parameter if present (before memref for positional ordering)
        if (tensor_type->tensor_view_.has_value()) {
            oss << ", tensor_view=" << PrintTensorView(tensor_type->tensor_view_.value());
        }

        // Add optional memref as positional arg
        if (tensor_type->memref_.has_value()) {
            oss << ", " << PrintMemRef(*tensor_type->memref_.value());
        }

        oss << "]";
        return oss.str();
    }

    if (auto tile_type = As<TileType>(type)) {
        std::ostringstream oss;
        PrintShapedTypePrefix(oss, "Tile", tile_type->shape_, tile_type->dtype_);

        // Add optional tile_view parameter if present (before memref for positional ordering)
        if (tile_type->tileView_.has_value()) {
            oss << ", tile_view=" << PrintTileView(tile_type->tileView_.value());
        }

        // Add optional hardware_info parameter if present
        if (tile_type->hardwareInfo_.has_value()) {
            oss << ", hardware_info=" << PrintHardwareInfo(tile_type->hardwareInfo_.value());
        }

        // Add optional memref as positional arg
        if (tile_type->memref_.has_value()) {
            oss << ", " << PrintMemRef(*tile_type->memref_.value());
        }
        oss << "]";
        return oss.str();
    }

    if (auto tuple_type = As<TupleType>(type)) {
        std::ostringstream oss;
        oss << prefix_ << ".Tuple([";
        for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << Print(tuple_type->types_[i]);
        }
        oss << "])";
        return oss.str();
    }

    if (auto memref_type = As<MemRefType>(type)) {
        return prefix_ + ".MemRefType";
    }

    return prefix_ + ".UnknownType";
}

std::string IRPythonPrinter::GetIndent() const { return std::string(static_cast<size_t>(indent_level_ * 4), ' '); }

void IRPythonPrinter::IncreaseIndent() { indent_level_++; }

void IRPythonPrinter::DecreaseIndent()
{
    if (indent_level_ > 0) {
        indent_level_--;
    }
}

// Expression visitors - reuse precedence logic from base printer
void IRPythonPrinter::VisitExpr_(const VarPtr& op) { stream_ << op->name_; }

void IRPythonPrinter::VisitExpr_(const MemRefPtr& op) { stream_ << op->name_; }

void IRPythonPrinter::VisitExpr_(const ConstIntPtr& op)
{
    // INT64 and INDEX both represent 64-bit integer constants
    // in the Python DSL, so they print as bare integers. Other integer types (INT8,
    // INT32, etc.) need explicit dtype annotation.
    if (op->dtype() == DataType::INT64 || op->dtype() == DataType::INDEX) {
        stream_ << op->value_;
    } else {
        stream_ << prefix_ << ".const(" << op->value_ << ", " << prefix_ << "." << BlockDslDTypeToString(op->dtype())
                << ")";
    }
}

void IRPythonPrinter::VisitExpr_(const ConstFloatPtr& op)
{
    if (op->dtype() != DataType::FP32) {
        stream_ << prefix_ << ".const(" << FormatFloatLiteral(op->value_) << ", " << prefix_ << "."
                << BlockDslDTypeToString(op->dtype()) << ")";
    } else {
        stream_ << FormatFloatLiteral(op->value_);
    }
}

void IRPythonPrinter::VisitExpr_(const ConstBoolPtr& op) { stream_ << (op->value_ ? "True" : "False"); }

bool IRPythonPrinter::TryPrintProgramCall(const CallPtr& op)
{
    if (current_program_) {
        // Check if the op name matches a function in the current program
        if (current_program_->GetFunction(op->name_)) {
            stream_ << "self." << op->name_ << "(";
            for (size_t i = 0; i < op->args_.size(); ++i) {
                if (i > 0)
                    stream_ << ", ";
                VisitExpr(op->args_[i]);
            }
            stream_ << ")";
            return true;
        }
    }
    return false;
}

std::string IRPythonPrinter::FormatCallOpName(const std::string& op_name) const
{
    // Operations are stored with internal names like "tensor.add_scalar"
    // but need to be printed in parseable format like "pl.tensor.add"
    if (op_name.find('.') == std::string::npos) {
        return op_name;
    }

    std::string normalized_op_name = op_name;
    size_t scalar_pos = normalized_op_name.find("_scalar");
    if (scalar_pos != std::string::npos) {
        normalized_op_name = normalized_op_name.substr(0, scalar_pos);
    }
    return prefix_ + "." + normalized_op_name;
}

void IRPythonPrinter::PrintCallArg(const CallPtr& op, size_t index)
{
    // Special handling for block.alloc's first argument (memory_space)
    if (op->name_ == "block.alloc" && index == 0) {
        if (auto const_int = std::dynamic_pointer_cast<const ConstInt>(op->args_[index])) {
            int space_value = static_cast<int>(const_int->value_);
            stream_ << prefix_ << ".MemorySpace." << MemorySpaceToString(static_cast<MemorySpace>(space_value));
            return;
        }
    }
    VisitExpr(op->args_[index]);
}

void IRPythonPrinter::PrintCallArgs(const CallPtr& op)
{
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        PrintCallArg(op, i);
    }
}

void IRPythonPrinter::PrintKwargValue(const std::string& key, const std::any& value)
{
    if (value.type() == typeid(int)) {
        int int_val = AnyCast<int>(value, "printing kwarg: " + key);
        if (key == "set_pipe" || key == "wait_pipe") {
            stream_ << prefix_ << ".PipeType." << PipeTypeToString(static_cast<PipeType>(int_val));
        } else {
            stream_ << int_val;
        }
    } else if (value.type() == typeid(bool)) {
        stream_ << (AnyCast<bool>(value, "printing kwarg: " + key) ? "True" : "False");
    } else if (value.type() == typeid(std::string)) {
        stream_ << "'" << AnyCast<std::string>(value, "printing kwarg: " + key) << "'";
    } else if (value.type() == typeid(double)) {
        stream_ << FormatFloatLiteral(AnyCast<double>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(float)) {
        stream_ << FormatFloatLiteral(static_cast<double>(AnyCast<float>(value, "printing kwarg: " + key)));
    } else if (value.type() == typeid(DataType)) {
        stream_ << prefix_ << "." << BlockDslDTypeToString(AnyCast<DataType>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(MemorySpace)) {
        stream_ << prefix_ << ".MemorySpace."
                << MemorySpaceToString(AnyCast<MemorySpace>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(std::vector<int>)) {
        const auto& vec = AnyCast<std::vector<int>>(value, "printing kwarg: " + key);
        stream_ << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0)
                stream_ << ", ";
            stream_ << vec[i];
        }
        stream_ << "]";
    } else {
        CHECK(false) << "Invalid kwarg type for key: " << key
                     << ", expected int, bool, std::string, double, float, DataType, MemorySpace, "
                        "or std::vector<int>, but got "
                     << DemangleTypeName(value.type().name());
    }
}

void IRPythonPrinter::PrintCallKwargs(const CallPtr& op, bool need_comma)
{
    for (const auto& [key, value] : op->kwargs_) {
        if (need_comma) {
            stream_ << ", ";
        }
        need_comma = true;
        stream_ << key << "=";
        PrintKwargValue(key, value);
    }
}

void IRPythonPrinter::VisitExpr_(const CallPtr& op)
{
    INTERNAL_CHECK(!op->name_.empty()) << "Call has empty name";
    if (TryPrintProgramCall(op)) {
        return;
    }

    stream_ << FormatCallOpName(op->name_) << "(";
    PrintCallArgs(op);
    PrintCallKwargs(op, !op->args_.empty());
    stream_ << ")";
}

void IRPythonPrinter::VisitExpr_(const MakeTuplePtr& op)
{
    stream_ << "[";
    for (size_t i = 0; i < op->elements_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->elements_[i]);
    }
    stream_ << "]";
}

void IRPythonPrinter::VisitExpr_(const GetItemExprPtr& op)
{
    VisitExpr(op->value_);
    stream_ << "[";
    VisitExpr(op->slice_);
    stream_ << "]";
}

// Binary and unary operators - reuse from base printer logic
void IRPythonPrinter::PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left)
{
    PrintChildExprWithParens(*this, stream_, parent, child, is_left);
}

void IRPythonPrinter::PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol)
{
    PrintChild(op, op->left_, true);
    stream_ << " " << op_symbol << " ";
    PrintChild(op, op->right_, false);
}

void IRPythonPrinter::PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name)
{
    stream_ << func_name << "(";
    VisitExpr(op->left_);
    stream_ << ", ";
    VisitExpr(op->right_);
    stream_ << ")";
}

void IRPythonPrinter::PrintPrefixUnaryOp(const UnaryExprPtr& op, const char* op_symbol, Precedence operand_precedence)
{
    stream_ << op_symbol;
    Precedence operand_prec = GetPrecedence(op->operand_);
    if (operand_prec < operand_precedence) {
        stream_ << "(";
        VisitExpr(op->operand_);
        stream_ << ")";
    } else {
        VisitExpr(op->operand_);
    }
}

// Arithmetic binary operators
void IRPythonPrinter::VisitExpr_(const AddPtr& op) { PrintBinaryOp(op, "+"); }
void IRPythonPrinter::VisitExpr_(const SubPtr& op) { PrintBinaryOp(op, "-"); }
void IRPythonPrinter::VisitExpr_(const MulPtr& op) { PrintBinaryOp(op, "*"); }
void IRPythonPrinter::VisitExpr_(const FloorDivPtr& op) { PrintBinaryOp(op, "//"); }
void IRPythonPrinter::VisitExpr_(const FloorModPtr& op) { PrintBinaryOp(op, "%"); }
void IRPythonPrinter::VisitExpr_(const FloatDivPtr& op) { PrintBinaryOp(op, "/"); }
void IRPythonPrinter::VisitExpr_(const PowPtr& op) { PrintBinaryOp(op, "**"); }

// Function-style binary operators
void IRPythonPrinter::VisitExpr_(const MinPtr& op) { PrintFunctionBinaryOp(op, "min"); }
void IRPythonPrinter::VisitExpr_(const MaxPtr& op) { PrintFunctionBinaryOp(op, "max"); }

// Comparison operators
void IRPythonPrinter::VisitExpr_(const EqPtr& op) { PrintBinaryOp(op, "=="); }
void IRPythonPrinter::VisitExpr_(const NePtr& op) { PrintBinaryOp(op, "!="); }
void IRPythonPrinter::VisitExpr_(const LtPtr& op) { PrintBinaryOp(op, "<"); }
void IRPythonPrinter::VisitExpr_(const LePtr& op) { PrintBinaryOp(op, "<="); }
void IRPythonPrinter::VisitExpr_(const GtPtr& op) { PrintBinaryOp(op, ">"); }
void IRPythonPrinter::VisitExpr_(const GePtr& op) { PrintBinaryOp(op, ">="); }

// Logical operators
void IRPythonPrinter::VisitExpr_(const AndPtr& op) { PrintBinaryOp(op, "and"); }
void IRPythonPrinter::VisitExpr_(const OrPtr& op) { PrintBinaryOp(op, "or"); }
void IRPythonPrinter::VisitExpr_(const XorPtr& op) { PrintBinaryOp(op, "xor"); }

// Bitwise operators
void IRPythonPrinter::VisitExpr_(const BitAndPtr& op) { PrintBinaryOp(op, "&"); }
void IRPythonPrinter::VisitExpr_(const BitOrPtr& op) { PrintBinaryOp(op, "|"); }
void IRPythonPrinter::VisitExpr_(const BitXorPtr& op) { PrintBinaryOp(op, "^"); }
void IRPythonPrinter::VisitExpr_(const BitShiftLeftPtr& op) { PrintBinaryOp(op, "<<"); }
void IRPythonPrinter::VisitExpr_(const BitShiftRightPtr& op) { PrintBinaryOp(op, ">>"); }

// Unary operators
void IRPythonPrinter::VisitExpr_(const NegPtr& op) { PrintPrefixUnaryOp(op, "-", Precedence::kUnary); }

void IRPythonPrinter::VisitExpr_(const AbsPtr& op)
{
    stream_ << "abs(";
    VisitExpr(op->operand_);
    stream_ << ")";
}

void IRPythonPrinter::VisitExpr_(const CastPtr& op)
{
    auto scalar_type = As<ScalarType>(op->GetType());
    INTERNAL_CHECK(scalar_type) << "Cast has non-scalar type";
    stream_ << prefix_ << ".cast(";
    VisitExpr(op->operand_);
    stream_ << ", " << prefix_ << "." << BlockDslDTypeToString(scalar_type->dtype_) << ")";
}

void IRPythonPrinter::VisitExpr_(const NotPtr& op) { PrintPrefixUnaryOp(op, "not ", Precedence::kNot); }

void IRPythonPrinter::VisitExpr_(const BitNotPtr& op) { PrintPrefixUnaryOp(op, "~", Precedence::kUnary); }

// Statement visitors with proper Python syntax
void IRPythonPrinter::VisitStmt_(const AssignStmtPtr& op)
{
    // Print with type annotation: var: type = value
    // First print variable name
    VisitExpr(op->var_);
    stream_ << ": " << Print(op->var_->GetType()) << " = ";
    VisitExpr(op->value_);
}

void IRPythonPrinter::VisitStmt_(const IfStmtPtr& op)
{
    // SSA-style if with pl.yield_()
    stream_ << "if ";
    VisitExpr(op->condition_);
    stream_ << ":\n";

    IncreaseIndent();
    VisitStmtBody(op->thenBody_, op->returnVars_);
    DecreaseIndent();

    if (op->elseBody_.has_value()) {
        stream_ << "\n" << GetIndent() << "else:\n";
        IncreaseIndent();
        VisitStmtBody(*op->elseBody_, op->returnVars_);
        DecreaseIndent();
    }
}

void IRPythonPrinter::VisitStmt_(const YieldStmtPtr& op)
{
    // Note: In function context, this will be changed to "return" by VisitFunction
    stream_ << prefix_ << ".yield_(";
    for (size_t i = 0; i < op->value_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(op->value_[i]);
    }
    stream_ << ")";
}

void IRPythonPrinter::VisitStmt_(const ReturnStmtPtr& op) { PrintReturnStmtValues(*this, stream_, op->value_); }

void IRPythonPrinter::VisitStmt_(const ForStmtPtr& op)
{
    // SSA-style for with pl.range() or pl.parallel() - no inline type annotations in unpacking
    PrintPythonRangeHeader(stream_, prefix_, op, [this](const ExprPtr& expr) { VisitExpr(expr); });

    IncreaseIndent();
    VisitStmtBody(op->body_, op->returnVars_);
    DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const WhileStmtPtr& op)
{
    // Check if this is SSA-style (with iter_args) or natural style
    if (op->iterArgs_.empty()) {
        // Natural while loop without iter_args
        stream_ << "while ";
        VisitExpr(op->condition_);
        stream_ << ":\n";

        IncreaseIndent();
        VisitStmtBody(op->body_, op->returnVars_);
        DecreaseIndent();
    } else {
        // SSA-style while with iter_args - print as explicit DSL syntax
        PrintPythonWhileHeader(stream_, prefix_, op, [this](const ExprPtr& expr) { VisitExpr(expr); });

        IncreaseIndent();

        // Print condition as pl.cond() call as first body statement
        stream_ << GetIndent() << prefix_ << ".cond(";
        VisitExpr(op->condition_);
        stream_ << ")\n";

        VisitStmtBody(op->body_, op->returnVars_);
        DecreaseIndent();
    }
}

void IRPythonPrinter::VisitStmt_(const SectionStmtPtr& op)
{
    // Map SectionKind to DSL function name
    static const std::unordered_map<SectionKind, std::string> section_kind_to_dsl = {
        {SectionKind::Vector, "section_vector"},
        {SectionKind::Cube, "section_cube"},
    };

    auto it = section_kind_to_dsl.find(op->sectionKind_);
    INTERNAL_CHECK(it != section_kind_to_dsl.end())
        << "Internal error: Unknown SectionKind in python_printer: " << SectionKindToString(op->sectionKind_);

    stream_ << "with " << prefix_ << "." << it->second << "():\n";

    IncreaseIndent();
    PrintStmtBlock(op->body_);
    DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SeqStmtsPtr& op)
{
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        PrintStmtBlock(op->stmts_[i]);
        if (i < op->stmts_.size() - 1) {
            stream_ << "\n";
        }
    }
}

void IRPythonPrinter::VisitStmt_(const OpStmtsPtr& op)
{
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        PrintStmtBlock(op->stmts_[i]);
        if (i < op->stmts_.size() - 1) {
            stream_ << "\n";
        }
    }
}

void IRPythonPrinter::PrintStmtBlock(const StmtPtr& stmt)
{
    if (auto seq = As<SeqStmts>(stmt)) {
        for (size_t i = 0; i < seq->stmts_.size(); ++i) {
            PrintStmtBlock(seq->stmts_[i]);
            if (i < seq->stmts_.size() - 1)
                stream_ << "\n";
        }
    } else if (auto ops = As<OpStmts>(stmt)) {
        for (size_t i = 0; i < ops->stmts_.size(); ++i) {
            PrintStmtBlock(ops->stmts_[i]);
            if (i < ops->stmts_.size() - 1)
                stream_ << "\n";
        }
    } else {
        stream_ << GetIndent();
        VisitStmt(stmt);
    }
}

void IRPythonPrinter::VisitStmt_(const EvalStmtPtr& op)
{
    // Print expression statement: expr
    VisitExpr(op->expr_);
}

void IRPythonPrinter::VisitStmt_(const BreakStmtPtr& /*op*/) { stream_ << "break"; }

void IRPythonPrinter::VisitStmt_(const ContinueStmtPtr& /*op*/) { stream_ << "continue"; }

void IRPythonPrinter::VisitStmt_(const StmtPtr& op) { stream_ << op->TypeName(); }

void IRPythonPrinter::PrintYieldAssignmentVars(const std::vector<VarPtr>& return_vars)
{
    // Helper to print left-hand side of yield assignment
    // For single variable: print with type annotation (var: type)
    // For multiple variables: print without type annotations (var1, var2)
    if (return_vars.size() == 1) {
        stream_ << return_vars[0]->name_ << ": " << Print(return_vars[0]->GetType());
    } else {
        for (size_t i = 0; i < return_vars.size(); ++i) {
            if (i > 0)
                stream_ << ", ";
            stream_ << return_vars[i]->name_;
        }
    }
}

void IRPythonPrinter::PrintReturnFromYield(const YieldStmtPtr& yield_stmt)
{
    stream_ << GetIndent() << "return";
    if (!yield_stmt->value_.empty()) {
        stream_ << " ";
        for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
            if (i > 0)
                stream_ << ", ";
            VisitExpr(yield_stmt->value_[i]);
        }
    }
}

void IRPythonPrinter::PrintYieldAssignment(const YieldStmtPtr& yield_stmt, const std::vector<VarPtr>& return_vars)
{
    stream_ << GetIndent();
    PrintYieldAssignmentVars(return_vars);
    stream_ << " = " << prefix_ << ".yield_(";
    for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
        if (i > 0)
            stream_ << ", ";
        VisitExpr(yield_stmt->value_[i]);
    }
    stream_ << ")";
}

void IRPythonPrinter::VisitSeqStmtBody(const SeqStmtsPtr& seq_stmts, const std::vector<VarPtr>& return_vars)
{
    for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
        auto stmt = seq_stmts->stmts_[i];
        bool is_last = (i == seq_stmts->stmts_.size() - 1);
        auto inner_yield = As<YieldStmt>(stmt);

        if (inner_yield && is_last && !inner_yield->value_.empty() && !return_vars.empty()) {
            PrintYieldAssignment(inner_yield, return_vars);
        } else if (inner_yield) {
            stream_ << GetIndent();
            VisitStmt(stmt);
        } else {
            PrintStmtBlock(stmt);
        }

        if (i < seq_stmts->stmts_.size() - 1) {
            stream_ << "\n";
        }
    }
}

void IRPythonPrinter::VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars)
{
    // Helper to visit statement body and wrap YieldStmt with assignment if needed
    if (auto yield_stmt = As<YieldStmt>(body)) {
        // If parent has return_vars, wrap yield as assignment
        if (!yield_stmt->value_.empty() && !return_vars.empty()) {
            PrintYieldAssignment(yield_stmt, return_vars);
        } else {
            stream_ << GetIndent();
            VisitStmt(yield_stmt);
        }
    } else if (auto seq_stmts = AsMut<SeqStmts>(body)) {
        VisitSeqStmtBody(seq_stmts, return_vars);
    } else {
        PrintStmtBlock(body);
    }
}

void IRPythonPrinter::PrintFunctionDecorator(const FunctionPtr& func)
{
    stream_ << GetIndent() << "@" << prefix_ << ".function";
    if (func->funcType_ != FunctionType::OPAQUE) {
        stream_ << "(type=" << prefix_ << ".FunctionType." << FunctionTypeToString(func->funcType_) << ")";
    }
    stream_ << "\n";
}

void IRPythonPrinter::PrintFunctionParam(const VarPtr& var)
{
    stream_ << var->name_ << ": ";
    stream_ << Print(var->GetType());
}

void IRPythonPrinter::PrintFunctionSignature(const FunctionPtr& func)
{
    stream_ << GetIndent() << "def " << func->name_ << "(";
    if (current_program_) {
        stream_ << "self";
    }

    for (size_t i = 0; i < func->params_.size(); ++i) {
        if (i > 0 || current_program_)
            stream_ << ", ";
        PrintFunctionParam(func->params_[i]);
    }
    stream_ << ")";
}

void IRPythonPrinter::PrintFunctionReturnType(const FunctionPtr& func)
{
    PrintFunctionReturnAnnotation(stream_, func->returnTypes_, [this](const TypePtr& type) { return Print(type); });
}

void IRPythonPrinter::PrintFunctionBody(const FunctionPtr& func)
{
    if (!func->body_) {
        return;
    }

    const auto& seq_stmts = func->body_;
    for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
        if (auto yield_stmt = As<YieldStmt>(seq_stmts->stmts_[i])) {
            PrintReturnFromYield(yield_stmt);
        } else {
            PrintStmtBlock(seq_stmts->stmts_[i]);
        }
        if (i + 1 < seq_stmts->stmts_.size()) {
            stream_ << "\n";
        }
    }
}

void IRPythonPrinter::VisitFunction(const FunctionPtr& func)
{
    PrintFunctionDecorator(func);
    PrintFunctionSignature(func);
    PrintFunctionReturnType(func);
    stream_ << ":\n";

    IncreaseIndent();
    PrintFunctionBody(func);
    DecreaseIndent();
}

// Helper class to collect function call Op names from a function's body
class FuncCallCollector : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    std::set<std::string> collected_func_names;

    void VisitExpr_(const CallPtr& op) override
    {
        INTERNAL_CHECK(!op->name_.empty()) << "Call has empty name";
        collected_func_names.insert(op->name_);
        // Visit arguments
        IRVisitor::VisitExpr_(op);
    }
};

// Topologically sort functions so called functions come before callers
// This ensures that when reparsing, function return types are known when needed
static std::vector<std::pair<std::string, FunctionPtr>> TopologicalSortFunctions(
    const std::map<std::string, FunctionPtr>& functions)
{
    // Build dependency graph: function name -> set of function names it calls
    std::map<std::string, std::set<std::string>> dependencies;

    for (const auto& [name, func] : functions) {
        // Collect all function names referenced in the function body
        FuncCallCollector collector;
        if (func->body_) {
            collector.VisitStmt(func->body_);
        }
        // Only keep names that are actually functions in this program
        for (const auto& called_name : collector.collected_func_names) {
            if (functions.count(called_name) > 0) {
                dependencies[name].insert(called_name);
            }
        }
    }

    // Topological sort using DFS
    std::vector<std::pair<std::string, FunctionPtr>> sorted;
    std::set<std::string> visited;
    std::set<std::string> in_progress; // For cycle detection

    std::function<bool(const std::string&)> dfs = [&](const std::string& name) -> bool {
        if (visited.count(name))
            return true;
        if (in_progress.count(name))
            return false; // Cycle detected

        in_progress.insert(name);

        // Visit dependencies first (dependencies = functions this function calls)
        if (dependencies.count(name)) {
            for (const auto& dep : dependencies[name]) {
                if (!dfs(dep))
                    return false; // Cycle detected
            }
        }

        in_progress.erase(name);
        visited.insert(name);
        // Add to sorted AFTER visiting dependencies, so dependencies come first
        sorted.emplace_back(name, functions.at(name));
        return true;
    };

    // Visit all functions
    for (const auto& function_entry : functions) {
        const auto& name = function_entry.first;
        if (!dfs(name)) {
            // Cycle detected, fall back to original order
            sorted.clear();
            for (const auto& fallback_entry : functions) {
                sorted.emplace_back(fallback_entry);
            }
            return sorted;
        }
    }

    return sorted;
}

void IRPythonPrinter::VisitProgram(const ProgramPtr& program)
{
    // Print program header comment
    stream_ << "# pypto_block.program: " << (program->name_.empty() ? "Program" : program->name_) << "\n";

    // Print import statement based on prefix
    if (prefix_ == "pl") {
        stream_ << "import pypto_block.language as pl\n\n";
    } else {
        stream_ << "import pypto_block.language as " << prefix_ << "\n\n";
    }

    // Print as @pl.program class with @pl.function methods
    stream_ << "@" << prefix_ << ".program\n";
    stream_ << "class " << (program->name_.empty() ? "Program" : program->name_) << ":\n";

    IncreaseIndent();

    // Sort functions in dependency order (called functions before callers)
    auto sorted_functions = TopologicalSortFunctions(program->functions_);

    // Print each function as a method, delegating to VisitFunction
    // Setting current_program_ enables self parameter and self.method() call printing
    auto prev_program = current_program_;
    current_program_ = program;

    bool first = true;
    for (const auto& entry : sorted_functions) {
        const auto& func = entry.second;
        if (!first) {
            stream_ << "\n"; // Blank line between functions
        }
        first = false;

        VisitFunction(func);
    }

    current_program_ = prev_program;
    DecreaseIndent();
}

void IRPythonPrinter::PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape)
{
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            oss << ", ";
        // For ConstInt shape dims, print raw value to avoid dtype annotations
        if (auto const_int = As<ConstInt>(shape[i])) {
            oss << const_int->value_;
        } else {
            IRPythonPrinter temp_printer(prefix_);
            oss << temp_printer.Print(shape[i]);
        }
    }
}

void IRPythonPrinter::PrintExprList(std::ostringstream& oss, const std::vector<ExprPtr>& exprs)
{
    for (size_t i = 0; i < exprs.size(); ++i) {
        if (i > 0)
            oss << ", ";
        IRPythonPrinter temp_printer(prefix_);
        oss << temp_printer.Print(exprs[i]);
    }
}

// Helper methods for MemRef and TileView printing
std::string IRPythonPrinter::PrintMemRef(const MemRef& memref)
{
    std::ostringstream oss;
    oss << prefix_ << ".MemRef(" << prefix_ << ".MemorySpace." << MemorySpaceToString(memref.memorySpace_) << ", ";

    // Print address expression
    IRPythonPrinter temp_printer(prefix_);
    oss << temp_printer.Print(memref.addr_);

    // Print size and id
    oss << ", " << memref.size_ << ")";
    return oss.str();
}

std::string IRPythonPrinter::PrintTileView(const TileView& tile_view)
{
    std::ostringstream oss;
    oss << prefix_ << ".TileView(valid_shape=[";
    PrintExprList(oss, tile_view.validShape);
    oss << "], stride=[";
    PrintExprList(oss, tile_view.stride);
    oss << "], start_offset=";

    {
        IRPythonPrinter temp_printer(prefix_);
        oss << temp_printer.Print(tile_view.startOffset);
    }

    oss << ")";
    return oss.str();
}

std::string IRPythonPrinter::PrintHardwareInfo(const HardwareInfo& hw_info)
{
    std::ostringstream oss;
    oss << prefix_ << ".HardwareInfo(";
    oss << "blayout=" << prefix_ << ".TileLayout." << TileLayoutToPythonName(hw_info.blayout);
    oss << ", slayout=" << prefix_ << ".TileLayout." << TileLayoutToPythonName(hw_info.slayout);
    oss << ", fractal=" << hw_info.fractal;
    oss << ", pad=" << prefix_ << ".TilePad." << TilePadToPythonName(hw_info.pad);
    oss << ", compact=" << prefix_ << ".CompactMode." << CompactModeToPythonName(hw_info.compact);
    oss << ")";
    return oss.str();
}

std::string IRPythonPrinter::PrintTensorView(const TensorView& tensor_view)
{
    std::ostringstream oss;
    oss << prefix_ << ".TensorView(";

    // Print valid_shape if non-empty
    if (!tensor_view.validShape.empty()) {
        oss << "valid_shape=[";
        for (size_t i = 0; i < tensor_view.validShape.size(); ++i) {
            if (i > 0)
                oss << ", ";
            IRPythonPrinter temp_printer(prefix_);
            oss << temp_printer.Print(tensor_view.validShape[i]);
        }
        oss << "], ";
    }

    oss << "stride=[";

    // Print stride
    for (size_t i = 0; i < tensor_view.stride.size(); ++i) {
        if (i > 0)
            oss << ", ";
        IRPythonPrinter temp_printer(prefix_);
        oss << temp_printer.Print(tensor_view.stride[i]);
    }

    oss << "], layout=" << prefix_ << ".TensorLayout.";

    // Print layout enum value
    switch (tensor_view.layout) {
        case TensorLayout::ND:
            oss << "ND";
            break;
        case TensorLayout::DN:
            oss << "DN";
            break;
        case TensorLayout::NZ:
            oss << "NZ";
            break;
        default:
            INTERNAL_CHECK(false) << "Unknown TensorLayout in PrintTensorView";
            break;
    }

    oss << ")";
    return oss.str();
}

// ================================
// Public API
// ================================
// Keep the block DSL printer as a distinct entry point from the generic
// outer IRPrinter. Header declarations and external callsites will be
// migrated in a follow-up include/binding step.
std::string PythonDslPrint(const IRNodePtr& node, const std::string& prefix)
{
    IRPythonPrinter printer(prefix);
    return printer.Print(node);
}

std::string PythonDslPrint(const TypePtr& type, const std::string& prefix)
{
    IRPythonPrinter printer(prefix);
    return printer.Print(type);
}

} // namespace ir
} // namespace pypto
