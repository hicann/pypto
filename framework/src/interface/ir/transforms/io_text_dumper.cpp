/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file dumper_text.cpp
 * \brief IR text dumper implementation following the declared IR Syntax grammar.
 */

#include "ir/transforms/io_text.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <typeindex>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Operator name lookup
// ---------------------------------------------------------------------------
const char* IRDumperText::BinaryOpName(const BinaryExprPtr& op)
{
    return GetBinaryOpDict().Find(op->GetKind(), "UnknownBinOp").c_str();
}

const char* IRDumperText::UnaryOpName(const UnaryExprPtr& op)
{
    return GetUnaryOpDict().Find(op->GetKind(), "UnknownUnOp").c_str();
}

// ---------------------------------------------------------------------------
// Attr value printing
// ---------------------------------------------------------------------------
void IRDumperText::PrintAttrValue(const std::string& key, const std::any& value)
{
    if (value.type() == typeid(int)) {
        stream_ << AnyCast<int>(value, key);
    } else if (value.type() == typeid(uint64_t)) {
        stream_ << AnyCast<uint64_t>(value);
    } else if (value.type() == typeid(double)) {
        stream_ << AnyCast<double>(value, key);
    } else if (value.type() == typeid(float)) {
        stream_ << static_cast<double>(AnyCast<float>(value, key));
    } else if (value.type() == typeid(bool)) {
        stream_ << (AnyCast<bool>(value, key) ? IR_KW_TRUE : IR_KW_FALSE);
    } else if (value.type() == typeid(DataType)) {
        stream_ << AnyCast<DataType>(value, key).ToCTypeString();
    } else if (value.type() == typeid(MemorySpace)) {
        stream_ << MemorySpaceToString(AnyCast<MemorySpace>(value, key));
    } else if (value.type() == typeid(std::vector<int>)) {
        const auto& vals = AnyCast<std::vector<int>>(value, key);
        stream_ << IR_PUN_LBRACKET;
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i > 0) {
                stream_ << IR_PUN_COMMA << " ";
            }
            stream_ << vals[i];
        }
        stream_ << IR_PUN_RBRACKET;
    } else if (value.type() == typeid(std::vector<int64_t>)) {
        const auto& vals = AnyCast<std::vector<int64_t>>(value, key);
        stream_ << IR_PUN_LBRACKET;
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i > 0) {
                stream_ << IR_PUN_COMMA << " ";
            }
            stream_ << vals[i];
        }
        stream_ << IR_PUN_RBRACKET;
    } else {
        stream_ << "Unsupported";
    }
}

void IRDumperText::PrintAttr(const std::string& key, const std::any& value)
{
    stream_ << " " << IR_PUN_ATTRNAME << key << IR_PUN_LPAREN;
    PrintAttrValue(key, value);
    stream_ << IR_PUN_RPAREN;
}

// ---------------------------------------------------------------------------
// Type printing via VisitType_
// ---------------------------------------------------------------------------
void IRDumperText::PrintType(const TypePtr& type)
{
    if (!type) {
        stream_ << IR_KW_UNKNOWN;
        return;
    }
    VisitType(type);
}

void IRDumperText::VisitType_(const UnknownTypePtr&) { stream_ << IR_KW_UNKNOWN; }

void IRDumperText::VisitType_(const ScalarTypePtr& op) { stream_ << op->dtype_.ToCTypeString(); }

void IRDumperText::VisitType_(const TensorTypePtr& op)
{
    stream_ << IR_KW_TENSOR << IR_PUN_LT;
    PrintShape(op->shape_);
    stream_ << IR_PUN_COMMA << " " << op->dtype_.ToCTypeString();
    stream_ << IR_PUN_COMMA << " " << IR_KW_TENSOR_VIEW << IR_PUN_LT;
    if (op->tensor_view_.has_value()) {
        const auto& tv = op->tensor_view_.value();
        PrintShape(tv.validShape);
        stream_ << IR_PUN_COMMA << " ";
        PrintShape(tv.stride);
        stream_ << IR_PUN_COMMA << " " << GetTensorLayoutDict().Find(tv.layout);
        if (tv.ptr.has_value()) {
            stream_ << IR_PUN_COMMA << " ";
            VisitExpr(tv.ptr.value());
        }
    }
    stream_ << IR_PUN_GT;
    stream_ << IR_PUN_GT;
}

void IRDumperText::VisitType_(const TileTypePtr& op)
{
    stream_ << IR_KW_TILE << IR_PUN_LT;
    PrintShape(op->shape_);
    stream_ << IR_PUN_COMMA << " " << op->dtype_.ToCTypeString();
    stream_ << IR_PUN_COMMA << " " << IR_KW_TILE_VIEW << IR_PUN_LT;
    if (op->tileView_.has_value()) {
        const auto& tv = op->tileView_.value();
        PrintShape(tv.validShape);
        stream_ << IR_PUN_COMMA << " ";
        PrintShape(tv.stride);
        stream_ << IR_PUN_COMMA << " ";
        if (tv.startOffset) {
            VisitExpr(tv.startOffset);
        } else {
            stream_ << IR_KW_NULL;
        }
    }
    stream_ << IR_PUN_GT;
    stream_ << IR_PUN_COMMA << " " << IR_KW_HW_INFO << IR_PUN_LT;
    if (op->hardwareInfo_.has_value()) {
        const auto& hw = op->hardwareInfo_.value();
        stream_ << GetTileLayoutDict().Find(hw.blayout);
        stream_ << IR_PUN_COMMA << " " << GetTileLayoutDict().Find(hw.slayout);
        stream_ << IR_PUN_COMMA << " " << hw.fractal;
        stream_ << IR_PUN_COMMA << " " << GetTilePadDict().Find(hw.pad);
        stream_ << IR_PUN_COMMA << " " << GetCompactModeDict().Find(hw.compact);
    }
    stream_ << IR_PUN_GT;
    stream_ << IR_PUN_GT;
}

void IRDumperText::VisitType_(const TupleTypePtr& op)
{
    stream_ << IR_KW_TUPLE << IR_PUN_LT;
    for (size_t i = 0; i < op->types_.size(); ++i) {
        if (i > 0) {
            stream_ << IR_PUN_COMMA << " ";
        }
        VisitType(op->types_[i]);
    }
    stream_ << IR_PUN_GT;
}

void IRDumperText::VisitType_(const MemRefTypePtr&) { stream_ << IR_KW_MEMREF_TYPE; }

void IRDumperText::VisitType_(const PtrTypePtr& op)
{
    stream_ << IR_KW_PTR << IR_PUN_LT << op->dtype_.ToCTypeString() << IR_PUN_GT;
}

void IRDumperText::VisitType_(const TokenTypePtr&) { stream_ << IR_KW_TOKEN; }

void IRDumperText::VisitType_(const NoneTypePtr&) { stream_ << IR_KW_NONE; }

void IRDumperText::VisitType_(const LogicalTensorTypePtr&) { stream_ << IR_KW_LOGICAL_TENSOR; }

void IRDumperText::PrintShape(const std::vector<ExprPtr>& shape)
{
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            stream_ << " " << IR_KW_DIM << " ";
        }
        VisitExpr(shape[i]);
    }
}

void IRDumperText::PrintExprList(const std::vector<ExprPtr>& exprs)
{
    bool first = true;
    for (size_t i = 0; i < exprs.size(); ++i) {
        if (!first) {
            stream_ << IR_PUN_COMMA << " ";
        }
        VisitExpr(exprs[i]);
        first = false;
    }
}

void IRDumperText::PrintTokenList(const std::vector<VarPtr>& tokenList)
{
    bool first = true;
    for (const auto& t : tokenList) {
        if (!first) {
            stream_ << IR_PUN_COMMA << " ";
        }
        PrintVarRef(t);
        first = false;
    }
}

// ---------------------------------------------------------------------------
// Var printing
// ---------------------------------------------------------------------------
void IRDumperText::PrintVarDef(const VarPtr& var)
{
    PrintType(var->GetType());
    stream_ << " " << IR_PUN_VARNAME << var->name_;
}

void IRDumperText::PrintVarRef(const VarPtr& var) { stream_ << IR_PUN_VARNAME << var->name_; }

void IRDumperText::PrintVarDefList(const std::vector<VarPtr>& varList, const std::vector<VarPtr>& tokenList)
{
    bool first = true;
    for (const auto& v : varList) {
        if (!first) {
            stream_ << IR_PUN_COMMA << " ";
        }
        PrintVarDef(v);
        first = false;
    }
    for (const auto& t : tokenList) {
        if (t) {
            if (!first) {
                stream_ << IR_PUN_COMMA << " ";
            }
            PrintVarDef(t);
            first = false;
        }
    }
    if (!first) {
        stream_ << " " << IR_PUN_EQ << " ";
    }
}

void IRDumperText::PrintIterArgs(const std::vector<IterArgPtr>& iterArgs)
{
    stream_ << " " << IR_KW_ITER << " " << IR_PUN_LBRACE;
    for (const auto& ia : iterArgs) {
        PrintVarDefList({ia->iterVar_});
        VisitExpr(ia->initValue_);
        stream_ << IR_PUN_SEMI;
    }
    stream_ << IR_PUN_RBRACE;
}

// ---------------------------------------------------------------------------
// Expression visitors
// ---------------------------------------------------------------------------
void IRDumperText::VisitExpr_(const VarPtr& op) { PrintVarRef(op); }

void IRDumperText::VisitExpr_(const MemRefPtr& op)
{
    stream_ << IR_KW_MEMREF << IR_PUN_LT << MemorySpaceToString(op->memorySpace_) << ", ";
    VisitExpr(op->addr_);
    stream_ << IR_PUN_COMMA << " " << op->size_ << IR_PUN_GT;
}

void IRDumperText::VisitExpr_(const ConstIntPtr& op) { stream_ << op->value_; }

void IRDumperText::VisitExpr_(const ConstFloatPtr& op) { stream_ << std::setprecision(17) << op->value_; }

void IRDumperText::VisitExpr_(const ConstBoolPtr& op) { stream_ << (op->value_ ? IR_KW_TRUE : IR_KW_FALSE); }

void IRDumperText::VisitExpr_(const CallPtr& op)
{
    stream_ << op->name_ << IR_PUN_LPAREN;
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0) {
            stream_ << IR_PUN_COMMA << " ";
        }
        VisitExpr(op->args_[i]);
    }
    for (const auto& [key, value] : op->kwargs_) {
        if (!op->args_.empty() || key != op->kwargs_.front().first) {
            stream_ << IR_PUN_COMMA << " ";
        }
        stream_ << key << IR_PUN_EQ;
        PrintAttrValue(key, value);
    }
    stream_ << IR_PUN_RPAREN;
}

void IRDumperText::VisitExpr_(const MakeTuplePtr& op)
{
    stream_ << IR_KW_TUPLE << IR_PUN_LPAREN;
    PrintExprList(op->elements_);
    stream_ << IR_PUN_RPAREN;
}

void IRDumperText::VisitExpr_(const GetItemExprPtr& op)
{
    stream_ << IR_KW_GETITEM << IR_PUN_LPAREN;
    VisitExpr(op->value_);
    stream_ << IR_PUN_COMMA << " ";
    VisitExpr(op->slice_);
    stream_ << IR_PUN_RPAREN;
}

void IRDumperText::VisitExpr_(const ScalarExprPtr& /*op*/) { stream_ << IR_KW_SCALAR_EXPR; }

void IRDumperText::VisitBinaryExpr_(const BinaryExprPtr& op)
{
    stream_ << IR_PUN_LPAREN;
    VisitExpr(op->left_);
    stream_ << " " << BinaryOpName(op) << " ";
    VisitExpr(op->right_);
    stream_ << IR_PUN_RPAREN;
}

void IRDumperText::VisitUnaryExpr_(const UnaryExprPtr& op)
{
    stream_ << IR_PUN_LPAREN << UnaryOpName(op);
    if (op->GetKind() == ObjectKind::Cast) {
        auto scalarType = As<ScalarType>(op->GetType());
        if (scalarType) {
            stream_ << " " << scalarType->dtype_.ToCTypeString();
        }
    }
    stream_ << " ";
    VisitExpr(op->operand_);
    stream_ << IR_PUN_RPAREN;
}

// ---------------------------------------------------------------------------
// Statement visitors
// ---------------------------------------------------------------------------
void IRDumperText::PrintBody(const StmtPtr& body)
{
    stream_ << " ";
    if (std::dynamic_pointer_cast<const SeqStmts>(body)) {
        VisitStmt(body);
    } else {
        indent_++;
        VisitStmt(body);
        indent_--;
    }
}

void IRDumperText::VisitStmt_(const AssignStmtPtr& op)
{
    PrintVarDefList({op->var_});
    VisitExpr(op->value_);
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const SeqStmtsPtr& op)
{
    stream_ << IR_PUN_LBRACE;
    indent_++;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        stream_ << "\n" << GetIndent();
        VisitStmt(op->stmts_[i]);
    }
    indent_--;
    if (!op->stmts_.empty()) {
        stream_ << "\n" << GetIndent();
    }
    stream_ << IR_PUN_RBRACE;
}

void IRDumperText::VisitStmt_(const IfStmtPtr& op)
{
    PrintVarDefList(op->returnVars_);
    stream_ << IR_KW_IF << " ";
    VisitExpr(op->condition_);
    stream_ << " " << IR_KW_THEN;
    PrintBody(op->thenBody_);
    stream_ << "\n" << GetIndent() << IR_KW_ELSE;
    if (op->elseBody_.has_value()) {
        PrintBody(op->elseBody_.value());
    } else {
        stream_ << " " << IR_PUN_LBRACE << IR_PUN_RBRACE;
    }
}

void IRDumperText::VisitStmt_(const YieldStmtPtr& op)
{
    stream_ << IR_KW_YIELD;
    if (!op->value_.empty()) {
        stream_ << " ";
        PrintExprList(op->value_);
    }
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const ReturnStmtPtr& op)
{
    stream_ << IR_KW_RETURN;
    if (!op->value_.empty()) {
        stream_ << " ";
        PrintExprList(op->value_);
    }
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const ForStmtPtr& op)
{
    PrintVarDefList(op->returnVars_);
    stream_ << IR_KW_FOR << " ";
    PrintVarRef(op->loopVar_);
    stream_ << " " << IR_KW_INRANGE << " ";
    VisitExpr(op->start_);
    stream_ << IR_PUN_COMMA << " ";
    VisitExpr(op->stop_);
    stream_ << IR_PUN_COMMA << " ";
    VisitExpr(op->step_);
    PrintIterArgs(op->iterArgs_);
    if (!op->attrs_.empty()) {
        for (const auto& [key, value] : op->attrs_) {
            PrintAttr(key, value);
        }
    }
    PrintBody(op->body_);
}

void IRDumperText::VisitStmt_(const WhileStmtPtr& op)
{
    PrintVarDefList(op->returnVars_);
    stream_ << IR_KW_WHILE << " ";
    VisitExpr(op->condition_);
    PrintIterArgs(op->iterArgs_);
    PrintBody(op->body_);
}

void IRDumperText::VisitStmt_(const SectionStmtPtr& op)
{
    stream_ << IR_KW_SECTION << " " << SectionKindToString(op->sectionKind_);
    PrintBody(op->body_);
}

void IRDumperText::VisitStmt_(const EvalStmtPtr& op)
{
    VisitExpr(op->expr_);
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const BreakStmtPtr& op)
{
    stream_ << IR_KW_BREAK;
    if (!op->value_.empty()) {
        stream_ << " ";
        PrintExprList(op->value_);
    }
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const ContinueStmtPtr& op)
{
    stream_ << IR_KW_CONTINUE;
    if (!op->value_.empty()) {
        stream_ << " ";
        PrintExprList(op->value_);
    }
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const ScalarOpStmtPtr& op)
{
    PrintVarDefList({op->result_}, std::vector<VarPtr>{op->result_token_});
    stream_ << op->opcode_ << IR_PUN_LPAREN;
    PrintExprList(op->args_);
    stream_ << IR_PUN_RPAREN << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const TensorOpStmtPtr& op)
{
    PrintVarDefList(op->result_, std::vector<VarPtr>{op->result_token_});
    stream_ << op->opcode_ << IR_PUN_LPAREN;
    PrintExprList(op->args_);
    stream_ << IR_PUN_RPAREN;

    stream_ << " " << IR_KW_TOKEN << IR_PUN_LPAREN;
    PrintTokenList(op->tokens_);
    stream_ << IR_PUN_RPAREN;

    for (const auto& [key, value] : op->attrs_) {
        PrintAttr(key, value);
    }
    stream_ << IR_PUN_SEMI;
}

void IRDumperText::VisitStmt_(const StmtPtr& op)
{
    stream_ << IR_PUN_LT << "unknown stmt: " << op->TypeName() << IR_PUN_GT;
}

// ---------------------------------------------------------------------------
// Function / Program
// ---------------------------------------------------------------------------
void IRDumperText::VisitFunction(const FunctionPtr& func)
{
    stream_ << GetIndent() << IR_KW_FUNCTION << " " << func->name_;
    stream_ << " " << IR_KW_INCAST << IR_PUN_LPAREN;
    for (size_t i = 0; i < func->params_.size(); ++i) {
        if (i > 0) {
            stream_ << IR_PUN_COMMA << " ";
        }
        PrintVarDef(func->params_[i]);
    }
    stream_ << IR_PUN_RPAREN;
    stream_ << " " << IR_KW_OUTCAST << IR_PUN_LPAREN;
    for (size_t i = 0; i < func->returnTypes_.size(); ++i) {
        if (i > 0) {
            stream_ << IR_PUN_COMMA << " ";
        }
        PrintType(func->returnTypes_[i]);
    }
    stream_ << IR_PUN_RPAREN;
    PrintAttr(IR_KW_TYPE, FunctionTypeToString(func->funcType_));
    PrintAttr(IR_KW_ENTRY, func->entry_ ? IR_KW_TRUE : IR_KW_FALSE);
    PrintBody(func->body_);
}

void IRDumperText::VisitProgram(const ProgramPtr& program)
{
    std::string entryName;
    for (const auto& [name, func] : program->functions_) {
        if (func->entry_) {
            entryName = name;
            break;
        }
    }
    stream_ << IR_KW_PROGRAM;
    PrintAttr(IR_KW_ENTRY, entryName);
    stream_ << " " << IR_PUN_LBRACE << "\n";
    indent_++;
    bool first = true;
    for (const auto& [name, func] : program->functions_) {
        (void)name;
        if (!first) {
            stream_ << "\n";
        }
        first = false;
        VisitFunction(func);
    }
    indent_--;
    stream_ << "\n" << IR_PUN_RBRACE;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
std::string IRDumperText::Dump(const IRNodePtr& node)
{
    IRDumperText dumper;
    if (auto program = As<Program>(node)) {
        dumper.VisitProgram(program);
    } else if (auto func = As<Function>(node)) {
        dumper.VisitFunction(func);
    } else if (auto stmt = As<Stmt>(node)) {
        dumper.VisitStmt(stmt);
    } else if (auto expr = As<Expr>(node)) {
        dumper.VisitExpr(expr);
    } else {
        return "<unsupported>";
    }
    return dumper.stream_.str();
}

std::string IRDumperText::DumpType(const TypePtr& type)
{
    IRDumperText dumper;
    dumper.PrintType(type);
    return dumper.stream_.str();
}

std::string TextDump(const IRNodePtr& node) { return IRDumperText::Dump(node); }
std::string TextDumpType(const TypePtr& type) { return IRDumperText::DumpType(type); }

} // namespace ir
} // namespace pypto
