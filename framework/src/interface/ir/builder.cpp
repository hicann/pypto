/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/builder.h"

#include <any>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static Span combine_span(const Span& begin_span, const Span& end_span)
{
    return Span(
        begin_span.Filename(), begin_span.BeginLine(), begin_span.BeginColumn(), end_span.EndLine(),
        end_span.EndColumn());
}

// ========== IRBuilder Implementation ==========

IRBuilder::IRBuilder() = default;

// ========== Function Building ==========

void IRBuilder::BeginFunction(const std::string& name, const Span& span, FunctionType type)
{
    if (InFunction()) {
        throw RuntimeError(
            "Cannot begin function '" + name + "': already inside function '" +
            static_cast<FunctionContext*>(CurrentContext())->GetName() + "' at " +
            CurrentContext()->GetBeginSpan().ToString());
    }

    context_stack_.push_back(std::make_unique<FunctionContext>(name, span, type));
}

VarPtr IRBuilder::FuncArg(const std::string& name, const TypePtr& type, const Span& span)
{
    ValidateInFunction("FuncArg");

    auto var = std::make_shared<ir::Var>(name, type, span);
    static_cast<FunctionContext*>(CurrentContext())->AddParam(var);
    return var;
}

void IRBuilder::ReturnType(const TypePtr& type)
{
    ValidateInFunction("ReturnType");
    static_cast<FunctionContext*>(CurrentContext())->AddReturnType(type);
}

FunctionPtr IRBuilder::EndFunction(const Span& end_span)
{
    ValidateInFunction("EndFunction");

    auto* func_ctx = static_cast<FunctionContext*>(CurrentContext());

    // Build body from accumulated statements
    const auto& stmts = func_ctx->GetStmts();
    StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

    // Combine begin and end spans
    auto new_span = combine_span(func_ctx->GetBeginSpan(), end_span);

    // Create function
    auto func = std::make_shared<Function>(
        func_ctx->GetName(), func_ctx->GetParams(), func_ctx->GetReturnTypes(), body, new_span,
        func_ctx->GetFuncType());

    // Pop context
    context_stack_.pop_back();

    return func;
}

// ========== For Loop Building ==========

void IRBuilder::BeginForLoop(
    const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step, const Span& span)
{
    if (context_stack_.empty()) {
        throw RuntimeError(
            "Cannot begin for loop: not inside a function or another valid context at " + span.ToString());
    }

    context_stack_.push_back(std::make_unique<ForLoopContext>(loop_var, start, stop, step, span));
}

void IRBuilder::AddIterArg(const IterArgPtr& iter_arg)
{
    ValidateInLoop("AddIterArg");
    static_cast<ForLoopContext*>(CurrentContext())->AddIterArg(iter_arg);
}

void IRBuilder::AddReturnVar(const VarPtr& var)
{
    ValidateInLoop("AddReturnVar");
    static_cast<ForLoopContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndForLoop(const Span& end_span)
{
    ValidateInLoop("EndForLoop");

    auto* loop_ctx = static_cast<ForLoopContext*>(CurrentContext());

    // Validate iter_args and return_vars match
    if (loop_ctx->GetIterArgs().size() != loop_ctx->GetReturnVars().size()) {
        // Pop context before throwing to maintain stack consistency
        context_stack_.pop_back();

        std::ostringstream oss;
        oss << "For loop has " << loop_ctx->GetIterArgs().size() << " iteration arguments but "
            << loop_ctx->GetReturnVars().size() << " return variables. They must match.";
        throw RuntimeError(oss.str());
    }

    // Build body from accumulated statements
    const auto& stmts = loop_ctx->GetStmts();
    StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

    // Combine begin and end spans
    auto new_span = combine_span(loop_ctx->GetBeginSpan(), end_span);

    // Create for statement
    auto for_stmt = std::make_shared<ForStmt>(
        loop_ctx->GetLoopVar(), loop_ctx->GetStart(), loop_ctx->GetStop(), loop_ctx->GetStep(), loop_ctx->GetIterArgs(),
        body, loop_ctx->GetReturnVars(), new_span);

    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(for_stmt);
    }

    return for_stmt;
}

// ========== While Loop Building ==========

void IRBuilder::BeginWhileLoop(const ExprPtr& condition, const Span& span)
{
    if (context_stack_.empty()) {
        throw RuntimeError(
            "Cannot begin while loop: not inside a function or another valid context at " + span.ToString());
    }

    context_stack_.push_back(std::make_unique<WhileLoopContext>(condition, span));
}

void IRBuilder::AddWhileIterArg(const IterArgPtr& iter_arg)
{
    ValidateInWhileLoop("AddWhileIterArg");
    static_cast<WhileLoopContext*>(CurrentContext())->AddIterArg(iter_arg);
}

void IRBuilder::AddWhileReturnVar(const VarPtr& var)
{
    ValidateInWhileLoop("AddWhileReturnVar");
    static_cast<WhileLoopContext*>(CurrentContext())->AddReturnVar(var);
}

void IRBuilder::SetWhileLoopCondition(const ExprPtr& condition)
{
    ValidateInWhileLoop("SetWhileLoopCondition");
    static_cast<WhileLoopContext*>(CurrentContext())->SetCondition(condition);
}

StmtPtr IRBuilder::EndWhileLoop(const Span& end_span)
{
    ValidateInWhileLoop("EndWhileLoop");

    auto* loop_ctx = static_cast<WhileLoopContext*>(CurrentContext());

    // Validate iter_args and return_vars match
    if (loop_ctx->GetIterArgs().size() != loop_ctx->GetReturnVars().size()) {
        // Pop context before throwing to maintain stack consistency
        context_stack_.pop_back();

        std::ostringstream oss;
        oss << "While loop has " << loop_ctx->GetIterArgs().size() << " iteration arguments but "
            << loop_ctx->GetReturnVars().size() << " return variables. They must match.";
        throw RuntimeError(oss.str());
    }

    // Build body from accumulated statements
    const auto& stmts = loop_ctx->GetStmts();
    StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

    // Combine begin and end spans
    auto new_span = combine_span(loop_ctx->GetBeginSpan(), end_span);

    // Create while statement
    auto while_stmt = std::make_shared<WhileStmt>(
        loop_ctx->GetCondition(), loop_ctx->GetIterArgs(), body, loop_ctx->GetReturnVars(), new_span);

    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(while_stmt);
    }

    return while_stmt;
}

// ========== If Statement Building ==========

void IRBuilder::BeginIf(const ExprPtr& condition, const Span& span)
{
    IRCHECK(!context_stack_.empty()) << "Cannot begin if statement: not inside a function or another valid context at "
                                     << span.ToString();
    context_stack_.push_back(std::make_unique<IfStmtContext>(condition, span));
}

void IRBuilder::BeginElse(const Span& span)
{
    ValidateInIf("BeginElse");

    auto* if_ctx = static_cast<IfStmtContext*>(CurrentContext());
    IRCHECK(!if_ctx->InElseBranch()) << "Cannot begin else branch: already in else branch at " << span.ToString();

    if_ctx->BeginElseBranch();
}

void IRBuilder::AddIfReturnVar(const VarPtr& var)
{
    ValidateInIf("AddIfReturnVar");
    static_cast<IfStmtContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndIf(const Span& end_span)
{
    ValidateInIf("EndIf");

    auto* if_ctx = static_cast<IfStmtContext*>(CurrentContext());

    // Build then body
    const auto& then_stmts = if_ctx->GetStmts();
    StmtPtr then_body = (then_stmts.size() == 1) ? then_stmts[0] : std::make_shared<SeqStmts>(then_stmts, end_span);

    // Build else body (optional)
    std::optional<StmtPtr> else_body;
    if (if_ctx->InElseBranch()) {
        const auto& else_stmts = if_ctx->GetElseStmts();
        if (!else_stmts.empty()) {
            else_body = (else_stmts.size() == 1) ? else_stmts[0] : std::make_shared<SeqStmts>(else_stmts, end_span);
        }
    }

    // Combine begin and end spans
    auto new_span = combine_span(if_ctx->GetBeginSpan(), end_span);

    // Create if statement
    auto if_stmt =
        std::make_shared<IfStmt>(if_ctx->GetCondition(), then_body, else_body, if_ctx->GetReturnVars(), new_span);

    // Pop context
    context_stack_.pop_back();

    // Emit to parent context if it exists
    if (!context_stack_.empty()) {
        CurrentContext()->AddStmt(if_stmt);
    }

    return if_stmt;
}

// ========== Program Building ==========
void IRBuilder::BeginProgram(const std::string& name, const Span& span)
{
    if (InProgram()) {
        throw RuntimeError(
            "Cannot begin program '" + name + "': already inside program '" +
            static_cast<ProgramContext*>(CurrentContext())->GetName() + "' at " +
            CurrentContext()->GetBeginSpan().ToString());
    }

    context_stack_.push_back(std::make_unique<ProgramContext>(name, span));
}

void IRBuilder::AddFunction(const FunctionPtr& func)
{
    ValidateInProgram("AddFunction");
    static_cast<ProgramContext*>(CurrentContext())->AddFunction(func);
}

ProgramPtr IRBuilder::EndProgram(const Span& end_span)
{
    ValidateInProgram("EndProgram");

    auto* prog_ctx = static_cast<ProgramContext*>(CurrentContext());

    // Combine begin and end spans
    auto new_span = combine_span(prog_ctx->GetBeginSpan(), end_span);

    // Create program from functions vector
    auto program = std::make_shared<Program>(prog_ctx->GetFunctions(), prog_ctx->GetName(), new_span);

    // Pop context
    context_stack_.pop_back();

    return program;
}

bool IRBuilder::InProgram() const
{
    for (const auto& ctx : context_stack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            return true;
        }
    }
    return false;
}

std::vector<TypePtr> IRBuilder::GetFunctionReturnTypes(const GlobalVarPtr& gvar) const
{
    // Find the program context in the stack
    for (const auto& ctx : context_stack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            auto* prog_ctx = static_cast<const ProgramContext*>(ctx.get());
            return prog_ctx->GetReturnTypes(gvar);
        }
    }
    return {};
}

// ========== Statement Recording ==========

void IRBuilder::Emit(const StmtPtr& stmt)
{
    if (context_stack_.empty()) {
        throw RuntimeError("Cannot emit statement: not inside any context");
    }

    auto* ctx = CurrentContext();
    ctx->AddStmt(stmt);
}

AssignStmtPtr IRBuilder::Assign(const VarPtr& var, const ExprPtr& value, const Span& span)
{
    auto assign = std::make_shared<AssignStmt>(var, value, span);
    Emit(assign);
    return assign;
}

VarPtr IRBuilder::Var(const std::string& name, const TypePtr& type, const Span& span)
{
    return std::make_shared<ir::Var>(name, type, span);
}

ReturnStmtPtr IRBuilder::Return(const std::vector<ExprPtr>& values, const Span& span)
{
    auto return_stmt = std::make_shared<ReturnStmt>(values, span);
    Emit(return_stmt);
    return return_stmt;
}

ReturnStmtPtr IRBuilder::Return(const Span& span)
{
    auto return_stmt = std::make_shared<ReturnStmt>(span);
    Emit(return_stmt);
    return return_stmt;
}

// ========== Context State Queries ==========

BuildContext* IRBuilder::CurrentContext()
{
    if (context_stack_.empty()) {
        return nullptr;
    }
    return context_stack_.back().get();
}

bool IRBuilder::InFunction() const
{
    for (const auto& ctx : context_stack_) {
        if (ctx->GetType() == BuildContext::Type::FUNCTION) {
            return true;
        }
    }
    return false;
}

bool IRBuilder::InLoop() const
{
    if (context_stack_.empty()) {
        return false;
    }
    return context_stack_.back()->GetType() == BuildContext::Type::FOR_LOOP;
}

bool IRBuilder::InIf() const
{
    if (context_stack_.empty()) {
        return false;
    }
    return context_stack_.back()->GetType() == BuildContext::Type::IF_STMT;
}

bool IRBuilder::InWhileLoop() const
{
    if (context_stack_.empty()) {
        return false;
    }
    return context_stack_.back()->GetType() == BuildContext::Type::WHILE_LOOP;
}

// ========== Private Helpers ==========

template <typename T>
T* IRBuilder::GetCurrentContextAs()
{
    auto* ctx = CurrentContext();
    if (!ctx) {
        return nullptr;
    }
    return dynamic_cast<T*>(ctx);
}

void IRBuilder::ValidateInFunction(const std::string& operation)
{
    IRCHECK(InFunction()) << operation << " can only be called inside a function context";
    IRCHECK(CurrentContext()->GetType() == BuildContext::Type::FUNCTION)
        << operation << " must be called directly in function context, not nested";
}

void IRBuilder::ValidateInLoop(const std::string& operation)
{
    IRCHECK(InLoop()) << operation << " can only be called inside a for loop context";
}

void IRBuilder::ValidateInIf(const std::string& operation)
{
    IRCHECK(InIf()) << operation << " can only be called inside an if statement context";
}

void IRBuilder::ValidateInWhileLoop(const std::string& operation)
{
    IRCHECK(InWhileLoop()) << operation << " can only be called inside a while loop context";
}

void IRBuilder::ValidateInProgram(const std::string& operation)
{
    IRCHECK(InProgram()) << operation << " can only be called inside a program context";
}

void ProgramContext::AddFunction(const FunctionPtr& func)
{
    INTERNAL_CHECK(func) << "Cannot add null function to program";

    for (auto cfun : functions_) {
        if (cfun->name_ == func->name_) {
            throw RuntimeError("Function '" + func->name_ + "' already declared in program");
        }
    }

    // Store return types for this function
    return_types_[func->name_] = func->returnTypes_;

    functions_.push_back(func);
}

std::vector<TypePtr> ProgramContext::GetReturnTypes(const GlobalVarPtr& gvar) const
{
    auto it = return_types_.find(gvar->name_);
    if (it != return_types_.end()) {
        return it->second;
    }
    return {};
}

} // namespace ir
} // namespace pypto
