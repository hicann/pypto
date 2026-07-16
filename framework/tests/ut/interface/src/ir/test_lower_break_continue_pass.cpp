/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_lower_break_continue_pass.cpp
 * \brief Coverage tests for LowerBreakContinue pass (lower_break_continue_pass.cpp)
 */

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"
#include "test_ir.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

class LowerBreakContinuePassTest : public testing::Test {
protected:
    VarPtr MakeVar(const std::string& name, TypePtr type = Scalar(DataType::INT32))
    {
        return std::make_shared<Var>(name, type, Sp());
    }

    ExprPtr Int(int64_t value) { return std::make_shared<ConstInt>(value, DataType::INT32, Sp()); }

    ExprPtr Bool(bool value) { return std::make_shared<ConstBool>(value, Sp()); }

    FunctionPtr MakeFunc(const std::string& name, const StmtPtr& body)
    {
        auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
        return std::make_shared<Function>(name, std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body, Sp());
    }

    ProgramPtr MakeProg(const FunctionPtr& func)
    {
        return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", Sp());
    }

    ProgramPtr RunPass(const ProgramPtr& prog)
    {
        auto p = pass::LowerBreakContinue();
        return p(prog);
    }

    FunctionPtr RunOnFunc(const FunctionPtr& func)
    {
        auto result = RunPass(MakeProg(func));
        return result->GetFunction(func->name_);
    }

    ExprPtr LowerContinueGuardCondition(const ExprPtr& cond)
    {
        auto i = MakeVar("i");
        auto if_cont = std::make_shared<IfStmt>(cond, std::make_shared<ContinueStmt>(Sp()), std::nullopt,
                                                std::vector<VarPtr>{}, Sp());
        auto assign = std::make_shared<AssignStmt>(MakeVar("y"), i, Sp());
        auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{if_cont, assign}, Sp());
        auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(4), Int(1), std::vector<IterArgPtr>{}, body,
                                                  std::vector<VarPtr>{}, Sp());
        auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
        if (!result_func) {
            return nullptr;
        }
        auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
        EXPECT_NE(result_for, nullptr);
        if (!result_for || result_for->body_->stmts_.empty()) {
            return nullptr;
        }
        auto guard_if = std::dynamic_pointer_cast<const IfStmt>(result_for->body_->stmts_[0]);
        EXPECT_NE(guard_if, nullptr);
        return guard_if ? guard_if->condition_ : nullptr;
    }

    bool ContainsNodeOfKind(const StmtPtr& stmt, const std::string& kind)
    {
        if (!stmt)
            return false;
        if (stmt->TypeName() == kind)
            return true;
        if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
            for (const auto& s : seq->stmts_) {
                if (ContainsNodeOfKind(s, kind))
                    return true;
            }
        }
        if (auto ifS = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            if (ContainsNodeOfKind(ifS->thenBody_, kind))
                return true;
            if (ifS->elseBody_ && ContainsNodeOfKind(*ifS->elseBody_, kind))
                return true;
        }
        if (auto forS = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            return ContainsNodeOfKind(forS->body_, kind);
        }
        if (auto whileS = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            return ContainsNodeOfKind(whileS->body_, kind);
        }
        return false;
    }
};

// ============================================================================
// No break/continue: pass returns original
// ============================================================================

TEST_F(LowerBreakContinuePassTest, TestNoBreakContinueNoChange)
{
    auto x = MakeVar("x");
    auto i = MakeVar("i");
    auto zero = Int(0);
    auto ten = Int(10);
    auto one = Int(1);
    auto assign = std::make_shared<AssignStmt>(x, one, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto forBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign, yield}, Sp());
    auto forStmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, forBody,
                                             std::vector<VarPtr>{}, Sp());
    auto func = MakeFunc("f", forStmt);

    auto result_func = RunOnFunc(func);
    ASSERT_NE(result_func, nullptr);
    EXPECT_EQ(result_func.get(), func.get());
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "ContinueStmt"));
}

// ============================================================================
// For loop with break
// ============================================================================

TEST_F(LowerBreakContinuePassTest, TestForLoopWithBreak)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto cond = std::make_shared<Eq>(i, std::make_shared<ConstInt>(5, DataType::INT32, Sp()), DataType::BOOL, Sp());
    auto brk = std::make_shared<BreakStmt>(Sp());
    auto ifBreak = std::make_shared<IfStmt>(cond, brk, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto assign = std::make_shared<AssignStmt>(x, i, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto forBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{ifBreak, assign, yield}, Sp());
    auto forStmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, forBody,
                                             std::vector<VarPtr>{}, Sp());
    auto func = MakeFunc("f", forStmt);
    auto prog = MakeProg(func);
    auto result = RunPass(prog);
    auto result_func = result->GetFunction("f");
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
    auto forResult = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(forResult, nullptr);
    ASSERT_EQ(forResult->iterArgs_.size(), 1u);
    ASSERT_EQ(forResult->returnVars_.size(), 1u);
    EXPECT_EQ(forResult->iterArgs_[0]->iterVar_->name_, "_can_continue");
    EXPECT_EQ(forResult->returnVars_[0]->name_, "_can_continue_final");
}

// ============================================================================
// While loop with break
// ============================================================================

TEST_F(LowerBreakContinuePassTest, TestWhileLoopWithBreak)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto innerCond = std::make_shared<Eq>(x, std::make_shared<ConstInt>(5, DataType::INT32, Sp()), DataType::BOOL,
                                          Sp());
    auto brk = std::make_shared<BreakStmt>(Sp());
    auto ifBreak = std::make_shared<IfStmt>(innerCond, brk, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto whileBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{ifBreak, assign, yield}, Sp());
    auto whileStmt = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, whileBody, std::vector<VarPtr>{},
                                                 Sp());
    auto func = MakeFunc("f", whileStmt);
    auto prog = MakeProg(func);
    auto result = RunPass(prog);
    auto result_func = result->GetFunction("f");
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
    auto whileResult = std::dynamic_pointer_cast<const WhileStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(whileResult, nullptr);
    ASSERT_EQ(whileResult->iterArgs_.size(), 1u);
    ASSERT_EQ(whileResult->returnVars_.size(), 1u);
    auto condition = std::dynamic_pointer_cast<const Var>(whileResult->condition_);
    ASSERT_NE(condition, nullptr);
    EXPECT_EQ(condition->name_, whileResult->iterArgs_[0]->iterVar_->name_);
}

// ============================================================================
// Bare break in for loop
// ============================================================================

TEST_F(LowerBreakContinuePassTest, TestForLoopWithBareBreak)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto brk = std::make_shared<BreakStmt>(Sp());
    auto forBody = brk;
    auto forStmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, forBody,
                                             std::vector<VarPtr>{}, Sp());
    auto func = MakeFunc("f", forStmt);
    auto prog = MakeProg(func);
    auto result = RunPass(prog);
    auto result_func = result->GetFunction("f");
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
}

// ============================================================================
// Nested loops: break in inner loop only
// ============================================================================

TEST_F(LowerBreakContinuePassTest, TestNestedForBreakInnerOnly)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto j = std::make_shared<Var>("j", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    auto cond = std::make_shared<Eq>(j, std::make_shared<ConstInt>(3, DataType::INT32, Sp()), DataType::BOOL, Sp());
    auto brk = std::make_shared<BreakStmt>(Sp());
    auto ifBreak = std::make_shared<IfStmt>(cond, brk, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto innerYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto innerBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{ifBreak, innerYield}, Sp());
    auto innerFor = std::make_shared<ForStmt>(j, zero, ten, one, std::vector<IterArgPtr>{}, innerBody,
                                              std::vector<VarPtr>{}, Sp());
    auto outerYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto outerBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{innerFor, outerYield}, Sp());
    auto outerFor = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, outerBody,
                                              std::vector<VarPtr>{}, Sp());
    auto func = MakeFunc("f", outerFor);
    auto prog = MakeProg(func);
    auto result = RunPass(prog);
    auto result_func = result->GetFunction("f");
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestNullBodyFunctionIsReturned)
{
    auto func = std::make_shared<Function>(Sp());
    func->name_ = "f";
    auto result_func = RunOnFunc(func);
    ASSERT_NE(result_func, nullptr);
    EXPECT_EQ(result_func.get(), func.get());
    EXPECT_EQ(result_func->body_, nullptr);
}

TEST_F(LowerBreakContinuePassTest, TestContinueGuardConditionNegationVariants)
{
    auto lhs = MakeVar("lhs");
    auto rhs = Int(1);
    EXPECT_NE(As<Ge>(LowerContinueGuardCondition(std::make_shared<Lt>(lhs, rhs, DataType::BOOL, Sp()))), nullptr);
    EXPECT_NE(As<Gt>(LowerContinueGuardCondition(std::make_shared<Le>(lhs, rhs, DataType::BOOL, Sp()))), nullptr);
    EXPECT_NE(As<Le>(LowerContinueGuardCondition(std::make_shared<Gt>(lhs, rhs, DataType::BOOL, Sp()))), nullptr);
    EXPECT_NE(As<Lt>(LowerContinueGuardCondition(std::make_shared<Ge>(lhs, rhs, DataType::BOOL, Sp()))), nullptr);
    EXPECT_NE(As<Ne>(LowerContinueGuardCondition(std::make_shared<Eq>(lhs, rhs, DataType::BOOL, Sp()))), nullptr);
    EXPECT_NE(As<Eq>(LowerContinueGuardCondition(std::make_shared<Ne>(lhs, rhs, DataType::BOOL, Sp()))), nullptr);

    auto flag = MakeVar("flag", Scalar(DataType::BOOL));
    auto double_negated = LowerContinueGuardCondition(std::make_shared<Not>(flag, DataType::BOOL, Sp()));
    ASSERT_NE(double_negated, nullptr);
    EXPECT_EQ(double_negated, flag);
    EXPECT_NE(As<Not>(LowerContinueGuardCondition(flag)), nullptr);
}

TEST_F(LowerBreakContinuePassTest, TestBareContinueWithIterArgsBecomesYield)
{
    auto i = MakeVar("i");
    auto iter_arg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), Int(0), Sp());
    auto ret_var = MakeVar("sum_out");
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{iter_arg},
                                              std::make_shared<ContinueStmt>(Sp()), std::vector<VarPtr>{ret_var}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->body_->stmts_.size(), 1u);
    auto yield = std::dynamic_pointer_cast<const YieldStmt>(result_for->body_->stmts_[0]);
    ASSERT_NE(yield, nullptr);
    ASSERT_EQ(yield->value_.size(), 1u);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "ContinueStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestBareContinueWithoutIterArgsDropsLoopBody)
{
    auto i = MakeVar("i");
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{},
                                              std::make_shared<ContinueStmt>(Sp()), std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    EXPECT_TRUE(result_for->body_->stmts_.empty());
}

TEST_F(LowerBreakContinuePassTest, TestThenContinueWithPreStmtsAddsIterArgElseYield)
{
    auto i = MakeVar("i");
    auto iter_arg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), Int(0), Sp());
    auto ret_var = MakeVar("sum_out");
    auto pre_assign = std::make_shared<AssignStmt>(MakeVar("tmp"), Int(2), Sp());
    auto then_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{pre_assign, std::make_shared<ContinueStmt>(Sp())},
                                                Sp());
    auto if_cont = std::make_shared<IfStmt>(Bool(true), then_body, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{iter_arg}, if_cont,
                                              std::vector<VarPtr>{ret_var}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->body_->stmts_.size(), 1u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_for->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_TRUE(result_if->elseBody_.has_value());
    ASSERT_EQ((*result_if->elseBody_)->stmts_.size(), 1u);
    auto else_yield = std::dynamic_pointer_cast<const YieldStmt>((*result_if->elseBody_)->stmts_[0]);
    ASSERT_NE(else_yield, nullptr);
    ASSERT_EQ(else_yield->value_.size(), 1u);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "ContinueStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestThenContinueWithPreStmtsMovesRemainingToElse)
{
    auto i = MakeVar("i");
    auto pre_assign = std::make_shared<AssignStmt>(MakeVar("tmp"), Int(2), Sp());
    auto then_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{pre_assign, std::make_shared<ContinueStmt>(Sp())},
                                                Sp());
    auto if_cont = std::make_shared<IfStmt>(Bool(true), then_body, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto trailing_assign = std::make_shared<AssignStmt>(MakeVar("after"), Int(3), Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{if_cont, trailing_assign}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{}, body,
                                              std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->body_->stmts_.size(), 1u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_for->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_EQ(result_if->thenBody_->stmts_.size(), 1u);
    ASSERT_TRUE(result_if->elseBody_.has_value());
    ASSERT_EQ((*result_if->elseBody_)->stmts_.size(), 1u);
    EXPECT_NE(std::dynamic_pointer_cast<const AssignStmt>((*result_if->elseBody_)->stmts_[0]), nullptr);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "ContinueStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestElseContinueAbsorbsRemainingStatementsAndAddsYield)
{
    auto i = MakeVar("i");
    auto iter_arg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), Int(0), Sp());
    auto ret_var = MakeVar("sum_out");
    auto then_assign = std::make_shared<AssignStmt>(MakeVar("tmp"), i, Sp());
    auto if_cont = std::make_shared<IfStmt>(Bool(true), then_assign,
                                            std::make_optional<StmtPtr>(std::make_shared<ContinueStmt>(Sp())),
                                            std::vector<VarPtr>{}, Sp());
    auto trailing_assign = std::make_shared<AssignStmt>(MakeVar("after"), Int(3), Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{if_cont, trailing_assign}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{iter_arg}, body,
                                              std::vector<VarPtr>{ret_var}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->body_->stmts_.size(), 1u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_for->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    ASSERT_TRUE(result_if->elseBody_.has_value());
    auto else_yield = std::dynamic_pointer_cast<const YieldStmt>((*result_if->elseBody_)->stmts_[0]);
    ASSERT_NE(else_yield, nullptr);
    ASSERT_EQ(else_yield->value_.size(), 1u);
    ASSERT_GE(result_if->thenBody_->stmts_.size(), 2u);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "ContinueStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestForBreakWithIterArgsPreservesOriginalReturnsAfterControlFlag)
{
    auto i = MakeVar("i");
    auto iter_arg = std::make_shared<IterArg>("sum", Scalar(DataType::INT32), Int(0), Sp());
    auto ret_var = MakeVar("sum_out");
    auto if_break = std::make_shared<IfStmt>(std::make_shared<Eq>(i, Int(4), DataType::BOOL, Sp()),
                                             std::make_shared<BreakStmt>(Sp()), std::nullopt, std::vector<VarPtr>{},
                                             Sp());
    auto new_sum = std::make_shared<Add>(iter_arg->iterVar_, i, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{new_sum}, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{if_break, yield}, Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{iter_arg}, body,
                                              std::vector<VarPtr>{ret_var}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->iterArgs_.size(), 2u);
    ASSERT_EQ(result_for->returnVars_.size(), 2u);
    EXPECT_EQ(result_for->iterArgs_[1]->iterVar_->name_, "sum");
    EXPECT_EQ(result_for->returnVars_[1]->name_, "sum_out");
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestWhileBreakWithIterArgsBuildsOuterConditionGuard)
{
    auto iter_arg = std::make_shared<IterArg>("count", Scalar(DataType::INT32), Int(0), Sp());
    auto ret_var = MakeVar("count_out");
    auto cond = std::make_shared<Lt>(iter_arg->iterVar_, Int(10), DataType::BOOL, Sp());
    auto if_break = std::make_shared<IfStmt>(std::make_shared<Eq>(iter_arg->iterVar_, Int(4), DataType::BOOL, Sp()),
                                             std::make_shared<BreakStmt>(Sp()), std::nullopt, std::vector<VarPtr>{},
                                             Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{iter_arg->iterVar_}, Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{if_break, yield}, Sp());
    auto while_stmt = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{iter_arg}, body,
                                                  std::vector<VarPtr>{ret_var}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", while_stmt));
    ASSERT_NE(result_func, nullptr);
    auto result_while = std::dynamic_pointer_cast<const WhileStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_while, nullptr);
    ASSERT_EQ(result_while->iterArgs_.size(), 2u);
    ASSERT_EQ(result_while->returnVars_.size(), 2u);
    auto new_condition = std::dynamic_pointer_cast<const Var>(result_while->condition_);
    ASSERT_NE(new_condition, nullptr);
    EXPECT_EQ(new_condition->name_, result_while->iterArgs_[0]->iterVar_->name_);
    ASSERT_EQ(result_while->body_->stmts_.size(), 2u);
    auto outer_if = std::dynamic_pointer_cast<const IfStmt>(result_while->body_->stmts_[0]);
    ASSERT_NE(outer_if, nullptr);
    ASSERT_TRUE(outer_if->elseBody_.has_value());
    auto else_yield = std::dynamic_pointer_cast<const YieldStmt>((*outer_if->elseBody_)->stmts_[0]);
    ASSERT_NE(else_yield, nullptr);
    ASSERT_EQ(else_yield->value_.size(), 2u);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
}

TEST_F(LowerBreakContinuePassTest, TestMultipleBreakSitesAreLoweredRecursively)
{
    auto i = MakeVar("i");
    auto first_break = std::make_shared<IfStmt>(std::make_shared<Eq>(i, Int(2), DataType::BOOL, Sp()),
                                                std::make_shared<BreakStmt>(Sp()), std::nullopt, std::vector<VarPtr>{},
                                                Sp());
    auto second_break = std::make_shared<IfStmt>(std::make_shared<Eq>(i, Int(6), DataType::BOOL, Sp()),
                                                 std::make_shared<BreakStmt>(Sp()), std::nullopt, std::vector<VarPtr>{},
                                                 Sp());
    auto body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{first_break, std::make_shared<EvalStmt>(Int(1), Sp()), second_break,
                             std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp())},
        Sp());
    auto for_stmt = std::make_shared<ForStmt>(i, Int(0), Int(10), Int(1), std::vector<IterArgPtr>{}, body,
                                              std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", for_stmt));
    ASSERT_NE(result_func, nullptr);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "BreakStmt"));
    auto result_for = std::dynamic_pointer_cast<const ForStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_for, nullptr);
    ASSERT_EQ(result_for->body_->stmts_.size(), 2u);
}

TEST_F(LowerBreakContinuePassTest, TestOuterWhileRebuiltWhenOnlyNestedLoopChanges)
{
    auto i = MakeVar("i");
    auto inner_if = std::make_shared<IfStmt>(Bool(true), std::make_shared<ContinueStmt>(Sp()), std::nullopt,
                                             std::vector<VarPtr>{}, Sp());
    auto inner_body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{inner_if, std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp())}, Sp());
    auto inner_for = std::make_shared<ForStmt>(i, Int(0), Int(3), Int(1), std::vector<IterArgPtr>{}, inner_body,
                                               std::vector<VarPtr>{}, Sp());
    auto outer_body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{inner_for, std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp())}, Sp());
    auto outer_while = std::make_shared<WhileStmt>(Bool(true), std::vector<IterArgPtr>{}, outer_body,
                                                   std::vector<VarPtr>{}, Sp());

    auto result_func = RunOnFunc(MakeFunc("f", outer_while));
    ASSERT_NE(result_func, nullptr);
    EXPECT_FALSE(ContainsNodeOfKind(result_func->body_, "ContinueStmt"));
    auto result_while = std::dynamic_pointer_cast<const WhileStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_while, nullptr);
    EXPECT_TRUE(result_while->iterArgs_.empty());
}

} // namespace ir
} // namespace pypto
