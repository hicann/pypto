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
 * \file test_const_fold_simplify_pass.cpp
 * \brief Coverage tests for ConstFoldAndSimplify pass (const_fold_simplify_pass.cpp)
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
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

class ConstFoldSimplifyPassTest : public testing::Test {
protected:
    VarPtr MakeVar(const std::string& name) { return std::make_shared<Var>(name, Scalar(DataType::INT32), Sp()); }

    ExprPtr Int(int64_t value) { return std::make_shared<ConstInt>(value, DataType::INT32, Sp()); }

    FunctionPtr MakeFunc(const std::string& name, const StmtPtr& body)
    {
        auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
        return std::make_shared<Function>(name, std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                          body, Sp());
    }

    ProgramPtr MakeProg(const FunctionPtr& func)
    {
        return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", Sp());
    }

    ProgramPtr RunPass(const ProgramPtr& prog)
    {
        auto p = pass::ConstFoldAndSimplify();
        return p(prog);
    }

    FunctionPtr RunOnFunc(const FunctionPtr& func)
    {
        auto result = RunPass(MakeProg(func));
        return result->GetFunction(func->name_);
    }

    ExprPtr RunAssignValue(const ExprPtr& value)
    {
        auto assign = std::make_shared<AssignStmt>(MakeVar("res"), value, Sp());
        auto result_func = RunOnFunc(MakeFunc("f", assign));
        if (!result_func || result_func->body_->stmts_.empty()) {
            ADD_FAILURE() << "ConstFoldAndSimplify did not return an assignment";
            return nullptr;
        }
        auto result_assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
        if (!result_assign) {
            ADD_FAILURE() << "ConstFoldAndSimplify result is not an AssignStmt";
            return nullptr;
        }
        return result_assign->value_;
    }
};

// ============================================================================
// Constant folding: scalar arithmetic and comparisons
// ============================================================================

TEST_F(ConstFoldSimplifyPassTest, TestFoldConstantExpressions)
{
    struct Case {
        std::string name;
        ExprPtr expr;
        int64_t expected;
    };
    std::vector<Case> cases{
        {"add", std::make_shared<Add>(Int(3), Int(7), DataType::INT32, Sp()), 10},
        {"sub", std::make_shared<Sub>(Int(10), Int(3), DataType::INT32, Sp()), 7},
        {"mul", std::make_shared<Mul>(Int(4), Int(5), DataType::INT32, Sp()), 20},
        {"floordiv", std::make_shared<FloorDiv>(Int(17), Int(5), DataType::INT32, Sp()), 3},
        {"floormod", std::make_shared<FloorMod>(Int(17), Int(5), DataType::INT32, Sp()), 2},
        {"eq_true", std::make_shared<Eq>(Int(5), Int(5), DataType::BOOL, Sp()), 1},
        {"eq_false", std::make_shared<Eq>(Int(5), Int(3), DataType::BOOL, Sp()), 0},
        {"lt", std::make_shared<Lt>(Int(3), Int(5), DataType::BOOL, Sp()), 1},
        {"le", std::make_shared<Le>(Int(5), Int(5), DataType::BOOL, Sp()), 1},
        {"gt", std::make_shared<Gt>(Int(5), Int(3), DataType::BOOL, Sp()), 1},
        {"ge", std::make_shared<Ge>(Int(3), Int(5), DataType::BOOL, Sp()), 0},
        {"neg", std::make_shared<Neg>(Int(42), DataType::INT32, Sp()), -42},
    };

    for (const auto& c : cases) {
        auto folded = std::dynamic_pointer_cast<const ConstInt>(RunAssignValue(c.expr));
        ASSERT_NE(folded, nullptr) << c.name;
        EXPECT_EQ(folded->value_, c.expected) << c.name;
    }
}

TEST_F(ConstFoldSimplifyPassTest, TestSimplifyIdentityAndAnnihilatorExpressions)
{
    auto x = MakeVar("x");
    struct Case {
        std::string name;
        ExprPtr expr;
    };
    std::vector<Case> identity_cases{
        {"add_zero_right", std::make_shared<Add>(x, Int(0), DataType::INT32, Sp())},
        {"add_zero_left", std::make_shared<Add>(Int(0), x, DataType::INT32, Sp())},
        {"sub_zero", std::make_shared<Sub>(x, Int(0), DataType::INT32, Sp())},
        {"mul_one_right", std::make_shared<Mul>(x, Int(1), DataType::INT32, Sp())},
        {"mul_one_left", std::make_shared<Mul>(Int(1), x, DataType::INT32, Sp())},
    };

    for (const auto& c : identity_cases) {
        auto simplified = std::dynamic_pointer_cast<const Var>(RunAssignValue(c.expr));
        ASSERT_NE(simplified, nullptr) << c.name;
        EXPECT_EQ(simplified->name_, "x") << c.name;
    }

    for (const auto& expr : {
             std::make_shared<Mul>(x, Int(0), DataType::INT32, Sp()),
             std::make_shared<Mul>(Int(0), x, DataType::INT32, Sp()),
         }) {
        auto folded_zero = std::dynamic_pointer_cast<const ConstInt>(RunAssignValue(expr));
        ASSERT_NE(folded_zero, nullptr);
        EXPECT_EQ(folded_zero->value_, 0);
    }
}

// ============================================================================
// Constant folding: boundary cases
// ============================================================================

TEST_F(ConstFoldSimplifyPassTest, TestFloorDivByZeroNoFold)
{
    auto div = std::make_shared<FloorDiv>(Int(17), Int(0), DataType::INT32, Sp());
    EXPECT_NE(std::dynamic_pointer_cast<const FloorDiv>(RunAssignValue(div)), nullptr);
}

// ============================================================================
// If-stmt simplification: constant true/false conditions
// ============================================================================

TEST_F(ConstFoldSimplifyPassTest, TestSimplifyIfTrue)
{
    auto cond = std::make_shared<ConstInt>(1, DataType::BOOL, Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto then_body = std::make_shared<AssignStmt>(x, val1, Sp());
    auto else_body = std::make_shared<AssignStmt>(x, val2, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body), std::vector<VarPtr>{},
                                           Sp());
    auto func = MakeFunc("f", ifStmt);
    auto result = RunPass(MakeProg(func));
    auto result_func = result->GetFunction("f");
    auto result_stmt = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_stmt, nullptr);
    auto ci = std::dynamic_pointer_cast<const ConstInt>(result_stmt->value_);
    ASSERT_NE(ci, nullptr);
    EXPECT_EQ(ci->value_, 1);
}

TEST_F(ConstFoldSimplifyPassTest, TestSimplifyIfFalseWithElse)
{
    auto cond = std::make_shared<Eq>(std::make_shared<ConstInt>(3, DataType::INT32, Sp()),
                                     std::make_shared<ConstInt>(4, DataType::INT32, Sp()), DataType::BOOL, Sp());
    auto dst = std::make_shared<Var>("selected", Scalar(DataType::INT32), Sp());
    auto then_body = std::make_shared<AssignStmt>(dst, std::make_shared<ConstInt>(9, DataType::INT32, Sp()), Sp());
    auto else_body = std::make_shared<AssignStmt>(dst, std::make_shared<ConstInt>(4, DataType::INT32, Sp()), Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body), std::vector<VarPtr>{},
                                           Sp());
    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 1u);
    auto result_stmt = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_.front());
    ASSERT_NE(result_stmt, nullptr);
    auto ci = std::dynamic_pointer_cast<const ConstInt>(result_stmt->value_);
    ASSERT_NE(ci, nullptr);
    EXPECT_EQ(ci->value_, 4);
}

TEST_F(ConstFoldSimplifyPassTest, TestSimplifyIfFalseNoElse)
{
    auto cond = std::make_shared<ConstInt>(0, DataType::BOOL, Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto then_body = std::make_shared<AssignStmt>(x, val, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, then_body, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto func = MakeFunc("f", ifStmt);
    auto result = RunPass(MakeProg(func));
    auto result_func = result->GetFunction("f");
    auto result_stmt = result_func->body_;
    ASSERT_NE(result_stmt, nullptr);
    EXPECT_TRUE(result_stmt->stmts_.empty());
}

// ============================================================================
// If-stmt simplification: same yield fold
// ============================================================================

TEST_F(ConstFoldSimplifyPassTest, TestFoldSameYieldBranches)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto then_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp());
    auto else_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, then_yield, std::optional<StmtPtr>(else_yield),
                                           std::vector<VarPtr>{ret}, Sp());
    auto func = MakeFunc("f", ifStmt);
    auto result = RunPass(MakeProg(func));
    auto result_func = result->GetFunction("f");
    auto result_body = result_func->body_;
    ASSERT_NE(result_body, nullptr);
    ASSERT_GE(result_body->stmts_.size(), 1u);
    StmtPtr folded_stmt = result_body->stmts_[0];
    auto inner = std::dynamic_pointer_cast<const SeqStmts>(result_body->stmts_[0]);
    if (inner) {
        ASSERT_GE(inner->stmts_.size(), 1u);
        folded_stmt = inner->stmts_[0];
    }
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(folded_stmt);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->var_->name_, "ret");
    auto ci = std::dynamic_pointer_cast<const ConstInt>(assign->value_);
    ASSERT_NE(ci, nullptr);
    EXPECT_EQ(ci->value_, 42);
}

TEST_F(ConstFoldSimplifyPassTest, TestFoldSameYieldBranchesFromSeqBodiesAndVarConst)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto alias = std::make_shared<Var>("alias", Scalar(DataType::INT32), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto tmp = std::make_shared<Var>("tmp", Scalar(DataType::INT32), Sp());
    auto seven = std::make_shared<ConstInt>(7, DataType::INT32, Sp());
    auto aliasAssign = std::make_shared<AssignStmt>(alias, seven, Sp());

    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{alias}, Sp());
    auto thenBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{thenYield}, Sp());
    auto elseAssign = std::make_shared<AssignStmt>(tmp, std::make_shared<ConstInt>(9, DataType::INT32, Sp()), Sp());
    auto elseYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(7, DataType::INT32, Sp())}, Sp());
    auto elseBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{elseAssign, elseYield}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{ret},
                                           Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{aliasAssign, ifStmt}, Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", body)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 2u);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[1]);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->var_->name_, "ret");
    auto var = std::dynamic_pointer_cast<const Var>(assign->value_);
    ASSERT_NE(var, nullptr);
    EXPECT_EQ(var->name_, "alias");
}

TEST_F(ConstFoldSimplifyPassTest, TestFoldSameYieldBranchesFromTupleGetItem)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto tupleType = std::make_shared<TupleType>(
        std::vector<TypePtr>{Scalar(DataType::INT32), Scalar(DataType::INT32)});
    auto tupleVar = std::make_shared<Var>("tuple_value", tupleType, Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto tuple = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(3, DataType::INT32, Sp()),
                             std::make_shared<ConstInt>(11, DataType::INT32, Sp())},
        Sp());
    auto tupleAssign = std::make_shared<AssignStmt>(tupleVar, tuple, Sp());
    auto item = std::make_shared<GetItemExpr>(tupleVar, std::make_shared<ConstInt>(1, DataType::INDEX, Sp()), Sp());
    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{item}, Sp());
    auto elseYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(11, DataType::INT32, Sp())}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{ret},
                                           Sp());
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{tupleAssign, ifStmt}, Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", body)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 2u);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[1]);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->var_->name_, "ret");
    EXPECT_NE(std::dynamic_pointer_cast<const GetItemExpr>(assign->value_), nullptr);
}

TEST_F(ConstFoldSimplifyPassTest, TestDifferentYieldBranchesKeepIf)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto thenYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())}, Sp());
    auto elseYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(2, DataType::INT32, Sp())}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{ret},
                                           Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    EXPECT_EQ(result_if->returnVars_.size(), 1u);
}

TEST_F(ConstFoldSimplifyPassTest, TestIfStmtRebuiltWhenBranchesFold)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto thenAdd = std::make_shared<Add>(std::make_shared<ConstInt>(1, DataType::INT32, Sp()),
                                         std::make_shared<ConstInt>(2, DataType::INT32, Sp()), DataType::INT32, Sp());
    auto elseAdd = std::make_shared<Add>(std::make_shared<ConstInt>(3, DataType::INT32, Sp()),
                                         std::make_shared<ConstInt>(4, DataType::INT32, Sp()), DataType::INT32, Sp());
    auto thenBody = std::make_shared<AssignStmt>(x, thenAdd, Sp());
    auto elseBody = std::make_shared<AssignStmt>(x, elseAdd, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{},
                                           Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    auto thenAssign = std::dynamic_pointer_cast<const AssignStmt>(result_if->thenBody_->stmts_[0]);
    ASSERT_NE(thenAssign, nullptr);
    auto thenConst = std::dynamic_pointer_cast<const ConstInt>(thenAssign->value_);
    ASSERT_NE(thenConst, nullptr);
    EXPECT_EQ(thenConst->value_, 3);
    ASSERT_TRUE(result_if->elseBody_.has_value());
    auto elseAssign = std::dynamic_pointer_cast<const AssignStmt>((*result_if->elseBody_)->stmts_[0]);
    ASSERT_NE(elseAssign, nullptr);
    auto elseConst = std::dynamic_pointer_cast<const ConstInt>(elseAssign->value_);
    ASSERT_NE(elseConst, nullptr);
    EXPECT_EQ(elseConst->value_, 7);
}

TEST_F(ConstFoldSimplifyPassTest, TestFoldSameYieldBranchesWithRebuiltYield)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto add = std::make_shared<Add>(std::make_shared<ConstInt>(2, DataType::INT32, Sp()),
                                     std::make_shared<ConstInt>(5, DataType::INT32, Sp()), DataType::INT32, Sp());
    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{add}, Sp());
    auto elseYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(7, DataType::INT32, Sp())}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{ret},
                                           Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto ci = std::dynamic_pointer_cast<const ConstInt>(assign->value_);
    ASSERT_NE(ci, nullptr);
    EXPECT_EQ(ci->value_, 7);
}

TEST_F(ConstFoldSimplifyPassTest, TestSameYieldVarNamesFold)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto thenVar = std::make_shared<Var>("same", Scalar(DataType::INT32), Sp());
    auto elseVar = std::make_shared<Var>("same", Scalar(DataType::INT32), Sp());
    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{thenVar}, Sp());
    auto elseYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{elseVar}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{ret},
                                           Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(assign, nullptr);
    auto var = std::dynamic_pointer_cast<const Var>(assign->value_);
    ASSERT_NE(var, nullptr);
    EXPECT_EQ(var->name_, "same");
}

TEST_F(ConstFoldSimplifyPassTest, TestYieldBranchesWithNonConstTupleIndexKeepIf)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto tupleType = std::make_shared<TupleType>(
        std::vector<TypePtr>{Scalar(DataType::INT32), Scalar(DataType::INT32), Scalar(DataType::INT32)});
    auto tupleVar = std::make_shared<Var>("tuple_value", tupleType, Sp());
    auto idxVar = std::make_shared<Var>("idx", Scalar(DataType::INDEX), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto tuple = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(3, DataType::INT32, Sp()),
                             std::make_shared<ConstInt>(11, DataType::INT32, Sp()),
                             std::make_shared<ConstInt>(19, DataType::INT32, Sp())},
        Sp());
    auto tupleAssign = std::make_shared<AssignStmt>(tupleVar, tuple, Sp());
    auto item = std::make_shared<GetItemExpr>(tupleVar, idxVar, Sp());
    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{item}, Sp());
    auto elseYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(19, DataType::INT32, Sp())}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{ret},
                                           Sp());
    std::vector<StmtPtr> stmts{tupleAssign};
    stmts.push_back(ifStmt);

    auto result = RunPass(MakeProg(MakeFunc("f", std::make_shared<SeqStmts>(stmts, Sp()))));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    ASSERT_EQ(result_func->body_->stmts_.size(), 2u);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_.back());
    ASSERT_NE(result_if, nullptr);
    EXPECT_EQ(result_if->returnVars_.size(), 1u);
}

TEST_F(ConstFoldSimplifyPassTest, TestNonYieldBranchPreventsSameYieldFold)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto thenAdd = std::make_shared<Add>(std::make_shared<ConstInt>(1, DataType::INT32, Sp()),
                                         std::make_shared<ConstInt>(2, DataType::INT32, Sp()), DataType::INT32, Sp());
    auto thenBody = std::make_shared<AssignStmt>(x, thenAdd, Sp());
    auto elseYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(3, DataType::INT32, Sp())}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{ret},
                                           Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    auto thenAssign = std::dynamic_pointer_cast<const AssignStmt>(result_if->thenBody_->stmts_[0]);
    ASSERT_NE(thenAssign, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<const ConstInt>(thenAssign->value_), nullptr);
}

TEST_F(ConstFoldSimplifyPassTest, TestReturnVarCountMismatchKeepsIf)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret0 = std::make_shared<Var>("ret0", Scalar(DataType::INT32), Sp());
    auto ret1 = std::make_shared<Var>("ret1", Scalar(DataType::INT32), Sp());
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp());
    auto elseYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield),
                                           std::vector<VarPtr>{ret0, ret1}, Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    EXPECT_EQ(result_if->returnVars_.size(), 2u);
}

TEST_F(ConstFoldSimplifyPassTest, TestNoElsePreventsSameYieldFold)
{
    auto cond = std::make_shared<Var>("c", Scalar(DataType::BOOL), Sp());
    auto ret = std::make_shared<Var>("ret", Scalar(DataType::INT32), Sp());
    auto thenYield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())}, Sp());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenYield, std::nullopt, std::vector<VarPtr>{ret}, Sp());

    auto result = RunPass(MakeProg(MakeFunc("f", ifStmt)));
    auto result_func = result->GetFunction("f");
    ASSERT_NE(result_func, nullptr);
    auto result_if = std::dynamic_pointer_cast<const IfStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_if, nullptr);
    EXPECT_FALSE(result_if->elseBody_.has_value());
}

// ============================================================================
// No-change: function with no foldable content
// ============================================================================

TEST_F(ConstFoldSimplifyPassTest, TestNoChangeWhenNothingToFold)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto add = std::make_shared<Add>(x, y, DataType::INT32, Sp());
    auto res = std::make_shared<Var>("res", Scalar(DataType::INT32), Sp());
    auto assign = std::make_shared<AssignStmt>(res, add, Sp());
    auto func = MakeFunc("f", assign);
    auto prog = MakeProg(func);
    auto result = RunPass(prog);
    auto result_func = result->GetFunction("f");
    EXPECT_EQ(result_func.get(), func.get());
    auto result_assign = std::dynamic_pointer_cast<const AssignStmt>(result_func->body_->stmts_[0]);
    ASSERT_NE(result_assign, nullptr);
    auto result_add = std::dynamic_pointer_cast<const Add>(result_assign->value_);
    ASSERT_NE(result_add, nullptr);
    EXPECT_EQ(result_add.get(), add.get());
}

} // namespace ir
} // namespace pypto
