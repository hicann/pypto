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
 * \file test_structural_equal_ext.cpp
 * \brief Coverage tests for structural_equal.cpp — exercising all EQUAL_DISPATCH paths
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }
static auto Node(const IRNodePtr& p) { return p; }

// ============================================================================
// Expression equality
// ============================================================================

class IRStructEqExprTest : public testing::Test {};

TEST_F(IRStructEqExprTest, TestConstAndLeafExprsEqual)
{
    // ConstFloat
    auto f1 = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    auto f2 = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    EXPECT_TRUE(structural_equal(Node(f1), Node(f2)));
    auto f3 = std::make_shared<ConstFloat>(2.71, DataType::FP32, Sp());
    EXPECT_FALSE(structural_equal(Node(f1), Node(f3)));

    // ConstBool
    auto b1 = std::make_shared<ConstBool>(true, Sp());
    auto b2 = std::make_shared<ConstBool>(true, Sp());
    EXPECT_TRUE(structural_equal(Node(b1), Node(b2)));
    EXPECT_FALSE(structural_equal(Node(b1), Node(std::make_shared<ConstBool>(false, Sp()))));

    // MemRef
    auto off1 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto off2 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto mr1 = std::make_shared<MemRef>(MemorySpace::DDR, off1, 1024, Sp());
    auto mr2 = std::make_shared<MemRef>(MemorySpace::DDR, off2, 1024, Sp());
    EXPECT_TRUE(structural_equal(Node(mr1), Node(mr2)));
    auto off3 = std::make_shared<ConstInt>(100, DataType::INT64, Sp());
    auto mr3 = std::make_shared<MemRef>(MemorySpace::DDR, off3, 1024, Sp());
    EXPECT_FALSE(structural_equal(Node(mr1), Node(mr3)));

    // IterArg — same pointer
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ia = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), init, Sp());
    EXPECT_TRUE(structural_equal(Node(ia), Node(ia)));

    // IterArg — auto-mapping
    auto init2 = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ia2 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), init2, Sp());
    EXPECT_TRUE(structural_equal(Node(ia), Node(ia2), true));
}

TEST_F(IRStructEqExprTest, TestCallAndTupleExprsEqual)
{
    auto a1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto a2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // Call
    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a1}, Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a2}, Sp());
    EXPECT_TRUE(structural_equal(Node(c1), Node(c2)));
    auto c3 = std::make_shared<Call>("op_b", std::vector<ExprPtr>{a1}, Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c3)));

    // MakeTuple
    auto b1 = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto b2 = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto t1 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a1, b1}, Sp());
    auto t2 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a2, b2}, Sp());
    EXPECT_TRUE(structural_equal(Node(t1), Node(t2)));
    auto t3 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a1, a1}, Sp());
    EXPECT_FALSE(structural_equal(Node(t1), Node(t3)));

    // GetItemExpr
    auto idx1 = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto idx2 = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto g1 = std::make_shared<GetItemExpr>(t1, idx1, Sp());
    auto g2 = std::make_shared<GetItemExpr>(t2, idx2, Sp());
    EXPECT_TRUE(structural_equal(Node(g1), Node(g2)));

    // BinaryExpr
    auto add1 = std::make_shared<Add>(a1, b1, DataType::INT32, Sp());
    auto add2 = std::make_shared<Add>(a2, b2, DataType::INT32, Sp());
    EXPECT_TRUE(structural_equal(Node(add1), Node(add2)));
    auto sub1 = std::make_shared<Sub>(a1, b1, DataType::INT32, Sp());
    EXPECT_FALSE(structural_equal(Node(add1), Node(sub1)));

    // UnaryExpr
    auto neg1 = std::make_shared<Neg>(a1, DataType::INT32, Sp());
    auto neg2 = std::make_shared<Neg>(a2, DataType::INT32, Sp());
    EXPECT_TRUE(structural_equal(Node(neg1), Node(neg2)));
}

// ============================================================================
// Statement equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestAllStmtsEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v42a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto v42b = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{v42a}, Sp());

    // AssignStmt
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<AssignStmt>(x, v42a, Sp())), Node(std::make_shared<AssignStmt>(x, v42b, Sp()))));

    // IfStmt without else
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp())),
        Node(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp()))));

    // IfStmt with else
    auto then = std::make_shared<AssignStmt>(x, v1, Sp());
    auto els = std::make_shared<AssignStmt>(x, v2, Sp());
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<IfStmt>(cond, then, els, std::vector<VarPtr>{}, Sp())),
        Node(std::make_shared<IfStmt>(cond, then, els, std::vector<VarPtr>{}, Sp()))));

    // YieldStmt, ReturnStmt
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<YieldStmt>(std::vector<ExprPtr>{v1}, Sp())),
        Node(std::make_shared<YieldStmt>(std::vector<ExprPtr>{v2}, Sp()))));
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v42a}, Sp())),
        Node(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v42b}, Sp()))));

    // ForStmt, WhileStmt
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<ForStmt>(
            i, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp())),
        Node(std::make_shared<ForStmt>(
            i, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp()))));
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp())),
        Node(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp()))));

    // SeqStmts
    auto a = std::make_shared<AssignStmt>(x, v1, Sp());
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a, a}, Sp())),
        Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a, a}, Sp()))));

    // BreakStmt, ContinueStmt, EvalStmt
    EXPECT_TRUE(structural_equal(Node(std::make_shared<BreakStmt>(Sp())), Node(std::make_shared<BreakStmt>(Sp()))));
    EXPECT_TRUE(
        structural_equal(Node(std::make_shared<ContinueStmt>(Sp())), Node(std::make_shared<ContinueStmt>(Sp()))));
    EXPECT_TRUE(
        structural_equal(Node(std::make_shared<EvalStmt>(call, Sp())), Node(std::make_shared<EvalStmt>(call, Sp()))));
}

// ============================================================================
// Function and Program equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestFunctionAndProgramEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto body1 = std::make_shared<AssignStmt>(x, v1, Sp());
    auto body2 = std::make_shared<AssignStmt>(x, v2, Sp());

    // Function equal
    auto f1 = std::make_shared<Function>(
        "f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)}, body1, Sp());
    auto f2 = std::make_shared<Function>(
        "f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)}, body2, Sp());
    EXPECT_TRUE(structural_equal(Node(f1), Node(f2)));

    // Function not equal (different body)
    auto v99 = std::make_shared<ConstInt>(99, DataType::INT32, Sp());
    auto body3 = std::make_shared<AssignStmt>(x, v99, Sp());
    auto f3 = std::make_shared<Function>(
        "f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)}, body3, Sp());
    EXPECT_FALSE(structural_equal(Node(f1), Node(f3)));

    // Program equal
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto fbody = std::make_shared<AssignStmt>(x, v, Sp());
    auto f = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, fbody, Sp());
    auto p1 = std::make_shared<Program>(std::vector<FunctionPtr>{f}, "prog", Sp());
    auto p2 = std::make_shared<Program>(std::vector<FunctionPtr>{f}, "prog", Sp());
    EXPECT_TRUE(structural_equal(Node(p1), Node(p2)));
}

// ============================================================================
// Type equality
// ============================================================================

class IRStructEqTypeTest : public testing::Test {};

TEST_F(IRStructEqTypeTest, TestTensorAndTileTypeEqual)
{
    auto d16a = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto d16b = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto d32 = std::make_shared<ConstInt>(32, DataType::INT64, Sp());

    // TensorType equal / dtype mismatch / shape mismatch
    auto t1 = std::make_shared<TensorType>(std::vector<ExprPtr>{d16a}, DataType::FP32);
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{d16b}, DataType::FP32);
    EXPECT_TRUE(structural_equal(t1, t2));
    EXPECT_FALSE(structural_equal(t1, std::make_shared<TensorType>(std::vector<ExprPtr>{d16a}, DataType::FP16)));
    EXPECT_FALSE(structural_equal(t1, std::make_shared<TensorType>(std::vector<ExprPtr>{d32}, DataType::FP32)));

    // TileType equal / dtype mismatch / rank mismatch
    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP32);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16b}, DataType::FP32);
    EXPECT_TRUE(structural_equal(tl1, tl2));
    EXPECT_FALSE(structural_equal(tl1, std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP16)));
    EXPECT_FALSE(structural_equal(tl1, std::make_shared<TileType>(std::vector<ExprPtr>{d16a, d16a}, DataType::FP32)));
}

TEST_F(IRStructEqTypeTest, TestTupleAndOtherTypes)
{
    auto s32 = std::make_shared<ScalarType>(DataType::INT32);

    // TupleType size mismatch
    EXPECT_FALSE(structural_equal(
        std::make_shared<TupleType>(std::vector<TypePtr>{s32}),
        std::make_shared<TupleType>(std::vector<TypePtr>{s32, s32})));

    // Type name mismatch
    EXPECT_FALSE(structural_equal(s32, std::make_shared<TupleType>(std::vector<TypePtr>{})));
    EXPECT_FALSE(structural_equal(
        s32, std::make_shared<TensorType>(
                 std::vector<ExprPtr>{std::make_shared<ConstInt>(16, DataType::INT64, Sp())}, DataType::FP32)));

    // MemRefType singleton
    EXPECT_TRUE(structural_equal(GetMemRefType(), GetMemRefType()));
}

// ============================================================================
// assert_structural_equal mismatch error paths
// ============================================================================

class IRStructEqAssertTest : public testing::Test {};

TEST_F(IRStructEqAssertTest, TestAssertExprMismatches)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());

    // Type mismatch (ConstInt vs ConstFloat)
    EXPECT_THROW(
        assert_structural_equal(Node(a), Node(std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp()))), ValueError);

    // Null mismatch
    EXPECT_THROW(assert_structural_equal(Node(a), nullptr), ValueError);

    // Int mismatch
    EXPECT_THROW(assert_structural_equal(Node(a), Node(b)), ValueError);

    // Float mismatch
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp())),
            Node(std::make_shared<ConstFloat>(2.0, DataType::FP32, Sp()))),
        ValueError);

    // Bool mismatch
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<ConstBool>(true, Sp())), Node(std::make_shared<ConstBool>(false, Sp()))),
        ValueError);

    // Call name mismatch
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<Call>("op_a", std::vector<ExprPtr>{a}, Sp())),
            Node(std::make_shared<Call>("op_b", std::vector<ExprPtr>{a}, Sp()))),
        ValueError);

    // Call arg count mismatch
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<Call>("op", std::vector<ExprPtr>{a}, Sp())),
            Node(std::make_shared<Call>("op", std::vector<ExprPtr>{a, b}, Sp()))),
        ValueError);
}

TEST_F(IRStructEqAssertTest, TestAssertStmtMismatches)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    // Function body mismatch
    auto body1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto body2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body1, Sp())),
            Node(std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body2, Sp()))),
        ValueError);

    // Optional field mismatch (IfStmt with/without else)
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp())),
            Node(std::make_shared<IfStmt>(cond, yield, yield, std::vector<VarPtr>{}, Sp()))),
        ValueError);

    // Vector size mismatch (SeqStmts)
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1}, Sp())),
            Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp()))),
        ValueError);
}

TEST_F(IRStructEqAssertTest, TestAssertTypeMismatches)
{
    auto d = std::make_shared<ConstInt>(16, DataType::INT64, Sp());

    // Scalar dtype
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<ScalarType>(DataType::INT32), std::make_shared<ScalarType>(DataType::FP32)),
        ValueError);

    // Tuple elements
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::INT32)}),
            std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::FP32)})),
        ValueError);

    // Tensor dtype / shape rank
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP32),
            std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP16)),
        ValueError);
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP32),
            std::make_shared<TensorType>(std::vector<ExprPtr>{d, d}, DataType::FP32)),
        ValueError);

    // Tile dtype / shape rank
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP32),
            std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP16)),
        ValueError);
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP32),
            std::make_shared<TileType>(std::vector<ExprPtr>{d, d}, DataType::FP32)),
        ValueError);

    // Tuple size
    auto s = std::make_shared<ScalarType>(DataType::INT32);
    EXPECT_THROW(
        assert_structural_equal(
            std::make_shared<TupleType>(std::vector<TypePtr>{s}),
            std::make_shared<TupleType>(std::vector<TypePtr>{s, s})),
        ValueError);
}

// ============================================================================
// Auto-mapping variable equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestVarAutoMappingEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // Auto-mapping: x↔y mapped, same structure
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<AssignStmt>(x, v1, Sp())), Node(std::make_shared<AssignStmt>(y, v2, Sp())), true));

    // Without auto-mapping, distinct vars differ
    EXPECT_FALSE(structural_equal(Node(x), Node(y), false));

    // Auto-mapping with type mismatch
    auto z = std::make_shared<Var>("z", Scalar(DataType::FP32), Sp());
    EXPECT_FALSE(structural_equal(
        Node(std::make_shared<AssignStmt>(x, v1, Sp())), Node(std::make_shared<AssignStmt>(z, v1, Sp())), true));
}

TEST_F(IRStructEqExprTest, TestVarConsistentMapping)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // x used twice on left, y used twice on right — consistent mapping
    auto body1 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(x, v1, Sp()), std::make_shared<AssignStmt>(x, v2, Sp())},
        Sp());
    auto body2 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(y, v1, Sp()), std::make_shared<AssignStmt>(y, v2, Sp())},
        Sp());
    EXPECT_TRUE(structural_equal(Node(body1), Node(body2), true));
}

// ============================================================================
// ForStmt with iterArgs — IterArg initValue comparison
// ============================================================================

TEST_F(IRStructEqExprTest, TestForStmtWithIterArgs)
{
    auto i1 = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto i2 = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto yv = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    auto ia1 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ia2 = std::make_shared<IterArg>(
        "acc", Scalar(DataType::INT32), std::make_shared<ConstInt>(0, DataType::INT32, Sp()), Sp());
    auto rv1 = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());
    auto rv2 = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());

    auto body1 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yv}, Sp());
    auto body2 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yv}, Sp());

    // Equal with same init
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<ForStmt>(
            i1, zero, ten, one, std::vector<IterArgPtr>{ia1}, body1, std::vector<VarPtr>{rv1}, Sp())),
        Node(std::make_shared<ForStmt>(
            i2, zero, ten, one, std::vector<IterArgPtr>{ia2}, body2, std::vector<VarPtr>{rv2}, Sp()))));

    // Not equal with different init
    auto ia3 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), one, Sp());
    EXPECT_FALSE(structural_equal(
        Node(
            std::make_shared<ForStmt>(
                i1, zero, ten, one, std::vector<IterArgPtr>{ia1}, body1, std::vector<VarPtr>{rv1}, Sp())),
        Node(
            std::make_shared<ForStmt>(
                i1, zero, ten, one, std::vector<IterArgPtr>{ia3}, body1, std::vector<VarPtr>{rv1}, Sp()))));
}

} // namespace ir
} // namespace pypto
