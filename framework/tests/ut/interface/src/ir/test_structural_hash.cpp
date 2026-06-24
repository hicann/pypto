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
 * \file test_structural_hash.cpp
 * \brief Coverage tests for structural_hash.cpp
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }
static auto Node(const IRNodePtr& p) { return p; }

class IRStructHashTest : public testing::Test {};

// ============================================================================
// Var hashing — the main dispatch path in StructuralHasher
// ============================================================================

TEST_F(IRStructHashTest, TestHashVarDeterministicAndNonZero)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(a)), structural_hash(Node(a)));
    EXPECT_NE(structural_hash(Node(a)), 0u);
}

TEST_F(IRStructHashTest, TestHashDifferentVarsDifferentHash)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    EXPECT_NE(structural_hash(Node(a)), structural_hash(Node(b)));
}

TEST_F(IRStructHashTest, TestHashVarAutoMappingEqual)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(a), true), structural_hash(Node(b), true));
}

TEST_F(IRStructHashTest, TestHashVarNoAutoMappingByName)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(a), false), structural_hash(Node(b), false));
}

// ============================================================================
// Type hashing — ScalarType, TensorType, TileType, TupleType, MemRefType
// ============================================================================

TEST_F(IRStructHashTest, TestHashScalarTypeDeterministicAndDifferent)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::INT32);
    auto t3 = std::make_shared<ScalarType>(DataType::FP32);
    EXPECT_NE(structural_hash(t1), 0u);
    EXPECT_EQ(structural_hash(t1), structural_hash(t2));
    EXPECT_NE(structural_hash(t1), structural_hash(t3));
}

TEST_F(IRStructHashTest, TestHashTensorTypeWithVarShape)
{
    auto d = std::make_shared<Var>("N", Scalar(DataType::INT64), Sp());
    auto t = std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP32);
    EXPECT_NE(structural_hash(t), 0u);

    auto d2 = std::make_shared<Var>("N", Scalar(DataType::INT64), Sp());
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{d2}, DataType::FP32);
    EXPECT_EQ(structural_hash(t, false), structural_hash(t2, false));
}

TEST_F(IRStructHashTest, TestHashTileType)
{
    auto d = std::make_shared<Var>("T", Scalar(DataType::INT64), Sp());
    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP32);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP16);
    EXPECT_NE(structural_hash(t1), 0u);
    EXPECT_NE(structural_hash(t1), structural_hash(t2));
}

TEST_F(IRStructHashTest, TestHashTupleType)
{
    auto t1 = std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::INT32)});
    auto t2 = std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::INT32)});
    auto t3 = std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::FP32)});
    EXPECT_NE(structural_hash(t1), 0u);
    EXPECT_EQ(structural_hash(t1), structural_hash(t2));
    EXPECT_NE(structural_hash(t1), structural_hash(t3));
}

TEST_F(IRStructHashTest, TestHashMemRefAndUnknownType)
{
    EXPECT_NE(structural_hash(GetMemRefType()), 0u);
    EXPECT_NE(structural_hash(std::make_shared<UnknownType>()), 0u);
}

} // namespace ir
} // namespace pypto
