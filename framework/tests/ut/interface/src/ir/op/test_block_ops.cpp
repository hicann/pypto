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
 * \file test_block_ops_type_deduction.cpp
 * \brief Coverage tests for block_ops type deduction via OpRegistry::Create
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// memory.cpp: get_block_idx, get_block_num, get_subblock_idx
// ============================================================================

class BlockOpsMemoryTest : public testing::Test {};

TEST_F(BlockOpsMemoryTest, GetBlockIdx_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_block_idx", {}, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetBlockNum_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_block_num", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetSubblockIdx_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_subblock_idx", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetBlockIdx_WithArgs_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("get_block_idx", {MakeScalarVar("x", DataType::INT32)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetBlockNum_WithArgs_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("get_block_num", {MakeScalarVar("x", DataType::INT32)}, Sp()), npu::tile_fwk::Error);
}

// ============================================================================
// memory.cpp: index_cast
// ============================================================================

TEST_F(BlockOpsMemoryTest, IndexCast_ScalarArg_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("index_cast", {MakeScalarVar("i", DataType::INT32)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

TEST_F(BlockOpsMemoryTest, IndexCast_NoArgs_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("index_cast", {}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, IndexCast_NonScalarArg_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("index_cast", {MakeTileVar("t", {16}, DataType::FP16)}, Sp()), npu::tile_fwk::Error);
}

// ============================================================================
// memory.cpp: block.make_tile
// ============================================================================

TEST_F(BlockOpsMemoryTest, MakeTile_BasicShape_ReturnsTileType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto dim2 = std::make_shared<ConstInt>(32, DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP16}};
    auto call = reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(BlockOpsMemoryTest, MakeTile_WithMemRef_ReturnsTileWithMemRef)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim}, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", DataType::INT32},
        {"memref_addr", int(0)},
        {"memref_size", int(256)},
        {"memref_id", int(1)},
    };
    auto call = reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_TRUE(rt->memref_.has_value());
}

TEST_F(BlockOpsMemoryTest, MakeTile_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP32}};
    EXPECT_THROW((void)reg.Create("block.make_tile", {}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, MakeTile_WithLayoutKwargs_ReturnsTileType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim1 = std::make_shared<ConstInt>(int64_t(16), DataType::INT64, Sp());
    auto dim2 = std::make_shared<ConstInt>(int64_t(32), DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", DataType::FP16},
        {"blayout", int(1)},
        {"slayout", int(0)},
        {"fractal", int(256)},
        {"pad", int(1)},
        {"compact", int(0)},
    };
    auto call = reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// memory.cpp: block.getval, block.setval
// ============================================================================

TEST_F(BlockOpsMemoryTest, GetVal_TileAndIndex_ReturnsScalarOfTileDtype)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.getval", {MakeTileVar("t", {64}, DataType::FP16), MakeScalarVar("i", DataType::INT32)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsMemoryTest, GetVal_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.getval", {MakeTileVar("t", {64}, DataType::FP32)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetVal_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.getval", {MakeScalarVar("a", DataType::FP32), MakeScalarVar("b", DataType::INT32)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetVal_NonIntIndex_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.getval", {MakeTileVar("t", {64}, DataType::FP16), MakeScalarVar("i", DataType::FP32)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_TileIndexValue_ReturnsTileType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.setval",
        {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::INT64), MakeScalarVar("v", DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsMemoryTest, SetVal_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.setval", {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::INT64)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.setval",
                   {MakeScalarVar("s", DataType::FP32), MakeScalarVar("i", DataType::INT32),
                    MakeScalarVar("v", DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_NonIntIndex_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.setval",
                   {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::FP32),
                    MakeScalarVar("v", DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_NonScalarValue_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.setval",
                   {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::INT64),
                    MakeTileVar("v", {128}, DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// struct_ops.cpp: struct.declare, struct.get, struct.set, struct.ref
// ============================================================================

class BlockOpsStructTest : public testing::Test {};

TEST_F(BlockOpsStructTest, StructDeclare_NoArgs_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("struct.declare", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

TEST_F(BlockOpsStructTest, StructGet_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("struct.get", {MakeScalarVar("idx", DataType::INT64)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

TEST_F(BlockOpsStructTest, StructSet_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("struct.set", {MakeScalarVar("idx", DataType::INT64), MakeScalarVar("val", DataType::INT64)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

TEST_F(BlockOpsStructTest, StructRef_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("struct.ref", {MakeScalarVar("idx", DataType::INT64)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

// ============================================================================
// out_elementwise.cpp: Tile x Tile binary ops
// ============================================================================

class BlockOpsOutElemwiseTest : public testing::Test {};

TEST_F(BlockOpsOutElemwiseTest, BlockAdd_ThreeTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.add", {lhs, rhs, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAdd_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.add", {MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {16, 32}, DataType::FP16)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAdd_NonTileArg_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.add",
                   {MakeScalarVar("s", DataType::FP32), MakeTileVar("r", {16, 32}, DataType::FP32),
                    MakeTileVar("o", {16, 32}, DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

// Representative coverage for REGISTER_BLOCK_OUT_BINARY_TILE macro-registered ops.
// All share the same DeduceBlockOutBinaryTile logic: validate 3 tile args, return out's type.

TEST_F(BlockOpsOutElemwiseTest, BlockSub_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP32);
    auto call = reg.Create("block.sub", {lhs, rhs, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockBitwiseAnd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::INT32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::INT32);
    auto out = MakeTileVar("out", {16, 32}, DataType::INT32);
    auto call = reg.Create("block.and", {lhs, rhs, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAddRelu_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.add_relu", {lhs, rhs, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockFusedMulAdd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.fused_mul_add", {lhs, rhs, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingBinaryTileOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);

    for (const auto& op_name : {
             "block.mul", "block.div", "block.rem", "block.maximum", "block.minimum",
             "block.or", "block.shl", "block.shr",
             "block.sub_relu", "block.mul_add_dst", "block.fused_mul_add_relu",
             "block.partadd", "block.partmax", "block.partmin", "block.partmul"}) {
        auto call = reg.Create(op_name, {lhs, rhs, out}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::FP16) << op_name;
    }
}

// block.add_relu_cast / sub_relu_cast / mul_cast: same deduction as binary tile ops
TEST_F(BlockOpsOutElemwiseTest, BlockAddReluCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP32);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_type", DataType::FP16}, {"mode", std::string("round")}};
    auto call = reg.Create("block.add_relu_cast", {lhs, rhs, out}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSubReluCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP32);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_type", DataType::FP16}, {"mode", std::string("round")}};
    auto call = reg.Create("block.sub_relu_cast", {lhs, rhs, out}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockMulCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP32);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_type", DataType::FP16}, {"mode", std::string("round")}};
    auto call = reg.Create("block.mul_cast", {lhs, rhs, out}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// out_elementwise.cpp: Tile x Scalar binary ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockAdds_TileScalarOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.adds",
        {MakeTileVar("t", {16, 16}, DataType::FP16), MakeScalarVar("s", DataType::FP16),
         MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAdds_NonScalarSecond_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.adds",
                   {MakeTileVar("t", {16, 16}, DataType::FP16), MakeTileVar("r", {16, 16}, DataType::FP16),
                    MakeTileVar("o", {16, 16}, DataType::FP16)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingBinaryScalarOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("t", {16, 16}, DataType::FP16);
    auto scalar = MakeScalarVar("s", DataType::FP16);
    auto out = MakeTileVar("o", {16, 16}, DataType::FP16);

    for (const auto& op_name : {
             "block.subs", "block.muls", "block.divs", "block.rems",
             "block.maxs", "block.mins", "block.lrelu", "block.axpy"}) {
        auto call = reg.Create(op_name, {tile, scalar, out}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::FP16) << op_name;
    }
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingBinaryScalarIntOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("t", {16, 16}, DataType::INT32);
    auto scalar = MakeScalarVar("s", DataType::INT32);
    auto out = MakeTileVar("o", {16, 16}, DataType::INT32);

    for (const auto& op_name : {"block.ands", "block.ors", "block.shls", "block.shrs"}) {
        auto call = reg.Create(op_name, {tile, scalar, out}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::INT32) << op_name;
    }
}

// ============================================================================
// out_elementwise.cpp: Unary ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockNeg_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.neg", {MakeTileVar("s", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockNeg_NonTileSrc_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.neg", {MakeScalarVar("s", DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingUnaryOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("s", {16, 16}, DataType::FP16);
    auto out = MakeTileVar("o", {16, 16}, DataType::FP16);

    for (const auto& op_name : {"block.exp", "block.sqrt", "block.rsqrt", "block.recip",
                                 "block.log", "block.abs", "block.relu"}) {
        auto call = reg.Create(op_name, {src, out}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::FP16) << op_name;
    }
}

TEST_F(BlockOpsOutElemwiseTest, BlockNot_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("s", {16, 16}, DataType::BOOL);
    auto out = MakeTileVar("o", {16, 16}, DataType::BOOL);
    auto call = reg.Create("block.not", {src, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::BOOL);
}

TEST_F(BlockOpsOutElemwiseTest, BlockCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", DataType::FP32}, {"mode", std::string("none")}};
    auto call = reg.Create(
        "block.cast", {MakeTileVar("s", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP32)}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockUnary_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.neg", {MakeTileVar("s", {16, 16}, DataType::FP16)}, Sp()), npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: Ternary / multi-input ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockXor_FourTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.xor",
        {MakeTileVar("l", {16, 16}, DataType::INT32), MakeTileVar("r", {16, 16}, DataType::INT32),
         MakeTileVar("tmp", {16, 16}, DataType::INT32), MakeTileVar("o", {16, 16}, DataType::INT32)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockXor_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.xor",
                   {MakeTileVar("l", {16, 16}, DataType::INT32), MakeTileVar("r", {16, 16}, DataType::INT32),
                    MakeTileVar("o", {16, 16}, DataType::INT32)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAddc_FourTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.addc",
        {MakeTileVar("a", {16, 16}, DataType::FP16), MakeTileVar("b", {16, 16}, DataType::FP16),
         MakeTileVar("c", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockXors_FourArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.xors",
        {MakeTileVar("lhs", {16, 16}, DataType::INT32), MakeScalarVar("s", DataType::INT32),
         MakeTileVar("tmp", {16, 16}, DataType::INT32), MakeTileVar("o", {16, 16}, DataType::INT32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockPrelu_FourArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.prelu",
        {MakeTileVar("tile", {16, 16}, DataType::FP16), MakeTileVar("slope", {16, 16}, DataType::FP16),
         MakeTileVar("tmp", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSubc_FourTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.subc",
        {MakeTileVar("a", {16, 16}, DataType::FP16), MakeTileVar("b", {16, 16}, DataType::FP16),
         MakeTileVar("c", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAddsc_FourArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.addsc",
        {MakeTileVar("lhs", {16, 16}, DataType::FP16), MakeScalarVar("s", DataType::FP16),
         MakeTileVar("rhs2", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSubsc_FourArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.subsc",
        {MakeTileVar("lhs", {16, 16}, DataType::FP16), MakeScalarVar("s", DataType::FP16),
         MakeTileVar("rhs2", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSels_FourArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.sels",
        {MakeTileVar("lhs", {16, 16}, DataType::FP16), MakeTileVar("rhs", {16, 16}, DataType::FP16),
         MakeScalarVar("mode", DataType::INT32), MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSel_FiveTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.sel",
        {MakeTileVar("l", {16, 16}, DataType::FP16), MakeTileVar("m", {16, 16}, DataType::FP16),
         MakeTileVar("r", {16, 16}, DataType::FP16), MakeTileVar("tmp", {16, 16}, DataType::FP16),
         MakeTileVar("o", {16, 16}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSel_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.sel",
                   {MakeTileVar("l", {16, 16}, DataType::FP16), MakeTileVar("m", {16, 16}, DataType::FP16),
                    MakeTileVar("r", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
                   Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: Comparison ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockCmp_ThreeTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"cmp_type", int(0)}};
    auto call = reg.Create(
        "block.cmp",
        {MakeTileVar("l", {16, 16}, DataType::FP16), MakeTileVar("r", {16, 16}, DataType::FP16),
         MakeTileVar("o", {16, 16}, DataType::FP16)},
        kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockCmps_TileScalarOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"cmp_type", int(1)}};
    auto call = reg.Create(
        "block.cmps",
        {MakeTileVar("t", {16, 16}, DataType::FP16), MakeScalarVar("s", DataType::FP16),
         MakeTileVar("o", {16, 16}, DataType::FP16)},
        kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_elementwise.cpp: Scalar broadcast, layout
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockExpands_ScalarAndOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.expands", {MakeScalarVar("s", DataType::FP16), MakeTileVar("o", {16, 32}, DataType::FP16)}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockCreateVecIdx_ScalarAndOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.create_vec_idx", {MakeScalarVar("start", DataType::INT32), MakeTileVar("o", {1, 128}, DataType::INT32)}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockReshape_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto shape = MakeOffsetsTuple({32, 16});
    auto call = reg.Create(
        "block.reshape", {MakeTileVar("s", {16, 32}, DataType::FP16), shape, MakeTileVar("o", {32, 16}, DataType::FP16)}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockTranspose_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"axis1", int(0)}, {"axis2", int(1)}};
    auto call = reg.Create(
        "block.transpose",
        {MakeTileVar("s", {16, 32}, DataType::FP16), MakeTileVar("o", {32, 16}, DataType::FP16)}, kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_elementwise.cpp: Quantize / dequantize
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockQuant_3Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"mode", std::string("sym")}};
    auto call = reg.Create(
        "block.quant",
        {MakeTileVar("src", {16, 32}, DataType::FP32), MakeTileVar("scale", {16, 1}, DataType::FP32),
         MakeTileVar("out", {16, 32}, DataType::INT8)},
        kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT8);
}

TEST_F(BlockOpsOutElemwiseTest, BlockDequant_4Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.dequant",
        {MakeTileVar("src", {16, 32}, DataType::INT8), MakeTileVar("scale", {16, 1}, DataType::FP32),
         MakeTileVar("offset", {16, 1}, DataType::FP32), MakeTileVar("out", {16, 32}, DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// out_matmul.cpp: block.matmul, block.matmul_acc, block.matmul_bias, block.gemv, block.gemv_acc, block.gemv_bias
// ============================================================================

class BlockOpsOutMatmulTest : public testing::Test {};

TEST_F(BlockOpsOutMatmulTest, BlockMatmul_LhsRhsOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.matmul",
        {MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16),
         MakeTileVar("o", {16, 16}, DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmulAcc_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.matmul_acc",
        {MakeTileVar("acc", {16, 16}, DataType::FP32), MakeTileVar("l", {16, 32}, DataType::FP16),
         MakeTileVar("r", {32, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP32)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMatmulTest, BlockGemv_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.gemv",
        {MakeTileVar("l", {1, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16),
         MakeTileVar("o", {1, 16}, DataType::FP32)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmulBias_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.matmul_bias",
        {MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16),
         MakeTileVar("bias", {1, 16}, DataType::FP32), MakeTileVar("o", {16, 16}, DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockGemvAcc_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.gemv_acc",
        {MakeTileVar("acc", {1, 16}, DataType::FP32), MakeTileVar("l", {1, 32}, DataType::FP16),
         MakeTileVar("r", {32, 16}, DataType::FP16), MakeTileVar("o", {1, 16}, DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockGemvBias_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.gemv_bias",
        {MakeTileVar("l", {1, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16),
         MakeTileVar("bias", {1, 16}, DataType::FP32), MakeTileVar("o", {1, 16}, DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmul_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.matmul", {MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16)}, Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.load, block.store, block.store_fp
// ============================================================================

class BlockOpsOutMemoryTest : public testing::Test {};

TEST_F(BlockOpsOutMemoryTest, BlockLoad_TensorOffsetsOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("tensor", {64, 128}, DataType::FP16);
    auto offsets = MakeOffsetsTuple({0, 0});
    auto out = MakeTileVar("out", {64, 128}, DataType::FP16);
    auto call = reg.Create("block.load", {tensor, offsets, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutMemoryTest, BlockLoad_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.load", {MakeTensorVar("t", {64, 128}, DataType::FP16), MakeOffsetsTuple({0, 0})}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockLoad_NonTensorFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.load",
                   {MakeScalarVar("s", DataType::FP32), MakeOffsetsTuple({0, 0}), MakeTileVar("o", {16, 32}, DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_TileOffsetsTensor_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("tile", {16, 32}, DataType::FP32);
    auto offsets = MakeOffsetsTuple({0, 0});
    auto tensor = MakeTensorVar("tensor", {16, 32}, DataType::FP32);
    auto call = reg.Create("block.store", {tile, offsets, tensor}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.store", {MakeTileVar("t", {16, 32}, DataType::FP32), MakeOffsetsTuple({0, 0})}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.store",
                   {MakeScalarVar("s", DataType::FP32), MakeOffsetsTuple({0, 0}),
                    MakeTensorVar("t", {16, 32}, DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockStoreFp_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("tile", {16, 32}, DataType::FP32);
    auto fp_tile = MakeTileVar("fp", {16, 32}, DataType::FP32);
    auto offsets = MakeOffsetsTuple({0, 0});
    auto tensor = MakeTensorVar("tensor", {16, 32}, DataType::FP32);
    auto call = reg.Create("block.store_fp", {tile, fp_tile, offsets, tensor}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockStoreFp_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.store_fp",
                   {MakeTileVar("t", {16, 32}, DataType::FP32), MakeOffsetsTuple({0, 0}),
                    MakeTensorVar("t2", {16, 32}, DataType::FP32)},
                   Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.move, block.move_fp, block.ub_copy
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockMove_SrcOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.move", {MakeTileVar("s", {16, 32}, DataType::FP16), MakeTileVar("o", {16, 32}, DataType::FP16)}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutMemoryTest, BlockMove_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.move", {MakeTileVar("s", {16, 32}, DataType::FP16)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockMoveFp_3Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.move_fp",
        {MakeTileVar("s", {16, 32}, DataType::FP32), MakeTileVar("fp", {16, 32}, DataType::FP32),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutMemoryTest, BlockMoveFp_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.move_fp", {MakeTileVar("s", {16, 32}, DataType::FP32), MakeTileVar("o", {16, 32}, DataType::FP16)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockUbCopy_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.ub_copy", {MakeTileVar("s", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_memory.cpp: block.insert
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockInsert_4Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.insert",
        {MakeTileVar("src", {16, 16}, DataType::FP16), MakeScalarVar("row", DataType::INT32),
         MakeScalarVar("col", DataType::INT32), MakeTileVar("o", {64, 64}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockInsert_5Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.insert",
        {MakeTileVar("src", {16, 16}, DataType::FP16), MakeScalarVar("row", DataType::INT32),
         MakeScalarVar("col", DataType::INT32), MakeScalarVar("off", DataType::INT32),
         MakeTileVar("o", {64, 64}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockInsert_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.insert", {MakeTileVar("s", {16, 16}, DataType::FP16), MakeScalarVar("r", DataType::INT32)}, Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.full, block.ssbuf_store, block.ssbuf_load
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockFull_ScalarOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.full", {MakeScalarVar("v", DataType::FP16), MakeTileVar("o", {16, 32}, DataType::FP16)}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockSsbufStore_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.ssbuf_store", {MakeScalarVar("off", DataType::INT32)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

TEST_F(BlockOpsOutMemoryTest, BlockSsbufLoad_ReturnsIndexScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.ssbuf_load", {MakeScalarVar("off", DataType::INT32)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INDEX);
}

// ============================================================================
// out_memory.cpp: block.set_validshape
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockSetValidshape_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.set_validshape",
        {MakeScalarVar("row", DataType::INT32), MakeScalarVar("col", DataType::INT32),
         MakeTileVar("t", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_memory.cpp: block.fillpad, block.fillpad_expand
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockFillpad_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {16, 32}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::zero);
    auto call = reg.Create("block.fillpad", {src, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpad_NoHardwareInfo_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("src", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    EXPECT_THROW((void)reg.Create("block.fillpad", {src, out}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpad_PadNull_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("src", {16, 32}, DataType::FP16);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::null);
    EXPECT_THROW((void)reg.Create("block.fillpad", {src, out}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadExpand_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {16, 32}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {32, 64}, DataType::FP16, TilePad::zero);
    auto call = reg.Create("block.fillpad_expand", {src, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadExpand_SrcLargerThanOut_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {32, 64}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::zero);
    EXPECT_THROW((void)reg.Create("block.fillpad_expand", {src, out}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadInplace_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithMemRef("src", {16, 32}, DataType::FP16, MemorySpace::Vec, 0, 1024);
    auto addr_expr = std::make_shared<ConstInt>(int64_t(0), DataType::INDEX, Sp());
    auto memref = std::make_shared<MemRef>(MemorySpace::Vec, addr_expr, uint64_t(1024));
    HardwareInfo hw(TileLayout::row_major, TileLayout::none_box, 512, TilePad::zero, CompactMode::null);
    auto out_type = std::make_shared<TileType>(
        std::vector<int64_t>{16, 32},
        DataType::FP16, std::optional<MemRefPtr>(memref), std::nullopt, std::optional<HardwareInfo>(hw));
    auto out = std::make_shared<Var>("out", out_type, Sp());
    auto call = reg.Create("block.fillpad_inplace", {src, out}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadInplace_NoMemRef_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {16, 32}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::zero);
    EXPECT_THROW((void)reg.Create("block.fillpad_inplace", {src, out}, Sp()), npu::tile_fwk::Error);
}

// ============================================================================
// out_reduction.cpp: reductions and broadcasts
// ============================================================================

class BlockOpsOutReductionTest : public testing::Test {};

TEST_F(BlockOpsOutReductionTest, BlockRowSum_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_sum",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 1}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_max",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 1}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_min",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 1}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_max",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::FP16),
         MakeTileVar("o", {1, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowArgmax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_argmax",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::INT32),
         MakeTileVar("o", {16, 1}, DataType::INT32)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpand_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand",
        {MakeTileVar("s", {16, 1}, DataType::FP16), MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandAdd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_add",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandAdd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_add",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColSum_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_sum",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::FP16),
         MakeTileVar("o", {1, 32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_min",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::FP16),
         MakeTileVar("o", {1, 32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowProd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_prod",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 1}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColProd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_prod",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::FP16),
         MakeTileVar("o", {1, 32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowReduce_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_reduce",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 1}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColReduce_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_reduce",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::FP16),
         MakeTileVar("o", {1, 32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowArgmin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_argmin",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {16, 1}, DataType::INT32),
         MakeTileVar("o", {16, 1}, DataType::INT32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutReductionTest, BlockColArgmax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_argmax",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::INT32),
         MakeTileVar("o", {1, 32}, DataType::INT32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutReductionTest, BlockColArgmin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_argmin",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("tmp", {1, 32}, DataType::INT32),
         MakeTileVar("o", {1, 32}, DataType::INT32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpand_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand",
        {MakeTileVar("cv", {1, 32}, DataType::FP16), MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandSub_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_sub",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandMul_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_mul",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandExpdif_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_expdif",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandBinop_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_binop",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandMul_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_mul",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandExpdif_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_expdif",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandBinop_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_binop",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandDiv_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_div",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_max",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.row_expand_min",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("rv", {16, 1}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandDiv_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_div",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandSub_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_sub",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_max",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.col_expand_min",
        {MakeTileVar("t", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16),
         MakeTileVar("o", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// sort.cpp: block.sort32, block.mrgsort, block.mrgsort2, block.histogram
// ============================================================================

class BlockOpsSortTest : public testing::Test {};

TEST_F(BlockOpsSortTest, BlockSort32_3Args_ReturnsDstType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.sort32",
        {MakeTileVar("src", {32}, DataType::FP16), MakeTileVar("idx", {32}, DataType::UINT32),
         MakeTileVar("dst", {32}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsSortTest, BlockSort32_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.sort32", {MakeTileVar("src", {32}, DataType::FP16)}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsSortTest, BlockMrgsort_SrcDst_ReturnsDstType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"block_len", int(32)}};
    auto call = reg.Create(
        "block.mrgsort",
        {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("dst", {64}, DataType::FP16)},
        kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsSortTest, BlockMrgsort_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.mrgsort", {MakeTileVar("src", {64}, DataType::FP16)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsSortTest, BlockMrgsort2_4Args_ReturnsDstType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.mrgsort2",
        {MakeTileVar("src0", {64}, DataType::FP16), MakeTileVar("dst", {64}, DataType::FP16),
         MakeTileVar("tmp", {64}, DataType::FP16), MakeTileVar("src1", {64}, DataType::FP16)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsSortTest, BlockMrgsort2_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.mrgsort2",
                   {MakeTileVar("s0", {64}, DataType::FP16), MakeTileVar("dst", {64}, DataType::FP16),
                    MakeTileVar("tmp", {64}, DataType::FP16)},
                   Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsSortTest, BlockHistogram_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.histogram",
        {MakeTileVar("src", {32}, DataType::UINT16), MakeTileVar("idx", {32}, DataType::UINT8),
         MakeTileVar("dst", {1, 256}, DataType::UINT32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::UINT32);
}

TEST_F(BlockOpsSortTest, BlockHistogram_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.histogram", {MakeTileVar("src", {32}, DataType::UINT16), MakeTileVar("idx", {32}, DataType::UINT8)}, Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: block.gather (index form and compare form)
// ============================================================================

class BlockOpsGatherTest : public testing::Test {};

TEST_F(BlockOpsGatherTest, BlockGather_IndexForm3Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.gather",
        {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("indices", {64}, DataType::INT32),
         MakeTileVar("out", {64}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsGatherTest, BlockGather_CompareForm5Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"cmp_mode", int(0)}, {"offset", int(0)}};
    auto call = reg.Create(
        "block.gather",
        {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("kval", {64}, DataType::FP16),
         MakeTileVar("cdst", {64}, DataType::FP16), MakeTileVar("tmp", {64}, DataType::FP16),
         MakeTileVar("out", {64}, DataType::FP16)},
        kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsGatherTest, BlockGatherb_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.gatherb",
        {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("offsets", {64}, DataType::INT32),
         MakeTileVar("out", {64}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsGatherTest, BlockGathermask_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pattern_mode", int(1)}};
    auto call = reg.Create(
        "block.gathermask",
        {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("out", {64}, DataType::FP16)},
        kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockGather_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.gather",
                   {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("indices", {64}, DataType::INT32)},
                   Sp()),
        std::runtime_error);
}

TEST_F(BlockOpsGatherTest, BlockScatter_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.scatter",
        {MakeTileVar("src", {16, 32}, DataType::FP16), MakeTileVar("indices", {16, 32}, DataType::INT32),
         MakeTileVar("dst", {16, 32}, DataType::FP16)},
        Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// OpRegistry::GetEntry and GetOp
// ============================================================================

class OpRegistryQueryTest : public testing::Test {};

TEST_F(OpRegistryQueryTest, GetEntry_RegisteredOp_ReturnsEntry)
{
    auto& reg = OpRegistry::GetInstance();
    const auto& entry = reg.GetEntry("get_block_idx");
    EXPECT_EQ(entry.GetName(), "get_block_idx");
    EXPECT_NE(entry.GetOp(), nullptr);
}

TEST_F(OpRegistryQueryTest, GetEntry_UnregisteredOp_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.GetEntry("nonexistent_op_xyz"), npu::tile_fwk::Error);
}

TEST_F(OpRegistryQueryTest, GetOp_RegisteredOp_ReturnsOp)
{
    auto& reg = OpRegistry::GetInstance();
    auto op = reg.GetOp("block.add");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->name_, "block.add");
}

TEST_F(OpRegistryQueryTest, GetOp_UnregisteredOp_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.GetOp("nonexistent_op_abc"), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
