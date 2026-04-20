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
 * \file binary_brcinline.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_BRCINLINE__H
#define TILEOP_TILE_OPERATOR_BINARY_BRCINLINE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

enum class BrcMode : uint8_t {
    NONE,
    BRC_W,     // [m, 1] [m, n] / [m, n] [m, 1]
    BRC_H,     // [1, n] [m, n] / [m, n] [1, n]
    BRC_HW,    // [1, 1] [m ,n] / [m, n] [1, 1]
    BRC_W0_H1, // [m, 1] [1, n]
    BRC_H0_W1  // [1, n] [m, 1]
};

template <TileOp::BroadcastOperand WBrcSide, TileOp::PenuBroadcastOperand HBrcSide>
TILEOP constexpr BrcMode GetBrcMode()
{
    if constexpr (WBrcSide != TileOp::BroadcastOperand::NONE && HBrcSide == TileOp::PenuBroadcastOperand::NONE) {
        return BrcMode::BRC_W;
    } else if constexpr (WBrcSide == TileOp::BroadcastOperand::NONE && HBrcSide != TileOp::PenuBroadcastOperand::NONE) {
        return BrcMode::BRC_H;
    } else if constexpr (
        (WBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND &&
         HBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) ||
        (WBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND &&
         HBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND)) {
        return BrcMode::BRC_HW;
    } else if constexpr (
        WBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND && HBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND) {
        return BrcMode::BRC_W0_H1;
    } else if constexpr (
        WBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND && HBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND) {
        return BrcMode::BRC_H0_W1;
    } else {
        return BrcMode::NONE;
    }
}

#define EXTRACT_LAST_USE_3DIM(LastUse)                                     \
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value; \
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value; \
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;

#define BINARY_EXPAND_DISPATCH(PREFIX, PrecisionType)                                       \
    if constexpr (op == BinaryOp::ADD) {                                                    \
        PTO_WITH_LAST_USE(pto::T##PREFIX##ADD(dst, src0, src1), n1, n2, n3);                \
        return;                                                                             \
    } else if constexpr (op == BinaryOp::SUB) {                                             \
        PTO_WITH_LAST_USE(pto::T##PREFIX##SUB(dst, src0, src1), n1, n2, n3);                \
        return;                                                                             \
    } else if constexpr (op == BinaryOp::MUL) {                                             \
        PTO_WITH_LAST_USE(pto::T##PREFIX##MUL(dst, src0, src1), n1, n2, n3);                \
        return;                                                                             \
    } else if constexpr (op == BinaryOp::DIV) {                                             \
        PTO_WITH_LAST_USE(pto::T##PREFIX##DIV<PrecisionType>(dst, src0, src1), n1, n2, n3); \
        return;                                                                             \
    } else if constexpr (op == BinaryOp::MAX) {                                             \
        PTO_WITH_LAST_USE(pto::T##PREFIX##MAX(dst, src0, src1), n1, n2, n3);                \
        return;                                                                             \
    } else if constexpr (op == BinaryOp::MIN) {                                             \
        PTO_WITH_LAST_USE(pto::T##PREFIX##MIN(dst, src0, src1), n1, n2, n3);                \
        return;                                                                             \
    } else if constexpr (op == BinaryOp::EXPANDEXPDIF) {                                    \
        PTO_WITH_LAST_USE(pto::T##PREFIX##EXPDIF(dst, src0, src1), n1, n2, n3);             \
        return;                                                                             \
    }

template <
    BinaryOp op, pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse, typename T0,
    typename T1, typename T2>
TILEOP void BinaryRowExpandComputeImpl(T0 dst, T1 src0, T2 src1)
{
    EXTRACT_LAST_USE_3DIM(LastUse)
    BINARY_EXPAND_DISPATCH(ROWEXPAND, PrecisionType)
}

template <
    BinaryOp op, pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse, typename T0,
    typename T1, typename T2>
TILEOP void BinaryColExpandComputeImpl(T0 dst, T1 src0, T2 src1)
{
    EXTRACT_LAST_USE_3DIM(LastUse)
    BINARY_EXPAND_DISPATCH(COLEXPAND, PrecisionType)
    if constexpr (op == BinaryOp::MOD) {
        pto::TROWEXPANDDIV<PrecisionType>(dst, src0, src1);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TCVT(dst, dst, pto::RoundMode::CAST_TRUNC);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TROWEXPANDMUL(dst, dst, src1);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TROWEXPANDSUB(dst, src0, dst);
    }
}

template <
    BinaryOp op, pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, TileOp::BroadcastOperand WBrcSide,
    TileOp::PenuBroadcastOperand HBrcSide, typename Src0TileInfo, typename Src1TileInfo, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryMixBrcCompute(T0 dst, T1 src0, T2 src1)
{
    const auto dstLayout = dst.GetLayout();
    const auto src0Layout = src0.GetLayout();
    const auto src1Layout = src1.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto src0Shape4 = src0Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto src1Shape4 = src1Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr bool src0IsColMajor = (Src0TileInfo::tileW == 1 && WBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND);
    constexpr bool src1IsColMajor = (Src1TileInfo::tileW == 1 && WBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND);
    using Src0PtoTile =
        typename std::conditional<src0IsColMajor, PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;
    using Src1PtoTile =
        typename std::conditional<src1IsColMajor, PtoTile<T2, pto::BLayout::ColMajor>, PtoTile<T2>>::type;
    auto dstTile = PtoTile<T0>(1, shape4).Data();
    auto src0Tile = Src0PtoTile(1, src0Shape4).Data();
    auto src1Tile = Src1PtoTile(1, src1Shape4).Data();
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < shape3; ++n3Index) {
                    auto dsttileOffset = GenTileOffset(dst, TileOffset4Dim(n0Index, n1Index, n2Index, n3Index));
                    auto src0tileOffset = GenTileOffset(src0, TileOffset4Dim(Src0TileInfo::tile0 == 1 ? 0 : n0Index, Src0TileInfo::tile1 == 1 ? 0 : n1Index,
                                                                Src0TileInfo::tile2 == 1 ? 0 : n2Index,
                                                                HBrcSide == TileOp::PenuBroadcastOperand::LEFT_OPERAND ? 0 : n3Index));
                    auto src1tileOffset = GenTileOffset(src1, TileOffset4Dim(Src1TileInfo::tile0 == 1 ? 0 : n0Index, Src1TileInfo::tile1 == 1 ? 0 : n1Index,
                                                                Src1TileInfo::tile2 == 1 ? 0 : n2Index,
                                                                HBrcSide == TileOp::PenuBroadcastOperand::RIGHT_OPERAND ? 0 : n3Index));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dsttileOffset * sizeof(typename T0::Type)));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0tileOffset * sizeof(typename T1::Type)));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1tileOffset * sizeof(typename T2::Type)));
                    BinaryRowExpandComputeImpl<op, PrecisionType, LastUse>(dstTile, src0Tile, src1Tile);
                }
            }
        }
    }
}
#endif
