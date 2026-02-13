/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file binary_scalar.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#define TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryScalarOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == BinaryScalarOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::SUB) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, -src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEAND) {
        pto::TANDS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEOR) {
        pto::TORS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MOD) {
        pto::TFMODS(dst, src0, src1);
        return;
    }
}

template <BinaryScalarOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                BinaryScalarComputeImpl<op, LastUse>(dstTile.Data(), src0Tile.Data(), src1);
            }
        }
    }
}
#define OP_TILE_OP_ADDS TAddS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::ADD, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::SUB, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MUL, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIVS TDivS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::DIV, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAXS TMaxS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMaxS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MAX, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MINS TMinS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMinS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MIN, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEANDS TBitwiseAndS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseAndS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::BITWISEAND, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEORS TBitwiseOrS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseOrS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::BITWISEOR, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MODS TModS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TModS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MOD, LastUse>(dst, src0, src1);
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpComputeImpl(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    if constexpr (op == BinaryScalarOp::BITWISEXOR) {
        pto::TXORS(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpCompute(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto tmpTile = PtoTile<T2>(tmp);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                BinaryScalarTmpComputeImpl<op>(dstTile.Data(), src0Tile.Data(), src1, tmpTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXORS TBitwiseXorS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TBitwiseXorS(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    BinaryScalarTmpCompute<BinaryScalarOp::BITWISEXOR>(dst, src0, src1, tmp);
}
#endif