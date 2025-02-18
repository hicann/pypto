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
 * \file binary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY__H
#define TILEOP_TILE_OPERATOR_BINARY__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryOp op, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    if constexpr (op == BinaryOp::ADD) {
        pto::TADD(dst, src0, src1);
        return;
    }
    if constexpr (op == BinaryOp::SUB) {
        pto::TSUB(dst, src0, src1);
    }

    if constexpr (op == BinaryOp::MUL) {
        pto::TMUL(dst, src0, src1);
    }

    if constexpr (op == BinaryOp::DIV) {
        pto::TDIV(dst, src0, src1);
    }

    if constexpr (op == BinaryOp::MAX) {
        pto::TMAX(dst, src0, src1);
    }

    if constexpr (op == BinaryOp::MIN) {
        pto::TMIN(dst, src0, src1);
    }
}

template <BinaryOp op, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0>().Tile();
        auto src0Tile = PtoTile<T1>().Tile();
        auto src1Tile = PtoTile<T2>().Tile();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryComputeImpl<op>(dstTile, src0Tile, src1Tile);
        return;
    }
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, expectSize>();
    auto stride0 = dstLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<DIM_3RD, expectSize>();
    const auto src0Layout = src0.GetLayout();
    auto src0Stride0 = src0Layout.template GetStrideDim<DIM_1ST, expectSize>();
    auto src0Stride1 = src0Layout.template GetStrideDim<DIM_2ND, expectSize>();
    auto src0Stride2 = src0Layout.template GetStrideDim<DIM_3RD, expectSize>();
    const auto src1Layout = src1.GetLayout();
    auto src1Stride0 = src1Layout.template GetStrideDim<DIM_1ST, expectSize>();
    auto src1Stride1 = src1Layout.template GetStrideDim<DIM_2ND, expectSize>();
    auto src1Stride2 = src1Layout.template GetStrideDim<DIM_3RD, expectSize>();

    auto dstTile = DynPtoTile<T0>(dst).Tile();
    auto src0Tile = DynPtoTile<T1>(src0).Tile();
    auto src1Tile = DynPtoTile<T2>(src1).Tile();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto src0TypeSize = sizeof(typename T1::Type);
    constexpr auto src1TypeSize = sizeof(typename T2::Type);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2;
                auto src0Offset = n0Index * src0Stride0 + n1Index * src0Stride1 + n2Index * src0Stride2;
                auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 + n2Index * src1Stride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize));
                pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize));
                BinaryComputeImpl<op>(dstTile, src0Tile, src1Tile);
            }
        }
    }
}

template <typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::ADD>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::SUB>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MUL>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::DIV>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MAX>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MIN>(dst, src0, src1);
}
#endif