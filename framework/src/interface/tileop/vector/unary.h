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
 * \file vec_unary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY__H
#define TILEOP_TILE_OPERATOR_VEC_UNARY__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <UnaryOp op, typename T0, typename T1>
TILEOP void UnaryComputeImpl(T0 dst, T1 src) {
    if constexpr (op == UnaryOp::EXP) {
        pto::TEXP(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::RSQRT) {
        pto::TRSQRT(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::SQRT) {
        pto::TSQRT(dst, src);
        return;
    }
}

template <UnaryOp op, typename T0, typename T1>
TILEOP void UnaryCompute(T0 dst, T1 src) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        auto dstTile = PtoTile<T0>().Tile();
        auto srcTile = PtoTile<T1>().Tile();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        UnaryComputeImpl<op>(dstTile, srcTile);
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
    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, expectSize>();

    auto dstTile = DynPtoTile<T0>(dst).Tile();
    auto srcTile = DynPtoTile<T1>(src).Tile();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                UnaryComputeImpl<op>(dstTile, srcTile);
            }
        }
    }
}

template <typename T0, typename T1>
TILEOP void TExp(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::EXP>(dst, src);
}

template <typename T0, typename T1>
TILEOP void TRsqrt(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::RSQRT>(dst, src);
}

template <typename T0, typename T1>
TILEOP void TSqrt(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::SQRT>(dst, src);
}
#endif