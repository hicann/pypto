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

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    if constexpr (op == BinaryScalarOp::ADD) {
        pto::TADDS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        pto::TMULS(dst, src0, src1);
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        pto::TDIVS(dst, src0, src1);
    }
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, expectSize>();
    auto stride0 = dstLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<DIM_3RD, expectSize>();
    const auto srcLayout = src0.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, expectSize>();

    auto dstTile = DynPtoTile<T0>(dst).Tile();
    auto src0Tile = DynPtoTile<T1>(src0).Tile();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto src0TypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + srcOffset * src0TypeSize));
                BinaryScalarComputeImpl<op>(dstTile, src0Tile, src1);
            }
        }
    }
}

template <typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::ADD>(dst, src0, src1);
}

template <typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MUL>(dst, src0, src1);
}

template <typename Scalar, typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::DIV>(dst, src0, src1);
}
#endif