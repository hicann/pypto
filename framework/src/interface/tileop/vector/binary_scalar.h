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

    if constexpr (op == BinaryScalarOp::SUB) {
        pto::TADDS(dst, src0, -src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        pto::TMULS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        pto::TDIVS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MAX) {
        pto::TMAXS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MIN) {
        pto::TMINS(dst, src0, src1);
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
}

template <BinaryScalarOp op, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                BinaryScalarComputeImpl<op>(dstTile.Data(), src0Tile.Data(), src1);
            }
        }
    }
}
#define OP_TILE_OP_ADDS TAddS
template <typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::ADD>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::SUB>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MUL>(dst, src0, src1);
}

#define OP_TILE_OP_DIVS TDivS
template <typename Scalar, typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::DIV>(dst, src0, src1);
}

#define OP_TILE_OP_MAXS TMaxS
template <typename Scalar, typename T0, typename T1>
TILEOP void TMaxS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MAX>(dst, src0, src1);
}

#define OP_TILE_OP_MINS TMinS
template <typename Scalar, typename T0, typename T1>
TILEOP void TMinS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::MIN>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEANDS TBitwiseAndS
template <typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseAndS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::BITWISEAND>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEORS TBitwiseOrS
template <typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseOrS(T0 dst, T1 src0, Scalar src1) {
    BinaryScalarCompute<BinaryScalarOp::BITWISEOR>(dst, src0, src1);
}

#define OP_TILE_OP_MODS TModS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TModS(T0 dst, T1 src0, Scalar src1, T2 tmp) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto src0Layout = src0.GetLayout();
    constexpr auto dataTypeSize = sizeof(typename T0::Type);
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto src0Stride0 = src0Layout.template GetStrideDim<0, expectSize>();
    auto src0Stride1 = src0Layout.template GetStrideDim<1, expectSize>();
    auto src0Stride2 = src0Layout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto src0TileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto src0TileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    __ubuf__ float *castBufAddr = reinterpret_cast<__ubuf__ float*>(tmp.GetAddr());
    using DstTileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using Src0TileDefine =
        pto::Tile<pto::TileType::Vec, typename T1::Type, src0TileH, src0TileW, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    Src0TileDefine src0Tile(dstShape3, dstShape4);

    using DstType = typename T0::Type;
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto src0Offset = n0Index * src0Stride0 + n1Index * src0Stride1 + n2Index * src0Stride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dataTypeSize));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * dataTypeSize));

                if constexpr (std::is_same_v<DstType, float>) {
                    DstTileDefine divTmpTile(dstShape3, dstShape4);
                    DstTileDefine castTmpTile(dstShape3, dstShape4);
                    pto::TASSIGN(divTmpTile, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW));
                    pto::TASSIGN(castTmpTile, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 2));
                    pto::TDIVS(divTmpTile, src0Tile, src1);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TCVT(castTmpTile, divTmpTile, pto::RoundMode::CAST_TRUNC);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TMULS(castTmpTile, castTmpTile, src1);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TSUB(dstTile, src0Tile, castTmpTile);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                }  else if constexpr (std::is_same_v<DstType, half> || std::is_same_v<DstType, bfloat16_t>) {
                    float src1Tmp = static_cast<float>(src1);
                    using Fp32TmpTileDefine =
                        pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    Fp32TmpTileDefine dstTileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine src0TileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine castTileTmp(dstShape3, dstShape4);
                    pto::TASSIGN(dstTileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW));
                    pto::TASSIGN(src0TileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 2));
                    pto::TASSIGN(castTileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 3));
                    pto::TCVT(dstTileTmp, dstTile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(src0TileTmp, src0Tile, pto::RoundMode::CAST_NONE);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TDIVS(dstTileTmp, src0TileTmp, src1Tmp);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TCVT(castTileTmp, dstTileTmp, pto::RoundMode::CAST_TRUNC);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TMULS(dstTileTmp, castTileTmp, src1Tmp);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TSUB(dstTileTmp, src0TileTmp, dstTileTmp);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TCVT(dstTile, dstTileTmp, pto::RoundMode::CAST_NONE);
                }
            }
        }
    }
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
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
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