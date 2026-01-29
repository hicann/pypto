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

template <BinaryOp op, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1) {
    if constexpr (op == BinaryOp::ADD) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TADD(dst, src0, src1);
        } else {
            pto::TROWEXPANDADD(dst, src0, src1);
        }
        return;
    }
    if constexpr (op == BinaryOp::SUB) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TSUB(dst, src0, src1);
        } else {
            pto::TROWEXPANDSUB(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MUL) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TMUL(dst, src0, src1);
        } else {
            pto::TROWEXPANDMUL(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::DIV) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TDIV(dst, src0, src1);
        } else {
            pto::TROWEXPANDDIV(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MAX) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TMAX(dst, src0, src1);
        } else {
            pto::TROWEXPANDMAX(dst, src0, src1);
        }
    }

    if constexpr (op == BinaryOp::MIN) {
        if constexpr (operand == TileOp::BroadcastOperand::NONE) {
            pto::TMIN(dst, src0, src1);
        } else {
            pto::TROWEXPANDMIN(dst, src0, src1);
        }
    }
}

template <BinaryOp op, TileOp::BroadcastOperand operand, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto src0Tile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        auto src1Tile = PtoTile<T2, pto::BLayout::RowMajor, true>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryComputeImpl<op, operand>(dstTile, src0Tile, src1Tile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                BinaryComputeImpl<op, operand>(dstTile.Data(), src0Tile.Data(), src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ADD TAdd
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::ADD, operand>(dst, src0, src1);
}

#define OP_TILE_OP_SUB TSub
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::SUB, operand>(dst, src0, src1);
}

#define OP_TILE_OP_MUL TMul
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MUL, operand>(dst, src0, src1);
}

#define OP_TILE_OP_DIV TDiv
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::DIV, operand>(dst, src0, src1);
}

#define OP_TILE_OP_MAX TMax
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MAX, operand>(dst, src0, src1);
}

#define OP_TILE_OP_MIN TMin
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1) {
    BinaryCompute<BinaryOp::MIN, operand>(dst, src0, src1);
}

#define OP_TILE_OP_Mod TMod
template <TileOp::BroadcastOperand operand = TileOp::BroadcastOperand::NONE, typename T0, typename T1, typename T2,  typename T3>
TILEOP void TMod(T0 dst, T1 src0, T2 src1, T3 tmp) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto src0Layout = src0.GetLayout();
    const auto src1Layout = src1.GetLayout();
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

    auto src1Stride0 = src1Layout.template GetStrideDim<0, expectSize>();
    auto src1Stride1 = src1Layout.template GetStrideDim<1, expectSize>();
    auto src1Stride2 = src1Layout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto src0TileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto src0TileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr auto src1TileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr auto src1TileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    __ubuf__ float *castBufAddr = reinterpret_cast<__ubuf__ float*>(tmp.GetAddr());
    using DstTileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using Src0TileDefine =
        pto::Tile<pto::TileType::Vec, typename T1::Type, src0TileH, src0TileW, pto::BLayout::RowMajor, -1, -1>;
    using Src1TileDefine =
        pto::Tile<pto::TileType::Vec, typename T2::Type, src1TileH, src1TileW, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    Src0TileDefine src0Tile(dstShape3, dstShape4);
    Src1TileDefine src1Tile(dstShape3, dstShape4);

    using DstType = typename T0::Type;
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto src0Offset = n0Index * src0Stride0 + n1Index * src0Stride1 + n2Index * src0Stride2;
                auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 + n2Index * src1Stride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dataTypeSize));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * dataTypeSize));
                pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * dataTypeSize));
                if constexpr (std::is_same_v<DstType, float>) {
                    DstTileDefine divTmpTile(dstShape3, dstShape4);
                    DstTileDefine castTmpTile(dstShape3, dstShape4);
                    pto::TASSIGN(divTmpTile, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW));
                    pto::TASSIGN(castTmpTile, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 2));
                    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
                        pto::TDIV(divTmpTile, src0Tile, src1Tile);
                    } else {
                        pto::TROWEXPANDDIV(divTmpTile, src0Tile, src1Tile);
                    }
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TCVT(castTmpTile, divTmpTile, pto::RoundMode::CAST_TRUNC);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
                        pto::TMUL(castTmpTile, castTmpTile, src1Tile);
                    } else {
                        pto::TROWEXPANDMUL(castTmpTile, castTmpTile, src1Tile);
                    }
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
                        pto::TSUB(dstTile, src0Tile, castTmpTile);
                    } else {
                        pto::TROWEXPANDSUB(dstTile, src0Tile, castTmpTile);
                    }
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                } else if constexpr (std::is_same_v<DstType, half> || std::is_same_v<DstType, bfloat16_t>) {
                    using Fp32TmpTileDefine =
                        pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    Fp32TmpTileDefine dstTileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine src0TileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine src1TileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine castTileTmp(dstShape3, dstShape4);
                    pto::TASSIGN(dstTileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW));
                    pto::TASSIGN(src0TileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 2));
                    pto::TASSIGN(src1TileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 3));
                    pto::TASSIGN(castTileTmp, reinterpret_cast<uint64_t>(castBufAddr + dstTileH * dstTileW * 4));
                    pto::TCVT(dstTileTmp, dstTile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(src0TileTmp, src0Tile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(src1TileTmp, src1Tile, pto::RoundMode::CAST_NONE);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
                        pto::TDIV(dstTileTmp, src0TileTmp, src1TileTmp);
                    } else {
                        pto::TROWEXPANDDIV(dstTileTmp, src0TileTmp, src1TileTmp);
                    }
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TCVT(castTileTmp, dstTileTmp, pto::RoundMode::CAST_TRUNC);
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
                        pto::TMUL(dstTileTmp, castTileTmp, src1TileTmp);
                    } else {
                        pto::TROWEXPANDMUL(dstTileTmp, castTileTmp, src1TileTmp);
                    }
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    if constexpr (operand == TileOp::BroadcastOperand::NONE) {
                        pto::TSUB(dstTileTmp, src0TileTmp, dstTileTmp);
                    } else {
                        pto::TROWEXPANDSUB(dstTileTmp, src0TileTmp, dstTileTmp);
                    }
                    #ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
                    #endif
                    pto::TCVT(dstTile, dstTileTmp, pto::RoundMode::CAST_NONE);
                }
            }
        }
    }
}
#endif