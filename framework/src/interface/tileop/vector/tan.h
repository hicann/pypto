/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tan.h
 * \brief tan operator: compute tangent function using polynomial approximation
 */

#ifndef TILEOP_TILE_OPERATOR_TAN__H
#define TILEOP_TILE_OPERATOR_TAN__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename LastUse, typename DstTile, typename SrcTile, typename TmpTile, typename TmpInt32Tile, typename Tmp2Tile, typename Tmp3Tile, typename Tmp4Tile, typename Tmp5Tile, typename Tmp6Tile>
TILEOP void TanImpl(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile, TmpInt32Tile tmpInt32Tile, Tmp2Tile tmp2Tile, Tmp3Tile tmp3Tile, Tmp4Tile tmp4Tile, Tmp5Tile tmp5Tile, Tmp6Tile tmp6Tile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    constexpr float INV_PI = 0.3183098733425140380859375f;
    constexpr float PI_HIGH = 3.140625f;
    constexpr float PI_CORR_1 = 0.0009670257568359375f;
    constexpr float PI_CORR_2 = 6.2771141529083251953125e-7f;
    constexpr float PI_CORR_3 = 1.21644916362129151821136474609375e-10f;
    constexpr float PI_CORR_4 = -1.0291767438275201129727065563201904296875e-13f;
    constexpr float HALF_PI = 1.57079637050628662109375f;
    constexpr float HALF_PI_NEG = -1.57079637050628662109375f;
    constexpr float EPS = 0.00000004371139000189375f;
    constexpr float EPS_NEG = -0.00000004371139000189375f;
    constexpr float POLY_R0 = 0.0698520831551998762793f;
    constexpr float POLY_R1 = -6.8711573651634203789f;
    constexpr float POLY_R2 = 61.20362572811089435388f;
    constexpr float POLY_R3 = -24.8048928861126769186219f;

    pto::TMULS(tmpTile, srcTile, INV_PI);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(tmpInt32Tile, tmpTile, pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(tmpTile, tmpInt32Tile, pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmp3Tile, tmpTile, PI_HIGH);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, srcTile, tmp3Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmp3Tile, tmpTile, PI_CORR_1);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(tmp4Tile, tmp2Tile, HALF_PI);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmp3Tile, tmpTile, PI_CORR_2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmp5Tile, tmpTile, PI_CORR_3);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp5Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmp6Tile, tmpTile, PI_CORR_4);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp6Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMUL(dstTile, tmp2Tile, tmp2Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMULS(tmpTile, dstTile, POLY_R0);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(tmpTile, tmpTile, POLY_R1);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMUL(tmpTile, tmpTile, dstTile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(tmpTile, tmpTile, POLY_R2);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMUL(tmpTile, tmpTile, tmp2Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(dstTile, dstTile, POLY_R3);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp4Tile, tmp4Tile, tmp3Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(tmp4Tile, tmp4Tile, EPS_NEG);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp4Tile, tmp4Tile, tmp5Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp4Tile, tmp4Tile, tmp6Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMUL(dstTile, dstTile, tmp4Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(tmp2Tile, tmp2Tile, HALF_PI_NEG);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TADDS(tmp2Tile, tmp2Tile, EPS);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp5Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TSUB(tmp2Tile, tmp2Tile, tmp6Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TMUL(dstTile, dstTile, tmp2Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TDIV(dstTile, tmpTile, dstTile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

template <typename LastUse, typename DstTile, typename SrcTile, typename TmpTile, typename TmpFp32Tile, typename TmpInt32Tile, typename Tmp2Tile, typename Tmp3Tile, typename Tmp4Tile, typename Tmp5Tile, typename Tmp6Tile>
TILEOP void TanHalfImpl(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile, TmpFp32Tile tmpFp32Tile, TmpInt32Tile tmpInt32Tile, Tmp2Tile tmp2Tile, Tmp3Tile tmp3Tile, Tmp4Tile tmp4Tile, Tmp5Tile tmp5Tile, Tmp6Tile tmp6Tile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    pto::TCVT(tmpFp32Tile, srcTile, pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    TanImpl<LastUse, TmpFp32Tile, TmpFp32Tile, TmpFp32Tile, TmpInt32Tile, Tmp2Tile, Tmp3Tile, Tmp4Tile, Tmp5Tile, Tmp6Tile>(tmpFp32Tile, tmpFp32Tile, tmpFp32Tile, tmpInt32Tile, tmp2Tile, tmp3Tile, tmp4Tile, tmp5Tile, tmp6Tile);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
    pto::TCVT(dstTile, tmpFp32Tile, pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

#define OP_TILE_OP_TAN Ttan
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T2>
TILEOP void Ttan(T0 dst, T1 src, T2 tmp)
{
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();

    constexpr auto ALIGN32HALF = 16;
    constexpr auto ALIGN32FLOAT = 8;
    constexpr auto tmpTileW = (srcTileW + ALIGN32HALF - 1) / ALIGN32HALF * ALIGN32HALF;
    constexpr auto tmpTileW32Bit = (srcTileW + ALIGN32FLOAT - 1) / ALIGN32FLOAT * ALIGN32FLOAT;

    using DstTile =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile =
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpFp32Tile =
        pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using TmpInt32Tile =
        pto::Tile<pto::TileType::Vec, int32_t, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp2Tile =
        pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp3Tile =
        pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp4Tile =
        pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp5Tile =
        pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp6Tile =
        pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;

    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcShape3, srcShape4);
    TmpFp32Tile tmpFp32Tile(srcShape3, srcShape4);
    TmpInt32Tile tmpInt32Tile(srcShape3, srcShape4);
    Tmp2Tile tmp2Tile(srcShape3, srcShape4);
    Tmp3Tile tmp3Tile(srcShape3, srcShape4);
    Tmp4Tile tmp4Tile(srcShape3, srcShape4);
    Tmp5Tile tmp5Tile(srcShape3, srcShape4);
    Tmp6Tile tmp6Tile(srcShape3, srcShape4);

    constexpr auto tmpFp32TileSize = srcTileH * tmpTileW32Bit * sizeof(float);
    pto::TASSIGN(tmpFp32Tile, (uint64_t)(tmp.GetAddr()));
    pto::TASSIGN(tmpInt32Tile, (uint64_t)(tmp.GetAddr() + tmpFp32TileSize));
    pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + 2 * tmpFp32TileSize));
    pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + 3 * tmpFp32TileSize));
    pto::TASSIGN(tmp4Tile, (uint64_t)(tmp.GetAddr() + 4 * tmpFp32TileSize));
    pto::TASSIGN(tmp5Tile, (uint64_t)(tmp.GetAddr() + 5 * tmpFp32TileSize));
    pto::TASSIGN(tmp6Tile, (uint64_t)(tmp.GetAddr() + 6 * tmpFp32TileSize));

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

                if constexpr (std::is_same<typename T1::Type, float>::value) {
                    TanImpl<LastUse, DstTile, SrcTile, TmpFp32Tile, TmpInt32Tile, Tmp2Tile, Tmp3Tile, Tmp4Tile, Tmp5Tile, Tmp6Tile>(dstTile, srcTile, tmpFp32Tile, tmpInt32Tile, tmp2Tile, tmp3Tile, tmp4Tile, tmp5Tile, tmp6Tile);
                } else if constexpr (std::is_same<typename T1::Type, half>::value ||
                                     std::is_same<typename T1::Type, bfloat16_t>::value) {
                    TanHalfImpl<LastUse, DstTile, SrcTile, TmpFp32Tile, TmpFp32Tile, TmpInt32Tile, Tmp2Tile, Tmp3Tile, Tmp4Tile, Tmp5Tile, Tmp6Tile>(dstTile, srcTile, tmpFp32Tile, tmpFp32Tile, tmpInt32Tile, tmp2Tile, tmp3Tile, tmp4Tile, tmp5Tile, tmp6Tile);
                }
            }
        }
    }
}

#endif
