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
 * \file floor_div.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_FLOOR_DIV__H
#define TILEOP_TILE_OPERATOR_FLOOR_DIV__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_FLOORDIV TFloorDiv
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TFloorDiv(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    static_assert(std::is_same_v<typename T1::Type, int32_t>);

    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    auto offset =
                        n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 + n3Index * dstStride3;
#ifdef __DAV_V220
                    using FloatTileDefine =
                        pto::Tile<pto::TileType::Vec, float, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using IntTileDefine =
                        pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using MaskTileDefine =
                        pto::Tile<pto::TileType::Vec, uint8_t, 1, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

                    FloatTileDefine tmp0Fp32Tile(1, dstShape4);
                    FloatTileDefine tmp1Fp32Tile(1, dstShape4);
                    FloatTileDefine tmp2Fp32Tile(1, dstShape4);
                    IntTileDefine tmp0I32Tile(1, dstShape4);
                    IntTileDefine tmp2I32Tile(1, dstShape4);
                    IntTileDefine tmp3I32Tile(1, dstShape4);
                    IntTileDefine tmp4I32Tile(1, dstShape4);
                    IntTileDefine tmp5I32Tile(1, dstShape4);
                    MaskTileDefine tmp1MaskTile(1, dstShape4);
                    IntTileDefine src0Tile(1, dstShape4);
                    IntTileDefine src1Tile(1, dstShape4);
                    IntTileDefine dstTile(1, dstShape4);

                    pto::TASSIGN(tmp0Fp32Tile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1Fp32Tile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(tmp2Fp32Tile, (uint64_t)(tmp.GetAddr() + 2 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp0I32Tile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp2I32Tile, (uint64_t)(tmp.GetAddr() + 2 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp3I32Tile, (uint64_t)(tmp.GetAddr() + 3 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp4I32Tile, (uint64_t)(tmp.GetAddr() + 4 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp5I32Tile, (uint64_t)(tmp.GetAddr() + 5 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));

                    // Step 1: approximate quotient by float32 division, then floor and cast to int32.
                    // q = floor(float32(x1) / float32(x2))
                    pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(tmp1Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TDIV(tmp0Fp32Tile, tmp0Fp32Tile, tmp1Fp32Tile);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    pipe_barrier(PIPE_V);

                    // Step 2: compute exact int32 remainder: r = x1 - q * x2.
                    pto::TMUL(tmp0I32Tile, dstTile, src1Tile);
                    pipe_barrier(PIPE_V);
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
                    pipe_barrier(PIPE_V);

                    // Step 3: refine q with floor(float32(r) / float32(x2)).
                    // The first float32 quotient can be off by a small amount; this correction handles it.
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                    pto::TDIV(tmp2Fp32Tile, tmp2Fp32Tile, tmp1Fp32Tile);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    pipe_barrier(PIPE_V);

                    // Step 4: apply the remainder-based correction.
                    // q_corrected = q + correction
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    pipe_barrier(PIPE_V);

                    // Step 5: recompute r2 with q_corrected.
                    // A valid floor-div remainder must satisfy 0 <= r2 * sign(x2) < abs(x2).
                    pto::TMUL(tmp0I32Tile, dstTile, src1Tile);
                    pipe_barrier(PIPE_V);
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
                    pipe_barrier(PIPE_V);

                    // Step 6: final +/-1 correction.
                    // Use float32 only to produce comparison masks; keep abs(x2) and diff in int32.
                    pto::TCVT(tmp2Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
                    pipe_barrier(PIPE_V);

                    // Normalize remainder: r2_adj = (x2 < 0) ? -r2 : r2.
                    pto::TMULS(tmp2I32Tile, tmp0I32Tile, -1);
                    pipe_barrier(PIPE_V);
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp3I32Tile); // r2 * sign(x2)
                    pipe_barrier(PIPE_V);

                    // Compute abs(x2) in int32 using the same sign mask.
                    pto::TMULS(tmp2I32Tile, src1Tile, -1);
                    pipe_barrier(PIPE_V);
                    pto::TSEL(tmp2I32Tile, tmp1MaskTile, tmp2I32Tile, src1Tile, tmp3I32Tile); // abs(x2)
                    pipe_barrier(PIPE_V);

                    pto::TSUB(tmp3I32Tile, tmp0I32Tile, tmp2I32Tile); // diff = r2_adj - abs(x2)
                    pipe_barrier(PIPE_V);

                    // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
                    // tensor-scalar select path, whose first lane can be unstable across calls.
                    pto::TSUB(tmp4I32Tile, tmp2I32Tile, tmp2I32Tile); // zero
                    pipe_barrier(PIPE_V);

                    // If r2_adj < 0, q_corrected is too large: final_corr = -1; otherwise 0.
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
                    pipe_barrier(PIPE_V);
                    pto::TADDS(tmp2I32Tile, tmp4I32Tile, -1);
                    pipe_barrier(PIPE_V);
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp4I32Tile, tmp5I32Tile);
                    pipe_barrier(PIPE_V);

                    // If diff >= 0, q_corrected is too small: final_corr = 1.
                    pto::TCVT(tmp2Fp32Tile, tmp3I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
                    pipe_barrier(PIPE_V);
                    pto::TADDS(tmp2I32Tile, tmp4I32Tile, 1);
                    pipe_barrier(PIPE_V);
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp5I32Tile);
                    pipe_barrier(PIPE_V);

                    // res = q_corrected + final_corr
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    pipe_barrier(PIPE_V);
#else
                    using DataTileDefine =
                        pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using MaskTileDefine =
                        pto::Tile<pto::TileType::Vec, uint8_t, 1, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

                    DataTileDefine src0Tile(1, dstShape4);
                    DataTileDefine src1Tile(1, dstShape4);
                    DataTileDefine dstTile(1, dstShape4);
                    DataTileDefine tmp0DataTile(1, dstShape4);
                    DataTileDefine tmp1DataTile(1, dstShape4);
                    DataTileDefine tmp2DataTile(1, dstShape4);
                    DataTileDefine tmp3DataTile(1, dstShape4);

                    MaskTileDefine tmp0MaskTile(1, dstShape4);
                    MaskTileDefine tmp1MaskTile(1, dstShape4);

                    constexpr int32_t pos = 0x7FFF7F7F, neg = 0x80008080;

                    pto::TASSIGN(tmp0DataTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1DataTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(tmp2DataTile, (uint64_t)(tmp.GetAddr() + 2 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp3DataTile, (uint64_t)(tmp.GetAddr() + 3 * tileW * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));

                    // Reuse the same tmp as packed mask storage
                    pto::TASSIGN(tmp0MaskTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));

                    // Deal dividend is zero
                    pto::TCMPS(tmp0MaskTile, src0Tile, 0, pto::CmpMode::LT);
                    pto::TSELS(tmp1DataTile, tmp0MaskTile, tmp1DataTile, tmp1DataTile, pos);
                    pto::TCMPS(tmp0MaskTile, src0Tile, 0, pto::CmpMode::GE);
                    pto::TSELS(tmp1DataTile, tmp0MaskTile, tmp1DataTile, tmp1DataTile, neg);
                    pto::TCMPS(tmp0MaskTile, src1Tile, 0, pto::CmpMode::NE);
                    pto::TSEL(tmp2DataTile, tmp0MaskTile, src0Tile, tmp1DataTile, tmp1DataTile);
                    pto::TSELS(tmp3DataTile, tmp0MaskTile, src1Tile, tmp1DataTile, 1);

                    /*
                     * After zero-divisor handling:
                     * sign_differ = (src0 < 0) != (src1 < 0)
                     * quot = src0 / src1
                     * rem = src0 - quot * src1
                     * dst = (sign_differ && rem != 0) ? quot - 1 : quot
                     */
                    pto::TCMPS(tmp0MaskTile, tmp2DataTile, 0, pto::CmpMode::LT);
                    pto::TCMPS(tmp1MaskTile, tmp3DataTile, 0, pto::CmpMode::LT);
                    pto::TXOR(tmp0MaskTile, tmp0MaskTile, tmp1MaskTile, dstTile); // packed mask of sign_differ
                    pto::TDIV(dstTile, tmp2DataTile, tmp3DataTile);                       // quot
                    pto::TMUL(tmp1DataTile, tmp3DataTile, dstTile);
                    pto::TSUB(tmp2DataTile, tmp2DataTile, tmp1DataTile); // rem

                    pto::TCMPS(tmp1MaskTile, tmp2DataTile, 0, pto::CmpMode::NE);
                    pto::TAND(tmp0MaskTile, tmp0MaskTile, tmp1MaskTile);
                    pto::TADDS(tmp2DataTile, dstTile, -1);
                    pto::TSEL(dstTile, tmp0MaskTile, tmp2DataTile, dstTile, tmp1DataTile);
#endif
                }
            }
        }
    }
}

#define OP_TILE_OP_FLOORDIVS TFloorDivS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TFloorDivS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    static_assert(std::is_same_v<typename T1::Type, int32_t>);

    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    auto offset =
                        n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 + n3Index * dstStride3;
#ifdef __DAV_V220
                    using IntTileDefine =
                        pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using FloatTileDefine =
                        pto::Tile<pto::TileType::Vec, float, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using MaskTileDefine =
                        pto::Tile<pto::TileType::Vec, uint8_t, 1, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

                    IntTileDefine src0Tile(1, dstShape4);
                    IntTileDefine dstTile(1, dstShape4);
                    IntTileDefine tmp0I32Tile(1, dstShape4);
                    IntTileDefine tmp2I32Tile(1, dstShape4);
                    IntTileDefine tmp3I32Tile(1, dstShape4);
                    IntTileDefine tmp4I32Tile(1, dstShape4);
                    IntTileDefine tmp5I32Tile(1, dstShape4);
                    FloatTileDefine tmp0Fp32Tile(1, dstShape4);
                    FloatTileDefine tmp2Fp32Tile(1, dstShape4);
                    MaskTileDefine tmp1MaskTile(1, dstShape4);

                    pto::TASSIGN(tmp0I32Tile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp2I32Tile, (uint64_t)(tmp.GetAddr() + 2 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp3I32Tile, (uint64_t)(tmp.GetAddr() + 3 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp4I32Tile, (uint64_t)(tmp.GetAddr() + 4 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp5I32Tile, (uint64_t)(tmp.GetAddr() + 5 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp0Fp32Tile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp2Fp32Tile, (uint64_t)(tmp.GetAddr() + 2 * tileW * dstTypeSize));
                    pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));

                    // Step 1: approximate quotient by float32 division, then floor and cast to int32.
                    // q = floor(float32(x1) / float32(x2))
                    pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TDIVS(tmp0Fp32Tile, tmp0Fp32Tile, static_cast<float>(src1));
                    pipe_barrier(PIPE_V);
                    pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    pipe_barrier(PIPE_V);

                    // Step 2: compute exact int32 remainder: r = x1 - q * x2.
                    pto::TMULS(tmp0I32Tile, dstTile, src1);
                    pipe_barrier(PIPE_V);
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
                    pipe_barrier(PIPE_V);

                    // Step 3: refine q with floor(float32(r) / float32(x2)).
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                    pto::TDIVS(tmp2Fp32Tile, tmp2Fp32Tile, static_cast<float>(src1));
                    pipe_barrier(PIPE_V);
                    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    pipe_barrier(PIPE_V);

                    // Step 4: apply the remainder-based correction.
                    // q_corrected = q + correction
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    pipe_barrier(PIPE_V);

                    // Step 5: recompute r2 with q_corrected.
                    pto::TMULS(tmp0I32Tile, dstTile, src1);
                    pipe_barrier(PIPE_V);
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
                    pipe_barrier(PIPE_V);

                    // Step 6: final +/-1 correction. A valid floor-div remainder must satisfy
                    // 0 <= r2 * sign(x2) < abs(x2).
                    auto absSrc1 = src1;
                    if (src1 < 0) {
                        pto::TMULS(tmp0I32Tile, tmp0I32Tile, -1); // r2_adj = -r2
                        pipe_barrier(PIPE_V);
                        absSrc1 = -src1;
                    }

                    pto::TADDS(tmp3I32Tile, tmp0I32Tile, -absSrc1); // diff = r2_adj - abs(x2)
                    pipe_barrier(PIPE_V);

                    // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
                    // tensor-scalar select path, whose first lane can be unstable across calls.
                    pto::TSUB(tmp4I32Tile, tmp0I32Tile, tmp0I32Tile); // zero
                    pipe_barrier(PIPE_V);

                    // If r2_adj < 0, q_corrected is too large: final_corr = -1; otherwise 0.
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
                    pipe_barrier(PIPE_V);
                    pto::TADDS(tmp2I32Tile, tmp4I32Tile, -1);
                    pipe_barrier(PIPE_V);
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp4I32Tile, tmp5I32Tile);
                    pipe_barrier(PIPE_V);

                    // If diff >= 0, q_corrected is too small: final_corr = 1.
                    pto::TCVT(tmp2Fp32Tile, tmp3I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
                    pipe_barrier(PIPE_V);
                    pto::TADDS(tmp2I32Tile, tmp4I32Tile, 1);
                    pipe_barrier(PIPE_V);
                    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp5I32Tile);
                    pipe_barrier(PIPE_V);

                    // res = q_corrected + final_corr
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    pipe_barrier(PIPE_V);
#else
                    using DataTileDefine =
                        pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using MaskTileDefine =
                        pto::Tile<pto::TileType::Vec, uint8_t, 1, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

                    DataTileDefine src0Tile(1, dstShape4);
                    DataTileDefine dstTile(1, dstShape4);
                    DataTileDefine tmp0DataTile(1, dstShape4);
                    DataTileDefine tmp1DataTile(1, dstShape4);
                    DataTileDefine tmp2DataTile(1, dstShape4);

                    MaskTileDefine tmp0MaskTile(1, dstShape4);
                    MaskTileDefine tmp1MaskTile(1, dstShape4);

                    pto::TASSIGN(tmp0DataTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1DataTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(tmp2DataTile, (uint64_t)(tmp.GetAddr() + 2 * tileW * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));

                    // Reuse the same tmp as packed mask storage
                    pto::TASSIGN(tmp0MaskTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));

                    if (src1 == 0) {
                        constexpr int32_t pos = 0x7FFF7F7F, neg = 0x80008080;
                        pto::TCMPS(tmp0MaskTile, src0Tile, 0, pto::CmpMode::LT);
                        pto::TSELS(dstTile, tmp0MaskTile, dstTile, tmp1DataTile, pos);
                        pto::TCMPS(tmp0MaskTile, src0Tile, 0, pto::CmpMode::GE);
                        pto::TSELS(dstTile, tmp0MaskTile, dstTile, tmp1DataTile, neg);
                    } else {
                        uint8_t src1Mask = 0;
                        if (src1 < 0) {
                            src1Mask = 0xff;
                        }
                        pto::TCMPS(tmp0MaskTile, src0Tile, 0, pto::CmpMode::LT);
                        pto::TXORS(tmp1MaskTile, tmp0MaskTile, src1Mask, dstTile); // packed mask of sign_differ
                        pto::TDIVS(dstTile, src0Tile, src1);                       // quot
                        pto::TMULS(tmp0DataTile, dstTile, -src1);
                        pto::TADD(tmp2DataTile, tmp0DataTile, src0Tile); // rem

                        pto::TCMPS(tmp0MaskTile, tmp2DataTile, 0, pto::CmpMode::NE);
                        pto::TAND(tmp0MaskTile, tmp1MaskTile, tmp0MaskTile);
                        pto::TADDS(tmp2DataTile, dstTile, -1);
                        pto::TSEL(dstTile, tmp0MaskTile, tmp2DataTile, dstTile, tmp1DataTile);
                    }
#endif
                }
            }
        }
    }
}

#endif
