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
 * \brief Binary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_FLOOR_DIV_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_FLOOR_DIV_H

#include "utils/sync.h"
#include "basic.h"

template <auto pos, auto neg, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7>
TILEOP void IntFloorDiv(T0 dst, T1 src0, T2 src1, T3 tmp0, T4 tmp1, T5 tmp2, T6 tmp3, T7 tmp4)
{
    // MaskTile: tmp3, tmp4
    // DataTile: tmp0-tmp2
    // reuse tmp address: tmp2=tmp4

    // Deal dividend is zero
    pto::TCMPS(tmp3, src0, 0, pto::CmpMode::LT);
    pto::TSELS(tmp2, tmp3, tmp2, tmp2, pos);
    pto::TCMPS(tmp3, src0, 0, pto::CmpMode::GE);
    pto::TSELS(tmp2, tmp3, tmp2, tmp2, neg);
    pto::TCMPS(tmp3, src1, 0, pto::CmpMode::NE);
    pto::TMOV(tmp0, src0);
    pto::TSEL(tmp0, tmp3, tmp0, tmp2, tmp2);
    pto::TSELS(tmp1, tmp3, src1, tmp2, 1);

    /*
     * After zero-divisor handling:
     * sign_differ = (src0 < 0) != (src1 < 0)
     * quot = src0 / src1
     * rem = src0 - quot * src1
     * dst = (sign_differ && rem != 0) ? quot - 1 : quot
     */
    pto::TCMPS(tmp3, tmp0, 0, pto::CmpMode::LT);
    pto::TCMPS(tmp4, tmp1, 0, pto::CmpMode::LT);
    pto::TXOR(tmp3, tmp3, tmp4, tmp4);

    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(dst, tmp0, tmp1);

    // A5 TREM does not use the tmp argument.
    pto::TREM(tmp0, tmp0, tmp1, tmp0);

    pto::TCMPS(tmp4, tmp0, 0, pto::CmpMode::NE);
    pto::TAND(tmp3, tmp3, tmp4);
    pto::TADDS(tmp0, dst, -1);
    pto::TSEL(dst, tmp3, tmp0, dst, tmp2);
}

template <typename TmpTensor>
TILEOP uint64_t FloorDivTmpAddr(TmpTensor tmp, size_t byteOffset, size_t tileShapeSize, size_t tileIndex,
                                size_t elementSize)
{
    return (uint64_t)(tmp.GetAddr() + byteOffset + tileIndex * tileShapeSize * elementSize);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivFp32TmpCompute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                   size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, tmp1Tile);
    SyncV();
    pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_RINT);
    SyncV();
}

template <typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivFloatCompute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile)
{
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, src1Tile);
    SyncV();
    pto::TCVT(dstTile, dstTile, pto::RoundMode::CAST_FLOOR);
    SyncV();
}

#ifdef __DAV_V220
template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivV220Int8Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                    size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp3Tile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();
    pto::TCVT(tmp0Tile, tmp2Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::ON);
    SyncV();
}
template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivV220Int32Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                     size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;

    Fp32TileDefine tmp0Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Fp32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp0I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp2I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp3I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp4I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp5I32Tile(dstShape3, dstShape4);
    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);

    pto::TASSIGN(tmp0Fp32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1Fp32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2Fp32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp0I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp2I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp3I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp4I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(int32_t)));
    pto::TASSIGN(tmp5I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 5, sizeof(int32_t)));
    pto::TASSIGN(tmp1MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));

    // Step 1: approximate quotient by float32 division, then floor and cast to int32.
    // q = floor(float32(x1) / float32(x2))
    pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCVT(tmp1Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Fp32Tile, tmp0Fp32Tile, tmp1Fp32Tile);
    SyncV();
    pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // Step 2: compute exact int32 remainder: r = x1 - q * x2.
    pto::TMUL(tmp0I32Tile, dstTile, src1Tile);
    SyncV();
    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
    SyncV();

    // Step 3: refine q with floor(float32(r) / float32(x2)).
    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Fp32Tile, tmp2Fp32Tile, tmp1Fp32Tile);
    SyncV();
    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // Step 4: apply the remainder-based correction.
    // q_corrected = q + correction
    pto::TADD(dstTile, dstTile, tmp0I32Tile);
    SyncV();

    // Step 5: recompute r2 with q_corrected.
    // A valid floor-div remainder must satisfy 0 <= r2 * sign(x2) < abs(x2).
    pto::TMUL(tmp0I32Tile, dstTile, src1Tile);
    SyncV();
    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
    SyncV();

    // Step 6: final +/-1 correction.
    // Use float32 only to produce comparison masks; keep abs(x2) and diff in int32.
    pto::TCVT(tmp2Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();

    // Normalize remainder: r2_adj = (x2 < 0) ? -r2 : r2.
    pto::TMULS(tmp2I32Tile, tmp0I32Tile, -1);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp3I32Tile);
    SyncV();

    // Compute abs(x2) in int32 using the same sign mask.
    pto::TMULS(tmp2I32Tile, src1Tile, -1);
    SyncV();
    pto::TSEL(tmp2I32Tile, tmp1MaskTile, tmp2I32Tile, src1Tile, tmp3I32Tile);
    SyncV();

    pto::TSUB(tmp3I32Tile, tmp0I32Tile, tmp2I32Tile); // diff = r2_adj - abs(x2)
    SyncV();

    // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
    // tensor-scalar select path, whose first lane can be unstable across calls.
    pto::TSUB(tmp4I32Tile, tmp2I32Tile, tmp2I32Tile); // zero
    SyncV();

    // If r2_adj < 0, q_corrected is too large: final_corr = -1; otherwise 0.
    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();
    pto::TADDS(tmp2I32Tile, tmp4I32Tile, -1);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp4I32Tile, tmp5I32Tile);
    SyncV();

    // If diff >= 0, q_corrected is too small: final_corr = 1.
    pto::TCVT(tmp2Fp32Tile, tmp3I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
    SyncV();
    pto::TADDS(tmp2I32Tile, tmp4I32Tile, 1);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp5I32Tile);
    SyncV();

    // res = q_corrected + final_corr
    pto::TADD(dstTile, dstTile, tmp0I32Tile);
    SyncV();
}
#else
template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivNonV220Uint8Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                        size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Int16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    Int16TileDefine tmp2Tile(dstShape3, dstShape4);
    Int16TileDefine tmp3Tile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);
    pto::TCVT(dstTile, tmp2Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::ON);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivNonV220Int8Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                       size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Uint8TileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefine tmp0DataTile(dstShape3, dstShape4);
    HalfTileDefine tmp1DataTile(dstShape3, dstShape4);
    Uint8TileDefine tmp2MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));

    pto::TCVT(tmp0DataTile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1DataTile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCMPS(tmp2MaskTile, tmp1DataTile, 0, pto::CmpMode::NE);
    pto::TSELS(tmp0DataTile, tmp2MaskTile, tmp0DataTile, tmp0DataTile, 0);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0DataTile, tmp0DataTile, tmp1DataTile);
    pto::TCVT(dstTile, tmp0DataTile, pto::RoundMode::CAST_FLOOR);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivNonV220Int32Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                        size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Uint8TileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;
    Int32TileDefine tmp0DataTile(dstShape3, dstShape4);
    Int32TileDefine tmp1DataTile(dstShape3, dstShape4);
    Int32TileDefine tmp2DataTile(dstShape3, dstShape4);
    Uint8TileDefine tmp3MaskTile(dstShape3, dstShape4);
    Uint8TileDefine tmp4MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp4MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));

    IntFloorDiv<(int32_t)0x7FFF7F7F, (int32_t)0x80008080>(dstTile, src0Tile, src1Tile, tmp0DataTile, tmp1DataTile,
                                                          tmp2DataTile, tmp3MaskTile, tmp4MaskTile);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename SrcTile, typename DstTile>
TILEOP void FloorDivNonV220Int64Compute(DstTile dstTile, SrcTile src0Tile, SrcTile src1Tile, T3 tmp, size_t offset,
                                        size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Int64TileDefine = pto::Tile<pto::TileType::Vec, int64_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 8 * tileW, pto::BLayout::RowMajor, -1, -1>;
    Int64TileDefine tmp0DataTile(dstShape3, dstShape4);
    Int64TileDefine tmp1DataTile(dstShape3, dstShape4);
    Int64TileDefine tmp2DataTile(dstShape3, dstShape4);
    MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
    MaskTileDefine tmp4MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int64_t)));
    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int64_t)));
    pto::TASSIGN(tmp2DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int64_t)));
    pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int64_t)));
    pto::TASSIGN(tmp4MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int64_t)));

    IntFloorDiv<(int64_t)0x7FFFFFFFFFFFFFFF, (int64_t)0x8000000000000000>(
        dstTile, src0Tile, src1Tile, tmp0DataTile, tmp1DataTile, tmp2DataTile, tmp3MaskTile, tmp4MaskTile);
}
#endif

#define OP_TILE_OP_FLOORDIV TFloorDiv
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TFloorDiv(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    DataTileDefine dstTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T3::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                if constexpr (std::is_same_v<typename T0::Type, half> ||
                              std::is_same_v<typename T0::Type, bfloat16_t>) {
                    FloorDivFp32TmpCompute<T0, T3, tileH, tileW>(dstTile, src0ExecTile, src1ExecTile, tmp,
                                                                 tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, float>) {
                    FloorDivFloatCompute(dstTile, src0ExecTile, src1ExecTile);
                }

#ifdef __DAV_V220
                if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    FloorDivV220Int32Compute<T0, T3, tileH, tileW>(dstTile, src0ExecTile, src1ExecTile, tmp,
                                                                   tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t> ||
                                     std::is_same_v<typename T0::Type, uint8_t>) {
                    FloorDivV220Int8Compute<T0, T3, tileH, tileW>(dstTile, src0ExecTile, src1ExecTile, tmp,
                                                                  tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                }
#else
                if constexpr (std::is_same_v<typename T0::Type, uint8_t>) {
                    FloorDivNonV220Uint8Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t>) {
                    FloorDivNonV220Int8Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    FloorDivNonV220Int32Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int64_t>) {
                    FloorDivNonV220Int64Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                }
#endif
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_FLOOR_DIV_H
