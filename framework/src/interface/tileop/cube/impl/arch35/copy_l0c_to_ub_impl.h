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
 * \file copy_l0c_to_ub_impl.h
 * \brief L0C to UB Data Transfer Interface Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H

#include "../cube_utils.h"

template <typename tileUBTensor, typename tileL0CTensor>
TILEOP void TExtractL0CToUB(
    tileUBTensor& ubTile, tileL0CTensor& l0cTile, int64_t l0cOffset0, int64_t l0cOffset1, int16_t subblockId)
{
    if (subblockId == 0) {
        pto::TEXTRACT<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(
            ubTile, l0cTile, l0cOffset0, l0cOffset1);
    } else {
        pto::TEXTRACT<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(
            ubTile, l0cTile, l0cOffset0, l0cOffset1);
    }
}

template <typename tileUBTensor, typename tileL0CTensor>
TILEOP void TInsertL0CToUB(
    tileUBTensor& ubTile, tileL0CTensor& l0cTile, int64_t ubOffset0, int64_t ubOffset1, int16_t subblockId)
{
    if (subblockId == 0) {
        pto::TINSERT<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(
            ubTile, l0cTile, ubOffset0, ubOffset1);
    } else {
        pto::TINSERT<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(
            ubTile, l0cTile, ubOffset0, ubOffset1);
    }
}

// Copy data from L0C to UB
template <CopyOutMode layoutMode, CopyMode mode, typename Coord, typename DstTileData, typename SrcTileData>
INLINE void TCopyL0C2UBImpl(
    DstTileData& dst, SrcTileData& src, const Coord& dstCoord, const Coord& srcCoord, int16_t subblockId)
{
    static_assert(mode != CopyMode::UNKNOWN,
        "[TCopyL0C2UB Error]: Current CopyMode is UNKNOWN. CopyMode only support EXTRACT, INSERT and MOVE");
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename SrcTileData::Type);
    int64_t dstOffset0 = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM2, 0>(dstCoord);
    int64_t dstOffset1 = TileOp::GetTupleElement<Coord, 1, SHAPE_DIM2, 0>(dstCoord);
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM2, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, 1, SHAPE_DIM2, 0>(srcCoord);
    constexpr auto staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
    constexpr auto staticUBW = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    constexpr auto staticL0CH =
        Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    using tileUBTensor = pto::Tile<pto::TileType::Vec, typename DstTileData::Type, staticUBH, staticUBW,
        layoutMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        layoutMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using tileL0CTensor =
        pto::Tile<pto::TileType::Acc, typename SrcTileData::Type, staticL0CH, staticL0CW, pto::BLayout::ColMajor, -1,
        -1, pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    tileUBTensor ubTile(dstShape0, dstShape1);
    tileL0CTensor l0cTile(srcShape0, srcShape1);
    pto::TASSIGN(ubTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr());
    if constexpr (mode == CopyMode::EXTRACT || mode == CopyMode::MOVE) {
        TExtractL0CToUB<tileUBTensor, tileL0CTensor>(ubTile, l0cTile, srcOffset0, srcOffset1, subblockId);
    } else if (mode == CopyMode::INSERT) {
        TInsertL0CToUB<tileUBTensor, tileL0CTensor>(ubTile, l0cTile, dstOffset0, dstOffset1, subblockId);
    }
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H