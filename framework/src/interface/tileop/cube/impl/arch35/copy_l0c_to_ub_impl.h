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

// Copy data from L0C to UB
template <CopyOutMode mode, typename Coord, typename DstTileData, typename SrcTileData>
INLINE void TExtractL0C2UBImpl(DstTileData& dst, SrcTileData& src, const Coord& coord, int16_t subblockId)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename SrcTileData::Type);
    if constexpr (DstTileData::FORMAT == Hardware::UB && SrcTileData::FORMAT == Hardware::L0C) {
        int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
        int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
        constexpr auto staticUBH =
            Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
        constexpr auto staticUBW = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
        constexpr auto staticL0CH =
            Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
        constexpr auto staticL0CW = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
        int64_t srcShape0 = GetShape<0>(src);
        int64_t srcShape1 = GetShape<1>(src);
        int64_t dstShape0 = GetShape<0>(dst);
        int64_t dstShape1 = GetShape<1>(dst);
        int64_t l0cOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
        using tileUBTensor = pto::Tile<
            pto::TileType::Vec, typename DstTileData::Type, staticUBH, staticUBW,
            mode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
            mode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
        using tileL0CTensor = pto::TileAcc<typename SrcTileData::Type, staticL0CH, staticL0CW, -1, -1>;
        tileUBTensor UBTile(dstShape0, dstShape1);
        tileL0CTensor l0cTile(srcShape0, srcShape1);
        pto::TASSIGN(UBTile, static_cast<uint64_t>(dst.GetAddr()));
        pto::TASSIGN(l0cTile, static_cast<uint64_t>(src.GetAddr()) + l0cOffset);
        if (subblockId == 0) {
            pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
        } else {
            pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_L0C_TO_UB_IMPL__H