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
 * \file copy_ub_to_l1_impl.h
 * \brief UB to L1 Data Movement Interface Implementation (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_L1_IMPL__H
#define TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_L1_IMPL__H

#include "../cube_utils.h"

template <typename Coord, typename DstTileData, typename SrcTileData>
INLINE void TExtractUB2L1Impl(DstTileData& dst, SrcTileData& src, const Coord& coord)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileData::Shape>::value;
    constexpr bool isB4 = CheckIsB4<DstTileData>();
    constexpr int64_t c0Size = isB4 ? FP4_BLOCK_ALIGN_BYTE : BLOCK_ALIGN_BYTE / sizeof(typename SrcTileData::Type);
    int64_t offset0 = TileOp::GetTupleElement<Coord, DIM_1ST, SHAPE_DIM2, 0>(coord);
    int64_t offset1 = TileOp::GetTupleElement<Coord, DIM_2ND, SHAPE_DIM2, 0>(coord);
    constexpr int64_t staticUBH =
        Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename SrcTileData::TileShape>::type::value;
    constexpr int64_t staticL1H =
        Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileData::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename DstTileData::TileShape>::type::value;
    using tileUBTensor = pto::Tile<
        pto::TileType::Vec, typename SrcTileData::Type, staticUBH, staticUBW, pto::BLayout::ColMajor, staticUBH,
        staticUBW, pto::SLayout::RowMajor>;
    using tileL1Tensor = pto::Tile<
        pto::TileType::Mat, typename DstTileData::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, staticL1H,
        staticL1W, pto::SLayout::RowMajor>;
    tileUBTensor UBTile;
    tileL1Tensor l1Tile;
    pto::TASSIGN(UBTile, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l1Tile, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT(l1Tile, UBTile, offset0, offset1);
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_COPY_UB_TO_L1_IMPL__H