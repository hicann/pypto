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
 * \file distributed_common.cpp
 * \brief
 */

#include "distributed_common.h"
#include <vector>
#include "interface/utils/common.h"
#include "interface/inner/config.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace Distributed {

inline bool CheckTileShape(const std::array<int, MAX_DIST_DIM_SIZE> &shape, int total)
{
    if ((shape[DIST_HEAD_SHAPE] < 0) || (shape[DIST_HEAD_COUNT] < 0) || (shape[DIST_TAIL_SHAPE] < 0)) {
        return false;
    }
    if (shape[DIST_HEAD_SHAPE] * shape[DIST_HEAD_COUNT] + shape[DIST_TAIL_SHAPE] != total) {
        return false;
    }
    return true;
}

void CheckAndGetGroupInfo(const int groupIndex, const TileShape &tileShape, CommGroupInfo &groupInfo)
{
    groupInfo.groupIndex = groupIndex;

    auto rankShape = tileShape.GetDistTileRank();
    ASSERT((rankShape[DIST_HEAD_SHAPE] >= 0) && (rankShape[DIST_HEAD_COUNT] >= 0) && (rankShape[DIST_TAIL_SHAPE] >= 0));
    int rankSize = rankShape[DIST_HEAD_SHAPE] * rankShape[DIST_HEAD_COUNT] + rankShape[DIST_TAIL_SHAPE];
    ASSERT(rankSize > 0);
    groupInfo.rank = std::make_optional(rankShape);
    groupInfo.rankSize = std::make_optional(rankSize);
    ALOG_INFO_F("Distributed opinfo: rank=[%d %d %d], rankSize=%d", groupInfo.rank.value()[DIST_HEAD_SHAPE],
        groupInfo.rank.value()[DIST_HEAD_COUNT], groupInfo.rank.value()[DIST_TAIL_SHAPE], groupInfo.rankSize.value());

    int rankId = tileShape.GetDistRankId();
    if ((rankId >= 0) && (rankId < INT16_MAX) && (rankId < rankSize)) {
        groupInfo.rankId = std::make_optional(rankId);
        ALOG_INFO_F("Distributed opinfo: rankId=%d", groupInfo.rankId.value());
    }
    ASSERT(groupInfo.rankSize.has_value() && groupInfo.rankId.has_value());
}

void CheckAndGetTileInfo(int rowTotal, int colTotal, const TileShape &tileShape, TensorTileInfo &tileInfo)
{
    ASSERT(rowTotal > 0 && colTotal > 0);

    const auto rowShape = tileShape.GetDistTileRow();
    if (CheckTileShape(rowShape, rowTotal)) {
        tileInfo.row = rowShape;
    } else {
        tileInfo.row = {rowTotal, 1, 0};
    }

    const auto colShape = tileShape.GetDistTileCol();
    if (CheckTileShape(colShape, colTotal)) {
        tileInfo.col = colShape;
    } else {
        tileInfo.col = {colTotal, 1, 0};
    }
    ALOG_INFO_F("Distributed opinfo: row=[%d %d %d], col=[%d %d %d]",
        tileInfo.row[DIST_HEAD_SHAPE], tileInfo.row[DIST_HEAD_COUNT], tileInfo.row[DIST_TAIL_SHAPE],
        tileInfo.col[DIST_HEAD_SHAPE], tileInfo.col[DIST_HEAD_COUNT], tileInfo.col[DIST_TAIL_SHAPE]);
}

int GetTilingTensorSize(const TensorTileInfo &tileInfo, const CommGroupInfo &groupInfo)
{
    auto getCnt = [](const std::array<int, MAX_DIST_DIM_SIZE> &tile) {
        return tile[1] + static_cast<int>(tile[2] != 0);
    };
    int rowCnt = getCnt(tileInfo.row);
    int colCnt = getCnt(tileInfo.col);
    ASSERT(groupInfo.rankSize.has_value() && groupInfo.rank.has_value());
    int tileRankCnt = getCnt(groupInfo.rank.value());
    return rowCnt * colCnt * (tileRankCnt + groupInfo.rankSize.value()) *
            static_cast<int>(sizeof(TilingInfo) / sizeof(int));
}

} // namespace Distributed
} // namespace npu::tile_fwk
