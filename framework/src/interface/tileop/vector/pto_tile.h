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
 * \file pto_tile.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PTO_TILE__H
#define TILEOP_TILE_OPERATOR_PTO_TILE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <cstddef>

template <typename T>
class PtoTile {
public:
    static constexpr auto expect_size = 5;
    static constexpr auto size = Std::tuple_size<typename T::Shape>::value;
    using Type =
        pto::Tile<pto::TileType::Vec, typename T::Type, TileOp::GetOutterAxisMergeResult<size, typename T::TileShape>(),
            TileOp::GetTensorTileShapeDim<T, DIM_5TH, expect_size>(), pto::BLayout::RowMajor,
            TileOp::GetOutterAxisMergeResult<size, typename T::Shape>(),
            TileOp::GetTensorShapeDim<T, DIM_5TH, expect_size>()>;

    __aicore__ inline PtoTile() {}

    __aicore__ inline const Type &Tile() const { return tile_; }

private:
    Type tile_;
};

template <typename T>
class DynPtoTile {
public:
    static constexpr auto expect_size = 5;
    static constexpr auto size = Std::tuple_size<typename T::Shape>::value;
    using Type =
        pto::Tile<pto::TileType::Vec, typename T::Type, TileOp::GetTensorTileShapeDim<T, DIM_4TH, expect_size>(),
            TileOp::GetTensorTileShapeDim<T, DIM_5TH, expect_size>(), pto::BLayout::RowMajor, -1, -1>;

    __aicore__ inline DynPtoTile(T tensor)
        : tile_(tensor.GetLayout().template GetShapeDim<DIM_4TH, expect_size>(),
              tensor.GetLayout().template GetShapeDim<DIM_5TH, expect_size>()) {}

    __aicore__ inline const Type &Tile() const { return tile_; }

private:
    Type tile_;
};
#endif