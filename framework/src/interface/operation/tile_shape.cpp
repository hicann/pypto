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
 * \file tile_shape.cpp
 * \brief
 */

#include "tilefwk/tile_shape.h"
#include "interface/configs/config_manager_ng.h"

using namespace npu::tile_fwk;

TileShape &TileShape::Current() {
    static TileShape instance;
    return instance;
}

void TileShape::SetVecTile(const std::vector<int64_t> &tile) {
    vecTile = {tile};
    ConfigManagerNg::GetInstance().CurrentScope()->UpdateValue("vec_tile_shapes", tile);
}

void TileShape::SetVecTile(const VecTile &tile) {
    vecTile = tile;
    ConfigManagerNg::GetInstance().CurrentScope()->UpdateValue("vec_tile_shapes", tile.tile);
}

void TileShape::SetCubeTile(const std::array<int64_t, MAX_M_DIM_SIZE> &m,
                            const std::array<int64_t, MAX_K_DIM_SIZE> &k,
                            const std::array<int64_t, MAX_N_DIM_SIZE> &n,
                            bool setL1Tile, bool enableSplitK) {
    auto nk = k;
    if (nk[2] == 0) {
        nk[2] = nk[1];
    }
    cubeTile = {m, nk, n, setL1Tile, enableSplitK};
    ConfigManagerNg::GetInstance().CurrentScope()->UpdateValue("cube_tile_shapes", cubeTile);
}

void TileShape::SetMatrixSize(const std::vector<int64_t> &size) {
    this->matrixSize = size;
    ConfigManagerNg::GetInstance().CurrentScope()->UpdateValue("matrix_size", size);
}