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
 * \file set_heuristic_tile_shapes.h
 * \brief
 */

#ifndef PASS_SET_HEURISTIC_TILE_SHAPES_H_
#define PASS_SET_HEURISTIC_TILE_SHAPES_H_
 
#define CUBE_TILES // comment to disable
#define VECTOR_TILES // comment to disable

#include "tilefwk/platform.h"
#include "passes/pass_interface/pass.h"
 
namespace npu::tile_fwk {
// Neseccary params
constexpr int64_t M_DIM = 0;
constexpr int64_t K_DIM = 1;
constexpr int64_t N_DIM = 2;
constexpr int64_t FACTOR = 2;
constexpr int64_t MIN_TILE = 16;
constexpr int64_t MAX_MDIM = 2;
constexpr int64_t MAX_KDIM = 3;
constexpr int64_t MAX_NDIM = 2;

constexpr const int64_t MIN_TILE_SIZE = 2048;
constexpr const int64_t MAX_TILE_SIZE = 192 * 1024 / 4; // 4 - max num of In/Out/Tmp buffers
constexpr const int64_t DEFAULT_TILE_SIZE = 4096;
constexpr const int64_t BYTES_PER_REPEAT = 256;
constexpr const int64_t DEFAULT_MAX_PARALLELISM = 128;
constexpr const int64_t DEFAULT_LATENCY = 10;
constexpr const int64_t UINT8MAX = 255;
constexpr const int64_t TRANSPOSE_VNCHWCONV_LAST_DIM = 2;
constexpr const int64_t VNCHWCONV_POINTERS = 32;
 
// Additional cube variable parameters
constexpr int64_t DOUBLE_BUFFER = 1; // 1 - disable, 2 - enable
constexpr int64_t WHOLE_M_SCORE = 2;
constexpr int64_t WHOLE_K_SCORE = 2;
constexpr int64_t WHOLE_N_SCORE = 2;
constexpr int64_t WEIGHT_L0 = 200;
constexpr double TASKS_CUBE_WEIGHT = 0.5;
constexpr double RESIDUAL_CUBE_TASKS_WEIGHT = 0.2;
constexpr int64_t BALANCE_WEIGHT = 1;
constexpr int64_t CYCLES_WEIGHT = 2;
 
// Additional vector variable parameters
constexpr double LAST_AXIS_WEIGHT = 0.1;
constexpr double WEIGHT_UB = 10;
constexpr double TASKS_VECTOR_WEIGHT = 0.7;
constexpr double RESIDUAL_VECTOR_TASKS_WEIGHT = 2;
 
class SetHeuristicTileShapes : public Pass {
public:
   SetHeuristicTileShapes() : Pass("SetHeuristicTileShapes") {}
   ~SetHeuristicTileShapes() override = default;
   Status RunOnFunction(Function &function) override;
 
private:
   void SetHeuristicTileShapesFunc(Function &function) const;
};
}
#endif // PASS_SET_HEURISTIC_TILE_SHAPES_H_