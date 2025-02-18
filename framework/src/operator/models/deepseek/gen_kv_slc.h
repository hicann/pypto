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
 * \file gen_kv_slc.h
 * \brief
 */
#pragma once
#ifndef GEN_KV_SLC
#define GEN_KV_SLC

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
    struct KvSlcTileShapeConfig {
        std::array<int, TILE_VEC_DIMS> v0TileShape;
    };
    void KvSlcCompute(Tensor &topK_indcies, Tensor &topK_tensor_shape, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, int front, int near, int topk, int l_prime,
        int n2, Tensor &blockTable, int blockSize, Tensor &k_slcOut, Tensor &v_slcOut, Tensor &kvSlcActSeqs, KvSlcTileShapeConfig &tileConfig, bool debug = false);
    void GenKvSlc(Tensor &topK_indcies, Tensor &topK_tensor_shape, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, int front, int near, int topk, int l_prime, int n2,
        Tensor &blockTable, int blockSize, Tensor &k_slcOut, Tensor &v_slcOut, Tensor &kvSlcActSeqs, KvSlcTileShapeConfig &tileConfig);
} // namespace npu::tile_fwk

#endif // MLA_PROLOG