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
 * \file attention.h
 * \brief
 */

#pragma once

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "dynamic_mla.h"
#include "page_attention.h"

namespace npu::tile_fwk {

void Attention(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, Tensor &qNopeOut, Tensor &qRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    const MlaQuantInputs &quantInputs, const RoPETileShapeConfigNew &ropeConfig,/*---*/
    Tensor &blockTable, Tensor &actSeqs, Tensor &paOut, int blockSize, float softmaxScale, PaTileShapeConfig &paTileConfig, /*---*/
    Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW, Tensor &postOut,
    float epsilonCq = 1e-5f, float epsilonCkv = 1e-5f, std::string cacheMode = "BNSD");

} // namespace npu::tile_fwk
