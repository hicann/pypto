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
 * \file mla_prolog_quant_v32.h
 * \brief
 */

#pragma once
#ifndef MLA_PROLOG_QUANT_V32
#define MLA_PROLOG_QUANT_V32

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/deepseek_v3.2_exp/dynamic_mla_v32.h"

namespace npu::tile_fwk {

void MlaPrologQuantV32Compute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr,
    const Tensor &dequantScaleWUqQr, const Tensor &wUk, const Tensor &wDkvKr, const Tensor &rmsnormGammaCq,
    const Tensor &rmsnormGammaCkv, const Tensor &ropeCos, const Tensor &ropeSin, const Tensor &cacheIndex, Tensor &kvCache,
    Tensor &krCache, Tensor &kScaleCache, Tensor &qNormOut, Tensor &qNormScaleOut, Tensor &qNopeOut, Tensor &qRopeOut,
    Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &kScaleCacheOut, float rmsnormEpsilonCq,
    float rmsnormEpsilonCkv, const std::string &layoutKey, const MlaTileConfig &tileConfig);

void MlaPrologQuantV32(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr,
    const Tensor &dequantScaleWUqQr, const Tensor &wUk, const Tensor &wDkvKr, const Tensor &rmsnormGammaCq,
    const Tensor &rmsnormGammaCkv, const Tensor &ropeCos, const Tensor &ropeSin, const Tensor &cacheIndex, Tensor &kvCache,
    Tensor &krCache, Tensor &kScaleCache, Tensor &qNormOut, Tensor &qNormScaleOut, Tensor &qNopeOut, Tensor &qRopeOut,
    Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &kScaleCacheOut, float rmsnormEpsilonCq, float rmsnormEpsilonCkv,
    const std::string &layoutKey,const MlaTileConfig &tileConfig);

} // namespace npu::tile_fwk

#endif // MLA_PROLOG_QUANT_V32
