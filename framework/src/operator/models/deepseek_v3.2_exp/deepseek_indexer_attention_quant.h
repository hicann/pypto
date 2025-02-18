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
 * \file deepseek_indexer_attention_quant.h
 * \brief
 */

#pragma once
#ifndef DEEPSEEK_INDEXER_ATTENTION_QUANT_H
#define DEEPSEEK_INDEXER_ATTENTION_QUANT_H

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"

#include "dsia_common.h"

namespace npu::tile_fwk {

constexpr int NUM_100 = 100;
constexpr int NUM_10000 = 10000;
constexpr int NUM_20000 = 20000;

struct DiaQuantAttr {
    float rmsnormEpsilonCq = 1e-6f;
    float rmsnormEpsilonCkv = 1e-6f;
    float layernormEpsilonK = 1e-6f;
    float attnSoftmaxScale = static_cast<float>(1.0 / sqrtf(512 + 64));
    int selectedCount = 2048;
    std::string layeroutQuery = "TND";
    std::string layeroutKey = "PA_BSND";
};

void DeepSeekIndexerAttentionQuant(
    const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &rmsnormGammaCq, const Tensor &rmsnormGammaCkv, const Tensor &cos, const Tensor &sin,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, Tensor &kScaleCache, const Tensor &dequantScaleWUqQr,
    const Tensor &wQb, const Tensor &wQbScale, const Tensor &wK, const Tensor &wProj,
    const Tensor &layernormGammaK, const Tensor &layernormBetaK, const Tensor &hadamardQ, const Tensor &hadamardK,
    const Tensor &idxKCache, const Tensor &idxKScaleCache, const Tensor &actualSeqLengthsKey, const Tensor &blockTable,
    Tensor &attentionOut, DiaQuantAttr &attrs, const DSIASimpleParams &params,
    Tensor &debugQNopeOut, Tensor &debugQRopeOut, Tensor &debugRmsNormOut, Tensor &debugRmsNormScaleOut,
    Tensor &debugQInt8Out, Tensor &debugQScaleOut, Tensor &debugWeightsOut,
    Tensor &indexerTopkResTmp, Tensor &topkValueTmp, Tensor &topkTmpOut
);

} // namespace npu::tile_fwk

#endif // DEEPSEEK_INDEXER_ATTENTION_QUANT_H
