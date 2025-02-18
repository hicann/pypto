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
 * \file DYNAMIC_NSA_V1.h
 * \brief
 */

#pragma once
#ifndef DYNAMIC_NSA_V1
#define DYNAMIC_NSA_V1

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "operator/models/nsa/nsa_selected_attention.h"
#include "operator/models/deepseek/dynamic_mla.h"
#include "operator/models/nsa/win_attention.h"
#include "operator/models/nsa/attention_post.h"
#include "fused_compress_kv_select.h"

namespace npu::tile_fwk {
constexpr int NUM_65536 = 65536;

enum GateMode {
    standard,
    simple
};

struct NSAV1SimpleParams {
    int b;
    int s1;
    int s2;
    int n1;
    int n2;
    int h;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    int rope_dim;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int front;
    int near;
    int topk;
    std::string cacheMode;
    int blockSize;
    int winSize;
    int vHeadDim;
    float eps;
    static NSAV1SimpleParams getCommonParams() {
        NSAV1SimpleParams params;
        params.h = NUM_7168;
        params.q_lora_rank = NUM_1536;
        params.kv_lora_rank = NUM_512;
        params.qk_rope_head_dim = NUM_64;
        params.qk_nope_head_dim = NUM_128;
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.rope_dim = NUM_64;
        params.cmpBlockSize = NUM_32;
        params.cmpStride = NUM_16;
        params.slcBlockSize = NUM_64;
        params.front = NUM_1;
        params.near = NUM_2;
        params.topk = NUM_16;
        params.cacheMode = "BSND";
        params.blockSize = NUM_128;
        params.winSize = NUM_512;
        params.vHeadDim = NUM_128;
        params.eps = 1e-5f;
        return params;
    }

    static NSAV1SimpleParams getDecodeParams() {
        NSAV1SimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_1;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }

    static NSAV1SimpleParams getMTPParams() {
        NSAV1SimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_2;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }
};
void GenGatedScoreCompute(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, GateMode gateMode) ;

void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor& gatingScore, GateMode gateMode);

void GenAttn(Tensor &gatingScore, Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &attentionOut);

void DynamicNsa(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, float epsilonCq, float epsilonCkv, std::string cacheMode,
    Tensor &topkIndices, Tensor &kvActSeqs, Tensor &blockTable,
    int front, int near, int topk, int slcBlockSize, int blockSize,
    float softmaxScale, SATileShapeConfig saTileConfig,
    const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1, GateMode gateMode,
    Tensor &cmpAtten, int winSize, WinAttenTileShapeConfig &winAttntileConfig,
    PostTensors &postTensors, const PostTileConfig &postConfig,
    Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &postOut,
    const Tensor &cmpKvCache, const Tensor &cmpKrCache, const Tensor &cmpBlockTable,
    const Tensor &actSeqLen, const Tensor &actCmpSeqLen, const Tensor &mlpWk1, const Tensor &mlpWk2,
    const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpAttnOut, Tensor &cmpSoftmax, Tensor &fullK, Tensor &cmpK,
    Tensor &firstRope, Tensor &firstRopeInput, Tensor &topkRes, Tensor &topkInput,
    const int cmpBlockSize, const int cmpStride,
    CmpAttnTile &tileConfig_v2,bool debug = false);

} // namespace npu::tile_fwk

#endif // DYNAMIC_NSA_V1
