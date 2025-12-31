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
 * \file decode_sparse_attention.cpp
 * \brief
 */
 
#include "decode_indexer_attention.h"
#include "interface/operation/operation.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "dsia_common.h"
 
namespace npu::tile_fwk {
 
IndexerShapeParams GetindexParamsFromDSASimpleParams(const DSIASimpleParams &params) {
    IndexerShapeParams indexParams;
    indexParams.b = params.b;
    indexParams.seq = params.s1;
    indexParams.dim = params.h;
    indexParams.qLoraRank = params.q_lora_rank;
    indexParams.headDim = params.idx_head_dim;
    indexParams.headNum = params.idx_n_heads;
    indexParams.ropeHeadDim = params.qk_rope_head_dim;
    indexParams.blockSize = params.blockSize;
    indexParams.blockNum = params.blockNum;
    indexParams.nKV = params.n2;
    indexParams.s2 = params.s2;
    indexParams.indexerTileConfigs = params.indexerTileConfigs;
    indexParams.ropeTileConfigs = params.ropeTileConfigs;
    indexParams.tileBS = NUM_16;
    return indexParams;
}
 
void DecodeIndexerAttention(const Tensor &x, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, Tensor &blockTable, Tensor &actSeqs,
    const Tensor &qW, const Tensor &kW, const Tensor &projW, const Tensor &lnW, const Tensor &lnBias,
    const Tensor &indexKCache, Tensor &attentionOut, 
    Tensor &gatherResTmp, Tensor &topkInputTmp, Tensor &indexerTopkResTmp, Tensor &rowSumOutTmp, Tensor &rmsResOutTmp,
    Tensor &queryOutTmp,Tensor &weightOutTmp, Tensor &qNopeOutTmp,Tensor &qRopeOutTmp,const DSIASimpleParams &params) {
    auto dType = x.GetDataType();
    int blockSize = params.blockSize;
    int n1 = params.n1;
    int n2 = params.n2;
    int dn = params.kv_lora_rank;
    int dr = params.rope_dim;
    int idx_head_dim = params.idx_head_dim;
    Tensor kvCacheOut(dType, {GetInputShape(kvCache, 0) , blockSize , n2, dn}, "kvCacheOuTmp"); 
    Tensor krCacheOut(dType, {GetInputShape(krCache, 0) , blockSize , n2, dr}, "krCacheOuTmp");
    Tensor indexKCacheOut(dType, {GetInputShape(indexKCache, 0), blockSize, n2, idx_head_dim}, "indexKCacheOut");
    Tensor kNope2D(dType, {GetInputShape(kvCache, 0) * blockSize * n2, dn}, "kNope2D");
    Tensor kRope2D(dType, {GetInputShape(krCache, 0) * blockSize * n2, dr}, "kRope2D");
    FUNCTION(
        "main",
        {
            x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, blockTable, actSeqs,
            qW, kW, projW, lnW, lnBias, indexKCache,
#if DSIA_DEBUG == 1
            topkInputTmp
#endif
        },
        {attentionOut,
#if DSIA_DEBUG == 1
            rmsResOutTmp, queryOutTmp, weightOutTmp, qNopeOutTmp, qRopeOutTmp,
                rowSumOutTmp, indexerTopkResTmp, gatherResTmp
#endif
        },
        {{kvCacheOut, kvCache}, {krCacheOut, krCache}, {indexKCacheOut, indexKCache}}) {
        auto b = GetInputShape(x, 0);
        auto s1 = GetInputShape(x, 1);
        float softmaxScale = params.softmaxScale;
        Tensor queryNopeOut(dType, {b * s1, n1, dn}, "queryNopeOut");
        Tensor queryRopeOut(dType, {b * s1, n1, dr}, "queryRopeOut");
        Tensor rmsRes(dType, {b * s1, params.q_lora_rank}, "rmsRes");
        MlaPrologComputeV32(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs, params.mlaTileCfg, queryNopeOut, queryRopeOut, kvCacheOut, krCacheOut, rmsRes, params.eps,
            params.eps, params.cacheMode);
 
        LOOP("LOOP_RESHAPE_IN12", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            (void) batchId;
            kNope2D = Reshape(kvCacheOut, {GetInputShape(kvCache, 0) * blockSize * n2, dn}, true);
            kRope2D = Reshape(krCacheOut, {GetInputShape(krCache, 0) * blockSize * n2, dr}, true);
        }
 
        Tensor queryOut(dType, {b * s1, params.idx_n_heads, params.idx_head_dim}, "qOut");
        Tensor weightOut(dType, {b * s1, params.idx_n_heads});
 
        IndexerPrologInput inputs = {
            x, rmsRes, qW, kW, projW, lnW, lnBias, cos, sin, indexKCache, cacheIndex, blockTable};
        IndexerPrologOutput outputs = {queryOut, weightOut, indexKCacheOut};
        auto indexParams = GetindexParamsFromDSASimpleParams(params);
        LightningIndexerPrologCompute(inputs, outputs, indexParams);
 
        Tensor queryOut4D(dType, {b, s1, params.idx_n_heads, params.idx_head_dim}, "qOut4D");
        Tensor weightOut4D(dType, {b, s1, params.idx_n_heads}, "weightOut4D");
        LOOP("Indexer_prolog_reshape_3D_2_4D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
            (void)unUsedIdx;
            queryOut4D = Reshape(queryOut, {b, s1, params.idx_n_heads, params.idx_head_dim}, true);
            weightOut4D = Reshape(weightOut, {b, s1, params.idx_n_heads}, true);
        } // 后续需要优化掉
 
        std::set<int> indexerUnrollList = {64, 32, 16, 8, 4, 1};
#if DSIA_DEBUG == 1
        LightningIndexerTopkImpl(queryOut4D, indexKCacheOut, false, nullptr, nullptr, weightOut4D, actSeqs, blockTable, indexerTopkResTmp, params.topk, params.indexTileCfg, indexerUnrollList, &rowSumOutTmp);
        GatherAfterPrologCompute(topkInputTmp, kNope2D, kRope2D, blockTable, actSeqs, gatherResTmp, params, b, s1);
#else
        LightningIndexerTopk(queryOut4D, indexKCacheOut, weightOut4D, actSeqs, blockTable, indexerTopkResTmp, params.topk, params.indexTileCfg, indexerUnrollList);
        GatherAfterPrologCompute(indexerTopkResTmp, kNope2D, kRope2D, blockTable, actSeqs, gatherResTmp, params, b, s1);
#endif
 
        Tensor qNope(dType, {b * s1 * n1, dn}, "qNope");
        Tensor qRope(dType, {b * s1 * n1, dr}, "qRope");
        LOOP("LOOP_RESHAPE_SEL_ATTN", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            (void) batchId;
            qNope = Reshape(queryNopeOut, {b * s1 * n1, dn}, true);
            qRope = Reshape(queryRopeOut, {b * s1 * n1, dr}, true);
        }
 
        SparseFlashAttentionCompute(qNope, qRope, gatherResTmp, gatherResTmp, actSeqs, n1, n2, softmaxScale, params.topk,
            attentionOut, params.salTileCfg);
    }
}
 
} // namespace npu::tile_fwk