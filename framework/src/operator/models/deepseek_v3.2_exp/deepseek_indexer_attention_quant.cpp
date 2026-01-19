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
 * \file deepseek_indexer_attention_quant.cpp
 * \brief
 */

#include "deepseek_indexer_attention_quant.h"
#include "mla_prolog_quant_v32.h"
#include "lightning_indexer_prolog.h"
#include "quant_lightning_indexer_prolog.h"
#include "lightning_indexer_topk.h"
#include "gather_after_prolog.h"
#include "gather_selected_attention.h"

namespace npu::tile_fwk {

void DeepSeekIndexerAttentionQuant(
    const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &rmsnormGammaCq, const Tensor &rmsnormGammaCkv, const Tensor &cos, const Tensor &sin,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, Tensor &kScaleCache, const Tensor &dequantScaleWUqQr,
    const Tensor &wQb, const Tensor &wQbScale, const Tensor &wK, const Tensor &wProj,
    const Tensor &layernormGammaK, const Tensor &layernormBetaK, const Tensor &hadamardQ, const Tensor &hadamardK,
    const Tensor &idxKCache, const Tensor &idxKScaleCache, const Tensor &actualSeqLengthsKey, const Tensor &blockTable,
    Tensor &attentionOut, DiaQuantAttr &attrs, const DSIASimpleParams &params,
    // debug
    Tensor &debugQNopeOut, Tensor &debugQRopeOut, Tensor &debugRmsNormOut, Tensor &debugRmsNormScaleOut,
    Tensor &debugQInt8Out, Tensor &debugQScaleOut, Tensor &debugWeightsOut,
    [[maybe_unused]] Tensor &indexerTopkResTmp, Tensor &topkValueTmp, Tensor &topkTmpOut
) {
#if QUANT_DSIA_DEBUG == 0
    (void)debugQNopeOut;
    (void)debugQRopeOut;
    (void)debugRmsNormOut;
    (void)debugRmsNormScaleOut;
    (void)debugQInt8Out;
    (void)debugQScaleOut;
    (void)debugWeightsOut;
    (void)topkValueTmp;
    (void)topkTmpOut;
#endif
    int selectedCount = attrs.selectedCount;
    float softmaxScale = attrs.attnSoftmaxScale;
    auto dType = tokenX.GetStorage()->Datatype();
    auto cacheIndexDType = cacheIndex.GetStorage()->Datatype();
    int blockSize = params.blockSize;
    int n1 = params.n1;
    int n2 = params.n2;
    int dn = params.kv_lora_rank;
    int dr = params.rope_dim;
    int idx_head_dim = params.idx_head_dim;
    int h = tokenX.GetShape()[2];
    auto blockNum = GetInputShape(kvCache, 0);
    int maxBlockNumPerBatch = params.maxBlockNumPerBatch;

    Tensor kvCacheOut(DT_INT8, {blockNum, blockSize, n2, dn}, "kvCacheOuTmp");
    Tensor krCacheOut(dType, {blockNum, blockSize, n2, dr}, "krCacheOuTmp");
    Tensor kScaleCacheOut(DT_FP32, {GetInputShape(kvCache, 0), blockSize, n2, 4}, "kScaleCacheOutTmp");
    Tensor idxKCacheOut(DT_INT8, {GetInputShape(idxKCache, 0), blockSize, n2, idx_head_dim}, "idxKCacheOut");
    Tensor idxKScaleCacheOut(DT_FP16, {GetInputShape(idxKScaleCache, 0), blockSize, n2, 1}, "idxKScaleCacheOut");

    FUNCTION(
        "main",
        { // input
            tokenX, wDq, wUqQr, wUk, wDkvKr, rmsnormGammaCq, rmsnormGammaCkv, cos, sin, kvCache,
            krCache, kScaleCache, dequantScaleWUqQr, wQb, wQbScale, wK, wProj,
            layernormGammaK, layernormBetaK, hadamardQ, hadamardK,
            idxKCache, idxKScaleCache, blockTable, cacheIndex, actualSeqLengthsKey,
        },
        { // output
            attentionOut,
#if QUANT_DSIA_DEBUG == 1
            debugQNopeOut, debugQRopeOut, debugRmsNormOut, debugRmsNormScaleOut,
            debugQInt8Out, debugQScaleOut, debugWeightsOut,
            indexerTopkResTmp, topkValueTmp, topkTmpOut
#endif
        },
        { // inplace
            {kvCacheOut, kvCache}, {krCacheOut, krCache}, {kScaleCacheOut, kScaleCache},
            {idxKCacheOut, idxKCache}, {idxKScaleCacheOut, idxKScaleCache}
        })
    {
        //============= mla prolog ================
        auto b = GetInputShape(tokenX, 0);
        auto s1 = GetInputShape(tokenX, 1);
        auto t = b * s1;
        Tensor queryNopeOut(dType, {t, n1, dn}, "queryNopeOut");
        Tensor queryRopeOut(dType, {t, n1, dr}, "queryRopeOut");
        Tensor rmsRes(DT_INT8, {t, params.q_lora_rank}, "rmsRes");
        Tensor rmsScaleRes(DT_FP32, {t, 1}, "rmsScaleRes");

        MlaPrologQuantV32Compute(
            tokenX, wDq, wUqQr, dequantScaleWUqQr, wUk, wDkvKr, rmsnormGammaCq, rmsnormGammaCkv, cos, sin, cacheIndex,
            kvCache, krCache, kScaleCache, rmsRes, rmsScaleRes, queryNopeOut, queryRopeOut, kvCacheOut, krCacheOut,
            kScaleCacheOut, attrs.rmsnormEpsilonCq, attrs.rmsnormEpsilonCkv, attrs.layeroutKey, params.mlaTileCfg);

#if QUANT_DSIA_DEBUG == 1
        LOOP("Mla_Prolog_Post", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b * s1, 1), {}, true) {
            TileShape::Current().SetVecTile({32, 32, 128});
            auto xTemp = View(queryNopeOut, {1, n1, dn}, {bIdx, 0, 0});
            Assemble(xTemp, {bIdx, 0, 0}, debugQNopeOut);
            auto xTemp2 = View(queryRopeOut, {1, n1, dr}, {bIdx, 0, 0});
            Assemble(xTemp2, {bIdx, 0, 0}, debugQRopeOut);

            TileShape::Current().SetVecTile({32, 128});
            auto xTemp3 = View(rmsRes, {1, params.q_lora_rank}, {bIdx, 0});
            Assemble(xTemp3, {bIdx, 0}, debugRmsNormOut);
            auto xTemp4 = View(rmsScaleRes, {1, 1}, {bIdx, 0});
            Assemble(xTemp4, {bIdx, 0}, debugRmsNormScaleOut);
        }
#endif

        //===================== indexer prolog ============================
        Tensor qInt8Out(DT_INT8, {t, params.idx_n_heads, params.idx_head_dim}, "qInt8OutTmp");
        Tensor qScaleOut(DT_FP16, {t, params.idx_n_heads, 1}, "qScaleOutTmp");
        Tensor weightOut(DT_FP16, {t, params.idx_n_heads}, "weightOutTmp");

        Tensor x2D(dType, {t, h}, "x2DTmp");
        Tensor sin2D(dType, {t, dr}, "sin2DTmp");
        Tensor cos2D(dType, {t, dr}, "cos2DTmp");
        Tensor cacheIndex2D(cacheIndexDType, {t}, "cacheIndex2DTmp");
        LOOP("IndexerProlog_Pre_Reshape_Loop", FunctionType::DYNAMIC_LOOP, tIdx, LoopRange(1)) {
            (void)tIdx;
            Reshape(tokenX, x2D);
            Reshape(sin, sin2D);
            Reshape(cos, cos2D);
            Reshape(cacheIndex, cacheIndex2D);
        }

        QuantIndexerPrologInput inputs = {
            x2D, rmsRes, rmsScaleRes, wQb, wQbScale, wK, wProj, layernormGammaK, layernormBetaK,
            cos2D, sin2D, hadamardQ, hadamardK, idxKCache, idxKScaleCache, cacheIndex2D
        };
        QuantIndexerPrologOutput outputs = {qInt8Out, qScaleOut, idxKCacheOut, idxKScaleCacheOut, weightOut};

        QuantIndexerConfigs configs;
        configs.qLinear = {16, 16, 256, 256, 128, 128};
        configs.qHd = {64, 64, 128, 128, 128, 128};
        configs.kLinear = {16, 16, 256, 256, 128, 128};
        configs.wLinear = {16, 16, 256, 256, 128, 128};

        QuantIndexerPrologAttr attr = QuantIndexerPrologAttr();
        attr.eps = attrs.layernormEpsilonK;
        attr.layeroutQuery = attrs.layeroutQuery;
        attr.layeroutKey = attrs.layeroutKey;
        QuantLightningIndexerPrologCompute(inputs, outputs, attr, configs);

#if QUANT_DSIA_DEBUG == 1
        LOOP("Indexer_Prolog_Post", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b * s1, 1), {}, true) {
            TileShape::Current().SetVecTile({32, 32, 128});
            auto qInt8Temp = View(qInt8Out, {1, params.idx_n_heads, idx_head_dim}, {bIdx, 0, 0});
            Assemble(qInt8Temp, {bIdx, 0, 0}, debugQInt8Out);
            auto qScaleTmp = View(qScaleOut, {1, params.idx_n_heads, 1}, {bIdx, 0, 0});
            Assemble(qScaleTmp, {bIdx, 0, 0}, debugQScaleOut);

            TileShape::Current().SetVecTile({32, params.idx_n_heads});
            auto weightTmp = View(weightOut, {1, params.idx_n_heads}, {bIdx, 0});
            Assemble(weightTmp, {bIdx, 0}, debugWeightsOut);
        }
#endif

        //===================== indexer topk ============================
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_100 * NUM_1024 * NUM_1024);
        config::SetPassOption(SG_PG_LOWER_BOUND, NUM_1024);
        config::SetPassOption(SG_PG_UPPER_BOUND, NUM_1024 * NUM_1024);
        config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, NUM_32}});
        config::SetPassOption(SG_PARALLEL_NUM, NUM_2);
        config::SetPassOption(VEC_NBUFFER_MODE, NUM_2);
        config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{
            {-1, 16}
        });
        config::SetRuntimeOption<uint8_t>(
            DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
                                static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));
        config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, NUM_128);
        config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, NUM_128);

        Tensor queryOut4D(DT_INT8, {b, s1, params.idx_n_heads, params.idx_head_dim}, "qOut4D");
        Tensor qScaleOut4D(DT_FP16, {b, s1, params.idx_n_heads, 1}, "qScaleOut4D");
        Tensor weightOut4D(DT_FP16, {b, s1, params.idx_n_heads}, "weightOut4D");
        LOOP("Indexer_prolog_reshape_3D_2_4D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
            (void)unUsedIdx;
            Reshape(qInt8Out, queryOut4D);
            Reshape(qScaleOut, qScaleOut4D);
            Reshape(weightOut, weightOut4D);
        }

        std::set<int> indexerUnrollList = {64, 32, 16, 8, 4, 1};
#if QUANT_DSIA_DEBUG == 1
        Tensor &indexerTopkRes = indexerTopkResTmp;
        LightningIndexerTopkImpl(queryOut4D, idxKCacheOut, true, &qScaleOut4D, &idxKScaleCacheOut,
            weightOut4D, actualSeqLengthsKey, blockTable, indexerTopkRes,
            selectedCount, params.indexTileCfg, indexerUnrollList, &topkTmpOut, &topkValueTmp);
#else
        Tensor indexerTopkRes(DT_INT32, {b, s1, n2, selectedCount}, "indexerTopkResTmp");
        LightningIndexerTopkQuant(queryOut4D, idxKCacheOut, qScaleOut4D, idxKScaleCacheOut,
            weightOut4D, actualSeqLengthsKey, blockTable, indexerTopkRes,
            selectedCount, params.indexTileCfg, indexerUnrollList);
#endif

        //=================== gather selected attention ======================
        Tensor topkRes2D(DT_INT32, {b * s1, n2 * selectedCount}, "topkRes2D");
        LOOP("GATHER_4D_2_2D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
            (void)unUsedIdx;
            Reshape(indexerTopkRes, topkRes2D);
        }

        // reset the previous config
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{});
        config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 0}});
        config::SetPassOption(SG_PARALLEL_NUM, NUM_20);
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, 1 * NUM_1024 * NUM_1024);
        config::SetPassOption(SG_PG_UPPER_BOUND, NUM_20000);
        config::SetPassOption(SG_PG_LOWER_BOUND, NUM_512);
        // set config for attention
        config::SetPassOption(VEC_NBUFFER_MODE, NUM_2);
        config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{{-1, 2}});

        config::SetRuntimeOption<uint8_t>(
            DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH));
        config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, NUM_128);
        config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, NUM_128);

        Tensor qNope2D(dType, {b * s1 * n1, dn}, "qNope2D");
        Tensor qRope2D(dType, {b * s1 * n1, dr}, "qRope2D");
        Tensor kvCache2D(DT_INT8, {blockNum * blockSize, n2 * dn}, "kvCache2D");
        Tensor krCache2D(dType, {blockNum * blockSize, n2 * dr}, "krCache2D");
        Tensor kScaleCache2D(DT_FP32, {blockNum * blockSize, n2 * 4}, "kScaleCache2D");
        LOOP("LOOP_RESHAPE_SEL_ATTN", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
            (void) unUsedIdx;
            Reshape(queryNopeOut, qNope2D);
            Reshape(queryRopeOut, qRope2D);
            Reshape(kvCacheOut, kvCache2D);
            Reshape(krCacheOut, krCache2D);
            Reshape(kScaleCacheOut, kScaleCache2D);
        }

        SelectedAttentionComputeV2(
            qNope2D, qRope2D, kvCache2D, krCache2D, kScaleCache2D, topkRes2D, blockTable, actualSeqLengthsKey, n1, n2,
            softmaxScale, selectedCount, blockSize, maxBlockNumPerBatch, attentionOut, params.salTileCfg
        );
    }
}
} // namespace npu::tile_fwk
