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
 * \file dynamic_nsa.cpp
 * \brief
 */

#include "operator/models/nsa/dynamic_nsa_v1.h"

namespace npu::tile_fwk {

void GenGatedScoreCompute(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, GateMode gateMode) {
    FUNCTION("FusedCompressKvSelect", {x, gateW1, gateW2, gateSimW1}, {gatingScore}) {
        GenGatedScore(x, gateW1, gateW2, gateSimW1, gatingScore, gateMode);
    }
}

void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, GateMode gateMode) {
    (void)gateSimW1;
    (void)gateMode;
    DataType dType = x.GetStorage()->Datatype();

    int b = x.GetShape()[0];
    int s = x.GetShape()[1];
    int h = x.GetShape()[2];
    int n1 = gateW2.GetShape()[1] / 3;
    int tileB = b;
    int tileS = s;
    int tileBS = tileB * tileS;

    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    LOOP("LOOP_L0_bIdx_gated_score", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        LOOP("LOOP_L0_sIdx_gated_score", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            TileShape::Current().SetVecTile({tileB, tileS, h});
            TileShape::Current().SetCubeTile({tileBS, tileBS}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
            SymbolicScalar bOfs = bIdx * tileB;
            SymbolicScalar sOfs = sIdx * tileS;
            SymbolicScalar bsOfs = bOfs * sOfs;

            auto xReshape = Reshape(x, {b * s, h});
            auto xView = View(xReshape, {tileBS, h}, {bsOfs, 0});
            auto mm1Res = Matrix::Matmul(DT_FP32, xReshape, gateW1);

            TileShape::Current().SetVecTile({1, h});
            auto sigmoidRes = Sigmoid(mm1Res);
            sigmoidRes = Cast(sigmoidRes, dType);
            TileShape::Current().SetCubeTile({tileBS, tileBS}, {NUM_128, NUM_128}, {NUM_16, NUM_16});
            auto mm2Res = Matrix::Matmul(DT_FP32, sigmoidRes, gateW2);
            TileShape::Current().SetVecTile({tileBS, n1});

            auto res = Reshape(mm2Res, {tileB, tileS, 3, n1});
            TileShape::Current().SetVecTile({1, tileS, 3, n1});

            res = Transpose(res, {2, 3});
            if (gatingScore.GetStorage()->Datatype() != DT_FP32) {
                res = Cast(res, dType);
            }
            Assemble(res, {bOfs, sIdx, 0, 0}, gatingScore);
        }
    }
}

void GenAttn(Tensor &gatingScore, Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &attentionOut) {
    int nDimSize = cmpAtten.GetShape()[2]; // n1
    int vDimSize = cmpAtten.GetShape()[3]; // v_dim
    int tileB = 8;
    int tileS = 1;

    SymbolicScalar bDimSize = GetInputShape(cmpAtten, 0);
    SymbolicScalar sDimSize = GetInputShape(cmpAtten, 1);
    SymbolicScalar bLoop = bDimSize / tileB;
    SymbolicScalar sLoop = sDimSize / tileS;
    DataType dType = attentionOut.GetStorage()->Datatype();
    LOOP("LOOP_L0_bIdx_gen_attn", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop), {}, true) {
        SymbolicScalar bOffset = bIdx * tileB;
        SymbolicScalar actualBSize = std::min(tileB, (bDimSize - bIdx * tileB));
        LOOP("LOOP_L1_sIdx_gen_attn", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sLoop)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0, 0};
            SymbolicScalar actualsSize = std::min(tileS, (sDimSize - sIdx * tileS));
            TileShape::Current().SetVecTile(1, 1, NUM_16, vDimSize);
            auto cmpAttenTile = View(cmpAtten, {tileB, tileS, nDimSize, vDimSize},
                {actualBSize, actualsSize, nDimSize, vDimSize}, {bOffset, sOffset, 0, 0});
            auto selAttenTile = View(selAtten, {tileB, tileS, nDimSize, vDimSize},
                {actualBSize, actualsSize, nDimSize, vDimSize}, {bOffset, sOffset, 0, 0});
            auto winAttenTile = View(winAtten, {tileB, tileS, nDimSize, vDimSize},
                {actualBSize, actualsSize, nDimSize, vDimSize}, {bOffset, sOffset, 0, 0});
            auto cmpAttenFP32Tile = Cast(cmpAttenTile, DT_FP32);
            auto selAttenFP32Tile = Cast(selAttenTile, DT_FP32);
            auto winAttenFP32Tile = Cast(winAttenTile, DT_FP32);
            TileShape::Current().SetVecTile(1, 1, nDimSize, NUM_3);
            auto gatingScoreTile = View(gatingScore, {tileB, tileS, nDimSize, NUM_3},
                {actualBSize, actualsSize, nDimSize, NUM_3}, {bOffset, sOffset, 0, 0});
            auto gatingScoreFP32 = Cast(gatingScoreTile, DT_FP32);
            auto cmpWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 0});
            auto selWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 1});
            auto winWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 2});
            TileShape::Current().SetVecTile(1, 1, NUM_16, vDimSize);
            auto mulCmp = Mul(cmpAttenFP32Tile, cmpWeight);
            auto mulSel = Mul(selAttenFP32Tile, selWeight);
            auto mulWin = Mul(winAttenFP32Tile, winWeight);
            auto addCmpSel = Add(mulCmp, mulSel);
            auto outFP32 = Add(addCmpSel, mulWin);
            TileShape::Current().SetVecTile(1, 1, NUM_16, vDimSize);
            auto attentionOutTile = Cast(outFP32, dType, CAST_RINT);
            Assemble(attentionOutTile, outOffset, attentionOut);
        }
    }
}

void DynamicNsa(const Tensor &x, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, const MlaTileConfig &mlaTileConfig,
    float epsilonCq, float epsilonCkv, std::string cacheMode, // prolog
    Tensor &topkIndices, Tensor &kvActSeqs, Tensor &blockTable, int front, int near, int topk,
    int slcBlockSize, int blockSize, float softmaxScale,
    SATileShapeConfig saTileConfig, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    GateMode gateMode, Tensor &cmpAtten, int winSize,
    WinAttenTileShapeConfig &winAttntileConfig,                  // gen win
    PostTensors &postTensors, const PostTileConfig &postConfig,  // post
    Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &postOut, const Tensor &cmpKvCache, const Tensor &cmpKrCache,
    const Tensor &cmpBlockTable, const Tensor &actSeqLen, const Tensor &actCmpSeqLen, const Tensor &mlpWk1,
    const Tensor &mlpWk2, const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpAttnOut, Tensor &cmpSoftmax,
    Tensor &fullK, Tensor &cmpK, Tensor &firstRope, Tensor &firstRopeInput, Tensor &topkRes, Tensor &topkInput,
    const int cmpBlockSize, const int cmpStride, CmpAttnTile &tileConfig_v2,bool debug) {
    ASSERT(gateMode == standard); // 当前仅支持standard模式

    FUNCTION("main",
        {
            x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,

            cmpKvCache, cmpKrCache, blockTable, cmpBlockTable, actSeqLen, actCmpSeqLen, mlpWk1, mlpWk2, mlpCos, mlpSin,
            quantInputs.dequantScaleWUqQr, quantInputs.smoothScalesCq, // prolog
            topkIndices, kvActSeqs, // genKvSlc
            gateW1, gateW2, gateSimW1, // gatedScore
            cmpAtten, // genAttn
            postTensors.weightUV, postTensors.weightO, postTensors.weightUvScale, postTensors.smoothScalesWUv,
            postTensors.weightOScale, postTensors.smoothScalesWo,      //  paPost
    },
        {postOut, cmpAttnOut, cmpSoftmax, fullK, cmpK, topkRes, topkInput},
        {{kvCacheOut, kvCache}, {krCacheOut, krCache}}) {
        config::SetPassOption(VEC_NBUFFER_MODE, 1);
        config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, NUM_4}});
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_2 * NUM_1024 * NUM_1024);

        int b = x.GetShape()[0];
        int s = x.GetShape()[1]; // s=1
        int n1 = gateW2.GetShape()[1] / 3;
        int n2 = 1;
        int vDim = wUk.GetShape()[2];    // kvLoraRank
        int ropeDim = sin.GetShape()[2]; // [b,s,qkRopeHeadDim]
        auto dtype = x.GetStorage()->Datatype();

        /*********************************/
        /*有数据依赖的子图一定要加loop_barrier*/
        /** 将整个nsa分为以下进行串联
         * subgragh 0: mla_prolog
         * subgragh 1: gen_win_attn
         * subgragh 2: kv_compression
         * subgragh 3: gen_cmp_atten
         * subgragh 4: gen_slc_atten, 其中包括: gen_kv_slc及slc_attn
         * subgragh 5: gen_gated_score
         * subgragh 6: gen_attn
         * subgragh 7: pa_post
         */
        /** 依赖关系如下：
            {subgraph-1, subgraph-2, subgraph-4}: {subgraph-0}
            subgraph-6: {subgraph-1, subgraph-3, subgraph-4, subgraph-5}
            subgraph-3: {subgraph-2}
            subgraph-7: {subgraph-6}
        */
        /*********************************/

        // subgraph-0: prolog
        // queryOut: [b,s1,n1,kvLoraRank], queryRopeOut: [b,s1,n1,qkRopeHeadDim]
        // kvCacheOut: [blockNum * blockSize * n2, kvLoraRank], krCache: [blockNum * blockSize * n2, qkRopeHeadDim]
        Tensor queryOut(dtype, {b, s, n1, vDim}, "queryOut");
        Tensor queryRopeOut(dtype, {b, s, n1, ropeDim}, "queryRopeOut");
        MlaPrologCompute(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs, mlaTileConfig, queryOut, queryRopeOut, kvCacheOut, krCacheOut, epsilonCq, epsilonCkv,
            cacheMode);

        // [b,s1,n1,d] -> [b*s1*n1,d]
        Tensor qNope(dtype, {b * s * n1, vDim}, "qNope");
        Tensor qRope(dtype, {b * s * n1, ropeDim}, "qRope");
        LOOP("RESHAPE_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true) {
            SymbolicScalar bOffset = bIdx * 1;
            LOOP("RESHAPE_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, s, 1)) {
                SymbolicScalar sOffset = sIdx * 1;

                Tensor nopeView = View(queryOut, {1, 1, n1, vDim}, {bOffset, sOffset, 0, 0});
                TileShape::Current().SetVecTile({1, 1, 32, vDim});
                Tensor nopeRes = Reshape(nopeView, {1 * 1 * n1, vDim});
                Assemble(nopeRes, {(bOffset * s + sOffset) * n1, 0}, qNope);

                Tensor ropeView = View(queryRopeOut, {1, 1, n1, ropeDim}, {bOffset, sOffset, 0, 0});
                TileShape::Current().SetVecTile({1, 1, n1, ropeDim});
                Tensor ropeRes = Reshape(ropeView, {1 * 1 * n1, ropeDim});
                Assemble(ropeRes, {(bOffset * s + sOffset) * n1, 0}, qRope);
            }
        }

        // Loop_barrier
        // subgraph-1
        Tensor winAtten(DT_FP32, {b, s, n1, vDim}, "winAtten");
        WinAttentionCompute(qNope, kvCacheOut, qRope, krCacheOut, n1, n2, blockTable, kvActSeqs, winSize, blockSize,
            softmaxScale, winAtten, winAttntileConfig);
        // subgraph-2-3

        Tensor cmpAttnOut16Tmp(dtype, {b, s, n1, vDim}, "cmpAttnOut16Tmp");
        Tensor topkResTmp(DT_INT32, {b, s, 16}, "topkRes");
        if (debug) {
            FusedCompressKvSelectCompute(qNope, qRope, kvCacheOut, krCacheOut, cmpKvCache, cmpKrCache, blockTable,
                cmpBlockTable, actSeqLen, actCmpSeqLen, mlpWk1, mlpWk2, mlpCos, mlpSin, cmpAttnOut, cmpAttnOut16Tmp,
                cmpSoftmax, fullK, cmpK, firstRope, firstRopeInput, topkResTmp, topkInput, blockSize, cmpBlockSize, cmpStride,
                softmaxScale, n1, n2, tileConfig_v2);
        } else {
            FusedCompressKvSelectCompute(qNope, qRope, kvCacheOut, krCacheOut, cmpKvCache, cmpKrCache, blockTable,
                cmpBlockTable, actSeqLen, actCmpSeqLen, mlpWk1, mlpWk2, mlpCos, mlpSin, cmpAttnOut, cmpAttnOut16Tmp,
                cmpSoftmax, fullK, cmpK, firstRope, firstRopeInput, topkRes, topkInput, blockSize, cmpBlockSize, cmpStride,
                softmaxScale, n1, n2, tileConfig_v2);
        }

        // subgraph-4
        // kv_slc+slc_attn
        Tensor slcAttn(DT_FP32, {b, s, n1, vDim}, "slcAttn");
        if (debug) {
            SelectedAttentionCompute(topkResTmp, kvCacheOut, krCacheOut, kvActSeqs, blockTable, qNope, qRope, slcAttn,
                n1, n2, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig, debug);
        } else {
            SelectedAttentionCompute(topkIndices, kvCacheOut, krCacheOut, kvActSeqs, blockTable, qNope, qRope, slcAttn,
                n1, n2, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig, debug);
        }

        // subgraph-5
        /********gen gated_score ********/
        Tensor gatingScore(dtype, {b, s, n1, 3}, "gatingScore");
        GenGatedScore(x, gateW1, gateW2, gateSimW1, gatingScore, gateMode); // GenGatedScore 输出四维[b,s1,n1,3] fp16

        // Loop_barrier
        // subgraph-6
        /******** gen attn ********/
        Tensor attentionOut(dtype, {b, s, n1, vDim}, "attentionOut");
        GenAttn(gatingScore, cmpAttnOut16Tmp, slcAttn, winAtten, attentionOut); // [b,s,n1,vDim] fp16

        config::SetPassOption(SG_PG_UPPER_BOUND, 500000); // 500000
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
        // Loop_barrier
        // subgraph-7: postOut [b,s,h]
        PostCompute(attentionOut, postTensors, postConfig, postOut);
    }
}

} // namespace npu::tile_fwk
