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
 * \file attention.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
constexpr int NUM_100000 = 100000;
constexpr int NUM_500000 = 500000;


void Attention(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, Tensor &qNopeOut, Tensor &qRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    const MlaQuantInputs &quantInputs, const RoPETileShapeConfigNew &ropeConfig,/*---*/
    Tensor &blockTable, Tensor &actSeqs, Tensor &paOut, int blockSize, float softmaxScale,
    PaTileShapeConfig &paTileConfig, /*---*/
    Tensor &weightUV, Tensor &weightO, Tensor &weightOScaleW, Tensor &postOut, float epsilonCq, float epsilonCkv, std::string cacheMode) {
    TileOpFormat paFormat = cacheMode == "PA_NZ" ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    auto dtype = tokenX.GetStorage()->Datatype();
    int b = tokenX.GetShape()[0];
    int s = tokenX.GetShape()[1]; // s=1
    int h = tokenX.GetShape()[2];
    int s2 = kvCache.GetShape()[2];
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk.GetShape()[0];
    int qkNopeHeadDim = wUk.GetShape()[1];
    int kvLoraRank = wUk.GetShape()[2];
    int qkRopeHeadDim = sin.GetShape()[2]; // [b,s,qkRopeHeadDim]
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int tileB = b;
    int tileBS = tileB * s;

    // 入参B*S*N合轴
    int dN = kvLoraRank;
    int dR = qkRopeHeadDim;

    int nTile = paTileConfig.headNumQTile;
    auto c1Tile = paTileConfig.c1TileShape;
    auto v1Tile = paTileConfig.v1TileShape;
    auto c2Tile = paTileConfig.c2TileShape;
    auto v2Tile = paTileConfig.v2TileShape;

    int vHeadDim = weightUV.GetShape()[2];

    std::vector<int> paOutShape = {b * s * n, kvLoraRank};

    config::SetPassConfig("PVC2_OOO", "InferMemoryConflict", KEY_DISABLE_PASS, true);

    FUNCTION("main",
        {tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
         quantInputs.dequantScaleWUqQr, quantInputs.smoothScalesCq,
         blockTable, actSeqs, weightUV, weightO, weightOScaleW},
        {postOut}, {{kvCacheOut, kvCache}, {krCacheOut, krCache}}) {
        /******** mla_prolog ********/
        SymbolicScalar bLoop = b / tileB;
        config::SetPassOption(VEC_NBUFFER_MODE, 1);
        config::SetPassOption(CUBE_L1_REUSE_MODE, NUM_4); //CubeL1reusemode合并的左矩阵或者右矩阵数量
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});   //从NUM_3个mm开始设置CubeNBuffer数量为NUM_4；CubeNBuffer：设置同构的mm计算合并入一个图
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_2 * NUM_1024 * NUM_1024);   // CubeNBuffer、CubeL1reusemode合并时copyin的cycle上限
        config::SetPassOption(SG_PG_UPPER_BOUND, NUM_100000);    // 设置切图与合图后子图的Latency的上限
        config::SetPassOption(SG_PARALLEL_NUM, NUM_2);       // 设置子图合并的并行度下限（子图数量大于等于pgParallelLowerBound才可合并）

        LOOP("LOOP_L0_bIdx_mla_prolog", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
            SymbolicScalar bOffset = bIdx * tileB;
            std::vector<SymbolicScalar> outputOffset = {bOffset, 0, 0, 0};

            Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
            bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
            Tensor smoothScalesCq = quantInputs.smoothScalesCq;
            bool isSmooth = (smoothScalesCq.GetStorage() != nullptr);
            std::cout << "isQuant +++ " << isQuant << std::endl;
            auto xView = View(tokenX, {tileB, s, h}, {bOffset, 0, 0});
            config::SetSemanticLabel("mlaPre");

            auto qKv = mlaPre(xView, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs, false, isSmooth);
            Tensor q = qKv[0];     // [b*s, n*qHeadDim]
            Tensor kvTmp = qKv[1]; // [b,s,kvLoraRank+qkRopeHeadDim]

            // dequant: int32 -> fp32 -> *scale -> fp16/bf16
            if (isQuant) {
                config::SetSemanticLabel("Quant");
                std::vector<int64_t> tileShape = {std::min(NUM_32, tileBS), NUM_64};
                TileShape::Current().SetVecTile(tileShape);
                auto qTmpFp32 = Cast(q, DataType::DT_FP32);
                auto qTmpDequantScale = qKv[2];
                auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpDequantScale);
                auto qTmpDequantChannel = Mul(qTmpDequantPerToken, dequantScaleWUqQr);

                q = Cast(qTmpDequantChannel, dtype);
            }

            config::SetSemanticLabel("Reshape0");
            auto qTmp = Reshape(q, {tileB, s, n, qHeadDim});
            std::vector<int64_t> tileShape = {std::min(NUM_32, tileB), 1, 1, NUM_64};
            TileShape::Current().SetVecTile(tileShape);

            /******** q ********/
            config::SetSemanticLabel("q");
            Tensor qNope = View(qTmp, {tileB, s, n, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
            tileShape = {tileB, 1, 1, NUM_128};
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeRes = Reshape(qNope, {tileBS, n, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
            tileShape = {std::min(NUM_32, tileBS), 1, qkNopeHeadDim};     // {NUM_2, NUM_32, qkNopeHeadDim}
            TileShape::Current().SetVecTile(tileShape);
            config::SetSemanticLabel("Transpose0");
            Tensor qNopeTrans = Transpose(qNopeRes, {0, 1}); // [n,bs,qkNopeHeadDim]

            int c0 = NUM_16;
            int m = (std::min(NUM_32, tileBS) + c0 - 1) / c0 * c0;
            TileShape::Current().SetCubeTile({m, m}, {NUM_256, NUM_256}, {NUM_128, NUM_128}, true);
            config::SetSemanticLabel("BatchMatmul");
            Tensor qNopeNew = Matrix::BatchMatmul(dtype, qNopeTrans, wUk);

            tileShape = {1, std::min(NUM_32, tileBS), kvLoraRank}; // {NUM_16, NUM_2, kvLoraRank}
            TileShape::Current().SetVecTile(tileShape);
            config::SetSemanticLabel("Transpose1");
            Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]

            /******** kv ********/
            config::SetSemanticLabel("kv");
            Tensor compressedKv = View(kvTmp, {tileB, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
            tileShape = {NUM_2, 1, NUM_512};
            TileShape::Current().SetVecTile(tileShape);
            config::SetSemanticLabel("RmsNorm");
            Tensor compressedKvNorm = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b,s,kvLoraRank]
            Tensor kNope = Reshape(compressedKvNorm, {tileB, 1, s, kvLoraRank});   // [b,1,s,kvLoraRank]

            /******** RoPE ********/
            config::SetSemanticLabel("RoPE");
            Tensor kPeView = View(kvTmp, {tileB, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // [b,s,qkRopeHeadDim]
            tileShape = {std::min(NUM_32, tileB), 1, qkRopeHeadDim};
            TileShape::Current().SetVecTile(tileShape);
            Tensor kPeRes = Reshape(kPeView, {tileB, s, 1, qkRopeHeadDim}); // [b,s,1,qkRopeHeadDim]
            Tensor qPeView = View(qTmp, {tileB, s, n, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
            Tensor cosView = View(cos, {tileB, s, qkRopeHeadDim}, {bOffset, 0, 0});
            Tensor sinView = View(sin, {tileB, s, qkRopeHeadDim}, {bOffset, 0, 0});
            Tensor kRopeView(kPeRes.GetStorage()->Datatype(), {tileB, s, 1, qkRopeHeadDim}, "kRopeView"); // [b,1,s,qkRopeHeadDim]
            Tensor qRopeView(kPeRes.GetStorage()->Datatype(), {tileB, s, n, qkRopeHeadDim}, "qRopeView");
            config::SetSemanticLabel("ApplyRotaryPosEmbV2");
            ApplyRotaryPosEmbV2(qPeView, kPeRes, cosView, sinView, qRopeView, kRopeView, NUM_2, ropeConfig);

            if (cacheMode != "BNSD") {
                int blockNum = kvCache.GetShape()[0];
                int n2 = kvCache.GetShape()[2];
                Tensor kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank});
                Tensor krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim});
                auto cacheIndexDview = View(cacheIndex, {tileB, s}, {bOffset, 0});
                kNope = Reshape(kNope, {tileB * s, kvLoraRank}); // [b*s,kvLoraRank]
                Tensor kRopeRes = Reshape(kRopeView, {tileB * s * 1, qkRopeHeadDim});

                /******** kvCache ********/
                tileShape = {1, kvLoraRank};
                TileShape::Current().SetVecTile(tileShape);
                auto kvCacheOutDview = ScatterUpdate(kvCacheRes, cacheIndexDview, kNope, SCATTER_UPADATE_DIM, cacheMode, blockSize);

                /******** krCache ********/
                tileShape = {1, qkRopeHeadDim};
                TileShape::Current().SetVecTile(tileShape);
                auto krCacheOutDview = ScatterUpdate(krCacheRes, cacheIndexDview, kRopeRes, SCATTER_UPADATE_DIM, cacheMode, blockSize);

                kvCacheOut = Reshape(kvCacheOutDview, {blockNum, blockSize, n2, kvLoraRank});
                krCacheOut = Reshape(krCacheOutDview, {blockNum, blockSize, n2, qkRopeHeadDim});
            } else {
                config::SetSemanticLabel("Reshape1");
                Tensor kRopeRes = Reshape(kRopeView, {tileB, 1, s, qkRopeHeadDim});
                config::SetSemanticLabel("kvCache");
                auto cacheIndexDview = View(cacheIndex, {tileB, s}, {bOffset, 0});
                /******** kvCache ********/
                tileShape = {1, 1, 1, kvLoraRank};
                TileShape::Current().SetVecTile(tileShape);
                // kvCache: [b,1,s2,kvLoraRank], output3
                auto kvCacheDview = View(kvCache, {tileB, 1, s2, kvLoraRank}, {bOffset, 0, 0, 0});
                config::SetSemanticLabel("ScatterUpdate0");
                auto kvCacheOutDview = ScatterUpdate(kvCacheDview, cacheIndexDview, kNope, -2);

                /******** krCache ********/
                config::SetSemanticLabel("krCache");
                tileShape = {1, 1, 1, qkRopeHeadDim};
                TileShape::Current().SetVecTile(tileShape);
                // krCache: [b,1,s2,qkRopeHeadDim], output4
                auto krCacheDview = View(krCache, {tileB, 1, s2, qkRopeHeadDim}, {bOffset, 0, 0, 0});
                config::SetSemanticLabel("ScatterUpdate1");
                auto krCacheOutDview = ScatterUpdate(krCacheDview, cacheIndexDview, kRopeRes, -2);

                auto kvCacheOutDviewNew = Reshape(kvCacheOutDview, {tileB*1*s2, kvLoraRank});
                auto krCacheOutDviewNew = Reshape(krCacheOutDview, {tileB*1*s2, qkRopeHeadDim});
                Assemble(kvCacheOutDviewNew, {bOffset*s2, 0}, kvCacheOut);
                Assemble(krCacheOutDviewNew, {bOffset*s2, 0}, krCacheOut);
            }

            auto queryOutDviewNew = Reshape(qNopeNewTrans, {tileB*s*n, kvLoraRank});
            auto qRopeViewNew = Reshape(qRopeView, {tileB*s*n, qkRopeHeadDim});
            Assemble(queryOutDviewNew, {bOffset*s*n, 0}, qNopeOut);
            Assemble(qRopeViewNew, {bOffset*s, 0}, qRopeOut);
        }

        /******** pa ********/
        SymbolicScalar batchSizeScalar = blockTable.GetShape()[0];
        SymbolicScalar nQ = qNopeOut.GetShape()[0] / batchSizeScalar;
        SymbolicScalar nLoop = nQ / nTile;

        config::SetPassOption(CUBE_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
        config::SetPassOption(CUBE_L1_REUSE_MODE, 0);
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, 1 * NUM_1024 * NUM_1024);
        config::SetPassOption(SG_PG_UPPER_BOUND, NUM_100000);
        config::SetPassOption(SG_PARALLEL_NUM, NUM_2);
        config::SetOperationConfig(KEY_FORCE_COMBINE_AXIS, true);

        LOOP("LOOP_L0_bIdx_pa", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeScalar, 1), {}, true) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = (curSeq + blockSize - 1) / blockSize;
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                    config::SetSemanticLabel("pa");
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = View(qNopeOut, {curNTile, dN}, {curOffset, 0});
                    auto qr = View(qRopeOut, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    Assemble(qn, {0, 0}, qi);
                    Assemble(qr, {0, dN}, qi);

                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = View(kvCacheOut, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = View(krCacheOut, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj", paFormat);
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    auto vj = View(kvCacheOut, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    config::SetSemanticLabel("paQkMM");
                    TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                    auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kj, false, true); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    config::SetSemanticLabel("paQkvec1");
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                    auto sijScale = Mul(sij, Element(DataType::DT_FP32, static_cast<double>(softmaxScale))); // (curNTile, curS2Tile)

                    auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub = Sub(sijScale, tildaMij);
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF (IsLoopBegin(bn, 0)) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        config::SetSemanticLabel("paKvMm");
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto oiTmp = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            config::SetSemanticLabel("paKvVec2");
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, paOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        config::SetSemanticLabel("paUpdateVec2");
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        auto miNew = Maximum(mi, tildaMij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t1 = Sub(mi, miNew);           // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t6 = Mul(t2, li);       // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto liNew = Add(t6, t5);    // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)

                        auto q3 = Mul(oi, t2); // (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        config::SetSemanticLabel("paUpdateMM2");
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, paOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }

        /******** post ********/
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, 1 * NUM_1024 * NUM_1024);
        config::SetPassOption(SG_PG_UPPER_BOUND, NUM_500000);
        config::SetPassOption(SG_PARALLEL_NUM, NUM_20);
        config::SetOperationConfig(KEY_FORCE_COMBINE_AXIS, false);
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
        TileShape::Current().SetMatrixSize({});
        LOOP("PaPost", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop), {}, true) {
            config::SetSemanticLabel("Post");
            auto postInUnit = View(paOut, {tileB * s * n, kvLoraRank}, {bIdx * tileB * s * n, 0});

            auto r1Res = Reshape(postInUnit, {tileB*s, n, kvLoraRank}); // 128个
            TileShape::Current().SetVecTile({std::min(NUM_32, tileB * s), NUM_2, kvLoraRank});
            auto cast1 = Cast(r1Res, DT_FP16);
            auto t1Res = Transpose(cast1, {0, 1}); // (n, tileB * s, kvLoraRank)    // 128个

            TileShape::Current().SetCubeTile({std::min(NUM_32, tileB * s), std::min(NUM_32, tileB * s)},
                {std::min(256, kvLoraRank), std::min(512, kvLoraRank)}, {vHeadDim, vHeadDim},
                true);                                                 // raw tileB*1  512   128   // 128/4个
            auto bmmRes = Matrix::BatchMatmul(dtype, t1Res, weightUV); // (n, tileB, kvLoraRank) * (n, kvLoraRank, vHeadDim) -> (n, tileB, vHeadDim)

            TileShape::Current().SetVecTile(NUM_4, std::min(NUM_32, tileB * s), vHeadDim); // raw (128, tileB*1, 128)
            auto t3Res = Transpose(bmmRes, {0, 1}); // (n, tileB, vHeadDim) -> (tileB, n, vHeadDim) // 128个
            auto r2Res = Reshape(t3Res, {tileB * s, n * vHeadDim}); // (tileB * s, n, vHeadDim) -> (tileB * s, n*vHeadDim)

            TileShape::Current().SetVecTile(1, n * vHeadDim); // raw (tileB*1, 128*128)
            auto quantA = Quant(r2Res);
            auto quantizedA = std::get<0>(quantA); //(tileB * s, n*vHeadDim)
            auto dequantScaleA = std::get<1>(quantA); //(tileB * s, 1)

            TileShape::Current().SetCubeTile({std::min(32, tileB * s), std::min(32, tileB * s)},
                {std::min(512, n * vHeadDim), std::min(512, n * vHeadDim)}, {std::min(64, h), std::min(64, h)},
                true); // raw  tileB*1  16k  7168
            Tensor res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, weightO);

            TileShape::Current().SetVecTile(std::min(NUM_32, tileB * s), std::min(NUM_32, h)); // raw (tileB*1, 7168)
            res = Cast(res, DataType::DT_FP32);
            res = Mul(res, dequantScaleA);   // (B*s, 1)
            Tensor weightOScaleW2Dim = Reshape(weightOScaleW, {1, h});
            res = Mul(res, weightOScaleW2Dim);   // (1, h)  // 224个
            Tensor bmm5Res = Cast(res, DataType::DT_FP16, CAST_RINT);
            auto postOutTmp = Reshape(bmm5Res, {tileB, s, h});

            std::vector<SymbolicScalar> dynOffset = {bIdx * tileB, 0, 0};
            Assemble(postOutTmp, dynOffset, postOut);
        }
    }
}

} // namespace npu::tile_fwk
