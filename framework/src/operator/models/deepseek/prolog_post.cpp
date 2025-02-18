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
 * \file prolog_post.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void PrologPost(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, Tensor &weightUV, Tensor &weightO, int blockSize, float softmaxScale,
    Tensor &postOut, PaTileShapeConfig &tileConfig) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int sQ = 1;
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];
    int tile4 = 4;

    int nTile = tileConfig.headNumQTile;
    int vHeadDim = weightUV.GetShape()[2];  // (nQ, dN, vHeadDim)
    int hiddenSize = weightO.GetShape()[1]; // (nQ*VHeadDim, H)

    auto v0Tile = tileConfig.v0TileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    int batchSize = blockTable.GetShape()[0];
    int nQ = qNope.GetShape()[0] / batchSize; // B*1*N
    int nLoop = nQ / nTile;

    Tensor attentionOut(DT_FP32, qNope.GetShape(), "attentionOut");

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, weightUV, weightO}, {postOut}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batchSize)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            // nLoop是因为B*N*S合轴，计算N时是SymbolicScalar计算，此处不一定需要用Loop
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(nLoop)) {
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(bnPerBatch)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                    auto qn = View(qNope, {nTile, dN}, {curOffset, 0});
                    auto qr = View(qRope, {nTile, dR}, {curOffset, 0});
                    auto qi = Cat({qn, qr}, 1); // (nTileCur, dN+dR)
                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});

                    auto kn = View(kNopeCache, {blockSize, dN}, {curBlockIdx * blockSize, 0});
                    auto kr = View(kRopeCache, {blockSize, dR}, {curBlockIdx * blockSize, 0});
                    auto kj = Cat({kn, kr}, 1); // (s2TileCur, dN+dR)
                    auto vj = View(vNopeCache, {blockSize, dN}, {curBlockIdx * blockSize, 0});
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});

                    auto sij =
                        Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (nTileCur, dN+dR), (s2TileCur, dN+dR) -> (nTileCur, s2TileCur)
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                    auto sijScale = Mul(sij, Element(DataType::DT_FP32, softmaxScale)); // (nTileCur, s2TileCur)

                    auto tildaMij = Amax(sijScale, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (nTileCur, s2TileCur) - (nTileCur, 1) -> (nTileCur, s2TileCur)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF(bn == 0) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32,
                            tildaPijF16, vj); // (nTileCur, s2TileCur), (s2TileCur, dN) -> (nTileCur, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF(bnPerBatch == 1) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                        }
                        ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    }
                    ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        auto miNew = Maximum(mi, tildaMij); // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                        auto t1 = Sub(mi, miNew);           // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew); // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij); // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                        auto t6 = Mul(t2, li);       // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                        auto liNew = Add(t6, t5);    // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)

                        auto q3 = Mul(oi, t2); // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32,
                            tildaPijF16, vj); // (nTileCur, s2TileCur), (s2TileCur, dN) -> (nTileCur, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF(bn == bnPerBatch - 1) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                        }
                        ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                    Assemble(oiUpdate, oiOffset, attentionOut);
                }
            }
        }

        config::SetBuildStatic(true);
        FUNCTION("PaPost") {
            TileShape::Current().SetVecTile({32, dN});
            auto attenRes = Reshape(attentionOut, {batchSize, nQ, dN}); // (b*sQ*nQ, dN), sQ=1

            TileShape::Current().SetVecTile({2, 16, dN});
            auto castOut = Cast(attenRes, dtype);         // (b*sQ, nQ, dN)
            auto attenTrans = Transpose(castOut, {0, 1}); // (b*sQ, nQ, dN) -> (nQ, b*sQ, dN)

            // Bmm的Tile设置也是M, K, N, 与Batch无关；注意，MM的维度不足16时要按照16对齐设置
            TileShape::Current().SetCubeTile({16, 16}, {dN, dN}, {vHeadDim, vHeadDim});
            auto bmmRes = Matrix::BatchMatmul<false, false>(DataType::DT_FP32,
                attenTrans, weightUV); // (nQ, b*sQ, dN) * (nQ, dN, vHeadDim) -> (nQ, b*sQ, vHeadDim)

            // cast不支持跳写，这个Transpose必须与上面的tileshape后两维一致； 2、Transpose尾轴不能切
            TileShape::Current().SetVecTile(1, tile4, vHeadDim);
            auto bmmTrans = Transpose(bmmRes, {0, 1}); // (nQ, b*sQ, vHeadDim) -> (b*sQ, nQ, vHeadDim)

            TileShape::Current().SetVecTile({1, nQ, vHeadDim});
            auto bmmReshape =
                Reshape(bmmTrans, {batchSize * sQ, nQ * vHeadDim}); // (b*sQ, nQ, vHeadDim) -> (b*sQ, nQ*vHeadDim)

            TileShape::Current().SetCubeTile({16, 16}, {32, 32}, {hiddenSize, hiddenSize});
            Tensor postMm =
                Matrix::Matmul<false, false>(DataType::DT_FP32, bmmReshape, weightO); // (b*sQ, nQ*vHeadDim) * (nQ*VHeadDim, H) -> (b*sQ, H)

            TileShape::Current().SetVecTile({batchSize * sQ, 32});
            postOut = Reshape(postMm, {batchSize, sQ, hiddenSize});
        }
    }
}

void PageAttentionAddS(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
        Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    int batchSize = blockTable.GetShape()[0];
    int nQ = qNope.GetShape()[0] / batchSize; // B*1*N

    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut, postOut}) {
        SymbolicScalar nLoop = nQ / nTile;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1), PowersOf2(maxUnrollTimes)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = View(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = View(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    Assemble(qn, {0, 0}, qi);
                    Assemble(qr, {0, dN}, qi);

                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = View(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = View(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    config::SetSemanticLabel("MatMul");
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);

                    config::SetSemanticLabel("SoftMax");
                    auto sijScale = Mul(sij, Element(DataType::DT_FP32, softmaxScale)); // (curNTile, curS2Tile)

                    auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF (IsLoopBegin(bn, 0)) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        config::SetSemanticLabel("b1-matmul2");
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        config::SetSemanticLabel("b1-after-matmul2");
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        config::SetSemanticLabel("Softmax-acc");
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
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        config::SetSemanticLabel("bn-matmul2");
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        config::SetSemanticLabel("bn-after-matmul2");
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }

        SymbolicScalar B = attentionOut.GetShape()[0] / N; // S=1
        const int bTile = 32;
        LOOP("PaPost", FunctionType::DYNAMIC_LOOP, papostiter, LoopRange(0, B / bTile, 1), {}, true) {
                auto postInUnit = View(attentionOut, {bTile * S * N, kvLoraRank}, {papostiter * bTile * S * N, 0});
                TileShape::Current().SetVecTile({std::min(64, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

                // 使用AddS看能否进行LooP间数据传递
                auto t1Res = Add(postInUnit, Element(DataType::DT_FP32, F_0));

                std::vector<SymbolicScalar> dynOffset = {papostiter * bTile * S * N, 0};
                Assemble(t1Res, dynOffset, postOut);
        }
    }
}

void PageAttentionAddSSingleOutput(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
        Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    int batchSize = blockTable.GetShape()[0];
    int nQ = qNope.GetShape()[0] / batchSize; // B*1*N

    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {postOut}) {
        SymbolicScalar nLoop = nQ / nTile;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1), PowersOf2(maxUnrollTimes)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = View(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = View(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    Assemble(qn, {0, 0}, qi);
                    Assemble(qr, {0, dN}, qi);

                    SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, bn});
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = View(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = View(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    config::SetSemanticLabel("MatMul");
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);

                    config::SetSemanticLabel("SoftMax");
                    auto sijScale = Mul(sij, Element(DataType::DT_FP32, softmaxScale)); // (curNTile, curS2Tile)

                    auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF (IsLoopBegin(bn, 0)) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        config::SetSemanticLabel("b1-matmul2");
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        config::SetSemanticLabel("b1-after-matmul2");
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;

                        config::SetSemanticLabel("Softmax-acc");
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
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        config::SetSemanticLabel("bn-matmul2");
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        config::SetSemanticLabel("bn-after-matmul2");
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            Assemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }

        SymbolicScalar B = attentionOut.GetShape()[0] / N; // S=1
        const int bTile = 32;
        LOOP("PaPost", FunctionType::DYNAMIC_LOOP, papostiter, LoopRange(0, B / bTile, 1), {}, true) {
                auto postInUnit = View(attentionOut, {bTile * S * N, kvLoraRank}, {papostiter * bTile * S * N, 0});
                TileShape::Current().SetVecTile({std::min(64, bTile * S * N), kvLoraRank}); // raw (8*1*128, 512)

                // 使用AddS看能否进行LooP间数据传递
                auto t1Res = Add(postInUnit, Element(DataType::DT_FP32, F_0));

                std::vector<SymbolicScalar> dynOffset = {papostiter * bTile * S * N, 0};
                Assemble(t1Res, dynOffset, postOut);
        }
    }
}

} // namespace npu::tile_fwk
