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
 * \file page_attention.cpp
 * \brief
 */

#include "page_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void PageAttention(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes, bool isNzFormat) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        SymbolicScalar batchSize = blockTable.GetShape()[0];
        SymbolicScalar nQ = qNope.GetShape()[0] / batchSize;
        SymbolicScalar nLoop = nQ / nTile;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
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

                    TileOpFormat kjFormat = isNzFormat ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj", kjFormat);
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    kj = View(kj, {curS2Tile, dN + dR}, {std::min(curSeq - bn * blockSize, blockSize), dR + dN}, {0, 0});
                    auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    config::SetSemanticLabel("MatMul");
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    sij.SetName("sij");
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);

                    config::SetSemanticLabel("SoftMax");
                    auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale)); // (curNTile, curS2Tile)

                    config::SetSemanticLabel("SoftMax");
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
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        config::SetSemanticLabel("b1-after-matmul2");
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            config::SetSemanticLabel("b1-after-matmul2");
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
                        config::SetSemanticLabel("bn-matmul2");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16,
                            vj); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
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
    }
}

// this case only for tensor graph loop unroll test
void PageAttentionWithImmScalar(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    std::vector<std::vector<int>>  &blockTable, std::vector<int> &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes, bool isNzFormat) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache}, {attentionOut}) {
        int batchSize((int64_t)blockTable.size());
        SymbolicScalar nQ = qNope.GetShape()[0] / batchSize;
        SymbolicScalar nLoop = nQ / nTile;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq(static_cast<int64_t>(actSeqs[0])); // diff batch seq all same
            SymbolicScalar bnPerBatch = (curSeq + blockSize - 1) / blockSize;
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1), PowersOf2(maxUnrollTimes)) {
                    int curS2Tile = blockSize;
                    auto qn = View(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = View(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    Assemble(qn, {0, 0}, qi);
                    Assemble(qr, {0, dN}, qi);

                    // Force set to 0 for test only
                    SymbolicScalar curBlockIdx(0);
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = View(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = View(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});

                    TileOpFormat kjFormat = isNzFormat ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj", kjFormat);
                    Assemble(kn, {0, 0}, kj);
                    Assemble(kr, {0, dN}, kj);
                    kj = View(kj, {curS2Tile, dN + dR}, {std::min(curSeq - bn * blockSize, blockSize), dR + dN}, {0, 0});
                    auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});

                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    sij.SetName("sij");
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);

                    auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale)); // (curNTile, curS2Tile)

                    auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF (bn == 0) {
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF (bn == bnPerBatch - 1) {
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
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16,
                            vj); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (bn == bnPerBatch - 1) {
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
    }
}

void PageAttentionWithManualUnroll(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes) {
    auto dtype = qNope.GetStorage()->Datatype();
    // 入参B*S*N合轴
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];

    int nTile = tileConfig.headNumQTile;
    auto v0Tile = tileConfig.v0TileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    int div2 = 2;

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        SymbolicScalar batchSize = blockTable.GetShape()[0];
        SymbolicScalar nQ = qNope.GetShape()[0] / batchSize;
        SymbolicScalar nLoop = nQ / nTile;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batchSize)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(nLoop)) {
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(bnPerBatch), PowersOf2(maxUnrollTimes)) {
                    for (int unrollTimes = maxUnrollTimes; unrollTimes != 0; unrollTimes /= div2) {
                        UNROLL(unrollTimes) {
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto qn = View(qNope, {nTile, dN}, {curOffset, 0});
                            auto qr = View(qRope, {nTile, dR}, {curOffset, 0});
                            auto qi = Cat({qn, qr}, 1); // (nTileCur, dN+dR)
                            std::vector<Tensor> subKns;
                            std::vector<Tensor> subKrs;
                            std::vector<Tensor> subVjs;
                            for (int idxOffset = 0; idxOffset < unrollTimes; idxOffset++) {
                                auto curBlockIdx = GetTensorData(blockTable, {bIdx, bn + idxOffset});
                                subKns.emplace_back(
                                    View(kNopeCache, {blockSize, dN}, {curBlockIdx * blockSize, 0}));
                                subKrs.emplace_back(
                                    View(kRopeCache, {blockSize, dR}, {curBlockIdx * blockSize, 0}));
                                subVjs.emplace_back(
                                    View(vNopeCache, {blockSize, dN}, {curBlockIdx * blockSize, 0}));
                            }

                            auto kn = Cat(subKns, 0);
                            auto kr = Cat(subKrs, 0);
                            auto kj = Cat({kn, kr}, 1); // (s2TileCur, dN+dR)
                            auto vj = Cat(subVjs, 0);

                            TileShape::Current().SetCubeTile(
                                {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});

                            auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);
                            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                            auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale)); // (nTileCur, s2TileCur)

                            auto tildaMij = Amax(sijScale, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)
                            auto tsub = Sub(sijScale,
                                tildaMij); // (nTileCur, s2TileCur) - (nTileCur, 1) -> (nTileCur, s2TileCur)
                            auto tildaPij = Exp(tsub);
                            auto tildaPijF16 = Cast(tildaPij, dtype);
                            auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                            IF(IsLoopBegin(bn, 0)) {
                                TileShape::Current().SetCubeTile(
                                    {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                                auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                                TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                                IF(IsLoopEnd(bn, bnPerBatch)) {
                                    oiUpdate =
                                        Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
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
                                auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                                TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                                auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                                auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                                IF(IsLoopEnd(bn, bnPerBatch)) {
                                    oiUpdate =
                                        Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
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
            }
        }
    }
}

void PageAttentionHighThroughput(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut,
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

    FUNCTION("main",
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        SymbolicScalar batchSize = blockTable.GetShape()[0];
        SymbolicScalar nQ = qNope.GetShape()[0] / batchSize;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1), PowersOf2(maxUnrollTimes)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {bIdx});
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();

            int curNTile = nTile;
            Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
            // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的Assemble操作
            SymbolicScalar curOffset = bIdx * nQ;
            std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

            // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
            int curS2Tile = blockSize;
            auto qn = View(qNope, {curNTile, dN}, {curOffset, 0});
            auto qr = View(qRope, {curNTile, dR}, {curOffset, 0});
            Tensor qi(dtype, {curNTile, dN + dR}, "qi");
            Assemble(qn, {0, 0}, qi);
            Assemble(qr, {0, dN}, qi);

            SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, 0});
            curBlockIdx.AsIntermediateVariable();
            auto kn = View(kNopeCache, {curS2Tile, dN}, {std::min(curSeq, blockSize), dN},
                {curBlockIdx * blockSize, 0});
            auto kr = View(kRopeCache, {curS2Tile, dR}, {std::min(curSeq, blockSize), dR},
                {curBlockIdx * blockSize, 0});
            Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
            Assemble(kn, {0, 0}, kj);
            Assemble(kr, {0, dN}, kj);
            auto vj = View(vNopeCache, {curS2Tile, dN}, {std::min(curSeq, blockSize), dN},
                {curBlockIdx * blockSize, 0});

            TileShape::Current().SetCubeTile({c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
            auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
            auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale)); // (curNTile, curS2Tile)

            auto tildaMij = Amax(sijScale, -1, true); // (curNTile, curS2Tile) -> (curNTile, 1)
            auto tsub =
                Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
            auto tildaPij = Exp(tsub);
            auto tildaPijF16 = Cast(tildaPij, dtype);
            auto tildaLij = Sum(tildaPij, -1, true); // (nTileCur, s2TileCur) -> (nTileCur, 1)

            TileShape::Current().SetCubeTile({c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
            auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
            Assemble(oiUpdate, oiOffset, attentionOut);
        }
    }
}
} // namespace npu::tile_fwk
