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
 * \file r2_selected_attention.cpp
 * \brief
 */

#include "gather_selected_attention.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
void SelectedAttentionComputeV2(const Tensor &qNope, const Tensor &qRope, const Tensor &kNope2D, const Tensor &kRope2D,
    const Tensor &kNopeScales, const Tensor &topKIndcies, const Tensor &blockTable, const Tensor &kvSlcActSeqs, const int nQ, const int nKv,
    const float softmaxScale, const int topk, const int blockSize, const int maxBlockNumPerBatch, Tensor &attentionOut, SaTileShapeConfig tileConfig) {
    auto dtype = qNope.GetStorage()->Datatype();
    auto knDtype = kNope2D.GetStorage()->Datatype();
    int dN = qNope.GetStorage()->shape[SHAPE_DIM1];
    int dR = qRope.GetStorage()->shape[SHAPE_DIM1];
    int group = nQ / nKv;
    int gTile = tileConfig.gTile;
    int s2Tile = tileConfig.sKvTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    SymbolicScalar n2Sym = nKv; // n2
    SymbolicScalar batchSizeSym = GetInputShape(kvSlcActSeqs, 0); // b
    SymbolicScalar s1N1GSym = GetInputShape(qNope, 0) / batchSizeSym; // s1n1
    SymbolicScalar s1Sym = s1N1GSym / nQ; // s1
    SymbolicScalar s1S2Sym = s1Sym * topk; // s1s2
    SymbolicScalar gLoopSym = group / gTile;
    SymbolicScalar s2Sym = s1S2Sym / s1Sym; // s2

    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true) {
        SymbolicScalar curKvSlcSeq = GetTensorData(kvSlcActSeqs, {bIdx});
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) {
            SymbolicScalar curSeq = std::min(std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0), topk); // for MTP s1!= 1 casual计算, 并且与topk取min
            curSeq.AsIntermediateVariable();
            SymbolicScalar bnPerBatch = (curSeq + s2Tile - 1) / s2Tile;
            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景
                LOOP("LOOP_L3_g_SA", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1)) {
                    int curGTile = gTile;
                    Tensor oiUpdate(DT_FP32, {curGTile, dN}, "oiUpdate");
                    Tensor liUpdate(DT_FP32, {1, curGTile}, "liUpdate");
                    Tensor miUpdate(DT_FP32, {1, curGTile}, "miUpdate");
                    SymbolicScalar curOffset = bIdx * s1N1GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                    std::vector<SymbolicScalar> oiOffset = {bIdx, s1Idx, n2Idx * group + gIdx * curGTile, 0}; // 按最终结果(B,S1,N1,D)进行assemble
                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                        int curS2Tile = s2Tile;
                        SymbolicScalar curKvOffset = bIdx * s1S2Sym + s1Idx * s2Sym + s2Idx * curS2Tile;
                        // C1
                        config::SetSemanticLabel("Sa_QkMM");
                        auto qn = View(qNope, {curGTile, dN}, {curGTile, dN}, {curOffset, 0});
                        auto qr = View(qRope, {curGTile, dR}, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        Assemble(qn, {0, 0}, qi);
                        Assemble(qr, {0, dN}, qi);

                        SymbolicScalar bS1Idx = bIdx * s1Sym + s1Idx;
                        auto curTopKIndcies = View(topKIndcies, {1, curS2Tile}, {1, std::min(curSeq - s2Idx * curS2Tile, curS2Tile)}, {bS1Idx, s2Idx * curS2Tile});
                        auto curBlockTable = View(blockTable, {1, maxBlockNumPerBatch}, {bIdx, 0});
                        Tensor kn(dtype, {s2Tile, dN}, "kn");
                        if (knDtype == DataType::DT_INT8) {
                            // v0
                            TileShape::Current().SetVecTile(NUM_32, NUM_512);
                            // Gather tokNopeScales
                            auto kNopeScalesView = View(kNopeScales, {curS2Tile, 4}, { std::min(curSeq - s2Idx * curS2Tile, curS2Tile), 4}, {0, 0});
                            auto knScale = experimental::GatherInUB(kNopeScalesView, curTopKIndcies, curBlockTable, blockSize, -2);
                            // Gather kNope2D
                            auto kNope2DView = View(kNope2D, {curS2Tile, dN}, { std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN}, {0, 0});
                            auto knQuant = experimental::GatherInUB(kNope2DView, curTopKIndcies, curBlockTable, blockSize, -2);
                            auto knQuantF16 = Cast(knQuant, DT_FP16);
                            auto knQuantF32 = Cast(knQuantF16, DT_FP32);
                            // dequant
                            auto knQuantF32Tmp = Reshape(knQuantF32, {s2Tile*4, 128});  // 基于MLA prolog内krcache量化逻辑
                            auto knScaleTmp = Reshape(knScale, {s2Tile*4, 1});
                            TileShape::Current().SetVecTile(NUM_128, NUM_128);
                            auto knF32 = Mul(knQuantF32Tmp, knScaleTmp);
                            auto knF32Reshape = Reshape(knF32, {s2Tile, dN});
                            TileShape::Current().SetVecTile(NUM_32, NUM_512);
                            auto curKnFp32 = View(knF32Reshape, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN}, {0, 0});
                            kn = Cast(curKnFp32, dtype);
                        } else {
                            TileShape::Current().SetCubeTile({c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);
                            kn = experimental::GatherInL1<true, true>(kNope2D, curTopKIndcies, curBlockTable, blockSize, dN);
                        }
                        // C1
                        TileShape::Current().SetCubeTile({c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);
                        auto kr = experimental::GatherInL1<true, true>(kRope2D, curTopKIndcies, curBlockTable, blockSize, dR);
                        Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                        Assemble(kn, {0, 0}, kj);
                        Assemble(kr, {0, dN}, kj);
                        auto kjView = View(kj, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR}, {0, 0});
                        auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kjView, false, true);

                        // V1
                        config::SetSemanticLabel("Sa_Qkvec1");
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale));
                        auto tildaMijReduce = Amax(sijScale, -1, true); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tildaMij = Reshape(tildaMijReduce, {1, curGTile}); // (1, curGTile)
                        tildaMij.SetName("tildaMij");
                        auto tsub = Sub(sijScale, tildaMijReduce); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaPijF16 = Cast(tildaPij, dtype);
                        auto tildaLijReduce = Sum(tildaPij, -1, true); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tildaLij = Reshape(tildaLijReduce, {1, curGTile}); // (1, curGTile)
                        tildaLij.SetName("tildaLij");
                        config::SetSemanticLabel("Sa_KvMm");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, false);
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], kn.GetShape()[1]});

                        // MQA场景: v=kn
                        Tensor q1(dtype, {curGTile, dN});
                        if (knDtype == DataType::DT_INT8) {
                            auto vj = View(kn, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN}, {0, 0});
                            q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);
                        } else {
                            auto vj = experimental::GatherInL1<true, false>(kNope2D, curTopKIndcies, curBlockTable, blockSize, dN);
                            q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);
                        }
                        IF (IsLoopBegin(s2Idx, 0)) {
                            auto oiTmp = q1;
                            // C2
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH3
                                // V2
                                config::SetSemanticLabel("Sa_KvVec2");
                                oiUpdate = Div(oiTmp, tildaLijReduce);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = Cast(Reshape(oiUpdate, {1, 1, curGTile, dN}), dtype);
                                Assemble(oiUpdate4Dim, oiOffset, attentionOut);
                            } ELSE { // PATH2
                                oiUpdate = oiTmp;
                            }
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            liUpdate = tildaLij;
                            miUpdate = tildaMij;
                        }
                        ELSE {
                            config::SetSemanticLabel("Sa_UpdateVec2");
                            auto oi = oiUpdate;
                            auto li = liUpdate;
                            auto mi = miUpdate;
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            auto miNew = Maximum(mi, tildaMij); // (1, curGTile)
                            auto t1 = Sub(mi, miNew); // (1, curGTile)
                            auto t2 = Exp(t1); // (1, curGTile)
                            auto t3 = Sub(tildaMij, miNew); // (1, curGTile)
                            auto t4 = Exp(t3); // (1, curGTile)
                            auto t5 = Mul(t4, tildaLij); // (1, curGTile)
                            auto t6 = Mul(t2, li); // (1, curGTile)
                            auto liNew = Add(t6, t5); // (1, curGTile)
                            auto q3 = Mul(oi, Reshape(t2, {curGTile, 1}));
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            auto q2 = Mul(q1, Reshape(t4, {curGTile, 1}));
                            auto oiTmp = Add(q3, q2);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH1
                                oiUpdate = Div(oiTmp, Reshape(liNew, {curGTile, 1}));
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = Cast(Reshape(oiUpdate, {1, 1, curGTile, dN}), dtype);
                                Assemble(oiUpdate4Dim, oiOffset, attentionOut);
                            } ELSE { // PATH0
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

void SelectedAttentionV2(const Tensor &qNope, const Tensor &qRope, const Tensor &kNope2D, const Tensor &kRope2D,
    const Tensor &kNopeScales, const Tensor &topKIndcies, const Tensor &blockTable, const Tensor &kvSlcActSeqs, const int nQ, const int nKv,
    const float softmaxScale, const int topk, const int blockSize, const int maxBlockNumPerBatch, Tensor &attentionOut, SaTileShapeConfig tileConfig) {
    FUNCTION("R2_SA_MAIN_V2", {qNope, qRope, kNope2D, kRope2D, kNopeScales, topKIndcies, blockTable, kvSlcActSeqs}, {attentionOut}) {
        SelectedAttentionComputeV2(qNope, qRope, kNope2D, kRope2D, kNopeScales, topKIndcies, blockTable, kvSlcActSeqs,
                                nQ, nKv, softmaxScale, topk, blockSize, maxBlockNumPerBatch, attentionOut, tileConfig);
    }
}
} // namespace npu::tile_fwk
