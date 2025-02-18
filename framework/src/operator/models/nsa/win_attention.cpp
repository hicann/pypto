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
 * \file win_attention.cpp
 * \brief
 */

#include "interface/inner/tilefwk.h"
#include "win_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void WinAttentionCompute(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    auto dtype = qNope.GetStorage()->Datatype();

    // 入参B*S*N合轴
    int dNopeSize = qNope.GetShape()[1];
    int dRopeSize = qRope.GetShape()[1];
    ASSERT(nKv != 0) << "nKv cant't be zero!";
    auto gGroup = nQ / nKv;
    int gTile = tileConfig.gTile; // 128
    ASSERT(blockSize != 0) << "blockSize can't be zero!";

    auto nopeTile = tileConfig.vNopeTileShape;
    auto ropeTile = tileConfig.vRopeTileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    // loop config
    SymbolicScalar bSize = blockTable.GetShape()[0];
    SymbolicScalar bTile = 1;
    ASSERT(bTile != 0) << "bTile can't be zero!";
    ASSERT(nQ != 0) << "nQ can't be zero!";
    SymbolicScalar bLoop = bSize / bTile;
    SymbolicScalar s1Size = qNope.GetShape()[0] / bSize / nQ; // [B_s1_N1, D]
    SymbolicScalar s1Tile = 1;
    ASSERT(s1Tile != 0) << "s1Tile can't be zero!";
    SymbolicScalar s1Loop = s1Size / s1Tile;
    SymbolicScalar n2Tile = 1;
    ASSERT(n2Tile != 0) << "n2Tile can't be zero!";
    SymbolicScalar n2Loop = nKv / n2Tile;
    ASSERT(gTile != 0) << "gTile can't be zero!";
    SymbolicScalar gLoop = gGroup / gTile;

    // block tile config
    SymbolicScalar blockStartIndex = 0;
    SymbolicScalar blockStartOffset = 0;
    SymbolicScalar blockEndIndex = 0;
    SymbolicScalar winActualSize = 0;
    SymbolicScalar tableLoop = 0;

    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar curActualSeqSize = GetTensorData(actSeqs, {bIdx});
        LOOP("LOOP_L1_s1Idx", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Loop, 1)) {
            winActualSize = std::min(windowSize, (curActualSeqSize - s1Size + s1Idx + 1));
            blockEndIndex = (curActualSeqSize + blockSize - 1) / blockSize - 1;
            blockStartIndex = std::max(0, ((curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) / blockSize));
            blockStartOffset = (curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) % blockSize;
            tableLoop = blockEndIndex - blockStartIndex + 1;
            LOOP("LOOP_L2_n2Idx", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Loop, 1)) {
                LOOP("LOOP_L2_gIdx", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoop, 1)) {
                    // flash update
                    std::vector<SymbolicScalar> oiOffset = {bIdx , s1Idx , n2Idx * gGroup + gIdx * gTile, 0};
                    SymbolicScalar curOffset = bIdx * s1Size * nQ + s1Idx * nQ + n2Idx * gGroup + gIdx * gTile;

                    Tensor kPart(dtype, {NUM_9 * blockSize, (dNopeSize + dRopeSize)}, "kPart");
                    for(auto tIdx = 0; tIdx < NUM_9; tIdx++) {
                        SymbolicScalar curidx = blockStartIndex + tIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, curidx});

                        TileShape::Current().SetVecTile(nopeTile[0], nopeTile[1]);
                        auto kNope = View(vNopeCache, {blockSize, dNopeSize}, {curBlockIdx * blockSize, n2Idx * dNopeSize});
                        auto tmpK1 = Cast(kNope, DataType::DT_FP32);
                        auto tmpK2 = Cast(tmpK1, dtype);
                        Assemble(tmpK2, {tIdx * blockSize, 0}, kPart);

                        TileShape::Current().SetVecTile(ropeTile[0], ropeTile[1]);
                        auto kRope = View(kRopeCache, {blockSize, dRopeSize}, {curBlockIdx * blockSize, n2Idx * dRopeSize});
                        auto tmpKR1 = Cast(kRope, DataType::DT_FP32);
                        auto tmpKR2 = Cast(tmpKR1, dtype);
                        Assemble(tmpKR2, {tIdx * blockSize, dNopeSize}, kPart);
                    }

                    auto startOffset = blockStartOffset;
                    auto kActualPart = View(kPart, {windowSize, dNopeSize + dRopeSize},
                        {winActualSize, dNopeSize + dRopeSize}, {startOffset, 0});
                    auto vActualPart = View(kPart, {windowSize, dNopeSize}, {winActualSize, dNopeSize},
                        {startOffset, 0});
                    // query
                    Tensor qPart(dtype, {gTile, dNopeSize + dRopeSize}, "qPart");
                    auto qNopeL = View(qNope, {gTile, dNopeSize}, {gTile, dNopeSize}, {curOffset, 0});
                    Assemble(qNopeL, {0, 0}, qPart);
                    auto qRopeR = View(qRope, {gTile, dRopeSize}, {gTile, dRopeSize}, {curOffset, 0});
                    Assemble(qRopeR, {0, dNopeSize}, qPart);

                    // matmul_1
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                    auto qKT = Matrix::Matmul<false, true>(DataType::DT_FP32, qPart, kActualPart);
                    TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                    auto qKTScale = Mul(qKT, Element(qKT.GetStorage()->Datatype(), softmaxScale));

                    // softmax
                    auto tileMax = Amax(qKTScale, -1, true); // max
                    auto tileSub = Sub(qKTScale, tileMax); // sub max
                    auto tileExp = Exp(tileSub); // exp
                    auto tileExpF16 = Cast(tileExp, dtype);
                    auto tileSum = Sum(tileExp, -1, true);

                    // matmul_2
                    TileShape::Current().SetCubeTile(
                        {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                    auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tileExpF16, vActualPart);
                    TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                    // reshape and copyOut
                    auto out = Div(oiTmp, tileSum);
                    TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                    auto outFinal = Add(Reshape(out, {bTile, s1Tile, gTile, dNopeSize}),
                        Element(out.GetStorage()->Datatype(), float(0)));
                    Assemble(outFinal, oiOffset, attentionOut);
                }
            }
        }
    }
}

void WinAttentionComputeFlash(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    auto dtype = qNope.GetStorage()->Datatype();

    // 入参B*S*N合轴
    int dNopeSize = qNope.GetShape()[1];
    int dRopeSize = qRope.GetShape()[1];
    ASSERT(nKv != 0) << "nKv cant't be zero!";
    auto gGroup = nQ / nKv;
    int gTile = tileConfig.gTile; // 128
    int s2Tile = tileConfig.skvTile; // 1、skvTile == windowsize:非flash；2、skvTile < windowsize:flash
    ASSERT(blockSize != 0) << "blockSize can't be zero!";

    auto nopeTile = tileConfig.vNopeTileShape;
    auto ropeTile = tileConfig.vRopeTileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    // loop config
    SymbolicScalar bSize = blockTable.GetShape()[0];
    SymbolicScalar bTile = 1;
    ASSERT(bTile != 0) << "bTile can't be zero!";
    ASSERT(nQ != 0) << "nQ can't be zero!";
    SymbolicScalar bLoop = bSize / bTile;
    SymbolicScalar s1Size = qNope.GetShape()[0] / bSize / nQ; // [B_s1_N1, D]
    SymbolicScalar s1Tile = 1;
    ASSERT(s1Tile != 0) << "s1Tile can't be zero!";
    SymbolicScalar s1Loop = s1Size / s1Tile;
    SymbolicScalar n2Tile = 1;
    ASSERT(n2Tile != 0) << "n2Tile can't be zero!";
    SymbolicScalar n2Loop = nKv / n2Tile;
    ASSERT(gTile != 0) << "gTile can't be zero!";
    SymbolicScalar gLoop = gGroup / gTile;

    // block tile config
    SymbolicScalar blockStartIndex = 0;
    SymbolicScalar blockStartOffset = 0;
    SymbolicScalar blockEndIndex = 0;
    SymbolicScalar winActualSize = 0;
    SymbolicScalar tableLoop = 0;

    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar curActualSeqSize = GetTensorData(actSeqs, {bIdx});
        Tensor kPart(dtype, {bSize * s1Size * NUM_9 * blockSize, (dNopeSize + dRopeSize)}, "kPart");
        LOOP("LOOP_L1_s1Idx", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Loop, 1)) {
            winActualSize = std::min(windowSize, (curActualSeqSize - s1Size + s1Idx + 1));
            blockEndIndex = (curActualSeqSize + blockSize - 1) / blockSize - 1;
            blockStartIndex = std::max(0, ((curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) / blockSize));
            blockStartOffset = (curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) % blockSize;
            tableLoop = blockEndIndex - blockStartIndex + 1;
            LOOP("LOOP_L2_n2Idx", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Loop, 1)) {
                LOOP("LOOP_L2_gIdx", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoop, 1)) {
                    // flash update
                    std::vector<SymbolicScalar> oiOffset = {bIdx , s1Idx , n2Idx * gGroup + gIdx * gTile, 0};
                    SymbolicScalar curOffset = bIdx * s1Size * nQ + s1Idx * nQ + n2Idx * gGroup + gIdx * gTile;
                    SymbolicScalar s2Loop = (winActualSize + s2Tile - 1) / s2Tile;

                    Tensor oiUpdate(DT_FP32, {gTile, dNopeSize}, "oiUpdate");
                    Tensor liUpdate(DT_FP32, {gTile, 1}, "liUpdate");
                    Tensor miUpdate(DT_FP32, {gTile, 1}, "miUpdate");

                    SymbolicScalar kvTensorIdx = (bIdx * s1Size + s1Idx) * NUM_9 * blockSize;
                    LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx, LoopRange(0, tableLoop, 1)) {
                        SymbolicScalar curidx = blockStartIndex + tIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, curidx});

                        TileShape::Current().SetVecTile(nopeTile[0], nopeTile[1]);
                        auto kNope = View(vNopeCache, {blockSize, dNopeSize}, {curBlockIdx * blockSize, n2Idx * dNopeSize});
                        auto tmpK1 = Cast(kNope, DataType::DT_FP32);
                        auto tmpK2 = Cast(tmpK1, dtype);
                        Assemble(tmpK2, {kvTensorIdx + tIdx * blockSize, 0}, kPart);

                        TileShape::Current().SetVecTile(ropeTile[0], ropeTile[1]);
                        auto kRope = View(kRopeCache, {blockSize, dRopeSize}, {curBlockIdx * blockSize, n2Idx * dRopeSize});
                        auto tmpKR1 = Cast(kRope, DataType::DT_FP32);
                        auto tmpKR2 = Cast(tmpKR1, dtype);
                        Assemble(tmpKR2, {kvTensorIdx + tIdx * blockSize, dNopeSize}, kPart);
                    }

                    LOOP("LOOP_L2_s2Idx", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, s2Loop, 1)) {
                        auto startOffset = blockStartOffset + s2Idx * s2Tile + kvTensorIdx;
                        auto kActualPart = View(kPart, {s2Tile, dNopeSize + dRopeSize},
                            {std::min(winActualSize - s2Idx * s2Tile, s2Tile), dNopeSize + dRopeSize},
                            {startOffset, 0});
                        auto vActualPart = View(kPart, {s2Tile, dNopeSize},
                            {std::min(winActualSize - s2Idx * s2Tile, s2Tile), dNopeSize}, {startOffset, 0});
                        // query
                        Tensor qPart(dtype, {gTile, dNopeSize + dRopeSize}, "qPart");
                        auto qNopeL = View(qNope, {gTile, dNopeSize}, {gTile, dNopeSize}, {curOffset, 0});
                        Assemble(qNopeL, {0, 0}, qPart);
                        auto qRopeR = View(qRope, {gTile, dRopeSize}, {gTile, dRopeSize}, {curOffset, 0});
                        Assemble(qRopeR, {0, dNopeSize}, qPart);

                        // matmul_1
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        auto qKT = Matrix::Matmul<false, true>(DataType::DT_FP32, qPart, kActualPart);
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto qKTScale = Mul(qKT, Element(qKT.GetStorage()->Datatype(), softmaxScale));

                        // softmax
                        auto tileMax = Amax(qKTScale, -1, true); // max
                        auto tileSub = Sub(qKTScale, tileMax); // sub max
                        auto tileExp = Exp(tileSub); // exp
                        auto tileExpF16 = Cast(tileExp, dtype);
                        auto tileSum = Sum(tileExp, -1, true);

                        IF (IsLoopBegin(s2Idx, 0)) {
                            // matmul_2
                            TileShape::Current().SetCubeTile(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tileExpF16, vActualPart);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            IF (IsLoopEnd(s2Idx, s2Loop)) {
                                // reshape and copyOut
                                oiUpdate = Div(oiTmp, tileSum);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto outFinal = Add(Reshape(oiUpdate, {bTile, s1Tile, gTile, dNopeSize}),
                                    Element(oiUpdate.GetStorage()->Datatype(), float(0)));
                                Assemble(outFinal, oiOffset, attentionOut);
                            } ELSE {
                                oiUpdate = oiTmp;
                            }
                            liUpdate = tileSum;
                            miUpdate = tileMax;
                        } ELSE {
                            auto oi = oiUpdate;
                            auto li = liUpdate;
                            auto mi = miUpdate;

                            auto miNew = Maximum(mi, tileMax);
                            auto t1 = Sub(mi, miNew);
                            auto t2 = Exp(t1);
                            auto t3 = Sub(tileMax, miNew);
                            auto t4 = Exp(t3);
                            auto t5 = Mul(t4, tileSum);
                            auto t6 = Mul(t2, li);
                            auto liNew = Add(t6, t5);

                            auto q3 = Mul(oi, t2);
                            TileShape::Current().SetCubeTile(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tileExpF16, vActualPart);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            auto q2 = Mul(q1, t4);
                            auto oiTmp = Add(q3, q2);
                            IF (IsLoopEnd(s2Idx, s2Loop)) { // PATH1
                                oiUpdate = Div(oiTmp, liNew);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto outFinal = Add(Reshape(oiUpdate, {bTile, s1Tile, gTile, dNopeSize}),
                                    Element(oiUpdate.GetStorage()->Datatype(), float(0)));
                                Assemble(outFinal, oiOffset, attentionOut);
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

void WinAttentionDebugCompute(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    auto dtype = qNope.GetStorage()->Datatype();

    // 入参B*S*N合轴
    int dNopeSize = qNope.GetShape()[1];
    int dRopeSize = qRope.GetShape()[1];
    ASSERT(nKv != 0) << "nKv cant't be zero!";
    auto gGroup = nQ / nKv;
    int gTile = tileConfig.gTile; // 128
    ASSERT(blockSize != 0) << "blockSize can't be zero!";

    auto nopeTile = tileConfig.vNopeTileShape;
    auto ropeTile = tileConfig.vRopeTileShape;
    auto outTile = tileConfig.outTileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    // loop config
    SymbolicScalar bSize = blockTable.GetShape()[0];
    SymbolicScalar bTile = 1;
    ASSERT(bTile != 0) << "bTile can't be zero!";
    ASSERT(nQ != 0) << "nQ can't be zero!";
    SymbolicScalar bLoop = bSize / bTile;
    SymbolicScalar s1Size = qNope.GetShape()[0] / bSize / nQ; // [B_s1_N1, D]
    SymbolicScalar s1Tile = 1;
    ASSERT(s1Tile != 0) << "s1Tile can't be zero!";
    SymbolicScalar s1Loop = s1Size / s1Tile;
    SymbolicScalar n2Tile = 1;
    ASSERT(n2Tile != 0) << "n2Tile can't be zero!";
    SymbolicScalar n2Loop = nKv / n2Tile;
    ASSERT(gTile != 0) << "gTile can't be zero!";
    SymbolicScalar gLoop = gGroup / gTile;
    // block tile config
    SymbolicScalar blockStartIndex = 0;
    SymbolicScalar blockStartOffset = 0;
    SymbolicScalar blockEndIndex = 0;
    SymbolicScalar winActualSize = 0;
    SymbolicScalar tableLoop = 0;

    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar curActualSeqSize = GetTensorData(actSeqs, {bIdx});
        LOOP("LOOP_L1_s1Idx", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Loop, 1)) {
            winActualSize = std::min(windowSize, (curActualSeqSize - s1Size + s1Idx + 1));
            blockEndIndex = (curActualSeqSize + blockSize - 1) / blockSize - 1;
            blockStartIndex = std::max(0, ((curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) / blockSize));
            blockStartOffset = (curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) % blockSize;
            tableLoop = blockEndIndex - blockStartIndex + 1;
            LOOP("LOOP_L2_n2Idx", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Loop, 1)) {
                LOOP("LOOP_L2_gIdx", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoop, 1)) {
                    std::vector<SymbolicScalar> outOffset = {bIdx , s1Idx , n2Idx * gGroup + gIdx * gTile, 0};
                    Tensor kPart(dtype, {5 * blockSize, (dNopeSize + dRopeSize)}, "kPart");
                    Tensor vPart(dtype, {5 * blockSize, dNopeSize}, "vPart");
                    LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx, LoopRange(0, tableLoop, 1), {}, true) {
                        SymbolicScalar curidx = blockStartIndex + tIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, curidx});

                        auto kNope = View(vNopeCache, {blockSize, dNopeSize}, {curBlockIdx * blockSize, n2Idx * dNopeSize});
                        TileShape::Current().SetVecTile(nopeTile[0], nopeTile[1]);
                        auto tmpK1 = Cast(kNope, DataType::DT_FP32);
                        auto tmpK2 = Cast(tmpK1, dtype);
                        Assemble(tmpK2, {tIdx * blockSize, 0}, kPart);

                        auto kRope = View(kRopeCache, {blockSize, dRopeSize}, {curBlockIdx * blockSize, n2Idx * dRopeSize});
                        TileShape::Current().SetVecTile(ropeTile[0], ropeTile[1]);
                        auto tmpKR1 = Cast(kRope, DataType::DT_FP32);
                        auto tmpKR2 = Cast(tmpKR1, dtype);
                        Assemble(tmpKR2, {tIdx * blockSize, dNopeSize}, kPart);

                        auto vNope = View(vNopeCache, {blockSize, dNopeSize}, {curBlockIdx * blockSize, n2Idx * dNopeSize});
                        TileShape::Current().SetVecTile(nopeTile[0], nopeTile[1]);
                        auto tmpV1 = Cast(vNope, DataType::DT_FP32);
                        auto tmpV2 = Cast(tmpV1, dtype);
                        Assemble(tmpV2, {tIdx * blockSize, 0}, vPart);
                    }
                    LOOP("LOOP_L2_Idx", FunctionType::DYNAMIC_LOOP, oIdx, LoopRange(1), {}, true) {
                        (void) oIdx;
                        SymbolicScalar curOffset = bIdx * s1Size * nQ + s1Idx * nQ + n2Idx * gGroup + gIdx * gTile;
                        auto kActualPart = View(kPart, {windowSize, dNopeSize + dRopeSize},
                            {winActualSize, dNopeSize + dRopeSize}, {blockStartOffset, 0});
                        auto vActualPart = View(vPart, {windowSize, dNopeSize}, {winActualSize, dNopeSize},
                            {blockStartOffset, 0});
                        Tensor qPart(dtype, {gTile, dNopeSize + dRopeSize}, "qPart");
                        // query
                        auto qNopeL = View(qNope, {gTile, dNopeSize}, {gTile, dNopeSize}, {curOffset, 0});
                        Assemble(qNopeL, {0, 0}, qPart);
                        auto qRopeR = View(qRope, {gTile, dNopeSize}, {gTile, dRopeSize}, {curOffset, 0});
                        Assemble(qRopeR, {0, dNopeSize}, qPart);

                        // matmul_1
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        auto qKT = Matrix::Matmul<false, true>(DataType::DT_FP32, qPart, kActualPart);
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto qKTScale = Mul(qKT, Element(qKT.GetStorage()->Datatype(), softmaxScale));

                        // softmax
                        auto tileMax = Amax(qKTScale, -1, true); // max
                        auto tileSub = Sub(qKTScale, tileMax); // sub max
                        auto tileExp = Exp(tileSub); // exp
                        auto tilSum = Sum(tileExp, -1, true);
                        auto tileSoftmx = Div(tileExp, tilSum);
                        auto valueType16 = Cast(tileSoftmx, dtype);

                        // matmul_2
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        auto out = Matrix::Matmul<false, false>(DataType::DT_FP32, valueType16, vActualPart);

                        // reshape and copyOut
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto outNew = Reshape(out, {bTile, s1Tile, gTile, dNopeSize});
                        TileShape::Current().SetVecTile(1, 1, outTile[0], outTile[1]);
                        auto outFinal = Add(outNew, Element(outNew.GetStorage()->Datatype(), 0.0));
                        Assemble(outFinal, outOffset, attentionOut);
                    }
                }
            }
        }
    }
}

void WinAttentionDebug(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    FUNCTION("main", {qNope, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        WinAttentionDebugCompute(qNope, vNopeCache, qRope, kRopeCache, nQ, nKv, blockTable, actSeqs, windowSize, blockSize,
            softmaxScale, attentionOut, tileConfig);
    }
}

void WinAttention(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    FUNCTION("main", {qNope, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        WinAttentionCompute(qNope, vNopeCache, qRope, kRopeCache, nQ, nKv, blockTable, actSeqs, windowSize, blockSize,
            softmaxScale, attentionOut, tileConfig);
    }
}

void WinAttentionFlash(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    FUNCTION("main", {qNope, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        WinAttentionComputeFlash(qNope, vNopeCache, qRope, kRopeCache, nQ, nKv, blockTable, actSeqs, windowSize, blockSize,
            softmaxScale, attentionOut, tileConfig);
    }
}
} // namespace npu::tile_fwk
