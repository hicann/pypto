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
 * \file nsa_selected_attention.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "nsa_selected_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
/**
 * normal attention: q=qNope+qRope, kv是连续的
 * input:
    * topKIndcies: [b, s1, topk]
    * kvNopeCache: [blockNum * blockSize, n2 * v_dim]
    * kRopeCache: [blockNum * blockSize, n2 * rope_dim]
    * kvActSeqs: [b]
    * blockTableL {b, maxBlockNumPerBatch}
    * qNope: [b*s1*n2*g, k_dim] fp16/bf16
    * qRope: [b*s1*n2*g, rope_dim] fp16/bf16
 * output:
    * attentionOut: [b, s1, n2, g, v_dim] fp32

 * middle tensor:
    * kSlc: [b*s1*n2*s2, k_dim + rope_dim], nope与rope在gen_kv_slc中已经合并起来了 fp16/bf16
    * vSlc: [b*s1*n2*s2, v_dim] fp16/bf16
    * kvSlcActSeqs: [b, s1] int32
*/
void SelectedAttentionCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig, bool debug) {
    auto dtype = qNope.GetStorage()->Datatype();
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];
    int group = nQ / nKv;

    auto v0Tile = saTileConfig.kvSlcV0TileShape;
    int gTile = saTileConfig.gTile;
    auto c1Tile = saTileConfig.c1TileShape;
    auto v1Tile = saTileConfig.v1TileShape;
    auto c2Tile = saTileConfig.c2TileShape;
    auto v2Tile = saTileConfig.v2TileShape;

    /******** tune params ********/
    // config::SetPassOption(CUBE_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    // config::SetPassOption(CUBE_L1_REUSE_MODE, 0);
    // config::SetPassOption(MG_COPYIN_UPPER_BOUND, 1 * 1024 * 1024);
    // config::SetPassOption(SG_PG_UPPER_BOUND, 100000);
    // config::SetPassOption(PG_PARALLEL_LOWER_BOUND, 2);
    // config::SetOperationConfig("FORCE_COMBINE_AXIS", true);

    SymbolicScalar batchSizeSym = topKIndcies.GetShape()[0]; // b
    SymbolicScalar s1N2GSym = qNope.GetShape()[0] / batchSizeSym; // s1n2
    SymbolicScalar s1Sym = s1N2GSym / nQ; // s1
    SymbolicScalar gLoopSym = group / gTile;
    SymbolicScalar n2Sym = nKv;


    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true) {
        SymbolicScalar curActSeq = GetTensorData(kvActSeqs, {bIdx});
        curActSeq.AsIntermediateVariable();
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) {
            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景
                LOOP("LOOP_L3_g_SA", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1)) { // slc_attn
                    int curGTile = gTile;
                    SymbolicScalar curOffset = bIdx * s1N2GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                    std::vector<SymbolicScalar> oiOffset = {bIdx, s1Idx, n2Idx * group + gIdx * curGTile, 0}; // 按最终结果(B,S1,N1,D)进行assemble

                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, 1, 1), PowersOf2(1)) { // 非Flash
                        int curS2Tile = topk * slcBlockSize;
                        // kv_slc
                        config::SetSemanticLabel("kv_slc");
                        Tensor kSlc(dtype, {topk * slcBlockSize, dN + dR}, "kSlc");
                        SymbolicScalar curKvSlcSeq = 0;
                        SymbolicScalar sSlc = (curActSeq - s1Sym + 1 + s1Idx - cmpBlockSize + slcBlockSize) / slcBlockSize;
                        sSlc.AsIntermediateVariable();
                        SymbolicScalar positions = 0;
                        std::vector<AssembleItem> assembeItems;
                        for (int topKIdx = 0; topKIdx < topk; topKIdx++) {
                            if (topKIdx < front) {
                                // 获取到topk的position
                                // 头部的front个
                                positions = topKIdx * slcBlockSize;
                            } else if (topKIdx > (topk - near - front)) {
                                // 尾部的near个
                                positions = (sSlc - near + (topKIdx - (topk - front - near)) - 1) * slcBlockSize;
                            } else {
                                // 中间的topk-front-near个
                                SymbolicScalar topkIndex;
                                if (debug) {
                                    TileShape::Current().SetVecTile(1, 1, NUM16);
                                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, topKIdx - front});
                                } else {
                                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, topKIdx - front});
                                }

                                positions = topkIndex * slcBlockSize;
                            }
                            curKvSlcSeq = curKvSlcSeq + std::min(slcBlockSize, curActSeq - positions);
                            SymbolicScalar blockIdxInBatch = positions / blockSize;
                            SymbolicScalar tail = positions % blockSize;
                            SymbolicScalar slcBlockIdx = GetTensorData(blockTable, {bIdx, blockIdxInBatch});
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock = View(kvNopeCache, {slcBlockSize, dN}, {slcBlockIdx * blockSize + tail, n2Idx * dN});
                            auto krSlcBlock = View(kRopeCache, {slcBlockSize, dR}, {slcBlockIdx * blockSize + tail, n2Idx * dR});

                            config::SetSemanticLabel("kv_slc_cast_fp32");
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock_fp32 = Cast(kvSlcBlock, DataType::DT_FP32);
                            auto krSlcBlock_fp32 = Cast(krSlcBlock, DataType::DT_FP32);
                            config::SetSemanticLabel("kv_slc_cast");
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock_fp16 = Cast(kvSlcBlock_fp32, kSlc.GetStorage()->Datatype());
                            auto krSlcBlock_fp16 = Cast(krSlcBlock_fp32, kSlc.GetStorage()->Datatype());
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);

                            SymbolicScalar slcOutSOffset = topKIdx * slcBlockSize;
                            assembeItems.emplace_back(AssembleItem{kvSlcBlock_fp16, std::vector<SymbolicScalar>{slcOutSOffset, 0}});
                            assembeItems.emplace_back(AssembleItem{krSlcBlock_fp16, std::vector<SymbolicScalar>{slcOutSOffset, dN}});
                        }
                        Assemble(assembeItems, kSlc, true);

                        // qAssemble
                        config::SetSemanticLabel("Sa");
                        // View, 临时规避改成 View
                        auto qn = View(qNope, {curGTile, dN}, {curOffset, 0});
                        auto qr = View(qRope, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        TileShape::Current().SetVecTile(c1Tile[0], c1Tile[NUM_VALUE_2]);
                        Assemble({{qn, {0, 0}}, {qr, {0, dN}}}, qi, true);

                        // slc_attn
                        SymbolicScalar curSeq = std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0); // for MTP s1!= 1 casual计算
                        curSeq.AsIntermediateVariable();
                        auto kj = View(kSlc, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                        {s2Idx * curS2Tile, 0}); // kSlc已经合并了rope和nope
                        auto vj = View(kSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                        {s2Idx * curS2Tile, 0});

                        // C1
                        config::SetSemanticLabel("Sa_QkMM");
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                        auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                        // V1
                        config::SetSemanticLabel("Sa_Qkvec1");
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale));
                        auto tildaMij = Amax(sijScale, -1, true); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tsub = Sub(sijScale, tildaMij); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaLij = Sum(tildaPij, -1, true);
                        auto tSoftmax = Div(tildaPij, tildaLij);
                        auto tildaPijF16 = Cast(tSoftmax, dtype);

                        // C2
                        config::SetSemanticLabel("Sa_KvMm");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto oi = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);

                        // V2
                        config::SetSemanticLabel("Sa_KvVec2");
                        TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                        auto oi4Dim = Add(Reshape(oi, {1, 1, curGTile, dN}), Element(oi.GetStorage()->Datatype(), float(0)));
                        Assemble({{oi4Dim, oiOffset}}, attentionOut, true);
                    }
                }
            }
        }
    }
}

void SelectedAttention(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig) {
    FUNCTION("SA_MAIN",
        {topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable, qNope, qRope},
        {attentionOut}) {
        SelectedAttentionCompute(topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable,
            qNope, qRope, attentionOut,
            nQ, nKv, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig);
    }
}

} // namespace npu::tile_fwk
