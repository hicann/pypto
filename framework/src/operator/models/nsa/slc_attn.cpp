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
 * \file selected_attention.cpp
 * \brief
 */

#include "interface/inner/tilefwk.h"
#include "slc_attn.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
/**
 * normal attention: q=qNope+qRope, kv是连续的
 * input:
    * qNope: [b*s1*n2*g, k_dim] fp16/bf16
    * qRope: [b*s1*n2*g, rope_dim] fp16/bf16
    * kSlc: [b*s1*n2*s2, k_dim + rope_dim], nope与rope在gen_kv_slc中已经合并起来了 fp16/bf16
    * vSlc: [b*s1*n2*s2, v_dim] fp16/bf16
    * kvSlcActSeqs: [b, s1] int32
 * output:
    * attentionOut: [b, s1, n2, g, v_dim] fp32
*/
void SlcAttnCompute(const Tensor &qNope, const Tensor &qRope, const Tensor &kSlc, const Tensor &vSlc, const Tensor &kvSlcActSeqs, int nQ, int nKv,
    float softmaxScale, Tensor &attentionOut, SaTileShapeConfig tileConfig) {
    auto dtype = qNope.GetStorage()->Datatype();
    int dN = qNope.GetShape()[1];
    int dR = qRope.GetShape()[1];
    int group = nQ / nKv;

    int gTile = tileConfig.gTile;
    int s2Tile = tileConfig.sKvTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    /******** tune params ********/
    // config::SetPassOption(CUBE_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    // config::SetPassOption(CUBE_L1_REUSE_MODE, 0);
    // config::SetPassOption(MG_COPYIN_UPPER_BOUND, 1 * 1024 * 1024);
    // config::SetPassOption(SG_PG_UPPER_BOUND, 100000);
    // config::SetPassOption(SG_PARALLEL_NUM, 2);
    // config::SetOperationConfig(KEY_FORCE_COMBINE_AXIS, true);

    SymbolicScalar batchSizeSym = kvSlcActSeqs.GetShape()[0]; // b
    SymbolicScalar s1N2GSym = qNope.GetShape()[0] / batchSizeSym; // s1n2
    SymbolicScalar s1Sym = s1N2GSym / nQ; // s1
    SymbolicScalar gLoopSym = group / gTile;

    SymbolicScalar s1N2S2Sym = kSlc.GetShape()[0] / batchSizeSym; // s1n2s2
    SymbolicScalar n2S2Sym = s1N2S2Sym / s1Sym; // n2s2
    SymbolicScalar n2Sym = nKv;

    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true) {
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) {
            SymbolicScalar curKvSlcSeq = GetTensorData(kvSlcActSeqs, {bIdx, s1Idx}); // 每一个S1的k_slc/v_slc都不同
            SymbolicScalar curSeq = std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0); // for MTP s1!= 1 casual计算
            curSeq.AsIntermediateVariable();
            SymbolicScalar bnPerBatch = (curSeq + s2Tile - 1) / s2Tile;

            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景
                LOOP("LOOP_L3_g_SA", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1)) {
                    int curGTile = gTile;
                    Tensor oiUpdate(DT_FP32, {curGTile, dN}, "oiUpdate");
                    Tensor liUpdate(DT_FP32, {curGTile, 1}, "liUpdate");
                    Tensor miUpdate(DT_FP32, {curGTile, 1}, "miUpdate");

                    SymbolicScalar curOffset = bIdx * s1N2GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                    std::vector<SymbolicScalar> oiOffset = {bIdx, s1Idx, n2Idx * group + gIdx * curGTile, 0}; // 按最终结果(B,S1,N1,D)进行assemble

                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                        int curS2Tile = s2Tile;
                        SymbolicScalar curKvOffset = bIdx * s1N2S2Sym + s1Idx * n2S2Sym + s2Idx * curS2Tile;

                        config::SetSemanticLabel("Sa");
                        // View, 临时规避改成 View
                        auto qn = View(qNope, {curGTile, dN}, {curGTile, dN}, {curOffset, 0});
                        auto qr = View(qRope, {curGTile, dR}, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        Assemble(qn, {0, 0}, qi);
                        Assemble(qr, {0, dN}, qi);

                        auto kj = View(kSlc, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                        {curKvOffset, 0}); // kSlc已经合并了rope和nope
                        auto vj = View(vSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                        {curKvOffset, 0});

                        // C1
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        config::SetSemanticLabel("Sa_QkMM");
                        TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                        auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                        // V1
                        config::SetSemanticLabel("Sa_Qkvec1");
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale));
                        auto tildaMij = Amax(sijScale, -1, true); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tsub = Sub(sijScale, tildaMij); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaPijF16 = Cast(tildaPij, dtype);
                        auto tildaLij = Sum(tildaPij, -1, true);

                        IF (IsLoopBegin(s2Idx, 0)) {
                            // C2
                            TileShape::Current().SetCubeTile(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            config::SetSemanticLabel("Sa_KvMm");
                            TileShape::Current().SetMatrixSize(
                                {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                            auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH3
                                // V2
                                config::SetSemanticLabel("Sa_KvVec2");
                                oiUpdate = Div(oiTmp, tildaLij);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = Add(Reshape(oiUpdate, {1, 1, curGTile, dN}), Element(oiUpdate.GetStorage()->Datatype(), float(0)));
                                Assemble(oiUpdate4Dim, oiOffset, attentionOut);
                            } ELSE { // PATH2
                                oiUpdate = oiTmp;
                            }
                            liUpdate = tildaLij;
                            miUpdate = tildaMij;
                        } ELSE {
                            config::SetSemanticLabel("Sa_UpdateVec2");
                            auto oi = oiUpdate;
                            auto li = liUpdate;
                            auto mi = miUpdate;

                            auto miNew = Maximum(mi, tildaMij);
                            auto t1 = Sub(mi, miNew);
                            auto t2 = Exp(t1);
                            auto t3 = Sub(tildaMij, miNew);
                            auto t4 = Exp(t3);
                            auto t5 = Mul(t4, tildaLij);
                            auto t6 = Mul(t2, li);
                            auto liNew = Add(t6, t5);

                            auto q3 = Mul(oi, t2);
                            TileShape::Current().SetCubeTile(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            config::SetSemanticLabel("Sa_UpdateMM2");
                            TileShape::Current().SetMatrixSize(
                                {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                            auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            auto q2 = Mul(q1, t4);
                            auto oiTmp = Add(q3, q2);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH1
                                oiUpdate = Div(oiTmp, liNew);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = Add(Reshape(oiUpdate, {1, 1, curGTile, dN}), Element(oiUpdate.GetStorage()->Datatype(), float(0)));
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

void SlcAttn(const Tensor &qNope, const Tensor &qRope, const Tensor &kSlc, const Tensor &vSlc, const Tensor &kvSlcActSeqs, int nQ, int nKv,
    float softmaxScale, Tensor &attentionOut, SaTileShapeConfig tileConfig) {
    FUNCTION("SA_MAIN", {qNope, qRope, kSlc, vSlc, kvSlcActSeqs}, {attentionOut}) {
        SlcAttnCompute(qNope, qRope, kSlc, vSlc, kvSlcActSeqs, nQ, nKv, softmaxScale, attentionOut, tileConfig);
    }
}

} // namespace npu::tile_fwk
