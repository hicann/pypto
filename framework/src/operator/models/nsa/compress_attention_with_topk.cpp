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
 * \file fused_compress_kv_select.cpp
 * \brief
 */

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "compress_attention_with_topk.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void CompressAttentionWithTopK(const Tensor &qNope, const Tensor &qRope, const Tensor &cmpKvCache,
    const Tensor &cmpKrCache, const Tensor &cmpBlockTable, const Tensor &actSeq, const Tensor &auxTensor,
    Tensor &cmpAttnOut, Tensor &topkRes, const int blockSize, const int cmpBlockSize, const int cmpStride,
    const int slcBlockSize, const float softmaxScale, const int n1, const int topk, const int front, const int near,
    CmpAttnTopkTile &tileConfig) {
    /* bellows are function params support
    qNope: [b*s1*n1, dN], fp16/bf16
    qRope: [b*s1*n1, dR], fp16/bf16
    cmpKvCache: [cmpBlockNum, blockSize, n2, dN], fp16/bf16
    cmpKrCache: [cmpBlockNum, blockSize, n2, dR], fp16/bf16
    cmpBlockTable: [b, maxCmpBlock], int32
    actSeq: [b], int32
    actCmpSeqLen: [b], int32
    auxTensor: [slcBlockSize / cmpStride + cmpBlockSize / cmpStride - 1, n1], fp32
    cmpAttnOut: [b*s1*n1, dN], fp32
    topkRes: [b, s1, topk], int32
    =================================================================================
    b:1~96, means batch size
    s1:1~16, means sequence len of query, and s1 > 1 refers to multi-token-predication
    n1:128, means headNum of query
    dN:512, means headDim of no-rope part of query and key
    dR:64, means headDim of rope part of query and key
    blockSize: 128, means uints of discretized storage of key and value
    n2=1: means headNum of key and value, and value 1 refers to mla, not gqa
    cmpBlockNum: means total blocks of each actual compressed key and value sequence
    maxCmpBlock: means max BlockNum of all compressed sequence of key
    */

    auto c1Tile = tileConfig.cmpTile.c1Tile;
    auto v1Tile = tileConfig.cmpTile.v1Tile;
    auto c2Tile = tileConfig.cmpTile.c2Tile;
    auto v2Tile = tileConfig.cmpTile.v2Tile;

    auto qDtype = qNope.GetStorage()->Datatype();
    auto kDtype = cmpKvCache.GetStorage()->Datatype();

    const int b = cmpBlockTable.GetShape()[SHAPE_DIM0];
    ASSERT(n1 !=0) << "n1 can't be zero!";
    const int s1 = qNope.GetShape()[SHAPE_DIM0] / b / n1;

    const int dN = qNope.GetShape()[SHAPE_DIM1];
    const int dR = qRope.GetShape()[SHAPE_DIM1];
    const int dQK = dN + dR;
    const int maxCmpBlock = cmpBlockTable.GetShape()[SHAPE_DIM1];
    ASSERT(cmpStride !=0) << "n1 can't be zero!";
    const int slcSize = slcBlockSize / cmpStride;
    ASSERT(slcSize !=0) << "slcSize can't be zero!";
    const int blockSlcNum = blockSize / slcSize;
    const int cmpSize = cmpBlockSize / cmpStride;
    const int slcWindow = slcSize + cmpSize - 1;
    (void)v1Tile;
    (void)qDtype;
    (void)dQK;

    // 接入Topk的时候让后面子图接收FP32的index结果，另外此处的LOOP必须存在，不能和下面的LOOP合并，否则会存在UB上的View
    Tensor topkNumIdx(DT_FP32, {1, 1, maxCmpBlock * blockSlcNum}, "topkNumIdx");
    TileShape::Current().SetVecTile(tileConfig.topkTile);
    LOOP("GEN_TOPK_RANGE", FunctionType::DYNAMIC_LOOP, ubReshapeIdx, LoopRange(1), {}) {
        (void)ubReshapeIdx;
        auto dumpTensor = Full(Element(DataType::DT_FP32, 0.0f), DT_FP32, {1, 1, maxCmpBlock * blockSlcNum});
        auto firstKTop = std::get<1>(TopK(dumpTensor, maxCmpBlock * blockSlcNum, -1, true));
        topkNumIdx = std::get<0>(TopK(Cast(firstKTop, DT_FP32), maxCmpBlock * blockSlcNum, -1, false));
    }

    LOOP("CMP_ATTN_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b), {}) {
        SymbolicScalar curSeq = GetTensorData(actSeq, {bIdx});
        LOOP("CMP_ATTN_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1), {}, true) {
            // 因果推理，注意可用实际的s2长度
            ASSERT(cmpStride !=0) << "cmpStride can't be zero!";
            auto casCmpSeq = (curSeq - (s1 - s1Idx - 1) - cmpBlockSize) / cmpStride + 1;
            ASSERT(blockSize !=0) << "blockSize can't be zero!";
            auto curCmpBlock = (casCmpSeq + blockSize - 1) / blockSize;
            auto slcLoop = (casCmpSeq + slcSize - 1) / slcSize;
            auto qOffset = bIdx * s1 * n1 + s1Idx * n1;

            // Now only support Mla,  n2=1 and don't need Loop
            Tensor oiUpdate(DT_FP32, {n1, dN}, "oiUpdate");
            Tensor liUpdate(DT_FP32, {1, n1}, "liUpdate");
            Tensor miUpdate(DT_FP32, {1, n1}, "miUpdate");
            Tensor slcBeforeGReduce(DT_FP32, {maxCmpBlock * blockSlcNum, n1}, "slcBeforeGReduce");
            Tensor localMaxGather(DT_FP32, {maxCmpBlock, n1}, "localMaxGather"); // maxBlock
            Tensor slcPre(DT_FP32, {blockSlcNum, n1}, "slcPre");
            // Block Loop
            LOOP("CMP_ATTN_LOOP_BLOCK", FunctionType::DYNAMIC_LOOP, blockIdx, LoopRange(curCmpBlock), {}) {
                SymbolicScalar curBlockIdx = GetTensorData(cmpBlockTable, {bIdx, blockIdx});
                curBlockIdx.AsIntermediateVariable();

                auto curValidSeq = std::min(casCmpSeq - blockIdx * blockSize, blockSize);
                auto curSlcLoop = (curValidSeq + slcSize - 1) / slcSize;
                curSlcLoop.AsIntermediateVariable();

                Tensor slcCur(DT_FP32, {blockSlcNum, n1}, "slcCur");
                Tensor tildaPijPad(DT_FP32, {blockSize + slcWindow - 1, n1}, "tildaPijPad");
                Tensor miModify(DT_FP32, {1, n1}, "miModify");

                LOOP("AVOID_LOOP_1", FunctionType::DYNAMIC_LOOP, avoid_loop_1_idx, LoopRange(1), {}) {
                    (void)avoid_loop_1_idx;

                    int vecTile = 128;
                    TileShape::Current().SetVecTile(vecTile, vecTile);
                    auto curQn = View(qNope, {n1, dN}, {qOffset, 0});
                    auto curQr = View(qRope, {n1, dR}, {qOffset, 0});
                    auto curQAttn = Assemble({
                        {curQn,  {0, 0}},
                        {curQr, {0, dN}}
                    });
                    curQAttn.SetName("curQAttn");

                    // 注意这里需要申请对齐的shape，因为不确定尾块多大
                    auto cmpKvCache2D =
                        Reshape(cmpKvCache, {cmpKvCache.GetShape()[0] * cmpKvCache.GetShape()[1] * cmpKvCache.GetShape()[2], dN});
                    auto curCmpKv =
                        View(cmpKvCache2D, {blockSize, dN}, {curValidSeq, dN}, {curBlockIdx * blockSize, 0});
                    auto cmpKrCache2D =
                        Reshape(cmpKrCache, {cmpKrCache.GetShape()[0] * cmpKrCache.GetShape()[1] * cmpKrCache.GetShape()[2], dR});
                    auto curCmpKr =
                        View(cmpKrCache2D, {blockSize, dR}, {curValidSeq, dR}, {curBlockIdx * blockSize, 0});
                    auto curKAttn = Assemble({
                        {curCmpKv,  {0, 0}},
                        {curCmpKr, {0, dN}}
                    });
                    curKAttn.SetName("curKAttn");

                    auto curVAttn = curCmpKv; // cmpKv tensor can be reused by cmpV tensor
                    curVAttn.SetName("curVAttn");
                    config::SetSemanticLabel("Cmp-Attn-C1");
                    TileShape::Current().SetCubeTile(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul(
                        DataType::DT_FP32, curKAttn, curQAttn, false, true); // (blockSize, dQK), (n1, dQK)-> (blockSize, n1)
                    sij.SetName("sij");

                    TileShape::Current().SetVecTile(vecTile, vecTile);
                    sij = View(sij, {blockSize, n1}, {curValidSeq, n1}, {0, 0});
                    config::SetSemanticLabel("Cmp-Attn-V1");
                    auto sijScale = Mul(sij, Element(sij.GetStorage()->Datatype(), softmaxScale)); // (blockSize, n1)
                    // reduceMax首轴不支持切分
                    auto tildaMij = Amax(sijScale, 0, true); // (1, n1)
                    auto tsub = Sub(sijScale, tildaMij);       // (blockSize, n1) - (1, n1) -> (blockSize, n1)
                    auto tildaPij = Exp(tsub);                 // (blockSize, n1)
                    auto tildaPijB16 = Cast(tildaPij, kDtype); // (blockSize, n1)
                    auto tildaLij = Sum(tildaPij, 0, true); // (1, n1)
                    IF(IsLoopBegin(blockIdx, 0)) {
                        config::SetSemanticLabel("Cmp-Attn-First-Block-C2");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto oiTmp = Matrix::Matmul(DataType::DT_FP32, tildaPijB16,
                            curVAttn, true, false); // (n1, blockSize), (blockSize, dN) -> (n1, dN)
                        oiTmp.SetName("oiTmp");
                        config::SetSemanticLabel("Cmp-Attn-First-Block-V2");
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        IF(IsLoopEnd(blockIdx, curCmpBlock)) {
                            oiUpdate = Div(oiTmp, Reshape(tildaLij, {n1, 1})); // (n1, dN), (n1, 1) -> (n1, dN)
                            auto oiUpdateReshape = Reshape(oiUpdate, {1, 1, n1, dN});
                            TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                            auto oiUpdateCast = Assign(Cast(oiUpdateReshape, cmpAttnOut.GetStorage()->Datatype()));
                            Assemble(oiUpdateCast, {bIdx, s1Idx, 0, 0}, cmpAttnOut);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        }
                        ELSE {
                            oiUpdate = oiTmp; // (n1, dN)
                        }
                        liUpdate = tildaLij; // (1, n1)
                        miUpdate = tildaMij; // (1, n1)
                    }
                    ELSE {
                        config::SetSemanticLabel("Cmp-Attn-Other-Update-V1");
                        auto oi = oiUpdate;                 // (n1, dN)
                        auto li = liUpdate;                 // (1, n1)
                        auto mi = miUpdate;                 // (1, n1)
                        auto miNew = Maximum(mi, tildaMij); // (1, n1), (1, n1) -> (1, n1)
                        auto t1 = Sub(mi, miNew);           // (1, n1), (1, n1) -> (1, n1)
                        auto t2 = Exp(t1);                  // (1, n1)
                        auto t3 = Sub(tildaMij, miNew);     // (1, n1), (1, n1) -> (1, n1)
                        auto t4 = Exp(t3);                  // (1, n1)
                        auto t5 = Mul(t4, tildaLij);        // (1, n1), (1, n1) -> (1, n1)
                        auto t6 = Mul(t2, li);              // (1, n1), (1, n1) -> (1, n1)
                        auto liNew = Add(t6, t5);           // (1, n1), (1, n1) -> (1, n1)

                        auto q3 = Mul(oi, Reshape(t2, {n1, 1})); // (n1, dN), (n1, 1) -> (n1, dN)
                        config::SetSemanticLabel("Cmp-Attn-Other-Update-C2");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});

                        auto tildaPijB16T = Transpose(tildaPijB16, {0, 1}); // (blockSize, n1) -> (n1, blockSize)
                        auto q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijB16T,
                            curVAttn, false, false); // (n1, blockSize), (blockSize, dN) -> (n1, dN)
                        q1.SetName("q1");
                        config::SetSemanticLabel("Cmp-Attn-Other-Update-V2");
                        TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, Reshape(t4, {n1, 1})); // (n1, dN), (n1, 1) -> (n1, dN)
                        auto oiTmp = Add(q3, q2);                // (n1, dN), (n1, dN) -> (n1, dN)
                        IF(IsLoopEnd(blockIdx, curCmpBlock)) {
                            oiUpdate = Div(oiTmp, Reshape(liNew, {n1, 1})); // (n1, dN), (n1, 1) -> (n1, dN)
                            auto oiUpdateReshape = Reshape(oiUpdate, {1, 1, n1, dN});
                            TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                            auto oiUpdateCast = Assign(Cast(oiUpdateReshape, cmpAttnOut.GetStorage()->Datatype()));
                            Assemble(oiUpdateCast, {bIdx, s1Idx, 0, 0}, cmpAttnOut);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                        }
                        ELSE {
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            oiUpdate = oiTmp; // (n1, dN)
                        }
                        Assemble(miNew, {blockIdx - 1, 0}, localMaxGather);
                        miModify = Sub(mi, miNew);
                        miModify = Exp(miModify);

                        miUpdate = miNew; // (1, n1)
                        liUpdate = liNew; // (1, n1)

                        // update current slcBlock
                        auto subCur = Sub(tildaMij, miUpdate); // (1, n1), (1, n1) -> (1, n1)
                        auto expCur = Exp(subCur);             // (1, n1)
                        tildaPij = Mul(tildaPij, expCur);      // (blockSize, n1), (1, n1) -> (blockSize, n1)
                    }
                    Element src(DataType::DT_FP32, 0.0f);
                    auto zeros = Full(src, DT_FP32, {slcWindow - 1, n1});
                    tildaPijPad = Cat({tildaPij, zeros}, 0);
                }

                LOOP("CMP_ATTN_P_SLC_FIRST", FunctionType::DYNAMIC_LOOP, slcIdx, LoopRange(curSlcLoop), {}) {
                    // update current slc block
                    auto slcValid = std::min(curValidSeq - slcIdx * slcSize, slcWindow);
                    auto lastView =
                        View(tildaPijPad, {slcWindow, n1}, {slcValid, n1}, {slcIdx * slcSize, 0}); // last window
                    auto auxTmpTensor = View(auxTensor, {slcWindow, n1}, {slcValid, n1}, {0, 0});
                    auto slcLastNoReduce = Mul(lastView, auxTmpTensor); // (slcSize, n1), (slcSize, n1) -> (slcSize, n1)
                    auto slcLastReduce = Sum(slcLastNoReduce, 0, true); // (slcSize, n1) -> (1, n1)
                    Assemble(slcLastReduce, {slcIdx, 0}, slcCur);
                }

                LOOP("GEN_SLC_BEFORE_G_REDUCE", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1), {}) {
                    (void)unusedIdx;
                    IF(blockIdx != 0) {
                        slcPre = Mul(slcPre, miModify);
                        auto modifyTensor =
                            View(tildaPijPad, {cmpSize - 1, n1}, {std::min(cmpSize - 1, curValidSeq), n1}, {0, 0});
                        auto lastAuxTensor =
                            View(auxTensor, {cmpSize - 1, n1}, {std::min(cmpSize - 1, curValidSeq), n1},
                                {auxTensor.GetShape()[0] - std::min(cmpSize - 1, curValidSeq), 0});
                        modifyTensor = Mul(modifyTensor,
                            lastAuxTensor); // (cmpSize - 1, n1), (cmpSize - 1, n1) -> (cmpSize - 1, n1)
                        auto modifyTensorReduce = Sum(modifyTensor, 0, true); // (1, n1)
                        auto preViewTensor = View(slcPre, {1, n1}, {blockSlcNum - 1, 0});
                        preViewTensor = Add(preViewTensor, modifyTensor); // (1, n1)
                        Assemble(Assign(View(slcPre, {blockSlcNum - 1, n1}, {0, 0})),
                            {(blockIdx - 1) * blockSlcNum, 0}, slcBeforeGReduce);
                        Assemble(preViewTensor, {blockIdx * blockSlcNum - 1, 0}, slcBeforeGReduce);
                        slcPre = Assign(slcCur);
                    }
                    ELSE {
                        slcPre = Assign(slcCur);
                    }
                    IF(blockIdx == curCmpBlock - 1) {
                        Assemble(Assign(slcCur), {blockIdx * blockSlcNum, 0}, slcBeforeGReduce);
                    }
                }
            }

            Tensor slcBeforeGReduce2(DT_FP32, {maxCmpBlock * blockSlcNum, n1}, "slcBeforeGReduce2");
            LOOP("SOFTMAX_BLOCK", FunctionType::DYNAMIC_LOOP, blockIdx, LoopRange(curCmpBlock), {}) {
                auto slcBeforeGReduceBlock = View(slcBeforeGReduce, {blockSlcNum, n1}, {blockIdx * blockSlcNum, 0});
                IF(IsLoopEnd(blockIdx, curCmpBlock)) {
                    slcBeforeGReduceBlock = Div(slcBeforeGReduceBlock, liUpdate);
                }
                ELSE {
                    auto rowMaxBlock = View(localMaxGather, {1, n1}, {blockIdx, 0}); // (1, n1)
                    auto subTmp = Sub(rowMaxBlock, miUpdate);                         // (1, n1)
                    auto expTmp = Exp(subTmp);                                        // (1, n1)
                    slcBeforeGReduceBlock =
                        Mul(slcBeforeGReduceBlock, expTmp); // (blockSlcNum, n1), (1, n1) -> (blockSlcNum, n1)
                    slcBeforeGReduceBlock =
                        Div(slcBeforeGReduceBlock, liUpdate); // (blockSlcNum, n1), (1, n1) -> (blockSlcNum, n1)
                    slcBeforeGReduceBlock = Add(slcBeforeGReduceBlock, Element(DT_FP32, 0.0f));
                }
                Assemble(slcBeforeGReduceBlock, {blockIdx * blockSlcNum, 0}, slcBeforeGReduce2);
            }

            LOOP("TOPK_INDICES", FunctionType::DYNAMIC_LOOP, unusedIdx, LoopRange(1), {}) {
                (void)unusedIdx;
                Tensor slcReShape;
                LOOP("AVOID_LOOP_5", FunctionType::DYNAMIC_LOOP, ubReshapeIdx, LoopRange(1), {}) {
                    (void)ubReshapeIdx;
                    auto slcBeforeGReduceActual =
                        View(slcBeforeGReduce2, {maxCmpBlock * blockSlcNum, n1}, {slcLoop, n1}, {0, 0});
                    auto slcReduce = Sum(
                        slcBeforeGReduceActual, -1, true); // (maxSlcLoop, n1) - > (maxSlcLoop, 1), 有效数据为(slcLoop, 1)
                    TileShape::Current().SetVecTile(tileConfig.topkTile);
                    slcReShape = Reshape(slcReduce, {1, 1, maxCmpBlock * blockSlcNum}, {1, 1, slcLoop});
                    slcReShape = Add(slcReShape, Element(DT_FP32, 0.0f));
                }

                LOOP("AVOID_LOOP_6", FunctionType::DYNAMIC_LOOP, ubReshapeIdx, LoopRange(1), {}) {
                    (void)ubReshapeIdx;
                    IF(slcLoop < topk) {
                        auto curIdx =
                            View(topkNumIdx, {1, 1, maxCmpBlock * blockSlcNum}, {1, 1, slcLoop}, {0, 0, 0});
                        curIdx = Cast(curIdx, DT_INT32);
                        Assemble(curIdx, {bIdx, s1Idx, 0}, topkRes);
                    }
                    ELSE {
                        auto slcFront = View(topkNumIdx, {1, 1, front}, {0, 0, 0});
                        auto slcNear = View(topkNumIdx, {1, 1, near}, {0, 0, slcLoop - near});
                        auto slcReView = View(slcReShape, {1, 1, maxCmpBlock * blockSlcNum - front - near},
                            {1, 1, slcLoop - front - near}, {0, 0, front});
                        auto innerTopk =
                            std::get<1>(TopK(slcReView, topk - front - near, -1, true)); // (1, 1, topk-front-near)
                        innerTopk = Add(innerTopk, Element(DT_INT32, 1UL));
                        slcFront = Cast(slcFront, DT_INT32);
                        slcNear = Cast(slcNear, DT_INT32);
                        Assemble(slcFront, {bIdx, s1Idx, 0}, topkRes);
                        Assemble(innerTopk, {bIdx, s1Idx, front}, topkRes);
                        Assemble(slcNear, {bIdx, s1Idx, topk - near}, topkRes);
                    }
                }
            }
        }
    }
}
} // namespace npu::tile_fwk
