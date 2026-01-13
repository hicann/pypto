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

#include "operator/models/deepseek/deepseek_mla.h"
#include "tilefwk/tilefwk.h"
#include "interface/utils/common.h"
#include "fused_compress_kv_select.h"
#include "operator/models/deepseek/dynamic_nsa.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

Tensor MlpSingleRope(const Tensor &x, const Tensor &cos, const Tensor &sin, MlpRopeTile &tileConfig) {
    // x: [cmpBlockSize, dR], cos: [1, cmpBlockSize, dR], sin: [1, cmpBlockSize, dR]
    ASSERT(x.GetShape().size() == SHAPE_DIM3 && cos.GetShape().size() == SHAPE_DIM3 && sin.GetShape().size() == SHAPE_DIM3);

    auto cmpSize = x.GetShape()[NUM_VALUE_0];
    auto n2 = x.GetShape()[NUM_VALUE_1];
    auto dR = x.GetShape()[NUM_VALUE_2];
    auto xDtype = x.GetStorage()->Datatype();

    auto fourDim = tileConfig.fourDim;
    auto fiveDim = tileConfig.fiveDim;

    TileShape::Current().SetVecTile(NUM_32, 1, NUM_64);
    auto castX = Cast(x, DT_FP32);
    TileShape::Current().SetVecTile(1, NUM_32, NUM_64);
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);

    auto cosUnsqueeze = Unsqueeze(castCos, NUM_VALUE_2); // (1, cmpBlockSize, 1, dR)
    auto sinUnsqueeze = Unsqueeze(castSin, NUM_VALUE_2); // (1, cmpBlockSize, 1, dR)

    auto xView =
        Reshape(castX, {SHAPE_DIM1, cmpSize, n2, dR / NUM_VALUE_2, NUM_VALUE_2}); // (1, cmpBlockSize, n2, dR / 2, 2)
    TileShape::Current().SetVecTile(
        fiveDim[NUM_VALUE_0], fiveDim[NUM_VALUE_1], fiveDim[NUM_VALUE_2], fiveDim[NUM_VALUE_3], fiveDim[NUM_VALUE_4]);
    auto xTrans = Transpose(xView, {NUM_VALUE_3, NUM_VALUE_4});
    auto xReSecond = Reshape(xTrans, {SHAPE_DIM1, cmpSize, n2, dR}); // (1, cmpBlockSize, n2, dR)

    TileShape::Current().SetVecTile(
        fourDim[NUM_VALUE_0], fourDim[NUM_VALUE_1], fourDim[NUM_VALUE_2], fourDim[NUM_VALUE_3]);
    auto xEmbed =
        Add(Mul(xReSecond, cosUnsqueeze), Mul(RotateHalf(xReSecond), sinUnsqueeze)); // (1, cmpBlockSize, n2, dR)
    auto xReLast = Reshape(xEmbed, {cmpSize, n2, dR});                               // (cmpBlockSize, n2, dR)
    TileShape::Current().SetVecTile(NUM_32, 1, NUM_64);
    auto res = Cast(xReLast, xDtype); // (cmpSize, n2, dR)
    return res;
}

Tensor BatchMlpSingleRope(const Tensor &x, const Tensor &cos, const Tensor &sin, MlpRopeTile &tileConfig) {
    (void)tileConfig;
    assert(x.GetShape().size() == SHAPE_DIM2 && cos.GetShape().size() == SHAPE_DIM3 && sin.GetShape().size() == SHAPE_DIM3);

    auto cmpSize = x.GetShape()[NUM_VALUE_0];
    auto dR = x.GetShape()[NUM_VALUE_1];
    auto xDtype = x.GetStorage()->Datatype();

    TileShape::Current().SetVecTile(
        tileConfig.threeDim[NUM_VALUE_0], tileConfig.threeDim[NUM_VALUE_1], tileConfig.threeDim[NUM_VALUE_2]);
    auto castX = Cast(x, DT_FP32);
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);

    auto xView = Reshape(castX, {1, cmpSize, dR / NUM_2, NUM_2}); // (1, cmpBlockSize, dR / 2, 2)
    TileShape::Current().SetVecTile(tileConfig.fourDim[NUM_VALUE_0], tileConfig.fourDim[NUM_VALUE_1],
        tileConfig.fourDim[NUM_VALUE_2], tileConfig.fourDim[NUM_VALUE_3]);
    auto xTrans = Transpose(xView, {NUM_2, NUM_3});
    auto xReSecond = Reshape(xTrans, {1, cmpSize, dR}); // (1, cmpBlockSize, dR)

    TileShape::Current().SetVecTile(
        tileConfig.threeDim[NUM_VALUE_0], tileConfig.threeDim[NUM_VALUE_1], tileConfig.threeDim[NUM_VALUE_2]);
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin)); // (1, cmpBlockSize, dR)
    auto res = Cast(xEmbed, xDtype);                                                 // (n2, cmpBlockSize, dR)
    return res;
}

Tensor BatchMlpCompress(const Tensor &x, const Tensor &w1, const Tensor &w2, MlpCmpTile &tileConfig) {
    auto xDtype = x.GetStorage()->Datatype();
    auto c1Tile = tileConfig.c1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v1Tile = tileConfig.v1TileShape;

    TileShape::Current().SetVecTile(NUM_128, NUM_128);
    config::SetSemanticLabel("MlpCompress-1");
    TileShape::Current().SetCubeTile({c1Tile[NUM_VALUE_0], c1Tile[NUM_VALUE_1]},
        {c1Tile[NUM_VALUE_2], c1Tile[NUM_VALUE_3]}, {c1Tile[NUM_VALUE_4], c1Tile[NUM_VALUE_5]}, true);
    auto firstMm = Matrix::Matmul(DT_FP32, x, w1, false, false); // (b, 2 * cmpBlockSize * d)
    TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1]);
    config::SetSemanticLabel("MlpCompress-2");
    auto sigTensor = Sigmoid(firstMm);
    config::SetSemanticLabel("MlpCompress-3");
    auto castTensor = Cast(sigTensor, xDtype);

    config::SetSemanticLabel("MlpCompress-4");
    TileShape::Current().SetCubeTile({c2Tile[NUM_VALUE_0], c2Tile[NUM_VALUE_1]},
        {c2Tile[NUM_VALUE_2], c2Tile[NUM_VALUE_3]}, {c2Tile[NUM_VALUE_4], c2Tile[NUM_VALUE_5]}, true);
    auto res = Matrix::Matmul(xDtype, castTensor, w2, false, false); // (b, d)
    config::SetSemanticLabel("MlpCompress-5");
    TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1]);
    config::SetSemanticLabel("");
    return res;
}

Tensor MlpCompress(const Tensor &x, const Tensor &w1, const Tensor &w2, MlpCmpTile &tileConfig) {
    // x: (cmpBlockSize, n2, d) , w1: (cmpBlockSize*d, 2*cmpBlockSize*d), w2: (2*cmpBlockSize*d, d)
    auto xDtype = x.GetStorage()->Datatype();
    auto transTile = tileConfig.transTileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    const int s = x.GetShape()[NUM_VALUE_0];
    const int n = x.GetShape()[NUM_VALUE_1];
    const int d = x.GetShape()[NUM_VALUE_2];

    config::SetSemanticLabel("MlpCompress-0");
    TileShape::Current().SetVecTile(transTile[NUM_VALUE_0], transTile[NUM_VALUE_1], transTile[NUM_VALUE_2]);
    auto xCast = Cast(x, DT_FP32);
    auto xTrans = Transpose(xCast, {NUM_VALUE_0, NUM_VALUE_1}); // (n2, cmpBlockSize, d)
    TileShape::Current().SetVecTile(1, NUM_32, NUM_128);
    config::SetSemanticLabel("MlpCompress-1");
    auto xRe2 = Reshape(xTrans, {n, s * d}); // (n2, cmpBlockSize * d)
    TileShape::Current().SetVecTile(1, NUM_64);
    config::SetSemanticLabel("MlpCompress-1.5");
    auto xCast2 = Cast(xRe2, xDtype);

    config::SetSemanticLabel("MlpCompress-2");
    TileShape::Current().SetCubeTile({c1Tile[NUM_VALUE_0], c1Tile[NUM_VALUE_1]},
        {c1Tile[NUM_VALUE_2], c1Tile[NUM_VALUE_3]}, {c1Tile[NUM_VALUE_4], c1Tile[NUM_VALUE_5]}, true);
    auto firstMm = Matrix::Matmul(DT_FP32, xCast2, w1, false, false); // (n2, 2 * cmpBlockSize * d)
    TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1]);
    config::SetSemanticLabel("MlpCompress-3");
    auto sigTensor = Sigmoid(firstMm);
    config::SetSemanticLabel("MlpCompress-4");
    auto castTensor = Cast(sigTensor, x.GetStorage()->Datatype());

    config::SetSemanticLabel("MlpCompress-5");
    TileShape::Current().SetCubeTile({c2Tile[NUM_VALUE_0], c2Tile[NUM_VALUE_1]},
        {c2Tile[NUM_VALUE_2], c2Tile[NUM_VALUE_3]}, {c2Tile[NUM_VALUE_4], c2Tile[NUM_VALUE_5]}, true);
    auto res = Matrix::Matmul(DT_FP32, castTensor, w2, false, false); // (n2, d)
    config::SetSemanticLabel("MlpCompress-6");
    auto resRe = Reshape(res, {NUM_VALUE_1, n, d});
    config::SetSemanticLabel("MlpCompress-7");
    TileShape::Current().SetVecTile(v2Tile[NUM_VALUE_0], v2Tile[NUM_VALUE_1], v2Tile[NUM_VALUE_2]);
    auto resCast = Cast(resRe, xDtype); // (1, n2, d)
    config::SetSemanticLabel("");
    return resCast;
}

std::tuple<Tensor, Tensor> CmpAttn(
    const Tensor &q, const Tensor &k, const Tensor &v, const float scale, AttnTile &tileConfig) {
    auto c1Tile = tileConfig.c1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto qDtype = q.GetStorage()->Datatype();
    config::SetSemanticLabel("CmpAttention-MatMul1");
    TileShape::Current().SetCubeTile({c1Tile[NUM_VALUE_0], c1Tile[NUM_VALUE_1]},
        {c1Tile[NUM_VALUE_2], c1Tile[NUM_VALUE_3]}, {c1Tile[NUM_VALUE_4], c1Tile[NUM_VALUE_5]}, true);
    auto mm1 = Matrix::Matmul(DT_FP32, q, k, false, true); // (g, effSeq)
    TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1]);
    config::SetSemanticLabel("CmpAttention-Softmax");
    auto softmaxRes = SoftmaxNew(mm1);                         // (g, effSeq)
    auto scaleRes = Mul(softmaxRes, Element(DT_FP32, scale)); // (g, effSeq)
    auto castScale = Cast(scaleRes, qDtype);
    config::SetSemanticLabel("CmpAttention-MatMul2");
    TileShape::Current().SetCubeTile({c2Tile[NUM_VALUE_0], c2Tile[NUM_VALUE_1]},
        {c2Tile[NUM_VALUE_2], c2Tile[NUM_VALUE_3]}, {c2Tile[NUM_VALUE_4], c2Tile[NUM_VALUE_5]}, true);
    auto mm2 = Matrix::Matmul(DT_FP32, castScale, v, false, false); // (g, dN)
    return std::tie(softmaxRes, mm2);
}

void FusedCompressKvSelectCompute(const Tensor &qNope, const Tensor &qRope, const Tensor &kvCache,
    const Tensor &krCache, const Tensor &cmpKvCache, const Tensor &cmpKrCache, const Tensor &blockTable,
    const Tensor &cmpBlockTable, const Tensor &actSeqLen, const Tensor &actCmpSeqLen, const Tensor &mlpWk1,
    const Tensor &mlpWk2, const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpAttnOut, Tensor &cmpAttnOut16,
    Tensor &cmpSoftmax, Tensor &fullK, Tensor &cmpK, Tensor &firstRope, Tensor &firstRopeInput, Tensor &topkRes,
    Tensor &topkInput, const int blockSize, const int cmpBlockSize, const int cmpStride, const float softmaxScale,
    const int n1, const int n2, CmpAttnTile &tileConfig) {
    /* bellows are function params support
    qNope: [b*s1*n1, dN], fp16/bf16
    qRope: [b*s1*n1, dR], fp16/bf16
    kvCache: [blockNum*blockSize, n2*dN], fp16/bf16
    krCache: [blockNum*blockSize, n2*dR], fp16/bf16
    cmpKCache: [cmpBlockNum*blockSize, n2*dK], fp16/bf16
    cmpVCache: [cmpBlockNum*blockSize, n2*dN], fp16/bf16
    blockTable: [b, maxBlockNum], int32
    cmpBlockTable: [b, maxCmpBlockNum], int32
    actSeqLen: [b], int32
    actCmpSeqLen: [b], int32
    mlpWk1: [cmpBlockSize*dK, 2*cmpBlockSize*dK], fp16/bf16
    mlpWv1: [cmpBlockSize*dN, 2*cmpBlockSize*dN], fp16/bf16
    mlpWk2: [2*cmpBlockSize*dK, dK], fp16/bf16
    mlpWv2: [2*cmpBlockSize*dN, dN], fp16/bf16
    mlpCos: [b, cmpBlockSize, dr], fp16/bf16
    mlpSin: [b, cmpBlockSize, dR], fp16/bf16
    cmpAttnOut: [b*s1*n1, dN], fp32
    cmpSoftmax: [b*s1*n1, scmpMax], fp32
    fullK: [], fp32
    fullV: [], fp32
    ropeOut:[], fp32
    cmpK: [], fp32
    cmpV: [], fp32
    */
    (void)cmpKvCache;
    (void)cmpKrCache;
    (void)actCmpSeqLen;
    (void)firstRopeInput;

    auto qDtype = qNope.GetStorage()->Datatype();
    auto kDtype = kvCache.GetStorage()->Datatype();

    const int b = blockTable.GetShape()[NUM_VALUE_0];
    const int s1 = qNope.GetShape()[NUM_VALUE_0] / b / n1;

    const int dN = qNope.GetShape()[SHAPE_DIM1];
    const int dR = qRope.GetShape()[SHAPE_DIM1];
    const int dQ = dN + dR;
    const int dK = dQ;
    const int maxBlockNum = blockTable.GetShape()[SHAPE_DIM1];
    const int maxCmpBlockNum = cmpBlockTable.GetShape()[SHAPE_DIM1];
    int group = n1 / n2;
    int n = NUM_128;                           // NUM_128
    int s_cmp = cmpSoftmax.GetShape()[SHAPE_DIM1]; // 511
    int s_slc = (s_cmp + 3) / 4;               // NUM_128
    int loop = s_slc;
    int out_loop = 4;    // 4
    int actualTopk = 13; // 13

    LOOP("COMPRESS_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b), {}, true) {
        // Construct align-shape tensors for store dynamic seqs
        Tensor kvTensor(kDtype, {maxBlockNum * blockSize, n2, dN}, "kvTensor");
        Tensor krTensor(kDtype, {maxBlockNum * blockSize, n2, dR}, "krTensor");
        Tensor kCmpTensor(kDtype, {maxCmpBlockNum * blockSize, n2, dK}, "kCmpTensor");
        Tensor tmpOut(DataType::DT_FP32, {1, s_slc}, "tmpout");
        Tensor softmaxTmp(DataType::DT_FP32, {n, s_cmp}, "softmaxTmp");
        auto curKvLen = GetTensorData(actSeqLen, {bIdx});
        auto blockLoop = (curKvLen + blockSize - NUM_VALUE_1) / blockSize;
        auto actualVaildLen = (((curKvLen - NUM_32) / 16 + 1) + 3) / 4 - 3; // 125
        // Concat All Blocks
        LOOP("CMP_LOOP_BLOCKNUM", FunctionType::DYNAMIC_LOOP, blockIdx, LoopRange(blockLoop), {}, true) {
            config::SetSemanticLabel("BeforeBlockConcat");
            SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx});
            auto curKv = View(kvCache, {blockSize, n2 * dN}, {curBlockIdx * blockSize, 0});
            auto curKr = View(krCache, {blockSize, n2 * dR}, {curBlockIdx * blockSize, 0});
            TileShape::Current().SetVecTile(NUM_128, NUM_64);
            auto curkv1 = Cast(curKv, DT_FP32);
            auto curkr1 = Cast(curKr, DT_FP32);
            TileShape::Current().SetVecTile(NUM_128, 1, NUM_64);
            auto curKvRe = Reshape(curkv1, {blockSize, n2, dN});
            auto curkrRe = Reshape(curkr1, {blockSize, n2, dR});
            auto curKv2 = Cast(curKvRe, kDtype);
            auto curKr2 = Cast(curkrRe, kDtype);

            Assemble(curKv2, {blockIdx * blockSize, 0, 0}, kvTensor);
            Assemble(curKr2, {blockIdx * blockSize, 0, 0}, krTensor);
            Assemble(curKv2, {blockIdx * blockSize, 0, 0}, fullK);
            Assemble(curKr2, {blockIdx * blockSize, 0, dN}, fullK);
            config::SetSemanticLabel("AfterBlockConcat");
        }
        // Kv Compress
        auto mlpLoop = (curKvLen - cmpBlockSize) / cmpStride + NUM_VALUE_1;
        LOOP("MLP_LOOP_STRIDE", FunctionType::DYNAMIC_LOOP, cmpIdx, LoopRange(mlpLoop), {}, true) {
            auto kvTmp = View(kvTensor, {cmpBlockSize, n2, dN}, {cmpIdx * cmpStride, 0, 0});
            auto krTmp = View(krTensor, {cmpBlockSize, n2, dR}, {cmpIdx * cmpStride, 0, 0});
            auto cosTmp = View(mlpCos, {NUM_VALUE_1, cmpBlockSize, dR}, {bIdx, 0, 0});
            auto sinTmp = View(mlpSin, {NUM_VALUE_1, cmpBlockSize, dR}, {bIdx, 0, 0});
            TileShape::Current().SetVecTile(NUM_32, 1, NUM_64);
            auto kvTmp1 = Cast(kvTmp, DT_FP32);
            auto kvTmp2 = Cast(kvTmp1, kDtype);
            auto krTmp1 = Cast(krTmp, DT_FP32);
            auto krTmp2 = Cast(krTmp1, kDtype);

            // LocalRope
            config::SetSemanticLabel("MlpLocalRope");
            auto krRope = MlpSingleRope(krTmp2, cosTmp, sinTmp, tileConfig.mlpRopeTile); // (cmpBlockSize, n2, dR)

            TileShape::Current().SetVecTile(1, NUM_32, 1, NUM_64);
            auto krRopeTmp = Reshape(krRope, {1, cmpBlockSize, n2, dR});
            krRopeTmp = Cast(krRopeTmp, DT_FP32);
            krRopeTmp = Cast(krRopeTmp, kDtype);
            Assemble(krRopeTmp, {cmpIdx, 0, 0, 0}, firstRope);
            TileShape::Current().SetVecTile(NUM_32, 1, NUM_64);

            // Mlp
            Tensor kCat(kDtype, {cmpBlockSize, n2, dN + dR}, "kConcat");
            Assemble(kvTmp2, {0, 0, 0}, kCat);
            Assemble(krRope, {0, 0, dN}, kCat);
            config::SetSemanticLabel("MlpCompress");
            auto kMlp = MlpCompress(kCat, mlpWk1, mlpWk2, tileConfig.mlpCmpTile); // (1, n2, dK)
            Assemble(kMlp, {cmpIdx, 0, 0}, kCmpTensor); // (maxCmpBlockSize * blockSize, n2 * dK)
            auto MlpReshape = Reshape(kMlp, {1, 1, n2, dK});
            TileShape::Current().SetVecTile(1, 1, 1, NUM_128);
            auto kMlpCast = Cast(MlpReshape, DT_FP32);
            Assemble(kMlpCast, {bIdx, cmpIdx, 0, 0}, cmpK);
        }
        // Compress Attention
        LOOP("CMP_ATTN_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1), {}, true) {
            LOOP("CMP_ATTN_LOOP_N2", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(n2), {}, true) {
                auto qOffset = bIdx * s1 * n1 + s1Idx * n1 + n2Idx * group;
                Tensor curQAttn(qDtype, {group, dN + dR}, "query"); // (g, dQ)
                auto curQn = View(qNope, {group, dN}, {group, dN}, {qOffset, 0});
                auto curQr = View(qRope, {group, dN}, {group, dR}, {qOffset, 0});
                TileShape::Current().SetVecTile(NUM_16, NUM_64);
                auto qnCast1 = Cast(curQn, DT_FP32);
                auto qnCast2 = Cast(qnCast1, qDtype);
                auto qrCast1 = Cast(curQr, DT_FP32);
                auto qrCast2 = Cast(qrCast1, qDtype);
                Assemble(qnCast2, {0, 0}, curQAttn);
                Assemble(qrCast2, {0, dN}, curQAttn);

                // MTP casual calculation for s2
                auto curOffset = s1 - s1Idx - 1;
                auto effSeq = (curKvLen - curOffset - cmpBlockSize) / cmpStride + NUM_VALUE_1;
                auto curK = View(kCmpTensor, {maxCmpBlockNum * blockSize, 1, dK},
                    {std::min(effSeq, maxCmpBlockNum * blockSize), 1, dK}, {0, n2Idx, 0}); // (effSeq, dK)
                auto curV = View(kCmpTensor, {maxCmpBlockNum * blockSize, 1, dN},
                    {std::min(effSeq, maxCmpBlockNum * blockSize), 1, dN}, {0, n2Idx, 0}); // (effSeq, dN)

                TileShape::Current().SetVecTile(NUM_128, 1, NUM_128);
                auto curKCast = Cast(curK, DT_FP32);
                auto curVCast = Cast(curV, DT_FP32);

                auto curKRe = Reshape(curKCast, {maxCmpBlockNum * blockSize, dK}, {effSeq, dK});
                auto curVRe = Reshape(curVCast, {maxCmpBlockNum * blockSize, dN}, {effSeq, dN});

                TileShape::Current().SetVecTile(NUM_128, NUM_128);

                auto curKCast2 = Cast(curKRe, kDtype);
                auto curVCast2 = Cast(curVRe, kDtype);

                auto res = CmpAttn(curQAttn, curKCast2, curVCast2, softmaxScale, tileConfig.attnTile);
                auto curSoftmax = std::get<0>(res);                    // (g, effSeq)
                auto curRes = std::get<1>(res);                        // (g, dN)
                Assemble(curSoftmax, {qOffset, 0}, cmpSoftmax);       // (b*s1*n1, sCmpMax)
                Assemble(curRes, {qOffset, 0}, cmpAttnOut);           // (b*s1*n1, dN)
                Assemble(curSoftmax, {n2Idx * group, 0}, softmaxTmp); // (b*s1*n1, sCmpMax)

                auto curRes_tmp = Reshape(curRes, {1, 1, n1, dN});
                TileShape::Current().SetVecTile({1, 1, 16, dN});
                auto a = Cast(curRes_tmp, DT_FP16);
                Assemble(a, {bIdx, s1Idx, 0, 0}, cmpAttnOut16);
            }

            LOOP("CMP_LOOP_L0_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
                (void)sIdx;
                TileShape::Current().SetVecTile({4, s_cmp});
                auto viewer = View(softmaxTmp, {n, s_cmp}, {0, 0});
                auto input32 = Cast(viewer, DataType::DT_FP32); // NUM_128,511
                auto tmpTrans = Transpose(input32, {0, 1});     // 511,NUM_128
                TileShape::Current().SetVecTile({16, n});
                Tensor abc(DT_FP32, {loop, n}, "reduce0");
                for (int i = 0; i < loop; i++) {
                    auto maxLen0 = std::min(out_loop, s_cmp - i * out_loop);
                    auto view0 = View(tmpTrans, {maxLen0, n}, {i * out_loop, 0}); // 4,NUM_128
                    auto maxLen1 = std::min(out_loop, s_cmp - i * out_loop - 1);
                    TileShape::Current().SetVecTile({8, n});
                    auto reduce0 = Sum(view0, 0, true); // 1,NUM_128
                    if (maxLen1 > 0) {
                        auto view1 = View(tmpTrans, {maxLen1, n}, {i * out_loop + 1, 0}); // 4,NUM_128
                        auto reduce1 = Sum(view1, 0, true);                            // 1,NUM_128
                        auto sum = Add(reduce0, reduce1);                                 // 1,NUM_128
                        Assemble(sum, {i, 0}, abc);
                    } else {
                        Assemble(reduce0, {i, 0}, abc);
                    }
                }
                auto trans1 = Transpose(abc, {0, 1}); // NUM_128,NUM_128
                TileShape::Current().SetVecTile({n, 8});
                auto reduce2 = Sum(trans1, 0, true); // 1,NUM_128
                tmpOut = Reshape(reduce2, {1, s_slc});

                TileShape::Current().SetVecTile({1, 16});
                auto a = Add(tmpOut, Element(DT_FP32, 0.0f));
                Assemble(a, {bIdx, 0}, topkInput);
            }

            LOOP("CMP_LOOP_topk1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
                (void)sIdx;

                std::vector<Tensor> res = GenTopkIndices(tmpOut, s_slc, actualTopk, actualVaildLen, true);

                auto tmp = Reshape(res[1], {1, 1, 16});
                TileShape::Current().SetVecTile({1, 1, 16});
                tmp = Cast(tmp, DT_INT32);
                Assemble(tmp, {bIdx, s1Idx, 0}, topkRes);
            }
        }
    }
}

void FusedCompressKvSelect(const Tensor &qNope, const Tensor &qRope, const Tensor &kvCache, const Tensor &krCache,
    const Tensor &cmpKvCache, const Tensor &cmpKrCache, const Tensor &blockTable, const Tensor &cmpBlockTable,
    const Tensor &actSeqLen, const Tensor &actCmpSeqLen, const Tensor &mlpWk1, const Tensor &mlpWk2,
    const Tensor &mlpCos, const Tensor &mlpSin, Tensor &cmpAttnOut, Tensor &cmpAttnOut16, Tensor &cmpSoftmax,
    Tensor &fullK, Tensor &cmpK, Tensor &firstRope, Tensor &firstRopeInput, Tensor &topkRes, Tensor &topkInput,
    const int blockSize, const int cmpBlockSize, const int cmpStride, const float softmaxScale, const int n1,
    const int n2, CmpAttnTile &tileConfig) {
    FUNCTION("FusedCompressKvSelect",
        {qNope, qRope, kvCache, krCache, cmpKvCache, cmpKrCache, blockTable, cmpBlockTable, actSeqLen, actCmpSeqLen,
            mlpWk1, mlpWk2, mlpCos, mlpSin},
        {cmpAttnOut, cmpAttnOut16, cmpSoftmax, fullK, cmpK, firstRope, firstRopeInput, topkRes, topkInput}) {
        FusedCompressKvSelectCompute(qNope, qRope, kvCache, krCache, cmpKvCache, cmpKrCache, blockTable, cmpBlockTable,
            actSeqLen, actCmpSeqLen, mlpWk1, mlpWk2, mlpCos, mlpSin, cmpAttnOut, cmpAttnOut16, cmpSoftmax, fullK, cmpK,
            firstRope, firstRopeInput, topkRes, topkInput, blockSize, cmpBlockSize, cmpStride, softmaxScale, n1, n2,
            tileConfig);
    }
}

} // namespace npu::tile_fwk
