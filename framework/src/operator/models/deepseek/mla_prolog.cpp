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
 * \file mla_prolog.cpp
 * \brief
 */

#include "operator/models/deepseek/mla_prolog.h"
#include "operator/models/deepseek/deepseek_mla.h"

namespace npu::tile_fwk {
std::vector<Tensor> QkvPre(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, MlaQuantInputs quantInputs,bool splitReduceLastDim, bool splitK) {
    // quant
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;
    bool hasSmooth = (smoothScalesCq.GetStorage() != nullptr);
    int b = tokenX.GetShape()[0];
    int s = tokenX.GetShape()[1];
    int h = tokenX.GetShape()[2];
    int bs = b * s;
    int q_lora_rank = wDq.GetShape()[1];

    DataType dType = tokenX.GetStorage()->Datatype();
    DataType dTypeQuantOut = isQuant ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;

    Tensor input = Reshape(tokenX, {bs, h}); // [b,s,h] -> [b*s,h]

    /******** q ********/
    int c0 = NUM_16;
    int tieM = (bs + c0 - 1) / c0 * c0;
    // [b*s,h] * [h,q_lora_rank] = [b*s,q_lora_rank]
    Tensor qMmRes;
    if (splitK) {
        TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64}, false, true);
        Tensor qMmResF32 = Matrix::Matmul(DT_FP32, input, wDq);
        TileShape::Current().SetVecTile(std::min(NUM_32, bs), NUM_128);
        qMmRes = Cast(qMmResF32, dType);
    } else {
        TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
        qMmRes = Matrix::Matmul(dType, input, wDq); // bf16
    }

    if (splitReduceLastDim) {
        TileShape::Current().SetVecTile(std::min(NUM_16, bs), NUM_128);
    } else {
        TileShape::Current().SetVecTile(std::min(NUM_8, bs), q_lora_rank);
    }

    Tensor normRes = RmsNorm(qMmRes, gammaCq, epsilonCq);
    Tensor normDequantScale;
    std::tuple<Tensor, Tensor> normQuantRes;
    if (isQuant) {
        if (hasSmooth) {
            normQuantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            normQuantRes = Quant(normRes);  // int8
        }
        normRes = std::get<0>(normQuantRes);
        normDequantScale = std::get<1>(normQuantRes);
        TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_256, NUM_256});
    } else {
        TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    }
    // [b*s,qLoraRank] * [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor q = Matrix::Matmul(dTypeQuantOut, normRes, wUqQr, false, false); // bf16  // quant: A8W8O32 -> bf16
    qkvPreRes.emplace_back(q);

    /******** kv ********/
    TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (splitK) {
        Tensor kvMmResF32 = Matrix::Matmul(DT_FP32, input, wDkvKr);
        TileShape::Current().SetVecTile(std::min(NUM_32, bs), NUM_64);
        compressedKv = Cast(kvMmResF32, dType);
    } else {
        compressedKv = Matrix::Matmul(dType, input, wDkvKr); // bf16
    }
    Tensor compressedKvRes = Reshape(compressedKv, {b, s, wDkvKr.GetShape()[1]});
    qkvPreRes.emplace_back(compressedKvRes);

    if (isQuant) {
        qkvPreRes.emplace_back(normDequantScale);
    }

    return qkvPreRes;
}

void MlaProlog(Tensor tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, MlaQuantInputs quantInputs, const RoPETileShapeConfigNew &ropeConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq, float epsilonCkv,
    std::string cacheMode, bool splitReduceLastDim, bool splitK) {
    // params check
    assert(tokenX.GetShape().size() == SHAPE_DIM3 && wUk.GetShape().size() == SHAPE_DIM3 && sin.GetShape().size() == SHAPE_DIM3);
    assert(kvCache.GetShape().size() == SHAPE_DIM4 && krCache.GetShape().size() == SHAPE_DIM4);
    assert(cacheMode == "BNSD" || cacheMode == "PA_BSND" || cacheMode == "PA_NZ");

    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
    std::cout << "isQuant +++ " << isQuant << std::endl;

    DataType dType = tokenX.GetStorage()->Datatype();
    int b = tokenX.GetShape()[0];
    int s = tokenX.GetShape()[1]; // s=1
    int bs = b * s;
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk.GetShape()[0];
    int qkNopeHeadDim = wUk.GetShape()[1];
    int kvLoraRank = wUk.GetShape()[2];
    int qkRopeHeadDim = sin.GetShape()[2]; // [b,s,qkRopeHeadDim]
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    auto qKv = QkvPre(tokenX, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs, splitReduceLastDim, splitK);
    Tensor q = qKv[0];     // [b*s, n*qHeadDim]
    Tensor kvTmp = qKv[1]; // [b,s,kvLoraRank+qkRopeHeadDim]

    // dequant: int32 -> fp32 -> *scale -> fp16/bf16
    if (isQuant) {
        std::vector<int64_t> tileShape = {bs, NUM_64};
        TileShape::Current().SetVecTile(tileShape);
        auto qTmpFp32 = Cast(q, DataType::DT_FP32);
        auto qTmpDequantScale = qKv[2];
        auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpDequantScale);
        auto qTmpDequantChannel = Mul(qTmpDequantPerToken, dequantScaleWUqQr);

        q = Cast(qTmpDequantChannel, dType);
    }
    auto qTmp = Reshape(q, {b, s, n, qHeadDim});
    std::vector<int64_t> tileShape = {b, 1, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);

    /******** q ********/
    Tensor qNope = View(qTmp, {b, s, n, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
                                                                      // {NUM_2, 1, NUM_32, NUM_128}
    tileShape = {b, 1, 1, NUM_128};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeRes = Reshape(qNope, {bs, n, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
    tileShape = {bs, 1, qkNopeHeadDim};     // {NUM_2, NUM_32, qkNopeHeadDim}
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeTrans = Transpose(qNopeRes, {0, 1}); // [n,bs,qkNopeHeadDim]

    int c0 = NUM_16;
    int m = (bs + c0 - 1) / c0 * c0;
    TileShape::Current().SetCubeTile({m, m}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

    tileShape = {1, bs, kvLoraRank}; // {NUM_16, NUM_2, kvLoraRank}
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1});       // [bs,n,kvLoraRank]
    queryOut = Reshape(qNopeNewTrans, {b, s, n, kvLoraRank}); // [b,s,n,kvLoraRank], output1

    /******** kv ********/
    Tensor compressedKv = View(kvTmp, {b, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_512};
    TileShape::Current().SetVecTile(tileShape);
    Tensor compressedKvNorm = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b,s,kvLoraRank]

    /******** RoPE ********/
    Tensor qPe = View(qTmp, {b, s, n, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim}); // [b,s,n,qkRopeHeadDim]
    Tensor kPe = View(kvTmp, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank});         // [b,s,qkRopeHeadDim]
    tileShape = {bs, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    Tensor kPeRes = Reshape(kPe, {b, s, 1, qkRopeHeadDim}); // [b,s,1,qkRopeHeadDim]

    Tensor kRope(kPeRes.GetStorage()->Datatype(), {b, s, 1, qkRopeHeadDim}, "kRope"); // [b,1,s,qkRopeHeadDim]
    // queryRopeOut: [b,s,n,qkRopeHeadDim], output2
    ApplyRotaryPosEmbV2(qPe, kPeRes, cos, sin, queryRopeOut, kRope, 2, ropeConfig);

    if (cacheMode == "PA_BSND") {
        int blockNum = kvCache.GetShape()[0];
        int blockSize = kvCache.GetShape()[1];
        int n2 = kvCache.GetShape()[2];
        Tensor kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank});
        Tensor krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim});
        Tensor kNope = Reshape(compressedKvNorm, {b * s, kvLoraRank});       // [b*s,kvLoraRank]
        Tensor kRopeRes = Reshape(kRope, {b * s * 1, qkRopeHeadDim});

        /******** kvCache ********/
        tileShape = {1, kvLoraRank};
        TileShape::Current().SetVecTile(tileShape);
        // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
        Tensor kvCacheUpdate = ScatterUpdate(kvCacheRes, cacheIndex, kNope, -2, cacheMode);
        kvCacheOut = Reshape(kvCacheUpdate, {blockNum, blockSize, n2, kvLoraRank});

        /******** krCache ********/
        tileShape = {1, qkRopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);
        // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
        Tensor krCacheUpdate = ScatterUpdate(krCacheRes, cacheIndex, kRopeRes, -2, cacheMode);
        krCacheOut = Reshape(krCacheUpdate, {blockNum, blockSize, n2, qkRopeHeadDim});
    } else {
        Tensor kNope = Reshape(compressedKvNorm, {b, 1, s, kvLoraRank});       // [b,1,s,kvLoraRank]
        Tensor kRopeRes = Reshape(kRope, {b, 1, s, qkRopeHeadDim});

        /******** kvCache ********/
        tileShape = {1, 1, 1, kvLoraRank};
        TileShape::Current().SetVecTile(tileShape);
        // kvCache: [b,1,s2,kvLoraRank], output3
        kvCacheOut = ScatterUpdate(kvCache, cacheIndex, kNope, -2); // cacheIndex: [b,s]

        /******** krCache ********/
        tileShape = {1, 1, 1, qkRopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);
        // krCache: [b,1,s2,qkRopeHeadDim], output4
        krCacheOut = ScatterUpdate(krCache, cacheIndex, kRopeRes, -2); // cacheIndex: [b,s]
    }
}

} // namespace npu::tile_fwk
