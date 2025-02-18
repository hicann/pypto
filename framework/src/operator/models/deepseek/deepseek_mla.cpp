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
 * \file deepseek_mla.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"

namespace npu::tile_fwk {
DeepseekAttention::DeepseekAttention(
    std::map<std::string, std::variant<bool, int, float, std::string>> config, AttentionW aw, const int inLayerIdx)
    : layerIdx(inLayerIdx) {
    attentionDropout = std::get<int>(config["attentionDropout"]);
    hiddenSize = std::get<int>(config["hiddenSize"]);
    numHeads = std::get<int>(config["numAttentionHeads"]);
    maxPositionEmbeddings = std::get<int>(config["maxPositionEmbeddings"]);
    ropeTheta = std::get<int>(config["ropeTheta"]);
    qLoraRank = std::get<int>(config["qLoraRank"]);
    qkRopeHeadDim = std::get<int>(config["qkRopeHeadDim"]);
    kvLoraRank = std::get<int>(config["kvLoraRank"]);
    vHeadDim = std::get<int>(config["vHeadDim"]);
    qkNopeHeadDim = std::get<int>(config["qkNopeHeadDim"]);
    qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    isCausal = true;

    qAProjW = aw.qAProjW;
    qBProjW = aw.qBProjW;
    qBProjWScale = aw.qBProjWScale;
    kvAProjWithMqaW = aw.kvAProjWithMqaW;
    kvBProjWK = aw.kvBProjWK;
    kvBProjWV = aw.kvBProjWV;
    oProjW = aw.oProjW;

    softmaxScale = static_cast<float>(1.0 / std::sqrt(qHeadDim));

    if (std::get<int>(config["ropeScaling"]) == 1) {
        int factor = 40;
        float mscale = 1.0;
        float mscaleAllDim = 1.0;
        double valuePointOne = 0.1;
        if (mscaleAllDim > 1) {
            mscale = static_cast<float>(valuePointOne * mscale * std::log(factor) + 1.0);
        }
        softmaxScale = softmaxScale * mscale * mscale;
    }
}

Tensor DeepseekAttention::Attention(Tensor q, Tensor kv, Tensor attenMask) {
    // q: [b,numHeads,s, kvLoraRank + qkRopeHeadDim]
    // kv: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    int b = q.GetShape()[0];
    int n2 = kv.GetShape()[1]; // 1
    int s1 = q.GetShape()[2];
    int s2 = kv.GetShape()[2];
    int kvLoraRankV = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    DataType dType = q.GetStorage()->Datatype();

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s1), std::min(NUM_128, s1)}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
    // TileShape::Current().SetVecTile({NUM_128, NUM_64, NUM_128, NUM_64}); //  bmm接口增加一个config参数
    //  [b,n,s1, kvLoraRank + qkRopeHeadDim] * [b,1, kvLoraRank + qkRopeHeadDim, s2] = [b,n,s1,s2]
    Tensor qk = Matrix::BatchMatmul<false, true>(dType, q, kv);
    TileShape::Current().SetVecTile({1, 1, NUM_128, NUM_64});
    Tensor qkFp32 = Cast(qk, DataType::DT_FP32);
    qkFp32 = Mul(qkFp32, Element(DataType::DT_FP32, static_cast<double>(softmaxScale)));
    qkFp32 = Add(qkFp32, attenMask);
    Tensor qk16 = Cast(qkFp32, dType);
    Tensor softmax = SoftmaxNew(qk16); // [b,n,s1,s2]
    // no drop
    Tensor v = View(kv, {b, n2, s2, kvLoraRankV}, {0, 0, 0, 0});
    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s1), std::min(NUM_128, s1)}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
    // TileShape::Current().SetVecTile({NUM_128, NUM_64, NUM_128, NUM_64}); //  bmm接口增加一个config参数
    // [b,n,s1,s2] * [b,1, s2, kvLoraRank] = [b,n,s1,kvLoraRank]
    Tensor attenRes = Matrix::BatchMatmul(dType, softmax, v);
    return attenRes;
}

Tensor DeepseekAttention::AttentionPost(Tensor attenRes) {
    // attenRes: [b,n,s,kvLoraRank]
    int b = attenRes.GetShape()[0];
    int n = attenRes.GetShape()[1];
    int s = attenRes.GetShape()[2];
    int bs = b * s;
    DataType dType = attenRes.GetStorage()->Datatype();

    TileShape::Current().SetVecTile({1, 1, 1, NUM_512});
    Tensor attenRes0 = Transpose(attenRes, {1, 2});
    TileShape::Current().SetVecTile({1, 1, NUM_128, NUM_64});
    Tensor attenRes1 = Reshape(attenRes0, {b * s, n, kvLoraRank});
    TileShape::Current().SetVecTile({1, 1, NUM_512});
    Tensor attenRes2 = Transpose(attenRes1, {0, 1});
    TileShape::Current().SetVecTile({1, NUM_128, NUM_64});
    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    TileShape::Current().SetVecTile(NUM_128, NUM_64);
    // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
    Tensor mm7Res = Matrix::BatchMatmul(dType, attenRes2, kvBProjWV);

    TileShape::Current().SetVecTile(1, 1, NUM_128);
    Tensor mm7Res1 = Transpose(mm7Res, {0, 1}); // [bs,n,vHeadDim]
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    Tensor mm7Res2 = Reshape(mm7Res1, {b, s, n * vHeadDim});

    TileShape::Current().SetVecTile(NUM_128, NUM_64);
    // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
    Tensor attnOutW = Unsqueeze(oProjW, 0);
    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    TileShape::Current().SetVecTile(NUM_128, NUM_64);
    Tensor attenOutput = Matrix::BatchMatmul(dType, mm7Res2, attnOutW);

    return attenOutput;
}

Tensor DeepseekAttention::AttentionPost2(Tensor attenRes) {
    // attenRes: [b,n,s,kvLoraRank]
    int b = attenRes.GetShape()[0];
    int n = attenRes.GetShape()[1];
    int s = attenRes.GetShape()[2];
    int bs = b * s;
    int h = oProjW.GetShape()[1];
    DataType dType = attenRes.GetStorage()->Datatype();

    TileShape::Current().SetVecTile({NUM_16, NUM_16, 1, NUM_128});
    Tensor attenRes0 = Transpose(attenRes, {1, 2});
    TileShape::Current().SetVecTile({NUM_16, 1, NUM_16, NUM_128});
    Tensor attenRes1 = Reshape(attenRes0, {b * s, n, kvLoraRank});
    TileShape::Current().SetVecTile({NUM_16, NUM_16, NUM_128});
    Tensor attenRes2 = Transpose(attenRes1, {0, 1});
    TileShape::Current().SetCubeTile({std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128},
        {std::min(NUM_128, h), std::min(NUM_128, h)});
    // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
    Tensor mm7Res = Matrix::BatchMatmul(dType, attenRes2, kvBProjWV);

    TileShape::Current().SetVecTile(NUM_16, NUM_16, NUM_128);
    Tensor mm7Res1 = Transpose(mm7Res, {0, 1}); // [bs,n,vHeadDim]
    TileShape::Current().SetVecTile(NUM_16, NUM_16, NUM_128);
    Tensor mm7Res2 = Reshape(mm7Res1, {b, s, n * vHeadDim});

    TileShape::Current().SetVecTile(NUM_128, std::min(NUM_256, h));
    // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
    Tensor attnOutW = Unsqueeze(oProjW, 0);
    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {std::min(NUM_128, h), std::min(NUM_128, h)});
    Tensor attenOutput = Matrix::BatchMatmul(dType, mm7Res2, attnOutW);

    return attenOutput;
}

std::tuple<Tensor, Tensor> DeepseekAttention::QkvPre(Tensor hiddenStates) {
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    DataType dType = hiddenStates.GetStorage()->Datatype();

    TileShape::Current().SetVecTile(NUM_128, NUM_64);
    Tensor qAProjW1 = Unsqueeze(qAProjW, 0);
    Tensor qBProjW1 = Unsqueeze(qBProjW, 0);
    Tensor kvAProjWithMqaW1 = Unsqueeze(kvAProjWithMqaW, 0);

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    TileShape::Current().SetVecTile(NUM_128, NUM_64); // for Assemble
    // qAProj, bmm: (b, s, h) * (1, h, qLoraRank) = (b, s, qLoraRank)
    Tensor qAProj = Matrix::BatchMatmul(dType, hiddenStates, qAProjW1); // bf16

    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    Tensor qALayerNorm = RmsNorm(qAProj);

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    TileShape::Current().SetVecTile(NUM_128, NUM_64); // for Assemble
    // q_b_proj, bmm: (b, s, qLoraRank) * (qLoraRank, numHeads * qHeadDim) = (b, s, numHeads * qHeadDim)
    Tensor q = Matrix::BatchMatmul(dType, qALayerNorm, qBProjW1);

    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    // (b, s, numHeads, qHeadDim)
    Tensor q2 = Reshape(q, {b, s, numHeads, qHeadDim});

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    TileShape::Current().SetVecTile(NUM_128, NUM_64); // for Assemble
    // kv_a_proj_with_mqa, bmm: (b, s, h) * (h, kvLoraRank + qkRopeHeadDim) = (b, s, kvLoraRank +
    // qkRopeHeadDim)
    Tensor compressedKv = Matrix::BatchMatmul(dType, hiddenStates, kvAProjWithMqaW1); // bf16

    return std::tie(q2, compressedKv);
}

std::tuple<Tensor, Tensor> DeepseekAttention::QkvPreCv(Tensor hiddenStates) {
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    DataType dType = hiddenStates.GetStorage()->Datatype();

    TileShape::Current().SetVecTile(NUM_128, NUM_64);
    Tensor qAProjW1 = Unsqueeze(qAProjW, 0); // [NUM_256,NUM_512]
    Tensor qBProjW1 = Unsqueeze(qBProjW, 0); // [NUM_512,2*192]
    Tensor kvAProjWithMqaW1 = Unsqueeze(kvAProjWithMqaW, 0); // [NUM_256,576]

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    // qAProj, bmm: (b, s, h) * (1, h, qLoraRank) = (b, s, qLoraRank)
    // [2,1,256] * [1,256,512]
    Tensor qAProj = Matrix::BatchMatmul(dType, hiddenStates, qAProjW1); // bf16  2_1_512

    TileShape::Current().SetVecTile(NUM_2, 1, NUM_512);
    Tensor qALayerNorm = RmsNorm(qAProj); // 2_1_512

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    // q_b_proj, bmm: (b, s, qLoraRank) * (qLoraRank, numHeads * qHeadDim) = (b, s, numHeads * qHeadDim)
    // 2_1_512 * 1_512_2*192
    // 2_1_2*192
    Tensor q = Matrix::BatchMatmul(dType, qALayerNorm, qBProjW1);

    TileShape::Current().SetVecTile(NUM_2, 1, NUM_384);
    // (b, s, numHeads, qHeadDim) // 2_1_2_192
    Tensor q2 = Reshape(q, {b, s, numHeads, qHeadDim});

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    // kv_a_proj_with_mqa, bmm: (b, s, h) * (h, kvLoraRank + qkRopeHeadDim) = (b, s, kvLoraRank +
    // qkRopeHeadDim)
    // 2_1_256  1_256_576
    Tensor compressedKV = Matrix::BatchMatmul(dType, hiddenStates, kvAProjWithMqaW1); // bf16
    // 2_1_32_192 2_1_576
    return std::tie(q2, compressedKV);
}

std::vector<Tensor> DeepseekAttention::QkvPre2(Tensor hiddenStates, bool isQuant) {
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int h = hiddenStates.GetShape()[2];
    int bs = b * s;

    DataType dType = hiddenStates.GetStorage()->Datatype();
    DataType dTypeQuantOut = isQuant ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPre2Res;

    Tensor input = Reshape(hiddenStates, {bs, h});  // [b,s,h] -> [b*s,h]

    int c0 = NUM_16;
    int m = (std::min(NUM_32, bs) + c0 - 1) / c0 * c0;
    int tileM = std::min(NUM_16, m);
    TileShape::Current().SetCubeTile({tileM, tileM}, {NUM_256, NUM_256}, {NUM_128, NUM_128});
    // [b*s,h] * [h,qLoraRank] = [b*s,qLoraRank]
    Tensor qAProj = Matrix::Matmul<false, false>(dType, input, qAProjW);  // bf16

    TileShape::Current().SetVecTile(std::min(NUM_16, bs), NUM_128);
    Tensor qAProjNorm = RmsNorm(qAProj);

    Tensor qAProjNormScaleDequant;
    if (isQuant) {
        auto qAProjNormQuantRes = Quant(qAProjNorm);    //int8
        qAProjNorm = std::get<0>(qAProjNormQuantRes);
        qAProjNormScaleDequant = std::get<1>(qAProjNormQuantRes);
        TileShape::Current().SetCubeTile({tileM, tileM}, {NUM_256, NUM_256}, {NUM_256, NUM_256});
    } else {
        TileShape::Current().SetCubeTile({m, m}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    }
    // [b*s,qLoraRank] * [qLoraRank, N*qHeadDim] = [b*s, N*qHeadDim]
    Tensor q = Matrix::Matmul<false, false>(dTypeQuantOut, qAProjNorm, qBProjW);  // bf16  // quant  A8W8O32  ->  bf16
    qkvPre2Res.emplace_back(q);

    TileShape::Current().SetCubeTile({m, m}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv = Matrix::Matmul<false, false>(dType, input, kvAProjWithMqaW);  // bf16
    Tensor compressedKvRes = Reshape(compressedKv, {b, s, kvLoraRank + qkRopeHeadDim});
    qkvPre2Res.emplace_back(compressedKvRes);

    if (isQuant) {
        qkvPre2Res.emplace_back(qAProjNormScaleDequant);
    }

    return qkvPre2Res;
}

std::tuple<Tensor, Tensor> DeepseekAttention::QkvPreFp32(Tensor hiddenStates) {
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int h = hiddenStates.GetShape()[2];
    int bs = b * s;
    DataType dType = hiddenStates.GetStorage()->Datatype();

    Tensor input = Reshape(hiddenStates, {bs, h});  // [b,s,h] -> [b*s,h]

    TileShape::Current().SetCubeTile(
        {std::min(NUM_64, bs), std::min(NUM_64, bs)}, {NUM_256, NUM_256}, {NUM_128, NUM_128});
    // [b*s,h] * [h,qLoraRank] = [b*s,qLoraRank]
    // [NUM_32*1,NUM_256] * [NUM_256,NUM_512] = [NUM_32*1,NUM_512]
    Tensor qAProjFp32 = Matrix::Matmul<false, false>(DataType::DT_FP32, input, qAProjW);  // fp32

    TileShape::Current().SetVecTile(NUM_32, NUM_128);
    Tensor qAProjNormFp32 = RmsNorm(qAProjFp32);  // fp32

    std::vector<int64_t> tileShape = {NUM_32, NUM_128};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qAProjNorm = Cast(qAProjNormFp32, dType);  // bf16

    TileShape::Current().SetCubeTile(
        {std::min(NUM_64, bs), std::min(NUM_64, bs)}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,qLoraRank] * [qLoraRank, N*qHeadDim] = [b*s, N*qHeadDim]
    // [NUM_32*1,NUM_512] * [NUM_512, 2*192] = [NUM_32*1, 2*192]
    Tensor qFp32 = Matrix::Matmul<false, false>(DataType::DT_FP32, qAProjNorm, qBProjW);  // fp32
    Tensor qRes = Reshape(qFp32, {b, s, numHeads, qHeadDim});

    TileShape::Current().SetCubeTile(
        {std::min(NUM_64, bs), std::min(NUM_64, bs)}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    // [NUM_32*1,NUM_256] * [NUM_256,NUM_512+NUM_64] = [NUM_32*1,NUM_512+NUM_64]
    Tensor compressedKvFp32 = Matrix::Matmul<false, false>(DataType::DT_FP32, input, kvAProjWithMqaW);  // fp32
    Tensor compressedKvRes = Reshape(compressedKvFp32, {b, s, kvLoraRank + qkRopeHeadDim});

    return std::tie(qRes, compressedKvRes);
}


// mm/bmm: bf16 in, bf16 out
Tensor DeepseekAttention::Forward(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin,
    Tensor kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig) {
    // hiddenStates: (b,s,h), attention_mask: (b,1,s,s2), positionIds: (b,s)
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int bs = b * s;
    DataType dType = hiddenStates.GetStorage()->Datatype();

    /*** prepare q k v ***/
    auto qKv = QkvPre(hiddenStates);
    Tensor q = std::get<0>(qKv);
    Tensor compressedKv = std::get<1>(qKv);

    Tensor qNope = View(q, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0});
    Tensor qPe = View(q, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
    TileShape::Current().SetVecTile(1, 1, 1, NUM_64);
    qPe = Transpose(qPe, {1, 2}); // setTileShapes 4维
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);

    // 先View kPe,防止compress_kv变化影响k_pe
    Tensor kPe = View(compressedKv, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // (b,s,qkRopeHeadDim)
    compressedKv = View(compressedKv, {b, s, kvLoraRank}, {0, 0, 0});              // (b,s,kvLoraRank)
    // [b,s, qkRopeHeadDim] -> [b,s, 1,qkRopeHeadDim] -> [b,1,s,qkRopeHeadDim]
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    kPe = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // setTileShapes 4维

    TileShape::Current().SetVecTile(1, NUM_128, 1, NUM_64); // SetVecTileShapes(1, 1, NUM_128, NUM_64)
    // (b, s, n, qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) -> (b, s, numHeads, kvLoraRank)
    Tensor qNope1 = Reshape(qNope, {b * s, numHeads, qkNopeHeadDim});
    TileShape::Current().SetVecTile(1, 1, NUM_128);
    Tensor qNope2 = Transpose(qNope1, {0, 1}); // (n,bs,d)
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    TileShape::Current().SetVecTile(NUM_128, NUM_64); // for Assemble
    // bmm: (n,bs,d) * (n, d, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNope2, kvBProjWK);
    TileShape::Current().SetVecTile(1, 1, NUM_512);
    Tensor qNopeNew2 = Transpose(qNopeNew, {0, 1});
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    qNopeNew2 = Reshape(qNopeNew2, {b, s, numHeads, kvLoraRank});
    TileShape::Current().SetVecTile(1, 1, 1, NUM_512);
    qNopeNew2 = Transpose(qNopeNew2, {1, 2});
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64); // (b,n,s,kvLoraRank)

    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    Tensor kNope = RmsNorm(compressedKv); // (b,s,kvLoraRank)
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    kNope = Reshape(kNope, {b, 1, s, kvLoraRank}); // (b,1,s,kvLoraRank)

    Tensor qPeRope(qPe.GetStorage()->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope");
    // (b,numHeads,s,qkRopeHeadDim)
    Tensor kPeRope(kPe.GetStorage()->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // (b,1,s,qkRopeHeadDim)
    ApplyRotaryPosEmb(qPe, kPe, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);
    TileShape::Current().SetVecTile(1, 1, NUM_128, NUM_64);

    Tensor queryStates = Cat({qNopeNew2, qPeRope}, -1); // (b,numHeads,s, kvLoraRank + qkRopeHeadDim)
    Tensor keyStates = Cat({kNope, kPeRope}, -1);        // (b,1,s, kvLoraRank + qkRopeHeadDim)

    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    auto pastKeyStatesNew = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // 增量
    //
    Tensor attenRes = Attention(queryStates, pastKeyStatesNew, attenMask); // 增量

    return AttentionPost(attenRes);
}

// mm/bmm: bf16 in, bf16 out
std::tuple<Tensor, Tensor>  DeepseekAttention::AtentionPreForward(Tensor hiddenStates, Tensor attenMask,
    Tensor positionIds, Tensor cos, Tensor sin, Tensor kvLen, Tensor pastKeyStates,
    const RoPETileShapeConfig &ropeTileShapeConfig) {
    (void)attenMask;
    // hiddenStates: (b,s,h), attention_mask: (b,1,s,s2), positionIds: (b,s)
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int bs = b * s;
    DataType dType = hiddenStates.GetStorage()->Datatype();

    /*** prepare q k v ***/
    auto qKv = QkvPre(hiddenStates);
    Tensor q = std::get<0>(qKv);
    Tensor compressedKv = std::get<1>(qKv);

    Tensor qNope = View(q, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0});
    Tensor qPe = View(q, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
    TileShape::Current().SetVecTile(1, 1, 1, NUM_64);
    qPe = Transpose(qPe, {1, 2}); // setTileShapes 4维
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);

    // 先View kPe,防止compress_kv变化影响k_pe
    Tensor kPe = View(compressedKv, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // (b,s,qkRopeHeadDim)
    compressedKv = View(compressedKv, {b, s, kvLoraRank}, {0, 0, 0});              // (b,s,kvLoraRank)
    // [b,s, qkRopeHeadDim] -> [b,s, 1,qkRopeHeadDim] -> [b,1,s,qkRopeHeadDim]
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    kPe = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // setTileShapes 4维

    TileShape::Current().SetVecTile(1, NUM_128, 1, NUM_64); // SetVecTileShapes(1, 1, NUM_128, NUM_64)
    // (b, s, n, qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) -> (b, s, numHeads, kvLoraRank)
    Tensor qNope1 = Reshape(qNope, {b * s, numHeads, qkNopeHeadDim});
    TileShape::Current().SetVecTile(1, 1, NUM_128);
    Tensor qNope2 = Transpose(qNope1, {0, 1}); // (n,bs,d)
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    TileShape::Current().SetVecTile(NUM_128, NUM_64); // for Assemble
    // bmm: (n,bs,d) * (n, d, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNope2, kvBProjWK);
    TileShape::Current().SetVecTile(1, 1, NUM_512);
    Tensor qNopeNew2 = Transpose(qNopeNew, {0, 1});
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    qNopeNew2 = Reshape(qNopeNew2, {b, s, numHeads, kvLoraRank});
    TileShape::Current().SetVecTile(1, 1, 1, NUM_512);
    qNopeNew2 = Transpose(qNopeNew2, {1, 2});
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64); // (b,n,s,kvLoraRank)

    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    Tensor kNope = RmsNorm(compressedKv); // (b,s,kvLoraRank)
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);
    kNope = Reshape(kNope, {b, 1, s, kvLoraRank}); // (b,1,s,kvLoraRank)

    Tensor qPeRope(qPe.GetStorage()->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope");
    // (b,numHeads,s,qkRopeHeadDim)
    Tensor kPeRope(kPe.GetStorage()->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // (b,1,s,qkRopeHeadDim)
    ApplyRotaryPosEmb(qPe, kPe, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);
    TileShape::Current().SetVecTile(1, 1, NUM_128, NUM_64);

    Tensor queryStates = Cat({qNopeNew2, qPeRope}, -1); // (b,numHeads,s, kvLoraRank + qkRopeHeadDim)
    Tensor keyStates = Cat({kNope, kPeRope}, -1);        // (b,1,s, kvLoraRank + qkRopeHeadDim)

    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    auto pastKeyStatesNew = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // 增量
    return std::tie(queryStates, pastKeyStatesNew);
    //
}

// mm/bmm: bf16 in, bf16 out
std::tuple<Tensor, Tensor>  DeepseekAttention::AtentionPreForwardCv(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, 
        Tensor cos, Tensor sin, Tensor kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig) {
    (void)attenMask;
    // hiddenStates: (b,s,h), attention_mask: (b,1,s,s2), positionIds: (b,s)
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int bs = b * s;
    DataType dType = hiddenStates.GetStorage()->Datatype();

    /*** prepare q k v ***/
    // 2_1_32_192 2_1_576
    auto qKv = QkvPreCv(hiddenStates);
    Tensor q = std::get<0>(qKv); //2_1_32_192
    Tensor compressedKv = std::get<1>(qKv); //2_1_576

    Tensor qNope = View(q, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0}); // 2_1_32_128
    Tensor qPe = View(q, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim}); // 2_1_32_64
    TileShape::Current().SetVecTile(NUM_2, 1, NUM_32, NUM_64);
    qPe = Transpose(qPe, {1, 2}); // setTileShapes 4维 2_32_1_64

    // 先View kPe,防止compress_kv变化影响k_pe
    Tensor kPe = View(compressedKv, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // (b,s,qkRopeHeadDim) 2_1_64
    compressedKv = View(compressedKv, {b, s, kvLoraRank}, {0, 0, 0});              // (b,s,kvLoraRank) 2_1_512
    // [b,s, qkRopeHeadDim] -> [b,s, 1,qkRopeHeadDim] -> [b,1,s,qkRopeHeadDim]
    TileShape::Current().SetVecTile(NUM_2, 1, NUM_64);
    kPe = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // setTileShapes 4维 2_1_1_64

    TileShape::Current().SetVecTile(NUM_2, 1, NUM_32, NUM_128); // SetVecTileShapes(1, 1, NUM_128, NUM_64)
    // (b, s, n, qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) -> (b, s, numHeads, kvLoraRank)
    Tensor qNope1 = Reshape(qNope, {b * s, numHeads, qkNopeHeadDim}); // 2_32_128
    TileShape::Current().SetVecTile(NUM_2, NUM_32, NUM_128);
    Tensor qNope2 = Transpose(qNope1, {0, 1}); // (n,bs,d) 32_2_128
    TileShape::Current().SetVecTile(1, NUM_128, NUM_64);

    TileShape::Current().SetCubeTile(
        {std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,d) * (n, d, kvLoraRank) = (n, bs, kvLoraRank)
    // 32_2_128 * 32_128_512 = 32_2_512
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNope2, kvBProjWK);
    TileShape::Current().SetVecTile(NUM_16, NUM_2, NUM_512);
    Tensor qNopeNew2 = Transpose(qNopeNew, {0, 1}); // 2_32_512
    TileShape::Current().SetVecTile(1, NUM_32, NUM_512);
    qNopeNew2 = Reshape(qNopeNew2, {b, s, numHeads, kvLoraRank}); // 2_1_32_512
    TileShape::Current().SetVecTile(NUM_2, 1, NUM_32, NUM_256);
    qNopeNew2 = Transpose(qNopeNew2, {1, 2}); //2_32_1_512

    TileShape::Current().SetVecTile(NUM_2, 1, NUM_512);
    Tensor kNope = RmsNorm(compressedKv); // (b,s,kvLoraRank) 2_1_512
    TileShape::Current().SetVecTile(NUM_2, 1, NUM_512);
    kNope = Reshape(kNope, {b, 1, s, kvLoraRank}); // (b,1,s,kvLoraRank) 2_1_1_512

    Tensor qPeRope(qPe.GetStorage()->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope"); // 2_32_1_64
    // (b,numHeads,s,qkRopeHeadDim)
    Tensor kPeRope(kPe.GetStorage()->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // (b,1,s,qkRopeHeadDim) 2_1_1_64
    // 2_32_1_64  2_1_1_64  1_64  1_64  2_1  2_32_1_64  2_1_1_64
    ApplyRotaryPosEmb(qPe, kPe, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);
    TileShape::Current().SetVecTile(NUM_2, NUM_32, 1, NUM_64);
    // 2_32_1_512 + 2_32_1_64 = 2_32_1_576
    Tensor queryStates = Cat({qNopeNew2, qPeRope}, -1); // (b,numHeads,s, kvLoraRank + qkRopeHeadDim)
    //2_32_1_512 + 2_32_1_64 = 2_32_1_576
    Tensor keyStates = Cat({kNope, kPeRope}, -1);        // (b,1,s, kvLoraRank + qkRopeHeadDim)

    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    // 2_1_256_576
    TileShape::Current().SetVecTile(NUM_2, 1, NUM_128, NUM_64);
    auto pastKeyStatesNew = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // 增量
    return std::tie(queryStates, pastKeyStatesNew);
}

std::tuple<Tensor, Tensor> DeepseekAttention::MlaPrologAbForward(Tensor hiddenStates, Tensor qPeRope, bool isQuant) {
    // hiddenStates: (b,s,h), positionIds: (b,s)
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int bs = b * s;
    DataType dType = hiddenStates.GetStorage()->Datatype();

    auto qKv = QkvPre2(hiddenStates, isQuant);
    Tensor q = qKv[0];     // [b,s,n,qHeadDim]
    Tensor kvTmp = qKv[1];    // [b,s,kvLoraRank+qkRopeHeadDim]

    //dequant int32 -> fp32  -> *scale  -> fp16/bf16
    if (isQuant) {
        std::vector<int64_t> tileShape = {std::min(NUM_32, bs), NUM_64};
        TileShape::Current().SetVecTile(tileShape);
        auto qTmpFp32 = Cast(q, DataType::DT_FP32);
        auto qTmpScaleDequant = qKv[2];
        auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpScaleDequant);
        auto qTmpDequantChannel = Mul(qTmpDequantPerToken, qBProjWScale);
        q = Cast(qTmpDequantChannel, dType);
    }
    auto qTmp = Reshape(q, {b, s, numHeads, qHeadDim});

    /******** q ********/
    Tensor qNope = View(qTmp, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]

    std::vector<int64_t> tileShape = {NUM_2, 1, NUM_32, NUM_128};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeR = Reshape(qNope, {bs, numHeads, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
    tileShape = {NUM_2, NUM_32, qkNopeHeadDim};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeT = Transpose(qNopeR, {0, 1}); // [n,bs,qkNopeHeadDim] 32_2_128

    int c0 = NUM_16;
    int m = (std::min(NUM_32, bs) + c0 - 1) / c0 * c0;
    TileShape::Current().SetCubeTile({m, m}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeT, kvBProjWK);

    tileShape = {NUM_16, NUM_2, kvLoraRank};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeNewT = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]
    Tensor qNopeNewR = Reshape(qNopeNewT, {b, s, numHeads, kvLoraRank}); // [b,s,n,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_32, kvLoraRank};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeNewT2 = Transpose(qNopeNewR, {1, 2}); // [b,n,s,kvLoraRank]

    tileShape = {NUM_2, NUM_32, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    Tensor queryStates = Cat({qNopeNewT2, qPeRope}, -1); // [b,n,s, kvLoraRank + qkRopeHeadDim]

    return {queryStates, kvTmp};
}


std::vector<Tensor> DeepseekAttention::MlaPrologFoward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin,
        Tensor kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig, bool isQuant) {
    // hiddenStates: (b,s,h), positionIds: (b,s)
    int b = hiddenStates.GetShape()[0];
    int s = hiddenStates.GetShape()[1];
    int bs = b * s;
    DataType dType = hiddenStates.GetStorage()->Datatype();

    auto qKv = QkvPre2(hiddenStates, isQuant);
    Tensor q = qKv[0];     // [b,s,n,qHeadDim]
    Tensor kvTmp = qKv[1];    // [b,s,kvLoraRank+qkRopeHeadDim]

    //dequant int32 -> fp32  -> *scale  -> fp16/bf16
    if (isQuant) {
        std::vector<int64_t> tileShape = {std::min(NUM_32, bs), NUM_64};
        TileShape::Current().SetVecTile(tileShape);
        auto qTmpFp32 = Cast(q, DataType::DT_FP32);
        auto qTmpScaleDequant = qKv[2];
        auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpScaleDequant);
        auto qTmpDequantChannel = Mul(qTmpDequantPerToken, qBProjWScale);

        q = Cast(qTmpDequantChannel, dType);
    }
    auto qTmp = Reshape(q, {b, s, numHeads, qHeadDim});

    /******** q ********/
    Tensor qNope = View(qTmp, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
    std::vector<int64_t> tileShape = {NUM_32, 1, 1, NUM_128};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeR = Reshape(qNope, {bs, numHeads, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
    tileShape = {NUM_2, NUM_32, qkNopeHeadDim};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeT = Transpose(qNopeR, {0, 1}); // [n,bs,qkNopeHeadDim]

    int c0 = NUM_16;
    int m = (std::min(NUM_32, bs) + c0 - 1) / c0 * c0;
    TileShape::Current().SetCubeTile({m, m}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeT, kvBProjWK);

    tileShape = {NUM_16, NUM_2, kvLoraRank};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeNewT = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]
    Tensor qNopeNewR = Reshape(qNopeNewT, {b, s, numHeads, kvLoraRank}); // [b,s,n,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_32, kvLoraRank};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeNewT2 = Transpose(qNopeNewR, {1, 2}); // [b,n,s,kvLoraRank]

    /******** kv ********/
    Tensor compressedKv = View(kvTmp, {b, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_512};
    TileShape::Current().SetVecTile(tileShape);
    Tensor compressedKvNorm = RmsNorm(compressedKv); // [b,s,kvLoraRank]
    Tensor kNope = Reshape(compressedKvNorm, {b, 1, s, kvLoraRank}); // [b,1,s,kvLoraRank]

    /******** RoPE ********/
    // [b,s,n,qkRopeHeadDim]
    Tensor qPe = View(qTmp, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
    tileShape = {NUM_2, 1, NUM_32, qkNopeHeadDim};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qPeT = Transpose(qPe, {1, 2}); // [b,n,s,qkRopeHeadDim]

    Tensor kPe = View(kvTmp, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // [b,s,qkRopeHeadDim]
    tileShape = {std::min(NUM_32, bs), 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    Tensor kPeR = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // [b,1,s,qkRopeHeadDim]

    Tensor qPeRope(qPeT.GetStorage()->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope"); // [b,n,s,qkRopeHeadDim]
    Tensor kPeRope(kPeR.GetStorage()->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // [b,1,s,qkRopeHeadDim]
    ApplyRotaryPosEmb(qPeT, kPeR, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);

    /******** output q & kv ********/
    tileShape = {NUM_2, NUM_32, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    Tensor queryStates = Cat({qNopeNewT2, qPeRope}, -1); // [b,n,s, kvLoraRank + qkRopeHeadDim]

    tileShape = {1, 1, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    Tensor keyStates = Cat({kNope, kPeRope}, -1); // [b,1,s, kvLoraRank + qkRopeHeadDim]

    tileShape = {1, 1, NUM_256, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    pastKeyStates = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // increase

    std::vector<Tensor> res = {queryStates, pastKeyStates, qNopeNewT2, qPeRope};
    return res;
}

Tensor DeepseekV2MoE::MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2, 
        Tensor ffnWeight3, int nRoutedExperts) {
    // x: (b*s, h), topkIds, topkWeight: (b*s, num_experts_per_tok)
    int bs = topkIds.GetShape(0);
    int expertPerTok = topkIds.GetShape(1);
    std::vector<int64_t> zerosShape(NUM_2);
    zerosShape[0] = bs;
    zerosShape[1] = nRoutedExperts;
    Tensor randoms(topkIds.GetStorage()->Datatype(), zerosShape);

    Tensor cnts = Mul(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

    cnts = Scatter_(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

    Tensor tokensPerExpert = Sum(cnts, 0, true);

    TileShape::Current().SetVecTile(NUM_128);
    // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
    Tensor idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*num_experts_per_tok)

    TileShape::Current().SetVecTile({NUM_128, NUM_128});

    Tensor sortedTokens = TensorIndex(x,  Cast(Div(Cast(idxs, DataType::DT_FP32),
        Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
        DataType::DT_INT32, CAST_TRUNC));

    auto &sortedTokensShape = sortedTokens.GetShape();

    // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
    // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
    std::vector<int> tokensPerExpertCpu(NUM_256, 0);
    for (size_t i = 0; i < NUM_8; i++) {
        tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
    }

    std::vector<Tensor> outputs;
    int startIdx = 0;

    for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
        int numTokens = tokensPerExpertCpu[i];
        if (numTokens == 0) {
            continue;
        }
        const int endIdx = startIdx + numTokens;
        // sortedTokens只有两维
        Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
        std::cout<<"=numTokens===="<<numTokens<<std::endl;
        for (auto n : tokensForThisExpert.GetShape()){
            std::cout<<"=tokensForThisExpert.GetShape()"<< n <<std::endl;
        }

        // 这里没有选对应的expert，默认infer模式下所有的expert相同
        // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
        auto expertOut = expert.Forward(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3);
        outputs.emplace_back(expertOut);
        startIdx = endIdx;
    }

    Tensor outs = Cat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*num_experts_per_tok, h)
    Tensor newX(outs.GetDataType(), outs.GetShape()); // (b*s*num_experts_per_tok, h)

    for (auto n: outs.GetShape()){
        std::cout << "=outs.GetShape()" << n << std::endl;
    }
    TileShape::Current().SetVecTile({NUM_128, NUM_128});
    // newX[idxs] = outs  -->index_put: (b*s*num_experts_per_tok, h)[b*s*num_experts_per_tok] =
    // (b*s*num_experts_per_tok, h)
    auto newIdxs = Reshape(idxs, {1, idxs.GetShape(0)});
    newX = IndexPut(newX, {newIdxs}, outs);

    int newXSize = std::accumulate(
        newX.GetShape().begin(), newX.GetShape().end(), 1, [](const int &a, const int &b) { return a * b; });
    std::cout << "===newXSize" << newXSize << std::endl;

    std::vector<int64_t> newShape = {bs, expertPerTok, newXSize / (bs * expertPerTok)};
    // (b*s, expertPerTok, h)
    auto newXShape = Reshape(newX, newShape);  // [128,256] -> [16,8,256]
    TileShape::Current().SetVecTile(NUM_16, NUM_128, NUM_128);

    auto wShapes = topkWeight.GetShape();
    wShapes.emplace_back(1);
    auto newW = Unsqueeze(topkWeight, NUM_2); // (b*s, expertPerTok, 1)
    auto newMul = Mul(newXShape, newW);
    // (b*s, expertPerTok, h) * (b*s, expertPerTok, 1) = (b*s, expertPerTok, h)
    auto reduceRes = Sum(newMul, 1, true); // reudce轴1 ->(b*s, 1, h)
    for (auto n: reduceRes.GetShape()){
        std::cout << "=reduceRes.GetShape().shape" << n <<std::endl;
    }

    auto fOut = Reshape(reduceRes, {bs, newXSize / (bs * expertPerTok)});

    return fOut;
}

Tensor DeepseekV2MoE::MoeInferSingleMlp(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2,
        Tensor ffnWeight3, int nRoutedExperts) {
    // x: (b*s, h), topkIds, topkWeight: (b*s, num_experts_per_tok)
    (void)topkWeight;
    int bs = topkIds.GetShape(0);
    int expertPerTok = topkIds.GetShape(1);
    std::vector<int64_t> zerosShape(NUM_2);
    zerosShape[0] = bs;
    zerosShape[1] = nRoutedExperts;
    Tensor randoms(topkIds.GetStorage()->Datatype(), zerosShape);

    Tensor cnts = Mul(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

    cnts = Scatter_(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

    Tensor tokensPerExpert = Sum(cnts, 0, true);

    TileShape::Current().SetVecTile(NUM_128);
    // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
    Tensor idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*num_experts_per_tok)

    TileShape::Current().SetVecTile({NUM_128, NUM_128});

    // Tensor((b*s, h))[Tensor(b*s*num_experts_per_tok)] = (b*s*num_experts_per_tok, h)
    // 没有int类型除法 只能先cast成float做完除法再cast回int
    Tensor sortedTokens = TensorIndex(x,  Cast(Div(Cast(idxs, DataType::DT_FP32),
        Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
        DataType::DT_INT32, CAST_TRUNC));

    auto &sortedTokensShape = sortedTokens.GetShape();

    // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
    // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
    std::vector<int> tokensPerExpertCpu(NUM_256, 0);
    for (size_t i = 0; i < 1; i++) {
        tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
    }

    std::vector<Tensor> outputs;
    int startIdx = 0;

    for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
        int numTokens = tokensPerExpertCpu[i];
        if (numTokens == 0) {
            continue;
        }
        const int endIdx = startIdx + numTokens;
        // sortedTokens只有两维
        Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
        std::cout<<"=numTokens===="<<numTokens<<std::endl;
        for (auto n : tokensForThisExpert.GetShape()){
            std::cout<<"=tokensForThisExpert.GetShape().shape"<< n <<std::endl;
        }

        // 这里没有选对应的expert，默认infer模式下所有的expert相同
        // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
        auto expertOut = expert.Forward(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3);
        outputs.emplace_back(expertOut);
        startIdx = endIdx;
    }

    Tensor outs = Cat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*num_experts_per_tok, h)
    return outs;
}

Tensor DeepseekV2MoE::MoeInferSingleMlpQuant(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2,
        Tensor ffnWeight3, Tensor ffnwight1Scale, Tensor ffnwight2Scale,Tensor ffnwight3Scale, int nRoutedExperts) {
    (void)topkWeight;
    int bs = topkIds.GetShape(0);
    int expertPerTok = topkIds.GetShape(1);
    std::vector<int64_t> zerosShape(NUM_2);
    zerosShape[0] = bs;
    zerosShape[1] = nRoutedExperts;
    Tensor randoms(topkIds.GetStorage()->Datatype(), zerosShape);

    Tensor cnts = Mul(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

    cnts = Scatter_(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

    Tensor tokensPerExpert = Sum(cnts, 0, true);

    TileShape::Current().SetVecTile(NUM_128);
    // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
    Tensor idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*num_experts_per_tok)

    TileShape::Current().SetVecTile({NUM_32, NUM_512});

    // Tensor((b*s, h))[Tensor(b*s*num_experts_per_tok)] = (b*s*num_experts_per_tok, h)
    // 没有int类型除法 只能先cast成float做完除法再cast回int
    Tensor sortedTokens = TensorIndex(x,  Cast(Div(Cast(idxs, DataType::DT_FP32),
        Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
        DataType::DT_INT32, CAST_TRUNC));

    TileShape::Current().SetVecTile({NUM_256, NUM_256});
    auto &sortedTokensShape = sortedTokens.GetShape();

    // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
    // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
    std::vector<int> tokensPerExpertCpu(NUM_256, 0);
    for (size_t i = 0; i < 1; i++) {
        tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
    }

    std::vector<Tensor> outputs;
    int startIdx = 0;

    for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
        int numTokens = tokensPerExpertCpu[i];
        if (numTokens == 0) {
            continue;
        }
        const int endIdx = startIdx + numTokens;
        // sortedTokens只有两维
        Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
        std::cout<<"=numTokens===="<<numTokens<<std::endl;
        for (auto n : tokensForThisExpert.GetShape()){
            std::cout<<"=tokensForThisExpert.GetShape().shape"<< n <<std::endl;
        }

        // 这里没有选对应的expert，默认infer模式下所有的expert相同
        // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
        auto expertOut = expert.ForwardWithQuant(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3, ffnwight1Scale, ffnwight2Scale, ffnwight3Scale);
        outputs.emplace_back(expertOut);
        startIdx = endIdx;
    }

    Tensor outs = Cat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*num_experts_per_tok, h)
    return outs;
}

Tensor DeepseekV2MoE::MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3,
        Tensor &idxs, Tensor &sortedTokens, Tensor &outs, int nRoutedExperts) {
    // x: (b*s, h), topkIds, topkWeight: (b*s, numExpertsPerTok)
    int bs = topkIds.GetShape(0);
    int expertPerTok = topkIds.GetShape(1);
    std::vector<int64_t> zerosShape(NUM_2);
    zerosShape[0] = bs;
    zerosShape[1] = nRoutedExperts;
    Tensor randoms(topkIds.GetStorage()->Datatype(), zerosShape);

    Tensor cnts = Mul(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

    cnts = Scatter_(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

    Tensor tokensPerExpert = Sum(cnts, 0, true);

    TileShape::Current().SetVecTile(NUM_128);
    // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
    idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*numExpertsPerTok)

    TileShape::Current().SetVecTile({NUM_64, NUM_64});

    sortedTokens = TensorIndex(x,  Cast(Div(Cast(idxs, DataType::DT_FP32),
        Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
        DataType::DT_INT32, CAST_TRUNC));

    auto &sortedTokensShape = sortedTokens.GetShape();

    // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
    // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
    std::vector<int> tokensPerExpertCpu(NUM_256, 0);
    for (size_t i = 0; i < NUM_8; i++) {
        tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
    }

    std::vector<Tensor> outputs;
    int startIdx = 0;

    for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
        int numTokens = tokensPerExpertCpu[i];
        if (numTokens == 0) {
            continue;
        }
        const int endIdx = startIdx + numTokens;
        // sorted_tokens只有两维
        Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
        std::cout<<"=numTokens===="<<numTokens<<std::endl;
        for (auto n : tokensForThisExpert.GetShape()) {
            std::cout<<"=tokensForThisExpert.GetShape().shape"<< n <<std::endl;
        }

        // 这里没有选对应的expert，默认infer模式下所有的expert相同
        // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
        auto expertOut = expert.Forward(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3);
        outputs.emplace_back(expertOut);
        startIdx = endIdx;
    }

    outs = Cat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*numExpertsPerTok, h)
    Tensor newX(outs.GetDataType(), outs.GetShape()); // (b*s*numExpertsPerTok, h)

    for (auto n : outs.GetShape()){
        std::cout<<"=outs.GetShape().shape"<< n <<std::endl;
    }
    TileShape::Current().SetVecTile({NUM_128, NUM_128});
    // newX[idxs] = outs  -->index_put: (b*s*numExpertsPerTok, h)[b*s*numExpertsPerTok] =
    // (b*s*numExpertsPerTok, h)
    auto newIdxs = Reshape(idxs, {1, idxs.GetShape(0)});
    newX = IndexPut(newX, {newIdxs}, outs);

    int newXSize = std::accumulate(
        newX.GetShape().begin(), newX.GetShape().end(), 1, [](const int &a, const int &b) { return a * b; });
    std::cout<<"===newXSize"<<newXSize<<std::endl;

    std::vector<int64_t> newShape = {bs, expertPerTok, newXSize / (bs * expertPerTok)};
    // (b*s, expertPerTok, h)
    auto newXShape = Reshape(newX, newShape);  // [128,256] -> [16,8,256]
    TileShape::Current().SetVecTile(NUM_16, NUM_64, NUM_64);

    auto wShape = topkWeight.GetShape();
    wShape.emplace_back(1);
    auto newW = Unsqueeze(topkWeight, NUM_2); // (b*s, expertPerTok, 1)
    auto newMul = Mul(newXShape, newW);
    // (b*s, expertPerTok, h) * (b*s, expertPerTok, 1) = (b*s, expertPerTok, h)
    auto reduceRes = Sum(newMul, 1, true); // reudce轴1 ->(b*s, 1, h)
    for (auto n : reduceRes.GetShape()){
        std::cout<<"=reduceRes.GetShape().shape"<< n <<std::endl;
    }

    auto fOut = Reshape(reduceRes, {bs, newXSize / (bs * expertPerTok)});

    return fOut;
}

Tensor DeepseekV2MoE::MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, int nRoutedExperts) {
    // x: (b*s, h), topkIds, topkWeight: (b*s, numExpertsPerTok)
    int bs = topkIds.GetShape(0);
    int expertPerTok = topkIds.GetShape(1);
    const int twoDim = 2;
    std::vector<int64_t> zerosShape(twoDim);
    zerosShape[0] = bs;
    zerosShape[1] = nRoutedExperts;
    Tensor randoms(topkIds.GetStorage()->Datatype(), zerosShape);

    Tensor cnts = Mul(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

    cnts = Scatter_(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

    Tensor tokensPerExpert = Sum(cnts, 0, true);
    // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
    Tensor idxs = ArgSort(Reshape(topkIds, {bs * expertPerTok}), -1); // (b*s*numExpertsPerTok)

    // Tensor((b*s, h))[Tensor(b*s*numExpertsPerTok)] = (b*s*numExpertsPerTok, h)
    Tensor sortedTokens = TensorIndex(x, Div(idxs, Element(DataType::DT_FP32,
        static_cast<double>(expertPerTok)))); // int64除法
    auto &sortedTokensShape = sortedTokens.GetShape();

    // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
    // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
    std::vector<int> tokensPerExpertCpu(NUM_256, 0);
    tokensPerExpertCpu[0] = sortedTokensShape[0] / NUM_8;
    // tokensPerExpertCpu[1] = sortedTokensShape[0] / 8;
    // tokensPerExpertCpu[2] = sortedTokensShape[0] / 8;
    // tokensPerExpertCpu[3] = sortedTokensShape[0] / 8;
    // tokensPerExpertCpu[4] = sortedTokensShape[0] / 8;
    // tokensPerExpertCpu[5] = sortedTokensShape[0] / 8;
    // tokensPerExpertCpu[6] = sortedTokensShape[0] / 8;
    // tokensPerExpertCpu[7] = sortedTokensShape[0] / 8;

    std::vector<Tensor> outputs;
    int startIdx = 0;

    for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
        int numTokens = tokensPerExpertCpu[i];
        if (numTokens == 0) {
            continue;
        }
        const int endIdx = startIdx + numTokens;
        // sorted_tokens只有两维
        Tensor tokensForThisExpert = View(sortedTokens, {numTokens, sortedTokensShape[1]}, {startIdx, 0});
        // 这里没有选对应的expert，默认infer模式下所有的expert相同
        // (numTokens[i],
        // h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
        auto expertOut = expert.Forward(tokensForThisExpert);
        outputs.emplace_back(expertOut);
        startIdx = endIdx;
    }

    // newX = torch.empty_like(outs)
    auto outs = Cat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*numExpertsPerTok, h)
    Tensor newX(outs.GetDataType(), outs.GetShape()); // (b*s*numExpertsPerTok, h)

    // newX[idxs] = outs  -->index_put: (b*s*numExpertsPerTok, h)[b*s*numExpertsPerTok] =
    // (b*s*numExpertsPerTok, h)
    auto newIdxs = Reshape(idxs, {1, idxs.GetShape(0)});
    newX = IndexPut(newX, {newIdxs}, outs);

    int newXSize = std::accumulate(
        newX.GetShape().begin(), newX.GetShape().end(), 1, [](const int &a, const int &b) { return a * b; });
    std::vector<int64_t> newShape = {bs, expertPerTok, newXSize / (bs * expertPerTok)};
    // (b*s, expertPerTok, h)
    auto newXShape = Reshape(newX, newShape);
    TileShape::Current().SetVecTile(NUM_128, NUM_64, NUM_64); // for Assemble
    auto newl = Cast(newXShape, topkWeight.GetDataType());
    auto wShape = topkWeight.GetShape();
    wShape.emplace_back(1);
    auto newW = Unsqueeze(topkWeight, 2); // (b*s, expertPerTok, 1)
    auto newMul = Mul(newl, newW);
    // (b*s, expertPerTok, h) * (b*s, expertPerTok, 1) = (b*s, expertPerTok, h)
    auto fOut = Cast(Sum(newMul, 1, true), newX.GetDataType()); // reudce轴1 ->(b*s, h)
    TileShape::Current().SetVecTile(NUM_128, NUM_64);              // for Assemble

    return fOut;
}

Tensor DeepseekV2MoE::Forward(Tensor hiddenStates) {
    const Tensor identity = hiddenStates;
    const std::vector<int64_t> &origShape = hiddenStates.GetShape();

    auto moeGateRes = moeGate.Forward(hiddenStates);   // hiddenStates: (b*s, h)
    const Tensor &topkIdx = std::get<0>(moeGateRes);    // (b*s, numExpertsPerTok)
    const Tensor &topkWeight = std::get<1>(moeGateRes); // (b*s, numExpertsPerTok)

    // MoeInfer
    Tensor inferRes = MoeInfer(hiddenStates, topkIdx, topkWeight);
    inferRes = Reshape(inferRes, origShape); // (b*s, h)
    const Tensor &sharedMlp = sharedExpert.Forward(identity); // (b, s, h)

    return Add(inferRes, sharedMlp);
}

std::tuple<Tensor, Tensor> MoEGate::Forward(const Tensor &hiddenStates) {
    // hiddenStates: [b*s,h]
    int bs = hiddenStates.GetShape()[0];

    /* compute gating score */
    auto logits = Matrix::Matmul<false, true>(DataType::DT_FP32, hiddenStates, weight); // [b*s,h] @ [nRoutedExperts,h].t -> [b*s,256]
    auto scores = Sigmoid(logits);                          // [b*s,256]

    /* select top-k experts */
    auto scoresForChoice = Add(scores, eScoreCorrectionBias);
    // [b*s,256]+[1,256]->[b*s,256]
    // groupScores = (View(scoresForChoice, bsz * seq_len, self.nGroup, -1).topk(2, dim=-1)[0].sum())
    // groupIdx = torch.topk(groupScores, k=self.topkGroup, dim=-1, sorted=False)[1]
    std::vector<int64_t> shape = {
        scoresForChoice.GetShape()[0] * nGroup, scoresForChoice.GetShape()[1] / nGroup
    }; // [b*s,256]->[b*s*8,32]
    auto scoresForChoiceNewShape = Reshape(scoresForChoice, shape);
    auto scoresForChoiceIndex = std::get<0>(TopK(scoresForChoiceNewShape, 2, -1));
    // [b*s*8,32]->[b*s*8,2]

    auto groupScores = Sum(scoresForChoiceIndex, 1, true); // [b*s*8,2]->[b*s*8]

    auto groupScoresReshape = Reshape(groupScores, {groupScores.GetShape()[0] / nGroup, nGroup});
    // [b*s*8]->[b*s,8]
    auto groupIdx = std::get<1>(TopK(groupScoresReshape, topkGroup, 1)); // [b*s,8]->[b*s,4]

    // groupMask = torch.zeros_like(groupScores)
    auto groupMask = Mul(groupScoresReshape, Element(DataType::DT_FP32, F_0)); // [b*s,8]
    // groupMask.scatter_(1, groupIdx, 1)
    auto groupMaskScatter = Scatter_(groupMask, groupIdx, Element(DataType::DT_FP32, F_1), 1); // [b*s,8]
    // scoreMask
    int dim0 = groupMaskScatter.GetShape()[0] * groupMaskScatter.GetShape()[1];

    auto scoreMask = Expand(Reshape(groupMaskScatter, {dim0, 1}), {dim0, nRoutedExperts / nGroup}); // [b*s*8,1] -> [b*s*8,32]
    scoreMask = Reshape(scoreMask, {bs, nRoutedExperts}); // [b*s*8,32]->[b*s,256]
    auto scoreMaskNot = Mul(scoreMask, Element(DataType::DT_FP32, F_NEGA_1));

    // // tmpScores = scoresForChoice.masked_fill(~scoreMask.bool(), 0.0)

    auto tmpScores = Mul(scoresForChoice, scoreMaskNot);
    // _, topkIdx = torch.topk(tmpScores, k=self.top_k, dim=-1, sorted=False)
    auto topkIdx = std::get<1>(TopK(tmpScores, numExpertsPerTok, -1)); // [b*s,256]->[b*s,8]
    // topkWeight = scores.gather(1, topkIdx)
    auto topkWeight = GatherElements(scores, topkIdx, 1); // [b*s,8]

    /* norm gate to sum 1 */
    // denominator = topkWeight.sum(dim=-1, keepdim=True) + 1e-20
    auto topkWeightSum = Sum(topkWeight, 1, true);      // [b*s,8]->[b*s,1]
    auto denominator = Add(topkWeightSum, Element(DataType::DT_FP32, DF_1E_20)); // [b*s,1]
    // topkWeight = topkWeight / denominator
    topkWeight = Div(topkWeight, denominator); // [b*s,numExpertsPerTok]

    /* expert-level computation auxiliary loss */
    // aux_loss = None

    return std::make_tuple(topkIdx, topkWeight);
}

Tensor DeepseekV2MLP::Forward(Tensor x) {
    // x 可能多维
    auto &xShape = x.GetShape();
    auto mSize =
        std::accumulate(xShape.begin(), xShape.end() - 1, 1, [](const int &a, const int &b) { return a * b; });

    if (xShape.size() > NUM_2) {
        x = Reshape(x, {mSize, xShape[xShape.size() - 1]});
    }
    const Tensor &gateProj = Matrix::Matmul<false, false>(DataType::DT_FP32, x, gateProjW);
    // Silu
    const Tensor &gateSilu = Div(gateProj, Add(Exp(Mul(gateProj, Element(DataType::DT_FP32, F_NEGA_1))),
        Element(DataType::DT_FP32, F_1)));
    const Tensor &upProj = Matrix::Matmul<false, false>(DataType::DT_FP32, x, upProjW);
    const Tensor &mul = Mul(gateSilu, upProj);
    // (x.shape[:-1], intermediateSize) * (intermediateSize, hiddenSize) = (x.shape[:-1], hiddenSize)
    Tensor downProj = Matrix::Matmul<false, false>(DataType::DT_FP32, mul, downProjW);
    if (xShape.size() > NUM_2) {
        downProj = Reshape(downProj, xShape);
    }
    return downProj;
    }

Tensor DeepseekV2MLP::Forward(Tensor x, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3) {
    // static ffn
    auto castRes = Cast(x, DataType::DT_FP16);
    auto gate = Matrix::Matmul<false, false>(DataType::DT_FP32, castRes, ffnWeight1);  // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]

    // swish: x / (1 + e^(-x))
    auto swish = Mul(gate, Element(DataType::DT_FP32, F_NEGA_1));
    swish = Exp(swish);
    swish = Add(swish, Element(DataType::DT_FP32, F_1));
    swish = Div(gate, swish);

    // upProj
    auto up = Matrix::Matmul<false, false>(DataType::DT_FP32, castRes, ffnWeight2);  // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]
    swish = Mul(swish, up);
    auto swishFp16 = Cast(swish, DataType::DT_FP16);

    // downProj
    Tensor res = Matrix::Matmul<false, true>(DataType::DT_FP32, swishFp16, ffnWeight3);  // [b*s, n*d*3] [n*d, n*d*3]^T => [b*s, n*d]

    return res;
}

Tensor DeepseekV2MLP::ForwardWithQuant(Tensor x, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3, Tensor ffnwight1Scale, Tensor ffnwight2Scale,Tensor ffnwight3Scale) {
    // static ffn
    // quant
    TileShape::Current().SetVecTile({NUM_32, NUM_512});
    auto normQuantRes = Quant(x); // int8
    TileShape::Current().SetVecTile({NUM_256, NUM_256});
    Tensor castRes = std::get<0>(normQuantRes);
    Tensor castResScale = std::get<1>(normQuantRes);
    TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_128, NUM_128}, {NUM_128, NUM_128});

    auto gateInt32 = Matrix::Matmul<false, false>(DataType::DT_INT32, castRes, ffnWeight1);

    // dequant: int32 -> fp32 -> *scale -> fp16/bf16
    auto gateTmpFp32 = Cast(gateInt32, DataType::DT_FP32);
    auto gateTmpDequantPerToken = Mul(gateTmpFp32, castResScale);
    auto gate = Mul(gateTmpDequantPerToken, ffnwight1Scale);

    // swish: x / (1 + e^(-x))
    auto swish = Mul(gate, Element(DataType::DT_FP32, F_NEGA_1));
    swish = Exp(swish);
    swish = Add(swish, Element(DataType::DT_FP32, F_1));
    swish = Div(gate, swish);

    auto upInt32 = Matrix::Matmul<false, false>(DataType::DT_INT32, castRes, ffnWeight2);
    // upProj
    auto upTmpFp32 = Cast(upInt32, DataType::DT_FP32);
    auto upTmpDequantPerToken = Mul(upTmpFp32, castResScale);
    auto up = Mul(upTmpDequantPerToken, ffnwight2Scale);

    swish = Mul(swish, up);

    // downProj
    TileShape::Current().SetVecTile({NUM_32, NUM_512});
    auto swishQuantRes = Quant(swish); // int8
    TileShape::Current().SetVecTile({NUM_256, NUM_256});
    Tensor swishRes = std::get<0>(swishQuantRes);
    Tensor swishScale = std::get<1>(swishQuantRes);

    Tensor resInt32 = Matrix::Matmul<false, true>(DataType::DT_INT32, swishRes, ffnWeight3);
    auto resTmpFp32 = Cast(resInt32, DataType::DT_FP32);
    auto resTmpDequantPerToken = Mul(resTmpFp32, swishScale);
    Tensor ffnwight3ScaleTrans = Transpose(ffnwight3Scale, {0, 1});
    auto res = Mul(resTmpDequantPerToken, ffnwight3ScaleTrans);

    return res;
}

} // namespace npu::tile_fwk
