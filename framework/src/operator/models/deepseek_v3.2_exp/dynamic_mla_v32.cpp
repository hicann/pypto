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
 * \file dynamic_mla.cpp
 * \brief
 */
 
#include "dynamic_mla_v32.h"
 
namespace npu::tile_fwk {
 
Tensor DeQuantV32(DataType dType, const Tensor &input, const Tensor &scale, const Tensor &wScale) {
    Tensor dequantRes = Cast(input, DataType::DT_FP32);
    dequantRes = Mul(dequantRes, scale);
    dequantRes = Mul(dequantRes, wScale);
    return Cast(dequantRes, dType);
}
 
Tensor RopeV2(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM2 && cos.GetShape().size() == SHAPE_DIM2 && sin.GetShape().size() == SHAPE_DIM2);
 
    auto seqSize = x.GetShape()[NUM_0];
    auto dR = x.GetShape()[NUM_1];
    auto xDtype = x.GetDataType();
 
    TileShape::Current().SetVecTile(
        tileConfig.twoDim[NUM_0], tileConfig.twoDim[NUM_1]);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
        castX = Add(castX, Element(DT_FP32, 0.0f));
    }
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);
 
    auto xView = Reshape(castX, {1, seqSize, dR / NUM_2, NUM_2});
    TileShape::Current().SetVecTile(tileConfig.fourDim[NUM_0],
        tileConfig.fourDim[NUM_1], tileConfig.fourDim[NUM_2], tileConfig.fourDim[NUM_3]);
    auto xTrans = Transpose(xView, {NUM_2, NUM_3});
    auto xReSecond = Reshape(xTrans, {seqSize, dR});
 
    TileShape::Current().SetVecTile(
        tileConfig.twoDim[NUM_0], tileConfig.twoDim[NUM_1]);
    if (!(x.GetShape()[0] == cos.GetShape()[0] && x.GetShape()[1] == cos.GetShape()[1])) {
        castCos = Expand(castCos, x.GetShape());
        castSin = Expand(castSin, x.GetShape());
    }
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin));
    auto res = Cast(xEmbed, xDtype);
    return res;
}
 
Tensor Rope3DV2(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM3 && cos.GetShape().size() == SHAPE_DIM2 && sin.GetShape().size() == SHAPE_DIM2);
 
    TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
        castX = Add(castX, Element(DT_FP32, 0.0f));
    }
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);
 
    castCos = Reshape(castCos, {x.GetShape()[NUM_0], 1, x.GetShape()[NUM_2]});
    castSin = Reshape(castSin, {x.GetShape()[NUM_0], 1, x.GetShape()[NUM_2]});
 
    auto xView = Reshape(castX, {x.GetShape()[NUM_0], x.GetShape()[NUM_1], x.GetShape()[NUM_2] / NUM_2, NUM_2});
    TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128, NUM_128);
    auto xTrans = Transpose(xView, {NUM_2, NUM_3});
    auto xReSecond = Reshape(xTrans, x.GetShape());
    TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128, NUM_128);
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin));
    auto res = Cast(xEmbed, x.GetDataType());
    return res;
}
 
std::vector<Tensor> PreCompute2D(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs)
{
    // quant
    Tensor dequantScaleWDq = quantInputs.dequantScaleWDq;
    Tensor dequantScaleWDkvKr = quantInputs.dequantScaleWDkvKr;
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuantA = (dequantScaleWDq.GetStorage() != nullptr) && (dequantScaleWDkvKr.GetStorage() != nullptr);
    bool isQuantB = dequantScaleWUqQr.GetStorage() != nullptr;
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;
    bool isSmooth = (smoothScalesCq.GetStorage() != nullptr);
 
    int bs = tokenX.GetShape()[0];
    int q_lora_rank = wDq.GetShape()[1];
 
    DataType dType = tokenX.GetDataType();
    DataType dTypeQuantAOut = isQuantA ? DataType::DT_INT32 : dType;
    DataType dTypeQuantBOut = isQuantB ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;
 
    config::SetSemanticLabel("pre_reshape");
    Tensor inputQuant, inputQuantScale;
 
    /******** q ********/
    int c0 = 16; // 16
    int m = (std::min(32, bs) + c0 - 1) / c0 * c0;
    int mv = std::min(8, bs);
    // [b*s,h] @ [h,q_lora_rank] = [b*s,q_lora_rank]
    Tensor qAProj;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        config::SetSemanticLabel("Quant_x");
        auto quantRes = Quant(tokenX);
        inputQuant = std::get<0>(quantRes);
        inputQuantScale = std::get<1>(quantRes);
        config::SetSemanticLabel("QuantMatmul_qa");
        qAProj = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDq);
        config::SetSemanticLabel("Dequant_qa");
        qAProj = DeQuantV32(dType, qAProj, inputQuantScale, dequantScaleWDq);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        config::SetSemanticLabel("Matmul_qa");
        qAProj = Matrix::Matmul(dType, tokenX, wDq);
    }
 
    // rmsnorm
    TileShape::Current().SetVecTile(mv, q_lora_rank);
    config::SetSemanticLabel("RmsNorm_qa");
    Tensor normRes = RmsNorm(qAProj, gammaCq, epsilonCq);
 
    // [b*s,qLoraRank] @ [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor qBProj;
    Tensor normQuant, normQuantScale;
    if (isQuantB) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        config::SetSemanticLabel("Quant_qMmRes");
        std::tuple<Tensor, Tensor> quantRes;
        if (isSmooth) {
            quantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            quantRes = Quant(normRes, true, false);
        }
        normQuant = std::get<0>(quantRes);  // int8
        normQuantScale = std::get<1>(quantRes);  //fp32
        config::SetSemanticLabel("QuantMatmul_qb");
        qBProj = Matrix::Matmul(dTypeQuantBOut, normQuant, wUqQr);
        config::SetSemanticLabel("Dequant_qb");
        qBProj = DeQuantV32(dType, qBProj, normQuantScale, dequantScaleWUqQr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        config::SetSemanticLabel("Matmul_qb");
        qBProj = Matrix::Matmul(dType, normRes, wUqQr);
    }
    qkvPreRes.emplace_back(qBProj);
 
    /******** kv ********/
    // [b*s,h] @ [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        config::SetSemanticLabel("QuantMatmul_kva");
        compressedKv = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDkvKr);
        config::SetSemanticLabel("Dequant_kva");
        compressedKv = DeQuantV32(dType, compressedKv, inputQuantScale, dequantScaleWDkvKr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        config::SetSemanticLabel("Matmul_kva");
        compressedKv = Matrix::Matmul(dType, tokenX, wDkvKr);
    }
    qkvPreRes.emplace_back(compressedKv);
    if (isQuantB) {
        qkvPreRes.emplace_back(normQuant);
        qkvPreRes.emplace_back(normQuantScale);
    } else {
        qkvPreRes.emplace_back(normRes);
    }

    return qkvPreRes;
}
 
// MlaProlog, b and s is dynamic, support:
// b: 16, 32, 64, 24, 48, 96
// s: 1, 2
void MlaPrologComputeV32(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    Tensor &rmsRes, float epsilonCq, float epsilonCkv, std::string cacheMode) {
    // params check
    assert(tokenX.GetShape().size() == 3 && wUk.GetShape().size() == 3 && sin.GetShape().size() == 3); // shape dim 3
    assert(kvCache.GetShape().size() == 4 && krCache.GetShape().size() == 4); // shape dim 4
    assert(cacheMode == "PA_BSND" || cacheMode == "PA_NZ");
    DataType dType = tokenX.GetDataType();
    int h = tokenX.GetShape()[2]; // 2
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk.GetShape()[0];
    int q_lora_rank = wDq.GetShape()[1];
    int qkNopeHeadDim = wUk.GetShape()[1];
    int kvLoraRank = wUk.GetShape()[2];
    int qkRopeHeadDim = sin.GetShape()[2]; // [b,s,qkRopeHeadDim], 2
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    // kvCache: [block_num, block_size, n2, kv_lora_rank], n2=1
    SymbolicScalar blockNum = GetInputShape(kvCache, 0);
    int blockSize = kvCache.GetShape()[1];
    int n2 = kvCache.GetShape()[2];
    assert(qkNopeHeadDim == 128 || qkRopeHeadDim == 64); // support 128, 64
 
    int tileB = tileConfig.tileB;
    int tileS = tileConfig.tileS;
    int tileBS = tileB * tileS;
 
    RoPETileShapeConfigNew ropeConfig {
        {tileB, tileS, qkRopeHeadDim},           // (b,s,d)
        {tileB, tileS, 1, qkRopeHeadDim},        // (b,s,n,d) Q
        {tileB, tileS, 1, qkRopeHeadDim},        // (b,s,1,d) K
        {tileB, tileS, 1, qkRopeHeadDim / 2, 2}  // (b,s,n,d//2,2)
    };
 
    RopeTileShapeConfig ropeCfg {
                {128, 128},
        { 32, 128,   128},
        { 16, 128,  128, 128}
    };
 
    SymbolicScalar b = GetInputShape(tokenX, 0);
    SymbolicScalar s = GetInputShape(tokenX, 1);
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    SymbolicScalar bsLoop  = (b * s + tileBS - 1 ) / tileBS;
 
    Tensor x2D(tokenX.GetDataType(), {b * s, h}, "x2D");
    Tensor cos2D(cos.GetDataType(), {b * s, qkRopeHeadDim}, "cos2D");
    Tensor sin2D(sin.GetDataType(), {b * s, qkRopeHeadDim}, "sin2D");
    Tensor kCacheIndex2D(cacheIndex.GetDataType(), {b * s, 1}, "kCacheIndex2D");
 
    LOOP("LOOP_MLA_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
        (void) batchId;
        x2D = Reshape(tokenX, {b * s, h}, true);
        cos2D = Reshape(cos, {b * s, qkRopeHeadDim}, true);
        sin2D = Reshape(sin, {b * s, qkRopeHeadDim}, true);
        kCacheIndex2D = Reshape(cacheIndex, {b * s, 1}, true);
    }
 
    Tensor kvCacheRes(kvCache.GetDataType(), {blockNum * blockSize * n2, kvLoraRank}, "kvCacheRes");
    Tensor krCacheRes(krCache.GetDataType(), {blockNum * blockSize * n2, qkRopeHeadDim}, "krCacheRes");
    LOOP("MLA_RESHAPE", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
        (void)unUsedIdx;
        kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank}, true);
        krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim}, true);
    }
    
    LOOP("MLA_BS_Loop", FunctionType::DYNAMIC_LOOP, bsIdx, LoopRange(0, bsLoop, 1)) {
        SymbolicScalar bsOffset = bsIdx * tileBS;
        std::vector<SymbolicScalar> outputOffset = {bsOffset, 0, 0};
        TileShape::Current().SetVecTile({tileBS, 128}); // 128
        auto xView = View(x2D, {tileBS, h}, {bsOffset, 0});
        xView = Cast(Cast(xView, DataType::DT_FP32),dType);
 
        auto qKv = PreCompute2D(xView, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs);
        Tensor q = qKv[0];      // [b*s, n*qHeadDim]
        Tensor kvTmp = qKv[1];  // [b*s, kvLoraRank+qkRopeHeadDim]
        auto qTmp = Reshape(q, {tileBS, n, qHeadDim});
 
        /******** q ********/
        config::SetSemanticLabel("Prepare_qNope");
        Tensor qNope = View(qTmp, {tileBS, n, qkNopeHeadDim}, {0, 0, 0});  // [b,s,n,qkNopeHeadDim]
        std::vector<int64_t> tileShape = {std::min(32, tileBS), 1, qkNopeHeadDim};              // 32
        TileShape::Current().SetVecTile(tileShape);
        Tensor qNopeTrans = Transpose(qNope, {0, 1});  // [n,bs,qkNopeHeadDim]
 
        int c0 = 16;                                        // 16
        int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;  // 32
        config::SetSemanticLabel("Matmul_qNope_wUk");
        TileShape::Current().SetCubeTile({m, m}, {128, 128}, {128, 128});  // 128
        // bmm: (n,bs,qkNopeHeadDim) @ (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
        Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);
 
        config::SetSemanticLabel("queryOut");
        tileShape = {1, std::min(32, tileBS), kvLoraRank};  // 32
        TileShape::Current().SetVecTile(tileShape);
        Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1});  // [bs,n,kvLoraRank]
        config::SetSemanticLabel("Assemble_queryOut");
        TileShape::Current().SetVecTile({ 1, 32, 128});  // 32, 128
        Assemble(qNopeNewTrans, outputOffset, queryOut);   // output1
 
        Tensor qPeView = View(qTmp, {tileBS, n, qkRopeHeadDim}, {0, 0, qkNopeHeadDim});
        auto cos2DView = View(cos2D, {tileBS, qkRopeHeadDim}, {bsOffset, 0}); 
        auto sin2DView = View(sin2D, {tileBS, qkRopeHeadDim}, {bsOffset, 0});
        auto qRopeView = Rope3DV2(qPeView, cos2DView, sin2DView, ropeCfg);
        config::SetSemanticLabel("Assemble_qRope");
        TileShape::Current().SetVecTile({1, 32, 64});  // 32, 64
        Assemble(qRopeView, outputOffset, queryRopeOut);  // output2
 
        /******** RoPE ********/
        TileShape::Current().SetVecTile({2,512});
        config::SetSemanticLabel("RotaryPosEmb");
        Tensor kPeView = View(kvTmp, {tileBS, qkRopeHeadDim}, {0, kvLoraRank}); // [b*s,qkRopeHeadDim]
        auto kRopeView = RopeV2(kPeView, cos2DView, sin2DView, ropeCfg); 
 
        Tensor kRopeRes = Reshape(kRopeView, {tileBS ,1, 1, qkRopeHeadDim});
                    /******** krCache ********/
        config::SetSemanticLabel("ScatterUpdate_krCache");
        tileShape = {1, qkRopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);
 
        auto index = View(kCacheIndex2D, {tileBS, 1}, {bsOffset, 0});
        // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
        TileShape::Current().SetVecTile(NUM_4, NUM_128, NUM_128, NUM_128);
        krCacheOut = ScatterUpdate(krCache, index, kRopeRes, -2, cacheMode, blockSize); // -2

        Tensor compressedKv = View(kvTmp, {tileBS, kvLoraRank}, {0, 0}); // [b*s,kvLoraRank]
        tileShape = {2, 512}; // 2, 512
        config::SetSemanticLabel("RmsNorm_compressedKv");
        TileShape::Current().SetVecTile(tileShape);
        Tensor kNope = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b*s,kvLoraRank]
        kNope = Reshape(kNope,{tileBS ,1, 1, kvLoraRank});

        /******** kvCache ********/
        config::SetSemanticLabel("ScatterUpdate_kvCache");
        TileShape::Current().SetVecTile(NUM_4, NUM_128, NUM_128, NUM_512);
        // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
        kvCacheOut = ScatterUpdate(kvCache, index, kNope, -2, cacheMode, blockSize); // -2

        TileShape::Current().SetVecTile({tileBS,q_lora_rank});
        auto  rms3D = Cast(Cast(qKv[2], DataType::DT_FP32), dType);
        Assemble(rms3D, {bsOffset, 0}, rmsRes);
    }
}
 
void MlaPrologV32(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    float epsilonCq, float epsilonCkv, std::string cacheMode) {
    FUNCTION("main",
        {tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
         quantInputs.dequantScaleWDq, quantInputs.dequantScaleWDkvKr, quantInputs.dequantScaleWUqQr, quantInputs.smoothScalesCq},
        {queryOut, queryRopeOut, kvCacheOut, krCacheOut}) {
        // compute
        Tensor rmsRes(tokenX.GetDataType(), {GetInputShape(tokenX, 0), GetInputShape(tokenX, 1), wDq.GetShape()[1]}, "rmsRes");
        MlaPrologComputeV32(tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs, tileConfig, queryOut, queryRopeOut, kvCacheOut, krCacheOut, rmsRes, epsilonCq, epsilonCkv, cacheMode);
    }
}
 
} // namespace npu::tile_fwk